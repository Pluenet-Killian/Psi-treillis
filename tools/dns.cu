// SPDX-License-Identifier: Apache-2.0
// dns.cu - direct numerical simulation of forced homogeneous isotropic turbulence: the solver that produces the
// sequences.
//
// Incompressible Navier-Stokes in a 2 pi-periodic cube, pseudo-spectral method:
//   - non-linear term in rotational form u x omega, evaluated in physical space;
//   - dealiasing by the 2/3 rule, projection onto the divergence-free fields;
//   - viscosity treated exactly by an integrating factor, RK2 (Heun) time stepping;
//   - forcing: the energy of the modes |k| <= kf is kept constant (same principle as the isotropic dataset of the
//     Johns Hopkins Turbulence Databases).
// The program first makes the flow converge to a statistically stationary regime, then records a sequence of frames at
// a fixed time step (src/sequence.h).
//
// Layout: real array of index (z * N + y) * N + x; the reduced dimension of the real-to-complex transform is x (kx
// from 0 to N/2).

#include <cuda_runtime.h>
#include <cufft.h>

#include "sequence.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(call)                                                                        \
    do {                                                                                \
        cudaError_t e_ = (call);                                                        \
        if (e_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error: %s\n  %s:%d\n", cudaGetErrorString(e_), source_name(__FILE__), __LINE__); \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)
#define CF(call)                                                                        \
    do {                                                                                \
        cufftResult r_ = (call);                                                        \
        if (r_ != CUFFT_SUCCESS) {                                                      \
            fprintf(stderr, "cuFFT error %d\n  %s:%d\n", (int)r_, source_name(__FILE__), __LINE__);  \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

typedef cufftComplex cpx;

struct Grid {
    int n, nh;  // nh = n/2 + 1
};

// Wavenumbers of the mode of index i in the complex array.
__device__ inline void wavevector(const Grid g, size_t i, float& kx, float& ky, float& kz, float& weight) {
    const int ix = (int)(i % g.nh), iy = (int)((i / g.nh) % g.n), iz = (int)(i / ((size_t)g.nh * g.n));
    kx = (float)ix;
    ky = (float)(iy <= g.n / 2 ? iy : iy - g.n);
    kz = (float)(iz <= g.n / 2 ? iz : iz - g.n);
    weight = (ix == 0 || ix == g.n / 2) ? 1.0f : 2.0f;  // half-spectrum: the other modes count double
}
__device__ inline bool dealiased(const Grid g, float kx, float ky, float kz) {
    const float lim = (float)(g.n / 3);
    return fabsf(kx) <= lim && fabsf(ky) <= lim && fabsf(kz) <= lim;
}

// omega^ = i k x u^
__global__ void curl_k(Grid g, const cpx* u, const cpx* v, const cpx* w, cpx* ox, cpx* oy, cpx* oz, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    // i (k x u): component c = ky*w - kz*v, multiplied by i -> (-Im, Re)
    const cpx cx = {ky * w[i].x - kz * v[i].x, ky * w[i].y - kz * v[i].y};
    const cpx cy = {kz * u[i].x - kx * w[i].x, kz * u[i].y - kx * w[i].y};
    const cpx cz = {kx * v[i].x - ky * u[i].x, kx * v[i].y - ky * u[i].y};
    ox[i] = {-cx.y, cx.x};
    oy[i] = {-cy.y, cy.x};
    oz[i] = {-cz.y, cz.x};
}

// N = u x omega in physical space, with the 1/N^3 normalisation of the forward transform.
__global__ void cross_phys(const float* u, const float* v, const float* w, float* ox, float* oy, float* oz, float norm, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float a = u[i], b = v[i], c = w[i], p = ox[i], q = oy[i], r = oz[i];
    ox[i] = norm * (b * r - c * q);
    oy[i] = norm * (c * p - a * r);
    oz[i] = norm * (a * q - b * p);
}

// Dealiasing + projection onto the divergence-free fields.
__global__ void project_dealias(Grid g, cpx* a, cpx* b, cpx* c, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float k2 = kx * kx + ky * ky + kz * kz;
    if (k2 == 0.0f || !dealiased(g, kx, ky, kz)) {
        a[i] = b[i] = c[i] = {0.0f, 0.0f};
        return;
    }
    const float dr = (kx * a[i].x + ky * b[i].x + kz * c[i].x) / k2;
    const float di = (kx * a[i].y + ky * b[i].y + kz * c[i].y) / k2;
    a[i] = {a[i].x - kx * dr, a[i].y - kx * di};
    b[i] = {b[i].x - ky * dr, b[i].y - ky * di};
    c[i] = {c[i].x - kz * dr, c[i].y - kz * di};
}

// Substep 1: u* = E (u + dt N1), with E = exp(-nu k^2 dt).
__global__ void rk_stage1(Grid g, const cpx* u, const cpx* n1, cpx* us, float nu, float dt, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float e = expf(-nu * (kx * kx + ky * ky + kz * kz) * dt);
    us[i] = {e * (u[i].x + dt * n1[i].x), e * (u[i].y + dt * n1[i].y)};
}

// Substep 2: u <- E u + dt/2 (E N1 + N2).
__global__ void rk_stage2(Grid g, cpx* u, const cpx* n1, const cpx* n2, float nu, float dt, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float e = expf(-nu * (kx * kx + ky * ky + kz * kz) * dt);
    u[i] = {e * u[i].x + 0.5f * dt * (e * n1[i].x + n2[i].x), e * u[i].y + 0.5f * dt * (e * n1[i].y + n2[i].y)};
}

// Spectral statistics: acc[0] = energy, acc[1] = sum of k^2 E(k) (enstrophy), acc[2] = energy of the forced band. If
// shells != nullptr, also accumulates E(k) per integer shell.
__global__ void spectral_stats(Grid g, const cpx* u, const cpx* v, const cpx* w, float kf, bool band_only, double* acc, double* shells, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float k2 = kx * kx + ky * ky + kz * kz;
    const bool in_band = k2 > 0.0f && k2 <= kf * kf;
    if (band_only && !in_band) return;
    const double e = 0.5 * wt * ((double)u[i].x * u[i].x + (double)u[i].y * u[i].y + (double)v[i].x * v[i].x + (double)v[i].y * v[i].y +
                                 (double)w[i].x * w[i].x + (double)w[i].y * w[i].y);
    if (e == 0.0) return;
    if (in_band) atomicAdd(&acc[2], e);
    if (band_only) return;
    atomicAdd(&acc[0], e);
    atomicAdd(&acc[1], e * k2);
    if (shells) {
        const int s = (int)(sqrtf(k2) + 0.5f);
        if (s < g.n) atomicAdd(&shells[s], e);
    }
}

// Resets the energy of the forced band to its target value.
__global__ void force_band(Grid g, cpx* u, cpx* v, cpx* w, float kf, float factor, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float k2 = kx * kx + ky * ky + kz * kz;
    if (k2 > 0.0f && k2 <= kf * kf) {
        u[i].x *= factor; u[i].y *= factor;
        v[i].x *= factor; v[i].y *= factor;
        w[i].x *= factor; w[i].y *= factor;
    }
}

// Shapes the initial spectrum: amplitude in k^2 exp(-(k/k0)^2), normalisation 1/N^3.
__global__ void shape_initial(Grid g, cpx* a, float k0, float norm, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float kx, ky, kz, wt;
    wavevector(g, i, kx, ky, kz, wt);
    const float k = sqrtf(kx * kx + ky * ky + kz * kz);
    const float s = norm * k * k * expf(-(k / k0) * (k / k0));
    a[i].x *= s;
    a[i].y *= s;
}

__global__ void scale_cpx(cpx* a, float s, size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) { a[i].x *= s; a[i].y *= s; }
}

struct Solver {
    Grid g;
    size_t nr, nc;  // sizes of the real and complex arrays
    float nu, kf, ef;
    cufftHandle r2c, c2r;
    cpx *u[3], *us[3], *n1[3], *n2[3], *scratch;
    float* phys[6];  // u, v, w, then omega (reused for the cross product)
    double* acc;
    double* shells;
    unsigned gc() const { return (unsigned)((nc + 255) / 256); }
    unsigned gr() const { return (unsigned)((nr + 255) / 256); }

    void init(int n, float nu_, float kf_, float ef_) {
        g = {n, n / 2 + 1};
        nr = (size_t)n * n * n;
        nc = (size_t)n * n * g.nh;
        nu = nu_;
        kf = kf_;
        ef = ef_;
        CF(cufftPlan3d(&r2c, n, n, n, CUFFT_R2C));
        CF(cufftPlan3d(&c2r, n, n, n, CUFFT_C2R));
        for (int c = 0; c < 3; ++c) {
            CK(cudaMalloc(&u[c], nc * sizeof(cpx)));
            CK(cudaMalloc(&us[c], nc * sizeof(cpx)));
            CK(cudaMalloc(&n1[c], nc * sizeof(cpx)));
            CK(cudaMalloc(&n2[c], nc * sizeof(cpx)));
        }
        CK(cudaMalloc(&scratch, nc * sizeof(cpx)));
        for (int c = 0; c < 6; ++c) CK(cudaMalloc(&phys[c], nr * sizeof(float)));
        CK(cudaMalloc(&acc, 3 * sizeof(double)));
        CK(cudaMalloc(&shells, n * sizeof(double)));
    }

    // The complex-to-real transform can overwrite its input: a copy is used.
    void to_phys(const cpx* src, float* dst) {
        CK(cudaMemcpy(scratch, src, nc * sizeof(cpx), cudaMemcpyDeviceToDevice));
        CF(cufftExecC2R(c2r, scratch, dst));
    }

    // Projected and dealiased non-linear term of the state v -> out.
    void rhs(cpx* const v[3], cpx* out[3]) {
        curl_k<<<gc(), 256>>>(g, v[0], v[1], v[2], out[0], out[1], out[2], nc);
        for (int c = 0; c < 3; ++c) to_phys(v[c], phys[c]);
        for (int c = 0; c < 3; ++c) to_phys(out[c], phys[3 + c]);
        cross_phys<<<gr(), 256>>>(phys[0], phys[1], phys[2], phys[3], phys[4], phys[5], 1.0f / (float)nr, nr);
        for (int c = 0; c < 3; ++c) CF(cufftExecR2C(r2c, phys[3 + c], out[c]));
        project_dealias<<<gc(), 256>>>(g, out[0], out[1], out[2], nc);
    }

    void stats(bool band_only, double out[3], std::vector<double>* spectrum = nullptr) {
        CK(cudaMemset(acc, 0, 3 * sizeof(double)));
        if (spectrum) CK(cudaMemset(shells, 0, g.n * sizeof(double)));
        spectral_stats<<<gc(), 256>>>(g, u[0], u[1], u[2], kf, band_only, acc, spectrum ? shells : nullptr, nc);
        CK(cudaMemcpy(out, acc, 3 * sizeof(double), cudaMemcpyDeviceToHost));
        if (spectrum) {
            spectrum->resize(g.n);
            CK(cudaMemcpy(spectrum->data(), shells, g.n * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }

    void force() {
        double s[3];
        stats(true, s);
        if (s[2] > 0.0) force_band<<<gc(), 256>>>(g, u[0], u[1], u[2], kf, (float)std::sqrt(ef / s[2]), nc);
    }

    void step(float dt) {
        rhs(u, n1);
        for (int c = 0; c < 3; ++c) rk_stage1<<<gc(), 256>>>(g, u[c], n1[c], us[c], nu, dt, nc);
        rhs(us, n2);
        for (int c = 0; c < 3; ++c) rk_stage2<<<gc(), 256>>>(g, u[c], n1[c], n2[c], nu, dt, nc);
        force();
    }

    // Physical velocity in phys[0..2]; returns max(|u| + |v| + |w|) for the CFL condition.
    float physical(std::vector<float>& host) {
        for (int c = 0; c < 3; ++c) to_phys(u[c], phys[c]);
        host.resize(3 * nr);
        for (int c = 0; c < 3; ++c) CK(cudaMemcpy(host.data() + c * nr, phys[c], nr * sizeof(float), cudaMemcpyDeviceToHost));
        float m = 0.0f;
        for (size_t i = 0; i < nr; ++i) m = std::max(m, std::fabs(host[i]) + std::fabs(host[nr + i]) + std::fabs(host[2 * nr + i]));
        return m;
    }
};

int main(int argc, char** argv) {
    int n = 256, n_frames = 128, steps_per_frame = 4;
    float nu = 0.0035f, kf = 2.0f, ef = 0.5f, spinup = 12.0f, cfl = 0.4f;
    std::string out = "data/turb.seq";
    unsigned seed = 42;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return i + 1 < argc ? argv[++i] : (fprintf(stderr, "missing value for %s\n", a.c_str()), exit(2), ""); };
        if (a == "--n") n = atoi(next());
        else if (a == "--frames") n_frames = atoi(next());
        else if (a == "--steps-per-frame") steps_per_frame = atoi(next());
        else if (a == "--nu") nu = (float)atof(next());
        else if (a == "--ef") ef = (float)atof(next());
        else if (a == "--spinup") spinup = (float)atof(next());
        else if (a == "--cfl") cfl = (float)atof(next());
        else if (a == "--out") out = next();
        else if (a == "--seed") seed = (unsigned)atoi(next());
        else {
            fprintf(stderr, "Usage: dns [--n 256] [--frames 128] [--steps-per-frame 4] [--nu 0.0035] [--ef 0.5]\n"
                            "           [--spinup 12] [--cfl 0.4] [--out data/turb.seq] [--seed 42]\n");
            return 2;
        }
    }

    Solver s;
    s.init(n, nu, kf, ef);
    const float dx = 6.283185307f / n;
    const float kmax = (float)(n / 3);

    // Initial state: white noise shaped into a large-scale spectrum, then projected.
    {
        std::vector<float> noise(s.nr);
        std::mt19937 rng(seed);
        std::normal_distribution<float> G(0.0f, 1.0f);
        for (int c = 0; c < 3; ++c) {
            for (auto& v : noise) v = G(rng);
            CK(cudaMemcpy(s.phys[c], noise.data(), s.nr * sizeof(float), cudaMemcpyHostToDevice));
            CF(cufftExecR2C(s.r2c, s.phys[c], s.u[c]));
            shape_initial<<<s.gc(), 256>>>(s.g, s.u[c], 3.0f, 1.0f / (float)s.nr, s.nc);
        }
        project_dealias<<<s.gc(), 256>>>(s.g, s.u[0], s.u[1], s.u[2], s.nc);
        double st[3];
        s.stats(false, st);
        const float sc = (float)std::sqrt(1.5 * ef / st[0]);  // initial total energy = 1.5 x forced energy
        for (int c = 0; c < 3; ++c) scale_cpx<<<s.gc(), 256>>>(s.u[c], sc, s.nc);
        s.force();
    }

    auto report = [&](const char* tag, double t, float dt, float umax) {
        double st[3];
        s.stats(false, st);
        const double E = st[0], eps = 2.0 * nu * st[1];
        const double up = std::sqrt(2.0 * E / 3.0), eta = std::pow((double)nu * nu * nu / eps, 0.25);
        const double lam = std::sqrt(15.0 * nu * up * up / eps);
        printf("%s t=%7.3f dt=%.5f E=%.4f eps=%.4f u'=%.4f Re_lambda=%6.1f kmax*eta=%.3f CFL=%.3f\n", tag, t, dt, E, eps, up, up * lam / nu, kmax * eta, umax * dt / dx);
        fflush(stdout);
        return std::vector<double>{up, kmax * eta, up * lam / nu};
    };

    // --- Phase 1: spin-up, adaptive time step.
    std::vector<float> host;
    double t = 0.0;
    float umax = s.physical(host);
    float dt = cfl * dx / umax;
    const auto wall0 = std::chrono::steady_clock::now();
    long nstep = 0;
    while (t < spinup) {
        if (nstep % 20 == 0) {
            umax = s.physical(host);
            dt = cfl * dx / umax;
        }
        if (nstep % 400 == 0) report("spin-up  ", t, dt, umax);
        s.step(dt);
        t += dt;
        ++nstep;
    }
    const double wall_spin = std::chrono::duration<double>(std::chrono::steady_clock::now() - wall0).count();
    printf("Spin-up: %ld steps in %.1f s (%.1f ms per step)\n", nstep, wall_spin, 1e3 * wall_spin / nstep);

    // --- Phase 2: recording at a fixed step, with a margin on the CFL condition.
    umax = s.physical(host);
    dt = 0.75f * cfl * dx / umax;
    const float dt_frame = dt * steps_per_frame;
    const size_t slash = out.find_last_of("/\\");
    if (slash != std::string::npos) {
        const std::string cmd = "mkdir \"" + out.substr(0, slash) + "\" 2>nul";
        (void)system(cmd.c_str());
    }
    FILE* f = fopen(out.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "Cannot open %s\n", out.c_str());
        return 1;
    }
    SequenceHeader h{};
    memcpy(h.magic, "TLSEQ01", 8);
    h.n = n;
    h.n_frames = n_frames;
    h.dt_frame = dt_frame;
    h.nu = nu;
    fwrite(&h, sizeof(h), 1, f);

    double sum_up = 0.0, sum_ke = 0.0, sum_re = 0.0, worst_cfl = 0.0;
    for (int k = 0; k < n_frames; ++k) {
        const float m = s.physical(host);
        worst_cfl = std::max(worst_cfl, (double)m * dt / dx);
        fwrite(host.data(), sizeof(float), host.size(), f);
        if (k % 16 == 0 || k == n_frames - 1) {
            const auto r = report("recording", t, dt, m);
            sum_up += r[0];
            sum_ke += r[1];
            sum_re += r[2];
            h.reserved[0] += 1.0f;
        }
        for (int j = 0; j < steps_per_frame; ++j) s.step(dt);
        t += dt_frame;
    }
    const float cnt = h.reserved[0];
    h.u_rms = (float)(sum_up / cnt);
    h.kmax_eta = (float)(sum_ke / cnt);
    h.re_lambda = (float)(sum_re / cnt);
    h.reserved[0] = 0.0f;
    fseek(f, 0, SEEK_SET);
    fwrite(&h, sizeof(h), 1, f);
    fclose(f);

    std::vector<double> spec;
    double st[3];
    s.stats(false, st, &spec);
    printf("\nFinal energy spectrum E(k):\n");
    for (int k = 1; k <= n / 3; k = (k < 8 ? k + 1 : (int)(k * 1.4) + 1)) printf("  k=%3d  E=%.4e\n", k, spec[k]);
    printf("\nSequence: %s, %d frames of %d^3, dt_frame=%.5f (%.2f voxel per frame at u'), max CFL %.3f\n", out.c_str(), n_frames, n, dt_frame,
           h.u_rms * dt_frame / dx, worst_cfl);
    printf("Statistics: u'=%.4f, Re_lambda=%.1f, kmax*eta=%.3f, size %.2f GiB\n", h.u_rms, h.re_lambda, h.kmax_eta,
           (double)n_frames * sequence_frame_bytes(h) / 1073741824.0);
    return 0;
}
