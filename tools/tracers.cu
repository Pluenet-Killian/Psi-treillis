// SPDX-License-Identifier: Apache-2.0
// tracers.cu - the tracer experiment: a Lagrangian experiment in a frozen snapshot.
//
// Question: does the zero divergence of Psi-treillis change anything for passive tracers?
//
// Tracers drawn uniformly are advected by RK4 in ONE frame of the velocity field, held frozen, seen through a
// reconstruction path. In a stationary field of zero divergence the flow conserves volume: the tracers stay uniform,
// and their number per bin follows the Poisson law. If the reconstructed field has a divergence, the volume carried by
// a tracer drifts and the tracers accumulate where it is negative. The frozen field is a test of a numerical property,
// not a simulation of the turbulence: the recorded sequence lasts only a few Kolmogorov times.
//
// Paths (--path):
//   ref    Psi-treillis of degree 3 at a step of 1 voxel, fine quantisation step (--ref-eps): solenoidal reference
//          more accurate than the default settings. The log says how many coefficients it clips;
//   psi2   Psi-treillis of degree 2, default settings (step of 2 voxels);
//   psi3   Psi-treillis of degree 3, default settings;
//   psi3b  psi3 with the sparse fine level (--tolerance, bound on the largest error);
//   cubic  float32 frame, tricubic interpolation (Catmull-Rom);
//   tex    half-precision 3D texture, hardware trilinear filtering;
//   zfp    cuZFP at fixed rate (--zfp-rate), then half-precision texture;
//   grid   a grid decoded by another compressor (--grid FILE: three planes u, v, w of n^3 float32, indexed like a
//          frame), read as the raw frame is: half-precision texture, hardware trilinear filtering. --label names it
//          in the log;
//   grad   psi3 plus a known compressible perturbation, eps * grad(phi): sum of Fourier modes with integer wave
//          vectors, tuned for an RMS divergence of --grad-ratio times the RMS gradient of the field. It separates the
//          effect of the divergence from that of the velocity error, which Psi-treillis has too.
//
// Measurements, at the requested instants (--tau, in Kolmogorov times):
//   - index of dispersion of the number of tracers per bin (variance / mean; 1 for the Poisson law), for bins of
//     --bin voxels a side, then two and four times larger;
//   - D = integral of div u along each trajectory (the opposite of the logarithm of the Jacobian of the flow), the
//     divergence being taken by centred differences of the same sampler, at +-e (--fd, in voxels): mean and variance
//     over the tracers. For a field of zero divergence, D measures only the truncation of these differences.
//
//   tracers --data f.seq --frame 32 --path psi3 [--particles N] [--seed 1] [--tau 0,1,2,5,10] [--substeps 2]
//           [--fd 0.125] [--no-div] [--bin 1 | --bins 512] [--resort 32] [--eps 0.005] [--tolerance 0.05]
//           [--zfp-rate 2] [--grad-ratio 0.04] [--grad-k 48,96] [--csv]

#include "playback_kernels.cuh"

// ---------------------------------------------------------------------------------------
// Fields specific to the experiment
// ---------------------------------------------------------------------------------------

struct CubicField {  // a single float4 frame, tricubic interpolation
    const float4* f;
    int n;
    float inv_h;
};
__device__ inline V3 sample(const CubicField& f, V3 p, float) { return tricubic(f.f, f.n, f.inv_h, p); }

struct Tex1Field {  // a single texture, hardware trilinear filtering
    cudaTextureObject_t t;
    float inv_l;
};
__device__ inline V3 sample(const Tex1Field& f, V3 p, float) {
    const float4 a = tex3D<float4>(f.t, p.x * f.inv_l, p.y * f.inv_l, p.z * f.inv_l);
    return {a.x, a.y, a.z};
}

// Compressible perturbation: u += sum_m amp_m k_m cos(k_m.x + phase_m), the gradient of
// phi = sum_m amp_m sin(k_m.x + phase_m). Its divergence is -sum_m amp_m |k_m|^2 sin(k_m.x + phase_m).
constexpr int kGradModes = 16;
struct GradModes {
    int k[kGradModes][3];
    float amp[kGradModes], phase[kGradModes];
    int count;
};
template <class B>
struct WithGrad {
    B base;
    GradModes g;
};
template <class B>
__device__ inline V3 sample(const WithGrad<B>& f, V3 p, float w) {
    V3 u = sample(f.base, p, w);
    for (int m = 0; m < f.g.count; ++m) {
        const float c = f.g.amp[m] * cosf((float)f.g.k[m][0] * p.x + (float)f.g.k[m][1] * p.y + (float)f.g.k[m][2] * p.z + f.g.phase[m]);
        u.x += c * (float)f.g.k[m][0];
        u.y += c * (float)f.g.k[m][1];
        u.z += c * (float)f.g.k[m][2];
    }
    return u;
}

// ---------------------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------------------

// Uniform initial positions in [0, L)^3, reproducible: 24 bits per coordinate. Each coordinate is the hash of a term
// of a Weyl sequence (multiple of the golden ratio on 64 bits) shifted by a key drawn from the seed. Hashing the
// counter 3 i + a directly gave a draw that was too regular at 2^27 tracers: initial dispersion by bins of 512^3
// almost identical from one seed to the next (-4.5e-5 +- 3e-6 for +-1.2e-4 expected), verified in numpy.
__global__ void lagr_seed(float* px, float* py, float* pz, uint32_t n, unsigned long long seed) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long g = 0x9E3779B97F4A7C15ull, key = mix64(seed), b = 3ull * i;
    const float s = kL / 16777216.0f;
    px[i] = (float)(mix64((b + 1) * g + key) >> 40) * s;
    py[i] = (float)(mix64((b + 2) * g + key) >> 40) * s;
    pz[i] = (float)(mix64((b + 3) * g + key) >> 40) * s;
}
// One RK4 step in the frozen field. If dsum is not null, the divergence of the field at the starting point, taken by
// centred differences at +-e, is added to the integral D of the tracer (dsum) and kept (dlast).
template <class F>
__global__ void lagr_step(F f, float* px, float* py, float* pz, float* dsum, float* dlast, uint32_t n, float dt, float e) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    V3 p = {px[i], py[i], pz[i]};
    if (dsum) {
        const float ddx = sample(f, V3{p.x + e, p.y, p.z}, 0.0f).x - sample(f, V3{p.x - e, p.y, p.z}, 0.0f).x;
        const float ddy = sample(f, V3{p.x, p.y + e, p.z}, 0.0f).y - sample(f, V3{p.x, p.y - e, p.z}, 0.0f).y;
        const float ddz = sample(f, V3{p.x, p.y, p.z + e}, 0.0f).z - sample(f, V3{p.x, p.y, p.z - e}, 0.0f).z;
        const float d = (ddx + ddy + ddz) * (0.5f / e);
        dsum[i] += d * dt;
        dlast[i] = d;
    }
    const V3 k1 = sample(f, p, 0.0f);
    const V3 k2 = sample(f, p + (0.5f * dt) * k1, 0.0f);
    const V3 k3 = sample(f, p + (0.5f * dt) * k2, 0.0f);
    const V3 k4 = sample(f, p + dt * k3, 0.0f);
    p = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    const float inv_l = 1.0f / kL;
    px[i] = p.x - kL * floorf(p.x * inv_l);
    py[i] = p.y - kL * floorf(p.y * inv_l);
    pz[i] = p.z - kL * floorf(p.z * inv_l);
}
// Number of tracers per bin: nb bins per axis.
__global__ void lagr_count(const float* px, const float* py, const float* pz, uint32_t n, int nb, uint32_t* counts) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = (float)nb / kL;
    const int x = min(nb - 1, max(0, (int)(px[i] * s))), y = min(nb - 1, max(0, (int)(py[i] * s))), z = min(nb - 1, max(0, (int)(pz[i] * s)));
    atomicAdd(&counts[((size_t)z * nb + y) * nb + x], 1u);
}
__global__ void lagr_gather2(const float* a, const float* b, const uint32_t* idx, uint32_t n, float* oa, float* ob) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t j = idx[i];
    oa[i] = a[j];
    ob[i] = b[j];
}
// Sum of |k|^2 |u_hat|^2 over a row (ky, kz) of the half-spectrum of one component: used for the RMS gradient of the
// field.
__global__ void lagr_grad_rows(const cufftComplex* u, int n, double* rows) {
    const size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= (size_t)n * n) return;
    const int qy = (int)(r % n), qz = (int)(r / n), nh = n / 2 + 1;
    const double ky = qy <= n / 2 ? qy : qy - n, kz = qz <= n / 2 ? qz : qz - n;
    double sum = 0.0;
    for (int kx = 0; kx < nh; ++kx) {
        const cufftComplex v = u[r * nh + kx];
        const double w = (kx == 0 || kx == n / 2) ? 1.0 : 2.0;
        sum += w * ((double)kx * kx + ky * ky + kz * kz) * ((double)v.x * v.x + (double)v.y * v.y);
    }
    rows[r] = sum;
}

// ---------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------

struct LOptions {
    std::string data = "data/turb256.seq", path = "psi3";
    int frame = 32;
    long long particles = 0;        // 0: one tracer per voxel
    unsigned long long seed = 1;
    std::vector<double> tau = {0, 1, 2, 5, 10};  // instants of the reports, in Kolmogorov times
    int substeps = 2;               // time step: dt_frame / substeps
    float fd = 0.125f;              // half-spacing of the centred differences of the divergence, in voxels
    bool with_div = true;
    int bin = 1;                    // side of the smallest bins, in voxels
    int bins = 0;                   // number of bins per axis (replaces --bin; allows bins smaller than the voxel)
    int resort = 32;                // Morton sort of the tracers every n steps (0: never)
    float eps = 0.005f, tolerance = 0.05f;
    float ref_eps = 0.001f;         // quantisation step of the ref path
    double zfp_rate = 2.0;
    std::string grid;               // path grid: file of the decoded grid
    std::string label = "grid";     // path grid: name of the path in the log
    std::string dump_grid;          // path zfp: writes the grid decoded by cuZFP to this file, then exits
    float grad_ratio = 0.04f;
    int grad_kmin = 48, grad_kmax = 96;
    double ridge = 1e-5;
    bool csv = false;
};

static std::vector<double> parse_list(const char* s) {
    std::vector<double> v;
    const char* p = s;
    while (*p) {
        char* end = nullptr;
        v.push_back(strtod(p, &end));
        if (end == p) break;
        p = *end ? end + 1 : end;
    }
    return v;
}

static LOptions lparse(int argc, char** argv) {
    LOptions o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--data") o.data = next();
        else if (a == "--path") o.path = next();
        else if (a == "--frame") o.frame = atoi(next());
        else if (a == "--particles") o.particles = atoll(next());
        else if (a == "--seed") o.seed = (unsigned long long)atoll(next());
        else if (a == "--tau") o.tau = parse_list(next());
        else if (a == "--substeps") o.substeps = atoi(next());
        else if (a == "--fd") o.fd = (float)atof(next());
        else if (a == "--no-div") o.with_div = false;
        else if (a == "--bin") o.bin = atoi(next());
        else if (a == "--bins") o.bins = atoi(next());
        else if (a == "--resort") o.resort = atoi(next());
        else if (a == "--eps") o.eps = (float)atof(next());
        else if (a == "--ref-eps") o.ref_eps = (float)atof(next());
        else if (a == "--tolerance") o.tolerance = (float)atof(next());
        else if (a == "--zfp-rate") o.zfp_rate = atof(next());
        else if (a == "--grid") o.grid = next();
        else if (a == "--label") o.label = next();
        else if (a == "--dump-grid") o.dump_grid = next();
        else if (a == "--grad-ratio") o.grad_ratio = (float)atof(next());
        else if (a == "--grad-k") {
            const std::vector<double> k = parse_list(next());
            CHK(k.size() == 2 && k[0] >= 1 && k[1] >= k[0], "--grad-k kmin,kmax");
            o.grad_kmin = (int)k[0];
            o.grad_kmax = (int)k[1];
        } else if (a == "--csv") o.csv = true;
        else {
            fprintf(stderr, "Usage: tracers --data f.seq --frame 32 --path ref|psi2|psi3|psi3b|cubic|tex|zfp|grid|grad [--particles N]\n"
                            "               [--seed 1] [--tau 0,1,2,5,10] [--substeps 2] [--fd 0.125] [--no-div] [--bin 1] [--resort 32]\n"
                            "               [--eps 0.005] [--tolerance 0.05] [--zfp-rate 2] [--dump-grid FILE] [--grid FILE --label NAME]\n"
                            "               [--grad-ratio 0.04] [--grad-k 48,96] [--csv]\n");
            exit(2);
        }
    }
    return o;
}

// What does not depend on the path: the frame, its statistics, the tracers.
struct Ctx {
    SequenceHeader hd{};
    int n = 0;
    size_t voxels = 0;
    float* h_planar = nullptr;
    float* d_planar = nullptr;
    double grad_rms = 0.0, tau_eta = 0.0, u_frame = 0.0;
    uint32_t np = 0;
    float *p[3] = {}, *q[3] = {}, *dsum = nullptr, *dlast = nullptr, *dsum2 = nullptr, *dlast2 = nullptr;
    uint32_t *key = nullptr, *idx = nullptr, *counts = nullptr;
    int nb = 0;
};

// RMS gradient of the field of the frame (Frobenius norm of grad u), by a sum over the spectrum.
static double gradient_rms(const float* d_planar, int n) {
    const size_t voxels = (size_t)n * n * n, spec = (size_t)n * n * (n / 2 + 1);
    cufftHandle plan = 0;
    CHK(cufftPlan3d(&plan, n, n, n, CUFFT_R2C) == CUFFT_SUCCESS, "R2C plan");
    cufftComplex* d_u = nullptr;
    double* d_rows = nullptr;
    CK(cudaMalloc(&d_u, spec * sizeof(cufftComplex)));
    CK(cudaMalloc(&d_rows, (size_t)n * n * sizeof(double)));
    double total = 0.0;
    for (int c = 0; c < 3; ++c) {
        CHK(cufftExecR2C(plan, (float*)d_planar + (size_t)c * voxels, d_u) == CUFFT_SUCCESS, "R2C");
        lagr_grad_rows<<<blocks_for((size_t)n * n), 256>>>(d_u, n, d_rows);
        total += thrust::reduce(thrust::device_pointer_cast(d_rows), thrust::device_pointer_cast(d_rows) + (size_t)n * n, 0.0);
    }
    CK(cudaFree(d_u));
    CK(cudaFree(d_rows));
    cufftDestroy(plan);
    return std::sqrt(total) / (double)voxels;
}

static void sort_particles(Ctx& c) {
    morton_keys<<<blocks_for(c.np), 256>>>(c.p[0], c.p[1], c.p[2], c.np, c.key, c.idx);
    CK(cudaDeviceSynchronize());
    thrust::sort_by_key(thrust::device_pointer_cast(c.key), thrust::device_pointer_cast(c.key) + c.np, thrust::device_pointer_cast(c.idx));
    gather3<<<blocks_for(c.np), 256>>>(c.p[0], c.p[1], c.p[2], c.idx, c.np, c.q[0], c.q[1], c.q[2]);
    lagr_gather2<<<blocks_for(c.np), 256>>>(c.dsum, c.dlast, c.idx, c.np, c.dsum2, c.dlast2);
    CK(cudaDeviceSynchronize());
    for (int a = 0; a < 3; ++a) std::swap(c.p[a], c.q[a]);
    std::swap(c.dsum, c.dsum2);
    std::swap(c.dlast, c.dlast2);
}

// Report: dispersion by bins (three sizes), statistics of D and of the instantaneous divergence.
static void report(Ctx& c, const LOptions& o, const char* label, double t_tau, long long step, double wall) {
    const int nb = c.nb;
    const size_t bins = (size_t)nb * nb * nb;
    CK(cudaMemset(c.counts, 0, bins * sizeof(uint32_t)));
    lagr_count<<<blocks_for(c.np), 256>>>(c.p[0], c.p[1], c.p[2], c.np, nb, c.counts);
    static std::vector<uint32_t> h;
    h.resize(bins);
    CK(cudaMemcpy(h.data(), c.counts, bins * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    double disp[3] = {}, mean[3] = {}, empty[3] = {};
    for (int level = 0; level < 3; ++level) {
        const int f = 1 << level, m = nb / f;
        if (m < 2) continue;
        const size_t cells = (size_t)m * m * m;
        double s1 = 0.0, s2 = 0.0;
        size_t zero = 0;
        if (level == 0) {
            for (size_t i = 0; i < bins; ++i) {
                const double v = h[i];
                s1 += v;
                s2 += v * v;
                zero += h[i] == 0;
            }
        } else {
            for (int z = 0; z < m; ++z)
                for (int y = 0; y < m; ++y)
                    for (int x = 0; x < m; ++x) {
                        uint64_t v = 0;
                        for (int dz = 0; dz < f; ++dz)
                            for (int dy = 0; dy < f; ++dy) {
                                const uint32_t* row = h.data() + ((size_t)(z * f + dz) * nb + (y * f + dy)) * nb + (size_t)x * f;
                                for (int dx = 0; dx < f; ++dx) v += row[dx];
                            }
                        s1 += (double)v;
                        s2 += (double)v * (double)v;
                        zero += v == 0;
                    }
        }
        mean[level] = s1 / cells;
        disp[level] = (s2 / cells - mean[level] * mean[level]) / mean[level];
        empty[level] = (double)zero / cells;
    }
    double d_mean = 0.0, d_var = 0.0, div_rms = 0.0;
    if (o.with_div) {
        static std::vector<float> hd;
        hd.resize(c.np);
        CK(cudaMemcpy(hd.data(), c.dsum, (size_t)c.np * sizeof(float), cudaMemcpyDeviceToHost));
        double s1 = 0.0, s2 = 0.0;
        for (uint32_t i = 0; i < c.np; ++i) {
            s1 += hd[i];
            s2 += (double)hd[i] * hd[i];
        }
        d_mean = s1 / c.np;
        d_var = s2 / c.np - d_mean * d_mean;
        CK(cudaMemcpy(hd.data(), c.dlast, (size_t)c.np * sizeof(float), cudaMemcpyDeviceToHost));
        s2 = 0.0;
        for (uint32_t i = 0; i < c.np; ++i) s2 += (double)hd[i] * hd[i];
        div_rms = std::sqrt(s2 / c.np);
    }
    const double side = (double)c.n / nb;  // side of the smallest bin, in voxels
    printf("t = %6.2f tau_eta (step %lld): dispersion - 1 by bins of %.3g, %.3g, %.3g voxels: %+.5f, %+.5f, %+.5f; empty bins %.5f (Poisson %.5f)", t_tau, step,
           side, 2.0 * side, 4.0 * side, disp[0] - 1.0, disp[1] - 1.0, disp[2] - 1.0, empty[0], std::exp(-mean[0]));
    if (o.with_div)
        printf("; D: mean %+.3e, standard deviation %.4e; divergence seen by the tracers %.3e of the RMS gradient", d_mean, std::sqrt(std::max(d_var, 0.0)),
               step > 0 ? div_rms / c.grad_rms : 0.0);
    printf("; %.0f s\n", wall);
    fflush(stdout);
    if (o.csv)
        printf("CSV,lagr,%d,%d,%s,%llu,%d,%u,%.5g,%.4f,%lld,%.6e,%.6e,%.4g,%.6f,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.6e,%.1f\n", c.n, o.frame, label,
               o.seed, o.substeps, c.np, (double)o.fd, t_tau, step, c.grad_rms, c.tau_eta, side, mean[0], disp[0] - 1.0, disp[1] - 1.0, disp[2] - 1.0, empty[0],
               empty[1], empty[2], d_mean, d_var, o.with_div && step > 0 ? div_rms / c.grad_rms : 0.0, std::exp(-mean[0]), 0.0, wall);
}

template <class F>
static void advect(const F& f, Ctx& c, const LOptions& o, const char* label) {
    const float dt = c.hd.dt_frame / o.substeps, e = o.fd * (kL / c.n);
    std::vector<long long> stops;
    for (double t : o.tau) stops.push_back(std::llround(t * c.tau_eta / dt));
    std::sort(stops.begin(), stops.end());
    printf("Advection      : %s, %u tracers, seed %llu, time step %.6f (%.1f steps per Kolmogorov time, %.3f voxel per step at u'), "
           "centred differences at +-%.4f voxel\n",
           label, c.np, o.seed, dt, c.tau_eta / dt, c.hd.u_rms * dt / (kL / c.n), o.fd);
    lagr_seed<<<blocks_for(c.np), 256>>>(c.p[0], c.p[1], c.p[2], c.np, o.seed);
    CK(cudaMemset(c.dsum, 0, (size_t)c.np * sizeof(float)));
    CK(cudaMemset(c.dlast, 0, (size_t)c.np * sizeof(float)));
    if (o.resort > 0) sort_particles(c);
    const auto t0 = std::chrono::steady_clock::now();
    long long step = 0;
    for (size_t k = 0; k < stops.size(); ++k) {
        for (; step < stops[k]; ++step) {
            if (o.resort > 0 && step > 0 && step % o.resort == 0) sort_particles(c);
            lagr_step<F><<<blocks_for(c.np), 256>>>(f, c.p[0], c.p[1], c.p[2], o.with_div ? c.dsum : nullptr, c.dlast, c.np, dt, e);
        }
        CK(cudaDeviceSynchronize());
        CK(cudaGetLastError());
        report(c, o, label, step * (double)dt / c.tau_eta, step, seconds_since(t0));
    }
}

static cudaTextureObject_t half_texture(const float4* d_f4, int n) {
    const size_t voxels = (size_t)n * n * n;
    ushort4* d_h4 = nullptr;
    CK(cudaMalloc(&d_h4, voxels * sizeof(ushort4)));
    pack_half4<<<blocks_for(voxels), 256>>>(d_f4, d_h4, voxels);
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
    cudaArray_t arr = nullptr;
    CK(cudaMalloc3DArray(&arr, &desc, make_cudaExtent(n, n, n), 0));
    cudaMemcpy3DParms prm{};
    prm.dstArray = arr;
    prm.extent = make_cudaExtent(n, n, n);
    prm.srcPtr = make_cudaPitchedPtr(d_h4, n * sizeof(ushort4), n, n);
    prm.kind = cudaMemcpyDeviceToDevice;
    CK(cudaMemcpy3D(&prm));
    CK(cudaFree(d_h4));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td{};
    td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeWrap;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 1;
    cudaTextureObject_t tex = 0;
    CK(cudaCreateTextureObject(&tex, &res, &td, nullptr));
    return tex;
}

// Psi paths: encodes the frame as a key frame, then advects in the decoded field.
template <int D>
static void run_psi(Ctx& c, const LOptions& o, int mf, int levels, float eps, bool bounded, bool grad) {
    const int n = c.n, fb = 2;
    Geo g;
    g.build(n, D, mf, levels, fb);
    const double gain2 = D == 2 ? 2.0 * 1.0 * 0.55 * 0.55 : 2.0 * (2.0 / 3.0) * (2416.0 / 5040.0) * (2416.0 / 5040.0);
    const float delta = (float)(std::sqrt(12.0) * eps * c.hd.u_rms * g.hl[0] / std::sqrt(gain2 * (1.0 + std::pow(0.25, fb))));
    Sender<D> snd;
    snd.init(g, delta, o.ridge);
    if (bounded) {
        const float fine_eps = o.tolerance / 8.5f;
        snd.init_fine(o.tolerance * c.hd.u_rms, (float)(std::sqrt(12.0) * fine_eps * c.hd.u_rms * ((double)kL / n) / std::sqrt(gain2)));
    }
    snd.encode(c.d_planar, 0, 0);
    if (bounded) snd.refine(c.d_planar, 0);
    double rms = 0.0, mx = 0.0;
    snd.field_error(c.d_planar, 0, rms, mx, nullptr);
    const bool wrapped = snd.frames_wrapped != 0;
    printf("Field          : Psi-treillis of degree %d, step of %d voxel(s), %d levels, eps %.4g%s; nodal error %.4f %%, largest error %.4f u'; "
           "%lld clipping(s), %lld overflow(s) of the fine level; lattice read %s\n",
           D, mf, levels, eps, bounded ? ", sparse fine level" : "", 100.0 * rms, mx / c.hd.u_rms, snd.clip_all(), snd.n_fine, wrapped ? "by differences" : "directly");
    if (bounded)
        printf("Fine level     : tolerance %.2f %% of u', %u nodes retained (%.3f %%), %u voxel(s) still outside the tolerance\n", 100.0 * o.tolerance, snd.fs.n_nodes[0],
               100.0 * snd.fs.n_nodes[0] / (double)c.voxels, snd.fine_left);
    char label[64];
    if (grad) {
        // Modes with integer wave vectors, of modulus between kmin and kmax; amplitudes in 1/|k|^2, hence equal
        // contributions to the divergence. RMS divergence = A sqrt(M / 2) = ratio x RMS gradient.
        WithGrad<PsiPlain<D>> f{};
        CHK(!wrapped, "perturbation: wrapped lattice not supported");
        static_cast<PsiField<D>&>(f.base) = snd.base_field(0);
        std::mt19937 rng(777);
        std::uniform_int_distribution<int> K(-o.grad_kmax, o.grad_kmax);
        std::uniform_real_distribution<float> P(0.0f, kL);
        const double A = o.grad_ratio * c.grad_rms / std::sqrt(kGradModes / 2.0);
        double v2 = 0.0;
        for (int m = 0; m < kGradModes;) {
            const int kx = K(rng), ky = K(rng), kz = K(rng);
            const double k2 = (double)kx * kx + (double)ky * ky + (double)kz * kz;
            if (k2 < (double)o.grad_kmin * o.grad_kmin || k2 > (double)o.grad_kmax * o.grad_kmax) continue;
            f.g.k[m][0] = kx;
            f.g.k[m][1] = ky;
            f.g.k[m][2] = kz;
            f.g.amp[m] = (float)(A / k2);
            f.g.phase[m] = P(rng);
            v2 += 0.5 * A * A / k2;
            ++m;
        }
        f.g.count = kGradModes;
        printf("Perturbation   : %d modes, |k| from %d to %d; RMS divergence %.4f of the RMS gradient; added velocity %.4f %% of u' (RMS of one component)\n",
               kGradModes, o.grad_kmin, o.grad_kmax, o.grad_ratio, 100.0 * std::sqrt(v2 / 3.0) / c.hd.u_rms);
        snprintf(label, sizeof(label), "grad%.4g", o.grad_ratio);
        advect(f, c, o, label);
    } else if (bounded) {
        PsiFine<D, PsiField<D>> f = snd.total_field(0);
        advect(f, c, o, "psi3b");
    } else {
        // The ref path carries its step in its name: two references of different steps are not confused in a log.
        if (o.path == "ref") snprintf(label, sizeof(label), "ref%.4g", eps);
        else snprintf(label, sizeof(label), "%s", o.path.c_str());
        if (wrapped) {
            advect(snd.base_field(0), c, o, label);
        } else {
            PsiPlain<D> f{};
            static_cast<PsiField<D>&>(f) = snd.base_field(0);
            advect(f, c, o, label);
        }
    }
}

int main(int argc, char** argv) {
    const LOptions o = lparse(argc, argv);
    Ctx c;
    FILE* in = fopen(o.data.c_str(), "rb");
    CHK(in, "sequence not found");
    c.hd = sequence_read_header(in);
    const int n = c.n = c.hd.n;
    CHK((n & (n - 1)) == 0 && o.frame >= 0 && o.frame < c.hd.n_frames, "frame outside the sequence");
    c.voxels = (size_t)n * n * n;
    CK(cudaHostAlloc(&c.h_planar, 3 * c.voxels * sizeof(float), cudaHostAllocDefault));
    CK(cudaMalloc(&c.d_planar, 3 * c.voxels * sizeof(float)));
    sequence_read_frame(in, c.hd, o.frame, c.h_planar);
    fclose(in);
    CK(cudaMemcpy(c.d_planar, c.h_planar, 3 * c.voxels * sizeof(float), cudaMemcpyHostToDevice));
    {
        double s2 = 0.0;
        for (size_t i = 0; i < 3 * c.voxels; ++i) s2 += (double)c.h_planar[i] * c.h_planar[i];
        c.u_frame = std::sqrt(s2 / (3.0 * c.voxels));
    }
    c.grad_rms = gradient_rms(c.d_planar, n);
    const double eta = c.hd.kmax_eta / (n / 3);
    c.tau_eta = eta * eta / c.hd.nu;
    printf("Frame          : %s, %d^3, frame %d; u' %.4f (sequence %.4f); RMS gradient %.4f; Kolmogorov time %.5f (eta^2 / nu, that is %.1f frames; "
           "its inverse %.4f)\n",
           o.data.c_str(), n, o.frame, c.u_frame, c.hd.u_rms, c.grad_rms, c.tau_eta, c.tau_eta / c.hd.dt_frame, 1.0 / c.tau_eta);

    c.np = (uint32_t)(o.particles > 0 ? o.particles : (long long)c.voxels);
    CHK(o.bin >= 1 && n % o.bin == 0, "--bin: divisor of the resolution");
    c.nb = o.bins > 0 ? o.bins : n / o.bin;
    CHK(c.nb >= 8 && c.nb % 4 == 0 && c.nb <= 1024, "--bins: multiple of 4, from 8 to 1024");
    for (int a = 0; a < 3; ++a) {
        CK(cudaMalloc(&c.p[a], (size_t)c.np * sizeof(float)));
        CK(cudaMalloc(&c.q[a], (size_t)c.np * sizeof(float)));
    }
    CK(cudaMalloc(&c.dsum, (size_t)c.np * sizeof(float)));
    CK(cudaMalloc(&c.dlast, (size_t)c.np * sizeof(float)));
    CK(cudaMalloc(&c.dsum2, (size_t)c.np * sizeof(float)));
    CK(cudaMalloc(&c.dlast2, (size_t)c.np * sizeof(float)));
    CK(cudaMalloc(&c.key, (size_t)c.np * sizeof(uint32_t)));
    CK(cudaMalloc(&c.idx, (size_t)c.np * sizeof(uint32_t)));
    CK(cudaMalloc(&c.counts, (size_t)c.nb * c.nb * c.nb * sizeof(uint32_t)));

    if (o.path == "ref") run_psi<3>(c, o, 1, 4, o.ref_eps, false, false);
    else if (o.path == "psi2") run_psi<2>(c, o, 2, 3, o.eps, false, false);
    else if (o.path == "psi3") run_psi<3>(c, o, 2, 3, o.eps, false, false);
    else if (o.path == "psi3b") run_psi<3>(c, o, 2, 3, o.eps, true, false);
    else if (o.path == "grad") run_psi<3>(c, o, 2, 3, o.eps, false, true);
    else {
        float4* d_f4 = nullptr;
        CK(cudaMalloc(&d_f4, c.voxels * sizeof(float4)));
        planar_to_f4<<<blocks_for(c.voxels), 256>>>(c.d_planar, d_f4, c.voxels);
        if (o.path == "cubic") {
            printf("Field          : float32 frame, tricubic interpolation (Catmull-Rom)\n");
            advect(CubicField{d_f4, n, 1.0f / (kL / n)}, c, o, "cubic");
        } else if (o.path == "tex") {
            printf("Field          : half-precision 3D texture, hardware trilinear filtering\n");
            advect(Tex1Field{half_texture(d_f4, n), 1.0f / kL}, c, o, "tex");
        } else if (o.path == "grid") {
            CHK(!o.grid.empty(), "--path grid needs --grid FILE");
            std::vector<float> dec(3 * c.voxels);
            FILE* g = fopen(o.grid.c_str(), "rb");
            CHK(g, "grid file not found");
            const size_t got = fread(dec.data(), sizeof(float), dec.size(), g);
            const bool at_end = fgetc(g) == EOF;
            fclose(g);
            CHK(got == dec.size() && at_end, "the grid file does not hold three planes of n^3 float32");
            double se = 0.0, sr = 0.0, mx = 0.0;
            for (size_t i = 0; i < dec.size(); ++i) {
                const double d = (double)dec[i] - c.h_planar[i];
                se += d * d;
                sr += (double)c.h_planar[i] * c.h_planar[i];
                mx = std::max(mx, std::fabs(d));
            }
            float* d_gplanar = nullptr;
            CK(cudaMalloc(&d_gplanar, 3 * c.voxels * sizeof(float)));
            CK(cudaMemcpy(d_gplanar, dec.data(), dec.size() * sizeof(float), cudaMemcpyHostToDevice));
            planar_to_f4<<<blocks_for(c.voxels), 256>>>(d_gplanar, d_f4, c.voxels);
            printf("Field          : grid %s read from %s, then half-precision texture; nodal error %.4f %%, largest error %.4f u'\n",
                   o.label.c_str(), o.grid.c_str(), 100.0 * std::sqrt(se / sr), mx / c.hd.u_rms);
            advect(Tex1Field{half_texture(d_f4, n), 1.0f / kL}, c, o, o.label.c_str());
        } else if (o.path == "zfp") {
#ifdef PSIT_WITH_ZFP
            ZfpCodec codec;
            codec.configure(n, o.zfp_rate);
            float* d_zplanar = nullptr;
            void* d_zstream = nullptr;
            CK(cudaMalloc(&d_zplanar, 3 * c.voxels * sizeof(float)));
            CK(cudaMalloc(&d_zstream, codec.span));
            codec.attach(n, d_f4, d_zplanar, d_zstream);
            codec.compress();
            codec.decompress();
            CK(cudaDeviceSynchronize());
            std::vector<float> dec(3 * c.voxels);
            CK(cudaMemcpy(dec.data(), d_zplanar, dec.size() * sizeof(float), cudaMemcpyDeviceToHost));
            if (!o.dump_grid.empty()) {
                FILE* g = fopen(o.dump_grid.c_str(), "wb");
                CHK(g && fwrite(dec.data(), sizeof(float), dec.size(), g) == dec.size(), "cannot write the grid file");
                fclose(g);
                printf("Grid           : cuZFP at %.2f bits per value, decoded grid written to %s\n", codec.rate, o.dump_grid.c_str());
                return 0;
            }
            double se = 0.0, sr = 0.0, mx = 0.0;
            for (size_t i = 0; i < dec.size(); ++i) {
                const double d = (double)dec[i] - c.h_planar[i];
                se += d * d;
                sr += (double)c.h_planar[i] * c.h_planar[i];
                mx = std::max(mx, std::fabs(d));
            }
            interleave3<<<blocks_for(c.voxels), 256>>>(d_zplanar, d_zplanar + c.voxels, d_zplanar + 2 * c.voxels, d_f4, c.voxels);
            printf("Field          : cuZFP at %.2f bits per value (%.1f:1), then half-precision texture; nodal error %.4f %%, largest error %.4f u'\n",
                   codec.rate, 32.0 / codec.rate, 100.0 * std::sqrt(se / sr), mx / c.hd.u_rms);
            char label[32];
            snprintf(label, sizeof(label), "zfp%.3g", o.zfp_rate);
            advect(Tex1Field{half_texture(d_f4, n), 1.0f / kL}, c, o, label);
#else
            CHK(false, "binary compiled without zfp");
#endif
        } else {
            CHK(false, "unknown path (ref|psi2|psi3|psi3b|cubic|tex|zfp|grid|grad)");
        }
    }
    return 0;
}
