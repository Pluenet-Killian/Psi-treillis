// SPDX-License-Identifier: Apache-2.0
// psi_view.cu - visual comparator in split screen.
//
// Two complete chains run on the same GPU and on the same frames, each in its half of the screen, with its own
// particles and its own timers:
//   psi:file.psi   Psi-treillis read from its self-contained file (degree 2 or 3 according to the file);
//   zfp:file.zfs   cuZFP stream read from a file, decoded on the GPU to a half-precision 3D texture (the compressed
//                  reference);
//   raw:file.seq   raw float32 frames read from disk, sent to the GPU, converted to a half-precision 3D texture (what
//                  a standard tool does).
//
// Three views:
//   1  particles, coloured by the vorticity or the speed of the field of each side;
//   2  slice of the divergence of the continuous field that the particles see, relative to the gradient, in colour
//      over the norm of the velocity gradient in grey;
//   3  slice of the field error against the raw frame (needs --data).
//
// The on-screen text is in English, like the public interface of the project. No displayed value is hard-coded:
// times, bytes, memory, divergence and error are measured during the run. In "race" mode, each side advances at the
// pace given by its own measured costs (disk + bus + decoding + sort + advection), as if it were alone on the GPU.
//
//   psi_view packzfp --data f.seq --file f.zfs [--rate 2 --start 0 --frames 64]
//   psi_view --left zfp:f.zfs --right psi:f.psi [--data f.seq] [--particles 4000000]
//            [--substeps 2] [--width 1600 --height 900] [--view 1|2|3] [--sync] [--sort]
//            [--color vort|speed] [--snapshot image.png --iters 40]

#include "playback_kernels.cuh"

#include "view_window.h"

#include <memory>

// Threads per block of the kernel with the support cache. With 168 registers per thread, an SM holds three blocks of
// 128 but only one of 256: the RK4 step was measured 13 to 16 % shorter.
constexpr unsigned kCacheBlock = 128;

// ---------------------------------------------------------------------------------------
// Display geometry and colours
// ---------------------------------------------------------------------------------------

struct Cam {
    float eye[3], right[3], up[3], fwd[3];
    float focal;
    float pcy;           // ordinate of the principal point in the view, in pixels
    float fog_a, fog_b;  // weight of a particle according to its depth zc: fog_a - fog_b * zc, bounded (1 and 0: no effect)
    int vx, vy, vw, vh, W;
};
struct SliceGeo {
    float cx, cy, z, scale;  // centre of the slice, z coordinate, size of a pixel
    int vx, vy, vw, vh, W;
};

__host__ __device__ inline bool project(const Cam& c, float x, float y, float z, float& sx, float& sy) {
    const float dx = x - c.eye[0], dy = y - c.eye[1], dz = z - c.eye[2];
    const float zc = dx * c.fwd[0] + dy * c.fwd[1] + dz * c.fwd[2];
    if (zc < 0.05f) return false;
    sx = c.vx + 0.5f * c.vw + c.focal * (dx * c.right[0] + dy * c.right[1] + dz * c.right[2]) / zc;
    sy = c.vy + c.pcy - c.focal * (dx * c.up[0] + dy * c.up[1] + dz * c.up[2]) / zc;
    return true;
}

// "inferno" colour map (polynomial approximation), t between 0 and 1.
__host__ __device__ inline uint32_t inferno(float t, float gain = 1.0f) {
    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
    float r = 0.00021894f + t * (0.10651342f + t * (11.602493f + t * (-41.703996f + t * (77.162936f + t * (-71.319428f + t * 25.131126f)))));
    float g = 0.00165100f + t * (0.56395644f + t * (-3.9728540f + t * (17.436399f + t * (-33.402359f + t * (32.626064f + t * -12.242669f)))));
    float b = -0.01948090f + t * (3.9327124f + t * (-15.942394f + t * (44.354145f + t * (-81.807309f + t * (73.209520f + t * -23.070325f)))));
    r = (r < 0.0f ? 0.0f : (r > 1.0f ? 1.0f : r)) * gain;
    g = (g < 0.0f ? 0.0f : (g > 1.0f ? 1.0f : g)) * gain;
    b = (b < 0.0f ? 0.0f : (b > 1.0f ? 1.0f : b)) * gain;
    return ((uint32_t)(r * 255.0f + 0.5f) << 16) | ((uint32_t)(g * 255.0f + 0.5f) << 8) | (uint32_t)(b * 255.0f + 0.5f);
}

__device__ inline V3 slice_pos(const SliceGeo& g, int i, int j) {
    float x = g.cx + ((float)i - 0.5f * g.vw + 0.5f) * g.scale, y = g.cy + (0.5f * g.vh - (float)j - 0.5f) * g.scale;
    x -= kL * floorf(x / kL);
    y -= kL * floorf(y / kL);
    return {x, y, g.z};
}

// ---------------------------------------------------------------------------------------
// Particle view
// ---------------------------------------------------------------------------------------

// Displayed quantity: norm of the velocity (mode 0) or of the vorticity (mode 1, centred differences of half-step dl
// on the field of the path).
template <class F>
__device__ inline float scalar_at(const F& f, V3 p, float w, int mode, float dl) {
    if (mode == 0) {
        const V3 u = sample(f, p, w);
        return sqrtf(u.x * u.x + u.y * u.y + u.z * u.z);
    }
    const V3 xp = sample(f, p + V3{dl, 0.0f, 0.0f}, w), xm = sample(f, p - V3{dl, 0.0f, 0.0f}, w);
    const V3 yp = sample(f, p + V3{0.0f, dl, 0.0f}, w), ym = sample(f, p - V3{0.0f, dl, 0.0f}, w);
    const V3 zp = sample(f, p + V3{0.0f, 0.0f, dl}, w), zm = sample(f, p - V3{0.0f, 0.0f, dl}, w);
    const float k = 0.5f / dl;
    const float wx = ((yp.z - ym.z) - (zp.y - zm.y)) * k, wy = ((zp.x - zm.x) - (xp.z - xm.z)) * k, wz = ((xp.y - xm.y) - (yp.x - ym.x)) * k;
    return sqrtf(wx * wx + wy * wy + wz * wz);
}

// Each visible particle deposits a weight (square of the normalised quantity: the intense structures stand out) and
// this weight multiplied by the quantity, in the pixel where it projects.
template <class F>
__global__ void splat_kernel(F f, const float* px, const float* py, const float* pz, uint32_t n, float w, Cam c, int mode, float dl, float inv_scale,
                             float* acc) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const V3 p = {px[i], py[i], pz[i]};
    float sx, sy;
    if (!project(c, p.x, p.y, p.z, sx, sy)) return;
    sx -= 0.5f;
    sy -= 0.5f;
    const int ix = (int)floorf(sx), iy = (int)floorf(sy);
    if (ix < c.vx - 1 || ix >= c.vx + c.vw || iy < c.vy - 1 || iy >= c.vy + c.vh) return;
    const float t = scalar_at(f, p, w, mode, dl) * inv_scale;
    // Depth cue: the particles close to the eye weigh more than those at the back.
    const float zc = (p.x - c.eye[0]) * c.fwd[0] + (p.y - c.eye[1]) * c.fwd[1] + (p.z - c.eye[2]) * c.fwd[2];
    const float fog = fminf(1.6f, fmaxf(0.3f, c.fog_a - c.fog_b * zc));
    const float wt = fminf(t * t, 2.0f) * fog, ax = sx - (float)ix, ay = sy - (float)iy;
    // Bilinear deposit on the four neighbouring pixels.
    for (int j = 0; j < 2; ++j)
        for (int i = 0; i < 2; ++i) {
            const int x = ix + i, y = iy + j;
            if (x < c.vx || x >= c.vx + c.vw || y < c.vy || y >= c.vy + c.vh) continue;
            const float k = wt * (i ? ax : 1.0f - ax) * (j ? ay : 1.0f - ay);
            const size_t pix = (size_t)y * c.W + x;
            atomicAdd(&acc[2 * pix], k);
            atomicAdd(&acc[2 * pix + 1], k * t);
        }
}
template <class F>
__global__ void probe_kernel(F f, const float* px, const float* py, const float* pz, uint32_t n, float w, int mode, float dl, float* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = scalar_at(f, V3{px[i], py[i], pz[i]}, w, mode, dl);
}
__global__ void compose_particles(const float* acc, uint32_t* img, Cam c, float gain) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= c.vw * c.vh) return;
    const size_t pix = (size_t)(c.vy + k / c.vw) * c.W + c.vx + k % c.vw;
    const float a = acc[2 * pix];
    img[pix] = a > 0.0f ? inferno(0.12f + 0.88f * fminf(acc[2 * pix + 1] / a, 1.0f), 1.0f - expf(-a * gain)) : 0u;
}

// ---------------------------------------------------------------------------------------
// Slices: divergence of the continuous field, error against the raw frame
// ---------------------------------------------------------------------------------------

// Texture: analytic gradient of the trilinear interpolant, from the eight texels read without filtering.
// G[3 i + j] = d u_i / d x_j.
struct TexPoint {
    cudaTextureObject_t t0, t1;
    int n;
};
__device__ inline void tex_gradient(const TexPoint& f, V3 p, float w, float G[9]) {
    const float inv_h = (float)f.n / kL, inv_n = 1.0f / (float)f.n;
    const float gx = p.x * inv_h - 0.5f, gy = p.y * inv_h - 0.5f, gz = p.z * inv_h - 0.5f;
    const float flx = floorf(gx), fly = floorf(gy), flz = floorf(gz);
    const float fx[2] = {1.0f - (gx - flx), gx - flx}, fy[2] = {1.0f - (gy - fly), gy - fly}, fz[2] = {1.0f - (gz - flz), gz - flz};
    float v[2][2][2][3];
    for (int c = 0; c < 2; ++c)
        for (int b = 0; b < 2; ++b)
            for (int a = 0; a < 2; ++a) {
                const float tx = (flx + (float)a + 0.5f) * inv_n, ty = (fly + (float)b + 0.5f) * inv_n, tz = (flz + (float)c + 0.5f) * inv_n;
                const float4 A = tex3D<float4>(f.t0, tx, ty, tz), B = tex3D<float4>(f.t1, tx, ty, tz);
                v[c][b][a][0] = A.x + w * (B.x - A.x);
                v[c][b][a][1] = A.y + w * (B.y - A.y);
                v[c][b][a][2] = A.z + w * (B.z - A.z);
            }
    for (int i = 0; i < 3; ++i) {
        float dx = 0.0f, dy = 0.0f, dz = 0.0f;
        for (int b = 0; b < 2; ++b)
            for (int a = 0; a < 2; ++a) {
                dx += fy[a] * fz[b] * (v[b][a][1][i] - v[b][a][0][i]);
                dy += fx[a] * fz[b] * (v[b][1][a][i] - v[b][0][a][i]);
                dz += fx[a] * fy[b] * (v[1][b][a][i] - v[0][b][a][i]);
            }
        G[3 * i] = dx * inv_h;
        G[3 * i + 1] = dy * inv_h;
        G[3 * i + 2] = dz * inv_h;
    }
}
__global__ void div_slice_tex(TexPoint f, float w, SliceGeo sg, float* out2) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    float G[9];
    tex_gradient(f, slice_pos(sg, k % sg.vw, k / sg.vw), w, G);
    float g2 = 0.0f;
    for (int i = 0; i < 9; ++i) g2 += G[i] * G[i];
    out2[2 * k] = G[0] + G[4] + G[8];
    out2[2 * k + 1] = g2;
}

// Psi-treillis: analytic gradient of the continuous field, evaluated in double precision. Each partial derivative has
// its own accumulator: the divergence is the sum of three numbers computed separately, which cancel only to within
// round-off. G[3 i + j] = d u_i / d x_j.
// Spline of degree D at t (in lattice steps), in double precision: values, first and second derivatives on the nodes
// of the support; returns the index of the first node.
template <int D>
__device__ inline int spline_w012(double t, double w0[4], double w1[4], double w2[4]) {
    if (D == 2) {
        const double i = floor(t + 0.5), fr = t - i, a = 0.5 - fr, b = 0.5 + fr;
        w0[0] = 0.5 * a * a;
        w0[1] = 0.75 - fr * fr;
        w0[2] = 0.5 * b * b;
        w0[3] = 0.0;
        w1[0] = -a;
        w1[1] = -2.0 * fr;
        w1[2] = b;
        w1[3] = 0.0;
        w2[0] = 1.0;
        w2[1] = -2.0;
        w2[2] = 1.0;
        w2[3] = 0.0;
        return (int)i - 1;
    }
    const double i = floor(t), fr = t - i, q = 1.0 - fr;
    w0[0] = q * q * q / 6.0;
    w0[1] = ((3.0 * fr - 6.0) * fr * fr + 4.0) / 6.0;
    w0[2] = (((-3.0 * fr + 3.0) * fr + 3.0) * fr + 1.0) / 6.0;
    w0[3] = fr * fr * fr / 6.0;
    w1[0] = -0.5 * q * q;
    w1[1] = (1.5 * fr - 2.0) * fr;
    w1[2] = (-1.5 * fr + 1.0) * fr + 0.5;
    w1[3] = 0.5 * fr * fr;
    w2[0] = q;
    w2[1] = 3.0 * fr - 2.0;
    w2[2] = -3.0 * fr + 1.0;
    w2[3] = fr;
    return (int)i - 1;
}
template <int D>
__device__ inline void psi_gradient_d(const PsiField<D>& f, const double p[3], double w, double G[9]) {
    double w0[3][4], w1[3][4], w2[3][4];  // spline, first derivative, second derivative (per lattice step)
    int i0[3];
    for (int c = 0; c < 3; ++c) i0[c] = spline_w012<D>((p[c] - (double)f.x0) * (double)f.inv_h, w0[c], w1[c], w2[c]);
    const int m = f.n - 1;
    const int4 v0 = f.lat[((size_t)(i0[2] & m) * f.n + (i0[1] & m)) * f.n + (i0[0] & m)];  // base of the support (lattice modulo 2^16)
    for (int i = 0; i < 9; ++i) G[i] = 0.0;
    for (int c = 0; c <= D; ++c)
        for (int b = 0; b <= D; ++b)
            for (int a = 0; a <= D; ++a) {
                const int4 v = f.lat[((size_t)((i0[2] + c) & m) * f.n + ((i0[1] + b) & m)) * f.n + ((i0[0] + a) & m)];
                int e[3], l[3];
                lat_get(v, v0, f.early, e);
                lat_get(v, v0, f.early ^ 1, l);
                const double q[3] = {e[0] + w * (l[0] - e[0]), e[1] + w * (l[1] - e[1]), e[2] + w * (l[2] - e[2])};
                // Second derivatives of the basis function of the node: H[i][j] = d2 phi / d x_i d x_j.
                double H[3][3];
                H[0][0] = w2[0][a] * w0[1][b] * w0[2][c];
                H[1][1] = w0[0][a] * w2[1][b] * w0[2][c];
                H[2][2] = w0[0][a] * w0[1][b] * w2[2][c];
                H[0][1] = H[1][0] = w1[0][a] * w1[1][b] * w0[2][c];
                H[0][2] = H[2][0] = w1[0][a] * w0[1][b] * w1[2][c];
                H[1][2] = H[2][1] = w0[0][a] * w1[1][b] * w1[2][c];
                for (int j = 0; j < 3; ++j) {
                    G[j] += H[1][j] * q[2] - H[2][j] * q[1];
                    G[3 + j] += H[2][j] * q[0] - H[0][j] * q[2];
                    G[6 + j] += H[0][j] * q[1] - H[1][j] * q[0];
                }
            }
    const double sc = (double)f.s * (double)f.inv_h;
    for (int i = 0; i < 9; ++i) G[i] *= sc;
}
// Sparse fine level (files with an error bound): analytic gradient of its contribution, added to G. Same second
// derivatives as for the base lattice; the nodes come from the blocks of the bricks present (FineLevel in
// src/codec_kernels.cuh). Each partial derivative keeps its accumulator: the divergence of the complete field remains
// a sum of numbers computed separately, which cancel only to round-off.
template <int D>
__device__ inline void fine_gradient_d(const FineLevel& f, const double p[3], double w, int early, double G[9]) {
    double w0[3][4], w1[3][4], w2[3][4];
    int i0[3];
    for (int c = 0; c < 3; ++c) i0[c] = spline_w012<D>((p[c] - (double)f.x0) * (double)f.inv_h, w0[c], w1[c], w2[c]);
    const int m = f.n - 1, mb = f.nb - 1;
    const int ix = i0[0] & m, iy = i0[1] & m, iz = i0[2] & m;
    const int bx = ix >> kFineLog, by = iy >> kFineLog, bz = iz >> kFineLog;
    const int lx = ix & (kFineSide - 1), ly = iy & (kFineSide - 1), lz = iz & (kFineSide - 1);
    const uint32_t head = f.tab[((size_t)bz * f.nb + by) * f.nb + bx];
    if (!(head & 0xFFu)) return;
    int blk[8];
    for (int k = 0; k < 8; ++k) {
        blk[k] = -1;
        if (!((head >> k) & 1u)) continue;
        const uint32_t e = k ? f.tab[((size_t)((bz + (k >> 2)) & mb) * f.nb + ((by + ((k >> 1) & 1)) & mb)) * f.nb + ((bx + (k & 1)) & mb)] : head;
        blk[k] = (int)(e >> 8);
    }
    double A[9];
    for (int i = 0; i < 9; ++i) A[i] = 0.0;
    for (int c = 0; c <= D; ++c) {
        const int zc = lz + c, kz = zc >> kFineLog, z = zc & (kFineSide - 1);
        for (int b = 0; b <= D; ++b) {
            const int yb = ly + b, ky = yb >> kFineLog, y = yb & (kFineSide - 1);
            for (int a = 0; a <= D; ++a) {
                const int xa = lx + a, bl = blk[(kz << 2) | (ky << 1) | (xa >> kFineLog)];
                if (bl < 0) continue;
                const int2 v = f.pool[(size_t)bl * kFineNodes + ((((z << kFineLog) | y) << kFineLog) | (xa & (kFineSide - 1)))];
                const int ew = early ? v.y : v.x, lw = early ? v.x : v.y;
                double q[3];
                for (int i = 0; i < 3; ++i) {
                    const double e = (double)(signed char)(ew >> (8 * i)), l = (double)(signed char)(lw >> (8 * i));
                    q[i] = e + w * (l - e);
                }
                double H[3][3];
                H[0][0] = w2[0][a] * w0[1][b] * w0[2][c];
                H[1][1] = w0[0][a] * w2[1][b] * w0[2][c];
                H[2][2] = w0[0][a] * w0[1][b] * w2[2][c];
                H[0][1] = H[1][0] = w1[0][a] * w1[1][b] * w0[2][c];
                H[0][2] = H[2][0] = w1[0][a] * w0[1][b] * w1[2][c];
                H[1][2] = H[2][1] = w0[0][a] * w1[1][b] * w1[2][c];
                for (int j = 0; j < 3; ++j) {
                    A[j] += H[1][j] * q[2] - H[2][j] * q[1];
                    A[3 + j] += H[2][j] * q[0] - H[0][j] * q[2];
                    A[6 + j] += H[0][j] * q[1] - H[1][j] * q[0];
                }
            }
        }
    }
    const double sc = (double)f.s * (double)f.inv_h;
    for (int i = 0; i < 9; ++i) G[i] += sc * A[i];
}
template <int D>
__global__ void div_slice_psi_fine(PsiField<D> f, FineLevel fl, float w, SliceGeo sg, float* out2) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    const V3 q = slice_pos(sg, k % sg.vw, k / sg.vw);
    const double p[3] = {q.x, q.y, q.z};
    double G[9], g2 = 0.0;
    psi_gradient_d<D>(f, p, w, G);
    fine_gradient_d<D>(fl, p, w, f.early, G);
    for (int i = 0; i < 9; ++i) g2 += G[i] * G[i];
    out2[2 * k] = (float)(G[0] + G[4] + G[8]);
    out2[2 * k + 1] = (float)g2;
}
template <int D>
__global__ void div_slice_psi(PsiField<D> f, float w, SliceGeo sg, float* out2) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    const V3 q = slice_pos(sg, k % sg.vw, k / sg.vw);
    const double p[3] = {q.x, q.y, q.z};
    double G[9], g2 = 0.0;
    psi_gradient_d<D>(f, p, w, G);
    for (int i = 0; i < 9; ++i) g2 += G[i] * G[i];
    out2[2 * k] = (float)(G[0] + G[4] + G[8]);
    out2[2 * k + 1] = (float)g2;
}

// Field error at the instant of the most recent frame, against the raw frame interpolated tricubically (itself within
// 0.1 % of the exact field).
template <class F>
__global__ void err_slice_kernel(F f, const float4* truth, int n, SliceGeo sg, float* out2) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    const V3 p = slice_pos(sg, k % sg.vw, k / sg.vw);
    const V3 a = sample(f, p, 1.0f), b = tricubic(truth, n, (float)n / kL, p), d = a - b;
    out2[2 * k] = sqrtf(d.x * d.x + d.y * d.y + d.z * d.z);
    out2[2 * k + 1] = b.x * b.x + b.y * b.y + b.z * b.z;
}

// Divergence slice. The background is the norm of the velocity gradient, in grey, relative to three times its RMS:
// the flow can be read on both sides, even where the divergence is zero. The divergence is overlaid on it in colour,
// on a logarithmic scale from 10^-8 (no colour) to 1; from 10^-4 upwards it covers the background at 80 %, which lets
// the same flow be made out on both sides.
// inv_ref: inverse of the RMS gradient over the slice.
__global__ void colorize_div(const float* out2, uint32_t* img, SliceGeo sg, float inv_ref) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    const float t = fminf(1.0f, fmaxf(0.0f, (log10f(fmaxf(fabsf(out2[2 * k]) * inv_ref, 1e-30f)) + 8.0f) / 8.0f));
    const float g = fminf(1.0f, sqrtf(fmaxf(out2[2 * k + 1], 0.0f)) * inv_ref / 3.0f);
    const float grey = 255.0f * (0.05f + 0.80f * sqrtf(g)), a = fminf(0.8f, 2.0f * t);
    const uint32_t c = inferno(t);
    const float r = grey + a * ((float)((c >> 16) & 255) - grey), gr = grey + a * ((float)((c >> 8) & 255) - grey), b = grey + a * ((float)(c & 255) - grey);
    img[(size_t)(sg.vy + k / sg.vw) * sg.W + sg.vx + k % sg.vw] = ((uint32_t)(r + 0.5f) << 16) | ((uint32_t)(gr + 0.5f) << 8) | (uint32_t)(b + 0.5f);
}
__global__ void colorize_lin(const float* out2, uint32_t* img, SliceGeo sg, float inv_ref) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= sg.vw * sg.vh) return;
    img[(size_t)(sg.vy + k / sg.vw) * sg.W + sg.vx + k % sg.vw] = inferno(out2[2 * k] * inv_ref);
}

// ---------------------------------------------------------------------------------------
// Reference marks drawn over the image: graduated axes, grid, slice plane, scales
// ---------------------------------------------------------------------------------------

struct Clip {
    int x0, y0, x1, y1;  // drawing rectangle [x0, x1) x [y0, y1)
};
// Line blended into the image (alpha between 0 and 1). behind: only on the dark pixels, hence behind the particles.
static void blend_line(uint32_t* px, int W, Clip cl, float x0, float y0, float x1, float y1, uint32_t rgb, float alpha, bool behind) {
    const float dx = x1 - x0, dy = y1 - y0, span = std::max(fabsf(dx), fabsf(dy));
    if (!(span < 20000.0f)) return;
    const int steps = (int)span + 1;
    const float r = (float)((rgb >> 16) & 255), g = (float)((rgb >> 8) & 255), b = (float)(rgb & 255);
    for (int i = 0; i <= steps; ++i) {
        const float t = (float)i / (float)steps;
        const int x = (int)lroundf(x0 + t * dx), y = (int)lroundf(y0 + t * dy);
        if (x < cl.x0 || x >= cl.x1 || y < cl.y0 || y >= cl.y1) continue;
        uint32_t& p = px[(size_t)y * W + x];
        const float pr = (float)((p >> 16) & 255), pg = (float)((p >> 8) & 255), pb = (float)(p & 255);
        if (behind && std::max(pr, std::max(pg, pb)) > 56.0f) continue;
        p = ((uint32_t)(pr + alpha * (r - pr) + 0.5f) << 16) | ((uint32_t)(pg + alpha * (g - pg) + 0.5f) << 8) | (uint32_t)(pb + alpha * (b - pb) + 0.5f);
    }
}
// Segment in space, cut at the near plane of the camera then projected.
static bool seg3d(const Cam& c, const float a[3], const float b[3], float& ax, float& ay, float& bx, float& by) {
    float p[2][3] = {{a[0], a[1], a[2]}, {b[0], b[1], b[2]}}, z[2];
    for (int i = 0; i < 2; ++i) z[i] = (p[i][0] - c.eye[0]) * c.fwd[0] + (p[i][1] - c.eye[1]) * c.fwd[1] + (p[i][2] - c.eye[2]) * c.fwd[2];
    const float zn = 0.06f;
    if (z[0] < zn && z[1] < zn) return false;
    for (int i = 0; i < 2; ++i)
        if (z[i] < zn) {
            const float t = (zn - z[i]) / (z[i ^ 1] - z[i]);
            for (int d = 0; d < 3; ++d) p[i][d] += t * (p[i ^ 1][d] - p[i][d]);
        }
    return project(c, p[0][0], p[0][1], p[0][2], ax, ay) && project(c, p[1][0], p[1][1], p[1][2], bx, by);
}
static int utf8_len(const std::string& s) {
    int n = 0;
    for (unsigned char ch : s) n += (ch & 0xC0) != 0x80;
    return n;
}
// Scales of the turbulence from the header of a source: eta (Kolmogorov), lambda (Taylor), tau_eta. The simulation
// that produced the sequences takes kmax = n / 3; lambda = 15^(1/4) Re_lambda^(1/2) eta in isotropic turbulence.
static void turbulence_scales(int n, double u_rms, double kmax_eta, double re_lambda, double& eta, double& lambda, double& tau_eta) {
    if (!(kmax_eta > 0.0) || !(re_lambda > 0.0) || !(u_rms > 0.0) || n < 3) return;
    eta = kmax_eta / (double)(n / 3);
    lambda = eta * std::pow(15.0, 0.25) * std::sqrt(re_lambda);
    tau_eta = eta * eta / (u_rms * lambda / re_lambda);
}

// ---------------------------------------------------------------------------------------
// The two sides of the screen
// ---------------------------------------------------------------------------------------

struct Meter {
    double disk = 0.0, bus = 0.0, dec = 0.0, sort = 0.0, step = 0.0, render = 0.0;  // seconds, moving averages
    bool ready = false;
};

struct Side {
    std::string title, sub;
    int n = 0, n_frames = 0, first_frame = 0;
    float dt_frame = 0.0f, u_rms = 0.0f;
    double eta = 0.0, lambda = 0.0, tau_eta = 0.0;  // Kolmogorov and Taylor scales, Kolmogorov time (0: unknown)
    double re_lambda = 0.0;                          // Taylor Reynolds number (0: unknown)
    int latest = 0;           // most recent frame held (index in the source)
    long long advanced = 0;   // frames advanced since the launch
    double frame_mib = 0.0;   // MiB read for the last frame
    double vram_mib = 0.0;
    uint32_t np = 0;
    float *p[3] = {}, *q[3] = {};
    uint32_t *key = nullptr, *idx = nullptr;
    cudaStream_t st = nullptr;
    Meter m;
    double credit = 0.0, last_sort = 0.0;
    double div_ratio = -1.0, err_rms = -1.0;
    double err0 = -1.0;       // field error measured at launch, on the first frame
    double mib_sum = 0.0;     // MiB read since the launch, key frames included
    double mib_avg() const { return advanced ? mib_sum / (double)advanced : frame_mib; }

    virtual ~Side() {}
    virtual void ingest(int j, double& disk, double& bus, double& dec) = 0;
    virtual void advect(int jn, int substeps) = 0;  // from frame latest to frame jn
    virtual void splat(const Cam& c, int mode, float dl, float inv_scale, float* acc) = 0;
    virtual void probe(int mode, float dl, uint32_t count, float* out) = 0;
    virtual void div_slice(const SliceGeo& sg, float* out2) = 0;
    virtual void err_slice(const float4* truth, const SliceGeo& sg, float* out2) = 0;

    void init_particles(uint32_t count, unsigned seed) {
        if (!p[0]) {
            np = count;
            for (int c = 0; c < 3; ++c) {
                CK(cudaMalloc(&p[c], (size_t)np * sizeof(float)));
                CK(cudaMalloc(&q[c], (size_t)np * sizeof(float)));
            }
            CK(cudaMalloc(&key, (size_t)np * sizeof(uint32_t)));
            CK(cudaMalloc(&idx, (size_t)np * sizeof(uint32_t)));
        }
        std::vector<float> pos((size_t)np * 3);
        std::mt19937 rng(seed);
        std::uniform_real_distribution<float> U(0.0f, kL);
        for (auto& v : pos) v = U(rng);
        for (int c = 0; c < 3; ++c) CK(cudaMemcpy(p[c], pos.data() + (size_t)c * np, (size_t)np * sizeof(float), cudaMemcpyHostToDevice));
    }
    double sort_particles() {
        const auto t0 = std::chrono::steady_clock::now();
        morton_keys<<<blocks_for(np), 256, 0, st>>>(p[0], p[1], p[2], np, key, idx);
        CK(cudaStreamSynchronize(st));
        thrust::sort_by_key(thrust::device_pointer_cast(key), thrust::device_pointer_cast(key) + np, thrust::device_pointer_cast(idx));
        gather3<<<blocks_for(np), 256, 0, st>>>(p[0], p[1], p[2], idx, np, q[0], q[1], q[2]);
        CK(cudaStreamSynchronize(st));
        for (int c = 0; c < 3; ++c) std::swap(p[c], q[c]);
        return seconds_since(t0);
    }
    // Advances by one frame: read, bus, decoding, sort if any, then the RK4 steps of the interval.
    void step(int substeps, int nfe, bool sorted) {
        const int jn = (latest + 1) % nfe;
        double d = 0.0, b = 0.0, c = 0.0;
        ingest(jn, d, b, c);
        if (sorted && advanced % 16 == 0) last_sort = sort_particles();
        const auto t0 = std::chrono::steady_clock::now();
        advect(jn, substeps);
        CK(cudaStreamSynchronize(st));
        const double a = seconds_since(t0) / substeps, so = sorted ? last_sort / 16.0 : 0.0;
        latest = jn;
        ++advanced;
        mib_sum += frame_mib;
        const double k = m.ready ? 0.15 : 1.0;
        m.disk += k * (d - m.disk);
        m.bus += k * (b - m.bus);
        m.dec += k * (c - m.dec);
        m.step += k * (a - m.step);
        m.sort += k * (so - m.sort);
        m.ready = true;
    }
    double frame_seconds(int substeps) const { return m.disk + m.bus + m.dec + m.sort + substeps * m.step; }
};

template <int D>
struct PsiSide : Side {
    Receiver<D> rx;
    FILE* in = nullptr;
    std::vector<FrameRecord> index;
    unsigned char* h_buf = nullptr;
    int mf = 2;
    bool fine = false;  // file with an error bound (PSISTRM3): the sparse fine level is added to the field in all the views

    // so: what the opening read and checked (read_stream_index): header of the fine level of a PSISTRM3 file, frame
    // table, seal.
    PsiSide(const std::string& path, const StreamHeader& sh, const StreamOpen& so, FILE* f, cudaStream_t s) {
        st = s;
        in = f;
        fine = so.has_fine;
        const FineInfo* fine_info = fine ? &so.fi : nullptr;
        Geo g;
        g.build(sh.n, D, sh.mf, sh.levels, sh.fb);
        if (fine) {
            const size_t nb = (size_t)(sh.n >> kFineLog);
            g.cap += fine_section_cap(nb * nb * nb / 8, fine_info->max_bricks, fine_info->max_nodes);
        }
        index = so.index;
        g.cap = packet_bytes(index, g.cap);  // receive buffers at the size of the largest frame, as in play
        Coef coef[kMaxLevels];
        for (int l = 0; l < g.L; ++l) coef[l] = Coef{sh.coef[l][0], sh.coef[l][1], sh.coef[l][2]};
        rx.init(g, sh.delta, coef);
        rx.set_format(sh.magic[7], sh.fb);
        if (fine) rx.init_fine(*fine_info);
        CK(cudaHostAlloc(&h_buf, g.cap, cudaHostAllocDefault));
        n = sh.n;
        mf = sh.mf;
        n_frames = sh.n_frames;
        first_frame = sh.first_frame;
        dt_frame = sh.dt_frame;
        u_rms = sh.u_rms;
        turbulence_scales(n, u_rms, sh.kmax_eta, sh.re_lambda, eta, lambda, tau_eta);
        re_lambda = sh.re_lambda;
        const double file_mib = so.file_bytes / 1048576.0;
        char b[160];
        // On-screen text: in English, like the public interface of the project.
        if (fine) snprintf(b, sizeof b, "Psi-treillis, degree %d, %g %% bound", D, 100.0 * fine_info->tolerance);
        else snprintf(b, sizeof b, "Psi-treillis, degree %d", D);
        title = b;
        if (fine && fine_info->frames_over)
            snprintf(b, sizeof b, ".psi file of %.0f MiB, %d frames; bound missed on %u frames (%.2f %% of u')", file_mib, sh.n_frames, fine_info->frames_over,
                     100.0 * fine_info->max_error);
        else if (fine)
            snprintf(b, sizeof b, ".psi file of %.0f MiB, %d frames; largest certified error %.2f %% of u'", file_mib, sh.n_frames, 100.0 * fine_info->max_error);
        else snprintf(b, sizeof b, ".psi file of %.0f MiB, %d frames", file_mib, sh.n_frames);
        sub = b;
        (void)path;
    }
    void ingest(int j, double& disk, double& bus, double& dec) override {
        const FrameRecord& r = index[j];
        CHK(r.bytes <= rx.g.cap, "frame larger than expected");
        auto t0 = std::chrono::steady_clock::now();
        seek64(in, r.offset);
        CHK(fread(h_buf, 1, r.bytes, in) == r.bytes, "unreadable frame");
        disk = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        CK(cudaMemcpyAsync(rx.packet, h_buf, r.bytes, cudaMemcpyHostToDevice, st));
        rx.inspect(r.type, h_buf, r.bytes);  // check of the payload on the host, during the transfer, as in play
        CK(cudaStreamSynchronize(st));
        bus = seconds_since(t0);
        t0 = std::chrono::steady_clock::now();
        rx.decode(r.type, j & 1, r.mean, h_buf, r.bytes, st);
        CK(cudaStreamSynchronize(st));
        dec = seconds_since(t0);
        frame_mib = r.bytes / 1048576.0;
    }
    void advect(int jn, int substeps) override {
        (void)jn;
        const float dt = dt_frame / substeps, dw = 1.0f / substeps;
        for (int s = 0; s < substeps; ++s)
            if (fine)
                rk4_psi_fine<D, false><<<blocks_for(np, kCacheBlock), kCacheBlock, 0, st>>>(rx.field(latest & 1), rx.fs.level(D, rx.fine_delta), p[0], p[1], p[2], np, s * dw,
                                                                                         dw, dt);
            else rk4_psi<D><<<blocks_for(np, kCacheBlock), kCacheBlock, 0, st>>>(rx.field(latest & 1), p[0], p[1], p[2], np, s * dw, dw, dt);
    }
    PsiField<D> now() const { return rx.field((latest & 1) ^ 1); }  // with weight 1: the most recent frame
    PsiFine<D, PsiField<D>> now_fine() const {  // the complete field: base lattice and sparse fine level
        PsiFine<D, PsiField<D>> f{};
        f.base = now();
        f.fine = rx.fs.level(D, rx.fine_delta);
        return f;
    }
    void splat(const Cam& c, int mode, float dl, float inv_scale, float* acc) override {
        if (fine) splat_kernel<PsiFine<D, PsiField<D>>><<<blocks_for(np), 256, 0, st>>>(now_fine(), p[0], p[1], p[2], np, 1.0f, c, mode, dl, inv_scale, acc);
        else splat_kernel<PsiField<D>><<<blocks_for(np), 256, 0, st>>>(now(), p[0], p[1], p[2], np, 1.0f, c, mode, dl, inv_scale, acc);
    }
    void probe(int mode, float dl, uint32_t count, float* out) override {
        if (fine) probe_kernel<PsiFine<D, PsiField<D>>><<<blocks_for(count), 256, 0, st>>>(now_fine(), p[0], p[1], p[2], count, 1.0f, mode, dl, out);
        else probe_kernel<PsiField<D>><<<blocks_for(count), 256, 0, st>>>(now(), p[0], p[1], p[2], count, 1.0f, mode, dl, out);
    }
    void div_slice(const SliceGeo& sg, float* out2) override {
        if (fine) div_slice_psi_fine<D><<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(now(), rx.fs.level(D, rx.fine_delta), 1.0f, sg, out2);
        else div_slice_psi<D><<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(now(), 1.0f, sg, out2);
    }
    void err_slice(const float4* truth, const SliceGeo& sg, float* out2) override {
        if (fine) err_slice_kernel<PsiFine<D, PsiField<D>>><<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(now_fine(), truth, n, sg, out2);
        else err_slice_kernel<PsiField<D>><<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(now(), truth, n, sg, out2);
    }
};

struct ZfsHeader {
    char magic[8];  // "ZFPSTRM1"
    int32_t n, n_frames, first_frame, reserved;
    double rate;
    float dt_frame, u_rms;
    uint64_t span;  // bytes per frame
};

struct TexSide : Side {
    bool zfp = false;
    FILE* f = nullptr;
    SequenceHeader hd{};
    uint64_t span = 0;
    size_t voxels = 0;
    cudaArray_t arr[2] = {};
    cudaTextureObject_t lin[2] = {}, pt[2] = {};
    cudaSurfaceObject_t surf[2] = {};
    float *d_planar = nullptr, *h_planar = nullptr;
    void *h_z = nullptr, *d_z = nullptr;
    ZfpCodec codec;

    void build(int n_) {
        n = n_;
        voxels = (size_t)n * n * n;
        const cudaChannelFormatDesc desc = cudaCreateChannelDesc(16, 16, 16, 16, cudaChannelFormatKindFloat);
        for (int s = 0; s < 2; ++s) {
            CK(cudaMalloc3DArray(&arr[s], &desc, make_cudaExtent(n, n, n), cudaArraySurfaceLoadStore));
            cudaResourceDesc res{};
            res.resType = cudaResourceTypeArray;
            res.res.array.array = arr[s];
            cudaTextureDesc td{};
            td.addressMode[0] = td.addressMode[1] = td.addressMode[2] = cudaAddressModeWrap;
            td.filterMode = cudaFilterModeLinear;
            td.readMode = cudaReadModeElementType;
            td.normalizedCoords = 1;
            CK(cudaCreateTextureObject(&lin[s], &res, &td, nullptr));
            td.filterMode = cudaFilterModePoint;
            CK(cudaCreateTextureObject(&pt[s], &res, &td, nullptr));
            CK(cudaCreateSurfaceObject(&surf[s], &res));
        }
        CK(cudaMalloc(&d_planar, 3 * voxels * sizeof(float)));
    }
    // cuZFP stream stored in a file by "psi_view packzfp".
    TexSide(const std::string& path, const ZfsHeader& zh, FILE* file, cudaStream_t s) {
        st = s;
        zfp = true;
        f = file;
        build(zh.n);
        n_frames = zh.n_frames;
        first_frame = zh.first_frame;
        dt_frame = zh.dt_frame;
        u_rms = zh.u_rms;
        span = zh.span;
        codec.configure(n, zh.rate);
        CHK(codec.span == span, "the .zfs file does not match this rate");
        CK(cudaHostAlloc(&h_z, span, cudaHostAllocDefault));
        CK(cudaMalloc(&d_z, span));
        codec.attach(n, (float4*)d_planar, d_planar, d_z);  // the first buffer is used only for compression
        char b[160];
        snprintf(b, sizeof b, "cuZFP %g bits + half-precision 3D texture", zh.rate);
        title = b;
        snprintf(b, sizeof b, ".zfs file of %.0f MiB, %d frames", (sizeof(ZfsHeader) + (double)span * zh.n_frames) / 1048576.0, zh.n_frames);
        sub = b;
        (void)path;
    }
    // Raw float32 frames of the sequence.
    TexSide(const std::string& path, FILE* file, int first, int count, cudaStream_t s) {
        st = s;
        f = file;
        hd = sequence_read_header(f);
        build(hd.n);
        first_frame = first;
        n_frames = count > 0 ? std::min(count, hd.n_frames - first) : hd.n_frames - first;
        CHK(n_frames >= 2, "not enough frames in the sequence");
        dt_frame = hd.dt_frame;
        u_rms = hd.u_rms;
        turbulence_scales(n, u_rms, hd.kmax_eta, hd.re_lambda, eta, lambda, tau_eta);
        re_lambda = hd.re_lambda;
        CK(cudaHostAlloc(&h_planar, 3 * voxels * sizeof(float), cudaHostAllocDefault));
        title = "Raw frames + half-precision 3D texture";
        char b[160];
        snprintf(b, sizeof b, "raw float32 sequence of %.1f GiB, %d frames", (double)sequence_frame_bytes(hd) * n_frames / 1073741824.0, n_frames);
        sub = b;
        (void)path;
    }
    void store(int slot) {
        const dim3 block(16, 4, 4), grid((n + 15) / 16, (n + 3) / 4, (n + 3) / 4);
        planes_to_half_surface<<<grid, block, 0, st>>>(d_planar, d_planar + voxels, d_planar + 2 * voxels, surf[slot], n);
    }
    void ingest(int j, double& disk, double& bus, double& dec) override {
        auto t0 = std::chrono::steady_clock::now();
        if (zfp) {
            seek64(f, sizeof(ZfsHeader) + (uint64_t)j * span);
            CHK(fread(h_z, 1, span, f) == span, "unreadable cuZFP frame");
            disk = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            CK(cudaMemcpyAsync(d_z, h_z, span, cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
            bus = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            codec.decompress();
            CK(cudaDeviceSynchronize());
            frame_mib = span / 1048576.0;
        } else {
            sequence_read_frame(f, hd, first_frame + j, h_planar);
            disk = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            CK(cudaMemcpyAsync(d_planar, h_planar, 3 * voxels * sizeof(float), cudaMemcpyHostToDevice, st));
            CK(cudaStreamSynchronize(st));
            bus = seconds_since(t0);
            t0 = std::chrono::steady_clock::now();
            frame_mib = 3.0 * voxels * sizeof(float) / 1048576.0;
        }
        store(j & 1);
        CK(cudaStreamSynchronize(st));
        dec = seconds_since(t0);
    }
    void advect(int jn, int substeps) override {
        const float dt = dt_frame / substeps, dw = 1.0f / substeps;
        const TexField fld{lin[latest & 1], lin[jn & 1], 1.0f / kL};
        for (int s = 0; s < substeps; ++s) rk4_kernel<TexField><<<blocks_for(np), 256, 0, st>>>(fld, p[0], p[1], p[2], np, s * dw, dw, dt);
    }
    TexField now() const { return TexField{lin[(latest & 1) ^ 1], lin[latest & 1], 1.0f / kL}; }
    void splat(const Cam& c, int mode, float dl, float inv_scale, float* acc) override {
        splat_kernel<TexField><<<blocks_for(np), 256, 0, st>>>(now(), p[0], p[1], p[2], np, 1.0f, c, mode, dl, inv_scale, acc);
    }
    void probe(int mode, float dl, uint32_t count, float* out) override {
        probe_kernel<TexField><<<blocks_for(count), 256, 0, st>>>(now(), p[0], p[1], p[2], count, 1.0f, mode, dl, out);
    }
    void div_slice(const SliceGeo& sg, float* out2) override {
        div_slice_tex<<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(TexPoint{pt[(latest & 1) ^ 1], pt[latest & 1], n}, 1.0f, sg, out2);
    }
    void err_slice(const float4* truth, const SliceGeo& sg, float* out2) override {
        err_slice_kernel<TexField><<<blocks_for((size_t)sg.vw * sg.vh), 256, 0, st>>>(now(), truth, n, sg, out2);
    }
};

// ---------------------------------------------------------------------------------------
// Options, sources
// ---------------------------------------------------------------------------------------

struct VOptions {
    std::string left, right, data, file, snapshot;
    uint32_t particles = 4000000;
    int substeps = 2, width = 1600, height = 900, view = 0, iters = 40, start = -1, frames = 0, color = 1;
    double rate = 2.0;
    bool race = true, sort = false;
    unsigned seed = 1234;
    float yaw = 0.62f, pitch = 0.38f;  // initial angles of the 3D view, in radians
};

static VOptions vparse(int argc, char** argv, int first) {
    VOptions o;
    for (int i = first; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> const char* {
            if (i + 1 >= argc) {
                fprintf(stderr, "Missing value for %s\n", a.c_str());
                exit(2);
            }
            return argv[++i];
        };
        if (a == "--left") o.left = next();
        else if (a == "--right") o.right = next();
        else if (a == "--data") o.data = next();
        else if (a == "--file") o.file = next();
        else if (a == "--snapshot") o.snapshot = next();
        else if (a == "--particles") o.particles = (uint32_t)atoi(next());
        else if (a == "--substeps") o.substeps = std::max(1, atoi(next()));
        else if (a == "--width") o.width = atoi(next());
        else if (a == "--height") o.height = atoi(next());
        else if (a == "--view") o.view = std::min(2, std::max(0, atoi(next()) - 1));
        else if (a == "--iters") o.iters = atoi(next());
        else if (a == "--start") o.start = atoi(next());
        else if (a == "--frames") o.frames = atoi(next());
        else if (a == "--rate") o.rate = atof(next());
        else if (a == "--seed") o.seed = (unsigned)atoi(next());
        else if (a == "--color") o.color = std::string(next()) == "speed" ? 0 : 1;
        else if (a == "--sync") o.race = false;
        else if (a == "--sort") o.sort = true;
        else if (a == "--yaw") o.yaw = (float)atof(next());
        else if (a == "--pitch") o.pitch = (float)atof(next());
        else {
            fprintf(stderr,
                    "Usage: psi_view packzfp --data f.seq --file f.zfs [--rate 2] [--start 0] [--frames 64]\n"
                    "       psi_view --left SOURCE --right SOURCE [--data f.seq] [--particles 4000000] [--substeps 2]\n"
                    "                [--width 1600 --height 900] [--view 1|2|3] [--sync] [--sort] [--color vort|speed]\n"
                    "                [--start frame --frames n] [--snapshot image.png --iters 40] [--yaw 0.62 --pitch 0.38]\n"
                    "       SOURCE = psi:file.psi | zfp:file.zfs | raw:file.seq\n");
            exit(a == "--help" ? 0 : 2);
        }
    }
    return o;
}

static int packzfp(const VOptions& o) {
    FILE* in = fopen(o.data.c_str(), "rb");
    CHK(in, "sequence not found (--data)");
    const SequenceHeader hd = sequence_read_header(in);
    const int n = hd.n, start = std::max(0, o.start), count = o.frames > 0 ? std::min(o.frames, hd.n_frames - start) : hd.n_frames - start;
    CHK(count >= 2, "not enough frames");
    const size_t voxels = (size_t)n * n * n;
    FILE* out = fopen(o.file.c_str(), "wb");
    CHK(out, "cannot create the file (--file)");
    ZfpCodec codec;
    codec.configure(n, o.rate);
    float *h_planar = nullptr, *d_planar = nullptr;
    float4* d_src = nullptr;
    void* d_z = nullptr;
    CK(cudaHostAlloc(&h_planar, 3 * voxels * sizeof(float), cudaHostAllocDefault));
    CK(cudaMalloc(&d_planar, 3 * voxels * sizeof(float)));
    CK(cudaMalloc(&d_src, voxels * sizeof(float4)));
    CK(cudaMalloc(&d_z, codec.span));
    std::vector<unsigned char> h_z(codec.span);
    codec.attach(n, d_src, d_planar, d_z);
    ZfsHeader zh{};
    memcpy(zh.magic, "ZFPSTRM1", 8);
    zh.n = n;
    zh.n_frames = count;
    zh.first_frame = start;
    zh.rate = codec.rate;
    zh.dt_frame = hd.dt_frame;
    zh.u_rms = hd.u_rms;
    zh.span = codec.span;
    CHK(fwrite(&zh, sizeof(zh), 1, out) == 1, "write failed");
    for (int k = 0; k < count; ++k) {
        sequence_read_frame(in, hd, start + k, h_planar);
        CK(cudaMemcpy(d_planar, h_planar, 3 * voxels * sizeof(float), cudaMemcpyHostToDevice));
        planar_to_f4<<<blocks_for(voxels), 256>>>(d_planar, d_src, voxels);
        CK(cudaMemset(d_z, 0, codec.span));
        CK(cudaDeviceSynchronize());
        codec.compress();
        CK(cudaMemcpy(h_z.data(), d_z, codec.span, cudaMemcpyDeviceToHost));
        CHK(fwrite(h_z.data(), 1, codec.span, out) == codec.span, "write failed");
    }
    fclose(out);
    fclose(in);
    printf("packzfp        : %s, %d^3, frames %d to %d, cuZFP %.2f bits per value\n", o.data.c_str(), n, start, start + count - 1, codec.rate);
    printf("File           : %s, %.1f MiB, that is %.2f MiB per frame (%.1fx less than the raw frame)\n", o.file.c_str(),
           (sizeof(zh) + (double)codec.span * count) / 1048576.0, codec.span / 1048576.0, 3.0 * voxels * sizeof(float) / (double)codec.span);
    return 0;
}

static Side* open_side(const std::string& spec, const VOptions& o, cudaStream_t st, int first_hint, int count_hint) {
    const size_t colon = spec.find(':');
    CHK(colon != std::string::npos, "expected source: psi:file, zfp:file or raw:file");
    const std::string kind = spec.substr(0, colon), path = spec.substr(colon + 1);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "File not found: %s\n", path.c_str());
        exit(1);
    }
    if (kind == "psi") {
        StreamHeader sh{};
        CHK(fread(&sh, sizeof(sh), 1, f) == 1, "invalid .psi file");
        check_stream_header(sh);  // magic, degree, resolution, number of frames, frame table
        // Same opening checks as play: size of the file, header of the fine level (file with an error bound), frame
        // table, seal. A seal that does not match stops the program; its absence is reported.
        StreamOpen so;
        read_stream_index(f, sh, so);
        printf("%s\n", path.c_str());
        print_seal(so);
        if (sh.degree == 3) return new PsiSide<3>(path, sh, so, f, st);
        return new PsiSide<2>(path, sh, so, f, st);
    }
    if (kind == "zfp") {
        ZfsHeader zh{};
        CHK(fread(&zh, sizeof(zh), 1, f) == 1 && memcmp(zh.magic, "ZFPSTRM1", 8) == 0, "not a valid .zfs file");
        return new TexSide(path, zh, f, st);
    }
    CHK(kind == "raw", "unknown source (psi, zfp or raw)");
    return new TexSide(path, f, o.start >= 0 ? o.start : std::max(0, first_hint), o.frames > 0 ? o.frames : count_hint, st);
}

// Numbers displayed on screen: decimal point, the interface being in English.
static std::string num(double v, int prec) {
    char b[64];
    snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
static std::string sci(double v) {
    char b[64];
    snprintf(b, sizeof b, "%.1e", v);
    return b;
}

// ---------------------------------------------------------------------------------------
// Program
// ---------------------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "packzfp") return packzfp(vparse(argc, argv, 2));
    VOptions o = vparse(argc, argv, 1);
    CHK(!o.left.empty() && !o.right.empty(), "--left and --right are required (see --help)");
    const bool snap = !o.snapshot.empty();
    int W = o.width, H = o.height;
    if (!snap) view_fit(W, H);  // never larger than the screen
    W &= ~1;
    cudaStream_t st;
    CK(cudaStreamCreate(&st));

    // The sources with a header first: a raw source takes the same range of frames as the other side.
    Side* side[2] = {nullptr, nullptr};
    const std::string spec[2] = {o.left, o.right};
    size_t free0 = 0, free1 = 0, total = 0;
    for (int pass = 0; pass < 2; ++pass)
        for (int k = 0; k < 2; ++k) {
            const bool raw = spec[k].compare(0, 4, "raw:") == 0;
            if (raw != (pass == 1)) continue;
            const Side* other = side[k ^ 1];
            CK(cudaMemGetInfo(&free0, &total));
            side[k] = open_side(spec[k], o, st, other ? other->first_frame : 0, other ? other->n_frames : 0);
            side[k]->init_particles(o.particles, o.seed);
            CK(cudaMemGetInfo(&free1, &total));
            side[k]->vram_mib = (free0 - free1) / 1048576.0;
        }
    CHK(side[0]->n == side[1]->n, "the two sources do not have the same resolution");
    CHK(side[0]->first_frame == side[1]->first_frame, "the two sources do not start at the same frame (see --start)");
    const int n = side[0]->n, nfe = std::min(side[0]->n_frames, side[1]->n_frames) & ~1;
    CHK(nfe >= 2, "at least two common frames are required");
    const float h = kL / n, u_rms = side[0]->u_rms;
    // Physical scales of the reference marks: those of one of the two sides, otherwise those of the --data sequence.
    double eta = 0.0, lambda = 0.0, tau_eta = 0.0, re_lambda = 0.0;
    for (Side* s : side)
        if (eta == 0.0 && s->eta > 0.0) {
            eta = s->eta;
            lambda = s->lambda;
            tau_eta = s->tau_eta;
            re_lambda = s->re_lambda;
        }

    // Initial state: frame 0 in the two slots of each side.
    for (Side* s : side) {
        double d, b, c;
        s->ingest(1, d, b, c);
        s->ingest(0, d, b, c);
        s->latest = 0;
        if (o.sort) s->last_sort = s->sort_particles();
    }

    CHK(view_open(W, H, "Psi-treillis \xE2\x80\x94 visual comparator", !snap), "cannot create the window");
    // From top to bottom: the two views, the band of the measured differences, the status bar.
    const int bar_h = 54, band_h = 132;
    const int vw = W / 2 - 1, vh = H - bar_h - band_h;
    uint32_t* d_img = nullptr;
    float *d_acc = nullptr, *d_out2 = nullptr, *h_out2 = nullptr, *d_probe = nullptr;
    CK(cudaMalloc(&d_img, (size_t)W * H * sizeof(uint32_t)));
    CK(cudaMalloc(&d_acc, (size_t)2 * W * H * sizeof(float)));
    CK(cudaMalloc(&d_out2, (size_t)2 * vw * vh * sizeof(float)));
    CK(cudaHostAlloc(&h_out2, (size_t)2 * vw * vh * sizeof(float), cudaHostAllocDefault));
    const uint32_t n_probe = std::min<uint32_t>(o.particles, 1u << 16);
    CK(cudaMalloc(&d_probe, n_probe * sizeof(float)));

    // Raw reference frame for the error view.
    FILE* seq = nullptr;
    SequenceHeader seq_hd{};
    float *h_tp = nullptr, *d_tp = nullptr;
    float4* d_truth = nullptr;
    int truth_frame = -1;
    const size_t voxels = (size_t)n * n * n;
    if (!o.data.empty()) {
        seq = fopen(o.data.c_str(), "rb");
        CHK(seq, "sequence not found (--data)");
        seq_hd = sequence_read_header(seq);
        if (eta == 0.0) {
            turbulence_scales(seq_hd.n, seq_hd.u_rms, seq_hd.kmax_eta, seq_hd.re_lambda, eta, lambda, tau_eta);
            re_lambda = seq_hd.re_lambda;
        }
        CHK(seq_hd.n == n, "the --data sequence does not have the resolution of the sources");
    }
    auto ensure_truth = [&](int j) {
        if (truth_frame == j) return;
        if (!d_truth) {
            CK(cudaHostAlloc(&h_tp, 3 * voxels * sizeof(float), cudaHostAllocDefault));
            CK(cudaMalloc(&d_tp, 3 * voxels * sizeof(float)));
            CK(cudaMalloc(&d_truth, voxels * sizeof(float4)));
        }
        sequence_read_frame(seq, seq_hd, side[0]->first_frame + j, h_tp);
        CK(cudaMemcpy(d_tp, h_tp, 3 * voxels * sizeof(float), cudaMemcpyHostToDevice));
        planar_to_f4<<<blocks_for(voxels), 256>>>(d_tp, d_truth, voxels);
        CK(cudaDeviceSynchronize());
        truth_frame = j;
    };

    // State of the interface.
    int view = o.view, color = o.color, substeps = o.substeps;
    bool race = o.race, paused = false, sorted = o.sort;
    if (view == 2 && !seq) view = 0;
    if (view == 2) race = false;  // the error view compares the two sides on the same frame
    float yaw = o.yaw, pitch = std::min(1.5f, std::max(-1.5f, o.pitch)), dist = 3.2f * kL, bright = 1.0f;
    bool fog = true, spin = false;  // attenuation with depth; slow rotation of the view
    float cx = 0.5f * kL, cy = 0.5f * kL, zs = 0.5f * kL + 0.37f * h, zoom = 1.0f;
    float inv_scale = 1.0f;
    const float dl = 0.5f * h;
    auto rescale = [&]() {  // the coloured quantity is relative to three times its mean, measured on the left side
        side[0]->probe(color, dl, n_probe, d_probe);
        std::vector<float> v(n_probe);
        CK(cudaMemcpy(v.data(), d_probe, n_probe * sizeof(float), cudaMemcpyDeviceToHost));
        double s = 0.0;
        for (float x : v) s += x;
        inv_scale = (float)(1.0 / (3.0 * s / n_probe + 1e-30));
    };
    rescale();

    auto make_cam = [&](int vx) {
        Cam c{};
        const float cp = cosf(pitch), sp = sinf(pitch), cyw = cosf(yaw), syw = sinf(yaw);
        const float f[3] = {-cp * syw, -sp, -cp * cyw};
        float r[3] = {-f[2], 0.0f, f[0]};
        const float rn = sqrtf(r[0] * r[0] + r[2] * r[2]);
        r[0] /= rn;
        r[2] /= rn;
        const float u[3] = {r[1] * f[2] - r[2] * f[1], r[2] * f[0] - r[0] * f[2], r[0] * f[1] - r[1] * f[0]};
        for (int i = 0; i < 3; ++i) {
            c.fwd[i] = f[i];
            c.right[i] = r[i];
            c.up[i] = u[i];
            c.eye[i] = 0.5f * kL - dist * f[i];
        }
        c.focal = 0.5f * vh / tanf(0.5f * 0.66f);
        c.pcy = 0.56f * vh;  // the box is moved down below the text panel
        const float half = 0.87f * kL;  // half-diagonal of the box: weight 1.45 at the nearest, 0.55 at the back
        c.fog_a = fog ? 1.0f + 0.45f * dist / half : 1.0f;
        c.fog_b = fog ? 0.45f / half : 0.0f;
        c.vx = vx;
        c.vy = 0;
        c.vw = vw;
        c.vh = vh;
        c.W = W;
        return c;
    };
    auto slice_geo = [&](int vx) { return SliceGeo{cx, cy, zs, kL / std::min(vw, vh) / zoom, vx, 0, vw, vh, W}; };
    const int vx_of[2] = {0, W / 2 + 1};

    // Divergence of the field on a small slice (150 x 150 points), drawn or not as an inset.
    const int vs = 150;
    auto div_probe = [&](int k, bool draw) {
        Side& s = *side[k];
        const SliceGeo sv{cx, cy, zs, kL / vs, vx_of[k] + vw - vs - 10, vh - vs - 10, vs, vs, W};
        s.div_slice(sv, d_out2);
        CK(cudaMemcpyAsync(h_out2, d_out2, (size_t)2 * vs * vs * sizeof(float), cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        double a = 0.0, b = 0.0;
        for (size_t i = 0; i < (size_t)vs * vs; ++i) {
            a += (double)h_out2[2 * i] * h_out2[2 * i];
            b += h_out2[2 * i + 1];
        }
        s.div_ratio = std::sqrt(a / b);
        if (draw) colorize_div<<<blocks_for((size_t)vs * vs), 256, 0, st>>>(d_out2, d_img, sv, (float)(1.0 / std::sqrt(b / ((double)vs * vs))));
    };
    div_probe(0, false);
    div_probe(1, false);
    // Field error of each side against the raw frame, measured once at launch (a single read of a raw frame). The two
    // sides first advance by twelve frames: just after a key frame, the error of Psi-treillis is lower than in steady
    // state, and that would flatter it.
    int err_frame = 0;
    if (seq) {
        for (int i = 0; i < std::min(12, nfe - 1); ++i)
            for (Side* s : side) s->step(substeps, nfe, sorted);
        for (Side* s : side) {
            s->advanced = 0;
            s->mib_sum = 0.0;
            s->credit = 0.0;
        }
        err_frame = side[0]->latest;
        ensure_truth(err_frame);
        const int es = 256;
        for (Side* s : side) {
            const SliceGeo se{cx, cy, zs, kL / es, 0, 0, es, es, W};
            s->err_slice(d_truth, se, d_out2);
            CK(cudaMemcpyAsync(h_out2, d_out2, (size_t)2 * es * es * sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            double a = 0.0, b = 0.0;
            for (size_t i = 0; i < (size_t)es * es; ++i) {
                a += (double)h_out2[2 * i] * h_out2[2 * i];
                b += h_out2[2 * i + 1];
            }
            s->err0 = std::sqrt(a / b);
        }
    }

    auto render_side = [&](int k) {
        Side& s = *side[k];
        const auto t0 = std::chrono::steady_clock::now();
        if (view == 0) {
            const Cam c = make_cam(vx_of[k]);
            s.splat(c, color, dl, inv_scale, d_acc);
            const float gain = bright * 0.9f / (0.2f * (float)s.np / (0.45f * vw * vh));
            compose_particles<<<blocks_for((size_t)vw * vh), 256, 0, st>>>(d_acc, d_img, c, gain);
            div_probe(k, true);  // inset: the divergence slice, computed live
        } else {
            const SliceGeo sg = slice_geo(vx_of[k]);
            if (view == 1) s.div_slice(sg, d_out2);
            else s.err_slice(d_truth, sg, d_out2);
            CK(cudaMemcpyAsync(h_out2, d_out2, (size_t)2 * vw * vh * sizeof(float), cudaMemcpyDeviceToHost, st));
            CK(cudaStreamSynchronize(st));
            double a = 0.0, b = 0.0;
            for (size_t i = 0; i < (size_t)vw * vh; ++i) {
                a += (double)h_out2[2 * i] * h_out2[2 * i];
                b += h_out2[2 * i + 1];
            }
            if (view == 1) {
                s.div_ratio = std::sqrt(a / b);
                colorize_div<<<blocks_for((size_t)vw * vh), 256, 0, st>>>(d_out2, d_img, sg, (float)(1.0 / std::sqrt(b / ((double)vw * vh))));
            } else {
                s.err_rms = std::sqrt(a / b);
                colorize_lin<<<blocks_for((size_t)vw * vh), 256, 0, st>>>(d_out2, d_img, sg, 1.0f / (0.05f * 1.7320508f * u_rms));
            }
        }
        CK(cudaStreamSynchronize(st));
        const double t = seconds_since(t0);
        s.m.render += (s.m.render > 0.0 ? 0.15 : 1.0) * (t - s.m.render);
    };

    auto step_both = [&]() {
        side[0]->step(substeps, nfe, sorted);
        side[1]->step(substeps, nfe, sorted);
    };
    auto align = [&]() {  // same frame on both sides
        for (int guard = 0; guard < nfe && side[0]->latest != side[1]->latest; ++guard) {
            Side* lag = side[0]->advanced <= side[1]->advanced ? side[0] : side[1];
            lag->step(substeps, nfe, sorted);
        }
        side[0]->credit = side[1]->credit = 0.0;
    };
    auto advance = [&]() {
        if (!race || !side[0]->m.ready || !side[1]->m.ready) {
            if (!race) align();
            step_both();
            return;
        }
        // Race: the simulated interval is the cost of one frame of the faster side; the other advances only when its
        // time credit covers the measured cost of its own frame.
        const double c[2] = {side[0]->frame_seconds(substeps), side[1]->frame_seconds(substeps)};
        const double tau = std::min(c[0], c[1]);
        for (int k = 0; k < 2; ++k) {
            side[k]->credit += tau;
            for (int guard = 0; guard < 4 && side[k]->credit + 1e-9 >= c[k]; ++guard) {
                side[k]->step(substeps, nfe, sorted);
                side[k]->credit -= c[k];
            }
        }
    };

    double loop_s = 0.0;  // duration of one loop iteration (the two chains, the rendering, the display), moving average

    // ---- Reference marks. Pass 0: lines, drawn before the text panels; pass 1: texts and legends.
    const uint32_t ax_col[3] = {0xF07C6A, 0x86D985, 0x74AEFA};  // axes x, y, z
    const uint32_t c_soft = 0xB8C0CC, c_accent = 0xFFC857, c_dim = 0x7C8694, c_edge = 0x97A3B5;
    const uint32_t c_slice = 0x6FE8FF;  // slice plane: same colour in the 3D view, around the inset and in the miniature cube
    auto text_c = [&](float x, float y, uint32_t rgb, const std::string& t) {  // text centred on (x, y), in the area of the views
        if (y < 10.0f || y > (float)vh - 10.0f) return;
        view_text((int)lroundf(x - 4.7f * (float)utf8_len(t)), (int)lroundf(y) - 10, rgb, t.c_str());
    };
    auto text_r = [&](int x, int y, uint32_t rgb, const std::string& t) {  // text right-aligned on x
        view_text(x - (int)lroundf(9.4f * (float)utf8_len(t)), y, rgb, t.c_str());
    };
    static const int box_edge[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};

    // 3D view: grid of the floor and of the back walls, edges of the box, axes graduated in voxels, slice plane,
    // orientation triad; then physical scales and colour scale.
    auto frame3d = [&](int k, int pass) {
        const Side& s = *side[k];
        const Cam c = make_cam(vx_of[k]);
        const Clip cl{vx_of[k], 0, vx_of[k] + vw, vh};
        uint32_t* px = view_pixels();
        auto seg = [&](float x0, float y0, float z0, float x1, float y1, float z1, uint32_t rgb, float alpha, bool behind, bool thick) {
            const float a[3] = {x0, y0, z0}, b[3] = {x1, y1, z1};
            float ax, ay, bx, by;
            if (!seg3d(c, a, b, ax, ay, bx, by)) return;
            blend_line(px, W, cl, ax, ay, bx, by, rgb, alpha, behind);
            if (thick) {
                const bool steep = fabsf(by - ay) > fabsf(bx - ax);
                blend_line(px, W, cl, ax + (steep ? 1.0f : 0.0f), ay + (steep ? 0.0f : 1.0f), bx + (steep ? 1.0f : 0.0f), by + (steep ? 0.0f : 1.0f), rgb, alpha, behind);
            }
        };
        const float nr[3] = {c.fwd[0] < 0.0f ? kL : 0.0f, c.fwd[1] < 0.0f ? kL : 0.0f, c.fwd[2] < 0.0f ? kL : 0.0f};  // corner nearest to the eye
        const float fa[3] = {kL - nr[0], kL - nr[1], kL - nr[2]};                                                        // farthest corner
        const float yf = fa[1];  // the gridded "floor": the horizontal face at the back
        // The x and z axes are carried, among the four parallel edges, by the one that projects lowest: the numbers
        // stay under the box, whether the eye is above or below.
        float xy = 0.0f, xz = 0.0f, zy = 0.0f, zx = 0.0f, low = -1e30f, px0 = 0.0f, py0 = 0.0f;
        for (int j = 0; j < 4; ++j)
            if (project(c, 0.5f * kL, (float)(j & 1) * kL, (float)(j >> 1) * kL, px0, py0) && py0 > low) {
                low = py0;
                xy = (float)(j & 1) * kL;
                xz = (float)(j >> 1) * kL;
            }
        low = -1e30f;
        for (int j = 0; j < 4; ++j)
            if (project(c, (float)(j >> 1) * kL, (float)(j & 1) * kL, 0.5f * kL, px0, py0) && py0 > low) {
                low = py0;
                zy = (float)(j & 1) * kL;
                zx = (float)(j >> 1) * kL;
            }
        const float ox = xz > 0.0f ? 1.0f : -1.0f;  // outward direction, along z, of the edge that carries the x axis
        const float oz = zx > 0.0f ? 1.0f : -1.0f;  // outward direction, along x, of the edge that carries the z axis
        // The y axis is carried by the one of the two vertical silhouette edges that projects furthest to the left.
        const float cand[2][2] = {{nr[0], fa[2]}, {fa[0], nr[2]}};
        float sa = 0.0f, sb = 0.0f, tmp = 0.0f;
        project(c, cand[0][0], 0.5f * kL, cand[0][1], sa, tmp);
        project(c, cand[1][0], 0.5f * kL, cand[1][1], sb, tmp);
        const float yx = sa <= sb ? cand[0][0] : cand[1][0], yz = sa <= sb ? cand[0][1] : cand[1][1];
        const float tk = 0.03f * kL;  // length of the tick marks
        if (pass == 0) {
            for (int i = 0; i <= 8; ++i) {  // floor: one line every n / 8 voxels
                const float t = (float)i * kL / 8.0f;
                seg(t, yf, 0.0f, t, yf, kL, 0x2E3642, 1.0f, true, false);
                seg(0.0f, yf, t, kL, yf, t, 0x2E3642, 1.0f, true, false);
            }
            for (int i = 0; i <= 4; ++i) {  // back walls: one line every n / 4 voxels
                const float t = (float)i * kL / 4.0f;
                seg(fa[0], 0.0f, t, fa[0], kL, t, 0x242B34, 1.0f, true, false);
                seg(fa[0], t, 0.0f, fa[0], t, kL, 0x242B34, 1.0f, true, false);
                seg(t, 0.0f, fa[2], t, kL, fa[2], 0x242B34, 1.0f, true, false);
                seg(0.0f, t, fa[2], kL, t, fa[2], 0x242B34, 1.0f, true, false);
            }
            const int far_code = (fa[0] > 0.0f ? 1 : 0) | (fa[1] > 0.0f ? 2 : 0) | (fa[2] > 0.0f ? 4 : 0);
            for (int e = 0; e < 12; ++e) {  // edges: those of the back corner pass behind the particles
                const int a = box_edge[e][0], b = box_edge[e][1];
                const bool back = a == far_code || b == far_code;
                seg((a & 1) * kL, ((a >> 1) & 1) * kL, ((a >> 2) & 1) * kL, (b & 1) * kL, ((b >> 1) & 1) * kL, ((b >> 2) & 1) * kL, back ? 0x3A4350 : c_edge,
                    back ? 1.0f : 0.5f, back, false);
            }
            // Slice plane z = zs: that of the divergence inset and of views 2 and 3. Its sides carried by the faces
            // turned towards the eye are drawn on top, the others pass behind the particles.
            seg(0.0f, 0.0f, zs, kL, 0.0f, zs, c_slice, 0.9f, nr[1] != 0.0f, true);
            seg(0.0f, kL, zs, kL, kL, zs, c_slice, 0.9f, nr[1] == 0.0f, true);
            seg(0.0f, 0.0f, zs, 0.0f, kL, zs, c_slice, 0.9f, nr[0] != 0.0f, true);
            seg(kL, 0.0f, zs, kL, kL, zs, c_slice, 0.9f, nr[0] == 0.0f, true);
            // Graduated axes, on the silhouette edges.
            seg(0.0f, xy, xz, kL, xy, xz, ax_col[0], 0.95f, false, true);
            seg(zx, zy, 0.0f, zx, zy, kL, ax_col[2], 0.95f, false, true);
            seg(yx, 0.0f, yz, yx, kL, yz, ax_col[1], 0.95f, false, true);
            for (int i = 0; i <= 4; ++i) {
                const float t = (float)i * kL / 4.0f;
                seg(t, xy, xz, t, xy, xz + ox * tk, ax_col[0], 0.95f, false, false);
                seg(zx, zy, t, zx + oz * tk, zy, t, ax_col[2], 0.95f, false, false);
                seg(yx, t, yz, yx - c.right[0] * tk, t, yz - c.right[2] * tk, ax_col[1], 0.95f, false, false);
            }
            // Orientation triad, bottom left: the three axes seen from the angle of the camera, the farthest first.
            const float tx = (float)vx_of[k] + 56.0f, ty = (float)vh - 60.0f, len = 38.0f;
            int order[3] = {0, 1, 2};
            std::sort(order, order + 3, [&](int a, int b) { return c.fwd[a] > c.fwd[b]; });
            for (int j = 0; j < 3; ++j) {
                const int i = order[j];
                const float ex = tx + len * c.right[i], ey = ty - len * c.up[i];
                blend_line(px, W, cl, tx, ty, ex, ey, ax_col[i], 1.0f, false);
                blend_line(px, W, cl, tx + 1.0f, ty, ex + 1.0f, ey, ax_col[i], 1.0f, false);
                blend_line(px, W, cl, tx, ty + 1.0f, ex, ey + 1.0f, ax_col[i], 1.0f, false);
            }
            return;
        }
        // Numbers and names of the axes: offset in pixels, perpendicular to the projected edge, towards the outside of
        // the box.
        float ccx = 0.0f, ccy = 0.0f;
        project(c, 0.5f * kL, 0.5f * kL, 0.5f * kL, ccx, ccy);
        auto axis_labels = [&](const float a[3], const float b[3], int axis, float num_px, float name_px, const char* name, int skip) {
            float ax, ay, bx, by;
            if (!project(c, a[0], a[1], a[2], ax, ay) || !project(c, b[0], b[1], b[2], bx, by)) return;
            float nx = -(by - ay), ny = bx - ax;
            const float nn = sqrtf(nx * nx + ny * ny);
            if (nn < 1.0f) return;
            nx /= nn;
            ny /= nn;
            if (nx * (0.5f * (ax + bx) - ccx) + ny * (0.5f * (ay + by) - ccy) < 0.0f) {
                nx = -nx;
                ny = -ny;
            }
            // Neither on the divergence inset, nor on the right-hand column (scales and colour scale).
            auto place = [&](float x, float y, const std::string& t) {
                const float right = (float)(vx_of[k] + vw);
                if (x > right - (float)vs - 30.0f && y > (float)(vh - vs) - 64.0f) return;
                if (x > right - 170.0f && y < 384.0f) return;
                text_c(x, y, ax_col[axis], t);
            };
            for (int i = 0; i <= 4; ++i) {
                const float t = (float)i / 4.0f;
                float sx, sy;
                if (i != skip && project(c, a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2]), sx, sy))
                    place(sx + nx * num_px, sy + ny * num_px, std::to_string(i * n / 4));
            }
            place(0.5f * (ax + bx) + nx * name_px, 0.5f * (ay + by) + ny * name_px, name);
        };
        const float xa[3] = {0.0f, xy, xz}, xb[3] = {kL, xy, xz}, za[3] = {zx, zy, 0.0f}, zb[3] = {zx, zy, kL};
        const float ya[3] = {yx, 0.0f, yz}, yb[3] = {yx, kL, yz};
        // A corner shared by two axes carries only one number: that of the x axis, otherwise that of the z axis.
        const bool xz_meet = xy == zy, yx_meet = yz == xz, yz_meet = yx == zx;
        axis_labels(xa, xb, 0, 20.0f, 60.0f, "x (voxels)", -1);
        axis_labels(za, zb, 2, 20.0f, 60.0f, "z", xz_meet ? (xz > 0.0f ? 4 : 0) : -1);
        axis_labels(ya, yb, 1, 28.0f, 62.0f, "y", yx_meet ? (xy > 0.0f ? 4 : 0) : (yz_meet ? (zy > 0.0f ? 4 : 0) : -1));
        // Letters of the triad.
        const float tx = (float)vx_of[k] + 56.0f, ty = (float)vh - 60.0f, len = 50.0f;
        static const char* const ax_name[3] = {"x", "y", "z"};
        for (int i = 0; i < 3; ++i) text_c(tx + len * c.right[i], ty - len * c.up[i], ax_col[i], ax_name[i]);
        // Right-hand column: size of the box, scales of the turbulence, simulated time.
        const int rx = vx_of[k] + vw - 146;
        int ry = 10;
        view_text(rx, ry, c_soft, ("box " + std::to_string(n) + "³ vx").c_str());
        if (re_lambda > 0.0) view_text(rx, ry += 22, c_soft, ("Reλ = " + num(re_lambda, 0)).c_str());
        if (eta > 0.0) {
            view_text(rx, ry += 22, c_soft, ("η = " + num(eta / h, 2) + " vx").c_str());
            view_text(rx, ry += 22, c_soft, ("λ = " + num(lambda / h, 1) + " vx").c_str());
            view_text(rx, ry += 22, c_accent, ("t = " + num((double)s.advanced * s.dt_frame / tau_eta, 1) + " τη").c_str());
        }
        // Colour scale of the particles: the quantity relative to its mean (the brightness follows the number of
        // particles).
        const int bx = vx_of[k] + vw - 26, by = 158, bh = 180;
        for (int j = 0; j < bh; ++j)
            for (int i = 0; i < 12; ++i) px[(size_t)(by + j) * W + bx + i] = inferno(0.12f + 0.88f * (1.0f - (float)j / (float)(bh - 1)));
        text_r(bx + 12, by - 28, c_soft, color ? "vorticity" : "speed");
        text_r(bx - 8, by - 4, c_dim, "3 × mean");
        text_r(bx - 8, by + bh / 2 - 10, c_dim, "1.5 ×");
        text_r(bx - 8, by + bh - 16, c_dim, "0");
        const double mean = 1.0 / (3.0 * (double)inv_scale);
        if (color && tau_eta > 0.0) text_r(bx + 12, by + bh + 6, c_dim, "mean " + num(mean * tau_eta, 2) + "/τη");
        else if (!color && u_rms > 0.0f) text_r(bx + 12, by + bh + 6, c_dim, "mean " + num(mean / u_rms, 2) + " u′");
    };

    // Slice views: tick marks in voxels on the left and bottom edges, scale bar, and miniature cube that locates the
    // slice.
    auto frame2d = [&](int k, int pass) {
        const SliceGeo sg = slice_geo(vx_of[k]);
        const Clip cl{vx_of[k], 0, vx_of[k] + vw, vh};
        uint32_t* px = view_pixels();
        const float ppv = h / sg.scale;  // pixels per voxel
        int step = 1;
        while ((float)step * ppv < 70.0f && step < n) step *= 2;
        const float xl = (sg.cx - 0.5f * (float)vw * sg.scale) / h, yb = (sg.cy - 0.5f * (float)vh * sg.scale) / h;  // left and bottom edges, in voxels
        const int x0 = vx_of[k], x_end = x0 + vw - 330, y_top = 254, bar = (int)lroundf((float)step * ppv);
        const float mcx = (float)(x0 + vw - 76), mcy = 66.0f, ms = 54.0f;  // miniature cube
        if (pass == 0) {
            view_panel(x0, vh - 30, vw - 318, 30, 0.35f);
            view_panel(x0, y_top - 6, 58, vh - 30 - (y_top - 6), 0.35f);
            view_panel(x0 + 66, vh - 92, std::max(bar, 230) + 20, 54, 0.35f);
            view_panel(x0 + vw - 150, 4, 146, 172, 0.35f);
            return;
        }
        auto wrap = [&](int v) { return std::to_string(((v % n) + n) % n); };
        for (int v = (int)ceilf(xl / (float)step) * step;; v += step) {
            const float sx = (float)x0 + ((float)v - xl) * ppv;
            if (sx > (float)x_end - 60.0f) break;
            if (sx < (float)x0 + 64.0f) continue;
            blend_line(px, W, cl, sx, (float)vh - 1.0f, sx, (float)vh - 8.0f, 0xDDE3EA, 1.0f, false);
            text_c(sx, (float)vh - 18.0f, c_soft, wrap(v));
        }
        for (int v = (int)ceilf(yb / (float)step) * step;; v += step) {
            const float sy = (float)vh - ((float)v - yb) * ppv;
            if (sy < (float)y_top + 30.0f) break;
            if (sy > (float)vh - 40.0f) continue;
            blend_line(px, W, cl, (float)x0, sy, (float)x0 + 7.0f, sy, 0xDDE3EA, 1.0f, false);
            view_text(x0 + 12, (int)lroundf(sy) - 10, c_soft, wrap(v).c_str());
        }
        view_text(x0 + 12, y_top - 2, ax_col[1], "y");
        view_text(x_end - 46, vh - 28, ax_col[0], "x");
        view_text(x0 + 8, vh - 28, c_dim, "vx");
        // Scale bar: one tick step, in voxels and in Kolmogorov scales.
        const float gx = (float)x0 + 76.0f, gy = (float)vh - 50.0f;
        for (int t = 0; t < 2; ++t) blend_line(px, W, cl, gx, gy + (float)t, gx + (float)bar, gy + (float)t, 0xF2F2F2, 1.0f, false);
        blend_line(px, W, cl, gx, gy - 5.0f, gx, gy + 6.0f, 0xF2F2F2, 1.0f, false);
        blend_line(px, W, cl, gx + (float)bar, gy - 5.0f, gx + (float)bar, gy + 6.0f, 0xF2F2F2, 1.0f, false);
        view_text((int)gx, vh - 86, c_soft, (std::to_string(step) + " voxels" + (eta > 0.0 ? " = " + num((double)step * h / eta, 0) + " η" : std::string())).c_str());
        // Miniature cube: the box seen from the default angle of the 3D view, and the plane of the slice.
        const float cp = cosf(0.38f), sp = sinf(0.38f), cyw = cosf(0.62f), syw = sinf(0.62f);
        const float f3[3] = {-cp * syw, -sp, -cp * cyw};
        const float rn = sqrtf(f3[2] * f3[2] + f3[0] * f3[0]);
        const float r3[3] = {-f3[2] / rn, 0.0f, f3[0] / rn};
        const float u3[3] = {r3[1] * f3[2] - r3[2] * f3[1], r3[2] * f3[0] - r3[0] * f3[2], r3[0] * f3[1] - r3[1] * f3[0]};
        auto mini = [&](float x, float y, float z, float& sx, float& sy) {
            x -= 0.5f;
            y -= 0.5f;
            z -= 0.5f;
            sx = mcx + ms * (x * r3[0] + y * r3[1] + z * r3[2]);
            sy = mcy - ms * (x * u3[0] + y * u3[1] + z * u3[2]);
        };
        auto mline = [&](float xa, float ya, float za, float xb, float yb2, float zb, uint32_t rgb) {
            float ax, ay, bx, by;
            mini(xa, ya, za, ax, ay);
            mini(xb, yb2, zb, bx, by);
            blend_line(px, W, cl, ax, ay, bx, by, rgb, 1.0f, false);
        };
        for (int e = 0; e < 12; ++e) {
            const int a = box_edge[e][0], b = box_edge[e][1];
            mline((float)(a & 1), (float)((a >> 1) & 1), (float)((a >> 2) & 1), (float)(b & 1), (float)((b >> 1) & 1), (float)((b >> 2) & 1), 0x6F7A88);
        }
        const float zr = zs / kL;
        mline(0.0f, 0.0f, zr, 1.0f, 0.0f, zr, c_slice);
        mline(1.0f, 0.0f, zr, 1.0f, 1.0f, zr, c_slice);
        mline(1.0f, 1.0f, zr, 0.0f, 1.0f, zr, c_slice);
        mline(0.0f, 1.0f, zr, 0.0f, 0.0f, zr, c_slice);
        mline(0.0f, 0.0f, zr, 1.0f, 1.0f, zr, c_slice);
        mline(1.0f, 0.0f, zr, 0.0f, 1.0f, zr, c_slice);
        float lx = 0.0f, ly = 0.0f;
        mini(1.16f, 0.0f, 1.0f, lx, ly);
        text_c(lx, ly, ax_col[0], "x");
        mini(0.0f, 1.16f, 1.0f, lx, ly);
        text_c(lx, ly, ax_col[1], "y");
        mini(1.0f, 0.0f, -0.2f, lx, ly);
        text_c(lx, ly, ax_col[2], "z");
        text_c(mcx, 134.0f, c_slice, "slice z = " + num(zs / h, 0) + " vx");
        text_c(mcx, 156.0f, c_dim, "keys ↑ ↓");
    };
    auto overlay = [&]() {
        const uint32_t white = 0xF2F2F2, soft = 0xB8C0CC, accent = 0xFFC857, dim = 0x7C8694;
        const uint32_t col[2] = {0x5AA9E6, 0xFFC857};  // colour of each side in the band
        for (int k = 0; k < 2; ++k) {
            if (view == 0) frame3d(k, 0);
            else frame2d(k, 0);
        }
        view_line(W / 2 - 1, 0, W / 2 - 1, vh, 0x30363F);
        view_line(W / 2, 0, W / 2, vh, 0x30363F);
        for (int k = 0; k < 2; ++k) {
            const Side& s = *side[k];
            const int x = vx_of[k] + 14;
            int y = 12;
            view_panel(vx_of[k] + 6, 6, 640, view == 0 ? 190 : 236, 0.30f);
            view_rect(x, y + 9, 12, 12, col[k]);
            view_text(x + 20, y, white, s.title.c_str(), 1);
            y += 30;
            view_text(x, y, soft, s.sub.c_str());
            y += 22;
            std::string l = "frame " + std::to_string(s.latest + 1) + "/" + std::to_string(nfe) + "  \xC2\xB7  " + std::to_string(s.advanced) +
                            " frames advanced  \xC2\xB7  " + num(s.mib_avg(), s.mib_avg() < 100.0 ? 2 : 0) + " MiB read per frame";
            view_text(x, y, soft, l.c_str());
            y += 22;
            l = "disk " + num(1e3 * s.m.disk, 2) + "  \xC2\xB7  bus " + num(1e3 * s.m.bus, 2) + "  \xC2\xB7  decoding " + num(1e3 * s.m.dec, 2) + " ms";
            view_text(x, y, soft, l.c_str());
            y += 22;
            l = "advection " + std::to_string(substeps) + " \xC3\x97 " + num(1e3 * s.m.step, 2) + " ms" + (sorted ? "  \xC2\xB7  sort " + num(1e3 * s.m.sort, 2) + " ms" : "");
            view_text(x, y, soft, l.c_str());
            y += 22;
            const double fs = s.frame_seconds(substeps);
            l = "cost of one frame " + num(1e3 * fs, 1) + " ms  \xE2\x86\x92  " + (fs > 0.0 ? num(1.0 / fs, 1) : std::string("-")) + " frames/s by its own costs";
            view_text(x, y, accent, l.c_str());
            y += 22;
            l = "GPU memory " + num(s.vram_mib, 0) + " MiB  \xC2\xB7  rendering " + num(1e3 * s.m.render, 1) + " ms";
            view_text(x, y, soft, l.c_str());
            y += 22;
            if (view == 1 && s.div_ratio >= 0.0) {
                l = "divergence / gradient, RMS over the slice: " + sci(s.div_ratio);
                view_text(x, y, accent, l.c_str());
                // What the image shows. Below 1e-8, the bottom of the scale, no colour covers the flow: the divergence
                // is zero to round-off.
                view_text(x, y + 22, soft, s.div_ratio < 1e-8 ? "grey only: divergence below the scale, zero to round-off" : "colour: divergence (log scale); grey: velocity gradient");
            } else if (view == 2 && s.err_rms >= 0.0) {
                l = "field error, RMS over the slice: " + num(100.0 * s.err_rms, 2) + " %";
                view_text(x, y, accent, l.c_str());
                view_text(x, y + 22, soft, "image: error of this side against the raw frame, not the field");
            }
            if (view == 0) {
                frame3d(k, 1);
                // Border and legend of the divergence inset.
                const int qx = vx_of[k] + vw - vs - 10, qy = vh - vs - 10;
                view_line(qx - 1, qy - 1, qx + vs, qy - 1, c_slice);
                view_line(qx + vs, qy - 1, qx + vs, qy + vs, c_slice);
                view_line(qx + vs, qy + vs, qx - 1, qy + vs, c_slice);
                view_line(qx - 1, qy + vs, qx - 1, qy - 1, c_slice);
                text_r(vx_of[k] + vw - 6, qy - 44, soft, "divergence " + sci(s.div_ratio));
                text_r(vx_of[k] + vw - 6, qy - 22, c_slice, "slice z = " + num(zs / h, 0) + " vx");
            } else {  // colour scale
                uint32_t* px = view_pixels();
                const int bx = vx_of[k] + vw - 300, by = vh - 46;
                view_panel(bx - 12, by - 28, 300, 64, 0.30f);
                for (int j = 0; j < 12; ++j)
                    for (int i = 0; i < 256; ++i) px[(size_t)(by + j) * W + bx + i] = inferno(i / 255.0f);
                if (view == 1) {
                    view_text(bx, by - 24, soft, "divergence / RMS gradient");
                    view_text(bx - 4, by + 14, dim, "1e-8");
                    view_text(bx + 112, by + 14, dim, "1e-4");
                    view_text(bx + 250, by + 14, dim, "1");
                } else {
                    view_text(bx, by - 24, soft, "error / RMS velocity");
                    view_text(bx - 4, by + 14, dim, "0");
                    view_text(bx + 226, by + 14, dim, "5 %");
                }
                frame2d(k, 1);
            }
        }
        // Band: the measured differences, readable at a glance. For each quantity, smaller is better.
        {
            const int y0 = vh, cols = side[0]->err0 >= 0.0 ? 5 : 4, cw = W / cols;
            const std::string err_name = "field error (frame " + std::to_string(err_frame + 1) + ")";
            view_line(0, y0, W, y0, 0x30363F);
            struct Metric {
                const char* name;
                double v[2];
                int log;  // bars on a logarithmic scale
                const char* unit;
                const char* gain;
            };
            const Metric ms[5] = {
                {"data read per frame", {side[0]->mib_avg(), side[1]->mib_avg()}, 0, " MiB", "less"},
                {"cost of one frame", {1e3 * side[0]->frame_seconds(substeps), 1e3 * side[1]->frame_seconds(substeps)}, 0, " ms", "faster"},
                {"GPU memory", {side[0]->vram_mib, side[1]->vram_mib}, 0, " MiB", "less"},
                {"divergence of the field", {side[0]->div_ratio, side[1]->div_ratio}, 1, "", "lower"},
                {err_name.c_str(), {100.0 * side[0]->err0, 100.0 * side[1]->err0}, 0, " %", "lower"}};
            for (int i = 0; i < cols; ++i) {
                const Metric& mt = ms[i];
                const int x = i * cw + 14;
                view_text(x, y0 + 8, soft, mt.name);
                if (!(mt.v[0] > 0.0) || !(mt.v[1] > 0.0)) continue;
                const int win = mt.v[1] <= mt.v[0] ? 1 : 0;
                const double ratio = mt.v[win ^ 1] / mt.v[win];
                std::string head = win ? "right: " : "left: ";
                if (ratio < 1.05) head = "tie";
                else if (mt.log && ratio > 1e6) head += cw >= 300 ? "zero to round-off" : "zero";
                else head += num(ratio, ratio < 10.0 ? 1 : 0) + "\xC3\x97 " + mt.gain;
                view_text(x, y0 + 28, accent, head.c_str(), 1);
                const double mx = std::max(mt.v[0], mt.v[1]);
                for (int k = 0; k < 2; ++k) {
                    const double f = mt.log ? std::min(1.0, std::max(0.0, (std::log10(mt.v[k]) + 16.0) / 16.0)) : mt.v[k] / mx;
                    const int bw = std::max(3, (int)(f * (cw - 150))), yb = y0 + 68 + 26 * k;
                    view_rect(x, yb, bw, 12, col[k]);
                    const std::string val = (mt.log ? sci(mt.v[k]) : num(mt.v[k], mt.v[k] < 100.0 ? 2 : 0)) + mt.unit;
                    view_text(x + bw + 8, yb - 4, soft, val.c_str());
                }
            }
        }
        const char* names[3] = {"particles", "divergence of the field", "field error against the raw frame"};
        std::string top = std::string(names[view]);
        if (view == 0) top += color ? " (colour: vorticity)" : " (colour: speed)";
        top += race ? "  \xC2\xB7  RACE: each side advances at the pace of its own measured costs" : "  \xC2\xB7  SYNC: same frame on both sides";
        if (paused) top += "  \xC2\xB7  PAUSED";
        if (sorted) top += "  \xC2\xB7  sorted particles";
        if (loop_s > 0.0) top += "  \xC2\xB7  display " + num(1.0 / loop_s, 0) + " frames/s";
        view_panel(0, H - 54, W, 54, 0.30f);
        view_text(12, H - 50, white, top.c_str());
        view_text(12, H - 26, dim, "Space pause  N step  1 2 3 views  C race  S sort  V colour  A spin  F depth  +/- substeps  R reset  P capture");
    };

    // Main loop.
    int captures = 0;
    auto t_loop = std::chrono::steady_clock::now();
    for (int iter = 0;; ++iter) {
        ViewInput in;
        view_poll(in);
        if (in.quit && !snap) break;
        bool single = false;
        for (int i = 0; i < in.key_count; ++i) {
            const int key = in.keys[i];
            if (key == 0x1B) in.quit = true;
            else if (key == 0x20) paused = !paused;
            else if (key == 'N') single = true;
            else if (key == '1' || key == 0x61) view = 0;
            else if (key == '2' || key == 0x62) view = 1;
            else if ((key == '3' || key == 0x63) && seq) {
                view = 2;
                race = false;
                paused = true;
                align();
            } else if (key == 'C' && view != 2) {  // the error view stays in sync: the raw frame is loaded for one frame only
                race = !race;
                if (!race) align();
            } else if (key == 'S') {
                sorted = !sorted;
                if (sorted)
                    for (Side* s : side) s->last_sort = s->sort_particles();
            } else if (key == 'A') {
                spin = !spin;
            } else if (key == 'F') {
                fog = !fog;
            } else if (key == 'V') {
                color ^= 1;
                rescale();
            } else if (key == 0x6B || key == 0xBB) substeps = std::min(64, substeps * 2);
            else if (key == 0x6D || key == 0xBD) substeps = std::max(1, substeps / 2);
            else if (key == 'R') {
                for (Side* s : side) {
                    s->init_particles(o.particles, o.seed);
                    if (sorted) s->last_sort = s->sort_particles();
                }
            } else if (key == 0x26) zs += h;
            else if (key == 0x28) zs -= h;
            else if (key == 0x21) bright *= 1.25f;
            else if (key == 0x22) bright /= 1.25f;
            else if (key == 'P') {
                char name[64];
                snprintf(name, sizeof name, "psi_view_%03d.png", captures++);
                view_save_png(name);
            }
        }
        if (in.quit && !snap) break;
        if (view == 0) {
            if (spin && !snap) yaw += 0.25f * (float)loop_s;  // one turn in about 25 seconds
            yaw -= 0.006f * in.drag_dx;
            pitch = std::min(1.5f, std::max(-1.5f, pitch + 0.006f * in.drag_dy));
            dist = std::min(6.0f * kL, std::max(0.3f * kL, dist * powf(0.9f, in.wheel)));
        } else {
            const float sc = kL / std::min(vw, vh) / zoom;
            cx -= in.drag_dx * sc;
            cy += in.drag_dy * sc;
            zoom = std::min(64.0f, std::max(0.5f, zoom * powf(1.2f, in.wheel)));
        }
        zs -= kL * floorf(zs / kL);

        const bool last = snap && iter + 1 >= o.iters;
        if (!paused || single) {
            if (single) {
                align();
                step_both();
            } else {
                advance();
            }
        }
        if (!snap || last) {  // in capture mode, only the last image is rendered
            if (view == 2) ensure_truth(side[0]->latest);
            CK(cudaMemsetAsync(d_img, 0, (size_t)W * H * sizeof(uint32_t), st));
            if (view == 0) CK(cudaMemsetAsync(d_acc, 0, (size_t)2 * W * H * sizeof(float), st));
            render_side(0);
            render_side(1);
            CK(cudaMemcpy(view_pixels(), d_img, (size_t)W * H * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            overlay();
            view_present();
        }
        if (paused && !snap) std::this_thread::sleep_for(std::chrono::milliseconds(10));
        const double dt_loop = seconds_since(t_loop);
        t_loop = std::chrono::steady_clock::now();
        loop_s += (loop_s > 0.0 ? 0.1 : 1.0) * (dt_loop - loop_s);
        if (last) break;
    }
    CK(cudaGetLastError());

    if (snap) {
        CHK(view_save_png(o.snapshot.c_str()), "cannot write the capture");
        printf("Capture        : %s (%dx%d), view %d, %s, %d RK4 steps per frame, %u particles per side\n", o.snapshot.c_str(), W, H, view + 1,
               race ? "race" : "sync", substeps, o.particles);
    }
    for (int k = 0; k < 2; ++k) {
        const Side& s = *side[k];
        printf("%s : %s\n", k ? "Right" : "Left", s.title.c_str());
        printf("  per frame    : %.2f MiB read on average, disk %.3f ms, bus %.3f ms, decoding %.3f ms, sort %.3f ms, %d RK4 steps of %.3f ms\n", s.mib_avg(),
               1e3 * s.m.disk, 1e3 * s.m.bus, 1e3 * s.m.dec, 1e3 * s.m.sort, substeps, 1e3 * s.m.step);
        printf("  cost         : %.3f ms per frame, that is %.1f frames per second by its own costs; %lld frames advanced\n",
               1e3 * s.frame_seconds(substeps), 1.0 / s.frame_seconds(substeps), s.advanced);
        printf("  GPU memory   : %.0f MiB; rendering %.2f ms\n", s.vram_mib, 1e3 * s.m.render);
        if (s.div_ratio >= 0.0) printf("  divergence   : %.3e of the gradient (RMS over the slice)\n", s.div_ratio);
        if (s.err_rms >= 0.0) printf("  error        : %.4e (RMS over the slice, against the raw frame)\n", s.err_rms);
    }
    view_close();
    return 0;
}
