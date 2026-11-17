// SPDX-License-Identifier: Apache-2.0
// codec_kernels.cuh - the kernels of the codec, as the library uses them (src/psitreillis.cu, through
// src/codec_stream.cuh).
//
// Contents, in order:
//   - the sampling lattice (a vector potential Psi in B-splines, two frames per node) and its analytic curl: the
//     velocity u = curl Psi at a point, by direct reading of the lattice or by differences;
//   - the sparse fine level: a second potential, of step one voxel, present by bricks;
//   - the projection of a frame onto the potential, level by level, in Fourier space (fit_level);
//   - the quantisation on the pyramid of levels: key frames, temporal prediction with integer coefficients,
//     innovations, exceptions;
//   - the synthesis of the sampling lattice from the levels, by the exact integer refinement of the B-splines;
//   - rANS decoding, and the host functions that go with these kernels (limits of the codec, alias sums, rANS tables,
//     3x3 solver).
// The kernels of the measurement instruments (advection, ray marching, reference paths) are not here: they are in
// tools/playback_kernels.cuh.
#ifndef PSIT_CODEC_KERNELS_CUH
#define PSIT_CODEC_KERNELS_CUH

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cufft.h>

#include "sequence.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

// The programs stop on an error (message, exit code 1). A compilation unit that includes this file can define CK and
// CHK before the inclusion to decide otherwise: the library (src/psitreillis.cu) throws an exception, which it returns
// to the caller as an error code.
#ifndef CK
#define CK(call)                                                                                   \
    do {                                                                                           \
        cudaError_t e_ = (call);                                                                   \
        if (e_ != cudaSuccess) {                                                                   \
            fprintf(stderr, "CUDA error: %s\n  %s:%d\n", cudaGetErrorString(e_), source_name(__FILE__), __LINE__); \
            exit(1);                                                                               \
        }                                                                                          \
    } while (0)
#endif
#ifndef CHK
#define CHK(cond, msg)                                                         \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "Failure: %s\n  %s:%d\n", msg, source_name(__FILE__), __LINE__); \
            exit(1);                                                           \
        }                                                                      \
    } while (0)
#endif

constexpr float kL = 6.283185307179586f;
constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxLevels = 6;

// Guards: clipping sites of the codec, counted per level of the pyramid (kClipSites counters per level).
//   kClipKey      target of a key frame outside the range of the level;
//   kClipTarget   target of an innovation outside the range of the level;
//   kClipInnov    innovation outside +-127: the decoded state then departs from the target;
//   kClipState    reconstructed state (prediction + innovation) outside the range of the level;
//   kClipPartial  partial sum of the pyramid outside the range that the integer refinement (up_at) accepts without
//                 overflowing: +-(2^25 - 1) in degree 2, +-(2^22 - 1) in degree 3 (check_codec_limits rules it out
//                 beforehand);
//   kClipLattice  difference between two neighbouring nodes of the sampling lattice outside the margin of the reading
//                 by differences (counted at level 0, see synth_lattice).
// None of these counters changes an emitted byte: they only observe the clippings already in place.
enum { kClipKey = 0, kClipTarget = 1, kClipInnov = 2, kClipState = 3, kClipPartial = 4, kClipLattice = 5, kClipSites = 6 };
static const char* const kClipName[kClipSites] = {"key target", "innovation target", "innovation outside +-127", "reconstructed state",
                                                  "partial sum", "lattice, difference between neighbours"};

struct V3 {
    float x, y, z;
};
__host__ __device__ inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
__host__ __device__ inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
__host__ __device__ inline V3 operator*(float s, V3 a) { return {s * a.x, s * a.y, s * a.z}; }

// ---------------------------------------------------------------------------------------
// Sampling lattice and evaluation of u = curl Psi
// ---------------------------------------------------------------------------------------
//
// One node = one word of 16 bytes: slot 0 (x, y, z on 16 bits, 16 free bits) then slot 1. Frame k occupies slot k & 1:
// a new frame overwrites the oldest one in place, and a single aligned read brings back the two frames of the interval.

// The lattice is written modulo 2^16: only the differences between nodes of the same support have a meaning. Since the
// sum of the grad(phi) over a support is zero, subtracting from each node the value of the first node of the support
// (v0) does not change u = sum of grad(phi) x q, and the difference, taken modulo 2^16, is exact as long as it stays
// under 2^15.
// The direct reading (without v0) gives the same u as long as no wrap separates two nodes of the support. A wrap
// implies a value less than 2^14 from the edge of the range of a short: lat_set flags it with bit 16 of the second word
// of each slot, and the kernels switch to the reading by differences only if a node of the support carries this bit.
constexpr int kLatNear = 0x10000;
__device__ __forceinline__ void lat_get(const int4 v, int slot, int q[3]) {
    const int lo = slot ? v.z : v.x, hi = slot ? v.w : v.y;
    q[0] = (int)(short)(lo);
    q[1] = (int)(short)(lo >> 16);
    q[2] = (int)(short)(hi);
}
__device__ __forceinline__ void lat_get(const int4 v, const int4 v0, int slot, int q[3]) {
    const int lo = slot ? v.z : v.x, hi = slot ? v.w : v.y, lo0 = slot ? v0.z : v0.x, hi0 = slot ? v0.w : v0.y;
    q[0] = (int)(short)(lo - lo0);
    q[1] = (int)(short)((lo >> 16) - (lo0 >> 16));
    q[2] = (int)(short)(hi - hi0);
}
__device__ inline void lat_set(int4* p, int slot, const int q[3]) {
    // near: bits 15 and 14 of a component differ, so its wrapped value is less than 2^14 from the edge.
    const int near_edge = (((q[0] ^ (q[0] << 1)) | (q[1] ^ (q[1] << 1)) | (q[2] ^ (q[2] << 1))) >> 15) & 1;
    const int lo = (q[0] & 0xFFFF) | (int)((unsigned)q[1] << 16), hi = (q[2] & 0xFFFF) | (near_edge << 16);
    if (slot) {
        p->z = lo;
        p->w = hi;
    } else {
        p->x = lo;
        p->y = hi;
    }
}

// B-spline weights of degree D and their derivatives (per unit of step); returns the index of the first node of the
// support. The nodes are at the integer abscissas of t.
template <int D>
__device__ __forceinline__ int bspline(float t, float w[4], float d[4]) {
    if (D == 2) {
        const float i = floorf(t + 0.5f), f = t - i, a = 0.5f - f, b = 0.5f + f;
        w[0] = 0.5f * a * a;
        w[1] = 0.75f - f * f;
        w[2] = 0.5f * b * b;
        w[3] = 0.0f;
        d[0] = -a;
        d[1] = -2.0f * f;
        d[2] = b;
        d[3] = 0.0f;
        return (int)i - 1;
    }
    const float i = floorf(t), f = t - i, g = 1.0f - f;
    w[0] = g * g * g * (1.0f / 6.0f);
    w[1] = ((3.0f * f - 6.0f) * f * f + 4.0f) * (1.0f / 6.0f);
    w[2] = (((-3.0f * f + 3.0f) * f + 3.0f) * f + 1.0f) * (1.0f / 6.0f);
    w[3] = f * f * f * (1.0f / 6.0f);
    d[0] = -0.5f * g * g;
    d[1] = (1.5f * f - 2.0f) * f;
    d[2] = (-1.5f * f + 1.0f) * f + 0.5f;
    d[3] = 0.5f * f * f;
    return (int)i - 1;
}

template <int D>
struct PsiField {
    const int4* lat;
    int n;           // nodes per axis
    float inv_h;     // inverse of the step of the lattice
    float x0;        // abscissa of node 0 (half a step in degree 2, zero in degree 3)
    float s;         // quantisation step / (2^fb * step of the lattice)
    float m0[3], m1[3];  // mean velocity of frames k and k+1 (mode k = 0, outside the potential)
    int early;       // slot of frame k
};

// Contribution of a node: grad(phi) x q, q interpolated in time between the two slots (e: frame k, l: frame k+1).
__device__ __forceinline__ void node_cross(const int e[3], const int l[3], float w, float gx, float gy, float gz, float& ax, float& ay, float& az) {
    const float qx = (float)e[0] + w * (float)(l[0] - e[0]);
    const float qy = (float)e[1] + w * (float)(l[1] - e[1]);
    const float qz = (float)e[2] + w * (float)(l[2] - e[2]);
    ax += gy * qz - gz * qy;
    ay += gz * qx - gx * qz;
    az += gx * qy - gy * qx;
}
// Direct reading of the node.
__device__ __forceinline__ void node_accum(const int4 v, int early, float w, float gx, float gy, float gz, float& ax, float& ay, float& az) {
    int e[3], l[3];
    lat_get(v, early, e);
    lat_get(v, early ^ 1, l);
    node_cross(e, l, w, gx, gy, gz, ax, ay, az);
}
// Reading by differences with respect to node v0.
__device__ __forceinline__ void node_accum(const int4 v, const int4 v0, int early, float w, float gx, float gy, float gz, float& ax, float& ay,
                                           float& az) {
    int e[3], l[3];
    lat_get(v, v0, early, e);
    lat_get(v, v0, early ^ 1, l);
    node_cross(e, l, w, gx, gy, gz, ax, ay, az);
}

template <int D>
__device__ inline V3 sample(const PsiField<D>& f, V3 p, float w) {
    float wx[4], dx[4], wy[4], dy[4], wz[4], dz[4];
    const int ix = bspline<D>((p.x - f.x0) * f.inv_h, wx, dx), iy = bspline<D>((p.y - f.x0) * f.inv_h, wy, dy),
              iz = bspline<D>((p.z - f.x0) * f.inv_h, wz, dz), m = f.n - 1;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    int near_edge = 0;
    for (int c = 0; c <= D; ++c) {
        const size_t zo = (size_t)((iz + c) & m) * f.n;
        for (int b = 0; b <= D; ++b) {
            const size_t row = (zo + ((iy + b) & m)) * f.n;
            const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
            for (int a = 0; a <= D; ++a) {
                const int4 v = f.lat[row + ((ix + a) & m)];
                near_edge |= v.y | v.w;
                node_accum(v, f.early, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
            }
        }
    }
    if (near_edge & kLatNear) {  // a node of the support is near the edge of the range: read again by differences with respect to the first node
        const int4 v0 = f.lat[((size_t)(iz & m) * f.n + (iy & m)) * f.n + (ix & m)];
        ax = ay = az = 0.0f;
        for (int c = 0; c <= D; ++c) {
            const size_t zo = (size_t)((iz + c) & m) * f.n;
            for (int b = 0; b <= D; ++b) {
                const size_t row = (zo + ((iy + b) & m)) * f.n;
                const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
                for (int a = 0; a <= D; ++a) node_accum(f.lat[row + ((ix + a) & m)], v0, f.early, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
            }
        }
    }
    return {f.s * ax + f.m0[0] + w * (f.m1[0] - f.m0[0]), f.s * ay + f.m0[1] + w * (f.m1[1] - f.m0[1]),
            f.s * az + f.m0[2] + w * (f.m1[2] - f.m0[2])};
}

// ---------------------------------------------------------------------------------------
// Variants without wrap: direct reading of the lattice
// ---------------------------------------------------------------------------------------
//
// When no value of the lattice leaves the range of a short (synth_lattice reports it to the sender, which records it in
// the file), the direct reading is exact and the test of the proximity bit is unnecessary. These are the original
// kernels; the readers choose them as long as the lattice does not wrap.
template <int D>
struct PsiPlain : PsiField<D> {};
template <int D>
__device__ inline V3 sample(const PsiPlain<D>& f, V3 p, float w) {
    float wx[4], dx[4], wy[4], dy[4], wz[4], dz[4];
    const int ix = bspline<D>((p.x - f.x0) * f.inv_h, wx, dx), iy = bspline<D>((p.y - f.x0) * f.inv_h, wy, dy),
              iz = bspline<D>((p.z - f.x0) * f.inv_h, wz, dz), m = f.n - 1;
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int c = 0; c <= D; ++c) {
        const size_t zo = (size_t)((iz + c) & m) * f.n;
        for (int b = 0; b <= D; ++b) {
            const size_t row = (zo + ((iy + b) & m)) * f.n;
            const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
            for (int a = 0; a <= D; ++a) node_accum(f.lat[row + ((ix + a) & m)], f.early, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
        }
    }
    return {f.s * ax + f.m0[0] + w * (f.m1[0] - f.m0[0]), f.s * ay + f.m0[1] + w * (f.m1[1] - f.m0[1]),
            f.s * az + f.m0[2] + w * (f.m1[2] - f.m0[2])};
}

// ---------------------------------------------------------------------------------------
// Variant with a node of 8 bytes: the kernel samples two levels directly
// ---------------------------------------------------------------------------------------
//
// Instead of the synthesis lattice on 16 bits, the kernel reads:
//   - level 0 of the codec as it is: one word of 8 bytes per node, slot 0 (x, y, z on 8 bits) then slot 1;
//   - the sum of the levels >= 1 on the lattice of level 1, twice as coarse (word of 16 bytes, fixed point).
// u = curl Psi is the sum of the two splines; nothing is approximated.

__device__ inline int pack8(int x, int y, int z) { return (x & 0xFF) | ((y & 0xFF) << 8) | ((z & 0xFF) << 16); }

// Fine node: words of frame k (ew) and of frame k+1 (lw), 8 bits per component.
__device__ __forceinline__ void fine_accum(int ew, int lw, float w, float gx, float gy, float gz, float& ax, float& ay, float& az) {
    const int ex = (int)(signed char)(ew), ey = (int)(signed char)(ew >> 8), ez = (int)(signed char)(ew >> 16);
    const int lx = (int)(signed char)(lw), ly = (int)(signed char)(lw >> 8), lz = (int)(signed char)(lw >> 16);
    const float qx = (float)ex + w * (float)(lx - ex), qy = (float)ey + w * (float)(ly - ey), qz = (float)ez + w * (float)(lz - ez);
    ax += gy * qz - gz * qy;
    ay += gz * qx - gx * qz;
    az += gx * qy - gy * qx;
}

// ---------------------------------------------------------------------------------------
// Sparse fine level: a second potential, of step 1 voxel, present by bricks
// ---------------------------------------------------------------------------------------
//
// u = curl Psi_base + curl Psi_fine. The sum of two curls is divergence-free whatever coefficients of the fine level
// are left at zero: refining by zones costs the identity nothing. The fine nodes are stored by bricks of 4^3; an
// absent brick is zero everywhere.
//   occ  : one word of 64 bits per brick, one bit per node (z << 4 | y << 2 | x): set if the support that starts at
//          this node (D + 1 nodes per axis) contains a retained node. It is the only read outside the refined zones.
//   tab  : one word per brick. Bits 0 to 7: presence of the eight bricks (b + {0,1}^3), bit (dz << 2 | dy << 1 | dx)
//          for the brick shifted by (dx, dy, dz); bits 8 to 31: index of the block of the brick itself if bit 0 is set.
//   pool : blocks of 64 nodes, of index (z * 4 + y) * 4 + x; one node = the word of slot 0 then that of slot 1
//          (pack8: three components on 8 bits).
// The support of a point (D + 1 nodes per axis, D <= 3) fits in two bricks per axis: the word of `tab` read at the
// brick of the first node gives the blocks to read.
constexpr int kFineLog = 2, kFineSide = 1 << kFineLog, kFineNodes = kFineSide * kFineSide * kFineSide;
struct FineLevel {
    const unsigned long long* occ;
    const uint32_t* tab;
    const int2* pool;
    int n, nb;        // nodes and bricks per axis
    float inv_h, x0;  // inverse of the step, abscissa of node 0
    float s;          // quantisation step / step of the lattice
};
// Index of the first node of the support, as bspline<D> returns it, without computing the weights.
template <int D>
__device__ __forceinline__ int bspline_first(float t) {
    return (int)floorf(D == 2 ? t + 0.5f : t) - 1;
}
// Contribution of the fine level to the field. It is returned separately, and the callers add it to the base field by a
// plain addition: the base field thus keeps exactly the value of the kernels without a fine level.
// Slow path: at least one retained node under the support. The function is not inlined into the callers: all the
// kernels thus execute the same machine code and return the same contribution bit for bit (inlined, it differed by one
// last bit between kernels, the compiler not fusing the same multiplications and additions everywhere).
template <int D>
__device__ __noinline__ V3 fine_slow(const FineLevel& f, float tx, float ty, float tz, int bx, int by, int bz, int lx, int ly, int lz, float w, int early) {
    const int mb = f.nb - 1;
    const uint32_t head = f.tab[((size_t)bz * f.nb + by) * f.nb + bx];
    float wx[4], dx[4], wy[4], dy[4], wz[4], dz[4];
    bspline<D>(tx, wx, dx);
    bspline<D>(ty, wy, dy);
    bspline<D>(tz, wz, dz);
    int blk[8];
    for (int k = 0; k < 8; ++k) {
        blk[k] = -1;
        if (!((head >> k) & 1u)) continue;
        const uint32_t e = k ? f.tab[((size_t)((bz + (k >> 2)) & mb) * f.nb + ((by + ((k >> 1) & 1)) & mb)) * f.nb + ((bx + (k & 1)) & mb)] : head;
        blk[k] = (int)(e >> 8);
    }
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int c = 0; c <= D; ++c) {
        const int zc = lz + c, kz = zc >> kFineLog, z = zc & (kFineSide - 1);
        for (int b = 0; b <= D; ++b) {
            const int yb = ly + b, ky = yb >> kFineLog, y = yb & (kFineSide - 1);
            const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
            for (int a = 0; a <= D; ++a) {
                const int xa = lx + a, bl = blk[(kz << 2) | (ky << 1) | (xa >> kFineLog)];
                if (bl < 0) continue;
                const int2 v = f.pool[(size_t)bl * kFineNodes + ((((z << kFineLog) | y) << kFineLog) | (xa & (kFineSide - 1)))];
                fine_accum(early ? v.y : v.x, early ? v.x : v.y, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
            }
        }
    }
    return V3{f.s * ax, f.s * ay, f.s * az};
}
template <int D>
__device__ __forceinline__ V3 fine_at(const FineLevel& f, V3 p, float w, int early) {
    const float tx = (p.x - f.x0) * f.inv_h, ty = (p.y - f.x0) * f.inv_h, tz = (p.z - f.x0) * f.inv_h;
    const int m = f.n - 1;
    const int ix = bspline_first<D>(tx) & m, iy = bspline_first<D>(ty) & m, iz = bspline_first<D>(tz) & m;
    const int bx = ix >> kFineLog, by = iy >> kFineLog, bz = iz >> kFineLog;
    const int lx = ix & (kFineSide - 1), ly = iy & (kFineSide - 1), lz = iz & (kFineSide - 1);
    // Common case: no retained node under the support. One read, no weights and no call.
    if (!((f.occ[((size_t)bz * f.nb + by) * f.nb + bx] >> ((((lz << kFineLog) | ly) << kFineLog) | lx)) & 1ull)) return V3{0.0f, 0.0f, 0.0f};
    return fine_slow<D>(f, tx, ty, tz, bx, by, bz, lx, ly, lz, w, early);
}
// Complete field: the base lattice (direct reading or reading by differences, according to B) and the sparse fine
// level.
template <int D, class B>
struct PsiFine {
    B base;
    FineLevel fine;
};
template <int D, class B>
__device__ inline V3 sample(const PsiFine<D, B>& f, V3 p, float w) {
    const V3 u = sample(f.base, p, w);
    return u + fine_at<D>(f.fine, p, w, f.base.early);
}

// ---------------------------------------------------------------------------------------
// Sender: closed-form projection onto curl(spline), in Fourier space
// ---------------------------------------------------------------------------------------

struct Cd {
    double r, i;
};
__host__ __device__ inline Cd cmul(Cd a, Cd b) { return {a.r * b.r - a.i * b.i, a.r * b.i + a.i * b.r}; }

// Coefficient (kx, ky, kz signed) of a real field given by its R2C half-spectrum (n, n, n/2+1).
__device__ inline Cd spec_at(const cufftComplex* h, int n, int kx, int ky, int kz) {
    const bool conj = kx < 0;
    if (conj) {
        kx = -kx;
        ky = -ky;
        kz = -kz;
    }
    const int iy = ky < 0 ? ky + n : ky, iz = kz < 0 ? kz + n : kz;
    const cufftComplex v = h[((size_t)iz * n + iy) * (n / 2 + 1) + kx];
    return {(double)v.x, conj ? -(double)v.y : (double)v.y};
}
__device__ inline int wrap_signed(int k, int n) {
    int r = k % n;
    if (r < 0) r += n;
    if (r > n / 2) r -= n;
    return r;
}

// One thread per wave vector q of the lattice of nl nodes per axis (half-space). Sums over the aliases k congruent to
// q in the band of the data, subtracts the field of the coarser levels already decoded (if it is provided), then solves
// the regularised 3x3 normal system.
//   u*     : half-spectra of the three velocity components (not normalised, samples at voxel centres);
//   c*     : half-spectra of the lattice of ncs nodes per axis that carries the sum of the coarser levels (not
//            normalised; cscale brings them back to a potential), or nullptr;
//   ta, tc : tables per axis, indexed by k + n/2:
//            ta[k] = beta(k) e^{i k (x0 - h/2)},  tc[k] = beta(k) beta_c(k) e^{i k (x0 - x0c)};
//   a0..a2 : alias sums, sum of k^p beta(k)^2, per axis.
__global__ void fit_level(const cufftComplex* u0, const cufftComplex* u1, const cufftComplex* u2, int n, const cufftComplex* c0,
                          const cufftComplex* c1, const cufftComplex* c2, int ncs, double cscale, int nl, double ridge, const double* a0,
                          const double* a1, const double* a2, const Cd* ta, const Cd* tc, cufftComplex* o0, cufftComplex* o1,
                          cufftComplex* o2) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int nh = nl / 2 + 1;
    if (i >= (size_t)nl * nl * nh) return;
    const int qx = (int)(i % nh), qy = (int)((i / nh) % nl), qz = (int)(i / ((size_t)nh * nl));
    const int sx = qx, sy = qy <= nl / 2 ? qy : qy - nl, sz = qz <= nl / 2 ? qz : qz - nl;
    const int m = n / nl, half = n / 2;
    const double inv_v = 1.0 / ((double)n * n * n);
    Cd r[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    for (int jz = -m / 2; jz <= m / 2; ++jz) {
        const int kz = sz + nl * jz;
        if (kz <= -half || kz >= half) continue;
        for (int jy = -m / 2; jy <= m / 2; ++jy) {
            const int ky = sy + nl * jy;
            if (ky <= -half || ky >= half) continue;
            const Cd tyz = cmul(ta[ky + half], ta[kz + half]);
            for (int jx = -m / 2; jx <= m / 2; ++jx) {
                const int kx = sx + nl * jx;
                if (kx <= -half || kx >= half) continue;
                const double k[3] = {(double)kx, (double)ky, (double)kz};
                Cd A = cmul(ta[kx + half], tyz);
                A.r *= inv_v;
                A.i *= inv_v;
                const Cd U[3] = {spec_at(u0, n, kx, ky, kz), spec_at(u1, n, kx, ky, kz), spec_at(u2, n, kx, ky, kz)};
                // i k x U
                const Cd Y[3] = {{-(k[1] * U[2].i - k[2] * U[1].i), k[1] * U[2].r - k[2] * U[1].r},
                                 {-(k[2] * U[0].i - k[0] * U[2].i), k[2] * U[0].r - k[0] * U[2].r},
                                 {-(k[0] * U[1].i - k[1] * U[0].i), k[0] * U[1].r - k[1] * U[0].r}};
                for (int c = 0; c < 3; ++c) {
                    const Cd t = cmul(A, Y[c]);
                    r[c].r += t.r;
                    r[c].i += t.i;
                }
                if (c0) {
                    // -i k x (coarse field) = beta_c e^{-i k.x0c} (k (k.C) - |k|^2 C)
                    const int wx = wrap_signed(kx, ncs), wy = wrap_signed(ky, ncs), wz = wrap_signed(kz, ncs);
                    const Cd C[3] = {spec_at(c0, ncs, wx, wy, wz), spec_at(c1, ncs, wx, wy, wz), spec_at(c2, ncs, wx, wy, wz)};
                    Cd B = cmul(tc[kx + half], cmul(tc[ky + half], tc[kz + half]));
                    B.r *= cscale;
                    B.i *= cscale;
                    const double k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2];
                    const Cd kc = {k[0] * C[0].r + k[1] * C[1].r + k[2] * C[2].r, k[0] * C[0].i + k[1] * C[1].i + k[2] * C[2].i};
                    for (int c = 0; c < 3; ++c) {
                        const Cd t = cmul(B, Cd{k[c] * kc.r - k2 * C[c].r, k[c] * kc.i - k2 * C[c].i});
                        r[c].r += t.r;
                        r[c].i += t.i;
                    }
                }
            }
        }
    }
    const double s200 = a2[qx] * a0[qy] * a0[qz], s020 = a0[qx] * a2[qy] * a0[qz], s002 = a0[qx] * a0[qy] * a2[qz];
    const double s110 = a1[qx] * a1[qy] * a0[qz], s101 = a1[qx] * a0[qy] * a1[qz], s011 = a0[qx] * a1[qy] * a1[qz];
    const double tr = 2.0 * (s200 + s020 + s002);
    Cd c[3] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
    if (tr > 0.0 && !(qx == 0 && qy == 0 && qz == 0)) {
        const double lam = ridge * tr / 3.0;
        const double A = s020 + s002 + lam, E = s200 + s002 + lam, I = s200 + s020 + lam, B = -s110, C2 = -s101, F = -s011;
        const double i00 = E * I - F * F, i01 = C2 * F - B * I, i02 = B * F - C2 * E;
        const double i11 = A * I - C2 * C2, i12 = B * C2 - A * F, i22 = A * E - B * B;
        const double det = A * i00 + B * i01 + C2 * i02;
        if (fabs(det) > 0.0) {
            const double id = 1.0 / det;
            c[0] = {id * (i00 * r[0].r + i01 * r[1].r + i02 * r[2].r), id * (i00 * r[0].i + i01 * r[1].i + i02 * r[2].i)};
            c[1] = {id * (i01 * r[0].r + i11 * r[1].r + i12 * r[2].r), id * (i01 * r[0].i + i11 * r[1].i + i12 * r[2].i)};
            c[2] = {id * (i02 * r[0].r + i12 * r[1].r + i22 * r[2].r), id * (i02 * r[0].i + i12 * r[1].i + i22 * r[2].i)};
        }
    }
    o0[i] = {(float)c[0].r, (float)c[0].i};
    o1[i] = {(float)c[1].r, (float)c[1].i};
    o2[i] = {(float)c[2].r, (float)c[2].i};
}

// ---------------------------------------------------------------------------------------
// Codec: key frames, integer prediction, innovations, synthesis of the sampling lattice
// ---------------------------------------------------------------------------------------

__device__ inline int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

struct Coef {
    int a, b, c;  // prediction = (a q(k) + b q(k-1) + c laplacian(q(k)) + 512) >> 10
};

// Integer prediction of the three components of node i of a level of n nodes per axis.
template <class T>
__device__ inline void predict3(const T* qk, const T* qm, int n, size_t i, Coef k, int p[3]) {
    int lap[3] = {0, 0, 0};
    if (k.c != 0) {
        const int m = n - 1;
        const int ix = (int)(i % n), iy = (int)((i / n) % n), iz = (int)(i / ((size_t)n * n));
        const size_t plane = (size_t)n * n, row = (size_t)iz * plane + (size_t)iy * n, col = (size_t)iz * plane + ix,
                     pil = (size_t)iy * n + ix;
        const size_t nb[6] = {row + ((ix + 1) & m),                      row + ((ix - 1) & m),
                              col + (size_t)((iy + 1) & m) * n,          col + (size_t)((iy - 1) & m) * n,
                              pil + (size_t)((iz + 1) & m) * plane,      pil + (size_t)((iz - 1) & m) * plane};
        for (int c = 0; c < 3; ++c) {
            lap[c] = -6 * (int)qk[3 * i + c];
            for (int j = 0; j < 6; ++j) lap[c] += (int)qk[3 * nb[j] + c];
        }
    }
    for (int c = 0; c < 3; ++c) p[c] = (k.a * (int)qk[3 * i + c] + k.b * (int)qm[3 * i + c] + k.c * lap[c] + 512) >> 10;
}

// Exceptions of a frame: the coefficients that the 8-bit symbols do not carry.
//   - escape (bit 31 of idx at 0): the innovation leaves +-127. The symbol emitted is -128 and val is the state to set;
//   - overflow of the fine level (bit 31 at 1): the target leaves +-127. The state stays at +-127 and val, the excess,
//     is added to the sampling lattice after its synthesis.
// idx is the index of the coefficient in the order of the innovations (start of the level + 3 x node + component).
struct ExEntry {
    uint32_t idx;
    int val;
};
constexpr uint32_t kExFine = 0x80000000u;
struct ExSink {  // collection buffer of the sender; null buf: no exceptions, out-of-range values are clipped and counted
    ExEntry* buf;
    unsigned* count;
    unsigned cap;
};
__device__ inline bool ex_push(const ExSink s, uint32_t idx, int val) {
    if (!s.buf) return false;
    const unsigned k = atomicAdd(s.count, 1u);
    if (k >= s.cap) return false;
    s.buf[k].idx = idx;
    s.buf[k].val = val;
    return true;
}

// Key frame: direct quantisation. base: index of the first coefficient of the level in the order of the innovations.
template <class T>
__global__ void level_key(const float* t0, const float* t1, const float* t2, float inv_delta, int lim, T* q, size_t nodes, unsigned* sat,
                          unsigned long long base, ExSink sink) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes) return;
    const float t[3] = {t0[i], t1[i], t2[i]};
    int clipped = 0;
    for (int c = 0; c < 3; ++c) {
        const int raw = (int)rintf(t[c] * inv_delta), v = clampi(raw, -lim, lim);
        if (raw != v && !(sizeof(T) == 1 && ex_push(sink, kExFine | (uint32_t)(base + 3 * i + c), raw - v))) ++clipped;
        q[3 * i + c] = (T)v;
    }
    if (clipped) atomicAdd(&sat[kClipKey], (unsigned)clipped);
}
// Sender: innovation = quantised target - prediction, then update of its copy of the state.
// qnew and qm designate the same array (frame k+1 overwrites frame k-1, node by node).
template <class T>
__global__ void level_innov(const float* t0, const float* t1, const float* t2, float inv_delta, const T* qk, const T* qm, int n, Coef k,
                            int lim, signed char* e, T* qnew, unsigned* sat, unsigned long long base, ExSink sink) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    int p[3];
    predict3<T>(qk, qm, n, i, k, p);
    const float t[3] = {t0[i], t1[i], t2[i]};
    int c_target = 0, c_innov = 0, c_state = 0;
    for (int c = 0; c < 3; ++c) {
        const int raw = (int)rintf(t[c] * inv_delta), target = clampi(raw, -lim, lim);
        if (raw != target && !(sizeof(T) == 1 && ex_push(sink, kExFine | (uint32_t)(base + 3 * i + c), raw - target))) ++c_target;
        const int diff = target - p[c];
        int d = clampi(diff, -127, 127), s = p[c] + d;
        if (d != diff) {
            if (ex_push(sink, (uint32_t)(base + 3 * i + c), target)) {  // escape: the exception will set the state to the target
                d = -128;
                s = target;
            } else {
                ++c_innov;
            }
        }
        const int v = clampi(s, -lim, lim);
        c_state += s != v;
        e[3 * i + c] = (signed char)d;
        qnew[3 * i + c] = (T)v;
    }
    if (c_target) atomicAdd(&sat[kClipTarget], (unsigned)c_target);
    if (c_innov) atomicAdd(&sat[kClipInnov], (unsigned)c_innov);
    if (c_state) atomicAdd(&sat[kClipState], (unsigned)c_state);
}
// Receiver: state = prediction + received innovation.
template <class T>
__global__ void level_apply(const signed char* e, const T* qk, const T* qm, int n, Coef k, int lim, T* qnew) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    int p[3];
    predict3<T>(qk, qm, n, i, k, p);
    for (int c = 0; c < 3; ++c) qnew[3 * i + c] = (T)clampi(p[c] + (int)e[3 * i + c], -lim, lim);
}

// Receiver, escapes: sets the state of the coefficients whose innovation did not fit in 8 bits. To be launched after
// level_apply, which wrote a meaningless value at these indices. The indices of a frame are all distinct.
struct LevelPtrs {
    signed char* q0;
    short* q[kMaxLevels];
    unsigned long long seg[kMaxLevels], n3[kMaxLevels];  // start and number of coefficients of each level
    int L;
};
__global__ void apply_state_ex(const ExEntry* ex, unsigned count, LevelPtrs p) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const unsigned long long i = ex[t].idx;
    for (int l = 0; l < p.L; ++l)
        if (i >= p.seg[l] && i < p.seg[l] + p.n3[l]) {
            if (l == 0) p.q0[i - p.seg[l]] = (signed char)ex[t].val;
            else p.q[l][i - p.seg[l]] = (short)ex[t].val;
            return;
        }
}
// Receiver, overflows of the fine level: adds the excess to the sampling lattice, after its synthesis. The entries are
// sorted by index; the thread of the first entry of a node handles all those of this node.
// wrap (optional): set to 1 if a value then leaves the range of a short.
// nodes: number of nodes of the lattice. An entry whose index is not that of a coefficient of level 0 (altered file)
// is ignored: without this bound, it would cause a write outside the lattice. One test per exception, not per node.
__global__ void apply_lattice_ex(const ExEntry* ex, unsigned count, unsigned long long seg0, unsigned long long nodes, int4* lat, int slot, int fb,
                                 unsigned* wrap) {
    const unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= count) return;
    const unsigned long long node = ((ex[t].idx & ~kExFine) - seg0) / 3;
    if (node >= nodes) return;
    if (t > 0 && ((ex[t - 1].idx & ~kExFine) - seg0) / 3 == node) return;
    int q[3];
    lat_get(lat[node], slot, q);
    bool out = false;
    for (unsigned u = t; u < count; ++u) {
        const unsigned long long local = (ex[u].idx & ~kExFine) - seg0;
        if (local / 3 != node) break;
        const int c = (int)(local % 3);
        q[c] += ex[u].val << fb;
        out = out || q[c] > 32767 || q[c] < -32768;
    }
    lat_set(&lat[node], slot, q);
    if (wrap && out) *wrap = 1u;
}

// Exact refinement of the B-splines by a factor of 2: integer indices and weights, per axis, of the coarse nodes that
// contribute to the fine node i. Denominator 4 (degree 2) or 8 (degree 3).
template <int D>
__device__ inline int up_axis(int i, int m, int id[3], int wt[3]) {
    const int j = i >> 1;
    if (D == 2) {
        id[0] = j & m;
        wt[0] = 3;
        id[1] = ((i & 1) ? j + 1 : j - 1) & m;
        wt[1] = 1;
        return 2;
    }
    if (i & 1) {
        id[0] = j & m;
        wt[0] = 4;
        id[1] = (j + 1) & m;
        wt[1] = 4;
        return 2;
    }
    id[0] = (j - 1) & m;
    wt[0] = 1;
    id[1] = j & m;
    wt[1] = 6;
    id[2] = (j + 1) & m;
    wt[2] = 1;
    return 3;
}
// Value, at the fine node (ix, iy, iz), of the refined lattice `up` (nh nodes per axis, 3 integers per node).
template <int D>
__device__ inline void up_at(const int* up, int nh, int ix, int iy, int iz, int out[3]) {
    int idx[3], wx[3], idy[3], wy[3], idz[3], wz[3];
    const int m = nh - 1;
    const int cx = up_axis<D>(ix, m, idx, wx), cy = up_axis<D>(iy, m, idy, wy), cz = up_axis<D>(iz, m, idz, wz);
    int s[3] = {0, 0, 0};
    for (int c = 0; c < cz; ++c)
        for (int b = 0; b < cy; ++b) {
            const size_t row = ((size_t)idz[c] * nh + idy[b]) * nh;
            const int wyz = wy[b] * wz[c];
            for (int a = 0; a < cx; ++a) {
                const int* v = up + 3 * (row + idx[a]);
                const int w = wx[a] * wyz;
                s[0] += w * (int)v[0];
                s[1] += w * (int)v[1];
                s[2] += w * (int)v[2];
            }
        }
    const int sh = D == 2 ? 6 : 9, rnd = 1 << (sh - 1);
    for (int c = 0; c < 3; ++c) out[c] = (s[c] + rnd) >> sh;
}
// Partial sum of the pyramid at the step of the current level: (c << fb) + refinement of the coarser levels, on
// 32 bits. sat (optional): clipping counters of the level (site kClipPartial). The limit is the one beyond which
// up_at, which weights this sum by weights of sum 2^6 (degree 2) or 2^9 (degree 3), would overflow an int at the next
// level; check_codec_limits guarantees that it is never reached.
template <int D, class T>
__global__ void synth_level(const T* c, const int* up, int n, int fb, int* out, unsigned* sat) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    int u[3] = {0, 0, 0};
    if (up) up_at<D>(up, n / 2, (int)(i % n), (int)((i / n) % n), (int)(i / ((size_t)n * n)), u);
    const int lim = (1 << (D == 2 ? 25 : 22)) - 1;
    int clipped = 0;
    for (int k = 0; k < 3; ++k) {
        const int s = ((int)c[3 * i + k] << fb) + u[k];
        clipped += s > lim || s < -lim;
        out[3 * i + k] = s;
    }
    if (sat && clipped) atomicAdd(&sat[kClipPartial], (unsigned)clipped);
}
// Same sum for the finest level at node (ix, iy, iz), on 32 bits.
template <int D, class T>
__device__ inline void fine_sum(const T* c, const int* up, int n, int fb, int ix, int iy, int iz, int s[3]) {
    int u[3] = {0, 0, 0};
    if (up) up_at<D>(up, n / 2, ix, iy, iz, u);
    const size_t i = ((size_t)iz * n + iy) * n + ix;
    for (int k = 0; k < 3; ++k) s[k] = ((int)c[3 * i + k] << fb) + u[k];
}
// Writes this sum in a slot of the sampling lattice, modulo 2^16: the kernels read only differences between nodes of
// the same support (see lat_get), so the absolute value of the potential no longer has a ceiling.
// wrap (optional): set to 1 if a value leaves the range of a short, that is if the lattice wraps.
// sat (optional, sender): guard of the site kClipLattice. It counts the components whose difference from an immediate
// neighbour exceeds 32767 / (3 D): below that, the difference within a support of (D+1)^3 nodes stays under 2^15 and
// the reading by differences is exact. The criterion is sufficient, not necessary.
template <int D, class T>
__global__ void synth_lattice(const T* c, const int* up, int n, int fb, int4* lat, int slot, unsigned* sat, unsigned* wrap) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * n * n) return;
    const int ix = (int)(i % n), iy = (int)((i / n) % n), iz = (int)(i / ((size_t)n * n));
    int q[3];
    fine_sum<D, T>(c, up, n, fb, ix, iy, iz, q);
    lat_set(&lat[i], slot, q);
    if (wrap && (abs(q[0]) > 32767 || abs(q[1]) > 32767 || abs(q[2]) > 32767)) *wrap = 1u;  // all the threads write the same value
    if (sat) {
        const int m = n - 1, lim = 32767 / (3 * D);
        int nb[3], bad = 0;
        fine_sum<D, T>(c, up, n, fb, (ix + 1) & m, iy, iz, nb);
        for (int k = 0; k < 3; ++k) bad += abs(nb[k] - q[k]) > lim;
        fine_sum<D, T>(c, up, n, fb, ix, (iy + 1) & m, iz, nb);
        for (int k = 0; k < 3; ++k) bad += abs(nb[k] - q[k]) > lim;
        fine_sum<D, T>(c, up, n, fb, ix, iy, (iz + 1) & m, nb);
        for (int k = 0; k < 3; ++k) bad += abs(nb[k] - q[k]) > lim;
        if (bad) atomicAdd(&sat[kClipLattice], (unsigned)bad);
    }
}
__global__ void int_to_planes(const int* in, float* o0, float* o1, float* o2, size_t nodes) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nodes) return;
    o0[i] = (float)in[3 * i];
    o1[i] = (float)in[3 * i + 1];
    o2[i] = (float)in[3 * i + 2];
}

// rANS decoding: state on 32 bits, probabilities on 12 bits, renormalisation by bytes. One thread per block of 1024
// coefficients; each block is an independent stream that begins with its state. The tables are per level, stored in the
// order of the innovations (from the coarsest to the finest); lv_end[j] is the end index of the j-th level in this
// ordering.
//
// Reading bound of a block, whatever the content of the stream (altered file included): kRansBlockMax bytes from the
// start of the block. The state of a valid stream is at least 2^23; a smaller initial state can only come from an
// altered file, and it is brought back to 2^23 (one test per block, with no effect on a valid stream). From then on,
// before each symbol x >= 2^23, so x >> 12 >= 2^11; the frequency of the decoded symbol is at least 1 (the inverse
// table returns only symbols that are present), so the state is at least 2^11 after the symbol and the renormalisation
// reads two bytes at most. Without this floor, a zero state followed by zero bytes would make the thread read without
// end. The caller guarantees that the start of each block is in the buffer and that the buffer extends at least
// kRansBlockMax bytes beyond this start.
constexpr size_t kRansBlockMax = 4 + 2 * 1024;
__global__ void rans_decode(const unsigned char* payload, const uint32_t* blk_off, const unsigned char* lut, const unsigned short* freq,
                            const unsigned short* cum, const unsigned long long* lv_end, signed char* e, size_t ecount) {
    const size_t b = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t i0 = b * 1024;
    if (i0 >= ecount) return;
    const size_t i1 = i0 + 1024 < ecount ? i0 + 1024 : ecount;
    const unsigned char* p = payload + blk_off[b];
    uint32_t x = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    if (x < (1u << 23)) x = 1u << 23;
    p += 4;
    int lv = 0;
    for (size_t i = i0; i < i1; ++i) {
        while (i >= lv_end[lv]) ++lv;
        const uint32_t slot = x & 4095u;
        const uint32_t sym = lut[lv * 4096 + slot];
        x = (uint32_t)freq[lv * 256 + sym] * (x >> 12) + slot - (uint32_t)cum[lv * 256 + sym];
        while (x < (1u << 23)) x = (x << 8) | *p++;
        e[i] = (signed char)sym;
    }
}

static double seconds_since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
static unsigned blocks_for(size_t count, unsigned block = 256) { return (unsigned)((count + block - 1) / block); }

// Limits of the codec that nothing else guards; to be checked as soon as the geometry is known (ecount: number of
// coefficients of a frame, all levels taken together).
//   - ExEntry.idx carries the index of the coefficient on 31 bits, bit 31 distinguishing the two kinds of exception:
//     beyond that, the exceptions would designate wrong nodes, without an error. Three levels: 856 nodes per axis, so
//     512 (power of two).
//   - up_at weights the partial sums by weights of sum 2^6 (degree 2) or 2^9 (degree 3), in an int. The sum of the
//     L - 1 coarse levels, each bounded by 32768 << fb, must therefore stay under 2^25 or 2^22.
static void check_codec_limits(int D, int L, int fb, size_t ecount) {
    CHK(fb >= 0 && fb <= 12, "fractional bits (--fb): from 0 to 12");
    CHK(ecount < ((size_t)1 << 31), "too many coefficients per frame: the index of the exceptions fits in 31 bits");
    const unsigned long long coarse = (unsigned long long)(L > 1 ? L - 1 : 0) * (32768ull << fb);
    CHK(coarse < (1ull << (D == 2 ? 25 : 22)), "--fb too large for this number of levels: the integer refinement of the pyramid would overflow");
}
static double sinc_pi(double x) { return std::fabs(x) < 1e-12 ? 1.0 : std::sin(kPi * x) / (kPi * x); }

// Alias sums per axis: a_p[q] = sum over m of (q + nl m)^p sinc((q + nl m)/nl)^(2(D+1)).
static void alias_sums(int nl, int D, std::vector<double> a[3]) {
    for (int p = 0; p < 3; ++p) a[p].assign(nl, 0.0);
    const int R = 2000;
    for (int q = 0; q < nl; ++q)
        for (int m = -R; m <= R; ++m) {
            const double k = (double)q + (double)nl * m, s = sinc_pi(k / nl), w2 = std::pow(s * s, D + 1);
            a[0][q] += w2;
            a[1][q] += k * w2;
            a[2][q] += k * k * w2;
        }
}

// Runs fn(thread, start, end) on T contiguous slices of [0, count).
template <class F>
static void parallel_ranges(unsigned T, size_t count, F fn) {
    std::vector<std::thread> th;
    for (unsigned t = 0; t < T; ++t) th.emplace_back([=, &fn]() { fn(t, count * t / T, count * (t + 1) / T); });
    for (auto& x : th) x.join();
}

// rANS tables of a level from the histogram of its symbols: frequencies of sum 4096, at least 1 for every symbol that
// is present, cumulative sums, and inverse table (slot -> symbol).
static void rans_build(const size_t* cnt, uint16_t* freq, uint16_t* cum, uint8_t* lut) {
    size_t total = 0;
    for (int v = 0; v < 256; ++v) total += cnt[v];
    int f[256] = {}, sum = 0, best = 0;
    if (total == 0) {
        f[0] = 4096;
        sum = 4096;
    } else {
        for (int v = 0; v < 256; ++v) {
            if (!cnt[v]) continue;
            f[v] = std::max(1, (int)((double)cnt[v] * 4096.0 / (double)total + 0.5));
            sum += f[v];
            if (cnt[v] > cnt[best]) best = v;
        }
    }
    if (sum < 4096) f[best] += 4096 - sum;
    while (sum > 4096) {  // takes the excess away from the most frequent symbols
        int j = 0;
        for (int v = 1; v < 256; ++v)
            if (f[v] > f[j]) j = v;
        const int take = std::min(sum - 4096, f[j] - 1);
        f[j] -= take;
        sum -= take;
    }
    int c = 0;
    for (int v = 0; v < 256; ++v) {
        freq[v] = (uint16_t)f[v];
        cum[v] = (uint16_t)c;
        for (int k = 0; k < f[v]; ++k) lut[c + k] = (uint8_t)v;
        c += f[v];
    }
}

// Solves a symmetric positive definite 3x3 system (Gaussian elimination, partial pivoting).
static bool solve3(double G[3][3], double b[3], double x[3]) {
    double M[3][4];
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) M[i][j] = G[i][j];
        M[i][3] = b[i];
    }
    for (int c = 0; c < 3; ++c) {
        int piv = c;
        for (int r = c + 1; r < 3; ++r)
            if (std::fabs(M[r][c]) > std::fabs(M[piv][c])) piv = r;
        if (std::fabs(M[piv][c]) < 1e-300) return false;
        for (int j = 0; j < 4; ++j) std::swap(M[c][j], M[piv][j]);
        for (int r = c + 1; r < 3; ++r) {
            const double f = M[r][c] / M[c][c];
            for (int j = c; j < 4; ++j) M[r][j] -= f * M[c][j];
        }
    }
    for (int r = 2; r >= 0; --r) {
        double s = M[r][3];
        for (int c = r + 1; c < 3; ++c) s -= M[r][c] * x[c];
        x[r] = s / M[r][r];
    }
    return true;
}

#endif  // PSIT_CODEC_KERNELS_CUH
