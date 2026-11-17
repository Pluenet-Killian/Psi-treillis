// SPDX-License-Identifier: Apache-2.0
// playback_kernels.cuh - kernels of the measurement instruments (tools/psi_stream.cu, tools/tracers.cu,
// tools/viewer/psi_view.cu). The library does not use them.
//
// Contents, in order:
//   - the reference paths that the lattice is compared with: tricubic interpolation of the raw frame, 3D texture
//     (float4 or half precision) with hardware trilinear filtering and, when PSIT_WITH_ZFP is defined, the planes
//     decoded by cuZFP written to a half-precision texture (tools/zfp_codec.cuh);
//   - the sampling of the lattice with a cache of the support, and the advection (RK4) and ray marching kernels that
//     use it, with or without the sparse fine level;
//   - the generic workloads, for any field that has a sample() function: particle advection (RK4), ray marching;
//   - particle sorting (Morton key and permutation), the fingerprint of an array of floats, and the "Seal" line that
//     the programs print about the file they read.
// The kernels of the codec itself (lattice, analytic curl, decoding) are in src/codec_kernels.cuh.
#ifndef PSIT_PLAYBACK_KERNELS_CUH
#define PSIT_PLAYBACK_KERNELS_CUH

#include "codec_stream.cuh"

#ifdef PSIT_WITH_ZFP
#include "zfp_codec.cuh"
#endif

// ---------------------------------------------------------------------------------------
// Reference: float4 frames in linear memory, tricubic interpolation (Catmull-Rom)
// ---------------------------------------------------------------------------------------

__host__ __device__ inline void catmull_rom(float f, float w[4]) {
    w[0] = ((-f + 2.0f) * f - 1.0f) * f * 0.5f;
    w[1] = ((3.0f * f - 5.0f) * f * f + 2.0f) * 0.5f;
    w[2] = ((-3.0f * f + 4.0f) * f + 1.0f) * f * 0.5f;
    w[3] = (f - 1.0f) * f * f * 0.5f;
}
__device__ inline V3 tricubic(const float4* f, int n, float inv_h, V3 p) {
    const float gx = p.x * inv_h - 0.5f, gy = p.y * inv_h - 0.5f, gz = p.z * inv_h - 0.5f;
    const float fx = floorf(gx), fy = floorf(gy), fz = floorf(gz);
    float wx[4], wy[4], wz[4];
    catmull_rom(gx - fx, wx);
    catmull_rom(gy - fy, wy);
    catmull_rom(gz - fz, wz);
    const int ix = (int)fx - 1, iy = (int)fy - 1, iz = (int)fz - 1, m = n - 1;
    V3 acc = {0.0f, 0.0f, 0.0f};
    for (int c = 0; c < 4; ++c) {
        const size_t zo = (size_t)((iz + c) & m) * n;
        for (int b = 0; b < 4; ++b) {
            const size_t row = (zo + ((iy + b) & m)) * n;
            const float wyz = wy[b] * wz[c];
            for (int a = 0; a < 4; ++a) {
                const float4 v = f[row + ((ix + a) & m)];
                const float w = wx[a] * wyz;
                acc.x += w * v.x;
                acc.y += w * v.y;
                acc.z += w * v.z;
            }
        }
    }
    return acc;
}

// Industrial reference: 3D float4 textures (or half precision), hardware trilinear filtering.
struct TexField {
    cudaTextureObject_t t0, t1;
    float inv_l;
};
__device__ inline V3 sample(const TexField& f, V3 p, float w) {
    const float4 a = tex3D<float4>(f.t0, p.x * f.inv_l, p.y * f.inv_l, p.z * f.inv_l);
    const float4 b = tex3D<float4>(f.t1, p.x * f.inv_l, p.y * f.inv_l, p.z * f.inv_l);
    return {a.x + w * (b.x - a.x), a.y + w * (b.y - a.y), a.z + w * (b.z - a.z)};
}
__global__ void pack_half4(const float4* in, ushort4* out, size_t voxels) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= voxels) return;
    out[i] = make_ushort4(__half_as_ushort(__float2half_rn(in[i].x)), __half_as_ushort(__float2half_rn(in[i].y)),
                          __half_as_ushort(__float2half_rn(in[i].z)), 0);
}

#ifdef PSIT_WITH_ZFP
// Three decoded planes -> half-precision texture, written directly through a surface.
__global__ void planes_to_half_surface(const float* a, const float* b, const float* c, cudaSurfaceObject_t surf, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= n || y >= n || z >= n) return;
    const size_t i = ((size_t)z * n + y) * n + x;
    surf3Dwrite(make_ushort4(__half_as_ushort(__float2half_rn(a[i])), __half_as_ushort(__float2half_rn(b[i])), __half_as_ushort(__float2half_rn(c[i])), 0),
                surf, x * (int)sizeof(ushort4), y, z);
}
#endif
// Word of the lattice whose six components are replaced by their difference from those of v0 (the proximity bits are
// cleared): used to bring the cache of the support back to a common base, which is then read directly.
__device__ __forceinline__ int4 lat_rebase(const int4 v, const int4 v0) {
    return make_int4(((v.x - v0.x) & 0xFFFF) | (int)((unsigned)((v.x >> 16) - (v0.x >> 16)) << 16), (v.y - v0.y) & 0xFFFF,
                     ((v.z - v0.z) & 0xFFFF) | (int)((unsigned)((v.z >> 16) - (v0.z >> 16)) << 16), (v.w - v0.w) & 0xFFFF);
}

// Cached variant: the (D+1)^3 words of the support are kept in registers. The four stages of an RK4 step, or the
// neighbouring samples of a ray, almost always fall in the same support: the lattice is read again only when the
// support changes.
// Inlining is forced: if the compiler keeps sample_cached as a function, the cache is passed by reference, lives in
// local memory and the kernel becomes 3.7 times slower (measured in psi_stream: 6.5 ms against 1.75 ms, for the same
// source).
template <int D>
struct NodeCache {
    int4 v[(D + 1) * (D + 1) * (D + 1)];
    int ix, iy, iz;
};
template <int D>
__device__ __forceinline__ V3 sample_cached(const PsiField<D>& f, NodeCache<D>& k, V3 p, float w) {
    float wx[4], dx[4], wy[4], dy[4], wz[4], dz[4];
    const int ix = bspline<D>((p.x - f.x0) * f.inv_h, wx, dx), iy = bspline<D>((p.y - f.x0) * f.inv_h, wy, dy),
              iz = bspline<D>((p.z - f.x0) * f.inv_h, wz, dz), m = f.n - 1;
    if (ix != k.ix || iy != k.iy || iz != k.iz) {
        int near_edge = 0;
        for (int c = 0; c <= D; ++c) {
            const size_t zo = (size_t)((iz + c) & m) * f.n;
            for (int b = 0; b <= D; ++b) {
                const size_t row = (zo + ((iy + b) & m)) * f.n;
                for (int a = 0; a <= D; ++a) {
                    const int4 v = f.lat[row + ((ix + a) & m)];
                    near_edge |= v.y | v.w;
                    k.v[(c * (D + 1) + b) * (D + 1) + a] = v;
                }
            }
        }
        if (near_edge & kLatNear) {  // a node is near the edge of the range: the cache is brought back to the difference from the first node
            const int4 v0 = k.v[0];
            for (int j = 0; j < (D + 1) * (D + 1) * (D + 1); ++j) k.v[j] = lat_rebase(k.v[j], v0);
        }
        k.ix = ix;
        k.iy = iy;
        k.iz = iz;
    }
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int c = 0; c <= D; ++c)
        for (int b = 0; b <= D; ++b) {
            const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
            for (int a = 0; a <= D; ++a) node_accum(k.v[(c * (D + 1) + b) * (D + 1) + a], f.early, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
        }
    return {f.s * ax + f.m0[0] + w * (f.m1[0] - f.m0[0]), f.s * ay + f.m0[1] + w * (f.m1[1] - f.m0[1]),
            f.s * az + f.m0[2] + w * (f.m1[2] - f.m0[2])};
}
template <int D>
__device__ __forceinline__ V3 sample_cached_plain(const PsiField<D>& f, NodeCache<D>& k, V3 p, float w) {
    float wx[4], dx[4], wy[4], dy[4], wz[4], dz[4];
    const int ix = bspline<D>((p.x - f.x0) * f.inv_h, wx, dx), iy = bspline<D>((p.y - f.x0) * f.inv_h, wy, dy),
              iz = bspline<D>((p.z - f.x0) * f.inv_h, wz, dz), m = f.n - 1;
    if (ix != k.ix || iy != k.iy || iz != k.iz) {
        for (int c = 0; c <= D; ++c) {
            const size_t zo = (size_t)((iz + c) & m) * f.n;
            for (int b = 0; b <= D; ++b) {
                const size_t row = (zo + ((iy + b) & m)) * f.n;
                for (int a = 0; a <= D; ++a) k.v[(c * (D + 1) + b) * (D + 1) + a] = f.lat[row + ((ix + a) & m)];
            }
        }
        k.ix = ix;
        k.iy = iy;
        k.iz = iz;
    }
    float ax = 0.0f, ay = 0.0f, az = 0.0f;
    for (int c = 0; c <= D; ++c)
        for (int b = 0; b <= D; ++b) {
            const float pyz = wy[b] * wz[c], dyz = dy[b] * wz[c], ydz = wy[b] * dz[c];
            for (int a = 0; a <= D; ++a) node_accum(k.v[(c * (D + 1) + b) * (D + 1) + a], f.early, w, dx[a] * pyz, wx[a] * dyz, wx[a] * ydz, ax, ay, az);
        }
    return {f.s * ax + f.m0[0] + w * (f.m1[0] - f.m0[0]), f.s * ay + f.m0[1] + w * (f.m1[1] - f.m0[1]),
            f.s * az + f.m0[2] + w * (f.m1[2] - f.m0[2])};
}
template <int D>
__global__ void rk4_psi_plain(PsiField<D> f, float* px, float* py, float* pz, uint32_t n, float w, float dw, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    NodeCache<D> k;
    k.ix = k.iy = k.iz = (int)0x80000000;
    V3 p = {px[i], py[i], pz[i]};
    const V3 k1 = sample_cached_plain<D>(f, k, p, w);
    const V3 k2 = sample_cached_plain<D>(f, k, p + (0.5f * dt) * k1, w + 0.5f * dw);
    const V3 k3 = sample_cached_plain<D>(f, k, p + (0.5f * dt) * k2, w + 0.5f * dw);
    const V3 k4 = sample_cached_plain<D>(f, k, p + dt * k3, w + dw);
    p = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    const float inv_l = 1.0f / kL;
    px[i] = p.x - kL * floorf(p.x * inv_l);
    py[i] = p.y - kL * floorf(p.y * inv_l);
    pz[i] = p.z - kL * floorf(p.z * inv_l);
}
template <int D>
__global__ void raymarch_psi_plain(PsiField<D> f, float* img, int width, int steps, float w, float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * width) return;
    NodeCache<D> k;
    k.ix = k.iy = k.iz = (int)0x80000000;
    const V3 o = {((i % width) + 0.5f) / width * kL, ((i / width) + 0.5f) / width * kL, 0.0f};
    const V3 d = {0.36f, 0.48f, 0.80f};
    const float ds = kL / steps;
    float transmittance = 1.0f, radiance = 0.0f;
    for (int s = 0; s < steps; ++s) {
        const V3 v = sample_cached_plain<D>(f, k, o + ((s + 0.5f) * ds) * d, w);
        const float e = v.x * v.x + v.y * v.y + v.z * v.z;
        const float a = 1.0f - expf(-sigma * e * ds);
        radiance += transmittance * a * e;
        transmittance *= 1.0f - a;
    }
    img[i] = radiance;
}
// Advection with a cache of the support of the base lattice, plus the fine level: for unsorted particles, where the
// kernel without a cache is the slowest. PLAIN: direct reading of the base lattice (see PsiPlain).
template <int D, bool PLAIN>
__global__ void rk4_psi_fine(PsiField<D> f, FineLevel fl, float* px, float* py, float* pz, uint32_t n, float w, float dw, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    NodeCache<D> k;
    k.ix = k.iy = k.iz = (int)0x80000000;
    auto at = [&](V3 x, float ww) {
        const V3 u = PLAIN ? sample_cached_plain<D>(f, k, x, ww) : sample_cached<D>(f, k, x, ww);
        return u + fine_at<D>(fl, x, ww, f.early);
    };
    V3 p = {px[i], py[i], pz[i]};
    const V3 k1 = at(p, w);
    const V3 k2 = at(p + (0.5f * dt) * k1, w + 0.5f * dw);
    const V3 k3 = at(p + (0.5f * dt) * k2, w + 0.5f * dw);
    const V3 k4 = at(p + dt * k3, w + dw);
    p = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    const float inv_l = 1.0f / kL;
    px[i] = p.x - kL * floorf(p.x * inv_l);
    py[i] = p.y - kL * floorf(p.y * inv_l);
    pz[i] = p.z - kL * floorf(p.z * inv_l);
}

// ---------------------------------------------------------------------------------------
// Workloads: RK4 advection, ray marching, probe
// ---------------------------------------------------------------------------------------

template <class F>
__global__ void rk4_kernel(F f, float* px, float* py, float* pz, uint32_t n, float w, float dw, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    V3 p = {px[i], py[i], pz[i]};
    const V3 k1 = sample(f, p, w);
    const V3 k2 = sample(f, p + (0.5f * dt) * k1, w + 0.5f * dw);
    const V3 k3 = sample(f, p + (0.5f * dt) * k2, w + 0.5f * dw);
    const V3 k4 = sample(f, p + dt * k3, w + dw);
    p = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    const float inv_l = 1.0f / kL;
    px[i] = p.x - kL * floorf(p.x * inv_l);
    py[i] = p.y - kL * floorf(p.y * inv_l);
    pz[i] = p.z - kL * floorf(p.z * inv_l);
}
template <int D>
__global__ void rk4_psi(PsiField<D> f, float* px, float* py, float* pz, uint32_t n, float w, float dw, float dt) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    NodeCache<D> k;
    k.ix = k.iy = k.iz = (int)0x80000000;
    V3 p = {px[i], py[i], pz[i]};
    const V3 k1 = sample_cached<D>(f, k, p, w);
    const V3 k2 = sample_cached<D>(f, k, p + (0.5f * dt) * k1, w + 0.5f * dw);
    const V3 k3 = sample_cached<D>(f, k, p + (0.5f * dt) * k2, w + 0.5f * dw);
    const V3 k4 = sample_cached<D>(f, k, p + dt * k3, w + dw);
    p = p + (dt / 6.0f) * (k1 + 2.0f * k2 + 2.0f * k3 + k4);
    const float inv_l = 1.0f / kL;
    px[i] = p.x - kL * floorf(p.x * inv_l);
    py[i] = p.y - kL * floorf(p.y * inv_l);
    pz[i] = p.z - kL * floorf(p.z * inv_l);
}

// Volumetric ray marching: oblique orthographic rays, fixed step, emission-absorption model on the local kinetic
// energy |u|^2.
template <class F>
__global__ void raymarch_kernel(F f, float* img, int width, int steps, float w, float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * width) return;
    const V3 o = {((i % width) + 0.5f) / width * kL, ((i / width) + 0.5f) / width * kL, 0.0f};
    const V3 d = {0.36f, 0.48f, 0.80f};
    const float ds = kL / steps;
    float transmittance = 1.0f, radiance = 0.0f;
    for (int s = 0; s < steps; ++s) {
        const V3 v = sample(f, o + ((s + 0.5f) * ds) * d, w);
        const float e = v.x * v.x + v.y * v.y + v.z * v.z;
        const float a = 1.0f - expf(-sigma * e * ds);
        radiance += transmittance * a * e;
        transmittance *= 1.0f - a;
    }
    img[i] = radiance;
}
template <int D>
__global__ void raymarch_psi(PsiField<D> f, float* img, int width, int steps, float w, float sigma) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= width * width) return;
    NodeCache<D> k;
    k.ix = k.iy = k.iz = (int)0x80000000;
    const V3 o = {((i % width) + 0.5f) / width * kL, ((i / width) + 0.5f) / width * kL, 0.0f};
    const V3 d = {0.36f, 0.48f, 0.80f};
    const float ds = kL / steps;
    float transmittance = 1.0f, radiance = 0.0f;
    for (int s = 0; s < steps; ++s) {
        const V3 v = sample_cached<D>(f, k, o + ((s + 0.5f) * ds) * d, w);
        const float e = v.x * v.x + v.y * v.y + v.z * v.z;
        const float a = 1.0f - expf(-sigma * e * ds);
        radiance += transmittance * a * e;
        transmittance *= 1.0f - a;
    }
    img[i] = radiance;
}

__global__ void planar_to_f4(const float* planar, float4* out, size_t voxels) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < voxels) out[i] = make_float4(planar[i], planar[voxels + i], planar[2 * voxels + i], 0.0f);
}
// Particle sorting: Morton key on 10 bits per axis, then permutation.
__device__ inline uint32_t spread10(uint32_t v) {
    v &= 0x3FFu;
    v = (v | (v << 16)) & 0x030000FFu;
    v = (v | (v << 8)) & 0x0300F00Fu;
    v = (v | (v << 4)) & 0x030C30C3u;
    v = (v | (v << 2)) & 0x09249249u;
    return v;
}
__global__ void morton_keys(const float* px, const float* py, const float* pz, uint32_t n, uint32_t* key, uint32_t* idx) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = 1024.0f / kL;
    const uint32_t x = (uint32_t)fminf(1023.0f, fmaxf(0.0f, px[i] * s)), y = (uint32_t)fminf(1023.0f, fmaxf(0.0f, py[i] * s)),
                   z = (uint32_t)fminf(1023.0f, fmaxf(0.0f, pz[i] * s));
    key[i] = spread10(x) | (spread10(y) << 1) | (spread10(z) << 2);
    idx[i] = i;
}
__global__ void gather3(const float* ax, const float* ay, const float* az, const uint32_t* idx, uint32_t n, float* bx, float* by, float* bz) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint32_t j = idx[i];
    bx[i] = ax[j];
    by[i] = ay[j];
    bz[i] = az[j];
}
__global__ void hash_floats(const float* a, uint32_t n, unsigned long long salt, unsigned long long* out) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    atomicXor(out, mix64(salt + ((unsigned long long)i << 32) + __float_as_uint(a[i])));
}
static void print_seal(const StreamOpen& so) {
    if (so.sealed)
        printf("Seal           : present and verified at opening: header%s and frame table unchanged since writing\n",
               so.has_fine ? ", header of the fine level" : "");
    else
        printf("Seal           : absent (file from before the seal, or pack --no-seal): the header and the frame table are covered by no "
               "fingerprint\n");
}

#endif  // PSIT_PLAYBACK_KERNELS_CUH
