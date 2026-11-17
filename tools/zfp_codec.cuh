// SPDX-License-Identifier: Apache-2.0
// zfp_codec.cuh - the cuZFP path shared by the reference paths of the measurement instruments: fixed-rate compression
// and decoding of a velocity frame on the GPU, one stream per component, and the kernels that interleave the decoded
// planes. Included by tools/playback_kernels.cuh when PSIT_WITH_ZFP is defined; the library does not use it.
#pragma once

#include <cuda_runtime.h>

#ifdef PSIT_WITH_ZFP
#include <zfp.h>
#endif

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------------------
// cuZFP: fixed rate, one 3D scalar component per stream
// ---------------------------------------------------------------------------------------

#ifdef PSIT_WITH_ZFP
// Interleaves three contiguous scalar planes into float4 (u, v, w, 0).
__global__ void interleave3(const float* a, const float* b, const float* c, float4* out,
                            size_t count) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = make_float4(a[i], b[i], c[i], 0.0f);
}

// Variant without an intermediate buffer: interleaves and writes directly into the cudaArray through a surface (the
// array must be allocated with cudaArraySurfaceLoadStore). The abscissa of surf3Dwrite is expressed in bytes.
__global__ void interleave_to_surface(const float* a, const float* b, const float* c,
                                      cudaSurfaceObject_t surf, int n) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= n || y >= n || z >= n) return;
    const size_t i = ((size_t)z * n + y) * n + x;
    surf3Dwrite(make_float4(a[i], b[i], c[i], 0.0f), surf, x * (int)sizeof(float4), y, z);
}

// The three components are coded separately. Field and stream are device pointers; zfp detects them
// (cudaPointerGetAttributes) and then performs neither an allocation nor a hidden copy. Two output layouts for the
// decoder:
//   planar (default) : three contiguous planes, then the interleave3 kernel to the float4 buffer;
//   strided          : direct write into the float4 buffer (stride of 4 floats), without a rearrangement pass but
//                      with partial writes of sectors.
struct ZfpCodec {
    zfp_stream* zs[3] = {};
    bitstream* bs[3] = {};
    zfp_field* fld_strided[3] = {};  // strided view of the float4 buffer (compression, strided decoding)
    zfp_field* fld[3] = {};          // target of the timed decoding
    size_t seg = 0;        // capacity of a stream segment (one component), multiple of 8
    size_t span = 0;       // 3 segments: bytes transferred per frame
    double rate = 0.0;     // effective rate retained by zfp, in bits per value

    // Pass 1: sets the rate and computes the size of the segments (no pointer required).
    void configure(int n, double requested_rate) {
        zfp_field* probe = zfp_field_3d(nullptr, zfp_type_float, n, n, n);
        zfp_stream* z = zfp_stream_open(nullptr);
        rate = zfp_stream_set_rate(z, requested_rate, zfp_type_float, 3, zfp_false);
        seg = (zfp_stream_maximum_size(z, probe) + 7) & ~(size_t)7;
        span = 3 * seg;
        zfp_stream_close(z);
        zfp_field_free(probe);
    }

    // Pass 2: attaches the streams and the fields to the device buffers. d_planar == nullptr selects the strided
    // layout.
    void attach(int n, float4* d_field, float* d_planar, void* d_stream) {
        const size_t plane = (size_t)n * n * n;
        for (int c = 0; c < 3; ++c) {
            fld_strided[c] = zfp_field_3d((float*)d_field + c, zfp_type_float, n, n, n);
            zfp_field_set_stride_3d(fld_strided[c], 4, (ptrdiff_t)4 * n, (ptrdiff_t)4 * n * n);
            fld[c] = d_planar ? zfp_field_3d(d_planar + c * plane, zfp_type_float, n, n, n)
                              : fld_strided[c];
            zs[c] = zfp_stream_open(nullptr);
            zfp_stream_set_rate(zs[c], rate, zfp_type_float, 3, zfp_false);
            if (!zfp_stream_set_execution(zs[c], zfp_exec_cuda)) {
                fprintf(stderr, "zfp: CUDA execution policy refused\n");
                exit(1);
            }
            bs[c] = stream_open((char*)d_stream + c * seg, seg);
            zfp_stream_set_bit_stream(zs[c], bs[c]);
        }
    }

    size_t compress() {
        size_t total = 0;
        for (int c = 0; c < 3; ++c) {
            zfp_stream_rewind(zs[c]);
            const size_t sz = zfp_compress(zs[c], fld_strided[c]);
            if (sz == 0) {
                fprintf(stderr, "zfp: CUDA compression failed (component %d)\n", c);
                exit(1);
            }
            total += sz;
        }
        return total;
    }

    void decompress() {
        for (int c = 0; c < 3; ++c) {
            zfp_stream_rewind(zs[c]);
            if (zfp_decompress(zs[c], fld[c]) == 0) {
                fprintf(stderr, "zfp: CUDA decompression failed (component %d)\n", c);
                exit(1);
            }
        }
    }

    void release() {
        for (int c = 0; c < 3; ++c) {
            if (fld[c] && fld[c] != fld_strided[c]) zfp_field_free(fld[c]);
            if (fld_strided[c]) zfp_field_free(fld_strided[c]);
            if (zs[c]) zfp_stream_close(zs[c]);
            if (bs[c]) stream_close(bs[c]);
        }
    }
};
#endif

