/* SPDX-License-Identifier: Apache-2.0 */
/* psitreillis.h - public C API of the Psi-treillis codec.
 *
 * Psi-treillis compresses time sequences of incompressible, periodic velocity fields on a cubic grid. The velocity is
 * stored as the curl of a B-spline vector potential, so the decoded field is divergence-free by construction. A file
 * (.psi) can be decoded back to a grid, or sampled directly at arbitrary points without decoding the grid.
 *
 * Scope: cubic grid, n a power of two, periodic in the three directions, three float32 components per point, at least
 * five frames. The codec runs on an NVIDIA GPU (CUDA).
 *
 * Conventions
 *   - A frame is 3 * n^3 floats: three planes u, v, w, each indexed (z * n + y) * n + x.
 *   - Point coordinates are in grid-index units: the point (i, j, k) is the centre of voxel (x = i, y = j, z = k), so
 *     sampling there returns the decoded value of that voxel. Coordinates are periodic with period n.
 *   - Every function returns PSIT_OK (0) on success. On failure, psit_last_error() gives a message for the calling
 *     thread.
 *   - A reader is not thread-safe: use one reader per thread.
 */
#ifndef PSITREILLIS_H
#define PSITREILLIS_H

#include <stddef.h>
#include <stdint.h>

#define PSIT_VERSION_MAJOR 0
#define PSIT_VERSION_MINOR 2
#define PSIT_VERSION_PATCH 0
#define PSIT_VERSION "0.2.0"

#if defined(_WIN32)
#  if defined(PSIT_BUILD)
#    define PSIT_API __declspec(dllexport)
#  else
#    define PSIT_API __declspec(dllimport)
#  endif
#else
#  define PSIT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Return codes. */
enum {
    PSIT_OK = 0,
    PSIT_ERROR = 1,        /* invalid argument, unreadable or altered file, GPU error: see psit_last_error() */
    PSIT_CLIPPED = 3,      /* compression: file written, but some coefficients were clipped (lower eps is too fine) */
    PSIT_BOUND_MISSED = 4  /* compression: file written, but the requested error bound is not certified */
};

typedef struct psit_reader psit_reader;

/* What a .psi file contains. Errors are those measured by the encoder against the original frames. */
typedef struct psit_info {
    int32_t n;             /* grid points per axis */
    int32_t frames;        /* number of frames in the file */
    int32_t degree;        /* degree of the B-spline potential: 2 or 3 */
    int32_t first_frame;   /* index of the first frame in the source sequence */
    int32_t key_interval;  /* one key frame every key_interval frames */
    int32_t format;        /* file format revision: 1, 2 or 3 */
    int32_t sealed;        /* 1 if the file carries a seal, verified at opening */
    int32_t has_bound;     /* 1 if an error bound was requested at compression */
    int32_t bound_held;    /* 1 if the encoder certified that bound on every voxel of every frame */
    float dt;              /* time between two frames */
    float u_rms;           /* RMS of one velocity component, in the units of the data */
    float eps;             /* target RMS quantisation error, as a fraction of u_rms */
    float tolerance;       /* requested bound on the largest component error, as a fraction of u_rms (0 if none) */
    float max_error;       /* largest component error over the file, as a fraction of u_rms (0 if not measured) */
    float rms_error;       /* relative RMS field error, mean over the frames (negative if not measured) */
    uint64_t file_bytes;   /* size of the .psi file */
    uint64_t raw_bytes;    /* size of the same frames as raw float32 */
} psit_info;

/* One frame of a .psi file. */
typedef struct psit_frame_info {
    int32_t is_key;        /* 1 for a key frame (decodable on its own) */
    uint64_t bytes;        /* compressed size of the frame */
    float rms_error;       /* relative RMS field error of this frame (negative if not measured) */
    float max_error;       /* largest component error of this frame, as a fraction of u_rms (0 if not measured) */
    float mean[3];         /* mean velocity of the frame */
} psit_frame_info;

/* Compression settings. Always start from psit_default_options(). */
typedef struct psit_options {
    int32_t degree;         /* 2 or 3 (default 3: more accurate, slower to sample) */
    float eps;              /* target RMS quantisation error, as a fraction of u_rms (default 0.005) */
    float tolerance;        /* bound on the largest component error at voxel centres, as a fraction of u_rms; 0: none */
    float abs_tolerance;    /* the same bound in the units of the data; used when tolerance is 0 */
    int32_t key_interval;   /* one key frame every key_interval frames (default 32) */
    int32_t first_frame;    /* first frame of the source to compress (default 0) */
    int32_t frame_count;    /* number of frames to compress; 0: all frames from first_frame */
    float u_rms;            /* RMS of one velocity component; 0: computed from the frames (one extra pass over them) */
    int32_t seal;           /* 1: end the file with a seal that lets a reader detect an altered header (default 1) */
    int32_t verbose;        /* 1: let the encoder print its report on stdout (default 0) */
    int32_t reserved[8];    /* must be zero */
} psit_options;

/* Called by psit_compress to get frame `frame` of the source: write 3 * n^3 floats into dst. Return 0 on success. */
typedef int (*psit_read_frame)(void* user, int frame, float* dst);

PSIT_API const char* psit_version(void);
PSIT_API const char* psit_last_error(void);

/* ---- Reading ---------------------------------------------------------------------------------------------------- */

/* Opens a .psi file and allocates the decoder on the GPU. The file stays open until psit_close. */
PSIT_API int psit_open(const char* path, psit_reader** reader);
PSIT_API void psit_close(psit_reader* reader);

PSIT_API int psit_get_info(const psit_reader* reader, psit_info* info);
PSIT_API int psit_get_frame_info(const psit_reader* reader, int frame, psit_frame_info* info);

/* Decodes frame `frame` (0 .. frames - 1) to the grid: `out` receives 3 * n^3 floats.
 * Frames are cheapest to read in increasing order; a jump restarts from the previous key frame. */
PSIT_API int psit_decode(psit_reader* reader, int frame, float* out);

/* Samples the velocity of frame `frame` at `count` points, without decoding the grid.
 * x, y, z: coordinates in grid-index units (see Conventions). u, v, w: `count` floats each. */
PSIT_API int psit_sample(psit_reader* reader, int frame, const float* x, const float* y, const float* z, size_t count,
                         float* u, float* v, float* w);

/* ---- Compression ------------------------------------------------------------------------------------------------ */

PSIT_API void psit_default_options(psit_options* options);

/* Compresses `frames` frames of an n^3 field, obtained one by one from `read`, into the file `psi_path`.
 * dt: time between two frames (stored in the file; any positive value if it has no meaning for the data).
 * Returns PSIT_OK, PSIT_CLIPPED, PSIT_BOUND_MISSED (the file is written in the three cases) or PSIT_ERROR. */
PSIT_API int psit_compress(psit_read_frame read, void* user, int n, int frames, float dt, const char* psi_path,
                           const psit_options* options);

/* Same, from frames held in memory: `data` points to frames * 3 * n^3 floats, frame after frame. */
PSIT_API int psit_compress_memory(const float* data, int n, int frames, float dt, const char* psi_path,
                                  const psit_options* options);

/* Same, from a sequence file in the project's own format (.seq, written by the `dns` generator or by `psit convert`). */
PSIT_API int psit_compress_file(const char* seq_path, const char* psi_path, const psit_options* options);

#ifdef __cplusplus
}
#endif

#endif /* PSITREILLIS_H */
