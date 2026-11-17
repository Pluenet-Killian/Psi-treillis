// SPDX-License-Identifier: Apache-2.0
// sequence.h - file format of a sequence of velocity fields, 3D + time (.seq).
//
// Fixed header, then n_frames frames. Each frame contains three contiguous float32 planes (u, then v, then w), each of
// N^3 values, index (z * N + y) * N + x. The value of index (x, y, z) is taken at the centre of the voxel: position
// ((x + 0.5) h, ...), h = L / N, periodic domain of size L = 2 pi.
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>

struct SequenceHeader {
    char magic[8];        // "TLSEQ01\0"
    int32_t n;            // resolution per axis
    int32_t n_frames;     // number of frames
    float dt_frame;       // time interval between two frames
    float nu;             // kinematic viscosity of the simulation
    float u_rms;          // RMS of one velocity component (mean over the sequence)
    float kmax_eta;       // resolution of the smallest scale (kmax * eta)
    float re_lambda;      // Taylor Reynolds number
    float reserved[9];
};

// Name of a source file without its folders, for the error messages: a message must not carry the paths of the
// machine that built the program.
inline const char* source_name(const char* path) {
    const char* name = path;
    for (const char* p = path; *p; ++p)
        if (*p == '/' || *p == '\\') name = p + 1;
    return name;
}

inline size_t sequence_frame_bytes(const SequenceHeader& h) {
    return 3 * (size_t)h.n * h.n * h.n * sizeof(float);
}

// Refusal of an unreadable sequence: a message and exit code 1, like the other refusals of the programs. An exception,
// which nobody caught, used to stop the program without a message, on a crash code.
// The library (src/psitreillis.cu) defines PSI_SEQUENCE_FAIL to throw an exception instead of stopping the program.
[[noreturn]] inline void sequence_fail(const char* msg) {
#ifdef PSI_SEQUENCE_FAIL
    PSI_SEQUENCE_FAIL(msg);
#else
    fprintf(stderr, "Failure: %s\n", msg);
    std::exit(1);
#endif
}

inline SequenceHeader sequence_read_header(FILE* f) {
    SequenceHeader h{};
    if (fread(&h, sizeof(h), 1, f) != 1 || std::string(h.magic, 7) != "TLSEQ01")
        sequence_fail("invalid sequence file (TLSEQ01 header expected)");
    if (h.n < 8 || h.n > 8192 || h.n_frames < 1) sequence_fail("invalid sequence file (resolution or number of frames)");
    return h;
}

// Reads frame k into dst (3 N^3 floats). 64-bit positioning.
inline void sequence_read_frame(FILE* f, const SequenceHeader& h, int k, float* dst) {
    const long long off = (long long)sizeof(SequenceHeader) + (long long)k * (long long)sequence_frame_bytes(h);
#ifdef _WIN32
    _fseeki64(f, off, SEEK_SET);
#else
    fseeko(f, off, SEEK_SET);
#endif
    if (fread(dst, 1, sequence_frame_bytes(h), f) != sequence_frame_bytes(h))
        sequence_fail("incomplete frame read (truncated sequence, or frame requested beyond the end)");
}
