/* SPDX-License-Identifier: Apache-2.0 */
/* roundtrip.c - compress a small synthetic velocity field, read it back, sample it at a point.
 *
 * The field is a sum of Fourier modes built as the curl of a potential, so it is periodic and divergence-free, which is
 * what the codec expects. Build it with the project (target example_roundtrip), or against an installed library:
 *
 *     find_package(psitreillis REQUIRED)
 *     target_link_libraries(my_program PRIVATE psitreillis::psitreillis)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "psitreillis.h"

#define N 64
#define FRAMES 8
#define TWO_PI 6.283185307179586

/* u = curl(A) with A = (sin(y + t), sin(z + t), sin(x + t)) plus one finer mode: analytic, periodic, solenoidal. */
static void fill_frame(float* frame, double t) {
    const size_t voxels = (size_t)N * N * N;
    for (int k = 0; k < N; ++k)
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) {
                const double x = TWO_PI * (i + 0.5) / N, y = TWO_PI * (j + 0.5) / N, z = TWO_PI * (k + 0.5) / N;
                const size_t v = ((size_t)k * N + j) * N + i;
                frame[v] = (float)(-cos(z + t) + 0.3 * 3.0 * cos(3.0 * y - t));
                frame[voxels + v] = (float)(-cos(x + t) + 0.3 * 3.0 * cos(3.0 * z - t));
                frame[2 * voxels + v] = (float)(-cos(y + t) + 0.3 * 3.0 * cos(3.0 * x - t));
            }
}

static int fail(const char* what) {
    fprintf(stderr, "%s: %s\n", what, psit_last_error());
    return 1;
}

int main(void) {
    const size_t frame_floats = 3 * (size_t)N * N * N;
    float* data = (float*)malloc(FRAMES * frame_floats * sizeof(float));
    float* decoded = (float*)malloc(frame_floats * sizeof(float));
    if (!data || !decoded) return 1;
    for (int f = 0; f < FRAMES; ++f) fill_frame(data + (size_t)f * frame_floats, 0.05 * f);

    /* Compress, with a certified bound of 5 % of u_rms on the largest error of a component. */
    psit_options options;
    psit_default_options(&options);
    options.tolerance = 0.05f;
    const int rc = psit_compress_memory(data, N, FRAMES, 0.05f, "roundtrip.psi", &options);
    if (rc == PSIT_ERROR) return fail("compress");
    if (rc != PSIT_OK) fprintf(stderr, "warning: %s\n", psit_last_error());

    /* Read it back. */
    psit_reader* reader = NULL;
    if (psit_open("roundtrip.psi", &reader) != PSIT_OK) return fail("open");
    psit_info info;
    if (psit_get_info(reader, &info) != PSIT_OK) return fail("info");
    printf("%d^3, %d frames, %.1f times smaller than raw; bound %s\n", info.n, info.frames, (double)info.raw_bytes / (double)info.file_bytes,
           info.bound_held ? "certified" : "not certified");

    /* Decode the last frame to the grid and measure the largest error ourselves. */
    const int last = FRAMES - 1;
    if (psit_decode(reader, last, decoded) != PSIT_OK) return fail("decode");
    const float* original = data + (size_t)last * frame_floats;
    double worst = 0.0;
    for (size_t i = 0; i < frame_floats; ++i) {
        const double e = fabs((double)decoded[i] - (double)original[i]);
        if (e > worst) worst = e;
    }
    printf("largest error on the last frame: %.4f %% of u_rms (bound: %.4f %%)\n", 100.0 * worst / info.u_rms, 100.0 * info.tolerance);

    /* Sample the velocity at an arbitrary point, straight from the compressed file. */
    const float x = 10.25f, y = 31.5f, z = 7.75f;
    float u, v, w;
    if (psit_sample(reader, last, &x, &y, &z, 1, &u, &v, &w) != PSIT_OK) return fail("sample");
    printf("velocity at (%.2f, %.2f, %.2f): %.5f %.5f %.5f\n", x, y, z, u, v, w);

    psit_close(reader);
    free(data);
    free(decoded);
    return worst <= info.tolerance * info.u_rms ? 0 : 1;
}
