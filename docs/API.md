# Using Psi-treillis: library, command line, Python

Version 0.2.0. This page describes the public interface: the C library (`include/psitreillis.h`), the `psit` tool and
the Python module (`python/psitreillis`). The file format is specified in [FORMAT.md](FORMAT.md).

## What the codec accepts

- A time sequence of velocity fields on a **cubic grid**, `n` points per axis, **`n` a power of two, at least 64**.
- **Periodic** in the three directions. The encoder projects the field with a periodic FFT.
- **Incompressible.** The velocity is stored as the curl of a potential: only the divergence-free part of the input is
  kept. On a field that is not solenoidal, the error is the size of its compressible part.
- Three **float32** components per point, **at least 5 frames**.
- An **NVIDIA GPU**, for compression and for reading.

It is not a general-purpose compressor for arbitrary arrays.

## Conventions

- **A frame** is `3 * n^3` floats: three planes `u`, `v`, `w`, each indexed `(z * n + y) * n + x`. In NumPy, a frame is
  an array of shape `(3, n, n, n)` indexed `[component, z, y, x]`, and a sequence has the shape `(F, 3, n, n, n)`.
- **Points** are given in grid-index units: the point `(i, j, k)` is the centre of the voxel `x = i, y = j, z = k`.
  Sampling there returns the decoded value of that voxel. Coordinates are periodic with period `n`.
- **`u_rms`** is the RMS of one velocity component. Errors and tolerances are fractions of it.
- **Frames are numbered from 0 in the file**, whatever their index in the source (`first_frame` records it).

## Two ways to control the error

| Setting | Meaning | Default |
|---|---|---|
| `eps` | Target RMS quantisation error, as a fraction of `u_rms`. Sets the size and the average error. | 0.005 |
| `tolerance` | Bound on the **largest** error of a component at voxel centres, as a fraction of `u_rms`. The encoder adds a sparse refinement where the base misses it, then checks every voxel of every frame. | none |
| `abs_tolerance` | The same bound in the units of the data. | none |

What the bound covers: voxel centres, at the times of the frames. Between frames and off the grid nothing is
guaranteed. The check is made by the encoder with the arithmetic of the reader, less a margin of `1e-4 u_rms` that is
empirical.

A bound cannot be arbitrarily tight for a given `eps`. With the default `eps`, bounds of 5, 4 and 3 % of `u_rms` were
certified on the project's 256^3 and 512^3 turbulence sequences, and down to 3.4 % at 64^3. A tighter bound needs a
smaller `eps`; otherwise the compression is refused ("the error bound is too tight for this eps").

## Command line: `psit`

```
psit info FILE.psi
psit compress INPUT OUTPUT.psi [--degree 2|3] [--eps 0.005] [--tolerance T | --abs-tolerance A]
              [--key-interval 32] [--first K] [--count C] [--n N] [--dt DT] [--u-rms U] [--no-seal] [--verbose] [--quiet]
psit decompress FILE.psi OUTPUT [--first K] [--count C] [--quiet]
psit sample FILE.psi --frame K --at X,Y,Z [--at X,Y,Z ...]
psit convert INPUT OUTPUT [--n N] [--dt DT] [--first K] [--count C]
```

Field files are recognised by their extension:

| Extension | Content |
|---|---|
| `.npy` | NumPy array, little-endian float32, C order, shape `(F, 3, n, n, n)` |
| `.seq` | The project's own sequence format: a 72-byte header, then the frames |
| anything else | Raw little-endian float32, frame after frame; give the grid size with `--n` |

Files are read and written frame by frame: a sequence larger than memory is fine.

Exit codes: 0 success; 1 error; 3 file written but coefficients were clipped; 4 file written but the error bound is
not certified.

```powershell
psit compress flow.npy flow.psi --tolerance 0.05     # certified bound of 5 % of u_rms
psit info flow.psi
psit decompress flow.psi back.npy
psit sample flow.psi --frame 10 --at 12.5,40,7.25
```

From a `.seq` file, `psit compress` writes byte for byte the file that `psi_stream pack` writes with the same settings.

## Python

```python
import numpy as np
import psitreillis as psi

info = psi.compress(frames, "flow.psi", dt=0.01, tolerance=0.05)   # frames: float32, shape (F, 3, n, n, n)
print(info.ratio, info.rms_error, info.max_error, info.bound_held)

with psi.open("flow.psi") as f:
    u = f.decode(10)                     # (3, n, n, n) float32
    v = f.sample(10, points)             # points (N, 3) of (x, y, z) in grid-index units -> (N, 3)
    k = f.frame_info(10)                 # is_key, bytes, rms_error, max_error, mean
```

- `psi.compress(frames, path, *, dt, degree, eps, tolerance, abs_tolerance, key_interval, first_frame, frame_count,
  u_rms, seal, verbose)` returns the information of the file written. An `np.memmap` is read in place.
- `psi.compress_file(seq_path, path, ...)` does the same from a `.seq` file.
- `psi.open(path)` returns a `Reader`: `info`, `n`, `frames`, `decode(frame, out=None)`, `sample(frame, points)`,
  `frame_info(frame)`, `close()`. It is a context manager, `len(reader)` is the number of frames and `reader[k]`
  decodes frame `k`.
- A failure raises `psi.PsiError`. A file written without its bound certified, or with clipped coefficients, emits a
  `psi.PsiWarning`.

The module is a ctypes layer over the shared library. It looks for the library in the environment variable
`PSITREILLIS_LIBRARY`, next to the module, then in the build folders of a source checkout. It is not packaged for `pip`
yet: add `python/` to `PYTHONPATH`.

## C

```c
#include "psitreillis.h"

psit_options options;
psit_default_options(&options);
options.tolerance = 0.05f;
int rc = psit_compress_memory(data, n, frames, dt, "flow.psi", &options);   /* data: frames * 3 * n^3 floats */
if (rc == PSIT_ERROR) fprintf(stderr, "%s\n", psit_last_error());

psit_reader* reader;
if (psit_open("flow.psi", &reader) == PSIT_OK) {
    psit_info info;
    psit_get_info(reader, &info);
    psit_decode(reader, 10, grid);                          /* grid: 3 * n^3 floats */
    psit_sample(reader, 10, x, y, z, count, u, v, w);       /* count points */
    psit_close(reader);
}
```

| Function | Purpose |
|---|---|
| `psit_open`, `psit_close` | Open a file and allocate the decoder on the GPU; release everything |
| `psit_get_info`, `psit_get_frame_info` | Grid, frames, sizes, errors measured by the encoder, bound and whether it holds |
| `psit_decode` | One frame to the grid |
| `psit_sample` | Velocity of one frame at arbitrary points, without decoding the grid |
| `psit_default_options` | Fill `psit_options` with the defaults |
| `psit_compress` | Compress frames obtained from a callback |
| `psit_compress_memory` | Compress frames held in memory |
| `psit_compress_file` | Compress a `.seq` file |
| `psit_last_error`, `psit_version` | Message of the last failure on the calling thread; version |

Every function returns `PSIT_OK` (0) or `PSIT_ERROR` (1). The compression functions also return `PSIT_CLIPPED` (3) and
`PSIT_BOUND_MISSED` (4): the file is written, and `psit_last_error()` says what is wrong with it.

With CMake, after `cmake --install`:

```cmake
find_package(psitreillis REQUIRED)
target_link_libraries(my_program PRIVATE psitreillis::psitreillis)
```

## Behaviour to know

- **Reading order.** Frames are predicted from the previous ones. Reading in increasing order decodes one frame per
  call; a jump decodes from the previous key frame (one every `key_interval` frames).
- **Integrity.** Each decoded frame is checked against a fingerprint stored in the file, and a sealed file has its
  header checked at opening. An altered file is refused with an error; it does not crash the caller.
- **Memory.** `psit_close` releases what the reader allocated on the GPU. A compression releases what it allocated
  when it returns, whether it succeeded or not.
- **Threads.** A reader must not be used from two threads at once. Errors are per thread.
- **Wording of refusals.** The codec core words its refusals for its own command-line programs. The most common ones
  are reworded for a caller of the library, followed by the original message in square brackets.
- **Report of the encoder.** With `verbose`, the encoder prints its report on stdout.

## Not done yet

- The library builds on Linux (checked at each commit on a machine without a GPU) but has not been run there.
- Reading without a GPU: the two reference readers of `tools/` read a file and give the velocity at chosen points on a
  CPU, but they are verification tools, not part of this interface.
- No `pip` package, no HDF5 or NetCDF input.
- Grids that are not cubic, not periodic, or smaller than 64^3.
- Sampling between two frames in time.
