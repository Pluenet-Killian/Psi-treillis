# Comparison compressors

## What is here, and why

The measurements of `benchmarks/` compare Psi-treillis with four published compressors. This folder builds them from
their unmodified sources and gives the other scripts one way to call them, so that every measurement calls them
identically.

| Compressor | Source (git submodule of `third_party/`) | Revision measured |
|---|---|---|
| SZ3 | https://github.com/szcompressor/SZ3 | `db2eca3ea8ae526ee1328649a1caf1a691b45873` |
| SPERR | https://github.com/NCAR/SPERR | `9a16ab0ea3af8f4d534634a51d51d65acb9fcb37` |
| zfp, with its CUDA backend | https://github.com/LLNL/zfp | `4282154f3f0afcd9ea2f0d3e9a77b2cf38eaafab` |
| cuSZp | https://github.com/szcompressor/cuSZp | `b5d880f09ca16cc43fc834f1614920bb24ae874d` |

- `build_codecs.ps1` builds them into `build/codecs/` (Windows, MSVC 2022, Ninja).
- `cuszp_wrap/` is a wrapper CMake project for cuSZp, whose own CMakeLists does not configure with MSVC; it compiles
  the sources of cuSZp and its example program as they are.
- `sperr_capi.def` lists the two functions of the C interface of SPERR that the DLL exports.
- `comparison_codecs.py` calls each compressor on one velocity component at a time with an absolute bound on the
  largest error, reads the sequences, and measures the errors of every codec in the same way, in NumPy. The exact
  command of each compressor is at the top of the file.
- `decoded_grids.py` gives one frame as each chain decodes it, and caches it, so that the divergence, the spectra and
  the tracers are measured on the same decoded grids.

## How to use

```
git submodule update --init
.\benchmarks\codecs\build_codecs.ps1
```

## Caveats

- **Default settings.** No option of a compressor is tuned other than the bound. A specialist of one of them may
  obtain a better compression ratio with other settings.
- SPERR is built without OpenMP and runs on one thread: its command-line tool and its OpenMP loops do not compile with
  MSVC. Its speed is not measured here. Whether a multi-threaded build writes the same bytes has not been checked.
- zfp is called in fixed-accuracy mode for a bound and in fixed-rate mode for a rate; the GPU chain of the tracer
  experiment and of the visual comparator uses its CUDA backend at a fixed rate.
- The compressors are built and called on Windows only by these scripts.
