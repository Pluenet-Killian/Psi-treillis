# Reference files: compression, certified bound, read-back, encoding time

## What is measured and why

Twelve files are encoded by `psi_stream pack` on the two reference sequences, then read back by `psi_stream play`.
They are the files on which the claims about compression (COMP-1, COMP-2), about the certified bound (BOUND-1 to
BOUND-4), about the read-back (READBACK-1) and about the encoding time (ENC-1) rest, and other measurements reuse them
(`benchmarks/speed/`, `benchmarks/robustness/`).

| File | Sequence | Frames | Degree | Bound on the largest error |
|---|---|---|---|---|
| `f_t256_d3_t05`, `f_t256_d3_t04`, `f_t256_d3_t03` | `data/turb256.seq` | 16 to 72 | 3 | 5, 4 and 3 % of u′ |
| `f_t256_d2_t06`, `f_t256_d2_t05` | `data/turb256.seq` | 16 to 48 | 2 | 6 and 5 % of u′ |
| `f_t512_d3_t05`, `f_t512_d3_t04`, `f_t512_d3_t03` | `data/turb512.seq` | 0 to 54 | 3 | 5, 4 and 3 % of u′ |
| `f_t512_d2_t07`, `f_t512_d2_t06` | `data/turb512.seq` | 0 to 54 | 2 | 7 and 6 % of u′ |
| `z_t512_d2`, `z_t512_d3` | `data/turb512.seq` | 0 to 54 | 2 and 3 | none |

Every other setting of `pack` is its default. For each file:

- `pack` checks the fidelity itself, on all the voxels of every frame, and prints the compression ratio (raw bytes over
  bytes of the file, key frames included), the relative field error, the largest error of a component as a fraction
  of u′, and its encoding time per frame (the reading of the raw frames from disk is not in it);
- with `--tolerance`, `pack` also prints the certification threshold (the tolerance minus half of its arithmetic
  margin) and the number of frames whose largest error is above this threshold; if there is at least one, the file is
  written all the same and the exit code is 4;
- with `--probe`, `pack` writes the field of the encoder at points off the grid, in the last frame of the segment;
- `play` reads the file alone, with one million sorted particles, recomputes the fingerprint of the state after each
  frame and compares it with that of the frame table; with `--probe`, it compares its own field with that of the
  encoder at the points of the probe.

The files are written without a seal (`--no-seal`), so that their bytes stay those of the reference files measured so
far.

```
psi_stream pack --data SEQUENCE --file F.psi [--probe F.probe] --csv --no-seal --degree D --start S --frames N [--tolerance T]
psi_stream play --file F.psi --particles 1000000 --sort [--probe F.probe]
```

## How to replay

From the root of the repository, with `build/psi_stream` built (`tools/build.ps1 -Tools` on Windows, CMake with
`-DPSIT_BUILD_TOOLS=ON` elsewhere) and the two sequences in `data/`:

```
python benchmarks/compression/reference_files/reference_files.py
```

The script stops before any run if the GPU is busy (utilisation above 5 %, or a program of this repository computing
on it, as `nvidia-smi` reports them): the encoding time would be wrong. `--force` overrides this; the log then says
that the GPU was not free. Python 3.8 or later, standard library only; Windows and Linux.

## What the script writes

| Path | Content |
|---|---|
| `reference_files.csv` | one line per file, the columns below |
| `logs/pack_<file>.log`, `logs/play_<file>.log` | the raw output of each run: standard output, then standard error |
| `logs/gpu.log` | the state of the GPU at the start and at the end |
| `build/work/reference_files/<file>.psi`, `<file>.probe` | the encoded files and their probes (several GiB, not versioned) |

Columns of `reference_files.csv`. Every figure is copied as the program prints it; the script computes none.

| Column | Read from | Claims |
|---|---|---|
| `name` | the file, without its extension | |
| `pack_command`, `play_command` | the two commands, paths relative to the root | |
| `pack_exit_code` | exit code of `pack`: 0; 3, coefficients clipped; 4, bound missed; 1, refusal | |
| `pack_failure`, `play_failure` | first line `Failure:` or `CUDA error:` of the standard error, if any | |
| `file_bytes`, `sha256` | the file written | |
| `grid`, `degree`, `frames` | line `CSV,pack` | |
| `compression_ratio` | line `CSV,pack` | COMP-1, COMP-2, BOUND-2 |
| `encoding_ms_per_frame` | line `CSV,pack` | ENC-1 |
| `field_error_mean`, `field_error_worst` | line `CSV,pack`: relative field error, mean over the frames and worst frame | COMP-1, COMP-2 |
| `largest_error_over_u_rms` | line `CSV,pack`: largest error of a component, as a fraction of u′ | BOUND-1, BOUND-3, BOUND-4 |
| `tolerance_over_u_rms` | line `CSV,fine` (files with a bound) | BOUND-1 to BOUND-4 |
| `certification_threshold_over_u_rms` | line `CSV,passes` (files with a bound) | BOUND-1, BOUND-3, BOUND-4 |
| `frames_above_threshold` | line `CSV,fine` (files with a bound) | BOUND-1, BOUND-3, BOUND-4 |
| `play_exit_code` | exit code of `play`: 0; 1, a frame differs or the file is refused | |
| `frames_differing`, `final_positions` | line `Integrity` of `play`: frames whose state differs from that of the sender, fingerprint of the final positions of the particles | READBACK-1 |
| `probe_frame`, `probe_points`, `probe_largest_difference` | line `Probe` of `play` (files with a bound) | READBACK-1 |
| `reader_allocated_mib`, `receiver_state_mib` | line `GPU memory` of `play` | |
| `gpu_free_at_start` | `yes`, or `no` if the script was forced on a busy GPU | ENC-1 |

`benchmarks/compression/psi_stream_runs.py` holds the patterns that read these lines, each with the source line that
prints it. If a line is missing or has another shape, the script says which, for which file, and exits with code 1:
the log must then not be used. An exit code 4 or 1 of `pack` is a result, recorded as such; the script goes on.

## Caveats

- The encoding time is measured once per file, on one machine. It is valid only if the GPU ran nothing else:
  `gpu_free_at_start` must read `yes`. The sizes, errors, exit codes and fingerprints do not depend on the load.
- The largest error is the one `pack` measures at voxel centres, on the frames it encodes. Between two frames and off
  the grid, nothing is measured here; the probe compares the reader with the encoder, not with the original field,
  and in one frame only.
- The compression ratio is that of a whole file with one key frame every 32 frames: it is an inter-frame ratio.
- `sha256` identifies the bytes of a file, to compare two runs. Nothing here establishes that another toolchain or
  another GPU gives the same bytes.
- `reader_allocated_mib` is a difference of free memory of the GPU, read by the program before and after its
  allocations; `receiver_state_mib` is the size of the buffers of the receiver, which the program computes.
