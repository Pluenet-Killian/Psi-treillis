# Under-resolved turbulence: the encodings of the reference files on a less resolved sequence

## What is measured and why

The reference files (`benchmarks/compression/reference_files/`) are encoded on turbulence whose Kolmogorov scale is
0.74 voxel at 512³. This measurement repeats the encodings at 512³ on `data/turb512_nu0006.seq`, a sequence of the
same solver at a viscosity of 0.0006, whose Kolmogorov scale is 0.45 voxel (`benchmarks/data/` holds the production
log of each sequence). That is about the resolution, in voxels, of the public JHTDB isotropic dataset at 1,024³
(0.47 voxel, calculated from its description). It asks two questions: does the compression without a bound hold on
such a sequence (claim RESO-1), and does `pack --tolerance` still hold its bound (claim RESO-2)?

Same segment as the reference files at 512³ (frames 0 to 54), same settings:

| File | Degree | Bound on the largest error | Off-grid probe |
|---|---|---|---|
| `d2`, `d3` | 2 and 3 | none | no |
| `d3_t05`, `d3_t04`, `d3_t03` | 3 | 5, 4 and 3 % of u′ | yes |
| `d2_t07` | 2 | 7 % of u′ | yes |

```
psi_stream pack --data data/turb512_nu0006.seq --file F.psi --start 0 --frames 55 --csv --degree D [--tolerance T] [--probe F.probe]
psi_stream play --file F.psi --particles 4000000 --sort --csv [--probe F.probe]
```

The exit code of `pack` is the result that RESO-2 reads: 0, the bound is held; 4, the file is written but at least one
frame is above the certification threshold; 1, `pack` refuses and stops, which it does when the tolerance asks the fine
level to refine more than a quarter of the volume. A refused encoding leaves a file without a frame table; `play` is
run on it all the same, and must refuse it.

Three more files, without a bound, are then encoded with a probe. They bracket a difference between the field of the
reader and that of the encoder that the probe shows in degree 2 on this sequence: the same sequence in degree 2 and in
degree 3, and the resolved sequence in degree 2.

| File | Sequence | Degree |
|---|---|---|
| `probe_nu0006_d2`, `probe_nu0006_d3` | `data/turb512_nu0006.seq` | 2 and 3 |
| `probe_turb512_d2` | `data/turb512.seq` | 2 |

```
psi_stream pack --data SEQUENCE --file F.psi --probe F.probe --start 0 --frames 55 --csv --degree D
psi_stream play --file F.psi --particles 1000000 --sort --probe F.probe
```

Unlike the reference files, these nine files are written with their seal, the default of `pack`.

## How to replay

From the root of the repository, with `build/psi_stream` built (`tools/build.ps1 -Tools` on Windows, CMake with
`-DPSIT_BUILD_TOOLS=ON` elsewhere) and the two sequences in `data/`:

```
python benchmarks/compression/under_resolved/under_resolved.py
```

Python 3.8 or later, standard library only; Windows and Linux. The script does not wait for a free GPU: no claim reads
a time in its log.

## What the script writes

| Path | Content |
|---|---|
| `under_resolved.csv` | one line per file, with the columns of `reference_files.csv` (described in `benchmarks/compression/reference_files/README.md`) |
| `logs/pack_<file>.log`, `logs/play_<file>.log` | the raw output of each run: standard output, then standard error |
| `logs/gpu.log` | the state of the GPU at the start and at the end |
| `build/work/under_resolved/<file>.psi`, `<file>.probe` | the encoded files and their probes (several GiB, not versioned) |

What the claims read: RESO-1, the columns `compression_ratio`, `field_error_mean` and `largest_error_over_u_rms` of
the file `d3`; RESO-2, the column `pack_exit_code` of `d3_t05`, `d3_t04`, `d3_t03` and `d2_t07`. For a refused
encoding the columns of figures are empty, and `pack_failure` and `play_failure` hold the messages of the two
programs.

If a line that the script reads is missing or has another shape, the script says which, for which file, and exits
with code 1: the log must then not be used. The exit codes 4 and 1 of `pack` are results, recorded as such.

## Caveats

- One sequence, one realisation, one segment of 55 frames. The sweep of `benchmarks/validity/` covers other
  resolutions and other realisations at 256³.
- The resolution is counted in voxels per Kolmogorov scale, from the `kmax * eta` that the solver records in the
  header of the sequence. The spectrum of this solver is truncated at one third of the grid; a dataset of the same
  resolution in voxels produced otherwise is not this sequence.
- The times printed in the raw logs (encoding, decoding, RK4 step) are not controlled: the GPU may be shared.
  `gpu_free_at_start` and `logs/gpu.log` record its state.
- The probe compares the reader with the encoder, at points off the grid and in the last frame only; it says nothing
  of the distance to the original field.
