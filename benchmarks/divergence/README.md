# Divergence of a frame as each chain gives it back

## What is measured, and why

An incompressible velocity field has zero divergence. This measurement asks what is left of that property once a frame
has been through a compressor and is read back, for every chain compared in this repository (claims GRID-1, GRID-2,
READ-1 and READ-2 of `docs/CLAIMS.md`).

Two instruments, both in double precision, both divided by the RMS of the velocity gradient of the original frame:

- **M1, divergence on the grid.** The spectral divergence of the decoded grid, RMS over the grid. No interpolation is
  involved: it is what compression alone does to the frame.
- **M2, divergence as read.** The analytic divergence of the trilinear interpolant of the grid, at 4,194,304 points
  drawn uniformly, RMS. It is what a tracer sees when the grid is read by trilinear interpolation, without the
  rounding of a half-precision texture.

The instruments, the data, the settings and the criteria were fixed before the measurement. The frame is frame 32 of
the three 256³ sequences of the resolution sweep at viscosity 0.0028 (solver seeds 42, 7 and 11). Every error-bounded
compressor receives the same absolute bound, 5 % of the RMS velocity; the chains are listed in
`benchmarks/codecs/decoded_grids.py`.

## How to replay

```
python benchmarks/divergence/divergence.py run
python benchmarks/divergence/divergence.py summary
```

Needs `build/` with the instruments (`tools/build.ps1 -Tools`), the comparison compressors in `build/codecs/`
(`benchmarks/codecs/build_codecs.ps1`), NumPy and SciPy, and the three sequences `data/sweep/nu00280_seed{42,7,11}.seq`
(`python benchmarks/validity/validity.py simulate`).

## What the script writes

`divergence.csv`: one row per realisation and per chain, with the compression ratio and the errors of the decoded
frame, the RMS gradient of the original, M1 and M2.

## Caveats

- **Psi-treillis is not read through this grid.** Its field is the analytic curl of the potential, whose divergence is
  zero by identity; claim DIV-1 measures it. The rows `psi` and `psi_free` are the grid decoded by the library, a
  sampling of that field at voxel centres, measured like the others for what it is.
- M2 is the divergence of an exact trilinear interpolant. A GPU chain filters a half-precision texture with weights
  of limited precision: its field is not exactly this one.
- The Nyquist wavenumber of each axis is set to zero in M1, as for any odd derivative on an even grid.
- One frame per realisation, three realisations, one family of flows, one resolution.
- `cuzfp_2bits` is at a fixed rate, not at the bound: its largest error is the one the log gives.
