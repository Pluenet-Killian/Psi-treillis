# Tracer experiment

## What is measured, and why

Whether the divergence of a reconstructed field changes anything for passive tracers (claims LAGR-1 to LAGR-5 and
TRAC-1 of `docs/CLAIMS.md`). Tracers drawn uniformly are advected by RK4 in one frame of the velocity field, held
frozen, seen through a reconstruction path (`tools/tracers.cu`): Psi-treillis, tricubic interpolation of the raw frame,
a half-precision texture of the raw frame with hardware trilinear filtering, the same texture after cuZFP, or after a
grid decoded by SZ3, SZ3 in 4D or SPERR. In a stationary divergence-free field the flow preserves volume: the tracers
stay uniform and their count per bin follows Poisson's law.

Two measurements:

- **Clustering.** The excess of the dispersion index of the count of tracers per bin (variance / mean − 1), in bins of
  1 voxel after 10 Kolmogorov times, over tracer seeds 1 to 5, in standard errors. This criterion was fixed before the
  first analysis and has not changed: more than 5 standard errors for the trilinear texture and for cuZFP, less than 2
  in absolute value for Psi-treillis.
- **Volume carried by a tracer.** D, the integral along the trajectory of the divergence of the field the tracer sees,
  taken by centred differences at ±1/8 voxel of the sampler of the path. Its variance grows with time when the field
  has a divergence; the log gives the growth per Kolmogorov time.

The campaigns are listed by `python benchmarks/tracers/tracers.py list`.

## How to replay

```
python benchmarks/tracers/tracers.py run resolved_256 resolved_512 confirmation_256 confirmation_256_more_seeds
python benchmarks/tracers/tracers.py run long_256 offsets_256 compressors_256
python benchmarks/tracers/tracers.py analyse
```

Needs `build/tracers` (`tools/build.ps1 -Tools`, with `third_party/zfp`), the sequences of `data/`, and for
`compressors_256` the comparison compressors in `build/codecs/`. A campaign at 256³ with 8 tracers per voxel uses
several GiB of GPU memory.

## What the script writes

For each campaign: `<campaign>.csv` (one row per path, seed and instant), `logs/<campaign>_<path>_s<seed>.log` (the
output of each run), `<campaign>_analysis.txt` (the criterion, then the descriptive tables) and
`<campaign>_summary.csv` (one row per path: growth of var(D), standard deviation of D at the last instant, and the
figures of the criterion).

## Caveats

- **A frozen field is a test of a numerical property, not a simulation of turbulence.** The recorded sequences last
  only a few Kolmogorov times.
- **D depends on its instrument.** For Psi-treillis, whose field has zero divergence, D measures only the truncation of
  the centred differences. For a texture, the divergence seen depends on the spacing of the differences: the campaign
  `offsets_256` repeats the measurement at ±1/4 and ±1/16 voxel.
- The paths read from a decoded grid (`compressors_256`) are outside the clustering criterion fixed in advance: their
  excess dispersion is reported, not judged.
- `confirmation_256_more_seeds` was run after seeing the result of `confirmation_256`: it is descriptive.
- Tracers have no inertia. One frame per sequence.
