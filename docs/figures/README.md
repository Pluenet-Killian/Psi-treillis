# Figures

Two kinds of images are kept here. Neither is drawn or edited by hand.

## Graphs

Written by `python tools/figures.py` from the logs of `benchmarks/`, in a light and a dark version for `README.md`
(`NAME_light.png`, `NAME_dark.png`); the same script writes the vector versions of the article in `paper/figures/`.

| Figure | Log it is drawn from | What it shows |
|---|---|---|
| `divergence` | `benchmarks/divergence/divergence.csv` | Divergence of one frame as each chain gives it back, on the grid and as read by trilinear interpolation |
| `tracers` | `benchmarks/tracers/compressors_256.csv` | Variance of the logarithm of the volume carried by a tracer, against time |
| `equal_bound` | `benchmarks/equal_bound/equal_bound_256.csv` | Compression ratio against relative RMS error, at an equal bound on the largest error |
| `validity` | `benchmarks/validity/validity.csv` | The resolution sweep: error against the Kolmogorov scale of the data, and compression ratio at a bound of 5 % |

## Captures of the visual comparator

Written by `tools\viewer\figures.ps1`, which re-encodes the files compared with the current encoder and then runs
`psi_view` (Windows only) on two chains side by side, on the same frames, with 4 million particles per side. What
`psi_view` printed for each capture is kept in `benchmarks/playback/figures.log`.

| Capture | Grid | Left side | Right side | View |
|---|---|---|---|---|
| `particles_256.png` | 256³ | cuZFP at 2 bits per value, decoded to a half-precision texture | Psi-treillis, degree 3 | particles |
| `divergence_256.png` | 256³ | the same | the same | divergence of the field read |
| `error_256.png` | 256³ | the same | the same | error against the raw frame |
| `particles_512.png` | 512³ | the same | the same | particles |
| `raw_512.png` | 512³ | raw frames uploaded to a half-precision texture | the same | particles |

The band at the bottom of a capture displays what the comparator measured during that capture. The values quoted in the
article and in `README.md` are not read from these images: they come from `benchmarks/playback/playback.csv`, a
separate run of the same program on files encoded with the same options, with the same number of particles sorted in
the same way, and timings differ from one run to the next.
