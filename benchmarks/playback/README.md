# Cost of one replayed frame

## What is measured, and why

What it costs to replay a simulation on the GPU, frame after frame, with particles advected through it: the data read
from disk, the GPU memory held, and the time of one frame, for three chains (claims DIV-1 and PLAY-1 of
`docs/CLAIMS.md`):

- raw float32 frames read from disk and uploaded to a half-precision 3D texture;
- cuZFP at a fixed rate of 2 bits per value, decoded on the GPU to the same kind of texture;
- Psi-treillis of degree 3, default settings, whose lattice is decoded on the GPU and sampled directly.

The visual comparator `build/psi_view` runs two chains on the same GPU and on the same frames and measures each of them
during the run. One protocol for the whole table, fixed before the measurement: synchronised mode, 4 million particles
per side sorted in Morton order, 2 RK4 steps per frame, one run per grid size (256³, 512³) and per pair of chains.

## How to replay

```
python benchmarks/playback/playback.py run
python benchmarks/playback/playback.py summary
```

Needs `build/` with the instruments (`tools/build.ps1 -Tools`; `psi_view` exists on Windows only) and the sequences
`data/turb256.seq` and `data/turb512.seq`. The GPU must run nothing else: the script stops otherwise.

The screenshots of the documentation are taken by `tools/viewer/figures.ps1`, which writes `figures.log` here.

## What the script writes

- `playback.csv`: one row per grid size, pair and side, with the data read per frame, the disk, bus, decoding and sort
  times, the time of one RK4 step, the cost of one frame, the GPU memory, the divergence of the field the particles
  see on a slice, and the error of the field on a slice against the raw frame.
- `logs/`: the output of each run of `psi_stream pack`, `psi_view packzfp` and `psi_view`.

## Caveats

- **Timings are those of one machine**, in one run per case; the column `gpu_free` says whether the GPU was free.
- The cost of a frame is the sum of the times the comparator measures for one side; in synchronised mode both sides
  share the GPU while they are measured.
- The raw chain reads 3 × 4 bytes per voxel from disk at every frame: its cost is mostly that read, and depends on the
  disk and on the cache of the system.
- The divergence and the error are measured on one slice.
- Psi-treillis is measured without a bound; cuZFP at 2 bits is at a fixed rate, not at a bound.
