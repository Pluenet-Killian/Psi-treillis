# Energy spectrum of a frame as each chain gives it back

## What is measured, and why

Compression ratios and error norms do not say which scales of the flow a compressor alters. This measurement compares
the energy spectrum of a decoded frame with that of the original, for every chain compared in this repository
(claim SPEC-1 of `docs/CLAIMS.md`).

Instrument, in double precision: the energy spectrum by integer shells of the wavenumber, on a periodic domain of side
2π. From it: the energy (sum of E(k)), the enstrophy (sum of k² E(k)), and k10, the smallest shell where the decoded
spectrum departs from the original by more than 10 %, searched up to the truncation wavenumber of the solver (n/3).

Same frames, chains and bound as `benchmarks/divergence/`; the instrument and the criterion were fixed before the
measurement.

## How to replay

```
python benchmarks/spectra/spectra.py run
python benchmarks/spectra/spectra.py summary
```

Same requirements as `benchmarks/divergence/`. The decoded frames are shared with it through the cache of
`benchmarks/codecs/decoded_grids.py`.

## What the script writes

- `spectra.csv`: one row per realisation and per chain, with the energy and the enstrophy relative to the original and
  k10.
- `shells.csv`: E(k) for every realisation, chain and shell, for the figures.

## Caveats

- The spectrum of Psi-treillis is that of the grid decoded by the library, not of its continuous field.
- The shells above n/2 hold only the corner modes of the grid; they are in `shells.csv` and in the sums.
- k10 depends on its threshold of 10 %, fixed in advance; `shells.csv` allows another one to be tried.
- One frame per realisation, three realisations, one family of flows, one resolution.
