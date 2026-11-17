# The data

## What is here, and why

Every measurement of this repository uses sequences of forced isotropic turbulence produced by the solver of the
repository, `tools/dns.cu`: incompressible Navier-Stokes equations in a periodic cube of side 2π, pseudo-spectral,
dealiased by truncation at wavenumber n/3, forced at large scales. The sequences are not distributed: they weigh 9 to
96 GiB each. This folder says how to produce them and records what the solver wrote in the header of each
(`sequences.csv`), which is where the Kolmogorov scale and the Reynolds number quoted elsewhere come from.

## How to replay

Build the instruments (`tools/build.ps1 -Tools`), then, from the root of the repository:

```
build/dns --n 256 --nu 0.0028  --spinup 10 --frames 128 --steps-per-frame 4 --out data/turb256.seq
build/dns --n 512 --nu 0.00111 --spinup 10 --frames 64  --steps-per-frame 4 --out data/turb512.seq
build/dns --n 512 --nu 0.0006  --spinup 10 --frames 64  --steps-per-frame 4 --out data/turb512_nu0006.seq
python benchmarks/validity/validity.py simulate
python benchmarks/data/sequences.py
```

The solver seed is 42 unless `--seed` is given. `validity.py simulate` produces the 18 sequences of the resolution
sweep in `data/sweep/` (six viscosities, solver seeds 42, 7 and 11, 48 frames each). The tests produce their own 64³
sequence.

## What the script writes

`sequences.csv`: one row per sequence, with the grid, the number of frames, the time between frames, the viscosity,
the RMS of one velocity component, kmax·η, the Kolmogorov scale in voxels, the Taylor-scale Reynolds number and the
size of the file.

## Caveats

- **One solver, one family of flows.** The sequences are smooth at the scale of the grid by construction: their
  spectrum is truncated at n/3. Data from another solver may behave differently; no public dataset is measured.
- A simulation of turbulence is chaotic: another version of cuFFT, another compiler or another GPU changes the bytes
  of a sequence, and with them every figure measured on it, without any defect. The figures of this repository are
  those of the sequences produced on the machine recorded in `benchmarks/environment.txt`.
- The statistics of a header (RMS velocity, kmax·η, Reynolds number) are averages over the recorded frames. The
  realisation of solver seed 42 at viscosity 0.0028 exists twice, as `turb256.seq` (128 frames) and as
  `sweep/nu00280_seed42.seq` (48 frames): their common frames are identical, their headers are not.
