# Validity map: the codec against the resolution of the data (claims VALID-1 to VALID-5)

## What is measured, and why

Psi-treillis represents the velocity field on a lattice that is coarser than the grid of the data: a step of 2 voxels by
default. Whether it can then certify a bound on the largest error depends on how well the data resolve the smallest
scales of the flow. This folder measures where that domain ends, and how the codec compares with others inside and
outside it.

The measurement is a sweep of the resolution: forced isotropic turbulence at 256^3 from the solver of the repository
(`tools/dns.cu`), six viscosities (0.0011, 0.0013, 0.0016, 0.0019, 0.0023 and 0.0028) and three realisations each
(solver seeds 42, 7 and 11), 48 frames per sequence. The resolution of a sequence is its Kolmogorov scale in voxels,
read from its header. On each of the 18 sequences, frames 8 to 43 are encoded, and the script logs:

- **Psi-treillis** (`psi_stream pack`), in degree 3 at a lattice step of 2 voxels and of 1 voxel, without a bound and
  with a certified bound of 5 % and of 3 % of u' on the largest error of a component; and in degree 2 at a step of 2
  voxels, without a bound. The compression ratio is that of the group of the first 32 frames (one key frame, 31
  innovations). The errors are those that the encoder measured on every voxel and recorded in the frame table of the
  file. `pack` exits with code 1, and the bound is said to be refused, when the tolerance would ask to refine more
  than a quarter of the volume.
- **SPERR and SZ3, frame by frame**, on frames 8, 24 and 39, and **SZ3 in 4D** on the block of frames 8 to 39, each at
  the same absolute bound (bound x u', u' being the RMS velocity of the header of the sequence). They are called
  through `benchmarks/codecs/comparison_codecs.py`, with no option tuned other than the bound. Their errors are
  measured by the script: the decoded arrays are compared with the original in NumPy.

Only sizes, errors and exit codes are logged: no timing.

These measurements decide five claims of `docs/CLAIMS.md`, whose criteria were written before the first compression of
a sequence of the sweep:

| Claim | What it bears on |
|---|---|
| VALID-1 | A threshold of resolution above which the bound of 5 % is held and below which it is refused |
| VALID-2 | Where the bound of 5 % is held, the compression ratio of Psi-treillis against that of SPERR |
| VALID-3 | At a lattice step of 1 voxel, the bound of 5 % on all the sequences, and the compression ratio against SPERR |
| VALID-4 | The field error as a function of the resolution |
| VALID-5 | The spread of the compression ratio and of the error between the three realisations of a viscosity |

`identity` is a check of the data, not of the codec: it compares the SHA-256 fingerprints of frames 0, 32 and 47 of
the sequence of the sweep at viscosity 0.0028 and seed 42 with those of `data/turb256.seq`, which was produced with the
same viscosity and seed but with 128 frames. Asking the solver for 48 frames instead of 128 must change no frame.

## How to replay

From the root of the repository, on Windows with a CUDA GPU:

```powershell
.\tools\build.ps1 -Tools                           # build\dns.exe, build\psi_stream.exe
.\benchmarks\codecs\build_codecs.ps1 -Only sz3,sperr   # build\codecs\
python benchmarks\validity\validity.py simulate    # the sequences of data\sweep\ that are missing (GPU)
python benchmarks\validity\validity.py measure     # appends to validity.csv what is missing (GPU)
python benchmarks\validity\validity.py summary     # prints the table and writes summary.txt
python benchmarks\validity\validity.py identity    # needs data\turb256.seq; writes identity_turb256.txt
python tools\figures.py                            # docs\figures\validity_light.png and validity_dark.png
```

`simulate` skips a sequence that is already on disk with the expected size. `measure` skips every line that
`validity.csv` already holds, so an interrupted campaign resumes where it stopped; delete `validity.csv` to measure
everything again. The 18 sequences are 9.0 GiB each and are not versioned.

## What the script writes

| File | Content |
|---|---|
| `validity.csv` | One line per sequence, codec and bound |
| `logs/dns_<sequence>.log` | Output of the solver for each sequence, after the command that produced it |
| `logs/pack_<sequence>_<codec>_<bound>.log` | Output of each run of `psi_stream pack`, after its command; `none` stands for no bound |
| `summary.txt` | For each viscosity and each codec and bound: exit codes, and the range over the realisations of the compression ratio, of the RMS error and the largest error |
| `identity_turb256.txt` | The fingerprints compared by `identity`, and its verdict |

Columns of `validity.csv`:

| Column | Meaning |
|---|---|
| `sequence` | Name of the sequence: `nu<viscosity x 10^5, 5 digits>_seed<solver seed>` |
| `nu`, `solver_seed` | Viscosity and seed given to the solver |
| `kmax_eta`, `re_lambda`, `u_rms` | From the header of the sequence: resolution of the smallest scale, Taylor Reynolds number, RMS of one velocity component |
| `eta_voxels` | Kolmogorov scale in voxels, from `kmax_eta` (the solver truncates at kmax = n / 3) |
| `frame_interval_over_tau_eta` | Interval between two frames, in Kolmogorov times |
| `codec` | `psi_d3_mf2`, `psi_d2_mf2`, `psi_d3_mf1` (degree, then lattice step in voxels), `sperr`, `sz3`, `sz3_4d_32` |
| `bound_over_urms` | Bound on the largest error as a fraction of u'; empty without a bound |
| `exit_code` | Exit code of `psi_stream pack` (0; 1 for a failure, the refusal of a bound among them; 3 if coefficients were clipped; 4 if the bound was missed); 0 for the other codecs |
| `group_compression_ratio` | Psi-treillis: raw size of 32 frames over the bytes of the first 32 frames of the file. SZ3 in 4D: raw size over compressed size of the block of 32 frames. SPERR, SZ3: harmonic mean of the ratios of the three frames |
| `file_compression_ratio` | Psi-treillis only: raw size of the 36 frames over the size of the file, header and frame table included, without a seal |
| `rms_error_relative` | RMS of the error over the RMS of the field. Psi-treillis: mean of the values of the 36 frames. SPERR, SZ3: mean of the three frames. SZ3 in 4D: over the whole block |
| `largest_error_over_urms` | Largest error of a component at voxel centres, over u': the maximum over the frames measured |
| `note` | Psi-treillis: size of the first key frame and mean size of an innovation, or the refusal message of the encoder. Others: the frames measured |

The `.psi` files are written in `build/work/validity/` and deleted once read.

## Caveats

- One solver, one family of flows, one grid size. The simulations truncate their spectrum at n / 3.
- The realisations of one viscosity do not have the same resolution: the Kolmogorov scale of the header depends on the
  dissipation of each run. The spread that VALID-5 measures is therefore partly a spread of resolution.
- The frame interval differs between sequences (it is set by the CFL condition of each run), and the temporal
  prediction of Psi-treillis depends on it.
- Psi-treillis and SZ3 in 4D are measured on a group of 32 frames; SPERR and SZ3 frame by frame on three frames.
- The errors of Psi-treillis are those recorded by its encoder, over the 36 frames encoded, whereas its compression
  ratio is that of the first 32; the errors of the other codecs are measured by the script on the frames they
  compress.
- The figures of the frame-by-frame codecs are rounded per frame (three decimals for a ratio, seven significant digits
  for an RMS error) before they are averaged.
- No timing is logged here; the comparison compressors are called as external tools.
