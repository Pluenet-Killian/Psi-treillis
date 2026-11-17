# Reference reader on CPU, in C++ (claim REF-1)

## What is measured and why

`tools/reference_reader.cpp` (executable `psi_ref`) is a reader of `.psi` files in C++17, with neither CUDA nor any
library. It was written from `docs/FORMAT.md` by a working session that opened no source file of the codec. It puts
two things to the test at once: the specification (is it enough to write a reader?) and the GPU reader (does a second
reader, written without its code, find the same state and the same field?).

**The twelve reference files** (`reference_cpp.py`, first part; criterion of REF-1 in `docs/CLAIMS.md`).

- Step 1, the fingerprints. `psi_ref` decodes every frame of each file and recomputes its state fingerprint
  (`docs/FORMAT.md`, section 7), which is compared with that of the frame table of the file. This step needs neither a
  GPU nor a probe.
- Step 2, the field at the probe. At the frame of the probe, `psi_ref` rebuilds the sampling lattice and the fine
  level and compares them, node for node, with those that `psi_stream pack --probe` recorded. It then computes the
  velocity at the points of the probe, in double precision, and gives its largest difference from the velocities
  recorded by the encoder, as a fraction of u′. `psi_stream play --probe` gives, for its part, the largest difference
  between the GPU reader and this same probe. The sum of the two differences bounds the difference between the CPU
  reader and the GPU reader: it is the figure that the criterion of REF-1 compares with its threshold.

```
psi_ref F.psi --csv --probe F.probe
psi_stream play --file F.psi --particles 4096 --probe F.probe
```

The ten reference files with a bound have their probe beside them, written at the encoding. The two files without a
bound (`z_t512_d2`, `z_t512_d3`) have none: the script encodes them again with the command that produced them plus
`--probe`, and accepts the probe only if the re-encoded file has the SHA-256 of the reference file.

**Coverage at 64³** (`reference_cpp.py`, second part). The reference files have neither a seal nor a raw key. The
files that the test suite leaves in `build/tests/cases/` cover the rest: valid files (sealed or not, with raw keys,
with exception lists, with and without a fine level) and the `guard_*` files (altered, truncated, or left by an
interrupted encoding). Every one of them goes through `psi_ref`, with its probe when it has one, and through
`psi_stream play --file F.psi --particles 4096`. The outcome of each reader is one of:

| Outcome | Definition |
|---|---|
| `read` | exit code 0, and the line that reports the frames (`CSV,ref` for `psi_ref`, `Integrity` with no differing frame for `play`) |
| `counted` | exit code 1 and the same line, with at least one frame whose fingerprint (`psi_ref`) or state (`play`) differs |
| `refused` | exit code 1 and a message `Failure` |
| `anomaly (code N)` | anything else: a crash, or an exit code without the line that goes with it |

The two readers agree on a file when they have the same outcome and the same number of differing frames.

**The conventions established by trial** (`conventions.py`). When `psi_ref` was written, the specification did not
give everything that the velocity field needs (list below). Three trials decide between the possible readings by
comparing each with what an existing program produced, on four test files at 64³ and their probes:

1. refinement of the pyramid (section 6.5): alignment of the masks and rounding of the division, against the sampling
   lattice recorded in the probe; the state comes from `psi_ref --state`;
2. position of the nodes of the lattice and of the fine level (section 6.7), against the velocity field of the probe;
3. position of the centre of a voxel (section 6.7), against `psit sample`, which takes its points in grid indices.

For each reading tried, `conventions.txt` gives the number of nodes that differ (trial 1) or the largest velocity
difference in u′ (trials 2 and 3): the reading adopted is the one that gives back the lattice and the field, the
others do not.

### How the reader was written

- Read by the session: `docs/FORMAT.md`; the claims sheet and the script that renders it; the notice of the test
  suite; the user guide and the summary of results of the time, which are no longer in the repository; the script
  that encoded the reference files and its logs, for the commands; the build script, for the call of the compiler;
  the `.psi` and probe files; the help and the outputs of `psi_stream`.
- Not read: no source file of the codec, no prototype, and not the script of the test suite. The `file:line`
  references of `docs/FORMAT.md` were not followed.
- Order of events, as observed during the session (it is recorded in no log): step 1 was written from the
  specification alone, and the first build reproduced the fingerprints of test files at 64³, then those of the twelve
  reference files, with no touch-up of the decoding. Step 2 required trials against the probe files at 64³ (below);
  the files at 256³ and 512³ served only to verify, once the conventions were fixed.
- Since then the source has been translated into English: identifiers, options and output labels. Its decoding was
  not changed.

### What the specification did not give

This list describes `docs/FORMAT.md` as it was when `psi_ref` was written. It is the list that the comments
"OUTSIDE THE SPECIFICATION" of `tools/reference_reader.cpp` point to. "Deduction": resolved by reasoning, then
confirmed by the fingerprints or the probe. "Trial": several possible readings, decided between by comparing with a
file. Nothing was resolved by reading the code. What was resolved has since been carried into the specification,
under the marks "clarified" and "established by trial" (sections 5, 6.2 to 6.7, 7 and 11), then cross-checked against
the code by another reading; what was not is in its section 10. The second reader (`../python/`) was written from the
specification thus completed.

Blocking for the velocity field (step 2):

| Section | What was missing | How it was resolved |
|---|---|---|
| 6.5 | The "integer masks of the B-splines" of the refinement were not given, nor their alignment, nor the division, nor its rounding. | Trial, against the lattice of the probe at 64³ (`conventions.py`, trial 1). Per axis, degree 2: `fine[2i] = 3 g[i] + g[i-1]`, `fine[2i+1] = 3 g[i] + g[i+1]`; degree 3: `fine[2i] = g[i-1] + 6 g[i] + g[i+1]`, `fine[2i+1] = 4 g[i] + 4 g[i+1]`; tensor product over the three axes, then a single division, `(sum + 32) >> 6` in degree 2 and `(sum + 256) >> 9` in degree 3. The limits of section 9 hinted at the gains of the masks, not at the rounding. |
| 2, 6.5, 6.6 | No formula gave the velocity at a point: neither the basis (cardinal B-spline centred on its node), nor the position of the lattice nodes, nor the sign of the curl, nor the addition of the mean velocity. Section 6.6 located only the nodes of the fine level. | Deduction for the basis, the curl and the scale `delta / (2^fb * h0)` of section 2; trial for the position (`conventions.py`, trial 2): node `i` of the lattice is at `(i + 1/2) h0` in degree 2 and at `i h0` in degree 3, like those of the fine level. |
| 6.6 | The degree of the B-spline of the fine level was not stated. | Deduction (that of the header), confirmed by the probe. |
| none | The probe file of `pack --probe` and `play --probe` was not described. | Black box, from the size and the content of probe files at 64³, 256³ and 512³. The layout is now section 11 of the specification. |
| none | The unit of the `Probe` line of `play` was not in the specification. | Trial, by hand: on a test file whose overflows are altered, which the fingerprint does not see, the figure of `play` is that of `psi_ref` multiplied by u′. The line is in velocity units, and the script divides it by u′. |
| 6.5 | The arithmetic of the GPU reader (single precision, order of operations) was not described: its field cannot be reproduced bit for bit. | Not resolved when `psi_ref` was written; section 6.7 now describes it, and `psi_ref` does not follow it (see the caveats). |
| 6.7 | The position of the centre of a voxel, for a point given in grid indices. | Trial, made when the conventions were carried into the specification (`conventions.py`, trial 3): the centre of voxel `i` is at `(i + 1/2)` voxel. |

Resolved by deduction for the state of the frames (step 1). Each has a single reasonable reading, and the
fingerprints confirm it; a reader written by someone else could nevertheless get it wrong.

| Section | What was missing |
|---|---|
| 7 | `a[i]` in `mix64(s + 256 * i + a[i])`: unsigned byte, from 0 to 255. With a signed byte the fingerprint changes. |
| 6.4 | The folding was given only by the example "0, -1, 1, -2 become 0, 1, 2, 3". Formula adopted: `2 r` if `r >= 0`, `-2 r - 1` otherwise, on 16 bits. |
| 6.4 | It was not said that the Lorenzo residual is computed component by component, the neighbours being the neighbouring nodes and not the neighbouring values of the array. |
| 5, 6.4 | The order of the symbols inside a segment of a coded key was not stated: it is `3 * node + component`, as for the state. |
| 6.2 | `>> 10` on a negative value: arithmetic shift, rounding towards minus infinity. |
| 6.2 | Is the innovation the symbol read as a signed byte? Yes (the text said "a signed byte" for the innovation, not for the decoded symbol). |
| 6.3 | An overflow is added to the lattice before the reduction modulo 2^16, at position `idx - seg[0]` of level 0. |
| 5 | In `PSISTRM3`, the stream area of the base part stops at the start of the section of the fine level, which is known only from the end of the frame: `FineTrailer` must be read before checking the offsets of the base. |

Not resolved, with no effect on the files read:

| Section | What was missing |
|---|---|
| 9 | "`FrameRecord.bytes` at most equal to the worst case of the coder": the formula of the worst case was not given. `psi_ref` bounds a frame by the frame area of the file, no further. |
| 6.2 | Value of `q(k-1)` for an innovation of type 1 placed just after a key (`pack` writes none). `psi_ref` takes `q(k)`. |
| 6.1 | What an rANS block reads beyond the end of its frame in an altered file: the specification said "the remainder of a previous frame", which depends on the buffer of the GPU reader. `psi_ref` reads zeros. The coverage at 64³ says whether the two readers nevertheless count the same differing frames. |

## How to replay

From the root of the repository, with `build/psi_ref`, `build/psi_stream` and `build/psit` built
(`tools/build.ps1 -Tools` on Windows, CMake with `-DPSIT_BUILD_TOOLS=ON` elsewhere; the scripts compile nothing):

```
# the twelve reference files and their probes, then the test files at 64^3
python benchmarks/compression/reference_files/reference_files.py
python tests/test_codec.py

python benchmarks/independent_readers/cpp/reference_cpp.py
python benchmarks/independent_readers/cpp/conventions.py
python tools/claims.py
```

`reference_cpp.py` takes `--files FOLDER` (the reference files, `build/work/reference_files` by default),
`--cases FOLDER` (the test cases, `build/tests/cases` by default), `--no-gpu` and `--output FOLDER`. With `--no-gpu`
it does only what needs no GPU: the fingerprints of all the frames, and the comparison of `psi_ref` with the probes
already present; `psi_stream` is not run, and the columns of the GPU reader stay empty. It needs `data/turb512.seq`
for the probes of the two files without a bound. `conventions.py` takes `--cases FOLDER`, `--no-gpu` (trials 1 and 2
only) and `--output FOLDER`; it needs NumPy and stops if one of its four test files or their probes is missing.
Neither script reads a time: the GPU may be shared. `tools/claims.py` renders the verdict of REF-1 from
`reference_cpp.csv`.

## What the scripts write

| Path | Content |
|---|---|
| `reference_cpp.csv` | one line per reference file, the columns below |
| `coverage_64.csv` | one line per test file at 64³: `file`, `bytes`, `cpu_outcome`, `cpu_frames_differing`, `gpu_outcome`, `gpu_frames_differing`, `agreement` (`yes` or `no`), and with a probe `cpu_probe_difference_over_u_rms`, `lattice_nodes_differing`, `fine_nodes_differing` |
| `logs/reference_cpp.log` | what the script printed: SHA-256 of the programs, state of the GPU at the start and at the end, output of `psi_ref` for each reference file, lines `Probe`, `Integrity` and `Failure` of `play`, summary of the coverage |
| `logs/psi_ref_<file>.log`, `logs/play_<file>.log` | raw output of `psi_ref` and of `play` on each reference file |
| `logs/pack_<file>.log` | raw output of the re-encoding of a file without a bound |
| `logs/coverage_64.log` | raw output of every run of the second part, each under a line `== program file: exit code, outcome` |
| `conventions.txt` | the figures of the three trials, one line per file and per reading |
| `logs/conventions_state_<file>.log`, `logs/conventions_psit_<file>.log`, `logs/conventions_psi_ref_<file>_c<c>.log` | raw output of the programs run by `conventions.py` |
| `build/work/independent_readers/` | probes of the two files without a bound, with the SHA-256 of the file each stands for; working files of `conventions.py` (not versioned) |

Columns of `reference_cpp.csv`:

| Column | Meaning |
|---|---|
| `file` | name of the reference file |
| `n`, `degree`, `version` | grid, degree, and 2 or 3 for `PSISTRM2` or `PSISTRM3` |
| `frames`, `frames_read` | frames of the file, frames decoded by `psi_ref` |
| `fingerprints_differing` | frames whose recomputed fingerprint is not that of the frame table |
| `escapes`, `overflows` | escapes and overflows of level 0 met over the frames read |
| `probe_frame`, `probe_points` | frame of the probe, counted in the source sequence, and number of points |
| `lattice_nodes_differing`, `fine_nodes_differing` | nodes of the sampling lattice and of the fine level that differ from those of the encoder |
| `cpu_probe_difference_over_u_rms` | largest difference between the field of `psi_ref` and that of the probe, as a fraction of u′ |
| `gpu_probe_difference_over_u_rms` | largest difference between the field of `play` and that of the probe: its `Probe` line, divided by u′ |
| `cpu_gpu_bound_over_u_rms` | sum of the two: bound on the difference between the CPU reader and the GPU reader |
| `gpu_frames_differing` | frames whose state differs for `play` (its `Integrity` line) |
| `cpu_exit_code` | exit code of `psi_ref` |
| `probe_origin` | `written at encoding`, `identical re-encoding (SHA-256)`, or `absent` with the reason |

A difference left empty was not measured; a difference of zero is a measurement. In the raw logs, standard output
comes first, then standard error. The lines that the scripts read are quoted in them with the source line that prints
each; those of `play` are in `benchmarks/compression/psi_stream_runs.py`.

Exit codes. `reference_cpp.py`: 0 if the twelve files are read by `psi_ref` with exit code 0 and every line that a
figure comes from was found; 1 if a reference file is missing, if `psi_ref` ends otherwise on one of them, if such a
line is missing, or if the second part could not be done; 2 if `psi_ref` is missing. An outcome of the second part,
whatever it is, is a result: it is recorded. `conventions.py`: 0 if every trial asked for is written, 1 if the third
is missing or incomplete, 2 if `psi_ref` or a test file is missing.

## Caveats

- **The statement of REF-1 holds in full only for the state of the frames**, decoded from the specification alone.
  For the velocity field, the specification of the time was not sufficient: the conventions listed above were
  established by trial on files, without reading the code.
- **The independence of the reader from the encoder.** The conventions of step 2 were fitted on probes written by
  `pack`. A defect shared by `pack` and by `play` on these conventions would not be seen.
- **The GPU reader itself.** Its velocities are not observed: `play` gives only their largest difference from the
  probe. The comparison therefore bears on the field recorded by the encoder.
- **The field outside the probed frame.** One frame per file. On the other frames only the fingerprint of the state
  is compared; the lattice and the field follow from it by the same computation, but are not compared.
- **The origin of the difference between `psi_ref` and the probe is not established.** The hypothesis is the rounding
  of the single precision of the GPU reader, which `psi_ref` does not reproduce since it computes in double precision
  and not in the order of operations of section 6.7. The margin under the threshold at a grid larger than those of the
  reference files is not known either.
- **The kernels that read by differences.** The path by differences of `psi_ref` is exercised only by a file that has
  a wrapped frame: the `Probe` line of `psi_ref`, in `logs/psi_ref_<file>.log`, says how the lattice of each file was
  read.
- **What the logs do not hold.** `PSISTRM1` files, which `psi_ref` reads, and alterations other than those of the
  test suite were tried by hand only when the reader was written; no script of this folder replays them.
- **Another machine.** The logs are those of one machine and one compiler. Nothing is tried on a big-endian machine.
- **The reader was translated after it was written** (identifiers, options, labels of its output). The logs of this
  folder are those of the translated source, whose SHA-256 is at the top of `logs/reference_cpp.log`.
