# Second reader, in Python: is the specification sufficient on its own? (claim REF-2)

## What is measured and why

`tools/reference_reader.py` is a reader of `.psi` files in Python 3 and NumPy, without a GPU. It was written from
`docs/FORMAT.md` alone, by a working session that opened neither the sources of the codec, nor the first reference
reader (`tools/reference_reader.cpp`), nor its logs (`../cpp/`). The first reader had needed trials on files to
settle what the specification of the time did not give; the specification was then completed. The second reader says
whether the completed text can be read without the code. The criterion of REF-2 (`docs/CLAIMS.md`) has three clauses:
the reader gives back the fingerprint of each frame of the files below; it gives back the field of their probe to
less than the threshold of the claim; and the list of the passages where the text was not sufficient, kept during the
writing, is empty for everything that concerns the state and the field.

**The rule followed during the writing.** The reader was written by reading the text, section by section. Each time
the text did not say what to do or allowed two readings, the doubt was noted before any trial: the passage, the
possible readings, the one adopted and why. `writing_journal.md` is that journal, in the order in which it was kept;
its first part, the six doubts, was committed with the reader before its first run on a file (commit `7cc43a1` of the
development history, which is not public, and in which the reader still has its former file name). Deciding between readings by comparing with a file would have
counted as a gap in the text.

**The files of the criterion** (`reference_python.py`, groups `reference_256` and `test_64`). The five reference
files at 256³ (`f_t256_d3_t05`, `_t04`, `_t03`, `f_t256_d2_t06`, `_t05`) and every valid test file at 64³ (the
`d2_*.psi` and `d3_*.psi` that the test suite leaves in `build/tests/cases/`). For each file:

1. it is copied to the working folder; the originals are never opened for writing;
2. it is encoded again by `psi_stream pack --probe`, run as a black box with the settings read in its header, to
   obtain the probe of the encoder (`docs/FORMAT.md`, section 11). The probe stands for the file only if the
   re-encoding has the SHA-256 of the file. The header does not carry the cap on the passes of the fine level: when
   the re-encoding differs, the script tries `--fine-passes` from 1 to 8. If nothing gives back the file, the
   re-encoded file is read back and probed in its place, and the log says so (columns `probe_origin` and
   `probe_read_on`);
3. the reader decodes every frame and compares its fingerprint with that of the frame table;
4. at the frame of the probe, the sampling lattice that the reader synthesises, modulo 2^16, and its coefficients of
   the fine level are compared node for node with those that the encoder recorded, and the field, evaluated in double
   precision by the formulas of section 6.7, with the velocities of the probe.

```
psi_stream pack --data SEQUENCE --file W/reencoded/F.psi --probe W/probes/F.probe OPTIONS
```

**Complementary files, outside the criterion** (`complements.csv`). The files of the criterion exercise neither the
empty segments (doubt D3), nor the raw keys in `PSISTRM3` (section 5), nor the reading by differences (doubt D2). Six
files are produced for that by `pack --start 3 --frames 36` on the test sequence at 64³, and read back in the same
way. Each of these trials was noted in the journal before being run, the reading already adopted being kept.

| File | Options of `pack` | What it puts to the test |
|---|---|---|
| `c1_d2_no_brick`, `c1_d3_no_brick` | `--tolerance 0.5` | frames without any brick: two empty segments in the section of the fine level, whose frequency tables the reader checks like the others |
| `c2_d2_bound_raw_keys`, `c2_d3_bound_raw_keys` | `--tolerance 0.05` or `0.04`, `--raw-keys` | raw keys in a `PSISTRM3` file: the rule of rounding to 8 bytes of section 5 |
| `c3_d2_eps0005`, `c3_d3_eps0005` | `--eps 0.0005` | clippings, and a lattice that wraps, hence read by differences |

`pack` clips on the two `c3` files, writes them and reports it by its exit code: these are degraded files, taken here
because wrapping the lattice requires it. For a file read by differences the script also measures the range of the
differences inside the support of each probe point, and what the field would be with a forced direct reading or when
subtracting the last node of the support instead of the first: this says whether the file distinguishes the readings
of D2.

**Altered files at 64³** (`coverage_64.csv`). Every `guard_*.psi` file of the test suite is given to the reader, then
to `psi_stream play --file F.psi --particles 4096` run as a black box. The reader either refuses the file with a
message, or reads it in full and counts the frames whose fingerprint differs; `play` is recorded as `read`, with its
number of differing frames, when it prints its `Integrity` line, as `refused` otherwise. The two agree on a file when
they have the same outcome and the same number of differing frames.

**The list of the passages noted during the writing** (`gaps.csv`, kept by hand). One line per doubt, with what the
text says, the reading adopted, what it affects, how it was put to the test, and whether it counts.

| | Section | Passage | Reading adopted |
|---|---|---|---|
| D1 | 6.2 | The prediction formula does not say on how many bits the sum is computed, and nothing bounds the predictors. | Exact sum. |
| D2 | 6.5, 6.7 | The "first node of the support" is not named; the threshold of the proximity bit is not said to be strict or not. | The node of smallest indices, first in the order given in 6.7; subtracted everywhere. |
| D3 | 6.1, 6.6 | Does the table of an empty segment sum to 4096? | The letter of the text: yes, all of them. |
| D4 | 6.3 | Fate of a malformed exception in an altered file. | Entry ignored, or value truncated to the type of the state. |
| D5 | 10, point 7 | Innovation of type 1 just after a key. | Content of the slot; refusal at position 1. |
| D6 | 9 | "Under the maxima": less than or equal, or strictly less? | Less than or equal. |

The counting rule: a passage counts if, on a file that the checks of the format accept, two readers conforming to the
text can return a different state or field without the text saying which is right; or if the reading adopted had
proved wrong; or if a file had been needed to choose. The column `counts` of `gaps.csv` applies it, strictly: a
silence of the text counts even when it has no effect on the files read. `tools/claims.py` reads this column; one
passage that counts is enough for the last clause of the criterion to fail. The column `prediction_margin_32_bits` of
`reference_python.csv` gives, for D1, how far the predictors of each file are from the case where an exact sum and a
sum on 32 bits would differ, and the reader counts the sums that leave 32 bits.

### What the session read, and what it did not read

Read:

- `docs/FORMAT.md`, in full: the only source for the format. Its `file:line` references were not followed.
- The claims sheet (its first hundred lines) and the script that renders it, the first entries of the changelog, the
  notice of the measurements, `.gitignore`.
- The script that encoded the reference files and the first lines of its logs, for the encoding options.
- The help of `psi_stream pack` and the outputs of `pack` and of `play`.
- The `.psi` files and their probes, through the reader only.

Not opened, nor read, nor searched: the sources of the codec, first reader included; the prototypes; the folder of
the first reader (`../cpp/`); the Python module of the library; the test suite; the notes left by other working
sessions, including that of the first reader. The description of the interface of the library and the main notice of
the repository were allowed, and were not needed.

Known in advance: what `docs/FORMAT.md`, the changelog and the claims sheet say of the first reader (its differences,
the list of what had had to be established by trial), and the purpose and the organisation of the project, without
any detail of the format.

## How to replay

From the root of the repository, with `build/psi_stream` built (`tools/build.ps1 -Tools` on Windows, CMake with
`-DPSIT_BUILD_TOOLS=ON` elsewhere), `data/turb256.seq` in place, Python 3.9 or later and NumPy:

```
# the reference files, then the test files at 64^3 and their sequence
python benchmarks/compression/reference_files/reference_files.py
python tests/test_codec.py

python benchmarks/independent_readers/python/reference_python.py
python tools/claims.py
```

`reference_python.py` takes `--files FOLDER` (the reference files, `build/work/reference_files` by default),
`--cases FOLDER` (the test cases, `build/tests/cases` by default; the test sequence `turb64.seq` is taken in the
parent folder), `--no-gpu` and `--output FOLDER`. With `--no-gpu` it runs neither `pack` nor `play` and reads
everything back with the probes, the re-encodings and the outcomes of `play` that an earlier run left in the working
folder: the reading itself never needs the GPU. No time is read: the GPU may be shared.

For a single file:

```
python tools/reference_reader.py F.psi --probe F.probe
```

## What the script writes

| Path | Content |
|---|---|
| `reference_python.csv` | one line per file of the criterion, the columns below |
| `complements.csv` | one line per complementary file: `purpose`, `pack_options`, `pack_exit_code`, the columns of `reference_python.csv` except those of the group and of the re-encoding, then `largest_range_in_a_support`, `points_with_support_range_2_15_or_more`, and for a file read by differences `field_difference_direct_read_over_u_rms` and `field_difference_last_node_subtracted_over_u_rms` |
| `coverage_64.csv` | one line per altered file: `file`, `python_outcome` (`read`, `refused` or `PYTHON ERROR`), `python_frames_differing`, `gpu_outcome`, `gpu_frames_differing`, `play_exit_code`, `agreement` (`yes` or `NO`), `python_reason` (the message of the refusal) |
| `logs/reference_python.log` | what the script printed: SHA-256 of each file, options and exit code of `pack`, what the reader found, summary |
| `logs/pack_<file>.log` | raw output of every `pack` run for a file, each under a line `== command: exit code` |
| `logs/coverage_64.log` | raw output of every `play`, each under a line `== command: exit code` |
| `build/work/independent_readers/python/` | copies of the files read, re-encodings, probes, complementary files, and the options of `pack` and outcomes of `play` that `--no-gpu` takes again (not versioned) |

Kept by hand, not written by the script:

| Path | Content |
|---|---|
| `gaps.csv` | the list of the passages noted during the writing: `id`, `section`, `passage`, `what_the_text_says`, `reading_adopted`, `what_it_affects`, `put_to_the_test`, `counts` (`yes` or `no`); read by `tools/claims.py` |
| `writing_journal.md` | the journal kept during the writing, in order |

Both are translations of what was kept in French during the writing; the names of files and of columns quoted in
them are the present ones.

Columns of `reference_python.csv`:

| Column | Meaning |
|---|---|
| `file`, `group` | name of the file; `reference_256` or `test_64` |
| `version`, `n`, `degree`, `seal` | signature of the file, grid, degree; `absent` or `verified` |
| `frames`, `frames_read`, `fingerprints_differing` | frames of the file, frames decoded, frames whose fingerprint is not that of the frame table |
| `raw_keys`, `coded_keys`, `escapes`, `overflows` | what the frames read contain |
| `frames_without_brick` | frames whose fine level has no brick (`PSISTRM3` only) |
| `wrapped_frames`, `lattice_read` | wrapped frames according to the header; `direct` or `by differences` |
| `prediction_margin_32_bits` | 2^31 divided by the largest possible modulus of the sum of the prediction, computed on the predictors of the header: above 1, the sum fits in 32 bits whatever the states (doubt D1) |
| `probe_origin`, `probe_read_on` | `identical re-encoding` and `file`, or `different re-encoding` and `re-encoding` |
| `probe_frame_position`, `probe_points` | position in the file of the probed frame, number of points |
| `lattice_nodes_differing`, `fine_nodes_differing` | nodes that differ from those of the encoder |
| `field_difference_over_u_rms` | largest difference between the field of the reader and that of the probe, as a fraction of u′ |
| `scale_difference` | largest relative difference between the scales computed by the reader and those of the probe |
| `reencoding_frames_read`, `reencoding_fingerprints_differing` | only when the probe was read on the re-encoding |

Exit code: 0 if the log is complete; 1 if a file or a probe is missing, or if the reader raised an error on a file
(the last line of `logs/reference_python.log` counts these anomalies); 2 if a prerequisite is missing and nothing was
run. The script refuses to write `logs/reference_python.log` if an absolute path has entered it.

## Caveats

- **What the independence covers.** This reader tests the text as it was completed from the first reader and then
  cross-checked against the code. It says whether this text can be read without the code; it does not rediscover the
  conventions.
- **The reader after its first run.** Its decoding has not been changed since the commit that precedes its first
  run. Two touch-ups of its command line followed, then its translation into English: identifiers, options, messages.
  The logs of this folder are those of the translated reader.
- **The list of the passages is kept by hand**, by the session that wrote the reader, and the rule that says which
  passages count was applied by that same session. The strict reading was adopted because the criterion, fixed before
  the measurement, speaks of everything that concerns the state: whoever holds that a silence with no effect on the
  files read does not count reads the last clause otherwise.
- **Which arithmetic is authoritative for the prediction** (D1) is not established: a reading of the code would tell;
  it was not done here. It is not known either whether `pack` can write predictors that large.
- **The field outside one frame per file.** The fingerprint covers the state of each frame; the field is compared
  only at the frame of the probe.
- **The GPU reader.** The field is compared with the probe of the encoder, not with what `play` returns. The GPU
  reader is compared only through the outcome of the altered files, and the script calls `refused` every run of
  `play` that prints no `Integrity` line: the column `play_exit_code` tells a refusal from another end.
- **Where the difference between the reader and the probe comes from** is not established. Section 10 of the
  specification attributes it to the single precision of the GPU reader, as a hypothesis; section 6.7 describes the
  order of the operations in single precision, and this reader does not follow it.
- **The geometries that these files do not have.** When the reader was written, all the files had three levels,
  `mf` = 2 and `fb` = 2: a single level, six levels, `mf` = 1 or 4 and another `fb` were not tested, nor a coded key
  turned back into a raw key by the encoder. The options of `pack` that `logs/reference_python.log` prints for each
  file show `--levels` or `--mf` when a file departs from these values.
- **512³.** Nothing is read at this size.
- **`PSISTRM1`.** The reader refuses it: it does not support it.
- **The altered files** are those of the test suite. This is not a proof for every alteration, and D4 is not tested:
  it is not known whether any of these files carries a malformed exception.
- **Another machine, other data.** Same machine and same family of data as the other measurements.
- **What the text would gain by saying**, reported by the session and not corrected by it: in 6.2, the width of the
  sum of the prediction, or a limit on the predictors in section 9; in 6.5, which node is "the first node of the
  support"; in 6.3, what the reader does with a malformed exception; in 6.1 or 6.6, that the table of an empty
  segment also sums to 4096.
