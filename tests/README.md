# Regression tests of the pack / play chain

A short suite, on turbulence at 64³, which checks `psi_stream pack` and `psi_stream play` through their bytes, their
fingerprints, their exit codes and their messages. It measures no timing.

## Running

From the root of the repository, with Python 3.8 or later (standard library only) and an NVIDIA GPU:

```powershell
python tests/test_codec.py                        # produces the sequence if it is missing, runs the 50 cases
python tests/test_codec.py --build build          # build folder holding dns and psi_stream (default: build)
python tests/test_codec.py --only "d3.*"          # a subset of the cases (wildcards * and ?); several patterns may follow: --only "guard.*" "*.bound_*"
python tests/test_codec.py --update               # regenerates tests/fingerprints.json, see below
python tests/test_codec.py --keep-alterations     # keeps the 240 altered files of guard.random_alteration
```

The script compiles nothing. It needs a build folder that holds `dns` and `psi_stream`: `.\tools\build.ps1 -Tools` on
Windows, CMake with `-DPSIT_BUILD_TOOLS=ON` elsewhere (`cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
-DPSIT_BUILD_TOOLS=ON`, then `cmake --build build`). A relative `--build` is taken from the root of the repository. If
the binaries are missing, the script says so and exits with code 2.

It produces `turb64.seq` in `build/tests/` (40 frames, `dns --n 64 --nu 0.02 --seed 42`, kmax·η = 1.68, Re_λ = 32),
then prints one line per case and a total. Exit code: 0 if no case fails, 1 otherwise, 2 if the prerequisites are
missing (binaries, sequence that `dns` could not produce, options that select no case). The files of the cases are
in `build/tests/cases/`, the output of each command in `build/tests/logs/`.

| Line | Meaning |
|---|---|
| `OK` | the case passes; `guard.random_alteration` adds the count of its outcomes to it |
| `FAIL` | the case fails; the reason and the log to read follow |
| `KNOWN` | the case fails on a known and uncorrected defect of the code; it does not count as a failure |
| `UPDATED` | with `--update`, the fingerprint recorded |

The last line is the total: `TOTAL: 50 cases, 49 OK, 0 FAILED, 1 known defect(s)` on 17 November 2026.

## What is covered

Each case exists in degree 2 (`d2.`) and in degree 3 (`d3.`), on frames 3 to 38 of the sequence: two key frames,
two innovations "after a key", 32 innovations.

| Case | What it requires |
|---|---|
| `pack_play` | `pack` and `play` exit with 0; expected header and frame types; "0 frame(s) whose state differs" |
| `determinism` | the same command twice gives the same file and the same probe, without and with a fine level |
| `bound_held` | with `--tolerance`, code 0, non-empty fine level, largest error of the table ≤ tolerance − margin / 2, read-back intact |
| `bound_missed` | tolerance below the floor of the fine level: code 4, lines "Passes", "Floor", "Bound", message "Failure" that names the floor, file written and readable again |
| `pass_cap` | `--fine-passes 1`: code 4, the cause named is the cap on the passes |
| `old_behaviour` | `--fine-passes 12 --tolerance-margin 0`: code 0, zero margin in the header, error ≤ tolerance |
| `probe` | `pack --probe` then `play --probe`: difference 0.000e+00, without and with a fine level |
| `exceptions` | with a finer step, innovation escapes and overflows of the fine level; read-back intact, probe at 0 |
| `raw_keys` | `--raw-keys`: keys of type 0 instead of 3, same states, same final positions at read-back |
| `seal` | `pack` seals by default; `--no-seal` gives, byte for byte, the sealed file stripped of its last 48 bytes; both read back with the same final positions; `play` says which one carries a seal and `--require-seal` refuses the other. Without and with a fine level |
| `fingerprint.*` | SHA-256 of seven `.psi` files and fingerprint of the final positions, compared with `fingerprints.json` |

The files `old`, `no_seal` and `bound_no_seal` are written with `--no-seal`: their fingerprint is that of the
files from before the seal. Those of `no_seal` and `bound_no_seal` are equal to the fingerprints that `base` and
`bound` had in the reference file at commit `240736a` of the development history (not public), before the seal (verified on 15 November 2026, see
`benchmarks/robustness/README.md`).

Not tied to a degree: `sequence.fingerprint` (SHA-256 of the sequence) and the guards, which require a refusal with
code 1 and a message, without a crash or a hang (each command has a timeout):

| Case | Situation |
|---|---|
| `guard.fb7_degree3` | `pack --degree 3 --fb 7`: refused, no file written |
| `guard.clipping_fb7_degree2` | `pack --degree 2 --fb 7`: code 3 (lattice clipped), file read back by differences |
| `guard.fine_level_capacity` | `--tolerance 0.01`: capacity of the fine level exceeded; the file left behind, without a table, is refused by `play` |
| `guard.truncated_file` | `play` on a file reduced to 0, 100, 208, 272 bytes, or shortened by 1, 20, 49 or 96 bytes (in the middle of the seal, or of the table) |
| `guard.corrupt_magic`, `guard.invalid_degree` | magic altered; degree of the header at 0 or 7 |
| `guard.invalid_sequence` | `pack --data` on a file that is not a sequence |
| `guard.corrupt_frame` | one bit flipped in frame 5: code 1 and 27 frames whose state differs (5 to 31), key 32 resynchronises |
| `guard.swapped_degree` | degree of the header changed from 2 to 3 in a sealed file: refused, the message names the seal and the header |
| `guard.sealed_header` | 22 alterations of a sealed file: `n`, `mf`, `levels`, `fb`, `n_frames`, `delta`, `dt_frame`, `coef`, `index_offset`, count of the wrapped frames; `delta`, maxima and margin of the header of the fine level; mean velocity, fingerprint and type of a frame in the table; the seal itself. Each one is refused, with the expected message |

Two cases require, for each altered file, one of these outcomes: a refusal (code 1 and message "Failure:"), or a
read carried through to its end whose "Integrity" line counts the differing frames. Never a CUDA error, a
crash or a hang.

| Case | Situation |
|---|---|
| `guard.geometry_no_seal` | 42 alterations. 39 of them in files **without** a seal, where only the check of the frames against the geometry can hold: `n`, `mf`, `levels`; type, size and position of a frame in the table; number of exceptions, frequency table, offsets of the blocks, initial state of a block; maxima of the header of the fine level and end of frame of the fine level. The last three (`raw_levels2`, `raw_bytes`, `raw_exceptions`) alter the file with raw keys, which keeps its seal: the levels of the header and the size of a raw key in the table are refused by the seal, before any check of the geometry; only the exception count of a raw key reaches that check. Then `fb` changed from 2 to 5 in a file without a seal: nothing can see it, the file is read and the "Seal" line says that it is not protected |
| `guard.random_alteration` | 240 files: in four files (without and with a fine level, with and without a seal), one bit flipped or one byte replaced, at a position drawn in the header, the table (seal included), a frame, or the structure bytes of a frame. Fixed seed, generator written in the script. A third outcome is allowed: no differing frame **and** the final positions of the intact file, when the byte touched carries nothing (padding, reserved or informative field) |

On 15 November 2026, and again on 17 November, the 240 files give 100 refusals, 99 counts of differing frames, 37
reads without effect and 4 alterations not seen, all four in the header or the table of a file without a seal, where
nothing can see them. Read back by the earlier reader (commit `240736a`), the 43 files of `guard.geometry_no_seal`
gave 10 CUDA errors; the 240 random files gave none, but 26 alterations of the header or of the table of a
sealed file were read there without anything being reported. That comparison with an earlier reader is not replayed
in this repository; `benchmarks/robustness/` measures the current reader.

### Reference fingerprints

`fingerprints.json` is valid only for one toolchain: version of CUDA and of cuFFT, host compiler, architecture of
the GPU. The file describes it, and the script says at launch whether the current machine is the same (it reads the
compiler versions from what CMake recorded in the build folder). The sequence comes out of a chaotic simulation:
another FFT library or another GPU changes its bytes, and all the files encoded with it, without there being any
defect. On another machine: first verify that all the cases other than "fingerprint" pass (integrity is checked by
`play`), then run `--update`, which refuses to write if another case fails or if `--only` restricts the suite. On the
same toolchain, a fingerprint that changes is a regression of the encoder or an intended change of the format.

### Tolerances retained at 64³

Without a fine level, the largest error is 0.117 u' in degree 2 and 0.091 u' in degree 3. With the defaults of `pack`,
the bound is held down to 0.038 u' (degree 2) and 0.034 u' (degree 3). Lower, `pack` does not exit with code 4 but with
code 1: the tolerance requires refining more than a quarter of the bricks and the capacity guard stops
the encoding. At 64³, a tolerance that is too low therefore never gives code 4 on its own.

- Bound held: 0.05 (degree 2) and 0.04 (degree 3). The fine level retains 2.3 % and 3.7 % of the nodes there in 6 and
  4 passes, far from the capacity limit.
- Bound missed: the same tolerance with `--fine-eps 0.02`, the step that `pack` would choose for a tolerance of 0.17.
  The tolerance is then below the floor of the fine level: 36 frames out of 36 above the threshold, error of 0.079 u'
  (degree 2) and 0.089 u' (degree 3).
- Cap on the passes: the same tolerance with `--fine-passes 1`.

## Tests of the public interface (`test_api.py`)

A second suite checks what a developer calls: the library (`include/psitreillis.h`), the Python module
(`python/psitreillis`) and the `psit` tool. It runs on the same sequence at 64³, which it produces in
`build/tests/` if it is missing, and reads no timing. It needs NumPy.

```powershell
.\tools\build.ps1 -Tools         # library, psit, psi_stream and dns in build\
python tests/test_api.py         # 16 cases; exit code 0 if none fails
python tests/test_api.py --only "api.*"
```

| Case | What it requires |
|---|---|
| `cli.same_bytes_as_pack` | `psit compress` writes, byte for byte, the file of `psi_stream pack`, without and with a bound |
| `cli.info`, `api.info` | grid, frames, degree, seal, bound and key frames read back from the file |
| `cli.decompress_npy` | `psit decompress` writes a `.npy` that NumPy reads back, equal to what the library decodes |
| `api.errors_match_the_encoder` | the error of each decoded frame, measured here in NumPy, is the one that the encoder recorded |
| `api.bound_is_held` | with a tolerance of 4 %, every component of every voxel of every decoded frame is within the tolerance |
| `api.sample_matches_the_grid` | sampling at voxel centres returns the decoded grid; the coordinates are periodic |
| `api.divergence_free` | by centred differences at 50,000 off-grid points, the RMS divergence stays below 2·10⁻³ of the RMS gradient |
| `api.random_access` | a frame read directly is the frame read in sequence, across a key and backwards |
| `api.errors_are_reported` | missing, truncated or foreign file, frame outside the file, invalid grid: an error, the library remains usable |
| `api.altered_file_is_refused` | one bit flipped in a frame is seen when that frame is decoded |
| `py.compress_from_memory`, `py.u_rms_is_measured` | a NumPy array compresses like the `.seq` file; `u_rms` measured on the frames |
| `cli.npy_and_raw_inputs` | a `.npy` input and a raw input give the frames of the `.seq` input |
| `api.bound_too_tight` | a bound out of reach is refused with a message, and the same output path is then written again |
| `api.memory_is_released` | 40 openings, decodings and closings, then one compression, do not increase the memory occupied on the GPU |

On 16 November 2026, and again on 17 November: 16 cases, 16 passed (the logs kept are described in
`benchmarks/interface/README.md`). Not covered: the "bound missed" exit (code 4), which the public options do not
allow to be reached at 64³; Linux; the large sizes, except through
`benchmarks/interface/bounds_512/verify_bounds_512.py`.

## Known defects (`KNOWN` lines)

Observed on 15 November 2026, left as they are. Each case describes the expected behaviour and will pass once the
defect is corrected; `known` will then have to be removed from the case in the script.

- `guard.altered_overflow`: the value of an overflow of level 0 (`ExEntry.val`, bit 31 of `idx` set to 1) is altered
  in the last frame of a file that carries some. It touches only the sampling lattice, not the quantised
  state that the fingerprint of the frame covers. `play` exits with 0 and announces "0 frame(s) whose state differs",
  whereas the probe measures a difference of 4.78e-03 between the field of the reader and that of the encoder. Neither
  the seal (header and table) nor the check of the geometry covers the value of an exception.

Corrected the same evening, and moved to the ordinary cases:

- `guard.invalid_sequence`. `pack --data` on a file that is not a sequence stopped without a message, with the
  code 0xC0000409; `sequence_read_header` (src/sequence.h) threw an exception that nothing caught. It now refuses
  with a message and code 1.
- `guard.swapped_degree`. In a file of degree 2, the header modified to say 3 was not refused: `play` exited
  with 0, announced "0 frame(s) whose state differs" and returned other final positions. The seal that `pack` writes
  by default causes the file to be refused. In a file without a seal, the defect remains and the reader says so.

## Empty output of a read launched in parallel (unexplained)

`guard.random_alteration` launches its 244 reads six at a time. On 15 November 2026, at the first run of the
PowerShell script that this suite was ported from, two of them came back without any output, neither a line of `play`
nor a message, with the exit code of a normal read (0 for one, 1 for the other). Replayed by hand, they return their
complete output. The defect did not recur over the nine following runs of the case, nor over 540 reads of one and the
same file launched 1, 6 and 12 at a time; the Windows event log carries no application error. The cause is not
established: the launcher of that script, or the end of the process. The Python suite launches its commands
otherwise, and the defect did not appear over its first six runs of the case, on 17 November 2026; that is too few
to say that it is gone. The case still runs again alone, once, a read that came back empty, and writes it on its line
("N read(s) run again after an empty output"); empty again, it is a failure.

## What is not covered

- 256³ and 512³: everything that appears only at large size (real cap on the passes, projection floors,
  memory). The results of the measurements are not replayed here.
- `psi_view`, `tracers` and `psi_ref`: not run by these suites.
- Run times, throughputs and memory occupancy: none is read.
- Another machine: the fingerprints have been recorded only on the toolchain described in `fingerprints.json`.
- Linux: `test_codec.py` is written to run there (name of the executables, paths, cuFFT library found through the
  dynamic loader), but it has been run on Windows only. On Linux, only its start has been tried: the refusal with
  code 2 when the binaries are missing, and the reading of the headers of `.psi` files.
- Options of `play`: `--raymarch`, `--resort`, `--seek`, `--no-fine`, `--cache`, `--ray-cache`, `--block`, unsorted
  particles. Options of `pack`: `--key-interval`, `--mf`, `--levels`, `--legacy-step`, `--fit-at`, `--allow-clipping`,
  `--brick-dump`, `--check-every`. Reading a PSISTRM1 file (32 files of that format were read back by hand, see
  `benchmarks/robustness/README.md`).
- Altered files are tried only at 64³, and one alteration at a time. Two alterations that cancel each other, or
  an alteration that appears only at large size, are not covered.
- The alterations drawn at random do not bear on a file with exceptions: an altered overflow value would be
  read there without being seen (known defect above).
- A file with raw keys and without a seal is not altered: the three alterations of `guard.geometry_no_seal` that bear
  on raw keys are made in a sealed file (see the table above).
- Exit code 2 has been provoked for missing binaries, for a pattern that matches no case and for `--update` given
  with `--only`; not for a failure of `dns`. The stopping of a command that hangs has been tried with a stand-in
  program, not with `psi_stream`.

## Measured duration

On 17 November 2026, GPU free: 41 to 46 s for the 50 cases (four runs), 1 s to produce the sequence. Run alone,
`guard.random_alteration` takes 14 s: four `pack` commands, then its 244 reads, six at a time. These durations are
indicative.

## Mutation tests

On 15 November 2026, the rounding of the integer predictor (`predict3`, now in src/codec_kernels.cuh, `+ 512` replaced
by `+ 511`) was modified for the duration of one build, then restored. The suite then had 38 cases: 8 failed, the
fingerprints of the `.psi` files (`fingerprint.base`, `.bound`, `.old`, `.raw_keys`, in degree 2 and 3). The others
passed, including `pack_play` and `fingerprint.positions`: the encoder and the reader share this predictor, the
decoded state does not change, only the bytes emitted change. Yet this change breaks the reading of existing files
(the mutated reader finds 32 frames out of 36 whose state differs in a file written before): only the fingerprints
see it.

On 17 November 2026, the suite itself was mutated, then restored. One digit changed in the references of
`d3_bound.psi` and of `d2_base.positions`: `d3.fingerprint.bound` and `d2.fingerprint.positions` fail, the 15 other
fingerprint cases pass. Three expected messages changed in the script (the bound held announced by `play`, the cap of
1 pass named by `pack`, the frame table named by the seal): `d2.bound_held`, `d3.bound_held`, `d2.pass_cap`,
`d3.pass_cap` and `guard.sealed_header` fail, and `d2.seal`, run with them, passes.
