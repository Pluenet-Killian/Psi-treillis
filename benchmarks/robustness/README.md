# Robustness of the reader: altered files, the seal

## What is measured and why

A `.psi` file can reach the reader damaged or truncated. The reader therefore checks the header and the frame table at
opening, verifies the seal of the file when it has one, and checks each frame against the geometry of the header
before any decoding kernel runs (`docs/FORMAT.md`: the seal, and the format limits guarded by the code). Two
measurements look at these guards from outside, by running `psi_stream play` and reading how it ends.

**Altered files** (`altered_files.py`, claim ROBUST-1). The reader must refuse an altered file, or read it while
counting the frames that differ; it must never end in a CUDA error, a crash or a hang. The files are those that two
cases of the test suite write at 64³:

| Batch | Files | Alterations |
|---|---|---|
| chosen | `build/tests/cases/guard_geo_*.psi` (case `guard.geometry_no_seal`) | chosen alterations of the header, of the frame table and of the frames, in files without a seal |
| random | `build/tests/cases/random/*.psi` (case `guard.random_alteration`) | one bit flipped or one byte replaced at an offset drawn with a fixed seed, in four files of which two are sealed |

Each file is read by `psi_stream play --file F.psi --particles 1000 --sort`, six reads at a time, each with a watchdog
timeout of 60 s. The outcome of a read is one of:

| Outcome | Definition |
|---|---|
| `refused` | exit code 1 and a message `Failure:` |
| `counted` | complete read, at least one frame whose state differs from that of the sender (exit code 1) |
| `read` | complete read, no differing frame (exit code 0): the alteration has no effect, or is not seen |
| `CUDA error` | a message `CUDA error` |
| `crash` | exit code other than 0 and 1 |
| `hang` | read stopped after the watchdog timeout |

**The seal** (`seal_256.py`). `pack` seals a file by default: 48 bytes written after the frame table, which cover the
header and the table. At 256³, two encodings of `data/turb256.seq` (those of the reference files `z_t256_d2`, degree 2
without a bound, and `f_t256_d3_t05`, degree 3 with a bound of 5 % of u′) are written twice, sealed and with
`--no-seal`. The script records whether the sealed file is the other one followed by 48 bytes, and how the reader
reads each of them: exit code, differing frames, final positions of one million sorted particles, and what the reader
says of the seal.

## How to replay

From the root of the repository, with `build/psi_stream` built (`tools/build.ps1 -Tools` on Windows, CMake with
`-DPSIT_BUILD_TOOLS=ON` elsewhere):

```
# altered files: the suite writes them, the script reads them back
python tests/test_codec.py --keep-alterations --only "guard.geometry_no_seal" "guard.random_alteration"
python benchmarks/robustness/altered_files.py

# the seal: needs data/turb256.seq
python benchmarks/robustness/seal_256.py
```

`altered_files.py` takes `--cases FOLDER` (the folder of the cases of the suite, `build/tests/cases` by default),
`--simultaneous N` and `--timeout SECONDS`. The two scripts read no time: the GPU may be shared. Python 3.8 or
later, standard library only; Windows and Linux.

## What the scripts write

| Path | Content |
|---|---|
| `altered_files.csv` | one line per altered file: `batch`, `file`, `exit_code`, `outcome`, `message` (first line `Failure:` or `CUDA error:` of the reader, if any) |
| `logs/altered_files_chosen.log`, `logs/altered_files_random.log` | the raw output of every read of the batch, each under a line `== file: exit code, outcome` |
| `seal_256.csv` | one line per encoding, the columns below |
| `logs/seal_256_<encoding>_<run>.log` | the raw output of each run: `pack_sealed`, `pack_unsealed`, `play_sealed`, `play_unsealed` |
| `build/work/seal_256/` | the four files of the seal measurement (not versioned) |

Columns of `seal_256.csv`:

| Column | Meaning |
|---|---|
| `encoding`, `pack_options` | the encoding and the options given to `pack` |
| `sealed_pack_exit_code`, `unsealed_pack_exit_code` | exit codes of the two `pack` |
| `sealed_bytes`, `unsealed_bytes`, `extra_bytes` | sizes of the two files, and their difference |
| `unsealed_sha256`, `sealed_sha256_without_last_48_bytes`, `same_bytes_before_the_seal` | SHA-256 of the file without a seal, SHA-256 of the sealed file without its last 48 bytes, and `yes` if they are equal |
| `sealed_play_exit_code`, `sealed_frames_differing`, `sealed_final_positions`, `sealed_seal_seen_by_reader` | the sealed file read by `play`: exit code, line `Integrity`, and `present` or `absent` from its line `Seal` |
| `unsealed_play_exit_code`, `unsealed_frames_differing`, `unsealed_final_positions`, `unsealed_seal_seen_by_reader` | the same for the file without a seal |
| `same_final_positions` | `yes` if the two reads end on the same fingerprint of the final positions |

In the raw logs, standard output comes first, then standard error. The patterns that read the output of `play` are in
`benchmarks/compression/psi_stream_runs.py`, each with the source line that prints it.

Exit codes of the scripts. A CUDA error, a crash, a hang, a differing frame or a `no` in a column is a result: it is
recorded, and the script exits with code 0. A script exits with code 1 when it cannot name what happened: a read that
ends with exit code 0 or 1 without the line that goes with it (its outcome then reads `code N without an expected
line`), a file that `pack` did not write. It exits with code 2 when it has run nothing (no file to read, the reader
missing).

## Caveats

- The alterations are tried at 64³ only, one at a time, on the files of one sequence. That no read of these batches
  ends in a CUDA error does not prove that none can: the bounds on what the reader reads rest on reasoning about the
  code (`docs/FORMAT.md`, sections on the coding and on the format limits), not on these batches.
- `read` does not mean that the alteration was detected. In a file without a seal, nothing covers the header or the
  frame table: a field altered there can change the field that is rendered without changing the state that the
  fingerprints check. The test suite (`tests/README.md`) says which alterations of the random batch are in this case,
  and describes one known defect of the same kind.
- A read can come back with a normal exit code and no output at all; the test suite describes the one occurrence
  observed. `altered_files.py` does not run such a read again: it records `code 0 without an expected line` and exits
  with code 1.
- The seal is measured at 256³ on two encodings; no sealed file is written at 512³ here. The test suite checks the
  same property at 64³, and that an altered sealed file is refused.
- The reader of the visual comparator (`psi_view`) shares the opening checks of `play` but is not run by these
  scripts.
