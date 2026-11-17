# Regression tests and public interface (claims TEST-1, API-1, API-2)

## What is measured, and why

Three logs, which answer three questions about the code as it is published.

- **Is the pack / play chain deterministic, and does it still behave as recorded?** (TEST-1)
  `test_logs/test_codec_64.log` is the output of the regression suite `tests/test_codec.py`, on a turbulence sequence
  at 64³: one line per case, then a total. The cases check bytes, fingerprints, exit codes and messages of
  `psi_stream pack` and `psi_stream play`; they are described in `tests/README.md`.
- **Does the public interface give what the measurement front end gives?** (API-1)
  `test_logs/test_api_64.log` is the output of `tests/test_api.py`, on the same sequence at 64³: the library
  (`include/psitreillis.h`), its Python module (`python/psitreillis`) and the `psit` tool. Among its cases: the file
  written through the interface is that of `psi_stream pack` byte for byte, the decoded field respects the requested
  bound, the field sampled off the grid is divergence-free, and the GPU memory is released on closing.
- **At 512³, does a file decoded through the public interface have the error that the encoder recorded in it?**
  (API-2) `bounds_512/verify_bounds_512.py` opens six reference files at 512³ with the Python module (four with an
  error bound, two without), decodes four frames of each to the grid, and compares them in NumPy with the raw frames
  of `data/turb512.seq`. The frames are the first one, which is a key frame, the innovation that follows it, the
  first innovation after the next key frame, and the last one. The encoder measures the error of each frame while it
  writes the file, and records it in the frame table; this measurement reaches the same figures by another code path:
  the kernel of the library that decodes to the grid, then a comparison made on the host in double precision, outside
  the code of the project.

  A frame is `consistent` if its largest error is within 5·10⁻⁵ u′ of the value recorded in the frame table (half the
  certification margin: what two GPU kernels may differ by), if its RMS error is within 0.02 % of the recorded one,
  and if, for a file whose header says that the bound is held, its largest error does not exceed the tolerance of the
  file. u′ is the RMS velocity of the header of the sequence.

No timing is read by any of the three.

## How to replay

From the root of the repository, with PowerShell 7 (an NVIDIA GPU is needed for all three):

```powershell
.\tools\build.ps1 -Tools
python tests/test_codec.py > benchmarks/interface/test_logs/test_codec_64.log
python tests/test_api.py > benchmarks/interface/test_logs/test_api_64.log
python benchmarks/interface/bounds_512/verify_bounds_512.py
```

The two suites produce their sequence at 64³ themselves if it is missing. `verify_bounds_512.py` needs NumPy,
`data/turb512.seq` and, in `build/work/reference_files/`, the files `f_t512_d3_t03.psi`, `f_t512_d3_t04.psi`,
`f_t512_d3_t05.psi`, `f_t512_d2_t07.psi`, `z_t512_d3.psi` and `z_t512_d2.psi`, which are written by
`benchmarks/compression/reference_files/reference_files.py` (see the notice of that measurement). It says which file
is missing and exits with code 2. Its exit code is 0 if every frame is consistent, 1 otherwise.

## What is written

| File | Content |
|---|---|
| `test_logs/test_codec_64.log` | Output of `tests/test_codec.py`: one line per case (`OK`, `FAIL`, `KNOWN` for a known defect), the total |
| `test_logs/test_api_64.log` | Output of `tests/test_api.py`: the version of the library, one line per case (`ok`, `FAIL`), the total |
| `bounds_512/verify_bounds_512.csv` | One row per frame checked: `file`, `frame` (position in the file), `sequence_frame` (position in the sequence), `key_frame`, `has_bound`, `bound_held`, `tolerance`, `largest_error` and `largest_error_recorded` (fractions of u′), `rms_error` and `rms_error_recorded` (fractions of the RMS of the field of the frame), `verdict` |
| `bounds_512/verify_bounds_512.log` | The same rows, after a line that gives the version of the library, the sequence and u′, and before the count of frames checked and mismatched |

The files that the suites write while they run (sequence, encoded files, output of each command) stay in the build
folder; they are described in `tests/README.md`.

## Caveats

- **The two suites run at 64³ only.** They check the behaviour of the code, not the figures of the paper: sizes and
  errors at 64³ say nothing about 256³ or 512³.
- **The fingerprints of TEST-1 are valid for one toolchain**: version of CUDA and of cuFFT, host compiler,
  architecture of the GPU. `tests/fingerprints.json` describes it, and `tests/README.md` says what to do on another
  machine before concluding that there is a regression.
- **A `KNOWN` line is a defect that is known and not corrected.** It is shown at every run and does not count as a
  failure; `tests/README.md` describes each one.
- **API-1 does not cover everything**: the exit "bound missed" of the interface cannot be reached at 64³ through the
  public options, and the suites have been run on Windows only.
- **API-2 checks four frames per file, not all of them**, and one sequence. Three of the four frames are innovations,
  one is a key frame. The decoding uses the GPU kernel of the library; only the comparison is independent of the
  project.
- **For a file without a bound, or whose bound is not held, API-2 checks only that the measured errors are the
  recorded ones**; the tolerance column is then not a constraint.
- **API-2 depends on the reference files as they are on disk.** If they were written by another build of the encoder
  than the library that reads them, a mismatch can come from either side; rewrite them first.
- **The redirections above are written for PowerShell 7**, which writes UTF-8. Windows PowerShell 5 writes UTF-16,
  which `tools/claims.py` does not read.
