# Changelog

This changelog records what changes a result, a file format, an option or the organisation of the repository.

A change marked **[bytes]** alters the `.psi` files produced with the same options: earlier files and fingerprints can
no longer be compared without the compatibility option indicated.

## 0.2.0, 17 November 2026: the repository as published

The project is published in this form: an article, the library, its tests, and one folder per measurement. No file
format changes and the encoder writes the same bytes: the 19 reference fingerprints of the tests are unchanged (the
test sequence, 14 encoded files, 4 sets of final positions).

- **Organisation.** The codec core is split by who uses it: `src/codec_kernels.cuh` and `src/codec_stream.cuh` hold
  what the library needs; the kernels of the instruments (advection, ray marching, reference paths) are in
  `tools/playback_kernels.cuh`, and the measurement front end `psi_stream` (pack and play) is in `tools/`. The first
  bench program, which shared a file with the kernels, is no longer part of the code. The turbulence solver, the tracer
  experiment, the visual comparator and the two reference readers are in `tools/`; each measurement has its folder
  in `benchmarks/`, with its script, its logs and a notice. By default only the library, `psit` and the examples are
  built; `-DPSIT_BUILD_TOOLS=ON` builds the instruments.
- **Everything is in English**: comments, messages of the programs, identifiers, names of files, of columns and of
  test cases. `tools/check_english.py` checks it. Consequences for a user of the earlier state:
  - the programs print in English, and the scripts read these messages;
  - the reference reader `psi_ref` takes `--probe`, `--state`, `--frame`, `--frames` and `--threads`, and the Python
    reader `--header`, `--probe`, `--frame` and `--output`;
  - probe files take the extension `.probe`;
  - five families of claims are renamed (table in `docs/CLAIMS.md`).
- **Error messages** name the source file without the folders of the machine that built the program.
- **Library** version 0.2.0. The public interface is unchanged.
- **Tests.** The regression suite of the codec chain is rewritten in Python (`tests/test_codec.py`) and runs on any
  platform: 50 cases instead of 52, the two missing ones having tested the bench program that is gone. One case is
  corrected: `guard.geometry_no_seal` meant to alter three files with raw keys after removing their seal and altered
  them with the seal on; it now removes the seal, and the three files are still refused.
- **Measurements replayed.** Every log of `benchmarks/` was produced again by this code. Compression ratios, errors,
  exit codes and fingerprints are those of the logs of 15 and 16 November; timings were measured again.
- **Equal-bound bench, corrected protocol** (BENCH-1 to BENCH-3): every codec on the same 32 frames, the error of
  Psi-treillis measured like that of the others, SZ3 in 4D by blocks of 8 frames at 512³, zfp reported apart. The
  three verdicts are unchanged.
- **New claims, their criteria written before the measurements**: GRID-1, READ-1 and READ-2 (the divergence of a
  frame as each compressor gives it back, on the grid and as read), TRAC-1 (tracers through a grid decoded by SZ3 or
  SPERR), SPEC-1 (energy spectrum); they hold. PLAY-1 (cost of one replayed frame) is measured in one protocol.
- **Verdicts that change.** LINUX-1 goes from "holds" to "pending": the workflow that builds the project on Linux can
  only run once this repository is published; the same build was green in the development repository on 16 November.
  LAGR-2 is now read in absolute value, as its criterion says; its verdict does not change. REF-3 is added, pending:
  a third reader from the specification as completed.
- **Format specification.** `docs/FORMAT.md` cites the sources by the name of the function or structure, and is
  completed on the six passages that the writing of the second reader had shown to be insufficient. One of them is
  a limit of the format, now stated: nothing bounds the coefficients of the temporal predictor.
- **Not done.** No measurement on a public dataset: access to JHTDB was obtained, the transfers of the cutout broke
  repeatedly. No measurement on a second machine.

## Earlier history

The entries below were written during the development, before the reorganisation above. They are kept as they were
written: they name files, folders, options, messages and claim identifiers as they were then, and they point to
commits and to documents of the development repository, which is not public. They remain the record of how each
result and each verdict came about.

<!-- earlier history: names as they were -->

## 16 November 2026: validity map, bench rerun, Linux build, documents in English

No file format and no option changes. Nine verdicts of the sheet change, and the documents change language.

- **Resolution sweep** (`results/validite/`): 18 sequences at 256³, from 0.36 to 0.80 voxel per Kolmogorov scale, on
  criteria committed before the first measurement (`ec71f76`). VALID-1 holds: the bound of 5 % is certified on the 8
  sequences at 0.543 voxel or more and refused on the 10 below. VALID-3 holds: a lattice step of 1 voxel keeps the
  bound everywhere, at about half the compression ratio of SPERR. VALID-2, VALID-4 and VALID-5 do not hold: SPERR
  compresses more on the two sequences just above the threshold; the field error is 0.76 % at 0.702 voxel, above the
  0.70 % expected; between realisations the compression ratio varies by up to 13.1 %.
- **Windowed cutout** (`results/decoupe/`): CTRL-1 holds. The error at the centre of a windowed cutout is 1.062 and
  1.089 times that of the whole field on our two sequences at 512³. The trial on a cutout of the public JHTDB dataset
  (JHTDB-3, JHTDB-4, REPR-1) is not done: the download from the database did not complete.
- **Lagrangian confirmation** (`results/lagrangien/rang7_analyse_confirmation_256.txt`): LAGR-5 does not hold. Three
  of its four conditions are met on the new realisation (cuZFP at 40.8 standard errors, trilinear texture at 2.7,
  variance of D growing 4,641 times faster with the texture); one reference path is at -2.8 for a threshold of 2 in
  absolute value, and at -0.6 on ten seeds.
- **Bench at an equal bound, rerun with the current encoder** (`results/banc/`): BANC-1, BANC-2 and BANC-3 keep their
  verdicts and now read these logs. The compression ratios of Psi-treillis differ by less than 0.4 % from the archived
  ones.
- **Linux build**: LINUX-1 holds. `.github/workflows/linux.yml` compiles the library, the tools and `psi_ref` on
  Ubuntu 24.04 without a GPU at each commit and replays the claims sheet. The first run failed on the link of
  `examples/roundtrip.c` (mathematical library), fixed in `c8db27e`. `PSIT_CUDA_ARCHITECTURES` selects the CUDA
  architecture at configuration.
- **Documents in English.** `docs/RESULTATS.md`, `docs/REVENDICATIONS.md`, `docs/LISEZMOI.md`, `docs/FORMAT.md`, this
  changelog, the notices of `results/` and of `tests/` are translated, and renamed `docs/RESULTS.md`, `docs/CLAIMS.md`,
  `docs/GUIDE.md`, `README.md` in each folder; `tools/revendications.py` becomes `tools/claims.py`. The translation was
  checked file by file: same digits, same code spans, same commands. The criteria committed in French before the
  measurements are translated clause for clause; the originals are in the history (`ec71f76`). Not translated: the
  messages of the programs and the logs, which the claims sheet reads back; the comments of the sources; the names of
  the scripts and of the folders of `results/`; the writing journal of the second reader, kept verbatim; `archive/`.
- **Visual comparator.** The text on screen of `psi_view` is in English. Its divergence view draws the divergence in
  colour over the velocity gradient in grey, instead of a black image where the divergence is zero. The five
  screenshots are retaken with degree 3 and the current encoder by `scripts/figures.ps1`, in `docs/figures/`; the
  earlier ones move to `archive/vues_2026-11-15/`. The codec itself is not touched.
- **Plots.** `tools/figures.py` draws the validity map and the bench from the logs, for the light and the dark theme.
- `README.md` states what the codec is not: at an equal bound SZ3 in 4D compresses 2.5 times more, and sampling the
  lattice in degree 3 is not faster than sampling a decoded texture at 256³.
- The sheet counts 40 claims: 23 hold, 10 do not hold, 7 pending.

## 16 November 2026: second reader, in Python, verdict of REF-2

No format and no option changes, and `docs/FORMAT.md` is not touched. One verdict of the sheet changes.

- **REF-2 goes from "pending" to "does not hold"**, on the criterion committed before the log (`ec71f76`), by its last
  clause only: one pending hypothesis fewer, one claim that does not hold more.
- `tools/lecteur_psi.py`: a reader of `.psi` files in Python 3 and NumPy, without a GPU, written from `docs/FORMAT.md`
  alone, without reading `src/`, the first reader or `results/reference_cpu/`. It reads `PSISTRM2` and `PSISTRM3`, with
  or without a seal, raw and coded key frames, exceptions, sparse fine level; it recomputes the fingerprint of each
  frame, gives the velocity field at given points and refuses an invalid file with a message and the code 1. `PSISTRM1`
  is not supported.
- `results/reference_python/`: script, log, writing journal and notice. Measured on the five reference files at 256³
  and the 22 valid test files at 64³: 1,029 frames read back, no fingerprint different from that of the table; at the
  probe of each file, sampling lattice and fine level identical to those of the encoder, and velocity field at
  2.0·10⁻⁵ u′ at most, for a threshold of 5·10⁻⁵. The 27 probes come from a re-encoding identical to the file, to the
  SHA-256.
- The reader was committed before its first run (`7cc43a1`) and gave everything back at that run: no trial was used to
  decide between readings, and its decoding has not been touched.
- **Why "does not hold".** The criterion requires the list of the passages where the text was not sufficient to be
  empty for the state and the field. Six passages were noted during the writing; one counts for the state: the
  prediction formula (section 6.2) does not say on how many bits the sum is computed, and nothing bounds the
  predictors. It has no effect on the files read, whose predictors are a factor 21 away from the case where an exact
  sum and a sum on 32 bits differ (calculation). This is a strict reading of the criterion; whoever does not count
  this passage reads "holds".
- Outside the criterion: six files produced at 64³ for what the files of the criterion do not exercise. The tables of
  the empty segments of a fine level without a brick do sum to 4096; the rule of the raw key frames in `PSISTRM3`,
  which section 5 called untested, is exact; a file with a wrapped lattice, read by differences, gives back the field
  of the encoder to 5.6·10⁻⁶ u′, and shows that the choice of the node subtracted weighs 0.42 u′: the text does not
  name it.
- The 84 altered files at 64³: same outcome and same number of differing frames as with `psi_stream play`.
- The sheet counts 40 claims: 19 hold, 6 do not hold, 15 pending.

## 16 November 2026: eleven hypotheses, their criteria fixed before any measurement

No result changes: this commit puts a date on criteria. The measurements come after it.

- **Resolution sweep** (`results/validite/validite.py`): 18 sequences at 256³, 6 viscosities and 3 realisations.
  VALID-1, the resolution threshold above which the 5 % bound holds; VALID-2, the compression ratio against SPERR where
  it holds; VALID-3, the step of 1 voxel; VALID-4, the field error as a function of the resolution; VALID-5, the
  scatter between realisations.
- **Windowed cutout.** CTRL-1: on our 512³ sequences, does the error at the centre of a windowed cutout reproduce
  that of the whole field? JHTDB-3, JHTDB-4 and REPR-1: the same protocol on a cutout of the public JHTDB dataset.
- **JHTDB-1 and JHTDB-2 can no longer be tested** under the access conditions obtained: the database limits a cutout
  to 3 GB and reserves its use for small subsets, whereas 8 whole frames weigh about a hundred GB. They stay in the
  claims sheet, pending, with this reason.
- REF-2: a second reader written from the completed specification. LINUX-1: the build under Linux, verified on
  GitHub.
- The claims sheet has 40 claims: 19 hold, 5 do not hold, 16 pending.

## 16 November 2026: specification cross-checked against the code, line numbers realigned

No result, no format, no option, no verdict changes: only the text of `docs/FORMAT.md` changes.

- The "clarified" and "established by trial" passages added from the reference reader (next entry) have been re-read
  against `src/psi.cu` and `src/psi_stream.cu`. All conform to the code; each now carries the mention "Cross-checked"
  and the `fichier:ligne` reference that supports it.
- The code answers points 5 to 9 of section 10: the order of operations of the GPU reader (written in 6.7), the formula
  of the worst case of the coder, the value of `q(k-1)` for an innovation of type 1 that would follow a key frame, what
  a block reads beyond its frame, the drawing of the probe points.
- Section 11: the eighth integer of the header of a probe is not always zero. It is a flag, set to 1 when the lattice
  of an already encoded frame has wrapped.
- The 138 `fichier:ligne` references that the touch-ups to `src/` of the same day had shifted are realigned on the
  current tree.
- What remains true: no reader has yet been written from the completed text, and the field of the GPU reader cannot be
  reproduced bit for bit from the source.

## 16 November 2026: specification completed from the reference reader

No result, no format, no option, no verdict changes: only the text of `docs/FORMAT.md` grows longer.

- `docs/FORMAT.md` receives what the reference reader had shown to be missing. Under the mark "established by trial":
  the masks of the refinement, their alignment and the rounding of the division (6.5); a section 6.7, the velocity
  field at a point (coordinate system, position of the nodes, basis, curl, scales); a section 11, the probe file of
  `pack --probe`. Under the mark "clarified": the unsigned byte of the fingerprint (7), the folding and the order of
  the symbols of a coded key frame (6.4), the arithmetic shift and the order in which the escapes are applied (6.2),
  the position of an overflow (6.3), the reading order of a `PSISTRM3` frame (5).
- These additions do not come from a reading of the code and have not been cross-checked against it: they carry no
  `fichier:ligne` reference. The original text is not modified.
- One more convention is established on this occasion: the centre of the voxel of index `(i, j, k)` is at `(i + 1/2)`
  voxel on each axis, measured against `psit sample`.
- `results/reference_cpu/conventions.py` replays the trials that decided between the possible readings; log
  `conventions.txt`.
- Section 10 of `docs/FORMAT.md`, five points that the trials did not resolve: the single-precision arithmetic of the
  GPU reader, the worst case of the coder, the first innovation of type 1, reading beyond an altered frame, the
  drawing of the probe points.
- The caveat of REF-1 says so: the specification as it then stood was not enough for the velocity field; no reader has
  yet been written from the completed text.

## 16 November 2026: reference reader on CPU, verdict of REF-1

No format, no option of `psi_stream` changes. One verdict of the claims sheet changes.

- **REF-1 goes from "pending" to "holds"**, on the criterion committed before the log was taken (`ed3761c`): one more
  claim that holds, one fewer pending hypothesis.
- `src/psi_ref.cpp`, built by `scripts/build_ref.ps1` in `build_ref\`: a reader of `.psi` files on CPU, in C++17, with
  neither CUDA nor any library, written from `docs/FORMAT.md` without reading the rest of `src/`. It reads `PSISTRM1`
  to `3`, with or without a seal, raw and coded key frames, exceptions, sparse fine level; it refuses an invalid file
  with a message and code 1. Options: `--sonde` (comparison with a probe of `pack --probe`), `--points` (velocity field
  of a frame at given positions), `--etat` (decoded state of a frame).
- `results/reference_cpu/`: script, log and readme. Measured on the twelve reference files: 622 frames read back, no
  state fingerprint different from that of the table; at the probe of each file, sampling lattice and fine level
  identical to those of the encoder, and a velocity field within 2.8·10⁻⁵ u′ at most of that of the GPU reader, for a
  threshold of 5·10⁻⁵. On the 106 test files at 64³, valid and altered, the CPU reader and the GPU reader return the
  same outcome and the same number of differing frames.
- **What the log says about `docs/FORMAT.md`.** The state of the frames can be decoded from the specification alone.
  The velocity field cannot: the masks and the rounding of the refinement (section 6.5), the position of the lattice
  nodes and the format of the probe are not in it, and were established by trial on files. The list of the gaps is in
  `results/reference_cpu/README.md`; `docs/FORMAT.md` is not modified by this change.
- `.gitignore`: `build_ref/`.

## 16 November 2026: library and public interface (version 0.1.0)

No result, no format and no option of `psi_stream` changes. The 52 cases of `tests/run_tests.ps1` pass with the same
fingerprints after recompilation: the `.psi` files produced are the same, byte for byte.

### What is added

- **Library** `psitreillis` (`include/psitreillis.h`, `src/psitreillis.cu`): open a `.psi` file, decode a frame to a
  grid, sample the velocity at arbitrary points, compress a sequence from memory, from a callback or from a `.seq`
  file. An error is returned to the caller instead of stopping the program, and the GPU memory is released when a
  reader is closed and at the end of a compression.
- **Tool** `psit` (`src/psit_cli.cpp`): `info`, `compress`, `decompress`, `sample`, `convert`; inputs and outputs in
  `.npy`, in raw floats or in `.seq`.
- **Python module** `python/psitreillis` (ctypes and NumPy), **examples** in C and in Python (`examples/`).
- **Usage documentation in English**: `README.md` at the root, `docs/API.md`, `CONTRIBUTING.md`, `CITATION.cff`.
- `scripts/build_lib.ps1`; installation by `cmake --install` with a CMake package (`find_package(psitreillis)`); the
  CMake project is called `psitreillis`, version 0.1.0.
- `tests/test_api.py`: 16 cases on the public interface. `results/api/`: at 512³, the errors of six reference files
  read back through the public interface and NumPy.
- Claims sheet: API-1 and API-2, which hold. It has 29 claims: 18 hold, 5 do not hold, 6 pending.

### What moves

- The original README, in French, becomes `docs/LISEZMOI.md` (translated and renamed `docs/GUIDE.md` on 16 November): content
  intact, links and references corrected.

### Project name

- The project is called Psi-treillis everywhere in the repository: texts, scripts, messages, title of the window of the
  comparator. Its former working name no longer appears in it.
- The CMake option for the prototypes is called `PSIT_PROTOTYPES` (`-DPSIT_PROTOTYPES=ON`, or `.\scripts\build.ps1
  -Prototypes`) and the compile definition of the cuZFP reference `PSIT_WITH_ZFP`.
- In `archive/`, two reports and their notes folders change name: `Améliorations du treillis Psi` and
  `Audit de recherche Psi-treillis`.

### Existing sources

Three touch-ups, with no effect on the existing programs:

- `src/psi.cu`: the macros `CK` and `CHK` can be defined before the file is included.
- `src/sequence.h`: `sequence_fail` can throw an exception (`PSI_SEQUENCE_FAIL`).
- `src/psi_stream.cu`: `pack` reads its frames through a source (`FrameSource`); the version that opens a `.seq` file
  calls it.

### What is not done

- No Linux build has been verified. The Python module is not packaged for `pip`.
- The refusal messages of the codec core remain in French; the most common ones are also given in English.

## 16 November 2026: licence and public repository

No result, no format, no option changes.

- The project is distributed under the Apache 2.0 licence (`LICENSE`, at the root).
- Public repository: <https://github.com/Pluenet-Killian/Psi-treillis>.
- The commit identifiers cited in the documents are those of the published history.

## 16 November 2026: claims sheet

No result, no format, no option changes.

- `docs/CLAIMS.md`: 27 claims, each with a statement, a numerical criterion, the log that supports it, the
  command that reproduces it and a verdict. 16 hold, 5 do not, 6 await a measurement.
- `tools/claims.py` re-reads the logs of the repository, returns the verdicts and writes the generated part of
  the claims sheet. It exits with code 1 if a verdict changes, if a log is missing or if the sheet is no longer up to
  date. Without a GPU.
- `results/tests/suite_64.log`: log of the regression suite (52 cases, 51 OK, no failure, 1 known defect), the log of
  claim TEST-1.
- **Criteria fixed before the measurement** at this date, for six hypotheses: JHTDB-1 and JHTDB-2 (public dataset at
  1,024³), PORT-1 (another machine), REF-1 (reader independent of the code), LAGR-5 (another realisation of the
  turbulence), BANC-4 (SZ3 in 4D on 32 frames at 512³). The sixteen criteria of the claims that hold were written after
  the measurement, except those of LAGR-1 and LAGR-2.

## 15 November 2026, late evening: reader hardened against an altered file

Measurements and scripts: `results/durcissement/` (`mesures.md` gives their detail and provenance).

### Format

- **[bytes]** `pack` ends the file with a seal of 48 bytes, `StreamSeal` ("PSISEAL1"): the fingerprint of the header,
  that of the header of the fine level and that of the frame table (`docs/FORMAT.md`, section 4.1). Everything that
  precedes the seal is unchanged: a sealed file is the old file followed by 48 bytes. The signature (`PSISTRM2`,
  `PSISTRM3`) does not change, and a reader from before the seal reads a sealed file without change.
- To reproduce byte for byte a file from before the seal: `--no-seal`. The recipe for a file with a bound earlier than
  the evening of 15 November becomes `--fine-passes 12 --tolerance-margin 0 --no-seal`. The replay scripts of
  `archive/journal/2026-11-15/audit/` pass `--no-seal`: their references, their sizes and their fingerprints are those
  of files without a seal.
- `tests/empreintes.json` is regenerated: eight fingerprints of sealed files change, those of `ancien` (written without
  a seal) and of the final positions do not change, and four fingerprints of files written with `--no-seal` are added,
  equal to the old fingerprints of the same files.

### Reader (`psi_stream play`, `psi_view`)

- The seal is checked at opening. An altered header, header of the fine level or table is refused with a message that
  names the altered part. A file without a seal is read, and the "Sceau" (seal) line says so; `play --require-seal`
  refuses it.
- The size of the file must be the one the header announces, with or without a seal. Each frame of the table must fit
  between the headers and the table and carry a known type; the first must be a key frame.
- Before any decoding kernel, the payload of each frame is checked on the host against the geometry of the header:
  exact size of a raw key frame, list of exceptions, frequency tables and offset table contained in the frame, offsets
  increasing and within the stream area, section of the fine level contained. The check is made while the frame
  crosses the bus.
- Three safeguards in the kernels, with no effect on a valid file: the initial state of an rANS block is brought back
  to 2^23 if it is smaller (a block then reads 2052 bytes at most, and the receive buffer has this margin); an overflow
  whose index falls outside level 0 is ignored; the bricks and the nodes of the fine level that exceed the counts
  announced by the frame are treated as absent.
- `PSISTRM1` files: at each frame `play` counts the sums of the pyramid outside ±32767, which their original reader
  clipped, and says whether the decoded lattice is that reader's.

### Measured

- 283 altered files at 64³ (43 chosen alterations, 240 drawn at random with a fixed seed): with the previous reader, 10
  gave "Erreur CUDA: an illegal memory access"; with this one, none, neither a crash nor a hang. 135 are refused, 106
  give a count of differing frames, 37 are read with no effect on the final positions. The other 5 are alterations of
  the header or of the table of a file without a seal, read without being seen: without a seal, nothing can see them.
- The 82 files of `build_psi/garde/`: 80 read back by both readers with the same final positions, 2 interrupted
  encodings refused by both.
- Reading times at 512³, five then fifteen interleaved repetitions of the old and the new binary, on two files: the
  difference of the medians goes from -0.014 to +0.008 ms for the decoding of a frame (3.7 to 3.9 ms) and from -0.003
  to +0.003 ms for an RK4 step (4.0 to 4.2 ms), below the range of the repetitions (0.02 to 0.06 ms).
- The 32 `PSISTRM1` files of `build_psi/`: 29 with no sum outside ±32767, among them `turb256.psi`; 3 safeguard files,
  already clipped by their encoder, have some on all their frames.

### Known defects, not fixed

- **The value of an overflow of level 0 is covered by no fingerprint.** It affects only the sampling lattice, not the
  quantised state that the fingerprint of a frame covers. Altered, it changes the field: `play` exits with 0 and
  announces "0 trame(s) dont l'état diffère" (zero frames whose state differs). The suite reports it at each run
  (`garde.debordement_altere`).
- **A file without a seal keeps an unprotected header**: files written before the seal, or with `--no-seal`. The
  reader says so, it cannot see it. The seal does not protect against its own removal: stripped of its last 48 bytes,
  a sealed file is a valid file without a seal, which only `play --require-seal` refuses.
- **Three `PSISTRM1` safeguard files** have sums outside ±32767: the current reader reports it, it does not restore
  the clipping of the original reader.
- **An unexplained empty output in the tests.** Once in ten runs of the case `garde.alteration_aleatoire`, two reads
  out of 244 launched in parallel came back with no output at all, with the code of a normal read. Not reproduced
  since; the case reruns such a read and says so on its line (`tests/README.md`).

## 15 November 2026, evening: audit, reliability work, tidying up

### Encoder (`psi_stream pack`)

- **[bytes]** The cap on the passes of the fine level becomes an option, `--fine-passes`, and is 64 by default instead
  of 12. The 3 % bound holds at 512³ in degree 3; it was recorded as missed.
- **[bytes]** Arithmetic margin on the bound, `--tolerance-margin`, 10⁻⁴ u′ by default: marking at
  tolerance − margin, certification at tolerance − margin/2. The 7 % bound holds at 512³ in degree 2; it was recorded
  as missed by 10⁻⁵ u′.
- To reproduce byte for byte an earlier file with a bound: `--fine-passes 12 --tolerance-margin 0`.
- "Plancher" (floor) line and failure message that names the cause of a missed bound: cap on the passes, floor of the
  fine level, or rounding difference larger than half the margin.
- The base field is kept between the marking passes: fine level 36 to 45 % faster, identical files.
- Safeguards: exception index on 31 bits, `--fb` bounded by the actual overflow of the pyramid, 2²⁴ bricks at most.
  `pack` refuses instead of writing a wrong file.

- An invalid or truncated input sequence is refused with a message and code 1. It used to stop `pack` without a
  message, on a crash code.
- The size and the offsets of the structures of the format are asserted at compile time (`static_assert`).

### Reader (`psi_stream play`, `psi_view`)

- The header is checked before any allocation: magic, degree, resolution, number of frames, presence of the table.
- The receive buffers have the size of the largest frame of the file: state of the receiver reduced by 10 to 16 %.
- `play` displays the certification threshold and the margin recorded in the file.

### Defects brought to light, dealt with since

The test suite and the writing of the format specification had brought to light three defects: a header that no
fingerprint protected, a CUDA illegal memory access error on a header geometry inconsistent with the frames, and
`PSISTRM1` files whose lattice the original reader clipped. The first two are fixed and the third is measured and
reported at reading: see the late-evening entry, which keeps the list of known defects.

### Corrected measurements

- Largest error without a bound at 512³: 0.268 u′ in degree 2 and 0.217 u′ in degree 3, on the 55 frames. The
  published values (0.255 and 0.201 u′) covered only 4 frames. `rang5_reglages.sh` no longer passes `--check-every 16`.
- The reference files of `build_psi/garde/` are re-encoded with the current settings; the old ones are in
  `build_psi/garde/avant_audit/`.

### New measurement: a less resolved turbulence

- On a 512³ sequence whose Kolmogorov scale is 0.45 voxel (that of the public JHTDB dataset) instead of 0.74, the
  field error goes from 0.64–0.69 % to 1.5–1.7 %, the bounds of 3 to 5 % are refused in degree 3 and that of 7 % is
  missed in degree 2. `docs/RESULTS.md`, section 10; `results/resolution/`.

### Repository

- **The root now contains only the final project**: `src/`, `tests/`, `scripts/`, `tools/`, `docs/`, `results/`.
  Everything else is kept in `archive/`: `archive/prototypes/` (dense reference of phase 1, EDCF engine with
  micro-networks), `journal/` (dated measurements of the earlier versions), `rapports/`, `notes_de_recherche/`. The
  table of the old paths is in `archive/README.md`.
- The build and measurement scripts move from the root to `scripts/`; they are still launched from the root
  (`.\scripts\build_psi.ps1`).
- `results/` keeps only the measurements of the current version of the code, arranged by result, each with its script.
- The root `CMakeLists.txt` now builds only the retained solution; `.\scripts\build.ps1 -Prototypes` also builds the
  prototypes.
- Additions: `docs/RESULTS.md` (results and provenance), `docs/FORMAT.md` (`.psi` format), `tests/` (regression
  tests at 64³), `tools/environnement.ps1` (log of the measurement environment), `.gitattributes` (LF line endings on
  all machines), `.editorconfig`, this changelog.

## 15 November 2026: roadmap of the committee review

- Rank 1: `pack` counts the clippings per site and per level and exits with code 3 if it clips.
- Rank 3: `play` takes the kernel without cache in sorted degree 3.
- Rank 4: **[bytes]** `PSISTRM2` format. Lattice written modulo 2¹⁶, read directly or by differences; exception lists
  for the innovations and the fine coefficients out of range. `play` still reads `PSISTRM1`.
- Rank 5: **[bytes]** coded key frames (3D Lorenzo then rANS); formula of the quantisation step corrected, `--eps`
  becomes the target quantisation error. Compatibility: `--raw-keys`, `--legacy-step`, `--fit-at 0`.
- Rank 6: `PSISTRM3` format. Sparse fine level of the potential and certified bound, `pack --tolerance`, exit code 4
  if the bound is missed. Bench at equal bound against SZ3, SPERR, zfp and cuSZp.
- Rank 7: `psi_lagr`, Lagrangian experiment on tracers.

## 14 and 15 November 2026: from the micro-network engine to Psi-treillis

- Bench `psi`: vector potential in B-splines, integer coefficients, temporal prediction, rANS. Zero divergence by
  construction.
- Standalone chain `psi_stream` (`pack`, `play`), `PSISTRM1` format.
- cuZFP + half-precision texture reference, full budget of a frame and frontier (`scripts/run_budget.ps1`,
  `tools/frontiere.py`), fidelity against the exact field (`psi --exact`).
- Visual comparator `psi_view`.
- Kernels with a cache of the support, launched in blocks of 128 threads.
- Campaign of the EDCF engine, now in `archive/prototypes/`: it wins only on unsorted accesses, loses in ray casting,
  and its divergence goes from 8 to 35 % of the gradient.
