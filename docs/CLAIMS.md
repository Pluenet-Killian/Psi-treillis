# Psi-treillis claims sheet

What the project asserts, in a form that a third party can refute: a statement, a numerical criterion, the value read in
a log of the repository, the command that reproduces this log, and a verdict. The claims that do not hold appear here on
the same footing as the others, and the hypotheses not yet tested are written here with their criterion before any
measurement.

The notices of `benchmarks/` describe each measurement, what its script writes and its caveats. This sheet keeps only
what the project commits to seeing verified.

## How to read it

- **Three verdicts.** "Holds": the log satisfies the criterion. "Does not hold": it violates it. "Pending": the
  measurement has not been made. A claim whose log is not in the repository, or is not the log that the present script
  writes, gets no verdict: it is listed apart, with the verdict that the registry expects.
- **Three levels of evidence.** The "Measured" lines are read in a log or a CSV of the repository. The ratios between
  two values are calculations on these measurements. The pending claims are hypotheses, each with the test that
  settles it. A figure quoted in a caveat is read in a log too wherever a log carries it; the sentences around it are
  written by hand.
- **Criterion fixed before or after the measurement.** Each criterion says which. Most were written after the
  measurement: they are thresholds that a replay can violate, not predictions. The criteria written before their
  measurement are those of LAGR-1, LAGR-2 and LAGR-4 (before the analysis, with a caveat on the number of tracers); of
  REF-1, REF-2, LAGR-5, VALID-1 to VALID-5, CTRL-1 and LINUX-1 (16 November 2026); of GRID-1, READ-1, READ-2, TRAC-1
  and SPEC-1 (17 November 2026); and those of the pending claims. Only these are predictions. A measurement replayed
  later does not change when its criterion was fixed.
- **What cannot be verified here.** The order of the commits that shows a criterion was written before its measurement
  is in the development history of the project, which is not public: it cannot be verified from this repository.
- **Common scope.** One machine (RTX 5090 Laptop, Windows 11) and one family of data: periodic isotropic turbulence
  simulated here, at 256³ and 512³. No claim that holds is valid outside this setting.
- **Versions.** Every log of `benchmarks/` comes from the code of this repository. There is no archive folder here:
  the measurements made with earlier versions of the code are not published.

## Verifying

```
python tools/claims.py
```

The script reads the logs of the repository again and gives one verdict per claim. It runs no computation and needs
neither a GPU nor a library: Python 3.8 or later, standard library only. It exits with code 1 if a verdict is no longer
the one that the registry expects, if a log is missing or unreadable for a claim registered as measured, if a log does
not have the scope that its criterion states (a number of files or of repetitions, a GPU that was not free for a
timing), or if this sheet is no longer up to date. Its last line is the count: `N claims, M anomalies`.

To put a claim to the test: replay the command of its "Evidence" line, which rewrites the log, then run the script
again. The sizes, the errors and the exit codes are deterministic on a given toolchain; the timings are not.

## Rules

1. A criterion is "fixed before the measurement" only if the commit that introduces it precedes the log.
2. A claim that does not hold stays in the sheet.
3. A criterion modified after the measurement takes a new identifier; the old one stays, with its verdict.
4. A verdict that changes is written in `CHANGELOG.md`.
5. All the trials made for a claim are reported, not only the best.

## Identifiers renamed

The identifiers that came from French words were renamed for this edition. Each claim keeps its statement, its
criterion and its history; every other identifier is unchanged.

| Former identifier | Present identifier |
|---|---|
| BORNE-1 to BORNE-4 | BOUND-1 to BOUND-4 |
| BANC-1 to BANC-4 | BENCH-1 to BENCH-4 |
| LECT-1 | SPEED-1 |
| RELEC-1 | READBACK-1 |
| ROBU-1 | ROBUST-1 |

## Data, settings and instruments of the claims fixed on 17 November 2026

GRID-1, READ-1, READ-2, TRAC-1 and SPEC-1 share the data, the settings and the instruments below, fixed with their
criteria before any of their measurements. The text is that of the pre-registration, word for word, except the names
of the three sequences, which were `v_nu00280_g42.seq`, `v_nu00280_g7.seq` and `v_nu00280_g11.seq` when it was written.

- **Realisations.** The three 256³ sequences of the resolution sweep at viscosity 0.0028 (solver seeds 42, 7 and 11;
  files `data/sweep/nu00280_seed42.seq`, `data/sweep/nu00280_seed7.seq`, `data/sweep/nu00280_seed11.seq`), frame 32.
  Seed 42 is the realisation of `turb256.seq`.
- **Bound.** Every error-bounded compressor receives the same absolute bound on the largest error of a component:
  0.05 u′, u′ being the RMS velocity recorded in the header of the sequence.
- **Chains.** `raw`: the original float32 grid. `sz3`: SZ3, frame by frame. `sz3_4d`: SZ3 on the block of frames 16 to
  47, frame 32 extracted. `sperr`: SPERR, frame by frame. `zfp`: zfp in fixed-accuracy mode. `cuszp`: cuSZp.
  `cuzfp_2bits`: cuZFP at a fixed rate of 2 bits per value. `psi`: Psi-treillis of degree 3 with the bound, group of
  frames 16 to 47. The compressors are called with their default settings, as in the equal-bound bench.
- **Normalisation.** G is the RMS of the velocity gradient of the original frame, sqrt(mean over the grid of the sum
  over i, j of (∂u_i/∂x_j)²), computed spectrally in double precision.
- **M1, divergence on the grid.** Spectral divergence of a grid (sum over j of i k_j u_hat_j, periodic FFT, double
  precision), RMS over the grid, divided by G. It measures what compression alone does: no interpolation is involved.
- **M2, divergence as read.** Analytic divergence of the trilinear interpolant of a grid (values at voxel centres,
  periodic), in double precision, at 4,194,304 points drawn uniformly (NumPy generator `default_rng(20261117)`), RMS,
  divided by G. It measures what a tracer sees when the grid is read by trilinear interpolation, without the rounding
  of a half-precision texture. For Psi-treillis the field read is the analytic curl of the potential: its divergence
  is zero by identity, and claim DIV-1 already measures it in double precision.
- **M3, volume carried by a tracer.** The program of the tracer experiment, unchanged: tracers advected by RK4 in the
  frozen frame, D the integral along the trajectory of the divergence taken by centred differences at ±1/8 voxel of the
  sampler of the chain; growth of the variance of D per Kolmogorov time, that is the variance of D after 10 Kolmogorov
  times divided by 10, mean over the tracer seeds; excess dispersion of the count in bins of 1 voxel after 10
  Kolmogorov times, in standard errors over tracer seeds 1 to 5; 8 tracers per voxel. A decoded grid is read as the raw
  frame is: half-precision 3D texture, hardware trilinear filtering.
- **M4, spectrum.** Energy spectrum by integer shells of the wavenumber, double precision. Enstrophy = sum over shells
  of k² E(k). k10 = the smallest shell where |E_decoded(k) / E_original(k) − 1| exceeds 10 %; "none" if no shell does up
  to the truncation of the solver (n/3).

Names in the logs. In `benchmarks/divergence/divergence.csv` and `benchmarks/spectra/spectra.csv` the chains carry
the names above, and Psi-treillis without a bound is `psi_free`. In the logs of the tracer experiment a chain is a
path: `raw` is `tex`, `cuzfp_2bits` is `zfp2`, `psi` is `psi3b`, and the three compressors keep their names. The
pre-registration also holds GRID-2, which makes no prediction and is therefore not a claim: its measurement is in the
caveat of GRID-1.

<!-- verdicts: start of the part written by tools/claims.py, do not edit by hand -->

## Summary

| Identifier | Claim | Verdict |
|---|---|---|
| COMP-1 | Compression without a bound, degree 3 | holds |
| COMP-2 | Compression without a bound, degree 2 | holds |
| BOUND-1 | Certified bound, degree 3 | holds |
| BOUND-2 | Cost of the bound, degree 3 | holds |
| BOUND-3 | Certified bound, degree 2 | holds |
| READBACK-1 | Read-back consistent with the encoding | holds |
| DIV-1 | Zero divergence | holds |
| SPEED-1 | Reading speed | holds |
| ENC-1 | Encoding speed | holds |
| LAGR-1 | Clustering of tracers under cuZFP | holds |
| LAGR-2 | No clustering under Psi-treillis | holds |
| LAGR-3 | Volume carried by a tracer | holds |
| ROBUST-1 | Reader faced with an altered file | holds |
| TEST-1 | Determinism and non-regression | holds |
| API-1 | Public interface, round trip at 64³ | holds |
| API-2 | Errors read back at 512³ through another code path | holds |
| BENCH-1 | Against SPERR, at an equal bound | holds |
| BENCH-2 | Pareto front at 256³ | holds |
| REF-1 | A reader independent of the code | holds |
| LAGR-4 | Clustering under the trilinear texture | **does not hold** |
| BOUND-4 | Certified bound, degree 2, one point lower | **does not hold** |
| RESO-1 | Fidelity on a less resolved turbulence | **does not hold** |
| RESO-2 | Certified bound on a less resolved turbulence | **does not hold** |
| BENCH-3 | Against SZ3 in 4D, at an equal bound | **does not hold** |
| JHTDB-1 | Expected degradation on JHTDB | **pending** |
| JHTDB-2 | Usefulness for compression on JHTDB | **pending** |
| PORT-1 | Another machine | **pending** |
| LAGR-5 | Lagrangian confirmation on an independent realisation | **does not hold** |
| BENCH-4 | SZ3 in 4D on 32 frames at 512³ | **pending** |
| VALID-1 | Resolution threshold of the bound of 5 % | holds |
| VALID-2 | Ahead of SPERR where the bound holds | **does not hold** |
| VALID-3 | The step of 1 voxel holds the bound everywhere, without catching up on the compression ratio | holds |
| VALID-4 | Field error as a function of the resolution | **does not hold** |
| VALID-5 | Spread between realisations | **does not hold** |
| CTRL-1 | Windowed cutout protocol | holds |
| JHTDB-3 | Degradation on a JHTDB cutout | **pending** |
| JHTDB-4 | Step of 1 voxel on the JHTDB cutout | **pending** |
| REPR-1 | Does our under-resolved sequence stand in for JHTDB? | **pending** |
| REF-2 | A second reader from the completed specification | **does not hold** |
| LINUX-1 | Compilation under Linux | **pending** |
| GRID-1 | Compression alone breaks the law | holds |
| READ-1 | Reading alone breaks the law | holds |
| READ-2 | Compression adds to it | holds |
| TRAC-1 | Tracers after an error-bounded compressor | holds |
| SPEC-1 | Spectrum of Psi-treillis | holds |
| PLAY-1 | Cost of one replayed frame | holds |
| REF-3 | A third reader from the specification completed on 17 November 2026 | **pending** |

47 claims: 28 hold, 10 do not hold, 9 pending.

## Claims that hold

Each one is verified by replaying the command given, then this script.

### COMP-1. Compression without a bound, degree 3

- **Claim.** At 512³, with the default settings in degree 3, Psi-treillis compresses at least 140 times for a field error of at most 0.65 %.
- **Criterion.** Compression ratio ≥ 140 and mean field error ≤ 0.65 %, on frames 0 to 54 of `turb512.seq`. *Fixed after the measurement.*
- **Measured.** 143.2× at a field error of 0.640 %, over 55 frames.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, line `z_t512_d3` (output of the encoder: `benchmarks/compression/reference_files/logs/pack_z_t512_d3.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** The compression ratio is that of a file of 55 frames with a key frame every 32 frames: it is inter-frame.

### COMP-2. Compression without a bound, degree 2

- **Claim.** At 512³, with the default settings in degree 2, Psi-treillis compresses at least 115 times for a field error of at most 0.70 %.
- **Criterion.** Compression ratio ≥ 115 and mean field error ≤ 0.70 %, on frames 0 to 54 of `turb512.seq`. *Fixed after the measurement.*
- **Measured.** 118.8× at a field error of 0.690 %, over 55 frames.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, line `z_t512_d2` (output of the encoder: `benchmarks/compression/reference_files/logs/pack_z_t512_d2.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.

### BOUND-1. Certified bound, degree 3

- **Claim.** In degree 3, `pack --tolerance` holds a bound of 5, 4 and 3 % of u′ on the largest error of a component, at 256³ and at 512³.
- **Criterion.** For each of the six files: no frame above the certification threshold, checked on all the voxels of all the frames. *Fixed after the measurement.*
- **Measured.** 6 of 6 files below the certification threshold. Largest error, as a fraction of u′: 256³ at 5 %: 4.990 %; 256³ at 4 %: 3.990 %; 256³ at 3 %: 2.990 %; 512³ at 5 %: 4.990 %; 512³ at 4 %: 3.990 %; 512³ at 3 %: 2.990 %.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, lines `f_t256_d3_t0*` and `f_t512_d3_t0*` (outputs of the encoder: `benchmarks/compression/reference_files/logs/pack_f_t*_d3_t0*.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** The bound holds at voxel centres and at the times of the frames; between frames and off the grid, nothing is guaranteed. The certification margin is empirical.

### BOUND-2. Cost of the bound, degree 3

- **Claim.** With this bound, the compression ratio remains at least 130 times at 5 %, 120 times at 4 % and 95 times at 3 %.
- **Criterion.** Compression ratio of the whole file ≥ 130, 120 and 95 according to the tolerance, at 256³ and at 512³. *Fixed after the measurement.*
- **Measured.** 256³ at 5 %: 134.1×; 256³ at 4 %: 127.3×; 256³ at 3 %: 105.1×; 512³ at 5 %: 136.6×; 512³ at 4 %: 126.5×; 512³ at 3 %: 100.1×.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, lines `f_t256_d3_t0*` and `f_t512_d3_t0*` (outputs of the encoder: `benchmarks/compression/reference_files/logs/pack_f_t*_d3_t0*.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.

### BOUND-3. Certified bound, degree 2

- **Claim.** In degree 2, the bound is held at 6 % of u′ at 256³ and at 7 % at 512³.
- **Criterion.** No frame above the certification threshold in the two files. *Fixed after the measurement.*
- **Measured.** 2 of 2 files below the certification threshold. Largest error, as a fraction of u′: 256³ at 6 %: 5.990 %; 512³ at 7 %: 6.991 %.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, lines `f_t256_d2_t06` and `f_t512_d2_t07` (outputs of the encoder: `benchmarks/compression/reference_files/logs/pack_f_t256_d2_t06.log`, `pack_f_t512_d2_t07.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** See BOUND-4: one point lower, it is missed.

### READBACK-1. Read-back consistent with the encoding

- **Claim.** The reader gives back the state of the encoder: no frame differs, and the same field off the grid.
- **Criterion.** On the twelve reference files: 0 frames whose state differs; on the ten files with a bound: zero difference at the off-grid probe (114,688 points). *Fixed after the measurement.*
- **Measured.** 12 files read back, 0 frames whose state differs; off-grid probe on 10 files (114,688 points), largest difference 0.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, columns `frames_differing` and `probe_largest_difference` (outputs of the reader: `benchmarks/compression/reference_files/logs/play_*.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** A rounding difference between reader and encoder exists on the less resolved sequence, in degree 2: 1.0·10⁻⁵ for `d2_t07` and 1.0·10⁻⁵ for `probe_nu0006_d2`, in velocity units, u′ being 0.850 on that sequence (`benchmarks/compression/under_resolved/under_resolved.csv`, column `probe_largest_difference`). It is not explained.

### DIV-1. Zero divergence

- **Claim.** The field read back has zero divergence, up to rounding.
- **Criterion.** RMS divergence of the analytic gradient, in double precision, ≤ 10⁻¹² of the RMS gradient. *Fixed after the measurement.*
- **Measured.** 9.4·10⁻¹⁶ of the RMS gradient at most, on the 4 sides of the log that show Psi-treillis (256³ and 512³).
- **Verdict: holds.**
- **Evidence.** `benchmarks/playback/playback.csv`, column `divergence_over_gradient` of the sides that show Psi-treillis (outputs of the comparator: `benchmarks/playback/logs/view_*.log`); to reproduce: `python benchmarks/playback/playback.py run` (GPU running nothing else; `psi_view` exists on Windows only).
- **Caveat.** Measured on a slice of one frame. By finite differences, one measures the truncation of the scheme, not the field. Since 17 November 2026 the figure is read in the log of the playback measurement, on files of degree 3 without a bound, at 256³ and at 512³; the earlier log was one capture of the visual comparator, which showed a file without a bound and a file with a bound of 3 %: a file with a bound is no longer among those measured here.

### SPEED-1. Reading speed

- **Claim.** At 512³ in degree 3, on the measurement machine, a frame is decoded in less than 4 ms and one RK4 step of 4 million sorted particles takes less than 4.5 ms, without a bound and with a bound of 5 %.
- **Criterion.** Medians of 15 repetitions of the current reader: decoding ≤ 4.0 ms and RK4 step ≤ 4.5 ms, on both files. *Fixed after the measurement.*
- **Measured.** `f_t512_d3_t05.psi`: decoding 3.93 ms, RK4 step 4.20 ms (medians of 15 repetitions); `z_t512_d3.psi`: decoding 3.74 ms, RK4 step 4.01 ms (medians of 15 repetitions).
- **Verdict: holds.**
- **Evidence.** `benchmarks/speed/read_timing.csv`; to reproduce: `python benchmarks/speed/read_timing.py` (see `benchmarks/speed/README.md`).
- **Caveat.** A single machine (RTX 5090 Laptop), GPU running no other computation, system cache warm. Degree 2 has not been measured again with the current reader.

### ENC-1. Encoding speed

- **Claim.** At 512³ in degree 3, on the measurement machine, encoding takes less than 300 ms per frame with a bound of 5 % and less than 450 ms with a bound of 3 %.
- **Criterion.** Encoding time per frame ≤ 300 ms and ≤ 450 ms, excluding the reading of the raw frames from disk. *Fixed after the measurement.*
- **Measured.** 512³ at 5 %: 275 ms per frame; 512³ at 3 %: 435 ms per frame.
- **Verdict: holds.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, lines `f_t512_d3_t05` and `f_t512_d3_t03`, column `encoding_ms_per_frame` (outputs of the encoder: `benchmarks/compression/reference_files/logs/pack_f_t512_d3_t05.log`, `pack_f_t512_d3_t03.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** A single run per case, a single machine.

### LAGR-1. Clustering of tracers under cuZFP

- **Claim.** Tracers advected in a frozen frame seen through cuZFP at 2 bits cluster measurably.
- **Criterion.** Excess dispersion of the count in bins of 1 voxel, after 10 Kolmogorov times, seeds 1 to 5: more than 5 standard errors, at 256³ and at 512³. *Fixed before the analysis (at the top of `benchmarks/tracers/tracers.py`), except the number of tracers per voxel at 256³, raised from 1 to 8 after a first trial.*
- **Measured.** In standard errors. cuZFP 2 bits + texture, 256³: +31.7; cuZFP 2 bits + texture, 512³: +13.1.
- **Verdict: holds.**
- **Evidence.** `benchmarks/tracers/resolved_256_summary.csv`, `resolved_512_summary.csv` (lines of the tracer program: `resolved_256.csv`, `resolved_512.csv`; tables: `resolved_256_analysis.txt`, `resolved_512_analysis.txt`); to reproduce: `python benchmarks/tracers/tracers.py run resolved_256 resolved_512` (GPU; it runs the campaigns, then writes `<campaign>_analysis.txt` and `<campaign>_summary.csv`).
- **Caveat.** Frozen field, one frame per resolution, tracers without inertia. The campaigns replayed on 17 November 2026 count bins of 1 voxel directly at 256³; the first campaign counted bins of half a voxel and merged them: the count per bin of 1 voxel is the same quantity.

### LAGR-2. No clustering under Psi-treillis

- **Claim.** In the same protocol, no path of Psi-treillis shows clustering.
- **Criterion.** Excess dispersion below 2 standard errors for each Ψ path measured on five seeds. *Fixed before the analysis (at the top of `benchmarks/tracers/tracers.py`), except the number of tracers per voxel at 256³, raised from 1 to 8 after a first trial.*
- **Measured.** In standard errors. Ψ degree 3, 256³: -0.7; Ψ degree 2, 256³: +0.8; Ψ degree 3, bound 5 %, 256³: -0.7; Ψ reference (step 1 voxel, eps 0.002), 256³: +1.3; Ψ degree 3, 512³: +0.5.
- **Verdict: holds.**
- **Evidence.** `benchmarks/tracers/resolved_256_summary.csv`, `resolved_512_summary.csv` (lines of the tracer program: `resolved_256.csv`, `resolved_512.csv`; tables: `resolved_256_analysis.txt`, `resolved_512_analysis.txt`); to reproduce: `python benchmarks/tracers/tracers.py run resolved_256 resolved_512` (GPU; it runs the campaigns, then writes `<campaign>_analysis.txt` and `<campaign>_summary.csv`).
- **Caveat.** See LAGR-4: the trilinear texture does not pass the threshold of 5 either, so this protocol does not separate Psi-treillis from the texture. The campaigns replayed on 17 November 2026 count bins of 1 voxel directly at 256³; the first campaign counted bins of half a voxel and merged them: the count per bin of 1 voxel is the same quantity.

### LAGR-3. Volume carried by a tracer

- **Claim.** Along the trajectories, the variance of D (integral of the divergence, that is the logarithm of the volume carried) grows at least 1,000 times more slowly with Psi-treillis of degree 3 than with the trilinear texture or cuZFP at 2 bits.
- **Criterion.** Growth of var(D) per Kolmogorov time: that of the texture and that of cuZFP ≥ 1,000 times that of Psi-treillis, at 256³ and at 512³. *Fixed after the measurement.*
- **Measured.** Per Kolmogorov time. 256³: Ψ 1.9·10⁻⁸, texture 9.1·10⁻⁵ (4,790 times), cuZFP 2.6·10⁻⁴ (13,532 times); 512³: Ψ 1.9·10⁻⁸, texture 9.2·10⁻⁵ (4,761 times), cuZFP 2.8·10⁻⁴ (14,321 times).
- **Verdict: holds.**
- **Evidence.** `benchmarks/tracers/resolved_256_summary.csv`, `resolved_512_summary.csv` (lines of the tracer program: `resolved_256.csv`, `resolved_512.csv`; tables: `resolved_256_analysis.txt`, `resolved_512_analysis.txt`); to reproduce: `python benchmarks/tracers/tracers.py run resolved_256 resolved_512` (GPU; it runs the campaigns, then writes `<campaign>_analysis.txt` and `<campaign>_summary.csv`).
- **Caveat.** For Psi-treillis, D measures only the truncation of the probe by differences (±1/8 voxel); for the texture, the value depends on this offset (4.9·10⁻⁵ to 1.2·10⁻⁴ at ±1/4 and ±1/16 voxel, campaign `offsets_256`, `benchmarks/tracers/offsets_256_summary.csv`). Descriptive, on one seed, 1 tracer per voxel and 100 Kolmogorov times (campaign `long_256`): Ψ 2.0·10⁻⁸, texture 9.2·10⁻⁵ (4,659 times), cuZFP 2.6·10⁻⁴ (13,240 times).

### ROBUST-1. Reader faced with an altered file

- **Claim.** An altered file is refused, or read while counting the frames that differ; it causes neither a CUDA error, nor a crash, nor a hang.
- **Criterion.** On the 283 altered files at 64³, the only outcomes of the current reader are "refused", "counted" and "read". *Fixed after the measurement.*
- **Measured.** 283 altered files read back: 106 counted, 42 read, 135 refused.
- **Verdict: holds.**
- **Evidence.** `benchmarks/robustness/altered_files.csv` (outputs of the reader: `benchmarks/robustness/logs/altered_files_*.log`); to reproduce: `python tests/test_codec.py --keep-alterations --only "guard.geometry_no_seal" "guard.random_alteration"`, then `python benchmarks/robustness/altered_files.py`.
- **Caveat.** 64³ only, one alteration at a time. Four alterations of the header or of the table of a file without a seal are not seen, nor is the altered value of an overflow of level 0 (case `guard.altered_overflow`).

### TEST-1. Determinism and non-regression

- **Claim.** The same options give the same bytes, and the regression suite at 64³ passes.
- **Criterion.** `tests/test_codec.py`: no `FAIL`; cases `d2.determinism`, `d3.determinism` and `sequence.fingerprint` marked `OK`. *Fixed after the measurement.*
- **Measured.** 50 cases, 49 OK, 0 failed, 1 known defect; determinism in degree 2 and 3: OK; 16 fingerprint cases OK.
- **Verdict: holds.**
- **Evidence.** `benchmarks/interface/test_logs/test_codec_64.log`; to reproduce: `python tests/test_codec.py > benchmarks/interface/test_logs/test_codec_64.log`.
- **Caveat.** The fingerprints are valid only for the toolchain described in `tests/fingerprints.json`. See PORT-1.

### API-1. Public interface, round trip at 64³

- **Claim.** Through the public interface (library, Python module, `psit` tool), the compressed file is that of `pack` byte for byte, the decoded field respects the requested bound, off-grid sampling returns a divergence-free field and the GPU memory is released on closing.
- **Criterion.** `tests/test_api.py`: no failed case; cases `cli.same_bytes_as_pack`, `api.bound_is_held`, `api.sample_matches_the_grid`, `api.divergence_free` and `api.memory_is_released` passed. *Fixed after the measurement.*
- **Measured.** 16 cases, 16 passed, 0 failed; the five required cases: passed; largest error 3.9881 % of u′ for a bound of 4 %; divergence 1.9·10⁻⁴ of the gradient, by differences.
- **Verdict: holds.**
- **Evidence.** `benchmarks/interface/test_logs/test_api_64.log`; to reproduce: `python tests/test_api.py > benchmarks/interface/test_logs/test_api_64.log`.
- **Caveat.** 64³ and Windows only. The errors are measured by the test in NumPy, outside the code of the project. The "bound missed" exit (code 4) cannot be reached at 64³ through the public options: it is not covered.

### API-2. Errors read back at 512³ through another code path

- **Claim.** At 512³, the field decoded to the grid by the library, compared in NumPy with the raw frames, has the error that the encoder recorded in the file, and it is within the tolerance of each file whose bound is held.
- **Criterion.** On four frames of six reference files: largest error less than 5·10⁻⁵ u′ from the recorded value, field error to within 0.02 %, and largest error ≤ tolerance. *Fixed after the measurement.*
- **Measured.** 24 frames of 6 files, 0 non-conforming; with a bound, largest error at 99.86 % of the tolerance at most.
- **Verdict: holds.**
- **Evidence.** `benchmarks/interface/bounds_512/verify_bounds_512.csv` (the same lines, with the version of the library: `verify_bounds_512.log`); to reproduce: `python benchmarks/interface/bounds_512/verify_bounds_512.py`.
- **Caveat.** Four frames per file out of 55; the encoder, for its part, checks all the frames (BOUND-1). The kernel that decodes to the grid is that of the library; the comparison is made on the host in double precision.

### BENCH-1. Against SPERR, at an equal bound

- **Claim.** At an equal bound on the largest error, Psi-treillis of degree 3 compresses at least 15 % more than SPERR, with no higher field error.
- **Criterion.** Compression ratio ≥ 1.15 times that of SPERR and field error ≤ 1.01 times that of SPERR, at 5, 4 and 3 % (256³) and at 5 and 4 % (512³). *Fixed after the measurement.*
- **Measured.** 256³ at 5 %: 133.7× against 109.2× (1.22 times), field error 0.614 % against 0.831 %; 256³ at 4 %: 126.7× against 93.6× (1.35 times), field error 0.604 % against 0.709 %; 256³ at 3 %: 104.0× against 76.9× (1.35 times), field error 0.582 % against 0.578 %; 512³ at 5 %: 137.6× against 117.1× (1.17 times), field error 0.623 % against 0.818 %; 512³ at 4 %: 127.4× against 100.0× (1.27 times), field error 0.609 % against 0.700 %.
- **Verdict: holds.**
- **Evidence.** `benchmarks/equal_bound/equal_bound_256.csv`, `equal_bound_512.csv`; to reproduce: `python benchmarks/equal_bound/equal_bound.py 256` and `512` (GPU; needs the comparison codecs of `build/codecs/`).
- **Caveat.** Psi-treillis is measured on a group of 32 frames, with temporal prediction; SPERR is measured frame by frame: the lead is that of temporal prediction. Protocol corrected before the re-measurement of 17 November 2026, the criterion being kept as it was: every codec is measured on the same 32 frames (frames 16 to 47 of `turb256.seq`, frames 0 to 31 of `turb512.seq`), where the frame-by-frame codecs were measured on four frames at 256³ and two at 512³, one of them outside the group of Psi-treillis; their compression ratio is the total raw size over the total compressed size, where it was the mean of the ratios of the frames measured; the error of Psi-treillis is no longer the one its encoder reports: its file is decoded through the public interface and compared with the raw frames by the same code as the other codecs; at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames (63 GiB of RAM do not hold a block of 32); zfp is reported apart, its fixed-accuracy mode leaving the largest error well under the bound requested.

### BENCH-2. Pareto front at 256³

- **Claim.** At 256³, the three default points of Psi-treillis of degree 3 (bounds of 5, 4 and 3 %) are not dominated by any measured point of SZ3, SZ3 in 4D, SPERR, zfp or cuSZp.
- **Criterion.** No point of another codec has both a compression ratio at least equal and a field error at most equal. *Fixed after the measurement.*
- **Measured.** 3 of 3 points not dominated, among 20 points of the other codecs.
- **Verdict: holds.**
- **Evidence.** `benchmarks/equal_bound/equal_bound_256.csv`; to reproduce: `python benchmarks/equal_bound/equal_bound.py 256` and `512` (GPU; needs the comparison codecs of `build/codecs/`).
- **Caveat.** The window is narrow: field error from 0.58 to 0.61 %. The largest error does not enter this criterion: with a bound of 1.5 %, SZ3 in 4D has a field error of 0.60 %, for a compression ratio of 106.1×. The points of zfp are among those of the other codecs, as the statement names it, although it is not at an equal attained bound: asked for 5, 4 and 3 %, its largest error is 1.15, 0.62, 0.62 % of u′. Protocol corrected before the re-measurement of 17 November 2026, the criterion being kept as it was: every codec is measured on the same 32 frames (frames 16 to 47 of `turb256.seq`, frames 0 to 31 of `turb512.seq`), where the frame-by-frame codecs were measured on four frames at 256³ and two at 512³, one of them outside the group of Psi-treillis; their compression ratio is the total raw size over the total compressed size, where it was the mean of the ratios of the frames measured; the error of Psi-treillis is no longer the one its encoder reports: its file is decoded through the public interface and compared with the raw frames by the same code as the other codecs; at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames (63 GiB of RAM do not hold a block of 32); zfp is reported apart, its fixed-accuracy mode leaving the largest error well under the bound requested.

### REF-1. A reader independent of the code

- **Claim.** A `.psi` file can be read from its specification alone.
- **Criterion.** A CPU reader written from `docs/FORMAT.md`, without reading `src/`, gives back for each frame of the twelve reference files the state fingerprint recorded in the table of the file, and a velocity field less than 5·10⁻⁵ u′ from that of the GPU reader at the points of the probe. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** 12 files, 622 of 622 frames read back, 0 fingerprints different from the one in the table; at the probe of each file (114,688 points, one frame): 0 nodes of the lattice or of the fine level different from those of the encoder, and a velocity field at most 2.8·10⁻⁵ u′ from that of the GPU reader.
- **Verdict: holds.**
- **Evidence.** `benchmarks/independent_readers/cpp/reference_cpp.csv` (reader: `tools/reference_reader.cpp`, program `psi_ref`); to reproduce: `python benchmarks/independent_readers/cpp/reference_cpp.py`.
- **Caveat.** The statement holds in full only for the state of the frames, decoded from the specification alone. For the velocity field, the specification of the time was not sufficient: the masks and the rounding of the refinement, the position of the nodes and the format of the probe were established by trial on files, without reading the code (list of the gaps: `benchmarks/independent_readers/cpp/README.md`). They were then carried over into `docs/FORMAT.md` (sections 6.5, 6.7 and 11), then cross-checked against the code, with which they agree; no reader had yet been written from the text thus completed when this claim was first measured: that reader is the one of REF-2. A single frame probed per file. The GPU reader is compared only through the probe of the encoder, from which it is at 0. The difference grows with the size of the grid and is more than half the threshold at 512³; it is attributed to the single precision of the GPU reader, without proof. The threshold is half the certification margin; the largest rounding disagreement recorded between two GPU kernels is 2.3·10⁻⁵ u′.

### VALID-1. Resolution threshold of the bound of 5 %

- **Claim.** At a step of 2 voxels in degree 3, holding a bound of 5 % of u′ depends only on the resolution of the data: the bound is held above a threshold of the Kolmogorov scale, missed or refused below it, and this threshold is between 0.45 and 0.74 voxel.
- **Criterion.** Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), frames 8 to 43. There is a resolution η* such that `pack --degree 3 --tolerance 0.05` exits with code 0 on all the sequences with η > η* and with a non-zero code on all those with η < η*; η*, the midpoint between the largest resolution that fails and the smallest that succeeds, is between 0.45 and 0.74 voxel. Refuted if the two sets overlap, if one of them is empty, or if η* falls outside the interval. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** Bound held on 8 sequences, from 0.543 to 0.795 voxel per Kolmogorov scale; missed or refused on 10, from 0.363 to 0.530 voxel; η* = 0.536 voxel.
- **Verdict: holds.**
- **Evidence.** `benchmarks/validity/validity.csv` (output of each run in `benchmarks/validity/logs/`); to reproduce: `python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned).
- **Caveat.** The threshold is that of this sweep: 256³, one solver, lattice step of 2 voxels, bound of 5 %. The three realisations of one viscosity (0.0019) fall on both sides of it: the bound is refused at 0.530 voxel and held at 0.543 and 0.552. For a bound of 3 %, the same sweep puts the threshold between 0.643 and 0.702 voxel: refused on 15 sequences, held on 3. The Kolmogorov scale is that of the header of each sequence, computed by the solver from the dissipation of the run.

### VALID-3. The step of 1 voxel holds the bound everywhere, without catching up on the compression ratio

- **Claim.** At a step of 1 voxel in degree 3, the bound of 5 % is held whatever the resolution of the sweep, but the compression ratio then remains below that of SPERR at the same bound.
- **Criterion.** Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), frames 8 to 43. On the 18 sequences: `pack --degree 3 --mf 1 --tolerance 0.05` exits with code 0, and its compression ratio for the group of 32 frames is lower than that of SPERR at the same bound. Refuted if one sequence misses the bound, or if the compression ratio reaches that of SPERR on one sequence. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** Bound held on 18 of 18 sequences; compression ratio from 0.46 times that of SPERR (32.4× against 70.5×, at 0.614 voxel per Kolmogorov scale) to 0.70 times (19.3× against 27.6×, at 0.363 voxel).
- **Verdict: holds.**
- **Evidence.** `benchmarks/validity/validity.csv` (output of each run in `benchmarks/validity/logs/`); to reproduce: `python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned).
- **Caveat.** This claim holds, and it is not good news for the codec: a step of 1 voxel keeps the bound on every sequence, down to 0.36 voxel per Kolmogorov scale, but at about half the compression ratio of SPERR. Its field error is then 0.39 to 0.55 %, lower than that of SPERR at the same bound.

### CTRL-1. Windowed cutout protocol

- **Claim.** The error of the codec measured at the centre of a windowed cutout, not periodic originally, is the one it makes on the whole field.
- **Criterion.** On `turb512.seq` and `turb512_nu0006.seq`: the 256³ cutout of frames 0 to 7, multiplied by a smooth window (1 at the centre, half-cosine over 32 voxels at each edge) then projected onto the divergence-free fields, is compressed in degree 3 at a step of 2 voxels with the u′ of the sequence. Its relative RMS error on the central cube of 160³ equals, to within 10 %, that of the whole field at 512³ compressed with the same settings on the same frames, measured on the same voxels. True for both sequences. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** `turb512.seq`: 0.745 % at the centre of the windowed cutout against 0.702 % on the same voxels of the whole field, that is 1.062 times; `turb512_nu0006.seq`: 1.763 % at the centre of the windowed cutout against 1.619 % on the same voxels of the whole field, that is 1.089 times.
- **Verdict: holds.**
- **Evidence.** `benchmarks/cutout/control.csv` (output of each encoding in `benchmarks/cutout/logs/`); to reproduce: `python benchmarks/cutout/cutout.py control` (GPU, the two sequences at 512³).
- **Caveat.** The protocol overestimates the error, by 6 % on one sequence and by 9 % on the other: the correction made by the projection is smooth at the centre, but not zero. The bound of 5 % is held or refused in the same way on the whole field and on the cutout, for both sequences. The compression ratio, for its part, does not carry over: the windowed cutout compresses 150 times against 126 for the whole field at 0.74 voxel per Kolmogorov scale, its edges being almost zero.

### GRID-1. Compression alone breaks the law

- **Claim.** Compression alone breaks the law.
- **Criterion.** For each of `sz3`, `sz3_4d` and `sperr`, on each of the three realisations: M1 of the decoded grid ≥ 1·10⁻², while M1 of the original grid ≤ 1·10⁻⁵. Refuted if one of the nine decoded grids is below 1·10⁻², or if one original grid is above 1·10⁻⁵. `zfp`, `cuszp` and `cuzfp_2bits` are measured and reported without a criterion: I do not know where they fall. *Fixed on 17 November 2026, before any measurement.*
- **Measured.** M1 of the original grid: 1.8·10⁻⁶, 1.7·10⁻⁶, 2.0·10⁻⁶; of the decoded grid: `sz3` 1.1·10⁻¹, 1.0·10⁻¹, 1.1·10⁻¹; `sz3_4d` 1.1·10⁻¹, 1.1·10⁻¹, 1.1·10⁻¹; `sperr` 5.8·10⁻², 5.8·10⁻², 6.0·10⁻² (realisations of solver seeds 42, 7 and 11, in this order).
- **Verdict: holds.**
- **Evidence.** `benchmarks/divergence/divergence.csv`, column `grid_divergence_over_gradient`; to reproduce: `python benchmarks/divergence/divergence.py run` (GPU for the chains decoded there; needs the comparison codecs of `build/codecs/` and the three sequences `data/sweep/nu00280_seed<seed>.seq`).
- **Caveat.** The law is the zero divergence of the velocity field. The realisations, the bound, the chains and the instruments M1 to M4 are those of the section "Data, settings and instruments of the claims fixed on 17 November 2026" of this sheet. Measured without a criterion, M1 of the decoded grid: `zfp` 1.6·10⁻², 1.4·10⁻², 1.4·10⁻²; `cuszp` 3.8·10⁻¹, 3.5·10⁻¹, 3.3·10⁻¹; `cuzfp_2bits` 5.0·10⁻², 5.6·10⁻², 6.1·10⁻². GRID-2 of the same pre-registration makes no prediction and is not a claim: the grid decoded by the library samples a divergence-free field at voxel centres, and its spectral divergence M1 is not zero by construction; measured, it is 2.1·10⁻³, 2.8·10⁻³, 3.0·10⁻³ for Psi-treillis with the bound (`psi`) and 1.8·10⁻³, 2.0·10⁻³, 2.2·10⁻³ without a bound (`psi_free` in the log). One frame per realisation, one viscosity, 256³. The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written.

### READ-1. Reading alone breaks the law

- **Claim.** Reading alone breaks the law.
- **Criterion.** M2 of the original grid ≥ 1·10⁻² on each of the three realisations. *Fixed on 17 November 2026, before any measurement.*
- **Measured.** M2 of the original grid: 4.1·10⁻², 4.3·10⁻², 4.6·10⁻² (realisations of solver seeds 42, 7 and 11, in this order).
- **Verdict: holds.**
- **Evidence.** `benchmarks/divergence/divergence.csv`, column `read_divergence_over_gradient` of the chain `raw`; to reproduce: `python benchmarks/divergence/divergence.py run` (GPU for the chains decoded there; needs the comparison codecs of `build/codecs/` and the three sequences `data/sweep/nu00280_seed<seed>.seq`).
- **Caveat.** The law is the zero divergence of the velocity field. The realisations, the bound, the chains and the instruments M1 to M4 are those of the section "Data, settings and instruments of the claims fixed on 17 November 2026" of this sheet. M2 is computed in double precision on the trilinear interpolant: it leaves out the rounding of a half-precision texture. For Psi-treillis the field read is the analytic curl of the potential: its divergence is zero by identity, and DIV-1 measures it. One frame per realisation, one viscosity, 256³. The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written.

### READ-2. Compression adds to it

- **Claim.** Compression adds to it.
- **Criterion.** For each of `sz3`, `sz3_4d` and `sperr`, on each of the three realisations: M2 of the decoded grid ≥ 1.1 times M2 of the original grid. *Fixed on 17 November 2026, before any measurement.*
- **Measured.** M2 of the original grid: 4.1·10⁻², 4.3·10⁻², 4.6·10⁻²; of the decoded grid: `sz3` 9.2·10⁻², 8.7·10⁻², 8.9·10⁻², that is 2.23, 2.01, 1.92 times; `sz3_4d` 9.2·10⁻², 9.0·10⁻², 8.8·10⁻², that is 2.23, 2.08, 1.91 times; `sperr` 6.0·10⁻², 6.0·10⁻², 6.3·10⁻², that is 1.44, 1.40, 1.37 times (realisations of solver seeds 42, 7 and 11, in this order).
- **Verdict: holds.**
- **Evidence.** `benchmarks/divergence/divergence.csv`, column `read_divergence_over_gradient`; to reproduce: `python benchmarks/divergence/divergence.py run` (GPU for the chains decoded there; needs the comparison codecs of `build/codecs/` and the three sequences `data/sweep/nu00280_seed<seed>.seq`).
- **Caveat.** "It" is the divergence that reading alone gives (READ-1). The realisations, the bound, the chains and the instruments M1 to M4 are those of the section "Data, settings and instruments of the claims fixed on 17 November 2026" of this sheet. Measured without a criterion, M2 of the decoded grid: `zfp` 4.3·10⁻², 4.4·10⁻², 4.7·10⁻²; `cuszp` 2.0·10⁻¹, 1.9·10⁻¹, 1.8·10⁻¹; `cuzfp_2bits` 5.2·10⁻², 5.5·10⁻², 5.9·10⁻². One frame per realisation, one viscosity, 256³. The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written.

### TRAC-1. Tracers after an error-bounded compressor

- **Claim.** Tracers after an error-bounded compressor.
- **Criterion.** On the realisation of seed 42, for each of `sz3`, `sz3_4d` and `sperr`: the growth of the variance of D (M3) is at least that of `raw`, and at least 1,000 times that of `psi`. The excess dispersion of these three chains is measured and reported without a criterion: I do not know whether it passes the threshold of 5 standard errors that `cuzfp_2bits` passes and `raw` does not. *Fixed on 17 November 2026, before any measurement.*
- **Measured.** Growth of the variance of D per Kolmogorov time: `raw` 9.0·10⁻⁵, `psi` 1.9·10⁻⁸; `sz3` 2.4·10⁻³ (26.4 times that of `raw`, 124,864 times that of `psi`); `sz3_4d` 2.4·10⁻³ (26.4 times that of `raw`, 124,947 times that of `psi`); `sperr` 6.3·10⁻⁴ (7.0 times that of `raw`, 33,240 times that of `psi`).
- **Verdict: holds.**
- **Evidence.** `benchmarks/tracers/compressors_256_summary.csv` (lines of the tracer program: `compressors_256.csv`; tables: `compressors_256_analysis.txt`); to reproduce: `python benchmarks/tracers/tracers.py run compressors_256` (GPU; needs the comparison codecs of `build/codecs/` and `data/sweep/nu00280_seed42.seq`).
- **Caveat.** The label of the pre-registration is not a sentence; the hypothesis is the first sentence of the criterion. The realisations, the bound, the chains and the instruments M1 to M4 are those of the section "Data, settings and instruments of the claims fixed on 17 November 2026" of this sheet. In the log, `raw` is the path `tex`, `psi` the path `psi3b` and `cuzfp_2bits` the path `zfp2`. The path `psi3b` is not the chain `psi` exactly as the pre-registration defines it: the tracer program encodes frame 32 alone, as a key frame, with the bound of 5 %, and samples the potential analytically; it does not extract frame 32 from the group of frames 16 to 47. Measured without a criterion, the excess dispersion in bins of 1 voxel after 10 Kolmogorov times, in standard errors over tracer seeds 1 to 5: `sz3` +671.6; `sz3_4d` +720.3; `sperr` +107.2; `cuzfp_2bits` +32.8; `raw` +3.2; `psi` +0.4. The growth of the variance of D of `cuzfp_2bits`, outside the criterion: 2.5·10⁻⁴. Frozen field, one frame, one realisation, tracers without inertia. The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written.

### SPEC-1. Spectrum of Psi-treillis

- **Claim.** Spectrum of Psi-treillis.
- **Criterion.** On each of the three realisations, for `psi` without a bound (degree 3, default settings): the enstrophy of the decoded grid is within 5 % of that of the original, and k10 is at least half the truncation wavenumber of the solver. The other chains are measured and reported without a criterion. *Fixed on 17 November 2026, before any measurement.*
- **Measured.** Enstrophy of the decoded grid over that of the original: 1.0022, 1.0009, 1.0003; k10: 74, 79, 81, for a truncation wavenumber of 85 (realisations of solver seeds 42, 7 and 11, in this order).
- **Verdict: holds.**
- **Evidence.** `benchmarks/spectra/spectra.csv`, lines of the chain `psi_free` (energy of every shell: `benchmarks/spectra/shells.csv`); to reproduce: `python benchmarks/spectra/spectra.py run` (needs the comparison codecs of `build/codecs/` and the three sequences `data/sweep/nu00280_seed<seed>.seq`).
- **Caveat.** The label of the pre-registration is not a sentence; the hypothesis is the first sentence of the criterion. The realisations, the bound, the chains and the instruments M1 to M4 are those of the section "Data, settings and instruments of the claims fixed on 17 November 2026" of this sheet. In the log, `psi` without a bound is the chain `psi_free`, and the truncation wavenumber is the integer part of n/3. Measured without a criterion, enstrophy of the decoded grid over that of the original and k10, for each realisation: `sz3` 1.0163 and 56, 1.0164 and 60, 1.0181 and 62; `sz3_4d` 1.0163 and 56, 1.0177 and 60, 1.0161 and 64; `sperr` 1.0067 and 69, 1.0065 and 73, 1.0070 and 75; `zfp` 1.0007 and none, 1.0006 and none, 1.0005 and none; `cuszp` 1.4426 and 52, 1.3768 and 55, 1.3297 and 57; `psi` 1.0024 and 73, 1.0018 and 78, 1.0013 and 80; `cuzfp_2bits` 1.0077 and 75, 1.0095 and 78, 1.0112 and 78. One frame per realisation, one viscosity, 256³: the spectrum of the grid decoded at voxel centres, not that of the field sampled off the grid. The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written.

### PLAY-1. Cost of one replayed frame

- **Claim.** Replayed frame after frame on the GPU, Psi-treillis costs more time per frame than cuZFP at 256³; at 512³ it reads far less data than raw frames and holds less GPU memory.
- **Criterion.** One protocol and one run of the visual comparator per grid size and per pair of chains, sorted particles, GPU running no other computation: data read per frame, GPU memory, cost of one frame, for raw frames, cuZFP at 2 bits and Psi-treillis of degree 3 without a bound. At 256³ the frame of Psi-treillis costs more time than that of cuZFP; at 512³ Psi-treillis reads at least 100 times less data than the raw chain and uses at least 5 times less GPU memory. *Fixed after the measurement of 16 November 2026; protocol fixed before the re-measurement.*
- **Measured.** 256³: one frame costs 14.21 ms with Psi-treillis against 5.29 ms with cuZFP at 2 bits; 512³: 10.53 MiB read per frame against 1536.00 MiB for the raw chain (145.9 times less), 624 MiB of GPU memory against 3712 MiB (5.95 times less).
- **Verdict: holds.**
- **Evidence.** `benchmarks/playback/playback.csv` (outputs of the comparator: `benchmarks/playback/logs/view_*.log`); to reproduce: `python benchmarks/playback/playback.py run` (GPU running nothing else; `psi_view` exists on Windows only).
- **Caveat.** The statement is written for this sheet; the criterion is that of the pre-registration of 17 November 2026, which kept the one of the measurement of 16 November. The two times at 256³ are those of the run that shows cuZFP and Psi-treillis side by side (pair `cuzfp_psi`); the data and the memory at 512³ are those of the run that shows the raw frames and Psi-treillis (pair `raw_psi`), the memory of Psi-treillis being compared with that of the raw chain. Timings of one machine, one run per case; in synchronised mode both sides share the GPU while they are measured. cuZFP at 2 bits is at a fixed rate, not at a bound.

## Claims that do not hold

What the project would have liked to be able to assert and that its own measurements refute. They bound the domain of validity.

### LAGR-4. Clustering under the trilinear texture

- **Claim.** In the protocol of LAGR-1, the trilinear texture causes a clustering that Psi-treillis avoids.
- **Criterion.** Excess dispersion of the trilinear texture above 5 standard errors, at 256³ and at 512³. *Fixed before the analysis (at the top of `benchmarks/tracers/tracers.py`), except the number of tracers per voxel at 256³, raised from 1 to 8 after a first trial.*
- **Measured.** In standard errors. trilinear texture, 256³: +3.3; trilinear texture, 512³: +2.1.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/tracers/resolved_256_summary.csv`, `resolved_512_summary.csv` (lines of the tracer program: `resolved_256.csv`, `resolved_512.csv`; tables: `resolved_256_analysis.txt`, `resolved_512_analysis.txt`); to reproduce: `python benchmarks/tracers/tracers.py run resolved_256 resolved_512` (GPU; it runs the campaigns, then writes `<campaign>_analysis.txt` and `<campaign>_summary.csv`).
- **Caveat.** On resolved data, this protocol sees no effect of the divergence of the texture at the scale of the voxel. Only LAGR-3 separates Psi-treillis from the texture. The campaigns replayed on 17 November 2026 count bins of 1 voxel directly at 256³; the first campaign counted bins of half a voxel and merged them: the count per bin of 1 voxel is the same quantity.

### BOUND-4. Certified bound, degree 2, one point lower

- **Claim.** In degree 2, the bound is held at 5 % of u′ at 256³ and at 6 % at 512³.
- **Criterion.** No frame above the certification threshold in the two files. *Fixed after the measurement.*
- **Measured.** 0 of 2 files below the certification threshold. Largest error, as a fraction of u′: 256³ at 5 %: 5.190 %, 1 frame(s) above the threshold; 512³ at 6 %: 6.315 %, 1 frame(s) above the threshold.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/compression/reference_files/reference_files.csv`, lines `f_t256_d2_t05` and `f_t512_d2_t06` (outputs of the encoder: `benchmarks/compression/reference_files/logs/pack_f_t256_d2_t05.log`, `pack_f_t512_d2_t06.log`); to reproduce: `python benchmarks/compression/reference_files/reference_files.py`.
- **Caveat.** Floor of the fine level: one voxel, in one frame, that it does not bring back below the threshold once all its nodes are retained.

### RESO-1. Fidelity on a less resolved turbulence

- **Claim.** The result of COMP-1 holds on a turbulence whose Kolmogorov scale is 0.45 voxel instead of 0.74.
- **Criterion.** On `turb512_nu0006.seq`, same settings, same frames: compression ratio ≥ 140 and field error ≤ 0.65 %. *Fixed after the measurement.*
- **Measured.** 104.4× at a field error of 1.516 %; largest error 0.583 u′.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/compression/under_resolved/under_resolved.csv`, line `d3` (output of the encoder: `benchmarks/compression/under_resolved/logs/pack_d3.log`); to reproduce: `python benchmarks/compression/under_resolved/under_resolved.py`.
- **Caveat.** This is the resolution, in voxels, of the public JHTDB dataset at 1,024³ (0.47 voxel, calculated from its description).

### RESO-2. Certified bound on a less resolved turbulence

- **Claim.** On this same sequence, the bound of BOUND-1 remains attainable in degree 3.
- **Criterion.** `pack --tolerance` exits with code 0 at 5, 4 and 3 %. *Fixed after the measurement.*
- **Measured.** Degree 3: `pack` exits with code 1 at 5 %, 1 at 4 % and 1 at 3 % (0: bound held; 1: refusal, capacity of the fine level exceeded; 4: bound missed). Degree 2 at 7 %: code 4.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/compression/under_resolved/under_resolved.csv`, column `pack_exit_code` of the lines `d3_t05`, `d3_t04`, `d3_t03` and `d2_t07`; to reproduce: `python benchmarks/compression/under_resolved/under_resolved.py`.
- **Caveat.** The lattice step of 2 voxels is 4.4 Kolmogorov scales on this sequence, against 2.7 on the other.

### BENCH-3. Against SZ3 in 4D, at an equal bound

- **Claim.** At an equal bound, Psi-treillis compresses at least as much as SZ3 in 4D on 32 frames.
- **Criterion.** Compression ratio of Psi-treillis of degree 3 ≥ that of SZ3 in 4D, at 5, 4 and 3 %, at 256³. *Fixed after the measurement.*
- **Measured.** 256³ at 5 %: 133.7× against 334.0× (field error 0.614 % against 1.690 %); 256³ at 4 %: 126.7× against 267.8× (field error 0.604 % against 1.391 %); 256³ at 3 %: 104.0× against 201.6× (field error 0.582 % against 1.086 %).
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/equal_bound/equal_bound_256.csv`; to reproduce: `python benchmarks/equal_bound/equal_bound.py 256` and `512` (GPU; needs the comparison codecs of `build/codecs/`).
- **Caveat.** SZ3 in 4D compresses 1.9 to 2.5 times more, with a field error 1.9 to 2.8 times higher. At 512³, it is measured by blocks of 8 frames only, which this criterion does not read. Protocol corrected before the re-measurement of 17 November 2026, the criterion being kept as it was: every codec is measured on the same 32 frames (frames 16 to 47 of `turb256.seq`, frames 0 to 31 of `turb512.seq`), where the frame-by-frame codecs were measured on four frames at 256³ and two at 512³, one of them outside the group of Psi-treillis; their compression ratio is the total raw size over the total compressed size, where it was the mean of the ratios of the frames measured; the error of Psi-treillis is no longer the one its encoder reports: its file is decoded through the public interface and compared with the raw frames by the same code as the other codecs; at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames (63 GiB of RAM do not hold a block of 32); zfp is reported apart, its fixed-accuracy mode leaving the largest error well under the bound requested.

### LAGR-5. Lagrangian confirmation on an independent realisation

- **Claim.** The verdicts of LAGR-1 to LAGR-4 are found again on another realisation of the turbulence, all the parameters being fixed in advance.
- **Criterion.** 256³ sequence produced by `dns --seed 7` (other options as for `turb256.seq`), frame 32, 8 tracers per voxel, bins of 1 voxel, 10 Kolmogorov times, tracer seeds 1 to 5, time step of half a frame: cuZFP at 2 bits above 5 standard errors; each Ψ path below 2; trilinear texture below 5; growth of var(D) of the texture ≥ 1,000 times that of Psi-treillis of degree 3. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** In standard errors: cuZFP at 2 bits +40.8; trilinear texture +2.7; Ψ degree 3 -0.1; Ψ degree 2 -1.1; Ψ degree 3, bound 5 % -0.8; Ψ reference (step 1 voxel, eps 0.002) -2.8. Growth of var(D) per Kolmogorov time: texture 1.1·10⁻⁴, Ψ degree 3 2.3·10⁻⁸ (4,641 times).
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/tracers/confirmation_256_summary.csv` (lines of the tracer program: `confirmation_256.csv`; tables: `confirmation_256_analysis.txt`); to reproduce: `python benchmarks/validity/validity.py simulate` for the sequence `data/sweep/nu00280_seed7.seq`, then `python benchmarks/tracers/tracers.py run confirmation_256 confirmation_256_more_seeds` (GPU).
- **Caveat.** Three of the four conditions are met as predicted: cuZFP clusters, the trilinear texture stays under the threshold, and the volume drift separates the texture from Psi-treillis. The fourth fails on one path, read strictly. The criterion says "below 2", the analysis script that defines it says "in absolute value", and the reference path at a step of 1 voxel is at -2.8: an under-dispersion, not a clustering. Read as a signed threshold, the claim holds. On ten seeds instead of five, run after seeing this result and therefore descriptive only, the same path is at -0.6 (seeds 6 to 10: campaign `confirmation_256_more_seeds`; calculation of `tools/claims.py` on the lines of the two campaigns, with the arithmetic of the analysis): the five seeds of the criterion look like a fluctuation, which the criterion does not allow us to discount. The three paths of the codec itself (degree 3, degree 2, degree 3 with a bound) are within 1.1. Bins of 1 voxel were counted directly, where the first campaign counted bins of half a voxel and merged them. The sequence has 48 frames instead of 128, which does not change a frame: with the default seed, frames 0, 32 and 47 of the 48-frame run are byte-identical to those of `turb256.seq` (`benchmarks/validity/identity_turb256.txt`, written by `python benchmarks/validity/validity.py identity`). The sequence was named `v_nu00280_g7.seq` when the criterion was written.

### VALID-2. Ahead of SPERR where the bound holds

- **Claim.** On each sequence where the bound of 5 % is held (degree 3, step of 2 voxels), Psi-treillis compresses at least as much as SPERR at the same bound.
- **Criterion.** Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), frames 8 to 43. For each sequence where `pack --degree 3 --tolerance 0.05` exits with code 0: compression ratio of the group of 32 frames ≥ compression ratio of SPERR at the same absolute bound (harmonic mean over frames 8, 24 and 39). Does not hold if no sequence holds the bound. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** Ahead of SPERR on 6 of the 8 sequences where the bound holds: from 0.97 times its compression ratio (56.1× against 57.9×, at 0.552 voxel per Kolmogorov scale) to 1.22 times (133.2× against 109.3×, at 0.795 voxel).
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/validity/validity.csv` (output of each run in `benchmarks/validity/logs/`); to reproduce: `python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned).
- **Caveat.** Psi-treillis is ahead of SPERR on the 6 sequences at 0.61 voxel per Kolmogorov scale or more, and behind it on the two just above the threshold of VALID-1 (0.543 and 0.552 voxel). There the bound is held at the price of a fine level that costs almost half of the file: the compression ratio falls from 105 to 107 times without the bound to 56 to 60 times with it. SPERR is applied frame by frame; Psi-treillis uses the temporal prediction.

### VALID-4. Field error as a function of the resolution

- **Claim.** Without a bound, the field error of degree 3 at a step of 2 voxels remains below 0.70 % on resolved data and exceeds twice the value of COMP-1 on under-resolved data.
- **Criterion.** Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), frames 8 to 43. Field error ≤ 0.70 % for every sequence with η ≥ 0.69 voxel, and > 1.28 % for every sequence with η ≤ 0.45 voxel. At least one sequence is needed in each range. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** 3 sequences at 0.69 voxel or more: field error from 0.629 % to 0.763 %; 6 sequences at 0.45 voxel or less: from 2.197 % to 2.824 %.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/validity/validity.csv` (output of each run in `benchmarks/validity/logs/`); to reproduce: `python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned).
- **Caveat.** The second half is met, the first is not: one of the three sequences at 0.69 voxel or more is at 0.76 % (0.702 voxel). The threshold of 0.70 % came from the two sequences at 512³; at 256³ and at an equal resolution in voxels the field error is higher. Over the sweep it falls steadily, from 2.8 % at 0.36 voxel per Kolmogorov scale to 0.63 % at 0.80.

### VALID-5. Spread between realisations

- **Claim.** The compression measurements depend on the realisation of the turbulence only to within a few per cent.
- **Criterion.** Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), frames 8 to 43. For each of the 6 viscosities, in degree 3 at a step of 2 voxels and without a bound: the compression ratio of the group of 32 frames and the field error each vary by less than 5 % between the three realisations (difference between the largest and the smallest value, relative to their mean). *Fixed on 16 November 2026, before any measurement.*
- **Measured.** Over the 6 viscosities, between the three realisations: the compression ratio varies by 13.1 % at most (viscosity 0.0028), the field error by 19.3 % at most (viscosity 0.0028).
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/validity/validity.csv` (output of each run in `benchmarks/validity/logs/`); to reproduce: `python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned).
- **Caveat.** The three realisations of one viscosity do not have the same resolution: the Kolmogorov scale depends on the dissipation of each run and itself varies by up to 13 % between seeds (0.702 to 0.795 voxel at a viscosity of 0.0028). Part of the spread measured here is therefore a spread of resolution, which this criterion did not separate. What it does establish: a single sequence per viscosity, as in every earlier measurement of this project, fixes a compression ratio to within 5 to 13 % only.

### REF-2. A second reader from the completed specification

- **Claim.** The completed specification is sufficient, on its own, to write a reader: the state of the frames and the velocity field, without any trial to decide between readings.
- **Criterion.** A second reader, written from `docs/FORMAT.md` by someone who has read neither `src/`, nor the first reader (`tools/reference_reader.cpp`, which was in `src/` at the time), nor `benchmarks/independent_readers/cpp/`, gives back the fingerprint of each frame of the five reference files at 256³ and of the test files at 64³, and the field of their probe to less than 5·10⁻⁵ u′. The list of the passages where the text was not sufficient, kept during the writing, is empty for everything that concerns the state and the field. *Fixed on 16 November 2026, before any measurement.*
- **Measured.** 5 reference files at 256³ and 22 test files at 64³, 1,029 frames read back out of 1,029, 0 fingerprint different from that of the table; at the probe of each file, 0 node of the lattice or of the fine level different from those of the encoder, and a velocity field at 2.0·10⁻⁵ u′ at most; 6 passages noted during the writing, of which 1 counted as concerning the state or the field: D1, section 6.2.
- **Verdict: does not hold.**
- **Evidence.** `benchmarks/independent_readers/python/reference_python.csv` and `gaps.csv` (reader: `tools/reference_reader.py`; notice: `benchmarks/independent_readers/python/README.md`); to reproduce: `python benchmarks/independent_readers/python/reference_python.py`.
- **Caveat.** The criterion fails only on its last clause, read strictly. The first two are met: the reader, run as it was written, gave back every fingerprint and the field at its first run, and its decoding has not been touched since. The passage counted is a silence of the text with no effect on the files read: the prediction formula does not say on how many bits the sum is computed, and no limit bounds the predictors; an exact sum and a sum on 32 bits differ only for predictors at least 21 times larger than those of these files (calculation). Whoever does not count this passage reads "holds". The five other passages noted are not counted: two readings kept as they were and then confirmed on files produced for that purpose (tables of the empty segments, reading by differences), one reading that is the only one consistent with the text, one case that the text itself calls undefined, and the fate of malformed exceptions in an altered file. A single frame probed per file, the last. The field is evaluated in double precision and compared with the probe of the encoder, not with the GPU reader. `PSISTRM1` is not supported, and nothing is read at 512³. The text was completed on the six passages on 17 November 2026: whether it is now sufficient is REF-3.

## Pending claims

Hypotheses whose criterion is written before any measurement. Each one states the test that settles it and what is needed first.

### JHTDB-1. Expected degradation on JHTDB

- **Hypothesis.** On the public JHTDB dataset of isotropic turbulence at 1,024³, with the default settings in degree 3, the field error is more than twice that of `turb512.seq` and the bound of 5 % of u′ is not held.
- **Criterion.** On at least 8 consecutive frames: mean field error > 1.28 % (twice the 0.640 % of COMP-1) and `pack --tolerance 0.05` exits with a non-zero code. The hypothesis is refuted if either of the two conditions is missing. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** RESO-1 and RESO-2: at the same resolution in voxels, the field error is 1.516 % and the bound is refused.
- **What is needed first.** The encoder does not handle 1,024³ (about 31 GiB of GPU memory, calculation, for a GPU of 24 GiB), and a sub-cube is not periodic whereas the encoder relies on a periodic FFT. Since 16 November 2026, not testable under the access conditions obtained: the database limits a cutout to 3 GB from a local machine and reserves its use for small subsets, whereas 8 whole frames weigh about a hundred GB (calculation). JHTDB-3 replaces it with a cutout. Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete frame was retrieved; the scripts are in `benchmarks/cutout/`.

### JHTDB-2. Usefulness for compression on JHTDB

- **Hypothesis.** On this same dataset, there is a setting of Psi-treillis that holds a certified bound of 5 % of u′ or less and compresses at least as much as SPERR at the same bound.
- **Criterion.** For one tolerance among 5, 4 and 3 %: `pack` exits with code 0 and its compression ratio is ≥ that of SPERR set to the same absolute bound, on the same frames. All the settings tried are reported, not only the best. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** BENCH-1 on resolved data; RESO-2 leads one to expect the opposite with the lattice of step 2 voxels. I do not know which verdict will come out.
- **What is needed first.** The prerequisites of JHTDB-1, and a lattice of step 1 voxel (`--mf 1`) that fits in memory at this size. If the claim does not hold, what remains to Psi-treillis, on these data, is zero divergence and decoding on the GPU, not the compression ratio. Not testable under the access conditions obtained, for the same reason as JHTDB-1: JHTDB-4 replaces it with a cutout. Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete frame was retrieved; the scripts are in `benchmarks/cutout/`.

### PORT-1. Another machine

- **Hypothesis.** The chain gives the same test verdicts on another GPU.
- **Criterion.** On an NVIDIA GPU of an architecture other than that of `tests/fingerprints.json`: `tests/test_codec.py` counts no `FAIL` outside the fingerprint cases (`*.fingerprint.*`, `sequence.fingerprint`). *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** `tests/README.md` describes the procedure; it has never been followed anywhere other than on the measurement machine.
- **What is needed first.** A second machine. Under Linux, a build on a machine with a GPU is needed first: the project has been measured only under Windows with MSVC, and compiled under Linux only on a machine without a GPU (LINUX-1). The CPU reference reader was compiled with GCC 13.3 and run on Linux on 17 November 2026 on one 64³ test file (0 fingerprint differing), which is not this criterion.

### BENCH-4. SZ3 in 4D on 32 frames at 512³

- **Hypothesis.** At 512³, SZ3 in 4D on 32 frames compresses at least 1.5 times more than Psi-treillis of degree 3 at an equal bound of 5 %.
- **Criterion.** Compression ratio of SZ3 in 4D on frames 0 to 31 of `turb512.seq` ≥ 1.5 times that of Psi-treillis on the same frames, both measured with the current encoder. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** BENCH-3 at 256³ (2.5 times at 5 %); at 512³, by blocks of 8 frames over frames 0 to 31, SZ3 in 4D gives 225.2× against 137.6× for Psi-treillis on one group of 32 frames, that is 1.64 times (`benchmarks/equal_bound/equal_bound_512.csv`).
- **What is needed first.** More than 63 GiB of RAM, or a variant of SZ3 that processes the block out of core. The bench of BENCH-1 to BENCH-3 was measured again on 17 November 2026 with the corrected protocol: at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames, which is not the single block of 32 frames of this criterion.

### JHTDB-3. Degradation on a JHTDB cutout

- **Hypothesis.** On a 256³ cutout of the public JHTDB dataset `isotropic1024coarse`, processed by the protocol of CTRL-1, degree 3 at a step of 2 voxels makes more than twice the error of COMP-1 and does not hold the bound of 5 %.
- **Criterion.** 8 consecutive frames: relative RMS error at the centre (cube of 160³) > 1.28 %, and `--tolerance 0.05` exits with a non-zero code. Not applicable if CTRL-1 does not hold. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** RESO-1 and RESO-2, on our sequence of the same resolution in voxels.
- **What is needed first.** The access token placed in the environment variable `JHTDB_TOKEN`, then `benchmarks/cutout/download_jhtdb.py`, run with the Python of an environment that holds the client `givernylocal` of the database, and `python benchmarks/cutout/cutout.py jhtdb`: 8 cutouts of 0.2 GB, one request at a time (usage limits of the database: 3 GB per cutout, small subsets). Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete frame was retrieved; the scripts are in `benchmarks/cutout/`.

### JHTDB-4. Step of 1 voxel on the JHTDB cutout

- **Hypothesis.** On this same cutout, the step of 1 voxel holds the bound of 5 % but compresses less than SPERR.
- **Criterion.** `--mf 1 --tolerance 0.05` exits with code 0, and its compression ratio on the 8 frames is lower than that of SPERR applied to the same windowed cutout, at the same absolute bound. Not applicable if CTRL-1 does not hold. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** VALID-3, if it holds on our data.
- **What is needed first.** The prerequisites of JHTDB-3. Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete frame was retrieved; the scripts are in `benchmarks/cutout/`.

### REPR-1. Does our under-resolved sequence stand in for JHTDB?

- **Hypothesis.** `turb512_nu0006.seq`, with the same Kolmogorov scale in voxels, predicts the behaviour of the codec on JHTDB.
- **Criterion.** The relative RMS error at the centre of the windowed cutout of JHTDB (degree 3, step of 2 voxels, 8 frames) equals that of the windowed cutout of `turb512_nu0006.seq` to within 20 %. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** Same resolution in voxels (0.45 and 0.47). Against: our simulation truncates its spectrum at n/3, I do not know where the database truncates its own, and its interval between frames is not verified.
- **What is needed first.** The prerequisites of JHTDB-3. Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete frame was retrieved; the scripts are in `benchmarks/cutout/`.

### LINUX-1. Compilation under Linux

- **Hypothesis.** The project compiles under Linux.
- **Criterion.** On GitHub, at each commit of `main`, an Ubuntu machine without a GPU compiles the library, `psit`, `psi_stream`, `dns` and `psi_ref` with GCC and `nvcc`, then replays `python tools/claims.py`: the last run is green. Nothing is executed there on a GPU. *Fixed on 16 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** The same build was green in the development repository on 16 November 2026 (run 43806060947, on the previous layout of the repository).
- **What is needed first.** The publication of this repository, a completed run of `.github/workflows/linux.yml` on `main`, then `python benchmarks/linux_build/record.py`, which writes `benchmarks/linux_build/workflow_run.json`. This script then evaluates the record, and says that the verdict registered must be updated.
- **Caveat.** `benchmarks/linux_build/workflow_run.json` does not exist when this repository is first published: the workflow can only run after the publication, so the verdict registered is "pending". The same build was green in the development repository on 16 November 2026 (run 43806060947, on the previous layout of the repository), after a first run that had failed at the linking of the C example (mathematics library absent from the link line), corrected in the following commit. Compilation only: the workflow machine has no GPU, no kernel is executed there. Whether the chain gives under Linux the same files and the same test verdicts remains to be established: that is PORT-1.

### REF-3. A third reader from the specification completed on 17 November 2026

- **Hypothesis.** The specification completed on 17 November 2026 on the six gaps of REF-2 is sufficient, on its own, to write a reader.
- **Criterion.** A third reader, written from `docs/FORMAT.md` by someone who has read neither `src/` nor the two existing readers, gives back the fingerprint of each frame of the five reference files at 256³ and of the test files at 64³, and the field of their probe to less than 5·10⁻⁵ u′; the list of the passages where the text was not sufficient, kept during the writing, is empty for everything that concerns the state and the field. *Fixed on 17 November 2026, before any measurement.*
- **Verdict: pending.**
- **Basis.** REF-2: the second reader gave back every fingerprint and the field, and noted six passages where the text left a doubt, one of them counted; `docs/FORMAT.md` was completed on each of them on 17 November 2026 (marks "completed on 17 November 2026, after the second reader").
- **What is needed first.** A working session that has read neither `src/` nor the two existing readers (`tools/reference_reader.cpp`, `tools/reference_reader.py`) nor their logs, and that keeps the list of its doubts before any trial.

<!-- verdicts: end -->

## What the project does not claim

- **The best compression ratio.** SZ3 in 4D compresses more at an equal bound (BENCH-3). Psi-treillis is ahead only in
  the window of BENCH-2.
- **A bound between frames or off the grid.** BOUND-1 holds at voxel centres, at the times of the frames.
- **A proven bound.** The certification margin is empirical.
- **A result on under-resolved data.** RESO-1 and RESO-2 show the opposite.
- **A result outside periodic isotropic turbulence**, or at 1,024³: nothing is measured.
- **A result on a public dataset.** Every measurement is made on turbulence simulated here. The trial on a cutout of
  the public JHTDB database was prepared and could not be carried out (JHTDB-1 to JHTDB-4, REPR-1).
- **A clustering of tracers avoided compared with trilinear interpolation** (LAGR-4).
- **Timings valid elsewhere.** SPEED-1, ENC-1 and PLAY-1 are those of one machine.
- **A reader that sees every alteration**: the limits are in the caveat of ROBUST-1.
- **A format that bounds its predictor coefficients.** Neither the encoder nor the readers bound them, and a file
  whose predictors make the sum of the prediction leave 32 bits has no defined decoding (`docs/FORMAT.md`, section 9).
- **A specification put to the test for the velocity field.** `docs/FORMAT.md` was sufficient for the state of the
  frames. For the field, the missing conventions were established by trial on files, then added to the text and
  cross-checked against the code (caveat of REF-1). The reader written afterwards from the text thus completed left
  one passage counted as a gap (REF-2), and no reader has yet been written from the text completed on that passage
  (REF-3).

## Adding or changing a claim

The registry is the list `REGISTRY` of `tools/claims.py`: statement, criterion, expected verdict, log, and the
function that reads this log. After a change, `python tools/claims.py --write` rewrites the generated part of
this sheet; the rest is edited by hand.
