# Writing journal of the second reader

> *This is a translation of the journal that was kept in French during the writing of the second reader; the identifiers of the reader and the names of the files quoted in it have been given their present English names, and the passages of `docs/FORMAT.md` are quoted in the English of that document.*

Kept during the writing of `tools/reference_reader.py`, from `docs/FORMAT.md` alone (claim REF-2). Each entry is
written before the trial that could settle it. The summary and the verdict are in `README.md`.

## 1. Doubts noted on reading the text, before any trial on a file

At this stage the reader is written in full and has been run on no `.psi` file and on no probe. For each doubt: the
passage, the possible readings, the one I adopt, and what it can affect.

**D1. Width of the arithmetic of the prediction (6.2).** "Prediction: `(a * q(k) + b * q(k-1) + c * lap(q(k)) + 512)
>> 10`". The text says that the shift is arithmetic, not on how many bits the sum is computed. Two readings: 32 bits
with wrap-around when the sum exceeds them, or an exact sum. I adopt the exact sum (64 bits). The two readings differ
only if the sum leaves 32 bits, which requires `(|a| + |b| + 12 |c|) * 32767 + 512 >= 2^31` (calculation): the reader
counts the frames where this happens. Can affect the state, but only for enormous predictors.

**D2. "First node of the support" and threshold of the proximity bit (6.5, 6.7).** "the kernel subtracts from each node
the value of the first node of the support"; "value less than 2^14 from the edge of the range". The text does not say
which node is the first, nor whether the threshold is strict. I adopt: the node of smallest indices on the three axes
(the first in the traversal order given in 6.7), and, in reading by differences, I always subtract, without testing
the bit. The gradient is the same up to floating-point rounding whichever node is subtracted and whether one subtracts
or not, as long as no difference reaches 2^15 (the sums of the `w'` are zero, as 6.7 says). Does not affect the state.
Affects the field only in a support where a difference reaches 2^15, a case where the text itself says that the
reading is no longer exact.

**D3. Frequency table of an empty segment (6.1, 6.6).** "Their sum is exactly 4096, otherwise the reader refuses the
frame." In the section of the fine level, the segments of the masks and of the coefficients are empty when the frame
has no brick. The text does not say whether the table of an empty segment still sums to 4096. Two readings: check all
the tables, or only those of the non-empty segments. I adopt the letter of the text: all of them. If a valid frame
without a brick carries a zero table, my reader will refuse it wrongly and that will be a gap in the text. Can affect
the state (refusal of a valid frame).

**D4. Exceptions of an altered file (6.3).** The text does not say what the reader does (a) with an escape whose
position is `ecount` or more, (b) with an escape whose `val` leaves the range of the state, (c) with an entry whose
bit 31 is zero placed after an overflow. I adopt: (a) ignored; (b) `val` truncated to the type of the state (8 or 16
bits); (c) ignored. None of these cases exists in a valid file: the entries are sorted and the encoder writes only
states within the range. Affects only the state of an altered file.

**D5. Innovation of type 1 just after a key (10, point 7).** The text itself says that such a file "has no defined
decoding". I adopt: in sequential reading, `q(k-1)` is the content of the slot, as the text describes it; at position
1, where this content does not exist, refusal. `pack` writes none. This is not a gap in the text: it says so.

**D6. "under the maxima of `FineInfo`" (9).** Less than or equal, or strictly less? I adopt less than or equal: the
maxima are the "largest number of bricks present in a frame", which a frame therefore reaches. Can affect the state
(refusal of a valid frame) if the strict reading were the right one; only one reading is reasonable.

Nothing else stopped me on reading: the layout of the headers, of the table, of the seal and of the payloads, the order
of the segments and of the tables, the rANS decoding, the prediction, the escapes and the overflows, the Lorenzo
residual, the refinement, the fine level, the formula of the field and the fingerprint each have a single reading in
the text.

Choices that are not doubts about the text:

- `PSISTRM1` is refused: this reader does not support it (the task asks for `PSISTRM2` and `PSISTRM3`).
- The field is evaluated in double precision by the formulas of 6.7, not in the order and the precision of the GPU
  reader.

## 2. First run

The reader run here is that of commit `7cc43a1` of the development history (not public), without any touch-up between its writing and these runs.

- Twenty-two valid test files at 64³ (`d2_*.psi`, `d3_*.psi`): 792 frames read back, no fingerprint different from
  that of the table. Seal verified on the sixteen sealed files. (This line first said "twenty files, 720 frames,
  fourteen sealed": a counting error on my part, corrected from the output of the run.)
- Six probes already present beside these files (`d2_base`, `d3_base`, `d2_bound`, `d3_bound`, `d2_exceptions`,
  `d3_exceptions`): sampling lattice and fine level identical to those of the encoder, field at 3.2·10⁻⁶ to
  8.0·10⁻⁶ u′. These probes are not those of the log: the log re-encodes each file to produce its own.
- Five reference files at 256³: 237 frames read back, no differing fingerprint.
- 84 altered files at 64³ (`guard_*.psi`): each is refused with a message, or read in full with a count of the
  differing frames; no Python error.

No trial was used to decide between readings: nothing was modified.

What these files do not exercise, according to their headers: none has a wrapped frame (all are read directly: D2 is
not tested); no `PSISTRM3` file has a raw key (the rounding rule of section 5, which the text itself calls untested);
the predictors are far from the 32-bit limit of D1.

## 3. Complementary trials, noted before being run

Three more trials, on files produced by `psi_stream pack` run as a black box on the test sequence at 64³ (six files in
the end: one per degree), to put to the test what the files of the criterion do not exercise. For each I keep the
reading already adopted: if it fails, the text is at fault and that is a gap; if it passes, the doubt is settled
without anything having been chosen from a file.

- **C1, for D3.** A file with a bound some frames of which have no brick, if one can be obtained with a wide tolerance
  (`--tolerance 0.5`). Reading kept: all the frequency tables sum to 4096, those of the empty segments included.
- **C2, for section 5.** A `PSISTRM3` file with raw keys (`--raw-keys --tolerance 0.05`). Reading kept: `key_bytes`
  plus the exception list, rounded to 8 bytes, falls on the start of the section of the fine level.
- **C3, for D2.** A file whose lattice wraps, hence read by differences, if one can be obtained by lowering `--eps`.
  Reading kept: the node of smallest indices of the support is subtracted.

Outcome of the three trials, reader unchanged:

- **C1.** `--tolerance 0.5` does give files without any brick (36 frames out of 36, in degree 2 and in degree 3).
  The reader reads them with the check of all the tables: no differing fingerprint. The letter of the text was exact;
  D3 is settled without anything having been chosen from a file.
- **C2.** The two `PSISTRM3` files with raw keys are read: no differing fingerprint. The rounding rule of section 5 is
  tested.
- **C3.** `--eps 0.0005` gives in degree 3 a file with 36 wrapped frames, read by differences (`pack` exits with
  code 3: 590,961 clippings of site 5); in degree 2, clippings of site 5 but no wrapped frame. Both are read: no
  differing fingerprint, lattice identical to that of the encoder, field at 5.6·10⁻⁶ u′ and 2.9·10⁻⁶ u′.

## 4. First complete log, and two additions to the script

First run of `reference_python.py` (reader of commit `7cc43a1`): 27 files of the criterion, 1,029 frames, no differing
fingerprint; field at 2.0·10⁻⁵ u′ at most; the 84 altered files have the same outcome and the same number of
differing frames as with `psi_stream play`.

Two points to take up again, noted before rerunning:

- **"Cap" files.** The re-encoding of `d2_cap.psi` and of `d3_cap.psi` with the settings of the header differs from
  the file: the header does not say the cap on the passes of the fine level. Searched by hand: `--fine-passes 1` gives
  back the two files byte for byte. The script now tries `--fine-passes` from 1 to 8 when the re-encoding differs.
  This concerns the obtaining of the probe, not the reading of the format.
- **D2, range of the differences.** In `c3_d3_eps0005.psi`, the range of the differences inside a support reaches
  65,062, above 2^15: I thought, wrongly, that the clipping of site 5 kept it below. This file can therefore decide
  the choice of the subtracted node. My reading does not change (the node of smallest indices) and it already gives
  back the field of the encoder; I only add the measurement of what the last node of the support would give, to know
  whether the file distinguishes the two.
- Since the first run, two touch-ups of the command line of the reader, with no effect on the decoding: with
  `--points`, only the velocities are output; a missing file gives a message in French.

## 5. Final log

Second complete run of `reference_python.py`, with the reader in its final state: it is this run that wrote the files
of this folder.

- 27 files of the criterion, 1,029 frames, no differing fingerprint. The 27 re-encodings give back their file, by
  SHA-256 (`--fine-passes 1` for the two "cap" files): each probe stands for its file. Lattice and fine level
  identical to those of the encoder; field at 2.0·10⁻⁵ u′ at most.
- **D2.** In `c3_d3_eps0005.psi`, 118 of the 114,688 probe points have a support in which the range of the differences
  reaches 2^15. Subtracting the last node of the support instead of the first would put the field at 0.42 u′ from that
  of the encoder; the direct reading, at 25.7 u′. The file therefore distinguishes the readings, and the one I had
  adopted before the trial is the right one. I tried the other node only afterwards, to measure what the trial was
  worth.
- The 84 altered files have the same outcome and the same number of differing frames as with `psi_stream play`.

## 6. What I count, and why

A passage counts as "the text was not sufficient for the state or the field" if, on a file that the checks of the
format accept, two readers conforming to the text can return a different state or field without the text saying which
is right; or if the reading I had adopted had proved wrong; or if a file had been needed to choose between readings.

- **D1 counts.** Nothing in the text says the width of the sum of the prediction, and nothing in it bounds the
  predictors. I had no means of deciding, neither by the text nor by a file; I took the exact sum because it is the
  formula as written. On the files read, this changes nothing (margin of a factor 21).
- **D2 does not count.** "First node of the support" has only one reading once set beside the order given in 6.7, and
  it is confirmed. The text would gain by saying it in so many words: the choice weighs 0.42 u′ on a file that `pack`
  writes.
- **D3 does not count.** The text was explicit and exact; it was I who doubted.
- **D4 does not count for the state and the field**: it concerns only altered files. It is a gap in the text as
  regards robustness.
- **D5 and D6 do not count**: the text itself calls the first case undefined, and the definition of the maxima leaves
  only one reading of the second.

The list is therefore not empty for the state: REF-2 does not hold, by its last clause, although the reader gave
everything back at its first run. I note that this is a strict reading: if one holds that a silence with no effect on
the files read does not count, the list is empty and the claim holds.

## 7. Branch rebased onto `main`

After the log, the branch was rebased onto `main`, which had meanwhile received two unrelated commits (Linux workflow,
`CMakeLists.txt`). The commits therefore changed identifier: that of the reader, written before any run, was called
`df10f58` and is now called `7cc43a1`. Its content is the same and its author date is that of its writing; its commit
date, for its part, is that of the rebase, later than the runs. The references of this journal, of the notice and of
`CHANGELOG.md` were realigned on the new identifier. No file of the log changed.
