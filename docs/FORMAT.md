# The `.psi` file format

A `.psi` file contains a sequence of 3D velocity fields compressed by Psi-treillis. There are three versions, recognised by their first eight bytes: `PSISTRM1`, `PSISTRM2`, `PSISTRM3`. Since the evening of 15 November 2026, `pack` ends the file with a seal of 48 bytes (section 4.1), which does not change the signature.

This specification is drawn from the code of the working tree as of 15 November 2026. Its references to the code were redone on 17 November 2026, after the sources were reorganised: a reference names the function, structure, constant or kernel that holds the code and the file it is in, and gives no line number, since names are stable and lines are not. A name given without a file is in `src/codec_stream.cuh`, except `play`, its lambda `ingest` (written "`ingest` in `play`") and `main`, which are in `tools/psi_stream.cu`. Another file is named next to the name, or ahead of a group of names; once named, it holds up to the next semicolon, or to the end of the parenthesis or of the cell, and "same file" refers to the file named just before. `Receiver::inspect` designates the member `inspect` of the structure `Receiver`. In running text, `pack` and `play` are the two modes of the program `psi_stream`; in a reference, they are the functions of these names. The code is authoritative. What could not be settled by reading is marked "not established by reading" and taken up in section 10.

| File | What it holds |
|---|---|
| `src/codec_stream.cuh` | The structures of the format, the checks made when a file is opened, the geometry (`Geo`), the encoder (`Sender`, `pack`), the receiver (`Receiver`), the options (`SOptions`), the compile-time assertions on the layout. |
| `src/codec_kernels.cuh` | The kernels of the codec: sampling lattice and its curl, sparse fine level, prediction, exceptions, refinement, rANS decoding, limits of the codec. |
| `src/sequence.h` | The sequence file that `pack` encodes. |
| `tools/psi_stream.cu` | The program `psi_stream`: its command line and the function `play`. |
| `tools/playback_kernels.cuh` | The kernels of the measurement instruments (advection, ray marching) and `print_seal`. |
| `tools/viewer/psi_view.cu` | The visual comparator `psi_view`. |
| `src/psitreillis.cu`, `src/psit.cpp` | The library and its tool `psit`. |
| `tools/reference_reader.cpp`, `tools/reference_reader.py` | The two reference readers, written from this text and not from the code. |

The former bench program `psi` and the kernels that only it used are not part of the published code.

**Additions of 16 November 2026, of another provenance.** A reference reader on CPU, `tools/reference_reader.cpp` (executable `psi_ref`), was written from this specification without reading the sources of the codec. It showed what the specification did not say (`benchmarks/independent_readers/cpp/README.md`). These gaps are filled here, under two marks:

- "clarified": the text had only one reasonable reading but did not state it; the reading adopted reproduces the fingerprint of the 622 frames of the twelve reference files.
- "established by trial": the text was silent, and several readings were decided between by comparing with a file; the reading adopted reproduces, bit for bit, the sampling lattice that the encoder records in its probe, and its velocity field to within 2.8·10⁻⁵ u′ (`benchmarks/independent_readers/cpp/reference_cpp.csv`).

These passages do not come from a reading of the code: when they were written, they carried no reference to the code and had not been cross-checked against it. The rest of the text has not changed.

**Cross-check against the code, 16 November 2026.** These passages were then re-read against the kernels of the codec and the code of the stream (today `src/codec_kernels.cuh` and `src/codec_stream.cuh`). All conform to the code. Each has since ended with "Cross-checked", followed by the reference that supports it. The code has also answered points 5 to 9 of section 10, after each of them, and corrected a field of the header of the probe (section 11).

**Additions of 17 November 2026, after the second reader.** A second reader, `tools/reference_reader.py`, was then written from this specification thus completed, again without the sources of the codec. Its session kept the list of the passages where the text left a doubt (`benchmarks/independent_readers/python/gaps.csv`, D1 to D6). Each of these passages is completed here from a reading of the code that implements it, and carries the mark "(completed on 17 November 2026, after the second reader: gap Dn)": D1 in 6.2, with the limit it reveals in section 9 and in section 10, point 10; D2 in 6.5 and 6.7; D3 in 6.1 and 6.6; D4 in 6.3; D5 in section 10, point 7; D6 in sections 3 and 9. These completions describe the format as the code implements it; they do not change it.

## Vocabulary

| Term | Meaning |
|---|---|
| Frame | One instant of the sequence: a velocity field on a periodic grid of `n`³ voxels, of side 2π (`kL` in `src/codec_kernels.cuh`; header comment of `src/sequence.h`). |
| Lattice | Periodic grid of nodes. Each node carries three integer coefficients: the components of a vector potential Ψ in B-splines. The velocity is the curl of Ψ (in `src/codec_kernels.cuh`: header comment, `sample`). |
| Level | One of the `L` nested lattices of the pyramid. Level `l` has `nl[l] = (n / mf) >> l` nodes per axis. Level 0 is the finest: its coefficients fit in 8 bits, those of the other levels in 16 bits (`Geo::build`; the states `q0`, on 8 bits, and `q`, on 16 bits, of `Receiver` and of `Sender`). |
| State | The quantised coefficients of a level at a frame. |
| Sampling lattice | Sum of the levels, brought to the step of level 0. The reader rebuilds it in memory. It is never written in the file (`Receiver::decode_levels`; `synth_lattice` in `src/codec_kernels.cuh`). |
| Key frame | Frame that carries the complete state of all the levels. It is decoded on its own. |
| Innovation | Difference, on 8 bits, between the state of a coefficient and its integer prediction, computed from the two previous frames (in `src/codec_kernels.cuh`: `Coef`, `predict3`, `level_innov`). By extension: a frame that carries only these differences. |
| rANS | Entropy coder with an integer state (range Asymmetric Numeral Systems). It codes bytes according to a frequency table. |
| Sparse fine level | Second potential, of step one voxel, present only in places (`PSISTRM3`). Not to be confused with level 0 of the pyramid, which the code also calls "fine level" in the exceptions (`kExFine` and the comment of `ExEntry` in `src/codec_kernels.cuh`; "overflows of the fine level" on the "Exceptions" line of `pack`). |
| Slot | Location of a frame in the memory of the reader. Frame `j` occupies slot `j & 1` (`ingest` in `play`; the comment that precedes `kLatNear` in `src/codec_kernels.cuh`). |
| `u'` | `u_rms`, the RMS value of one velocity component (`SequenceHeader::u_rms` in `src/sequence.h`). |

## 1. Overview

| Offset | Section | Size | Presence |
|---|---|---|---|
| 0 | `StreamHeader` | 208 bytes | always |
| 208 | `FineInfo` | 64 bytes | `PSISTRM3` only |
| 208 or 272 | payloads of frames 0 to `n_frames - 1`, end to end | variable | always |
| `index_offset` | frame table: `n_frames` x `FrameRecord` | 48 bytes per frame | always |
| `index_offset + 48 * n_frames` | seal: `StreamSeal` | 48 bytes | except with `pack --no-seal`; absent from the files written before the seal |

- Writing order, all of it in `pack`: header, header of the fine level, frames, table, seal. Without a seal, the file ends after the table: its size is `index_offset + 48 * n_frames`. With the seal, it is 48 bytes more. The reader refuses any other size (`read_stream_index`).
- The header is written twice. The first time, `index_offset` and the reserved words are zero. They are filled in at the end, as are the maxima of `FineInfo` (`pack`, after the loop over the frames). The seal carries the fingerprint of these headers in their final state. An interrupted encoding leaves `index_offset = 0`: the reader refuses the file (`check_stream_header`).
- Endianness. The structures are written as they are in memory, by `fwrite` (`pack`). The code contains neither a byte-order conversion nor a packing directive. The format is therefore that of the build machine: x86-64, little-endian, IEEE 754 floats on 32 bits. The size and the offsets of the structures are asserted at compile time by `static_assert` declarations, at the end of `src/codec_stream.cuh`: if they change, the compilation fails. Three places assume little-endianness: the initial state of an rANS block (`rans_pack`; `rans_decode` in `src/codec_kernels.cuh`), the marker `kClipMarker` and the fingerprint (`hash_bytes`).
- Alignment of the structures. Offsets recomputed from the types: `StreamHeader`, `FineInfo`, `FrameRecord`, `FineTrailer` and `ExEntry` contain no padding. Each field falls on a multiple of the alignment of its type, and each total size is a multiple of the largest alignment. The only padding of the format is in the payloads (section 5).
- Alignment of the frames. In `PSISTRM1` and `PSISTRM2`, the frames are not aligned. In `PSISTRM3`, each payload is a multiple of 8 bytes (`Sender::append_fine`; required at reading, `Receiver::inspect`). Since 272 is a multiple of 8, each frame then starts at an offset that is a multiple of 8 (deduced, not checked by the code).
- The reader finds the frames only through the table (`ingest` in `play`). It does not assume that they follow one another.
- Order of the arrays. The node `(x, y, z)` of a level has the index `i = (z * nl + y) * nl + x`. Its three coefficients are at positions `3 * i + c`, `c` = 0, 1, 2 for the components x, y, z of Ψ (`predict3` in `src/codec_kernels.cuh`).

Decoding of the header, of the table and of the seal with the `struct` module of Python (these are the strings with which `tools/reference_reader.py` unpacks them):

```
StreamHeader  "<8s8i6f18iQ8Q"   208 bytes
FineInfo      "<4f2IiI8I"        64 bytes
FrameRecord   "<QQQi3fff"        48 bytes
StreamSeal    "<8s3Q2Q"          48 bytes
```

## 2. `StreamHeader` (208 bytes, `StreamHeader` in `src/codec_stream.cuh`)

| Offset | Size | Type | Field | Unit | Meaning |
|---|---|---|---|---|---|
| 0 | 8 | `char[8]` | `magic` | | `PSISTRM1`, `PSISTRM2` or `PSISTRM3`, with no final zero (`check_stream_header`, `pack`). |
| 8 | 4 | `int32` | `n` | voxels | Resolution per axis. Power of two, from 16 to 8192 (`check_stream_header`). |
| 12 | 4 | `int32` | `degree` | | Degree of the B-splines: 2 or 3 (`check_stream_header`). |
| 16 | 4 | `int32` | `mf` | voxels | Step of level 0, in voxels: `nl[0] = n / mf`. Power of two (`Geo::build`). |
| 20 | 4 | `int32` | `levels` | | Number of levels `L`, from 1 to 6 (`Geo::build`; `kMaxLevels` in `src/codec_kernels.cuh`). |
| 24 | 4 | `int32` | `fb` | bits | Fractional bits of the sum of the levels, from 0 to 12 (in `src/codec_kernels.cuh`: `fine_sum`, `check_codec_limits`). |
| 28 | 4 | `int32` | `key_interval` | frames | One key frame every `K` frames; 0: the first only (`SOptions::key_interval`, `pack`). Informative: no reader reads it, they rely on the `type` of each frame. |
| 32 | 4 | `int32` | `n_frames` | frames | Number of frames of the file, from 2 to 2^24 (`check_stream_header`). |
| 36 | 4 | `int32` | `first_frame` | frames | Position of the first frame in the source sequence (`pack --start`; set in `pack`). |
| 40 | 4 | `float32` | `delta` | velocity x length | Quantisation step of the potential. The decoded field is multiplied by `delta / (2^fb * h0)`, with `h0 = 2π / nl[0]` (`Geo::field_scale`; `PsiField::s` in `src/codec_kernels.cuh`). |
| 44 | 4 | `float32` | `dt_frame` | time | Time interval between two frames, copied from the sequence (`pack`; `SequenceHeader::dt_frame` in `src/sequence.h`). |
| 48 | 4 | `float32` | `u_rms` | velocity | `u'`, mean over the sequence (`SequenceHeader::u_rms` in `src/sequence.h`). |
| 52 | 4 | `float32` | `re_lambda` | | Taylor Reynolds number (`SequenceHeader::re_lambda` in `src/sequence.h`). |
| 56 | 4 | `float32` | `kmax_eta` | | Resolution of the smallest scale, kmax·η (`SequenceHeader::kmax_eta` in `src/sequence.h`). |
| 60 | 4 | `float32` | `eps` | fraction of `u'` | Target quantisation error (`pack --eps`; set in `pack`). Informative: no reader reads it. See section 8. |
| 64 | 72 | `int32[6][3]` | `coef` | 1/1024 | Predictor `(a, b, c)` of level `l`, at offset `64 + 12 * l`. Zero for `l >= levels` (`pack`; `Coef` in `src/codec_kernels.cuh`). |
| 136 | 8 | `uint64` | `index_offset` | bytes | Offset of the frame table (`pack`). |
| 144 | 64 | `uint64[8]` | `reserved` | | See below. |

Cross-check: 8 + 8 x 4 + 6 x 4 + 18 x 4 = 136. `index_offset` is indeed at offset 136, with no padding before it.

Reserved words (comment of `StreamHeader::reserved`; written by `pack`):

| Word | Offset | Content |
|---|---|---|
| 0 | 144 | Clipping marker `kClipMarker = 0x3150494C43495350`, that is the bytes `PSICLIP1`. It says that the seven following words are filled in. |
| 1 to 6 | 152 to 199 | Number of clippings at site `s` = 0 to 5, all levels and all frames of the file taken together. |
| 7 | 200 | Low-order 32 bits: number of clipped frames. High-order 32 bits: number of wrapped frames. |

- To clip a value is to bring it back to the bound of its range. Sites (`kClipKey` to `kClipLattice` in `src/codec_kernels.cuh`): 0 target of a key frame out of range; 1 target of an innovation out of range; 2 innovation outside ±127; 3 reconstructed state out of range; 4 partial sum of the pyramid out of range; 5 difference between neighbouring nodes of the sampling lattice too large.
- Clipped frame: a frame where at least one of these counters is non-zero (`Sender::encode`). The four frames used to fit the predictors are not counted (`Sender::reset_clip`, called by `pack` after the fit).
- Wrapped frame: a frame whose sampling lattice leaves the range of a 16-bit integer (`Sender::encode`; in `src/codec_kernels.cuh`: `synth_lattice`, `apply_lattice_ex`). See 6.5.
- If word 0 is not the marker, `play` displays no counter and reads a `PSISTRM2` or `PSISTRM3` file by differences (`play`, constant `plain`).

## 3. `FineInfo` (64 bytes, `PSISTRM3`, `FineInfo` in `src/codec_stream.cuh`)

It follows `StreamHeader`, at offset 208 (`read_stream_index`).

| Offset | Size | Type | Field | Unit | Meaning |
|---|---|---|---|---|---|
| 0 | 4 | `float32` | `tolerance` | fraction of `u'` | Bound requested on the largest error of a component at voxel centres. |
| 4 | 4 | `float32` | `eps` | fraction of `u'` | Quantisation error of the sparse fine level. By default `tolerance / 8.5` (`pack`). |
| 8 | 4 | `float32` | `delta` | velocity x length | Quantisation step of the fine potential (`pack`). |
| 12 | 4 | `float32` | `max_error` | fraction of `u'` | Largest error observed at encoding over the whole file (`pack`). |
| 16 | 4 | `uint32` | `max_bricks` | bricks | Largest number of bricks present in a frame (`pack`). |
| 20 | 4 | `uint32` | `max_nodes` | nodes | Largest number of nodes retained in a frame (`pack`). |
| 24 | 4 | `int32` | `brick` | nodes | Side of a brick. It is 4; any other value is refused (`read_stream_index`, `Receiver::init_fine`; `kFineSide` in `src/codec_kernels.cuh`). |
| 28 | 4 | `uint32` | `frames_over` | frames | Frames whose largest error exceeds the certification threshold (`pack`). |
| 32 | 32 | `uint32[8]` | `reserved` | | Word 0: arithmetic margin. Words 1 to 7: zero. |

- Arithmetic margin: `reserved[0]`, at offset 32, contains the bits of a `float32`, as a fraction of `u'` (comment of `FineInfo::reserved`, `fine_margin`, `pack`). It is zero in a file from before the evening of 15 November.
- Certification threshold: `tolerance - margin / 2`, `margin` being the arithmetic margin (`fine_certified`). A frame counts in `frames_over` if its largest error exceeds this threshold multiplied by `u_rms` (`pack`).
- Marking threshold, used by the encoder to retain nodes: `tolerance - margin`. It is not written; it is deduced. `pack` requires `0 <= margin <= tolerance / 10` (both in `pack`).
- `frames_over` non-zero: the file is written all the same, and `pack` exits with code 4, or with code 3 if coefficients were clipped as well and `--allow-clipping` is not given (end of `pack`). `play` displays it as a warning (`play`).
- `max_bricks` and `max_nodes` size the buffers of the reader and bound each frame (`Receiver::init_fine`, `FineState::init`, `Receiver::inspect`; `fine_section_cap`, called by `play`). A frame may reach them: the comparison is "less than or equal", and section 9 gives its exact form (completed on 17 November 2026, after the second reader: gap D6).

## 4. Frame table: `FrameRecord` (48 bytes, `FrameRecord` in `src/codec_stream.cuh`)

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 8 | `uint64` | `offset` | Offset of the payload, counted from the start of the file (`pack`). |
| 8 | 8 | `uint64` | `bytes` | Size of the payload, in bytes (`pack`). |
| 16 | 8 | `uint64` | `hash` | Fingerprint of the frame (section 7). |
| 24 | 4 | `int32` | `type` | 0, 1, 2 or 3, see below. |
| 28 | 12 | `float32[3]` | `mean` | Mean velocity of the frame, components x, y, z, in velocity units (`Sender::encode`, `pack`). |
| 40 | 4 | `float32` | `field_err` | Relative error of the field, measured at encoding. It is -1 if it was not measured (`pack`). |
| 44 | 4 | `float32` | `field_max` | Largest error of a component at voxel centres, divided by `u'`. Zero if it was not measured (`pack`). |

Cross-check: 3 x 8 + 4 + 3 x 4 + 4 = 44. `field_max` is indeed at offset 44 and the structure is 48 bytes, with no padding.

| `type` | Meaning |
|---|---|
| 0 | Raw key frame: the states, as they are. |
| 1 | Innovation, predicted with the coefficients `coef` of the header. |
| 2 | Innovation that follows a key frame. The prediction is persistence: the state of the previous frame (`persist` in `Receiver::decode_levels`). |
| 3 | Coded key frame: residual of a Lorenzo prediction, passed through rANS. |

- `pack` writes a key at frame 0, then at each `j` that is a multiple of `key_interval` if it is non-zero. The frame that follows a key is of type 2 (`pack`). A key is of type 3 except with `--raw-keys`, or if the coded form exceeds the raw form (`Sender::payload`). The first frame has `offset = 208`, or 272 in `PSISTRM3` (`pack`).
- The mean velocity is not in the payload. The reader takes it from the table (`ingest` in `play`, `Receiver::decode_levels`). The potential does not carry the mean mode (`PsiField::m0` and `PsiField::m1` in `src/codec_kernels.cuh`).
- `field_err` = square root of (sum of the squares of the differences / sum of the squares of the reference), over all the voxels and the three components (`field_err_bricks`, `Sender::field_error`).
- The two measurements are made on all the frames of a `PSISTRM3` file, and otherwise on the frames `j` that are multiples of `--check-every` (1 by default; 0: never) (`SOptions::check_every`, `pack`).

### 4.1 Seal: `StreamSeal` (48 bytes, `StreamSeal` in `src/codec_stream.cuh`)

It follows the table, at offset `index_offset + 48 * n_frames`. `pack` has written it by default since the evening of 15 November 2026; `pack --no-seal` does not write it (`SOptions::seal`, `pack`).

| Offset | Size | Type | Field | Meaning |
|---|---|---|---|---|
| 0 | 8 | `char[8]` | `magic` | `PSISEAL1`, with no final zero. |
| 8 | 8 | `uint64` | `header_hash` | Fingerprint of the 208 bytes of `StreamHeader`, as they are in the finished file. Salt `0x3000`. |
| 16 | 8 | `uint64` | `fine_hash` | Fingerprint of the 64 bytes of `FineInfo`, salt `0x3000 + (1 << 56)`. Zero if the file is not `PSISTRM3`. |
| 24 | 8 | `uint64` | `index_hash` | Fingerprint of the `48 * n_frames` bytes of the table, salt `0x3000 + (2 << 56)`. |
| 32 | 16 | `uint64[2]` | `reserved` | Zero. The reader refuses any other value. |

- The fingerprint is that of a byte array of section 7 (`mix64`, exclusive OR over the positions), computed on the host (`host_hash`).
- What the seal covers: all the fields of `StreamHeader` (`n`, `degree`, `mf`, `levels`, `fb`, `delta`, `dt_frame`, `coef`, `n_frames`, `index_offset`, and the clipping counters, which decide the reading mode of the lattice), all those of `FineInfo` (its `delta`, its maxima, the arithmetic margin) and the whole table (offset, size, fingerprint, type and mean velocity of each frame). What it does not cover: the payloads, which fall under the fingerprint of each frame (section 7).
- At reading (`read_stream_index`). The file ends with the table, and it has no seal, or 48 bytes further on. In the second case these 48 bytes must be a seal (`magic`, reserved words zero) and its three fingerprints those of the file. Otherwise the reader refuses the file with code 1 and says which part has changed: the header, the header of the fine level or the table. A file without a seal is read; `play` and `psi_view` announce it on a "Seal" line (`print_seal` in `tools/playback_kernels.cuh`), and `play --require-seal` refuses it (`main`).
- Why at the end of the file, and not under a new signature. The eight reserved words of the header are taken. Placed after the table, the seal leaves intact all the bytes that precede it: a sealed file is the file without a seal followed by 48 bytes. A reader from before the seal, which stops at the table, therefore reads it without change, and the SHA-256 of a file without a seal is recovered by removing these 48 bytes. A new signature would have made the new files unreadable by the existing binaries and required touching every version test. Both properties were measured on 15 November 2026: the second at 64³ and at 256³, the first at 256³ with the binary of commit `240736a` of the development history (not public). The second is replayed at 64³ by the cases `d2.seal` and `d3.seal` of `tests/test_codec.py`, and at 256³ by `benchmarks/robustness/seal_256.py` (`seal_256.csv`); the reading of a sealed file by the earlier binary is not replayed by a script of the repository.
- Tools that count the frames from the size of the file. `(size - index_offset) / 48`, `size` being the size of the file, gives one frame more in a sealed file. The seal has the size of a `FrameRecord` and its last sixteen bytes are zero: this extra frame has `field_err = 0` and `field_max = 0`. An audit script of 15 November 2026 counted in this way and read only `field_max`: its output was the same for a sealed file and for the same file without a seal (checked at 256³ on two encodings; that script is not in the repository). The Python scripts of the repository that read the table (`tools/reference_reader.py`, `benchmarks/validity/validity.py`, `tests/test_codec.py`) read `n_frames` in the header and do not see the seal as a frame.
- Limit. The seal protects against an alteration, not against its own removal: stripped of its last 48 bytes, a sealed file is a valid file without a seal. Only `play --require-seal` refuses it.

## 5. Payload of a frame

Quantities derived from the header (`Geo::build`):

| Name | Value | Reference |
|---|---|---|
| `nodes[l]` | `nl[l]`³ | `Geo::nodes` |
| `ecount` | `3 * (nodes[0] + ... + nodes[L-1])`: coefficients of a frame | `Geo::ecount` |
| `seg[l]` | Position of the first coefficient of level `l` in the order of the innovations. The coarsest level comes first: `seg[L-1] = 0`, `seg[l] = seg[l+1] + 3 * nodes[l+1]`. | `Geo::seg` |
| `n_blocks` | `ecount / 1024`, rounded up | `Geo::n_blocks`, computed in two steps through `Geo::units`; `rans_pack` |
| `key_bytes` | `3 * nodes[0] + 6 * (nodes[1] + ... + nodes[L-1])` | `Geo::key_bytes`, with `Geo::koff` |
| `kcount` | Equal to `key_bytes`: one symbol per byte of state | `Geo::kcount`, with `Geo::khi`, `Geo::klo` and `Geo::kseg_end` |
| `n_kblocks` | `kcount / 1024`, rounded up | `Geo::n_kblocks` |
| `ex_cap` | `max(65536, ecount / 64)`: exceptions per frame, at most | `Geo::ex_cap` |

The code rounds `3 * nodes[0]` up to the next multiple of 8 (`Geo::build`, first value of `koff`). This rounding never acts: `nl[0]` is a power of two of at least 8, so `nodes[0]` is a multiple of 512.

Example, `n = 256`, `mf = 2`, `L = 3`: `nl` = 128, 64, 32; `ecount` = 7,176,192; `n_blocks` = 7008; `key_bytes` = 8,060,928, that is 25 times less than a raw frame of 201,326,592 bytes (the second figure of the "Keys" line of `pack`, that of a key written raw).

**Type 0, raw key** (writing `Sender::payload`, reading `Receiver::decode_levels`):

| Offset | Size | Content |
|---|---|---|
| 0 | `3 * nodes[0]` | State of level 0: one `int8` per coefficient. |
| next | `6 * nodes[l]` per level | States of levels 1 to `L - 1`, in this order: one `int16` per coefficient. |
| `key_bytes` | `8 + 8 * n_ex` | Exception list (6.3). Absent in `PSISTRM1`. |

**Type 3, coded key** (writing `Sender::payload`, reading `Receiver::decode_levels`):

| Order | Size | Content |
|---|---|---|
| 1 | `8 + 8 * n_ex` | Exception list. |
| 2 | `(2 * L - 1) * 512` | Frequency tables, one per segment (6.4). |
| 3 | `4 * n_kblocks` | Offsets of the rANS blocks. |
| 4 | variable | rANS streams of the `kcount` symbols. |

**Types 1 and 2, innovation** (writing `Sender::payload`, reading `Receiver::decode_levels`):

| Order | Size | Content |
|---|---|---|
| 1 | `8 + 8 * n_ex` | Exception list. Absent in `PSISTRM1` (`Receiver::inspect`). |
| 2 | `L * 512` | Frequency tables, one per level, from the coarsest to the finest (`Geo::lv_end`). |
| 3 | `4 * n_blocks` | Offsets of the rANS blocks. |
| 4 | variable | rANS streams of the `ecount` innovations. |

**Section of the fine level**, added at the end of every frame of a `PSISTRM3` file, whatever its type (writing `Sender::append_fine`, reading `Receiver::inspect` and `Receiver::decode_fine`):

| Order | Size | Content |
|---|---|---|
| 1 | 0 to 7 | Zero padding, up to a multiple of 8 from the start of the payload. |
| 2 | `3 * 512` | Frequency tables: brick map, masks, coefficients. |
| 3 | `4 * nblk` | Offsets of the blocks, `nblk = count / 1024` rounded up. |
| 4 | variable | rANS streams of the `count` bytes (6.6). |
| 5 | 0 to 7 | Zero padding, up to a multiple of 8. |
| 6 | 16 | `FineTrailer`: the last sixteen bytes of the payload. |

`FineTrailer` (the structure of that name): four `uint32`, with no padding.

| Offset | Field | Meaning |
|---|---|---|
| 0 | `n_bricks` | Bricks present in the frame. |
| 4 | `n_nodes` | Nodes retained in the frame. |
| 8 | `bytes` | Size of parts 2 to 5 above. Multiple of 8. |
| 12 | `reserved` | Zero. |

- `count = bricks / 8 + 8 * n_bricks + 3 * n_nodes`, with `bricks = (n / 4)`³ (`Receiver::inspect`, `Receiver::decode_fine`; `FineState::bm_bytes` for the first term).
- The reader finds the section from the end: it starts at offset `FrameRecord.bytes - 16 - FineTrailer.bytes` of the payload (`Receiver::inspect`).
- Variable sizes. The size of an rANS stream is written nowhere: it results from the coding. A block starts at its offset and the decoder reads what it needs (`rans_decode` in `src/codec_kernels.cuh`), 2052 bytes at most (6.1). The size of the whole frame is `FrameRecord.bytes`. In `PSISTRM3`, the end of the base part is known only to within the padding (0 to 7 bytes).
- Reading order in `PSISTRM3` (clarified). The stream area of the base part stops at the start of the section of the fine level, and this start can be found only from the end of the frame. `FineTrailer` must therefore be read first, the start of the section deduced from it, then the offsets of the base part checked against this bound. For a raw key, the expected size (`key_bytes`, plus the exception list, rounded to 8 bytes) must fall exactly on this start; this last point was not tested by the first reader, none of the files it read being a `PSISTRM3` with raw keys; the second reader has since read two such files (`c2_d2_bound_raw_keys` and `c2_d3_bound_raw_keys`, `benchmarks/independent_readers/python/complements.csv`). Cross-checked: the base part stops at the start of the section of the fine level, and a raw key, rounded to 8 bytes, must fall exactly on it (`Receiver::inspect`).
- The tables are per level or per segment. The streams are per block of 1024 symbols, not per level: a block can straddle two levels (6.1).
- Check at reading. Before launching a decoding kernel, the reader checks on the host that the payload is consistent with the geometry of the header (`Receiver::inspect`): exact size of a raw key, exception list, tables and offsets contained in the frame, section of the fine level contained. The rules are in section 9. `play` and `psi_view` make this check while the frame crosses the bus (`ingest` in `play`; `PsiSide::ingest` in `tools/viewer/psi_view.cu`).

## 6. Coding

### 6.1 rANS blocks (`rans_pack` in `src/codec_stream.cuh`, `rans_decode` in `src/codec_kernels.cuh`)

- The symbols are bytes, cut into blocks of 1024. The last block can be shorter (`rans_pack`).
- A frequency table is 512 bytes: 256 `uint16`, one per byte value. Their sum is exactly 4096, otherwise the reader refuses the frame (`check_rans_section`, `rans_tables`). A symbol that is present has a frequency of at least 1 (`rans_build` in `src/codec_kernels.cuh`). The sum of 4096 holds for every table, with no exception for a segment that has no symbol (completed on 17 November 2026, after the second reader: gap D3). For such a segment the encoder writes the frequency 4096 for the value 0 and zero for the 255 others (`rans_build`, same file, when the total is zero). The reader requires the sum of 4096 of every table of a section, whether or not a symbol is decoded with it (`check_rans_section`, then `rans_tables`); it does not require that particular table. Only two segments can be empty: the masks and the coefficients of the fine level, in a frame without a brick (6.6). The others always have symbols: `nl[l] >= 8` gives at least 1536 symbols per level, and `n >= 32` at least 64 bytes of brick map.
- A segment is a range of consecutive symbols that shares one table. When a block crosses the end of a segment, the decoder moves to the next table (`rans_decode` in `src/codec_kernels.cuh`).
- Framing: 8 bytes per block. Four bytes in the offset table: a `uint32`, counted from the start of the stream area, which follows this table (`rans_pack`). Four bytes at the head of the stream of the block: the initial state, little-endian (`rans_pack`; `rans_decode` in `src/codec_kernels.cuh`).
- State on 32 bits, lower bound 2^23, probabilities on 12 bits, renormalisation by bytes. The encoder goes through the symbols of the block in reverse (`rans_pack`).
- The code reserves 2056 bytes per block for the worst case (`block_cap` in `rans_pack`, `cap` in `Geo::build`). The renormalisation outputs at most 2 bytes per symbol, that is 2052 bytes with the state (deduced from the renormalisation loop of `rans_pack`, not measured).
- Reading bound of the decoder: 2052 bytes per block (`kRansBlockMax` and the comment that precedes it in `src/codec_kernels.cuh`), whatever the content of the stream. An initial state less than 2^23, which a valid stream does not produce, is brought back to 2^23 (`rans_decode`, same file). Before each symbol the state is then at least 2^23; after it, at least 2^11, because the frequency of a decoded symbol is at least 1; the renormalisation therefore reads two bytes at most. Without this floor, a zero state followed by zero bytes would make the decoder read without end. This is a reasoning on the code, tested by the altered files of the tests, not a measurement of the bound.
- Check of the offsets at reading (`check_rans_section`): the tables and the offset table fit in the frame; the first offset is zero; each offset exceeds the previous one by at least 4 bytes, the size of a state; the last is at least 4 bytes from the end of the area. The start of each block is therefore in the frame. The receive buffer extends 2056 bytes beyond the largest frame of the file (`kPacketMargin`, `packet_bytes`): a block cannot make the decoder read outside the buffer. What it reads after the end of its frame is the remainder of a previous frame.

Decoding of a block, with `cum[v]` = sum of the frequencies of the values less than `v`:

```
x = little-endian uint32 read at the start of the block; p points to the next byte
if x < 2^23: x = 2^23               (altered file only)
for each symbol of the block, in order:
    s   = x & 4095
    sym = the value v such that cum[v] <= s < cum[v] + freq[v]
    x   = freq[sym] * (x >> 12) + s - cum[sym]
    while x < 2^23: x = (x << 8) | *p++
```

The packet of innovations by classes (units of 16 coefficients) was never in the `.psi` files. Only the former bench program used it, and neither that program nor the packet is part of the published code. What remains of it is `Geo::units`, an intermediate of `Geo::n_blocks` and of the size of the symbol buffer.

### 6.2 Innovations (types 1 and 2)

- The symbol at position `i` is the innovation of coefficient `i`, a signed byte. `i` follows the order of the innovations: `seg[l] + 3 * node + component` (in `src/codec_kernels.cuh`: comment of `ExEntry`, `level_innov`).
- Prediction: `(a * q(k) + b * q(k-1) + c * lap(q(k)) + 512) >> 10`, where `lap` is the sum of the six neighbours minus six times the centre, taken periodically (`predict3` in `src/codec_kernels.cuh`). `q(k)` is the previous frame, `q(k-1)` the one before it (`Receiver::decode_levels`). In type 2, `(a, b, c) = (1024, 0, 0)` (`persist` in `Receiver::decode_levels`).
- State = prediction + innovation, bounded to ±127 at level 0 and to ±32767 at the other levels (±`32767 >> fb` in `PSISTRM1`) (`Receiver::decode_levels`, `Receiver::set_format`; `level_apply` in `src/codec_kernels.cuh`).
- Clarified. The decoded symbol is read as a signed byte, from -128 to 127. The shift `>> 10` is arithmetic: it rounds towards minus infinity, including for a negative value. The Laplacian is taken component by component, over the six neighbouring nodes of the same level. The innovations of the whole frame are applied first; the escapes of the exception list (6.3) set their state afterwards. Cross-checked: prediction `predict3` and application `level_apply`, both in `src/codec_kernels.cuh`; order of the two passes `Receiver::decode_levels`.
- Width of the sum (completed on 17 November 2026, after the second reader: gap D1). The whole expression is evaluated in the type `int`, a signed integer of 32 bits on the machines of section 1, by the encoder and by the receiver alike: both call the same function (`predict3` in `src/codec_kernels.cuh`, called by `level_innov` for the encoder and by `level_apply` for the receiver). The predictors are `int` (`Coef`, same file), copied as they are from the `int32` of the header; each state is converted to `int`; the Laplacian, the three products, their sum with 512 and the shift are `int`. No intermediate is wider, and the sum is neither bounded nor tested. It fits in 32 bits, and the formula then has its exact value, whenever `(|a| + |b| + 12 * |c|) * M + 512 <= 2^31 - 1` for the level, `M` being the largest modulus that a state of this level can have: 127 at level 0 and 32767 at the other levels in a file written by `pack`; 128 and 32768 in any file that the reader accepts, since a raw key, a coded key or an escape can carry the smallest value of the type (6.3). The predictor of type 2, (1024, 0, 0), always satisfies it; the condition concerns the frames of type 1, and it bears on the header alone. Where it holds, a reader that computes the exact sum, on 64 bits for instance, obtains the state of the encoder. Where it does not, the code defines nothing: the overflow of a signed integer is undefined behaviour in C++, so that the source says neither that the sum wraps modulo 2^32, nor that it saturates, nor anything else, and what the compiled kernels do was not examined (section 10, point 10). Nothing in the encoder or in the readers bounds the predictors (section 9).

### 6.3 Exception lists (`ExEntry` in `src/codec_kernels.cuh`, `Sender::write_ex` in `src/codec_stream.cuh`)

A list begins with 8 bytes: the number of entries `n_ex` (`uint32`), then 4 reserved bytes, zero. Then follow `n_ex` entries `ExEntry` of 8 bytes:

| Offset | Type | Field | Meaning |
|---|---|---|---|
| 0 | `uint32` | `idx` | Bits 0 to 30: position of the coefficient in the order of the innovations. Bit 31: kind of exception. |
| 4 | `int32` | `val` | According to the kind, see below. |

- Bit 31 at 0, escape: the innovation leaves ±127. The symbol emitted is -128 and `val` is the state to set. All levels (in `src/codec_kernels.cuh`: `level_innov`, `apply_state_ex`).
- Bit 31 at 1 (`kExFine = 0x80000000`), overflow: the target of level 0 leaves ±127. The state stays at ±127. `val` is the excess: the reader adds `val << fb` to the sampling lattice after its synthesis. Level 0 only (in `src/codec_kernels.cuh`: `level_key`, `level_innov`, `apply_lattice_ex`). The reader ignores an overflow whose index is not that of a coefficient of level 0 (`apply_lattice_ex`, same file).
- The entries are sorted by increasing `idx`, read as an unsigned integer: the escapes first, the overflows afterwards (`Sender::encode`). The reader counts the escapes as the first entries whose bit 31 is zero (`n_esc` in `Receiver::inspect`). It refuses `n_ex > ex_cap`, and a list that leaves the frame (`Receiver::inspect`).
- A key frame carries only overflows: the encoder produces no escape in it (`level_key` in `src/codec_kernels.cuh`), and the reader applies none (`Receiver::decode_levels`).
- Beyond `ex_cap`, the encoder adds no further exception: the coefficient is clipped and counted (`ex_push` in `src/codec_kernels.cuh`).
- Clarified. An overflow at position `idx` (bit 31 removed) affects the value at position `idx - seg[0]` of level 0, that is `3 * node + component`. `val << fb` is added to the sum on 32 bits of section 6.5, before its reduction modulo 2^16. Confirmed by the lattice of the probe of `d2_exceptions.psi`, a test file at 64³ whose probed frame carries overflows (`benchmarks/independent_readers/cpp/coverage_64.csv`). Cross-checked: the reader makes the addition after the synthesis, on the 16-bit value of the node, and takes the result modulo 2^16 again, which amounts to the same thing (`Receiver::decode_levels`; `apply_lattice_ex` in `src/codec_kernels.cuh`).
- Lists that a valid file does not contain (completed on 17 November 2026, after the second reader: gap D4). The reader checks the count and the extent of a list, not its entries (`Receiver::inspect`): neither their order, nor their positions, nor their values. What it does with a malformed entry is read in the code and not tested, no test file being known to carry one:
  - Which entries are escapes. The escapes are the entries that precede the first entry whose bit 31 is set (`n_esc` in `Receiver::inspect`). Every entry from that one to the end of the list is treated as an overflow, its bit 31 being masked and not tested (`apply_lattice_ex` in `src/codec_kernels.cuh`). An entry whose bit 31 is zero, placed after an overflow, is therefore not an escape and is not ignored: it is an overflow.
  - Escape whose position is not that of a coefficient (`idx >= ecount`): ignored. Otherwise the state is set to `val` reduced to the type of the state, with no bound: its low-order 8 bits read as a signed byte at level 0, its low-order 16 bits read as a signed integer at the other levels. The state can thus be -128 or -32768, which the encoder never writes. The symbol at that position is not consulted: it need not be -128 (`apply_state_ex` in `src/codec_kernels.cuh`). Conversely, a symbol -128 without an escape is an innovation of -128 (`level_apply`, same file).
  - Escapes of a key frame: skipped, as said above. The entries that follow them are overflows all the same.
  - Overflow whose position, bit 31 removed, is not that of a coefficient of level 0: ignored. Its value is not bounded: `val << fb` is added on 32 bits to the 16-bit value of the node, and the low-order 16 bits of the result are kept (`apply_lattice_ex` in `src/codec_kernels.cuh`).
  - Repeated positions. Two escapes at the same position: the state is the value of one of them, and which one is not defined (one thread per entry, with no order). Overflows on the same node that follow one another in the list are all added, including two overflows of the same coefficient. If they do not follow one another, several threads update the node at the same time and the result is not defined.
  - What reports it. An escape changes the state, so the fingerprint of the frame reports a malformed one unless it sets the very value of the encoder. An overflow changes the sampling lattice alone, which nothing covers (section 7).

### 6.4 Coded key: Lorenzo residual (`key_residual`, `key_scan_x` and `key_scan` in `src/codec_stream.cuh`)

- Residual of a coefficient: `q(x,y,z) - q(x-1) - q(y-1) - q(z-1) + q(x-1,y-1) + q(x-1,z-1) + q(y-1,z-1) - q(x-1,y-1,z-1)`. A neighbour outside the domain is zero: no periodicity here (`key_residual`).
- Level 0: one symbol, the residual modulo 2^8 (`key_residual`).
- Other levels: the residual modulo 2^16 is folded (0, -1, 1, -2 become 0, 1, 2, 3), then split into low-order byte and high-order byte (`key_residual`).
- Order of the `2 * L - 1` segments: for `l` from `L - 1` to 1, the high-order bytes of level `l`, then its low-order bytes; lastly level 0 (`Geo::build`: `khi`, `klo`, `kseg_end`).
- Decoding: three cumulative sums, along x, then y, then z, modulo 2^8 or 2^16 (`key_scan_x`, `key_scan`, `Receiver::decode_levels`). The result is exact: the residual requires no exception (comment of `key_residual`).
- The coded form is kept only if tables, offsets and streams fit in `key_bytes` bytes. Otherwise the key is written raw, of type 0 (`limit` in `rans_pack`, `Sender::payload`).
- Clarified. The residual is computed component by component: `q(x-1)` designates the same component of the neighbouring node, not the neighbouring value in the array. Folding of a residual `r`, read as a signed 16-bit integer: `z = 2 r` if `r >= 0`, `z = -2 r - 1` otherwise; conversely, `r = (z >> 1) XOR -(z AND 1)`, modulo 2^16. Inside a segment, the symbols follow the order of the states: `3 * node + component`. At level 0, the state is the result of the cumulative sums modulo 2^8, read as a signed byte. Cross-checked: residual and folding `key_residual`, unfolding and cumulative sums `key_scan_x` and `key_scan`.

### 6.5 Sampling lattice: written modulo 2^16, read directly or by differences

- The reader computes, from the coarsest level to level 0: `sum = (state << fb) + refinement of the sum of the coarser level`, on 32 bits. The refinement applies the integer masks of the B-splines (`Receiver::decode_levels`; in `src/codec_kernels.cuh`: `up_axis`, `up_at`, `synth_level`).
- Refinement (established by trial). `g` is the sum of level `l + 1`, of `m` nodes per axis; the result has `2 m` nodes per axis, those of level `l`. The indices are periodic. Along one axis:

  | Degree | Even node, `2i` | Odd node, `2i + 1` | Sum of the weights |
  |---|---|---|---|
  | 2 | `3 g[i] + g[i-1]` | `3 g[i] + g[i+1]` | 4 |
  | 3 | `g[i-1] + 6 g[i] + g[i+1]` | `4 g[i] + 4 g[i+1]` | 8 |

  The three-dimensional mask is the product of these masks over x, y and z, applied to the three components separately, with no intermediate division: each fine node receives an integer sum of 8 terms in degree 2, of 8, 12, 18 or 27 terms in degree 3, of total weight 64 in degree 2 and 512 in degree 3. A single division follows, rounded to nearest by an arithmetic shift: `(sum + 32) >> 6` in degree 2, `(sum + 256) >> 9` in degree 3. This is the two-scale relation of the B-spline, node `i` being at `(i + 1/2)` steps in degree 2 and at `i` steps in degree 3 (6.7). The limits of section 9 on `fb` guarantee that the sum before division fits in 32 bits. With rounding down, 32,723 nodes out of 32,768 differ from those of the encoder in degree 2, and 32,750 in degree 3; with the other alignment of the masks, all differ (measured at 64³: `benchmarks/independent_readers/cpp/conventions.txt`, script `conventions.py`). Cross-checked: these are the weights, the alignment and the single rounding of the code (`up_axis` and `up_at` in `src/codec_kernels.cuh`); the sum of the coarsest level is `state << fb`, with no refinement (`synth_level` and `fine_sum`, same file, when `up` is null).
- At level 0, the sum is written modulo 2^16 in a word of 16 bytes per node: slot 0, then slot 1. Each slot holds x, y, z on 16 bits and a proximity bit (in `src/codec_kernels.cuh`: `lat_get`, `lat_set`, `synth_lattice`).
- The support of a point is the set of the `(degree + 1)`³ nodes on which the velocity at this point depends (`sample` in `src/codec_kernels.cuh`). Only the differences between nodes of the same support matter. Taken modulo 2^16, a difference is exact as long as it stays under 2^15 (the comment that precedes `kLatNear`, same file).
- Direct reading: the kernels read the values as they are. It is exact if no value leaves the range of a 16-bit integer (`sample` for `PsiPlain` in `src/codec_kernels.cuh`).
- Reading by differences: if a node of the support carries the proximity bit (value less than 2^14 from the edge of the range), the kernel subtracts from each node the value of the first node of the support (in `src/codec_kernels.cuh`: `kLatNear`, `lat_set`, `lat_get`, `sample` for `PsiField`).
- The first node, the proximity bit and the wrap, exactly (completed on 17 November 2026, after the second reader: gap D2). The first node of the support is the node whose index, on each of the three axes, is the first of the support in the table of 6.7: `m - 1` in degree 2, `i - 1` in degree 3, taken modulo `nl[0]`. It is the index that `bspline` returns, and the first node of the traversal of 6.7 (`v0` in `sample` for `PsiField`, in `src/codec_kernels.cuh`). Once reduced modulo `nl[0]`, it is not always the node of smallest index: at the edge of the box its index can be `nl[0] - 1`. Each of its three components is subtracted from the same component of every node of the support, itself included, modulo 2^16, and the difference is read back as a signed 16-bit integer (`lat_get` with `v0`, same file). The lattice in memory holds two frames: each is read with the value that the first node has in its own slot. The proximity bit is not in the file: the reader computes it when it writes a node (`lat_set`, same file), one bit per node and per slot for the three components together. It is set if, for at least one component, bits 14 and 15 of the value written differ, that is if this value, read as a signed 16-bit integer, is at least 2^14 or strictly below -2^14: from 16384 to 32767, or from -32768 to -16385. The kernel that tests it reads the support directly, then reads it again by differences if at least one node of the support carries the bit in one of the two slots; the kernels chosen for a file without a wrapped frame do not test it (`sample` for `PsiPlain`, same file). A reader that subtracts the first node on every support, with no proximity bit, computes the same gradient as this kernel in exact arithmetic: where no node carries the bit, all the values of the support lie between -16384 and 16383, their differences are below 2^15 in modulus and, the sums of the `w'` being zero, subtracting the first node changes nothing; elsewhere the kernel subtracts it too. A frame is counted as wrapped when a component of the sum of level 0 has a modulus above 32767 before its reduction (`synth_lattice`, same file: a sum of -32768 counts, although it fits in 16 bits), or when the addition of an overflow (6.3) takes the 16-bit value of a node outside the range from -32768 to 32767 (`apply_lattice_ex`, same file). What the second reader tried on a file read by differences is in `benchmarks/independent_readers/python/README.md` and `complements.csv`.
- Flag in the file: the high-order 32 bits of `reserved[7]`, the number of wrapped frames. `play` reads directly if the file is `PSISTRM1`, or if `reserved[0]` is the marker and this number is zero. Otherwise it reads by differences (`plain` in `play`). The library makes the same choice (`ReaderT::init` in `src/psitreillis.cu`). `psi_view` always keeps the kernels that test the proximity bit (`PsiSide::advect` and `PsiSide::now` in `tools/viewer/psi_view.cu`).

### 6.6 Sparse fine level (`FineInfo` and `FineState` in `src/codec_stream.cuh`, `FineLevel` in `src/codec_kernels.cuh`)

- Fine grid: `n` nodes per axis, one per voxel. It is cut into bricks of 4³ nodes: `nb = n / 4` bricks per axis, of index `b = (bz * nb + by) * nb + bx` (`FineState::init`, `fine_brick_masks`; `kFineLog` and `kFineSide` in `src/codec_kernels.cuh`).
- The section codes three segments of bytes, in this order (`Sender::append_fine`, `Receiver::decode_fine`):
  1. Brick map: `bricks / 8` bytes. Brick `b` is present if bit `b mod 32` of the `uint32` word at position `b div 32` is 1, that is bit `b mod 8` of byte `b div 8` (`fine_brick_masks`, `fine_bit`).
  2. Masks: one `uint64` per brick present, by increasing `b`. Bit `k = (z * 4 + y) * 4 + x` says whether the node of local coordinates `(x, y, z)` is retained (`fine_brick_masks`, `fine_compact_masks`).
  3. Coefficients: three `int8` per retained node, in the order of the bricks then of the increasing bits `k`. Values from -127 to 127 (`fine_compact_coefs`, `fine_quantize`).
- A frame without a brick has `n_bricks = 0` and `n_nodes = 0`: the masks and the coefficients are then two empty segments. The section carries its three frequency tables all the same, each of sum 4096 (6.1; completed on 17 November 2026, after the second reader: gap D3).
- A node that is not retained is zero. The fine level of a frame depends on no other frame (the comment that precedes `FineInfo`).
- In a valid frame, the map counts exactly `n_bricks` bricks and the masks `n_nodes` nodes. In an altered frame they can count more: the reader treats what exceeds them as absent (`FineSlots`, `fine_fill`), instead of reading outside the masks and the coefficients or writing outside the blocks of nodes. The fingerprint of the frame then reports the difference.
- The reader adds to the field the curl of this second potential, at the scale `FineInfo.delta * n / 2π`. Node 0 is half a voxel from the origin in degree 2, at the origin in degree 3 (`FineState::level`; in `src/codec_kernels.cuh`: `fine_at`, `sample` for `PsiFine`).
- Clarified. The B-spline of the fine level has the degree of the header, like that of the lattice. The evaluation is described in 6.7. Cross-checked: `FineState::level`; `fine_slow` and `fine_at` in `src/codec_kernels.cuh`.

### 6.7 Velocity field at a point (established by trial, cross-checked with `bspline`, `node_cross`, `sample`, `fine_accum`, `fine_slow` and `fine_at` in `src/codec_kernels.cuh`)

No formula gave the velocity at a point. This one reproduces the field that the encoder records in its probe (section 11), and that of `psit sample`. The trials that decided between the possible readings are replayed by `benchmarks/independent_readers/cpp/conventions.py`; their figures, quoted below, are in `conventions.txt`, measured on four test files at 64³.

- Coordinate system. The box is periodic, of side 2π on each axis. A voxel measures `v = 2π / n`. The centre of the voxel of index `(i, j, k)` is at the point `((i + 1/2) v, (j + 1/2) v, (k + 1/2) v)`: it is the point of grid index `(i, j, k)` of the public interface (`docs/API.md`). Against `psit sample`, this reading leaves a difference of 2.7·10⁻⁶ u′ at most; centres at `i v` leave one of 0.43 u′.
- Nodes. A lattice of step `p` has its node of index `i` at `(i + d) p` on each axis, with `d = 1/2` in degree 2 and `d = 0` in degree 3. For the sampling lattice, `p = h0 = 2π / nl[0]`; for the fine level, `p = v`. In degree 2 the nodes of the fine level are therefore at the centres of the voxels, in degree 3 at their corners. Against the field of the probe, the other value of `d` leaves a difference of 1.5 u′ for the lattice and of 0.07 to 0.08 u′ for the fine level, instead of 6·10⁻⁶ u′ at most.
- Basis. The cardinal B-spline of the degree of the header, centred on its node. For a coordinate `x`, set `u = x / p - d`.

  | Degree | Nodes of the support | Weights `w` | Derivatives `w'`, with respect to `u` |
  |---|---|---|---|
  | 2 | `m - 1`, `m`, `m + 1`, with `m = floor(u + 1/2)` and `r = u - m` | `(1/2 - r)² / 2`, `3/4 - r²`, `(1/2 + r)² / 2` | `-(1/2 - r)`, `-2 r`, `1/2 + r` |
  | 3 | `i - 1`, `i`, `i + 1`, `i + 2`, with `i = floor(u)` and `t = u - i` | `(1 - t)³ / 6`, `(3 t³ - 6 t² + 4) / 6`, `(-3 t³ + 3 t² + 3 t + 1) / 6`, `t³ / 6` | `-(1 - t)² / 2`, `(3 t² - 4 t) / 2`, `(-3 t² + 2 t + 1) / 2`, `t² / 2` |

  The indices of the nodes are taken modulo the number of nodes per axis.
- Gradient of the potential. For the component `c` of Ψ, of coefficients `T[c]`: `dΨc/dx` is the sum, over the `(degree + 1)`³ nodes of the support, of `T[c] * w'x * wy * wz`; likewise `dΨc/dy` with `wx * w'y * wz` and `dΨc/dz` with `wx * wy * w'z`. These derivatives are taken with respect to `u`: the step `p` is in the scale below. The `T[c]` of the sampling lattice are read directly or by differences (6.5): by differences, the value of the first node of the support, which 6.5 defines (completed on 17 November 2026, after the second reader: gap D2), is subtracted from each node, modulo 2^16, and the difference is read back as a signed 16-bit integer; the sums of the `w'` being zero, the gradient does not change as long as no difference reaches 2^15.
- Curl. `curl(Ψ) = (dΨz/dy - dΨy/dz, dΨx/dz - dΨz/dx, dΨy/dx - dΨx/dy)`.
- Velocity. `u = mean + E * curl(sampling lattice) + Ef * curl(fine level)`, with `mean` the mean velocity of the frame in the table (section 4), `E = delta / (2^fb * h0)` (section 2) and `Ef = FineInfo.delta * n / 2π` (6.6). The last term exists only in `PSISTRM3`; a fine node that is not retained is zero. The two scales computed in this way are, to within 8·10⁻⁸ in relative value, those that the encoder writes in the header of its probe (`benchmarks/independent_readers/cpp/logs/reference_cpp.log`, "Velocity" lines of `psi_ref`).
- Precision. These formulas, evaluated in double precision, are within 2.8·10⁻⁵ u′ at most of the field of the GPU reader on the twelve reference files, and within 8·10⁻⁶ u′ at 64³. The GPU reader computes in single precision, in an order that this specification does not give: its field cannot be reproduced bit for bit (section 10, point 5).
- Cross-checked. The coordinate system, the position of the nodes (`Geo::build`, `FineState::level`), the weights and their derivatives (`bspline` in `src/codec_kernels.cuh`), the sign of the curl (`node_cross`, same file) and the two scales (`Geo::field_scale`, `FineState::level`) are those of the code.
- Order of the operations of the GPU reader, read in the code (in `src/codec_kernels.cuh`: `bspline`, `node_cross`, `sample`). Everything is in single precision. On each axis, `u = (x - d p) * (1 / p)`, where `d p` and `1 / p` are two constants rounded to single precision. The weights and the derivatives are evaluated by the expressions of the table, in the form of the code: in degree 3, for example, the second weight is `((3 t - 6) t t + 4) * (1/6)`. No division is made: the weights of degree 3 are multiplied by the constant `1/6` rounded to single precision, and the halves are multiplications by `0.5` or are carried into the coefficients (in degree 3 the second derivative is `(1.5 t - 2) t`). The nodes of the support are traversed with z in the outer loop, then y, then x; the products `wy wz`, `w'y wz` and `wy w'z` are computed once per pair (y, z). For each node, with `(gx, gy, gz) = (w'x wy wz, wx w'y wz, wx wy w'z)` and `(qx, qy, qz)` its three integers converted to floats, three accumulators receive `ax += gy qz - gz qy`, `ay += gz qx - gx qz`, `az += gx qy - gy qx`. The velocity is `E * a + mean`, component by component. The fine level is accumulated separately, in the same way, multiplied by `Ef`, then added to the result (`fine_slow`, `fine_at` and `sample` for `PsiFine`, same file). Between two frames, the advection kernels interpolate each integer and the mean velocity linearly between the two slots before these products; at the instant of a frame, the weight of the other slot is zero.
- What this order does not say. The compiler can fuse a multiplication and an addition into one operation rounded only once, and it does not make the same fusions in all the kernels: two kernels that evaluate this same field, the one that marks the fine level and the one that certifies the bound at encoding, already differ by 2.3·10⁻⁵ u′ at most (measured on one voxel, over 55 frames at 512³; the figure is recorded in the comment of `SOptions::tolerance_margin`). Replaying the field bit for bit would require the machine code of the kernel, not its source.

## 7. Fingerprint of a frame (`hash`)

- It is 64 bits. `pack` computes it on the state of the encoder after the encoding of the frame (`pack`, with `state_hash` and `fine_hash`).
- Fingerprint of a byte array `a` with a salt `s`: the exclusive OR, over all the positions `i`, of `mix64(s + 256 * i + a[i])`, in arithmetic modulo 2^64 (`hash_bytes`, `host_hash`).
- `mix64(x)`: `x += 0x9E3779B97F4A7C15`; `x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9`; `x = (x ^ (x >> 27)) * 0x94D049BB133111EB`; result `x ^ (x >> 31)` (`mix64`).
- Clarified. `a[i]` is the byte at position `i` read as unsigned, from 0 to 255: a coefficient of -1 at level 0 counts as 255. The states are taken in the order of section 1 (`3 * node + component`), the `int16` in little-endian. For the fine level, the three arrays are the decoded bytes of its section, in the order of 6.6: the map, then the masks (8 little-endian bytes per brick present), then the coefficients. Cross-checked: `host_hash`, `hash_bytes`.
- It covers, by exclusive OR of the fingerprints (`state_hash`, `fine_hash`):

| Array | Bytes | Salt |
|---|---|---|
| State of level 0 (`int8`) | `3 * nodes[0]` | `0x1000` |
| State of level `l >= 1` (little-endian `int16`) | `6 * nodes[l]` | `0x1000 + (l << 56)` |
| Brick map (`PSISTRM3`) | `bricks / 8` | `0x2000` |
| Masks (`PSISTRM3`) | `8 * n_bricks` | `0x2000 + (1 << 56)` |
| Fine coefficients (`PSISTRM3`) | `3 * n_nodes` | `0x2000 + (2 << 56)` |

- It covers neither the mean velocity, nor the bytes of the payload, nor the headers, nor the table. The headers and the table, mean velocity included, fall under the seal (section 4.1). Nor does it cover the overflows of level 0: they affect only the sampling lattice, not the state (`apply_lattice_ex` in `src/codec_kernels.cuh`). Nothing therefore covers the value of an overflow: altered, it changes the field without being seen. This is a known defect, which the case `guard.altered_overflow` of `tests/test_codec.py` reproduces.
- `play` recomputes the fingerprint on its own state after the decoding of each frame and compares it with that of the table (`ingest` in `play`). It displays the number of differing frames on the "Integrity" line and exits with code 1 if there are any (end of `play`).
- With `--no-fine` on a `PSISTRM3` file, nothing is checked (`ingest` in `play`). The counter is reset to zero before the reading of frame 0: the frames decoded for `--seek` do not count (`play`). `psi_view` does not check the fingerprint of the frames (`PsiSide::ingest` in `tools/viewer/psi_view.cu`). It verifies the seal and checks each frame against the geometry, like `play` (`open_side` and `PsiSide::ingest`, same file).

## 8. Compatibility

| Version | What it adds | What the reader does |
|---|---|---|
| `PSISTRM1` | Base: header, frames of types 0, 1, 2, table. States of the coarse levels bounded to ±`32767 >> fb`. | Accepted (`check_stream_header`). No exception list (`Receiver::v1`, `Receiver::inspect`). Type 3 refused (`read_stream_index`, `Receiver::inspect`). Direct reading (`plain` in `play`). The sums outside ±32767 are counted and reported (`Receiver::legacy_beyond`, `play`). |
| `PSISTRM2` | Exception list in each frame. Coarse levels on ±32767. Lattice modulo 2^16 and count of the wrapped frames. Coded keys, type 3. | Reads the lists (`Receiver::inspect`) and the four types. Direct reading or reading by differences according to `reserved` (`plain` in `play`). It is the default format of `pack` (`pack`). |
| `PSISTRM3` | `FineInfo` at offset 208. Section of the fine level and `FineTrailer` at the end of each frame. The fingerprint covers the fine level. | Reads `FineInfo` (`read_stream_index`) and the section of each frame (`Receiver::inspect`, `Receiver::decode_fine`). `play --no-fine` reads back the base alone (`SOptions::no_fine`, `play`). `pack` writes it if `--tolerance` is positive (`pack`). |

No option of `pack` writes a `PSISTRM1` file. `psi_view` reads the three versions with the same opening checks as `play` (`open_side` in `tools/viewer/psi_view.cu`). Options that reproduce an old format or an old behaviour:

| Option | Effect | Scope in the file |
|---|---|---|
| `--raw-keys` | Raw key frames, type 0, instead of coded ones (`SOptions::raw_keys`, `Sender::payload`). | Layout of the keys. |
| `--legacy-step` | Old formula of the quantisation step (`SOptions::legacy_step`, `pack`). | Value of `delta` only. |
| `--fit-at 0` | Predictors fitted on frames 1 to 4 of the sequence, as in the files written before the present default existed. By default: on the four frames that follow `--start` (`SOptions::fit_at`, `pack`). | Values of `coef` only. |
| `--fine-passes 12 --tolerance-margin 0` | Fine level from before the evening of 15 November: 12 marking passes at most, marking and certification at the tolerance itself (`SOptions::fine_passes`, `SOptions::tolerance_margin`). | Nodes retained; `FineInfo.reserved[0]` zero. |
| `--no-seal` | No seal after the table (`SOptions::seal`, `pack`). To be added to the options above to reproduce byte for byte a file written before the seal. | The last 48 bytes of the file, nothing else. |

- To reproduce a file earlier than 15 November: `--legacy-step --fit-at 0 --raw-keys`. To reproduce byte for byte a file with a bound earlier than the evening of 15 November: `--fine-passes 12 --tolerance-margin 0 --no-seal` (the variant `old` of `tests/test_codec.py`, cases `d2.fingerprint.old` and `d3.fingerprint.old`). Any file written before the seal requires `--no-seal`; otherwise it differs from the old one only by the 48 bytes added after the table.
- The header does not say which formula gave `delta`. It can be found by recomputing `delta = sqrt(12) * eps * u_rms * h0 / sqrt(gain * noise)`, with `gain = 2 * 0.55²` in degree 2 and `2 * (2/3) * (2416/5040)²` in degree 3. `noise = 1 + 4^-fb` with the current formula, and `1 + 1/4 + ... + 4^-(L-1)` with the old one (`gain2` and `noise` in `pack`). With the old one, the actual error is 0.900 `eps` for three levels and `fb = 2` (comment of `noise` in `pack`).
- In the last versioned `PSISTRM1` encoder (commit `b41a0c2` of the development history, not public; structure `FrameRecord` of `src/psi_stream.cu` as that commit has it), offset 44 of `FrameRecord` was a padding integer. `field_max` is therefore zero there, which the convention "zero if not measured" absorbs.
- Caveat on `PSISTRM1`. This encoder and its reader clipped the sums of the pyramid to ±32767 (same commit, kernels `synth_level` and `synth_lattice` of its `src/psi.cu`), which the comment of `StreamHeader::magic` recalls ("its reader clipped the lattice"). The current reader does not clip: it keeps the sums on 32 bits (`synth_level` and `fine_sum` in `src/codec_kernels.cuh`), writes the lattice modulo 2^16 (`synth_lattice` and `lat_set`, same file) and reads it directly. The old reader and the current reader give the same lattice only if no sum leaves ±32767. The fingerprint does not detect it (section 7). Since the evening of 15 November 2026, `play` counts these sums at each frame of a `PSISTRM1` file, on the state it has just decoded, and says so on a "PSISTRM1" or "Warning" line (`count_beyond`, `count_lattice_beyond`, `Receiver::legacy_beyond`, `play`). Measured that evening on the 32 `PSISTRM1` files then kept on the development machine, none of which is in the repository, any more than the log of that measurement: 29 have no sum outside ±32767, among them `turb256.psi`, an encoding of the sequence at 256³ in degree 2; 3 safeguard files at 256³, which their encoder had itself clipped, have some on all their frames (one of degree 2 with `mf` = 2 and three levels, one of degree 3 and one of degree 2 with `mf` = 1 and four levels). For these three, the field returned by the current reader is not that of the original reader. The current reader reports it; it does not restore the clipping.

## 9. Format limits guarded by the code

| Limit | Reference |
|---|---|
| `magic` among the three signatures | `check_stream_header` |
| `degree` = 2 or 3 | `check_stream_header` |
| `n` a power of two, from 16 to 8192 | `check_stream_header` |
| `n_frames` from 2 to 2^24 | `check_stream_header`; `play`, for the frames it is asked to read |
| `index_offset >= 208` | `check_stream_header` |
| Size of the file: `index_offset + 48 * n_frames`, or 48 bytes more with the seal; `index_offset` after the headers | `read_stream_index` |
| Seal, if it is present: `magic`, reserved words zero, three fingerprints | `read_stream_index` |
| Table: `type` from 0 to 3, no type 3 in `PSISTRM1`; each frame between the end of the headers and the table; frame 0 is a key | `read_stream_index` |
| `levels` from 1 to 6; `mf` a power of two | `Geo::build` |
| `nl[l] >= 8` for each level | `Geo::build` |
| `fb` from 0 to 12 | `check_codec_limits` in `src/codec_kernels.cuh` |
| `ecount < 2^31`: the position of a coefficient fits in 31 bits of `ExEntry.idx`, bit 31 being taken | `check_codec_limits` in `src/codec_kernels.cuh`, and its comment |
| `(L - 1) * (32768 << fb)` less than 2^25 in degree 2, than 2^22 in degree 3: the integer refinement does not overflow | `check_codec_limits` in `src/codec_kernels.cuh`, and its comment |
| `FrameRecord.bytes` at most equal to the worst case of the coder | `packet_bytes`, `cap` in `Geo::build`, `fine_section_cap`, `ingest` in `play` |
| `n_ex <= ex_cap`; exception list contained in the frame | `Receiver::inspect` |
| `type` from 0 to 3; no type 3 in `PSISTRM1` (also checked for each frame) | `Receiver::inspect` |
| Raw key: size exactly equal to `key_bytes`, plus the exception list outside `PSISTRM1`; rounded to 8 bytes in `PSISTRM3` | `Receiver::inspect` |
| rANS sections (frames of types 1, 2, 3 and fine level): tables and offsets contained in the frame; first offset zero; offsets increasing by at least 4 bytes; the last at least 4 bytes from the end of the area | `check_rans_section`, called by `Receiver::inspect` |
| Sum of a frequency table = 4096 | `check_rans_section`, `rans_tables` |
| Fine level: `brick = 4`; `n >= 32` | `read_stream_index`, `Receiver::init_fine`, `FineState::init` |
| Fine level: at most 2^24 bricks, that is `n <= 1024`. The block index of a brick fits in 24 bits in memory. | `FineState::init`; comment of `FineLevel` in `src/codec_kernels.cuh`, on the word `tab` |
| Section of the fine level: payload a multiple of 8 and of at least 16 bytes; `n_bricks` and `n_nodes` at most equal to the maxima of `FineInfo` (less than or equal; exact form below); `FineTrailer.bytes` a multiple of 8 and contained in the frame; tables and offsets contained in the section | `Receiver::inspect` |

- Computed consequences. `n` and `mf` being powers of two, `ecount < 2^31` imposes `nl[0] <= 512`. The comment of `check_codec_limits` in `src/codec_kernels.cuh` gives 856 nodes per axis for three levels, without this constraint. With three levels, the limit of the refinement imposes `fb <= 8` in degree 2 and `fb <= 5` in degree 3.
- The maxima of the fine level (completed on 17 November 2026, after the second reader: gap D6). The comparison is "less than or equal": a frame is refused if `n_bricks > B` or `n_nodes > N` (`Receiver::inspect`), with `B = max(1, min(max_bricks, (n / 4)³))` and `N = max(1, min(max_nodes, n³))` (`cap_bricks` and `cap_nodes` in `FineState::init`). In a file written by `pack`, `B` and `N` are `max_bricks` and `max_nodes` themselves unless these are zero, and at least one frame reaches each maximum. The form of `B` and `N` has two consequences, which only a header that `pack` did not write can show: with a maximum of zero, a frame that announces one brick, or one node, passes this check; a maximum larger than the grid is brought back to the size of the grid. The check is made only when the fine level is read: with `play --no-fine`, `n_bricks`, `n_nodes` and the tables of the section are not checked.
- Implicit limit, with no check: the block offsets and `FineTrailer.bytes` are `uint32`, so a stream area stays under 2^32 bytes. Not checked at reading: the paddings and the reserved fields of the frames (their value is ignored), and, in a file without a seal, `key_interval`, `eps` and `first_frame`. Frame 0 is a key: `--seek`, which goes back up to a key (`play`), therefore stops. An `offset` outside the frame area is refused at opening (`read_stream_index`).
- Limit that nothing guards: the predictors (completed on 17 November 2026, after the second reader: gap D1). The prediction of 6.2 is computed on 32 bits. In a frame of type 1, with the states that a reader can meet, its sum fits if `|a| + |b| + 12 * |c| <= 65535` at a level other than level 0, and `<= 16777211` at level 0; beyond that it can overflow, and the decoding is then not defined (6.2). No check enforces it. The encoder does not bound the coefficients it fits: `pack` rounds the solution of its least-squares system to 1/1024 and writes it, and refuses only a degenerate system (`pack`; `solve3` in `src/codec_kernels.cuh`). The readers do not bound them either: `check_stream_header` and `read_stream_index` do not look at `coef`, `Geo::build` and `check_codec_limits` (the latter in `src/codec_kernels.cuh`) do not receive it, and `play`, the library (`ReaderT::init` in `src/psitreillis.cu`) and `psi_view` (`PsiSide` in `tools/viewer/psi_view.cu`) hand it to `Receiver::init` as it is. The seal protects `coef` against an alteration, not against a value that overflows. A file whose predictors exceed these limits is therefore accepted, and the format does not say what its frames decode to. It is not known whether `pack` can fit such predictors on real data; for the files that the second reader read, the column `prediction_margin_32_bits` of `benchmarks/independent_readers/python/reference_python.csv` says how far they are from it.
- What these checks guarantee. An altered file gives either a refusal (message, code 1), or a complete read that counts the differing frames on the "Integrity" line. Measured at 64³ on 283 altered files (`benchmarks/robustness/altered_files.csv`, script `altered_files.py`; cases `guard.geometry_no_seal` and `guard.random_alteration` of `tests/test_codec.py`), 10 of which gave a CUDA error, an illegal memory access, with the previous reader (commit `240736a`; that comparison, made on 15 November 2026, is not replayed by a script of the repository). This is not demonstrated for every alteration: the reading bounds are reasoned on the code (6.1, 6.3, 6.6), and the altered files are tried only at 64³.
- What they do not guarantee. Without a seal, a field of the header or of the table that does not change the decoded state (`degree`, `fb`, `delta`, mean velocity of a frame) is read without error and gives another field. With or without a seal, the value of an overflow of level 0 is covered by nothing (section 7).

## 10. Points not established by reading

1. **Another machine.** The sizes and the offsets are established for the build machine: `static_assert` declarations assert them at the end of `src/codec_stream.cuh` (208, 48, 64, 16 and 8 bytes; `index_offset` at offset 136, `field_max` at offset 44), and the compilation passes. They are cross-checked by readers that share no code with the codec and are applied to real files: the two reference readers (`tools/reference_reader.cpp`, which reads each field at its offset, and `tools/reference_reader.py`, with the strings of section 1) and, before them, on 15 November 2026, two scripts that are not in the repository. The behaviour on a big-endian machine is not established: a file would be neither read nor written correctly there.
2. **Fidelity of the read-back of a `PSISTRM1` file.** Established by measurement, on the evening of 15 November 2026, for the 32 files present on this machine (section 8): 29 are read back with the lattice of their original reader, 3 are not. The encoder of this format is no longer in the tree; its layout is known from the reader and from commit `b41a0c2`. Not done: playing a file with the binary of this commit and comparing the fingerprint of the final positions. It would not settle the matter on its own, the reading kernels having changed since.
3. **`PSISTRM2` files without a marker.** It is not established that any exist (written before the clipping counters or before the coded keys). The reader would accept them and read them by differences. To find out: read the word at offset 144 of the files kept.
4. **Actual maximum size of an rANS block.** 2052 bytes by the reasoning of 6.1, 2056 reserved by the code. Not measured: one would have to record the largest difference between two successive offsets on real files. The safety of the reader does not depend on it: it rests on the reading bound of the decoder and on the check of the offsets (6.1), not on the size of the blocks written.

Points added on 16 November 2026 by the reference reader. They are not "not established by reading" the code, which this reader did not read: they are gaps of this text, which a trial on a file did not resolve.

5. **Arithmetic of the GPU reader.** The velocity field of the GPU reader is computed in single precision; the order of its operations is not described here. The formulas of 6.7, in double precision, depart from it by 1.5 to 2.8·10⁻⁵ u′ on the twelve reference files, and the difference grows with the size of the grid (`benchmarks/independent_readers/cpp/reference_cpp.csv`). Attributing it to the rounding is a hypothesis: it is not demonstrated. To settle it, write here the order of the operations of the kernel, then replay it in single precision on CPU. **Answer of the code.** The order of the operations is now written in 6.7. It is not enough to reproduce the field bit for bit: the fusions of multiplication and addition made by the compiler are not in the source. The hypothesis of the rounding therefore remains to be settled by the trial described, in single precision and in this order.
6. **Worst case of the coder.** Section 9 says that `FrameRecord.bytes` is at most equal to the worst case of the coder, without giving its formula. The reference reader bounds a frame by the frame area of the file, nothing more. **Answer of the code.** The worst case is `max(key_bytes, L * 512 + 2060 * n_blocks) + 8 + 8 * ex_cap`, plus, in `PSISTRM3`, `8 + 3 * 512 + 2060 * nblk + 8 + 16`, where `nblk` is `(bricks / 8 + 8 * max_bricks + 3 * max_nodes) / 1024` rounded up (`cap` in `Geo::build`, `fine_section_cap`; check in `packet_bytes`). 2060 is the 4 bytes of offset and the 2056 bytes reserved per block of 1024 symbols (6.1).
7. **First innovation of type 1.** `q(k-1)` is not defined for an innovation of type 1 placed at position 1, just after the key of frame 0. `pack` writes none: the frame that follows a key is of type 2. The reference reader then takes `q(k)`. **Answer of the code.** Frame `j` is decoded in slot `j & 1`, and `q(k-1)` is what this slot contained, read node by node just before being overwritten (`Receiver::decode_levels`; `level_apply` in `src/codec_kernels.cuh`). For an innovation of type 1 that followed a key, it would be the frame before the key in sequential reading, an unrelated content after a direct access, and the initial content of the slot at position 1. The GPU reader therefore does not take `q(k)` as the reference reader does: such a file has no defined decoding. `pack` writes none. Exactly (completed on 17 November 2026, after the second reader: gap D5): the receiver does not refuse such a frame. `read_stream_index` requires only that frame 0 be a key, and `Receiver::inspect` checks the type of a frame against the version of the file, not against the frame that precedes it. `Receiver::decode_levels` then predicts with whatever the slot `j & 1` holds: zeros if no frame has been decoded into this slot since the file was opened (`Receiver::init` sets the states of both slots to zero), which is the case at position 1 in a reading from the start; the frame `j - 2` in sequential reading; another frame after a direct access (`play --seek`, or the library called in any order). A reader written from this text may refuse the frame, or decode it in one of these ways: none is wrong, since the format defines none, and only a file without such a frame is read alike by all readers.
8. **Reading beyond an altered frame.** An rANS block of an altered file can read after the end of its frame (6.1); what it finds there depends on the buffer of the GPU reader. The reference reader reads zeros there. The two readers nevertheless return the same outcome and the same number of differing frames on the 84 safeguard files at 64³ (`benchmarks/independent_readers/cpp/coverage_64.csv`). **Answer of the code.** The receive buffer is not reset to zero between two frames: a block that reads after the end of its frame finds there the bytes of a frame received before, or the initial content of the buffer (`kPacketMargin`, `packet_bytes`; `Receiver::init`, which allocates the buffer without setting it). This content is not defined by the format; reading zeros there is a conforming reading.
9. **Drawing of the probe points.** The rule that places the base points of the probe (section 11) was not sought: their positions are in the file. **Answer of the code.** The 16,384 base points are drawn by `std::mt19937` with seed 4321: three uniform draws in `[0, 2π)` per point, for x, y, then z (`Sender::write_probe`). The sequence obtained depends on the standard library: the positions of the file are authoritative.
10. **Prediction sum beyond 32 bits** (completed on 17 November 2026, after the second reader: gap D1). Read in the code: the sum of 6.2 is an `int` of 32 bits, and nothing keeps it from overflowing (section 9). Not established: what the compiled kernels compute when it does. The source leaves it undefined; the machine code of `level_innov` and of `level_apply` (`src/codec_kernels.cuh`) was not examined, and no file exercises the case. To settle it, one would have to encode and read back a sequence with predictors forced beyond the limits of section 9; the answer would hold only for the build tried.

## 11. Probe file (`pack --probe`, `play --probe`)

Established as a black box, from the size and the content of probe files at 64³, 256³ and 512³, and read back by `tools/reference_reader.cpp` on the twelve reference files. It is not a part of the `.psi` file: it is the file that `pack --probe f.probe` writes alongside, and that `play --probe f.probe` reads back to compare its field with that of the encoder.

It carries a single frame: the last of the file in all the files seen. The help of `pack` announces an option `--probe-frame` to choose another; it has not been tried. Little-endian. `P` is the number of points. Cross-checked: `--probe-frame j` chooses the frame, counted from `--start`; by default, the last (`SOptions::probe_frame`, `pack`).

| Offset | Type | Content |
|---|---|---|
| 0 | `char[8]` | `PSIPROBE`, with no final zero |
| 8 | `uint32` x 4 | `n`; `degree`; `nl[0]`; probed frame, counted in the source sequence (`first_frame` + position in the file) |
| 24 | `uint32` x 4 | `P` (114,688 in all the files seen); number of base points, `P / 7`; fine level present, 0 or 1; 1 if the lattice of a frame encoded up to then has wrapped (6.5), 0 otherwise. Zero in all the files seen by the reference reader (cross-checked: `Sender::write_probe`, `pack`) |
| 40 | `float32` x 2 | `E`, the scale of the lattice (6.7); `Ef`, that of the fine level, zero without a fine level |
| 48 | `float32` x 2 | One eighth of a voxel, `2π / (8 n)`; `u'` |
| 56 | `float32` | Without a fine level: zero. With one: `(tolerance - margin) * u'`, the marking threshold in velocity units (verified on three files) |
| 60 | `float32` x 3 | Mean velocity of the probed frame |
| 72 | `int16` x `3 * nl[0]`³ | Sampling lattice of the encoder, modulo 2^16, in the order of section 1, overflows included |
| next | `int8` x `3 * n`³ | Only if the fine level is present: its coefficients on the dense grid of step one voxel, same order, zero for a node that is not retained |
| next | `float32` x `3 * P` | Positions: the `P` x coordinates, then the `P` y coordinates, then the `P` z coordinates, in the coordinate system of 6.7 |
| next | `float32` x `3 * P` | Velocities of the encoder at these points: the `P` x components, then y, then z |

- The size of the file is `72 + 6 * nl[0]³ + 3 * n³ + 24 * P` with a fine level, `72 + 6 * nl[0]³ + 24 * P` without.
- The points go in sevens: a base point in the box, then its neighbours at +1/8 and -1/8 of a voxel along x, y, then z. A neighbour can leave the box by a few thousandths: the field is periodic.
- The "Probe" line of `play` gives the largest difference between its field and these velocities, in velocity units, not as a fraction of `u'` (`probe_diff` in `play`; confirmed by trial on `guard_overflow.psi`, the file of the case `guard.altered_overflow` of the test suite: `benchmarks/independent_readers/cpp/README.md`).
- Cross-checked with the code: writing `Sender::write_probe`, reading `play`, comparison `ingest` in `play`. The eight integers of the header are `int32`. `play --probe` refuses a probe whose `n` or `degree` are not those of the file (`play`), skips the lattice and the fine level, and reads back only the positions and the velocities.
