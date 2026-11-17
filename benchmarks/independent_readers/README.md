# Independent readers of the `.psi` format (claims REF-1 and REF-2)

Two readers of `.psi` files were each written from the specification of the format, `docs/FORMAT.md`, and not from
the codec. This folder holds the scripts that run them on the reference files and on the files of the test suite, and
the logs of what they give.

| Folder | Reader | Claim |
|---|---|---|
| `cpp/` | `tools/reference_reader.cpp`, executable `psi_ref`: C++17, on CPU, with neither CUDA nor any library | REF-1 |
| `python/` | `tools/reference_reader.py`: Python 3 and NumPy, on CPU | REF-2 |

## What "independent" means here

Each reader was written by a working session that had the specification and did not read the sources of the codec:
neither the encoder, nor the GPU reader, nor the library. The session of the second reader read neither the first
reader nor its logs. The programs of the codec were used only as black boxes: their help, their output, the `.psi`
files they write, and the probe that `psi_stream pack --probe` records beside a file (the field of the encoder at
points off the grid, `docs/FORMAT.md`, section 11). Each notice lists what its session read and what it did not.

Independent does not mean without any contact with the codec:

- the first reader needed trials against files written by the encoder to settle what the specification of the time
  did not give for the velocity field (`cpp/README.md`, `cpp/conventions.py`); these conventions were then carried
  into the specification;
- the second reader was written from the specification thus completed, which another reading had by then
  cross-checked against the code. Its session kept, during the writing, the list of the passages where the text left
  a doubt (`python/gaps.csv`, `python/writing_journal.md`);
- both compare their velocity field with the probe that the encoder writes, not with a field computed apart.

## What each establishes, and what it does not

**The first reader (`cpp/`, REF-1)** says whether the state of every frame of the reference files can be decoded from
the specification by a program that shares no code with the codec: each recomputed fingerprint is compared with that
of the frame table. It also compares its sampling lattice, its fine level and its velocity field with those of the
encoder at the probe, and its outcome on every file of the test suite, valid or altered, with that of the GPU reader.
It does not establish that the specification of the time was sufficient for the velocity field: it was not, and the
trials that filled the gaps are replayed by `cpp/conventions.py`.

**The second reader (`python/`, REF-2)** says whether the completed specification can be read without the code and
without any trial to decide between readings: the reader was committed before its first run on a file, and the list
of its doubts was kept before any trial. It does not rediscover the conventions that the first reader had to
establish: it reads them in the text.

Neither reader establishes anything about the speed of the format, nor about the arithmetic of the GPU reader, which
computes in single precision where both compute in double precision: the difference that remains at the probe is
recorded, and its origin is not established. Neither observes the velocities of the GPU reader: `psi_stream play`
gives only its largest difference from the same probe. The comparison of the fields bears on one frame per file.

The figures are in the logs of each folder; `tools/claims.py` renders the verdicts of REF-1 and REF-2 from them.

## How to replay

```
python benchmarks/compression/reference_files/reference_files.py   # the twelve reference files and their probes
python tests/test_codec.py                                         # the test files at 64^3
python benchmarks/independent_readers/cpp/reference_cpp.py
python benchmarks/independent_readers/cpp/conventions.py
python benchmarks/independent_readers/python/reference_python.py
```

The notices `cpp/README.md` and `python/README.md` say what each script runs, what it needs, what it writes, and the
caveats.
