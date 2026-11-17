# Measurements

Every number of the article, of `README.md` and of the claims sheet is in a log of this folder. Each subfolder holds
one measurement: the script that makes it, the logs it wrote, and a notice (`README.md`) that says what is measured and
why, how to replay it, what the script writes, and the caveats. All the logs were produced by the code of this
repository as published, on the machine recorded in `environment.txt`. The `Repository` section of that file names a
commit of the development history, which is not public: the measurements were made there, before this repository was
published. The binaries it fingerprints were built from the sources published here; one comment line of
`tools/psi_stream.cu` was edited after the build.

| Folder | What it measures | Claims | In the article |
|---|---|---|---|
| `data/` | The sequences of turbulence: how to produce them, and what the solver recorded | | table of the data |
| `divergence/` | Divergence of a frame as each compressor gives it back, on the grid and as read | GRID-1, READ-1, READ-2 | figure and table of the divergence |
| `tracers/` | Tracers advected in a frozen frame seen through each chain: clustering, volume carried | LAGR-1 to LAGR-5, TRAC-1 | figure and table of the tracers |
| `playback/` | Cost of one replayed frame: data read, GPU memory, time; divergence of the field read | PLAY-1, DIV-1 | table of the cost |
| `equal_bound/` | Compression against SZ3, SPERR, cuSZp and zfp at an equal bound, on the same frames | BENCH-1 to BENCH-4 | figure and table of the price |
| `validity/` | Resolution sweep on 18 sequences: where the bound can be held | VALID-1 to VALID-5 | figure of the validity map |
| `spectra/` | Energy spectrum of a decoded frame | SPEC-1 | one paragraph |
| `compression/` | The twelve reference files (ratio, error, bound, read-back, encoding time) and the under-resolved sequence | COMP-1, COMP-2, BOUND-1 to BOUND-4, READBACK-1, ENC-1, RESO-1, RESO-2 | compression, bound and speed |
| `speed/` | Decoding time of a frame and time of one RK4 step | SPEED-1 | compression, bound and speed |
| `robustness/` | The reader faced with altered files; the seal | ROBUST-1 | limits |
| `independent_readers/` | Two readers written from the specification alone, compared with the encoder | REF-1, REF-2 | reproducibility |
| `interface/` | Logs of the two test suites; errors read back at 512³ through the public interface | TEST-1, API-1, API-2 | reproducibility |
| `cutout/` | A protocol to measure the codec on a windowed, non-periodic cutout, and the prepared trial on JHTDB | CTRL-1; JHTDB-3, JHTDB-4, REPR-1 (pending) | limits |
| `linux_build/` | Record of the workflow that builds the project on Linux without a GPU | LINUX-1 | limits |
| `codecs/` | Build of the comparison compressors, and the one way every script calls them | | data, compressors and machine |

## Replaying everything

```
python tools/environment.py                 # the machine, the toolchain, the revisions of the compressors
python tools/claims.py                      # every claim against the logs (no GPU)
python tools/paper_numbers.py               # the numbers and tables of the article, and README.md
python tools/figures.py                     # the figures
```

The measurements themselves are replayed one folder at a time, with the commands of each notice. Compression ratios,
errors, exit codes and fingerprints are deterministic on a given toolchain and can be compared digit for digit;
timings cannot, and are measured with the GPU running nothing else.
