# Bench at an equal bound

## What is measured, and why

How much Psi-treillis compresses, and with what error, against general-purpose error-bounded compressors that are given
the same bound: SZ3 (frame by frame and in 4D), SPERR, cuSZp and zfp (claims BENCH-1, BENCH-2 and BENCH-3 of
`docs/CLAIMS.md`). This is the price of the method: the claims sheet records where it compresses less.

Protocol, fixed before the measurement:

- every codec receives the same absolute bound on the largest error of a component: a fraction of u′, the RMS velocity
  of the header of the sequence;
- every codec is measured on the same 32 frames: frames 16 to 47 of `turb256.seq`, frames 0 to 31 of `turb512.seq`.
  The frame-by-frame codecs are run on each of the 32 frames, and their compression ratio is the total raw size over
  the total compressed size. Psi-treillis encodes the 32 frames as one group. SZ3 in 4D takes them as one block at
  256³ and as four blocks of 8 frames at 512³, where a block of 32 does not fit in 63 GiB of RAM;
- the errors of every codec, Psi-treillis included, are measured in the same way: the decoded frames are compared
  with the raw frames in NumPy, in double precision, on every voxel. The file of Psi-treillis is written and decoded
  through the public Python module of the library.

How each compressor is called is written at the top of `benchmarks/codecs/comparison_codecs.py`: its own tool or
interface, default settings, nothing tuned but the bound.

## How to replay

```
python benchmarks/equal_bound/equal_bound.py 256
python benchmarks/equal_bound/equal_bound.py 512
python benchmarks/equal_bound/equal_bound.py summary
```

Needs `build/` (`tools/build.ps1`), the comparison compressors in `build/codecs/`
(`benchmarks/codecs/build_codecs.ps1`), NumPy, and the two sequences in `data/`.

## What the script writes

`equal_bound_256.csv` and `equal_bound_512.csv`: one row per codec and per bound, with the compression ratio, the bits
per value, the relative RMS error, the largest error over u′ and the PSNR (peak = value range of the original).

## Caveats

- **zfp is not at an equal attained bound.** Its fixed-accuracy mode leaves the largest error well under the bound
  requested; its rows are logged for the record.
- The compressors run with their default settings. SPERR runs on one thread and by chunks of 256³, as the build of
  `benchmarks/codecs/` allows on Windows: a 256³ frame is one chunk, a 512³ frame is eight.
- No timing is recorded: the compressors are called as external tools on a machine that is not reserved.
- One sequence per grid size. `benchmarks/validity/` repeats the comparison with SPERR and SZ3 in 4D on 18 sequences.
- SZ3 in 4D is measured on blocks of 8 frames at 512³, Psi-treillis on one group of 32.
