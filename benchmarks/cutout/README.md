# Windowed cutout: trying the codec on a sub-volume that is not periodic (claim CTRL-1)

## What is measured, and why

Psi-treillis requires a periodic, divergence-free field. A cutout of a larger field, such as those that the public
JHTDB database lets one download, is neither at its edges. This folder holds a protocol for trying the codec on such a
cutout all the same, the control of that protocol on our own data, and the scripts of a trial on JHTDB that was
prepared but could not be carried out.

**The protocol.**

1. Multiply the 256^3 cutout by a smooth window: 1 at the centre, a half-cosine over 32 voxels at each edge.
2. Project the result onto the periodic divergence-free fields (Leray projection, by FFT).
3. Compress that field, and measure the error only on the central cube of 160^3, where the window is 1.

At the centre, the projection adds to the original field only the gradient of a harmonic potential, which is smooth:
the small scales there should remain those of the original field. That is a hypothesis, and the control tests it.

**The control (claim CTRL-1).** On `data/turb512.seq` and `data/turb512_nu0006.seq`, the cutout of corner
(128, 128, 128) of frames 0 to 7 is windowed and projected, then compressed by `psi_stream pack` in degree 3 at a
lattice step of 2 voxels, without a bound and with a bound of 5 % of u'. The whole field at 512^3 is compressed with
the same settings on the same frames. Both files are decoded through the public Python module of the library, and the
error of each is measured on the same voxels, those of the central cube of the cutout: the windowed cutout against the
windowed field, the whole field against the original. The RMS error is relative to the RMS of the reference on that
cube; the largest error is divided by the u' of the original sequence. The criterion of CTRL-1, written before the
measurement, bounds the difference between the two errors (`docs/CLAIMS.md`).

Outside the criterion, the windowed cutout is also compressed at a lattice step of 1 voxel, and by SPERR on frames 0, 3
and 7 at the same absolute bound (compression ratio only). No timing is logged.

**The trial on JHTDB (claims JHTDB-3, JHTDB-4 and REPR-1): not done.** The same protocol was to be applied to a 256^3
cutout of the dataset `isotropic1024coarse`, on 8 consecutive stored instants. Access to the database was obtained.
The transfers of the cutout broke repeatedly, and no complete frame was retrieved. The trial on JHTDB is therefore not
done: `jhtdb.csv` does not exist, and the three claims remain pending. `download_jhtdb.py` and `cutout_jhtdb.py` are
published as they were prepared; the first has never run to its end, the second has never run on data.

## How to replay

From the root of the repository, on Windows with a CUDA GPU:

```powershell
.\tools\build.ps1 -Tools                           # build\psi_stream.exe, build\psitreillis.dll
.\benchmarks\codecs\build_codecs.ps1 -Only sperr   # build\codecs\sperr\
python benchmarks\cutout\cutout.py control         # needs data\turb512.seq and data\turb512_nu0006.seq
```

The trial on JHTDB, for whoever has an access token of the database and a link that holds:

```powershell
$env:JHTDB_TOKEN = '<token>'
<python of an environment with givernylocal> benchmarks\cutout\download_jhtdb.py
python benchmarks\cutout\cutout.py jhtdb
```

`download_jhtdb.py` asks the database, with its official client `givernylocal`, for the cutout of corner
(384, 384, 384) at stored instants 1001 to 1008, in blocks of 64^3, one request at a time, with a pause between two
requests and spaced retries. It writes each block to disk as soon as it is received and resumes where it stopped.
`cutout.py jhtdb` then applies the protocol to the 8 frames.

## What the scripts write

| File | Content |
|---|---|
| `control.csv` | Written by `cutout.py control`, from scratch at each run: one line per sequence, field, setting and bound |
| `jhtdb.csv` | Written by `cutout.py jhtdb`: the same lines for the cutout of JHTDB. Does not exist: the trial is not done |
| `logs/pack_<name>.log` | Output of each run of `psi_stream pack`, after its command |

Columns of `control.csv` and `jhtdb.csv`:

| Column | Meaning |
|---|---|
| `source` | `turb512`, `turb512_nu0006` or `jhtdb_isotropic1024coarse` |
| `eta_voxels` | Kolmogorov scale of the source in voxels, from its header; empty for JHTDB |
| `u_rms` | u' of the source: from the header of the sequence; for JHTDB, measured on the raw cutout |
| `field` | `whole field` (the sequence at 512^3) or `windowed cutout` |
| `setting` | `d3_mf2`, `d3_mf1` (degree 3, lattice step of 2 voxels or of 1 voxel) or `sperr` |
| `bound_over_urms` | Bound on the largest error as a fraction of u'; empty without a bound |
| `exit_code` | Exit code of `psi_stream pack`; 0 for SPERR |
| `compression_ratio` | Raw size over the size of the file (8 frames, without a seal); for SPERR, over three frames |
| `centre_rms_error_relative` | RMS of the error over the RMS of the reference, on the central cube, over the 8 frames |
| `centre_largest_error_over_urms` | Largest error of a component on the central cube, over u' |
| `note` | For SPERR, the frames measured |

The windowed sequences and the `.psi` files are written in `build/work/cutout/` and deleted once measured. The frames
downloaded from JHTDB go to `data/cutout/jhtdb/`, which is not versioned.

## Caveats

- The control rests on two sequences, a single cutout per sequence, 8 frames and a single window width.
- CTRL-1 bears on the error only. The compression ratio of a windowed cutout is logged, but its edges are almost zero
  by construction: it says nothing about the compression ratio on a whole field, and can only be compared with that of
  another codec applied to the same windowed cutout.
- Psi-treillis is measured here on a file of 8 frames, one of them a key frame; SPERR frame by frame on frames 0, 3
  and 7. On 8 frames the key frame weighs more than in a group of 32: these compression ratios cannot be compared with
  those of `benchmarks/equal_bound/`.
- The windowed sequence carries the u' of the original sequence in its header: the bound of 5 % is 5 % of that u',
  not of the RMS of the windowed field.
- Our simulations truncate their spectrum at n / 3. A cutout of a dataset whose spectrum goes further has not been
  tried: that is the subject of REPR-1, which is pending.
- Whatever its result, the trial on JHTDB would not give the compression ratio of Psi-treillis on the whole field of
  1,024^3 (the encoder does not handle that size, and a compression ratio measured on a windowed cutout is not that of
  a whole field), nor anything on more than 8 frames.
- The access token of the database is read from the environment variable `JHTDB_TOKEN` and from nowhere else. The
  scripts write it nowhere, and `download_jhtdb.py` redacts it from the error messages of the client before printing
  them.
