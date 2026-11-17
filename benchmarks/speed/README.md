# Reading time at 512³: decoding of one frame and one RK4 step

## What is measured and why

Claim SPEED-1 states how long the reader takes, at 512³ in degree 3, to decode one frame and to advance four million
sorted particles by one RK4 step, without a bound and with a bound of 5 % of u′. This measurement reads two of the
reference files (`benchmarks/compression/reference_files/`), 55 frames each:

| File | Content |
|---|---|
| `z_t512_d3.psi` | degree 3, no bound: no fine level |
| `f_t512_d3_t05.psi` | degree 3, bound of 5 % of u′: with its sparse fine level |

```
psi_stream play --file build/work/reference_files/F.psi --particles 4000000 --sort --csv
```

Protocol:

1. one warm-up read per file, not recorded (it fills the disk cache);
2. 15 repetitions; at each repetition, one read of each file, the two files alternating;
3. the times are those that `play` measures itself and prints on its line `CSV,play`, in milliseconds: per frame, the
   read from disk, the transfer over the bus, the decoding and the full budget; one RK4 step of the particles; the wall
   clock per frame; the decoding of the key frames read in steady state;
4. statistic: the median over the repetitions, per file and per quantity. The summary also gives the minimum, the
   maximum and their difference, which is the scatter against which a difference of medians must be read.

`play` exits with a non-zero code if the state of a frame differs from that of the sender: such a read stops the
script, and no time is recorded.

## How to replay

From the root of the repository, on a GPU that runs nothing else:

```
# writes the two files, if they are not there
python benchmarks/compression/reference_files/reference_files.py
# measures, then prints the summary
python benchmarks/speed/read_timing.py
# prints the summary again, from the log, without measuring
python benchmarks/speed/read_timing.py --summary
```

`--repetitions N` changes the number of repetitions (15 is the number that the claim states). Before any read, the
script asks `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` for the utilisation of the GPU and
`nvidia-smi --query-compute-apps=process_name --format=csv,noheader` for the programs that compute on it. It stops,
having run nothing, if the utilisation is above 5 %, if a program of this repository is among them, or if `nvidia-smi`
cannot tell; it stops before a later read if such a program has appeared. `--force` overrides both checks, and the log
then says that the GPU was not free.

Needs `build/psi_stream`, the two files, an NVIDIA GPU and `nvidia-smi`. Python 3.8 or later, standard library only;
Windows and Linux.

## What the script writes

| Path | Content |
|---|---|
| `read_timing.csv` | one line per recorded read, the columns below |
| `logs/warm_up_<file>.log` | the raw output of each warm-up read |
| `logs/read_<file>_<nn>.log` | the raw output of each recorded read, `nn` being the repetition |

| Column | Meaning |
|---|---|
| `repetition`, `file` | the read |
| `gpu_free_at_start` | `yes`, or `no` if the script was forced on a busy GPU |
| `gpu_utilisation_percent_at_start` | utilisation of the GPU seen before the first read, in percent |
| `gpu_utilisation_percent_before_read`, `gpu_memory_used_mib_before_read` | utilisation and memory in use just before this read |
| `disk_ms`, `bus_ms`, `decoding_ms` | per frame: read from disk, transfer over the bus, decoding |
| `rk4_step_ms` | one RK4 step of the particles (there are two per frame) |
| `frame_ms` | full budget of a frame: disk, bus, decoding, sort and the RK4 steps |
| `wall_ms` | wall clock per frame |
| `key_decoding_ms`, `key_frames` | decoding of a key frame read in steady state, and how many were read |
| `frames_differing` | frames whose state differs from that of the sender |
| `final_positions` | fingerprint of the final positions of the particles |

The times are copied as `play` prints them (four decimals). The log is written at the end only: a run that stops on
the way leaves the previous log untouched, and says so. The medians are not stored: `--summary` computes them from the
log.

## Caveats

- One machine. The times are valid only if the GPU ran nothing else: `gpu_free_at_start` must read `yes`, and the
  summary warns if it does not.
- `nvidia-smi` gives a mean over its last sampling period. Before every read but the first, the utilisation recorded
  is therefore the load of the read before it; it is the value seen at the start that tells whether the GPU was free.
  The memory in use, for its part, is that of the idle GPU as long as nothing else runs.
- The disk cache is warm: `disk_ms` is not the time of a cold read.
- There is one key frame per read of these files in steady state: `key_decoding_ms` is far more scattered than the
  other times.
- Not measured here: degree 2, particles that are not sorted, 256³, the reader of `psi_view`.
- The two files must be those of the current encoder: write them again with `reference_files.py` after a change of
  the encoder or of the format.
