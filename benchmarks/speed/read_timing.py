# SPDX-License-Identifier: Apache-2.0
"""Reading time at 512^3: decoding of one frame and one RK4 step, as `psi_stream play` times them (claim SPEED-1).

    python benchmarks/speed/read_timing.py [--repetitions 15] [--force]    measures; log read_timing.csv
    python benchmarks/speed/read_timing.py --summary                       the summary again, from the log, without measuring

What is measured. Two reference files at 512^3 in degree 3, 55 frames each, written by
benchmarks/compression/reference_files/reference_files.py: z_t512_d3.psi (no bound, hence no fine level) and
f_t512_d3_t05.psi (bound of 5 % of u', with its fine level). Each is read by

    psi_stream play --file build/work/reference_files/F.psi --particles 4000000 --sort --csv

Protocol: one warm-up read per file (disk cache; not recorded), then, at each repetition, one read of each file, the
two files alternating. The times are those that play measures itself and prints on its line "CSV,play", in
milliseconds: per frame, the read from disk, the transfer over the bus, the decoding, and the full budget; one RK4
step of the particles; the wall clock per frame; and the decoding of the key frames read in steady state (one per read
of these files: a value far more scattered than the others). Statistic: the median over the repetitions, per file and
per quantity; the summary also gives the minimum, the maximum and their difference.

The times are valid only on a free GPU. The script stops before any read if the utilisation given by
`nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader` is above 5 %, if a program of this repository computes
on the GPU, or if nvidia-smi cannot tell; and it stops before a later read if such a program has appeared. --force
overrides both, and the log then says that the GPU was not free.

What is written.
    read_timing.csv                 one line per read: repetition, file, whether the GPU was free at the start and its
                                    utilisation then, utilisation and memory in use just before the read, the times
                                    above, the number of key frames read in steady state, the frames whose state
                                    differs, the fingerprint of the final positions
    logs/warm_up_<file>.log         raw output of each warm-up read
    logs/read_<file>_<nn>.log       raw output of each recorded read
The log is written at the end only: a run that stops on the way (a read that fails, a busy GPU) leaves the previous log
untouched and says so. nvidia-smi gives a mean over its last sampling period: before every read but the first, the
utilisation recorded is the load of the read before it. The memory in use is that of the idle GPU as long as nothing
else runs.

Exit code: 0 if every read was recorded; 1 if a read fails, differs from the encoder or prints a line of another
shape; 2 if nothing was measured (prerequisite missing, GPU busy).

Needs build/psi_stream (tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere), the two files
above, an NVIDIA GPU and nvidia-smi. Python 3.8 or later, standard library only.
"""

import argparse
import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "compression"))
import psi_stream_runs as ps  # noqa: E402

FOLDER = "build/work/reference_files"
FILES = ["z_t512_d3.psi", "f_t512_d3_t05.psi"]
PARTICLES = "4000000"
LOG = HERE / "read_timing.csv"
COLUMNS = ["repetition", "file", "gpu_free_at_start", "gpu_utilisation_percent_at_start", "gpu_utilisation_percent_before_read",
           "gpu_memory_used_mib_before_read", "disk_ms", "bus_ms", "decoding_ms", "rk4_step_ms", "frame_ms", "wall_ms", "key_decoding_ms",
           "key_frames", "frames_differing", "final_positions"]
# Field of the line "CSV,play" (counted after the kind; see psi_stream_runs.py) -> column of the log.
PLAY_FIELDS = {7: "disk_ms", 8: "bus_ms", 9: "decoding_ms", 11: "rk4_step_ms", 13: "frame_ms", 14: "wall_ms", 17: "frames_differing",
               18: "final_positions", 19: "key_frames", 23: "key_decoding_ms"}
SUMMARISED = ["decoding_ms", "rk4_step_ms", "bus_ms", "key_decoding_ms", "disk_ms", "wall_ms"]


def read(file, log):
    """One read by play; returns the columns taken from its line "CSV,play". Stops the script if the read fails."""
    r = ps.run(["play", "--file", f"{FOLDER}/{file}", "--particles", PARTICLES, "--sort", "--csv"], log)
    try:
        if r.code != 0:
            raise ps.UnexpectedOutput(f"exit code {r.code}" + (f" ({ps.failure(r.error)})" if ps.failure(r.error) else ""))
        fields = ps.csv_line(r.output, "play")
        if fields[2] != PARTICLES or fields[4] != "1":
            raise ps.UnexpectedOutput(f'the line "CSV,play" gives {fields[2]} particles and a sort flag of {fields[4]}, {PARTICLES} sorted particles '
                                      "were asked: its fields are not the expected ones")
        for index in PLAY_FIELDS:
            if index not in (17, 18, 19):
                float(fields[index])
    except (ps.UnexpectedOutput, ValueError) as e:
        ps.stop(f"{r.command}: {e}. No time is recorded and {ps.relative(LOG)} is left as it was; raw output in {ps.relative(log)}.", 1)
    return {column: fields[index] for index, column in PLAY_FIELDS.items()}


def measure(repetitions, force):
    ps.require([ps.PSI_STREAM] + [f"{FOLDER}/{f}" for f in FILES],
               "Build the instruments (tools/build.ps1 -Tools) and run benchmarks/compression/reference_files/reference_files.py first.")
    gpu = ps.require_free_gpu(force)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)

    def before_read():
        ours, _ = ps.gpu_programs()
        if ours is None or ours:
            if not force:
                ps.stop(f"Another computation of this repository occupies the GPU ({'nvidia-smi silent' if ours is None else ', '.join(ours)}): "
                        f"no time is recorded and {ps.relative(LOG)} is left as it was.")
        return ps.gpu_utilisation(), ps.gpu_memory_used()

    for file in FILES:  # warm-up, not recorded
        before_read()
        read(file, logs / f"warm_up_{Path(file).stem}.log")
    rows = []
    for repetition in range(1, repetitions + 1):
        for file in FILES:
            utilisation, memory = before_read()
            row = dict(repetition=repetition, file=file, gpu_free_at_start="yes" if gpu["free"] else "no",
                       gpu_utilisation_percent_at_start="" if gpu["utilisation_percent"] is None else gpu["utilisation_percent"],
                       gpu_utilisation_percent_before_read="" if utilisation is None else utilisation,
                       gpu_memory_used_mib_before_read="" if memory is None else memory)
            row.update(read(file, logs / f"read_{Path(file).stem}_{repetition:02d}.log"))
            rows.append(row)
            print(f"repetition {repetition:2d}, {file}: decoding {row['decoding_ms']} ms, RK4 step {row['rk4_step_ms']} ms", flush=True)
    with open(LOG, "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary():
    """Prints the medians; returns 1 if a read of the log has a frame whose state differs, 0 otherwise."""
    if not LOG.is_file():
        ps.stop(f"{ps.relative(LOG)} does not exist: measure first.")
    with open(LOG, newline="", encoding="utf-8") as fi:
        rows = list(csv.DictReader(fi))
    if not rows or set(COLUMNS) - set(rows[0]):
        ps.stop(f"{ps.relative(LOG)} is empty or does not have the columns that this script writes.", 1)
    repetitions = max(int(r["repetition"]) for r in rows)
    print(f"Reading time at 512^3, {PARTICLES} sorted particles, {repetitions} repetitions ({ps.relative(LOG)})")
    print("Per file and per quantity, in ms: minimum / median / maximum over the repetitions, then range (maximum - minimum).")
    differing_reads = 0
    for file in sorted({r["file"] for r in rows}, key=[r["file"] for r in rows].index):
        series = [r for r in rows if r["file"] == file]
        positions = sorted({r["final_positions"] for r in series})
        differing = sum(1 for r in series if int(r["frames_differing"]) != 0)
        differing_reads += differing
        print(f"== {file}")
        print(f"   fingerprint of the final positions: {', '.join(positions)} ({len(positions)} value(s) over {len(series)} reads); "
              f"reads with differing frames: {differing}")
        for quantity in SUMMARISED:
            values = [float(r[quantity]) for r in series]
            print(f"   {quantity:16s} {min(values):.3f} / {statistics.median(values):.3f} / {max(values):.3f}   range {max(values) - min(values):.3f}")
    if any(r["gpu_free_at_start"] != "yes" for r in rows):
        print("WARNING: the GPU was not free at the start of this run (--force): these times must not be quoted.")
    return 1 if differing_reads else 0


def main():
    parser = argparse.ArgumentParser(description="Reading time at 512^3: decoding of one frame and one RK4 step.")
    parser.add_argument("--repetitions", type=int, default=15, help="recorded reads per file (default: 15, the number that claim SPEED-1 states)")
    parser.add_argument("--force", action="store_true", help="measure even if the GPU is busy; the log then says so")
    parser.add_argument("--summary", action="store_true", help="print the summary from the log, without measuring")
    arguments = parser.parse_args()
    if arguments.repetitions < 1:
        parser.error("--repetitions: at least 1")
    if not arguments.summary:
        measure(arguments.repetitions, arguments.force)
    return summary()


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
