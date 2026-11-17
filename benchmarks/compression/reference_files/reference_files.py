# SPDX-License-Identifier: Apache-2.0
"""The twelve reference files: encoded by `psi_stream pack`, then read back by `psi_stream play`
(claims COMP-1, COMP-2, BOUND-1 to BOUND-4, READBACK-1 and ENC-1).

    python benchmarks/compression/reference_files/reference_files.py [--force]

What is measured. Ten files with a bound on the largest error (`pack --tolerance`) and two without, in degree 2 and 3,
on the two reference sequences:

    f_t256_d3_t05, _t04, _t03    data/turb256.seq, frames 16 to 72, degree 3, bound of 5, 4 and 3 % of u'
    f_t256_d2_t06, _t05          data/turb256.seq, frames 16 to 48, degree 2, bound of 6 and 5 %
    f_t512_d3_t05, _t04, _t03    data/turb512.seq, frames 0 to 54, degree 3, bound of 5, 4 and 3 %
    f_t512_d2_t07, _t06          data/turb512.seq, frames 0 to 54, degree 2, bound of 7 and 6 %
    z_t512_d2, z_t512_d3         data/turb512.seq, frames 0 to 54, degree 2 and 3, no bound

Every other setting of pack is its default. The fidelity is checked by pack on all the voxels of every frame. The
files with a bound carry an off-grid probe (`--probe`): the field of the encoder at points off the grid, in one frame,
which play compares with its own. The files are written without a seal (`--no-seal`), so that their bytes stay those
of the reference files measured so far. Each file is read back by play with one million sorted particles.

    psi_stream pack --data SEQUENCE --file F.psi [--probe F.probe] --csv --no-seal --degree D --start S --frames N [--tolerance T]
    psi_stream play --file F.psi --particles 1000000 --sort [--probe F.probe]

What is written.
    reference_files.csv     one line per file: exit codes of pack and play, size and SHA-256 of the file, compression
                            ratio, encoding time per frame, field error, largest error, tolerance, certification
                            threshold, frames above this threshold, frames whose state differs at read-back,
                            fingerprint of the final positions, largest difference at the probe, memory of the reader
                            and of the receiver state, and whether the GPU was free at the start
    logs/pack_<file>.log    raw output of each pack
    logs/play_<file>.log    raw output of each play
    logs/gpu.log            state of the GPU at the start and at the end
The encoded files and their probes go to build/work/reference_files/ (several GiB); other measurements read them
there (benchmarks/speed/, benchmarks/robustness/).

An exit code 4 of pack (bound missed) or 1 (refusal) is a result: it is recorded, and the script goes on. The script
itself exits with code 0 if every run gave the lines it reads, 1 if a line is missing or has another shape (the list
is printed at the end), 2 if nothing was run.

The sizes, errors, exit codes and fingerprints do not depend on the load of the machine; the encoding time does. The
script therefore stops if the GPU is busy (benchmarks/compression/psi_stream_runs.py, require_free_gpu); --force
overrides this, and the column gpu_free_at_start then says "no".

Needs build/psi_stream (tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere), data/turb256.seq
and data/turb512.seq, an NVIDIA GPU. Python 3.8 or later, standard library only.
"""

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import psi_stream_runs as ps  # noqa: E402

WORK = "build/work/reference_files"
PARTICLES = "1000000"

D2_256 = ["--degree", "2", "--start", "16", "--frames", "33"]
D3_256 = ["--degree", "3", "--start", "16", "--frames", "57"]
D2_512 = ["--degree", "2", "--start", "0", "--frames", "55"]
D3_512 = ["--degree", "3", "--start", "0", "--frames", "55"]

# (file, sequence, options of pack, tolerance or None), in the order of the runs.
FILES = [
    ("f_t256_d3_t05", "data/turb256.seq", D3_256, "0.05"),
    ("f_t256_d3_t04", "data/turb256.seq", D3_256, "0.04"),
    ("f_t256_d3_t03", "data/turb256.seq", D3_256, "0.03"),
    ("f_t256_d2_t06", "data/turb256.seq", D2_256, "0.06"),
    ("f_t256_d2_t05", "data/turb256.seq", D2_256, "0.05"),
    ("f_t512_d3_t05", "data/turb512.seq", D3_512, "0.05"),
    ("f_t512_d3_t04", "data/turb512.seq", D3_512, "0.04"),
    ("f_t512_d3_t03", "data/turb512.seq", D3_512, "0.03"),
    ("f_t512_d2_t07", "data/turb512.seq", D2_512, "0.07"),
    ("f_t512_d2_t06", "data/turb512.seq", D2_512, "0.06"),
    ("z_t512_d2", "data/turb512.seq", D2_512, None),
    ("z_t512_d3", "data/turb512.seq", D3_512, None),
]


def commands(name, sequence, options, tolerance):
    """The arguments of pack and of play for one file."""
    file, probe = f"{WORK}/{name}.psi", f"{WORK}/{name}.probe"
    if tolerance is None:  # no bound: no probe
        return (["pack", "--data", sequence, "--file", file, "--csv", "--no-seal"] + options,
                ["play", "--file", file, "--particles", PARTICLES, "--sort"])
    return (["pack", "--data", sequence, "--file", file, "--probe", probe, "--csv", "--no-seal"] + options + ["--tolerance", tolerance],
            ["play", "--file", file, "--particles", PARTICLES, "--sort", "--probe", probe])


def main():
    parser = argparse.ArgumentParser(description="Encodes the twelve reference files and reads them back.")
    parser.add_argument("--force", action="store_true", help="run even if the GPU is busy; the log then says so")
    arguments = parser.parse_args()

    ps.require([ps.PSI_STREAM] + sorted({f[1] for f in FILES}),
               "Build the instruments (tools/build.ps1 -Tools) and produce the sequences of data/ first.")
    gpu = ps.require_free_gpu(arguments.force)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    gpu_log = [f"start: {ps.gpu_line(gpu)}"]

    problems = []
    with open(HERE / "reference_files.csv", "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=ps.COLUMNS, lineterminator="\n")
        writer.writeheader()
        for name, sequence, options, tolerance in FILES:
            pack_arguments, play_arguments = commands(name, sequence, options, tolerance)
            row, found = ps.pack_then_play(name, pack_arguments, play_arguments, logs)
            row["gpu_free_at_start"] = "yes" if gpu["free"] else "no"
            writer.writerow(row)
            fo.flush()
            problems += found
            print(ps.shown(row), flush=True)

    gpu_log.append(f"end: {ps.gpu_line(ps.gpu_state())}")
    (logs / "gpu.log").write_text("\n".join(gpu_log) + "\n", encoding="utf-8", newline="\n")
    print(f"Log: {ps.relative(HERE / 'reference_files.csv')}; raw outputs in {ps.relative(logs)}/")
    if problems:
        print(f"{len(problems)} PROBLEM(S): the log is incomplete or cannot be trusted for these runs:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
