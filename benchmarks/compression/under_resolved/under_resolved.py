# SPDX-License-Identifier: Apache-2.0
"""Psi-treillis on an under-resolved turbulence: the encodings of the reference files, on a sequence at 512^3 whose
Kolmogorov scale is 0.45 voxel (claims RESO-1 and RESO-2).

    python benchmarks/compression/under_resolved/under_resolved.py

What is measured. data/turb512_nu0006.seq is a turbulence at 512^3 of viscosity 0.0006: its Kolmogorov scale is
0.45 voxel, against 0.74 voxel for data/turb512.seq (the header of each sequence gives kmax * eta, see
benchmarks/data/). The same segment as the reference files at 512^3 (frames 0 to 54) is encoded with the same
settings, then read back:

    d2, d3                    degree 2 and 3, no bound
    d3_t05, d3_t04, d3_t03    degree 3, bound of 5, 4 and 3 % of u' on the largest error, with an off-grid probe
    d2_t07                    degree 2, bound of 7 %, with an off-grid probe

    psi_stream pack --data data/turb512_nu0006.seq --file F.psi --start 0 --frames 55 --csv --degree D [--tolerance T] [--probe F.probe]
    psi_stream play --file F.psi --particles 4000000 --sort --csv [--probe F.probe]

Then three files without a bound are encoded with a probe, to bracket a difference between the field of the reader and
that of the encoder that the probe shows in degree 2 on this sequence: the same sequence in degree 2 and in degree 3,
and data/turb512.seq in degree 2.

    probe_nu0006_d2, probe_nu0006_d3    data/turb512_nu0006.seq, degree 2 and 3
    probe_turb512_d2                    data/turb512.seq, degree 2

    psi_stream pack --data SEQUENCE --file F.psi --probe F.probe --start 0 --frames 55 --csv --degree D
    psi_stream play --file F.psi --particles 1000000 --sort --probe F.probe

Unlike the reference files, these nine files are written with their seal (the default of pack).

What is written.
    under_resolved.csv      one line per file, with the columns of benchmarks/compression/reference_files/reference_files.csv
    logs/pack_<file>.log    raw output of each pack
    logs/play_<file>.log    raw output of each play
    logs/gpu.log            state of the GPU at the start and at the end
The encoded files and their probes go to build/work/under_resolved/ (several GiB).

The exit code of pack is the result that claim RESO-2 reads: 0, the bound is held; 1, pack refuses and stops (it does so
when the tolerance asks the fine level to refine more than a quarter of the volume) and the file it leaves has no
frame table, which play must refuse in its turn; 4, the file is written but the bound is missed. All three are
recorded, with the message of the program, and the script goes on. The script itself exits with code 0 if every run gave the lines it reads, 1 if a line is missing
or has another shape (the list is printed at the end), 2 if nothing was run.

The sizes, errors, exit codes and fingerprints do not depend on the load of the machine; the times do, and no claim
reads a time in this log. The script does not wait for a free GPU: it records its state (column gpu_free_at_start and
logs/gpu.log).

Needs build/psi_stream (tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere),
data/turb512_nu0006.seq and data/turb512.seq, an NVIDIA GPU. Python 3.8 or later, standard library only.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import psi_stream_runs as ps  # noqa: E402

WORK = "build/work/under_resolved"
UNDER_RESOLVED = "data/turb512_nu0006.seq"
RESOLVED = "data/turb512.seq"
SEGMENT = ["--start", "0", "--frames", "55"]

# The encodings of the reference files: (file, options of pack, with a probe).
CASES = [
    ("d2", ["--degree", "2"], False),
    ("d3", ["--degree", "3"], False),
    ("d3_t05", ["--degree", "3", "--tolerance", "0.05"], True),
    ("d3_t04", ["--degree", "3", "--tolerance", "0.04"], True),
    ("d3_t03", ["--degree", "3", "--tolerance", "0.03"], True),
    ("d2_t07", ["--degree", "2", "--tolerance", "0.07"], True),
]
# Files without a bound, with a probe: (file, sequence, options of pack).
PROBE_CASES = [
    ("probe_nu0006_d2", UNDER_RESOLVED, ["--degree", "2"]),
    ("probe_nu0006_d3", UNDER_RESOLVED, ["--degree", "3"]),
    ("probe_turb512_d2", RESOLVED, ["--degree", "2"]),
]


def commands():
    """(file, arguments of pack, arguments of play) of every run, in order."""
    runs = []
    for name, options, with_probe in CASES:
        file, probe = f"{WORK}/{name}.psi", f"{WORK}/{name}.probe"
        pack = ["pack", "--data", UNDER_RESOLVED, "--file", file] + SEGMENT + ["--csv"] + options
        play = ["play", "--file", file, "--particles", "4000000", "--sort", "--csv"]
        if with_probe:
            pack += ["--probe", probe]
            play += ["--probe", probe]
        runs.append((name, pack, play))
    for name, sequence, options in PROBE_CASES:
        file, probe = f"{WORK}/{name}.psi", f"{WORK}/{name}.probe"
        runs.append((name, ["pack", "--data", sequence, "--file", file, "--probe", probe] + SEGMENT + ["--csv"] + options,
                     ["play", "--file", file, "--particles", "1000000", "--sort", "--probe", probe]))
    return runs


def main():
    if len(sys.argv) > 1:
        print(__doc__)
        return 2
    ps.require([ps.PSI_STREAM, UNDER_RESOLVED, RESOLVED], "Build the instruments (tools/build.ps1 -Tools) and produce the sequences of data/ first.")
    gpu = ps.gpu_state()
    print(f"GPU at the start: {ps.gpu_line(gpu)}", flush=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    gpu_log = [f"start: {ps.gpu_line(gpu)}"]

    problems = []
    with open(HERE / "under_resolved.csv", "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=ps.COLUMNS, lineterminator="\n")
        writer.writeheader()
        for name, pack_arguments, play_arguments in commands():
            row, found = ps.pack_then_play(name, pack_arguments, play_arguments, logs)
            row["gpu_free_at_start"] = "yes" if gpu["free"] else "no"
            writer.writerow(row)
            fo.flush()
            problems += found
            print(ps.shown(row), flush=True)

    gpu_log.append(f"end: {ps.gpu_line(ps.gpu_state())}")
    (logs / "gpu.log").write_text("\n".join(gpu_log) + "\n", encoding="utf-8", newline="\n")
    print(f"Log: {ps.relative(HERE / 'under_resolved.csv')}; raw outputs in {ps.relative(logs)}/")
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
