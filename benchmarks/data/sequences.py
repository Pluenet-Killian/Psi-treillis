# SPDX-License-Identifier: Apache-2.0
"""Records the header of each sequence of data/: the table of the data used by the measurements.

    python benchmarks/data/sequences.py

Writes sequences.csv next to this script: one row per sequence of data/ and data/sweep/, with what the solver wrote
in its header (grid, number of frames, time between frames, viscosity, RMS velocity, kmax * eta, Taylor-scale Reynolds
number) and the Kolmogorov scale in voxels deduced from it (the solver truncates at kmax = n / 3). It reads the 72
bytes of each header, nothing else, and runs no program.
"""

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "codecs"))
import comparison_codecs as cc  # noqa: E402

COLUMNS = ["sequence", "grid", "frames", "frame_interval", "viscosity", "u_rms", "kmax_eta", "eta_voxels", "re_lambda", "gib"]


def main():
    files = sorted(cc.DATA.glob("*.seq")) + sorted((cc.DATA / "sweep").glob("*.seq"))
    if not files:
        sys.exit("no sequence in data/: see README.md for the commands that produce them")
    with open(HERE / "sequences.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(COLUMNS)
        for path in files:
            h = cc.read_header(path)
            row = [path.relative_to(cc.DATA).as_posix(), h["n"], h["frames"], f"{h['dt']:.7g}", f"{h['nu']:.7g}", f"{h['u_rms']:.7g}",
                   f"{h['kmax_eta']:.7g}", f"{cc.eta_in_voxels(h):.4f}", f"{h['re_lambda']:.6g}", f"{path.stat().st_size / 2 ** 30:.1f}"]
            w.writerow(row)
            print("  ".join(str(x) for x in row))


if __name__ == "__main__":
    main()
