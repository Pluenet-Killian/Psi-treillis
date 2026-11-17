# SPDX-License-Identifier: Apache-2.0
"""Checks at 512^3, through the public interface, the errors of the reference files (claim API-2).

Each chosen frame is decoded to the grid by the library (psitreillis.Reader.decode), then compared in NumPy with the
raw frame of the sequence. This is another code path than that of the encoder: a kernel that decodes to the grid,
written for the library, and a comparison made on the host in double precision.

    python benchmarks/interface/bounds_512/verify_bounds_512.py

Needs: NumPy; the library of build/ (tools/build.ps1); data/turb512.seq; the six reference files at 512^3 of
build/work/reference_files/ (benchmarks/compression/reference_files/reference_files.py): f_t512_d3_t03, f_t512_d3_t04,
f_t512_d3_t05, f_t512_d2_t07 (with an error bound), z_t512_d3 and z_t512_d2 (without). The GPU is used for the
decoding only. No timing is recorded.

Writes, next to this script:
    verify_bounds_512.csv   one row per frame checked
    verify_bounds_512.log   the same rows between a line that names the library and the sequence, and the count

Columns of the CSV. file: reference file, without its extension. frame: position of the frame in the file.
sequence_frame: its position in the sequence. key_frame: 1 if the file holds it as a key frame. has_bound, bound_held:
what the header of the file says. tolerance: bound of the file, as a fraction of u' (0 without a bound).
largest_error, rms_error: measured here; largest_error_recorded, rms_error_recorded: recorded by the encoder in the
frame table. The largest errors are fractions of u', the RMS velocity of the header of the sequence; the RMS errors
are fractions of the RMS of the field of the frame, the three components together.

Verdict of a frame: "consistent" if the largest error measured is within 5e-5 u' of the recorded one, if the RMS error
measured is within 0.02 % of the recorded one and if, for a file whose bound is held, the largest error measured does
not exceed the tolerance; "MISMATCH" otherwise.

Exit code: 0 if every frame is consistent, 1 otherwise, 2 if a file needed is missing.
"""

import csv
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "benchmarks" / "codecs"))
import comparison_codecs as cc  # noqa: E402

SEQUENCE = "data/turb512.seq"
REFERENCE_FOLDER = "build/work/reference_files"
FILES = ("f_t512_d3_t03", "f_t512_d3_t04", "f_t512_d3_t05", "f_t512_d2_t07", "z_t512_d3", "z_t512_d2")
# The first frame, which is a key frame; the innovation that follows it; frame 33, the first innovation after the key
# frame 32; the last frame of the file.
FRAMES = (0, 1, 33, 54)
ROUNDING = 5e-5            # half the certification margin, in u': what two GPU kernels may differ by
RMS_TOLERANCE = 2e-4       # relative difference allowed between the RMS error measured here and the recorded one

COLUMNS = ["file", "frame", "sequence_frame", "key_frame", "has_bound", "bound_held", "tolerance", "largest_error",
           "largest_error_recorded", "rms_error", "rms_error_recorded", "verdict"]


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    os.chdir(ROOT)  # every path below is relative to the root of the repository
    missing = [p for p in [SEQUENCE] + [f"{REFERENCE_FOLDER}/{name}.psi" for name in FILES] if not Path(p).is_file()]
    if missing:
        print("missing: " + ", ".join(missing) + " (data/turb512.seq is produced by dns, the reference files by "
              "benchmarks/compression/reference_files/reference_files.py)", file=sys.stderr)
        return 2
    try:
        psi = cc.psitreillis()
    except ImportError:
        print("missing: the library of build/ (tools/build.ps1)", file=sys.stderr)
        return 2

    header = cc.read_header(SEQUENCE)
    n, frames, u_rms = header["n"], header["frames"], header["u_rms"]
    raw = np.memmap(SEQUENCE, dtype=np.float32, mode="r", offset=cc.SEQ_HEADER_BYTES, shape=(frames, 3, n, n, n))

    lines = [f"psitreillis {psi.version()}; sequence {SEQUENCE}, {n}^3, u' = {u_rms:.6f}", ",".join(COLUMNS)]
    print("\n".join(lines), flush=True)
    rows = []
    mismatches = 0
    for name in FILES:
        with psi.open(f"{REFERENCE_FOLDER}/{name}.psi") as f:
            info = f.info
            for frame in FRAMES:
                decoded = f.decode(frame)
                largest, square_error, square_field = 0.0, 0.0, 0.0
                for c in range(3):
                    original = raw[info.first_frame + frame, c]
                    error = decoded[c].astype(np.float64) - original
                    largest = max(largest, float(np.abs(error).max()))
                    square_error += float((error * error).sum())
                    square_field += float((original.astype(np.float64) ** 2).sum())
                largest /= u_rms
                rms = (square_error / square_field) ** 0.5
                recorded = f.frame_info(frame)
                as_recorded = (abs(largest - recorded.max_error) <= ROUNDING
                               and abs(rms - recorded.rms_error) <= RMS_TOLERANCE * recorded.rms_error)
                within_bound = not info.has_bound or not info.bound_held or largest <= info.tolerance
                verdict = "consistent" if as_recorded and within_bound else "MISMATCH"
                mismatches += verdict != "consistent"
                rows.append([name, frame, info.first_frame + frame, int(recorded.is_key), int(info.has_bound), int(info.bound_held),
                             f"{info.tolerance:.4f}", f"{largest:.6f}", f"{recorded.max_error:.6f}", f"{rms:.6e}",
                             f"{recorded.rms_error:.6e}", verdict])
                lines.append(",".join(str(v) for v in rows[-1]))
                print(lines[-1], flush=True)
    lines.append(f"{len(rows)} frames checked, {mismatches} mismatched")
    print(lines[-1])

    with open(HERE / "verify_bounds_512.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    with open(HERE / "verify_bounds_512.log", "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
