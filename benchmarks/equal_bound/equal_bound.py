# SPDX-License-Identifier: Apache-2.0
"""Bench at an equal bound: Psi-treillis against SZ3 (frame by frame and in 4D), SPERR, cuSZp and zfp.

    python benchmarks/equal_bound/equal_bound.py 256      turb256.seq, frames 16 to 47; log equal_bound_256.csv
    python benchmarks/equal_bound/equal_bound.py 512      turb512.seq, frames 0 to 31; log equal_bound_512.csv
    python benchmarks/equal_bound/equal_bound.py summary  the tables, from the logs

Protocol (fixed before the measurement):
  - every codec receives the same absolute bound on the largest error of a component, tolerance x u', u' being the RMS
    velocity of the header of the sequence;
  - every codec is measured on the same 32 frames. The frame-by-frame codecs are run on each of the 32 frames: their
    compression ratio is the total raw size over the total compressed size. Psi-treillis encodes the 32 frames as one
    group (one key frame, 31 predicted frames). SZ3 in 4D takes the 32 frames as one block at 256^3, and at 512^3 four
    consecutive blocks of 8 frames, because 63 GiB of RAM do not hold a block of 32;
  - the errors of every codec, Psi-treillis included, are measured here in the same way: the decoded frames are
    compared with the raw frames in NumPy, in double precision, on every voxel. The file of Psi-treillis is written
    and decoded through the public Python module of the library;
  - zfp is logged like the others but it is not at an equal attained bound: its fixed-accuracy mode leaves the largest
    error well under the bound requested;
  - no timing is recorded: the compressors are called as external tools, and the machine is not reserved.

Needs build/ (tools/build.ps1), build/codecs/ (benchmarks/codecs/build_codecs.ps1) and the sequences of data/.
"""

import csv
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "codecs"))
import comparison_codecs as cc  # noqa: E402

COLUMNS = ["grid", "codec", "bound_over_urms", "frames", "block_frames", "exit_code", "compression_ratio", "bits_per_value",
           "rms_error_relative", "largest_error_over_urms", "psnr_db", "note"]

# (sequence, first frame, bounds of Psi-treillis, bounds of the other codecs, frames per block of SZ3 in 4D)
BENCHES = {
    "256": dict(sequence="turb256.seq", first=16, psi_bounds=[None, 0.05, 0.04, 0.03, 0.015], bounds=[0.05, 0.04, 0.03, 0.015], block=32),
    "512": dict(sequence="turb512.seq", first=0, psi_bounds=[None, 0.05, 0.04, 0.03], bounds=[0.05, 0.04, 0.03], block=8),
}
FRAMES = 32
FRAME_CODECS = ["sz3", "sperr", "cuszp", "zfp"]


def row(grid, codec, bound, first, block, code, raw_bytes, compressed_bytes, errors, u_rms, note):
    measured = errors is not None
    return [grid, codec, "none" if bound is None else f"{bound:g}", f"{first}-{first + FRAMES - 1}", block, code,
            f"{raw_bytes / compressed_bytes:.3f}" if measured else "", f"{32.0 * compressed_bytes / raw_bytes:.5f}" if measured else "",
            f"{errors.relative_rms():.6e}" if measured else "", f"{errors.largest / u_rms:.5e}" if measured else "",
            f"{errors.psnr_db():.3f}" if measured else "", note]


def run(grid):
    b = BENCHES[grid]
    sequence = cc.DATA / b["sequence"]
    header = cc.read_header(sequence)
    n, u_rms, first = header["n"], header["u_rms"], b["first"]
    raw_bytes = FRAMES * 3 * n ** 3 * 4
    folder = cc.work_folder("equal_bound")
    log = HERE / f"equal_bound_{grid}.csv"
    psi = cc.psitreillis()
    with open(log, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(COLUMNS)

        def put(r):
            w.writerow(r)
            fo.flush()
            print("  ".join(str(x) for x in r), flush=True)

        # Psi-treillis, degree 3, default settings: one group of 32 frames, written and decoded by the library.
        for bound in b["psi_bounds"]:
            path = folder / "group.psi"
            path.unlink(missing_ok=True)
            code, message = psi.OK, ""
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    psi.compress_file(sequence, path, degree=3, abs_tolerance=None if bound is None else bound * u_rms, u_rms=u_rms,
                                      first_frame=first, frame_count=FRAMES, seal=False)
                except psi.PsiError as e:
                    code, message = psi.ERROR, str(e)
            if caught:
                message = str(caught[-1].message)
                code = psi.BOUND_MISSED if not _bound_held(psi, path) else psi.CLIPPED
            if code == psi.ERROR:
                put(row(grid, "psi_d3", bound, first, FRAMES, code, raw_bytes, 1, None, u_rms, f"not encoded: {message}"))
                continue
            errors = cc.ErrorSum()
            with psi.open(path) as f:
                for j in range(FRAMES):
                    errors.add(cc.read_frame(sequence, first + j, n), f.decode(j))
            put(row(grid, "psi_d3", bound, first, FRAMES, code, raw_bytes, path.stat().st_size, errors, u_rms,
                    f"one group of {FRAMES} frames, file without a seal" + (f"; {message}" if message else "")))
            path.unlink()

        # SZ3 in 4D: one component at a time, by blocks of b["block"] frames.
        for bound in b["bounds"]:
            errors, compressed = cc.ErrorSum(), 0
            for start in range(first, first + FRAMES, b["block"]):
                for c in range(3):
                    block = cc.read_component_block(sequence, start, b["block"], c, n)
                    size, decoded = cc.sz3_4d_component(block, bound * u_rms, folder)
                    compressed += size
                    for j in range(b["block"]):
                        errors.add(block[j], decoded[j])
                    del decoded
                    (folder / "block.dec").unlink()
            put(row(grid, "sz3_4d", bound, first, b["block"], 0, raw_bytes, compressed, errors, u_rms,
                    f"{FRAMES // b['block']} block(s) of {b['block']} frames, time as the fourth dimension"))

        # Frame-by-frame codecs, on each of the 32 frames.
        totals = {(kind, bound): [cc.ErrorSum(), 0] for kind in FRAME_CODECS for bound in b["bounds"]}
        for j in range(FRAMES):
            u = cc.read_frame(sequence, first + j, n)
            for kind in FRAME_CODECS:
                for bound in b["bounds"]:
                    size, decoded = cc.compress_frame(kind, u, bound * u_rms, folder)
                    totals[(kind, bound)][0].add(u, decoded)
                    totals[(kind, bound)][1] += size
            print(f"frame {first + j} done", flush=True)
        for kind in FRAME_CODECS:
            for bound in b["bounds"]:
                errors, compressed = totals[(kind, bound)]
                put(row(grid, kind, bound, first, 1, 0, raw_bytes, compressed, errors, u_rms, f"frame by frame, {FRAMES} frames"))
    summary([log])


def _bound_held(psi, path):
    with psi.open(path) as f:
        return (not f.info.has_bound) or f.info.bound_held


def summary(logs):
    for log in logs:
        rows = list(csv.DictReader(open(log, encoding="utf-8")))
        if not rows:
            continue
        print(f"\n{log.name}: compression ratio / relative RMS error / largest error over u' (frames {rows[0]['frames']})")
        bounds = sorted({r["bound_over_urms"] for r in rows}, key=lambda s: (s != "none", -float(s) if s != "none" else 0))
        names = []
        for r in rows:
            if r["codec"] not in names:
                names.append(r["codec"])
        print(f"{'bound':>8s}  " + "  ".join(f"{c:>26s}" for c in names))
        for bound in bounds:
            cells = []
            for c in names:
                r = next((x for x in rows if x["codec"] == c and x["bound_over_urms"] == bound), None)
                if r is None:
                    cells.append("")
                elif not r["compression_ratio"]:
                    cells.append(f"not encoded (code {r['exit_code']})")
                else:
                    cells.append(f"{float(r['compression_ratio']):7.1f}x {100 * float(r['rms_error_relative']):6.3f}% {100 * float(r['largest_error_over_urms']):6.2f}%")
            print(f"{bound:>8s}  " + "  ".join(f"{c:>26s}" for c in cells))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action in BENCHES:
        run(action)
    elif action == "summary":
        summary([p for p in (HERE / f"equal_bound_{g}.csv" for g in BENCHES) if p.is_file()])
    else:
        print(__doc__)
        sys.exit(2)
