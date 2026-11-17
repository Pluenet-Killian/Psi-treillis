#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Claim REF-2 (docs/CLAIMS.md): the second reader, tools/reference_reader.py (Python and NumPy, written from
docs/FORMAT.md alone), on the five reference files at 256^3 and the test files at 64^3.

    python benchmarks/independent_readers/python/reference_python.py [--no-gpu] [--files FOLDER] [--cases FOLDER] [--output FOLDER]

What is measured, in this order.

  1. The files to read are copied to the working folder W = build/work/independent_readers/python/. The originals are
     never opened for writing. The files of the criterion are the five reference files at 256^3 (f_t256_d3_t05, _t04,
     _t03, f_t256_d2_t06, _t05) and every valid test file at 64^3 (d2_*.psi and d3_*.psi of the folder of the cases).
  2. Each of them is encoded again by `psi_stream pack --probe`, run as a black box with the settings read in the
     header of the file, to obtain the probe of the encoder (docs/FORMAT.md, section 11):
         psi_stream pack --data SEQUENCE --file W/reencoded/F.psi --probe W/probes/F.probe OPTIONS
     The SHA-256 of the re-encoding is compared with that of the file: the probe stands for the file only if they are
     equal. The header does not say the cap on the passes of the fine level: if the re-encoding differs, --fine-passes
     is tried from 1 to 8. If nothing gives back the file, it is the re-encoded file that is read back and probed, and
     the log says so.
  3. Each file is read back with the reader, on CPU: fingerprint of each frame against that of the frame table, then
     sampling lattice, fine level and velocity field against the probe.
  4. Six complementary files at 64^3 are produced for what the files of the criterion do not exercise (frames without
     a brick, PSISTRM3 with raw keys, wrapped lattice read by differences), and read back in the same way:
         psi_stream pack --data build/tests/turb64.seq --file W/complements/F.psi --probe W/complements/F.probe --start 3 --frames 36 OPTIONS
  5. The altered files at 64^3 (guard_*.psi) go through the reader: refusal, or count of the frames that differ,
     compared with the outcome of `psi_stream play`, run as a black box:
         psi_stream play --file W/cases64/F.psi --particles 4096

Options.
    --files FOLDER    folder of the reference files, from the root (default: build/work/reference_files)
    --cases FOLDER    folder of the test cases at 64^3, from the root (default: build/tests/cases); the test sequence
                      turb64.seq is taken in its parent folder
    --no-gpu          runs neither pack nor play: takes again the probes, the re-encodings and the outcomes of play left
                      in the working folder by an earlier run. The reading itself never needs the GPU.
    --output FOLDER   where the logs are written (default: the folder of this script)

What is written.
    reference_python.csv        one line per file of the criterion: version, grid, degree, seal, frames, frames read,
                                fingerprints that differ, raw and coded keys, escapes, overflows, frames without a
                                brick, wrapped frames, how the lattice is read, margin of the prediction to 32 bits,
                                origin of the probe, what the probe was read on, position and points of the probe,
                                nodes that differ, largest difference of the field as a fraction of u', difference of
                                the scales, and what the re-encoding gives when it is not the file
    complements.csv             the six complementary files: the same columns, what each one puts to the test, the
                                options and the exit code of pack, the range of the differences inside a support
    coverage_64.csv             the altered files: outcome of the reader, outcome of play, agreement
    logs/reference_python.log   what the script printed: SHA-256 of each file, options of pack, summary
    logs/pack_<file>.log        raw output of every pack run for a file, each under a line "== command: exit code"
    logs/coverage_64.log        raw output of every play, each under a line "== command: exit code"
No time is read. The GPU may be shared.

The list of the passages where the specification left a doubt, gaps.csv, is kept by hand beside this script: the
script does not write it. tools/claims.py reads reference_python.csv and gaps.csv.

Exit code: 0 if the log is complete; 1 if a file or a probe is missing, or if the reader raised an error; 2 if a
prerequisite is missing (nothing was run).

Needs Python 3.9 or later and NumPy; the five reference files (benchmarks/compression/reference_files/reference_files.py)
and the test cases (python tests/test_codec.py); unless --no-gpu, build/psi_stream (tools/build.ps1 -Tools on Windows,
CMake with -DPSIT_BUILD_TOOLS=ON elsewhere), data/turb256.seq, build/tests/turb64.seq and an NVIDIA GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(HERE.parent.parent / "compression"))
import psi_stream_runs as ps  # noqa: E402
import reference_reader as rr  # noqa: E402

WORK = "build/work/independent_readers/python"
THRESHOLD = 5e-5   # criterion of REF-2, as a fraction of u'
REFERENCES = ["f_t256_d3_t05", "f_t256_d3_t04", "f_t256_d3_t03", "f_t256_d2_t06", "f_t256_d2_t05"]
COMPLEMENTS = {   # name: (options of pack, what the file puts to the test)
    "c1_d2_no_brick": (["--degree", "2", "--tolerance", "0.5"], "no brick: tables of the empty segments (doubt D3)"),
    "c1_d3_no_brick": (["--degree", "3", "--tolerance", "0.5"], "no brick: tables of the empty segments (doubt D3)"),
    "c2_d2_bound_raw_keys": (["--degree", "2", "--tolerance", "0.05", "--raw-keys"], "PSISTRM3 with raw keys (section 5)"),
    "c2_d3_bound_raw_keys": (["--degree", "3", "--tolerance", "0.04", "--raw-keys"], "PSISTRM3 with raw keys (section 5)"),
    "c3_d2_eps0005": (["--degree", "2", "--eps", "0.0005"], "clippings of site 5, lattice not wrapped"),
    "c3_d3_eps0005": (["--degree", "3", "--eps", "0.0005"], "wrapped lattice, read by differences (doubt D2)"),
}
COLUMNS = ["file", "group", "version", "n", "degree", "seal", "frames", "frames_read", "fingerprints_differing",
           "raw_keys", "coded_keys", "escapes", "overflows", "frames_without_brick", "wrapped_frames",
           "lattice_read", "prediction_margin_32_bits", "probe_origin", "probe_read_on", "probe_frame_position", "probe_points",
           "lattice_nodes_differing", "fine_nodes_differing", "field_difference_over_u_rms", "scale_difference",
           "reencoding_frames_read", "reencoding_fingerprints_differing"]

# The line of play that the script reads. tools/psi_stream.cu, play<D>, last lines of the report:
#   printf("Integrity      : %d frame(s) whose state differs from that of the sender; fingerprint of the final positions %016llx\n", bad_hash, pos_hash);
INTEGRITY = re.compile(r"^Integrity\s*:\s*(\d+) frame", re.M)

console: list = []


def say(text: str = "") -> None:
    print(text, flush=True)
    console.append(text)


def shortest(x: float) -> str:
    """Shortest decimal writing of a float32: 0.05 and not 0.05000000074505806."""
    return np.format_float_positional(np.float32(x), unique=True, trim="-")


def pack_options(f: rr.PsiFile) -> list:
    """Options of `psi_stream pack` that give back the file, from its header. An option at its default value (help of
    pack: --mf 2, --levels 3, --eps 0.005, --key-interval 32, --tolerance-margin 1e-4) is not passed. The header says
    neither --fine-passes, nor --fit-at, nor --legacy-step: a file written with one of them is not found again, except
    --fine-passes 12, which docs/FORMAT.md (section 8) associates with a zero margin."""
    e = f.header
    o = ["--degree", str(e.degree)]
    if e.mf != 2:
        o += ["--mf", str(e.mf)]
    if e.levels != 3:
        o += ["--levels", str(e.levels)]
    if shortest(e.eps) != "0.005":
        o += ["--eps", shortest(e.eps)]
    if e.key_interval != 32:
        o += ["--key-interval", str(e.key_interval)]
    o += ["--start", str(e.first_frame), "--frames", str(e.n_frames)]
    if any(r.type == 0 for r in f.table):
        o += ["--raw-keys"]
    if f.v3:
        i = f.fine_info
        o += ["--tolerance", shortest(i.tolerance)]
        if shortest(i.margin) != "0.0001":
            o += ["--tolerance-margin", shortest(i.margin)]
        if abs(i.eps / (i.tolerance / 8.5) - 1.0) > 1e-5:
            o += ["--fine-eps", shortest(i.eps)]
        if i.margin == 0:
            o += ["--fine-passes", "12"]
    if f.seal == "absent":
        o += ["--no-seal"]
    return o


def pack(arguments: list, file: str, probe: str, raw) -> int:
    """Runs `psi_stream pack` towards file and probe (removed first: what an earlier run left must not stand for this
    one), appends its raw output to the open log `raw`, and returns its exit code."""
    for stale in (file, probe):
        (ROOT / stale).unlink(missing_ok=True)
    r = ps.run(["pack"] + arguments[:2] + ["--file", file, "--probe", probe] + arguments[2:])
    raw.write(f"== {r.command}: exit code {r.code}\n{r.text()}")
    raw.flush()
    return r.code


def prediction_margin(f: rr.PsiFile) -> float:
    """2^31 divided by the largest possible modulus of the sum of the prediction (docs/FORMAT.md, 6.2): above 1, the
    sum fits in 32 bits whatever the states (computed on the predictors of the header)."""
    worst = max((abs(a) + abs(b) + 12 * abs(c)) * 32767 + 512 for a, b, c in f.header.coef[:f.header.levels])
    return (1 << 31) / max(worst, 1024 * 32767 + 512)


def read_back(path: str, probe: str | None):
    """Reads back all the frames of a file. Returns (open file, counters, comparison with the probe, probed frame, probe)."""
    f = rr.PsiFile(str(ROOT / path))
    s = rr.read_probe(str(ROOT / probe)) if probe is not None else None
    probe_position = s.frame - f.header.first_frame if s is not None else None
    c = dict(read=0, differing=0, raw=0, coded=0, escapes=0, overflows=0, without_brick=0)
    comparison = probed = None
    for t in f.frames():
        c["read"] += 1
        c["differing"] += 0 if t.conforming else 1
        c["raw"] += t.type == 0
        c["coded"] += t.type == 3
        c["escapes"] += t.n_escapes if t.type in (1, 2) else 0
        c["overflows"] += int(t.overflows[0].size)
        c["without_brick"] += 1 if (t.fine is not None and t.fine.n_bricks == 0) else 0
        if s is not None and t.position == probe_position:
            comparison, probed = rr.compare_with_probe(f, s, t), t
    if s is not None and comparison is None:
        raise rr.Refusal("the frame of the probe is outside the file")
    if f.overflows_32_bits:
        raise rr.Refusal("the sum of a prediction left 32 bits: doubt D1 is no longer without effect")
    return f, c, comparison, probed, s


def row_of(name: str, group: str, f: rr.PsiFile, c: dict) -> dict:
    e = f.header
    return {"file": name + ".psi", "group": group, "version": e.magic.decode(), "n": e.n, "degree": e.degree, "seal": f.seal,
            "frames": e.n_frames, "frames_read": c["read"], "fingerprints_differing": c["differing"],
            "raw_keys": c["raw"], "coded_keys": c["coded"], "escapes": c["escapes"],
            "overflows": c["overflows"], "frames_without_brick": c["without_brick"] if f.v3 else "",
            "wrapped_frames": e.wrapped_frames, "lattice_read": "direct" if f.direct_read else "by differences",
            "prediction_margin_32_bits": f"{prediction_margin(f):.1f}"}


def add_probe(row: dict, comp, origin: str, on: str) -> None:
    row.update({"probe_origin": origin, "probe_read_on": on, "probe_frame_position": comp.frame, "probe_points": comp.points,
                "lattice_nodes_differing": comp.differing_lattice_nodes, "fine_nodes_differing": comp.differing_fine_nodes,
                "field_difference_over_u_rms": f"{comp.max_difference_u:.3e}",
                "scale_difference": f"{max(comp.scale_difference, comp.fine_scale_difference):.1e}"})


def support_ranges(f: rr.PsiFile, lattice: np.ndarray, pts: np.ndarray):
    """Range, in the support of each point, of the differences from the first node read modulo 2^16. Under 2^15,
    subtracting another node of the support than the first gives the same differences up to a constant, hence the same
    gradient. Returns (largest range, number of points whose support reaches 2^15)."""
    N, degree = lattice.shape[0], f.header.degree
    step, d = 2.0 * math.pi / N, 0.5 if degree == 2 else 0.0
    ix, iy, iz = (rr._weights(pts[:, a] / step - d, degree)[0] for a in range(3))
    first = lattice[iz % N, iy % N, ix % N].astype(np.int64)
    lowest, highest = np.zeros_like(first), np.zeros_like(first)
    for kz in range(degree + 1):
        for ky in range(degree + 1):
            for kx in range(degree + 1):
                e = ((lattice[(iz + kz) % N, (iy + ky) % N, (ix + kx) % N].astype(np.int64) - first + 32768) & 0xFFFF) - 32768
                lowest, highest = np.minimum(lowest, e), np.maximum(highest, e)
    ranges = (highest - lowest).max(axis=1)
    return int(ranges.max()), int(np.count_nonzero(ranges >= 32768))


def lattice_by_differences(f: rr.PsiFile, lattice: np.ndarray, pts: np.ndarray, reference: int) -> np.ndarray:
    """E * curl of the lattice read by differences, subtracting the node of offset `reference` on each axis (0: the
    first node of the support, as the reader does; degree: the last). Serves to say whether a file decides between
    these two choices."""
    N, degree = lattice.shape[0], f.header.degree
    step, d = 2.0 * math.pi / N, 0.5 if degree == 2 else 0.0
    (ix, wx, dwx), (iy, wy, dwy), (iz, wz, dwz) = (rr._weights(pts[:, a] / step - d, degree) for a in range(3))
    subtracted = lattice[(iz + reference) % N, (iy + reference) % N, (ix + reference) % N].astype(np.int64)
    rot = np.zeros((pts.shape[0], 3))
    for kz in range(degree + 1):
        for ky in range(degree + 1):
            for kx in range(degree + 1):
                q = ((lattice[(iz + kz) % N, (iy + ky) % N, (ix + kx) % N].astype(np.int64) - subtracted + 32768) & 0xFFFF) - 32768
                gx, gy, gz = dwx[kx] * wy[ky] * wz[kz], wx[kx] * dwy[ky] * wz[kz], wx[kx] * wy[ky] * dwz[kz]
                rot[:, 0] += gy * q[:, 2] - gz * q[:, 1]
                rot[:, 1] += gz * q[:, 0] - gx * q[:, 2]
                rot[:, 2] += gx * q[:, 1] - gy * q[:, 0]
    return f.scale * rot


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="Claim REF-2: the second reader, in Python, on the reference files at 256^3 and the test files at 64^3.")
    p.add_argument("--files", default="build/work/reference_files", help="folder of the reference files, from the root")
    p.add_argument("--cases", default="build/tests/cases", help="folder of the test cases at 64^3, from the root; turb64.seq is taken in its parent folder")
    p.add_argument("--no-gpu", action="store_true", help="runs neither pack nor play: takes again what an earlier run left in the working folder")
    p.add_argument("--output", default=str(HERE), help="folder where the logs are written (default: the folder of this script)")
    a = p.parse_args()
    files, cases, gpu, output = ps.relative(a.files), ps.relative(a.cases), not a.no_gpu, Path(a.output).resolve()
    seq256, seq64 = "data/turb256.seq", ps.relative(ROOT / cases / ".." / "turb64.seq")
    valid = sorted(x.stem for x in (ROOT / cases).glob("d[23]_*.psi"))
    altered = sorted(x.stem for x in (ROOT / cases).glob("guard_*.psi"))
    if not valid:
        ps.stop(f"No valid test file (d2_*.psi, d3_*.psi) in {cases}/. Run python tests/test_codec.py first. Nothing was run.")
    if gpu:
        ps.require([ps.PSI_STREAM, seq256, seq64], "Build the instruments (tools/build.ps1 -Tools), produce the sequences of data/ and run "
                   "python tests/test_codec.py first; or pass --no-gpu to read again what an earlier run left.")
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    for d in ("ref256", "cases64", "reencoded", "probes", "complements"):
        (ROOT / WORK / d).mkdir(parents=True, exist_ok=True)

    say("Log of the second reader (tools/reference_reader.py), claim REF-2")
    say(f"Python {sys.version.split()[0]}, NumPy {np.__version__}; psi_stream {'run for the probes' if gpu else 'not run (--no-gpu)'}")
    say(f"Threshold of the criterion: field at the probe at less than {THRESHOLD:.0e} u'")
    anomalies = 0

    # --- Files of the criterion: five reference files at 256^3, valid test files at 64^3 ---
    batch = [(name, "reference_256", f"{files}/{name}.psi", f"{WORK}/ref256", seq256) for name in REFERENCES]
    batch += [(name, "test_64", f"{cases}/{name}.psi", f"{WORK}/cases64", seq64) for name in valid]
    say(f"Files of the criterion: {len(REFERENCES)} reference files at 256^3 and {len(valid)} valid test files at 64^3; "
        f"{len(altered)} altered files at 64^3 go to the coverage")
    rows = []
    memory_path = ROOT / WORK / "pack_options.json"
    memory_pack = json.loads(memory_path.read_text(encoding="utf-8")) if memory_path.is_file() else {}
    for name, group, source, folder, sequence in batch:
        copy, reenc, probe = f"{folder}/{name}.psi", f"{WORK}/reencoded/{name}.psi", f"{WORK}/probes/{name}.probe"
        say()
        if not (ROOT / source).is_file() and not (ROOT / copy).is_file():
            say(f"{name}.psi: NOT FOUND")
            anomalies += 1
            continue
        if (ROOT / source).is_file():
            shutil.copyfile(ROOT / source, ROOT / copy)
        try:
            opened = rr.PsiFile(str(ROOT / copy))
            options, file_sha = pack_options(opened), ps.sha256(ROOT / copy)
            if gpu:
                trials = [options]
                if opened.v3 and "--fine-passes" not in options:   # cap on the passes: absent from the header
                    trials += [options + ["--fine-passes", str(k)] for k in range(1, 9)]
                with open(logs / f"pack_{name}.log", "w", encoding="utf-8", newline="\n") as raw:
                    for trial in trials + [options]:               # if nothing gives back the file: the settings of the header
                        code = pack(["--data", sequence] + trial, reenc, probe, raw)
                        if (ROOT / reenc).is_file() and ps.sha256(ROOT / reenc) == file_sha:
                            options = trial
                            break
                memory_pack[name] = {"options": options, "code": code}
            elif name in memory_pack:
                options, code = memory_pack[name]["options"], memory_pack[name]["code"]
            else:
                code = "not recorded"
            if not (ROOT / reenc).is_file() or not (ROOT / probe).is_file():
                say(f"{name}.psi: neither re-encoding nor probe (pack {' '.join(options)}: exit code {code})")
                anomalies += 1
                continue
            reenc_sha = ps.sha256(ROOT / reenc)
            identical = file_sha == reenc_sha
            say(f"{name}.psi: SHA-256 {file_sha}")
            say(f"    pack {' '.join(options)} --probe: exit code {code}; re-encoding "
                + ("identical to the file" if identical else f"DIFFERENT from the file (SHA-256 {reenc_sha})"))
            f, c, comp, _, _ = read_back(copy, probe if identical else None)
            row = row_of(name, group, f, c)
            say(f"    {f.header.magic.decode()}, {f.header.n}^3, degree {f.header.degree}, seal {f.seal}, lattice read "
                f"{'directly' if f.direct_read else 'by differences'}; {c['read']} frames read back out of {f.header.n_frames}, {c['differing']} "
                f"fingerprint(s) differing; {c['raw']} raw key(s), {c['coded']} coded, {c['escapes']} escape(s), "
                f"{c['overflows']} overflow(s)")
            if identical:
                add_probe(row, comp, "identical re-encoding", "file")
            else:
                f2, c2, comp, _, _ = read_back(reenc, probe)
                add_probe(row, comp, "different re-encoding", "re-encoding")
                row.update({"reencoding_frames_read": c2["read"], "reencoding_fingerprints_differing": c2["differing"]})
                say(f"    re-encoding read back: {c2['read']} frames out of {f2.header.n_frames}, {c2['differing']} fingerprint(s) differing")
            say(f"    probe ({row['probe_read_on']}): frame at position {comp.frame}, {comp.points} points; {comp.differing_lattice_nodes} node(s) of the "
                f"lattice and {comp.differing_fine_nodes} node(s) of the fine level different from those of the encoder; field at "
                f"{comp.max_difference_u:.3e} u'; scales to within {max(comp.scale_difference, comp.fine_scale_difference):.1e}")
            rows.append(row)
        except rr.Refusal as r:
            say(f"{name}.psi: REFUSAL of the reader: {r}")
            anomalies += 1
        except Exception as r:  # a Python error is a defect of the reader: it counts
            say(f"{name}.psi: PYTHON ERROR: {type(r).__name__}")
            anomalies += 1
    if gpu:
        memory_path.write_text(json.dumps(memory_pack, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(output / "reference_python.csv", "w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, COLUMNS, restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    # --- Complementary files, produced here ---
    say()
    say("Complementary files at 64^3, outside the criterion (produced by pack --start 3 --frames 36 on the test sequence)")
    complements = []
    for name, (options, purpose) in COMPLEMENTS.items():
        file, probe = f"{WORK}/complements/{name}.psi", f"{WORK}/complements/{name}.probe"
        try:
            code = "not rerun"
            if gpu:
                with open(logs / f"pack_{name}.log", "w", encoding="utf-8", newline="\n") as raw:
                    code = pack(["--data", seq64, "--start", "3", "--frames", "36"] + options, file, probe, raw)
            if not (ROOT / file).is_file() or not (ROOT / probe).is_file():
                say(f"{name}.psi: not produced (pack exit code {code})")
                anomalies += 1
                continue
            f, c, comp, t, s = read_back(file, probe)
            row = row_of(name, "complement_64", f, c)
            add_probe(row, comp, "probe of the encoding", "file")
            row["purpose"], row["pack_options"], row["pack_exit_code"] = purpose, " ".join(options), code
            lattice = f.lattice(t)
            row["largest_range_in_a_support"], row["points_with_support_range_2_15_or_more"] = support_ranges(f, lattice, s.positions)
            if not f.direct_read:   # what the direct reading, which the header forbids here, and another subtracted node would give
                field = f.velocity(t, s.positions, lattice, f.dense_fine_level(t))
                first = lattice_by_differences(f, lattice, s.positions, 0)
                direct = field - first + f.scale * rr.curl(lattice, s.positions, 2.0 * math.pi / f.geo.nl[0], f.header.degree, False)
                last = field - first + lattice_by_differences(f, lattice, s.positions, f.header.degree)
                row["field_difference_direct_read_over_u_rms"] = f"{float(np.max(np.abs(direct - s.velocities))) / f.header.u_rms:.3e}"
                row["field_difference_last_node_subtracted_over_u_rms"] = f"{float(np.max(np.abs(last - s.velocities))) / f.header.u_rms:.3e}"
            say(f"{name}.psi ({purpose}): pack {' '.join(options)}, exit code {code}")
            say(f"    {f.header.magic.decode()}, lattice read {'directly' if f.direct_read else 'by differences'}, {f.header.wrapped_frames} wrapped "
                f"frame(s), clippings {f.header.clippings}; {c['read']} frames read back, {c['differing']} fingerprint(s) differing; {c['raw']} raw "
                f"key(s), {c['escapes']} escape(s), {c['overflows']} overflow(s), {c['without_brick']} frame(s) without a brick")
            say(f"    probe: {comp.differing_lattice_nodes} node(s) of the lattice and {comp.differing_fine_nodes} node(s) of the fine level "
                f"differing; field at {comp.max_difference_u:.3e} u'; range of the differences inside a support: "
                f"{row['largest_range_in_a_support']} at most, {row['points_with_support_range_2_15_or_more']} point(s) at 2^15 = 32768 or more"
                + (f"; the field would be at {row['field_difference_direct_read_over_u_rms']} u' with a forced direct reading, at "
                   f"{row['field_difference_last_node_subtracted_over_u_rms']} u' when subtracting the last node of the support instead of the first"
                   if "field_difference_direct_read_over_u_rms" in row else ""))
            complements.append(row)
        except rr.Refusal as r:
            say(f"{name}.psi: REFUSAL of the reader: {r}")
            anomalies += 1
        except Exception as r:
            say(f"{name}.psi: PYTHON ERROR: {type(r).__name__}")
            anomalies += 1
    with open(output / "complements.csv", "w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, ["purpose", "pack_options", "pack_exit_code"]
                           + [x for x in COLUMNS if not x.startswith("reencoding") and x not in ("group", "probe_origin", "probe_read_on")]
                           + ["largest_range_in_a_support", "points_with_support_range_2_15_or_more", "field_difference_direct_read_over_u_rms",
                              "field_difference_last_node_subtracted_over_u_rms"], restval="", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(complements)

    # --- Altered files: outcome of the reader, compared with that of psi_stream play ---
    say()
    say(f"Altered files at 64^3: {len(altered)}")
    memory_play = ROOT / WORK / "play_outcomes.json"
    play_outcomes = json.loads(memory_play.read_text(encoding="utf-8")) if memory_play.is_file() else {}
    coverage, agreements, errors = [], 0, 0
    raw = open(logs / "coverage_64.log", "w", encoding="utf-8", newline="\n") if gpu else None
    for name in altered:
        source, copy = f"{cases}/{name}.psi", f"{WORK}/cases64/{name}.psi"
        if (ROOT / source).is_file():
            shutil.copyfile(ROOT / source, ROOT / copy)
        row = {"file": name + ".psi"}
        try:
            f = rr.PsiFile(str(ROOT / copy))
            differing = sum(0 if t.conforming else 1 for t in f.frames())
            row.update({"python_outcome": "read", "python_frames_differing": differing, "python_reason": ""})
        except rr.Refusal as r:
            row.update({"python_outcome": "refused", "python_frames_differing": "", "python_reason": str(r)})
        except Exception as r:
            row.update({"python_outcome": "PYTHON ERROR", "python_frames_differing": "", "python_reason": type(r).__name__})
            errors += 1
        if gpu:
            r = ps.run(["play", "--file", copy, "--particles", "4096"])
            raw.write(f"== {r.command}: exit code {r.code}\n{r.text()}")
            raw.flush()
            integrity = INTEGRITY.search(r.text())
            if integrity:
                play_outcomes[name] = {"outcome": "read", "differing": int(integrity.group(1)), "code": r.code}
            else:
                play_outcomes[name] = {"outcome": "refused", "differing": "", "code": r.code}
        g = play_outcomes.get(name)
        if g is None:
            row.update({"gpu_outcome": "not recorded", "gpu_frames_differing": "", "play_exit_code": "", "agreement": ""})
        else:
            agreement = row["python_outcome"] == g["outcome"] and str(row["python_frames_differing"]) == str(g["differing"])
            agreements += agreement
            row.update({"gpu_outcome": g["outcome"], "gpu_frames_differing": g["differing"], "play_exit_code": g["code"],
                        "agreement": "yes" if agreement else "NO"})
        coverage.append(row)
    if gpu:
        raw.close()
        memory_play.write_text(json.dumps(play_outcomes, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(output / "coverage_64.csv", "w", newline="", encoding="utf-8") as s:
        w = csv.DictWriter(s, ["file", "python_outcome", "python_frames_differing", "gpu_outcome", "gpu_frames_differing", "play_exit_code",
                               "agreement", "python_reason"], lineterminator="\n")
        w.writeheader()
        w.writerows(coverage)
    refusals = sum(1 for row in coverage if row["python_outcome"] == "refused")
    compared = sum(1 for row in coverage if row["agreement"])
    say(f"    Python reader: {refusals} refusal(s), {len(coverage) - refusals - errors} complete read(s), {errors} Python error(s)")
    say(f"    same outcome and same number of differing frames as psi_stream play: {agreements} file(s) out of {compared} compared")
    for row in coverage:
        if row["agreement"] == "NO":
            say(f"    DISAGREEMENT {row['file']}: Python {row['python_outcome']} {row['python_frames_differing']} ({row['python_reason']}); "
                f"play {row['gpu_outcome']} {row['gpu_frames_differing']}")
    anomalies += errors

    # --- Summary ---
    say()
    frames, read = sum(row["frames"] for row in rows), sum(row["frames_read"] for row in rows)
    differing = sum(row["fingerprints_differing"] for row in rows)
    on_file = [row for row in rows if row["probe_read_on"] == "file"]
    nodes = sum(row["lattice_nodes_differing"] + row["fine_nodes_differing"] for row in rows)
    worst = max((float(row["field_difference_over_u_rms"]) for row in rows), default=float("nan"))
    say(f"SUMMARY of the criterion: {len(rows)} files out of {len(batch)}, {read} frames read back out of {frames}, {differing} fingerprint(s) "
        f"different from that of the table")
    say(f"    probes: {len(on_file)} file(s) probed on a re-encoding identical by SHA-256, {len(rows) - len(on_file)} on a "
        f"different re-encoding; {nodes} node(s) of the lattice or of the fine level different from those of the encoder; field at "
        f"{worst:.3e} u' at most, for a threshold of {THRESHOLD:.0e}")
    say(f"    complements: {len(complements)} file(s), {sum(row['fingerprints_differing'] for row in complements)} fingerprint(s) differing, field at "
        f"{max((float(row['field_difference_over_u_rms']) for row in complements), default=float('nan')):.3e} u' at most")
    say(f"    anomalies of the log: {anomalies}")

    text = "\n".join(console) + "\n"
    for path in {str(ROOT), ROOT.as_posix(), str(output), output.as_posix()}:   # no absolute path may enter the log
        if path in text:
            raise SystemExit("an absolute path entered the log: logs/reference_python.log is not written")
    (logs / "reference_python.log").write_text(text, encoding="utf-8", newline="\n")
    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
