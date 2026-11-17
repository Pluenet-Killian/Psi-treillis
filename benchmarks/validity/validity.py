# SPDX-License-Identifier: Apache-2.0
"""Validity map of Psi-treillis against the resolution of the data (claims VALID-1 to VALID-5).

    python benchmarks/validity/validity.py simulate   produces the sequences that are missing (data/sweep/)
    python benchmarks/validity/validity.py measure    measures what is missing, appends it to validity.csv
    python benchmarks/validity/validity.py summary    table per viscosity over the realisations; writes summary.txt
    python benchmarks/validity/validity.py identity   compares one sequence of the sweep with data/turb256.seq; writes
                                                      identity_turb256.txt

What is measured. A sweep of the resolution: forced isotropic turbulence at 256^3, six viscosities and three
realisations each (solver seeds 42, 7 and 11), 48 frames per sequence, frames 8 to 43 encoded. On each of the 18
sequences:
  - Psi-treillis (`psi_stream pack`) in degree 3 and degree 2 at a lattice step of 2 voxels, and in degree 3 at a step
    of 1 voxel, without a bound; in degree 3, at both steps, also with a bound of 5 % and of 3 % of u' on the largest
    error. The compression ratio is that of the group of the first 32 frames (one key frame, 31 innovations); the errors
    are those that the encoder measured on every voxel and recorded in the frame table of the file;
  - at the same absolute bound (bound x u', u' being the RMS velocity of the header of the sequence), SPERR and SZ3
    frame by frame on frames 8, 24 and 39, and SZ3 in 4D on the block of frames 8 to 39. Their errors are measured here,
    by comparing the decoded arrays with the original in NumPy.
Only sizes, errors and exit codes are logged: no timing. The criteria that these measurements decide are in
docs/CLAIMS.md (VALID-1 to VALID-5), written before the first measurement.

What it needs. build/ (tools/build.ps1 -Tools: dns and psi_stream), the comparison codecs of build/codecs/
(benchmarks/codecs/build_codecs.ps1), NumPy, and a GPU for `simulate` and `measure`. The 18 sequences are 9.0 GiB each
and are not versioned.

What it writes, in this folder:
  validity.csv           one line per sequence, codec and bound (columns: COLUMNS below). `measure` appends to it and
                         skips what it already holds: delete the file to measure everything again
  logs/dns_*.log         output of the solver for each sequence, after the command that produced it
  logs/pack_*.log        output of `psi_stream pack` for each sequence, setting and bound, after its command
  summary.txt            the table of `summary`
  identity_turb256.txt   the comparison of `identity`
The .psi files are written in build/work/validity/ and deleted once read.
"""

import csv
import hashlib
import math
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "codecs"))
import comparison_codecs as cc  # noqa: E402

SWEEP = cc.DATA / "sweep"
WORK = cc.BUILD / "work" / "validity"
LOG = HERE / "validity.csv"
LOGS = HERE / "logs"

# The sweep, fixed before any measurement. Viscosities chosen to cover from 0.4 to 0.8 voxel per Kolmogorov scale
# (scaling law in nu^(3/4) from turb256.seq: 0.0028 gives 0.80 voxel); the actual value is read from the header.
N, FRAMES, STEPS_PER_FRAME, SPINUP = 256, 48, 4, 10
VISCOSITIES = (0.0011, 0.0013, 0.0016, 0.0019, 0.0023, 0.0028)
SOLVER_SEEDS = (42, 7, 11)
FIRST, COUNT = 8, 36                  # encoded segment: frames 8 to 43, two key frames (ranks 0 and 32 of the file)
GROUP = 32                            # frames of the group whose compression ratio is logged, and of the block of SZ3 in 4D
BOUNDS = (0.05, 0.03)                 # bounds on the largest error, as fractions of u'
FRAMES_COMPARED = (8, 24, 39)         # frames of the frame-by-frame codecs
PSI = (("psi_d3_mf2", 3, 2), ("psi_d2_mf2", 2, 2), ("psi_d3_mf1", 3, 1))   # (codec, degree, lattice step in voxels)
COLUMNS = ["sequence", "nu", "solver_seed", "kmax_eta", "eta_voxels", "re_lambda", "u_rms", "frame_interval_over_tau_eta", "codec",
           "bound_over_urms", "exit_code", "group_compression_ratio", "file_compression_ratio", "rms_error_relative",
           "largest_error_over_urms", "note"]

DNS, PACK = cc.BUILD / f"dns{cc.EXE}", cc.BUILD / f"psi_stream{cc.EXE}"


def name(nu, seed):
    return f"nu{round(nu * 1e5):05d}_seed{seed}"


def sequences():
    return [(nu, seed, SWEEP / f"{name(nu, seed)}.seq") for nu in VISCOSITIES for seed in SOLVER_SEEDS]


def relative(path):
    """Path relative to the root, with forward slashes: the programs run from the root, and the logs then name no folder
    of the machine."""
    return Path(path).relative_to(ROOT).as_posix()


def resolution(sequence):
    """Header of a sequence, with its Kolmogorov scale in voxels and its frame interval in Kolmogorov times."""
    h = cc.read_header(sequence)
    eta = h["kmax_eta"] / (h["n"] / 3)             # Kolmogorov scale, the box having a side of 2 pi
    return dict(h, eta_voxels=eta * h["n"] / (2 * math.pi), frame_interval_over_tau_eta=h["dt"] / (eta * eta / h["nu"]))


def without_machine_paths(text):
    """A versioned log does not carry the path of the machine: the root of the repository is removed from each line, and
    a line that still starts with an absolute path (the source file named by a failed check of a program) is dropped."""
    for root in (str(ROOT) + os.sep, ROOT.as_posix() + "/"):
        text = text.replace(root, "")
    return "\n".join(line for line in text.splitlines() if not re.match(r"[A-Za-z]:[\\/]|/", line.strip()))


def simulate():
    SWEEP.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    for nu, seed, sequence in sequences():
        if sequence.is_file() and sequence.stat().st_size == cc.SEQ_HEADER_BYTES + FRAMES * 12 * N ** 3:
            continue
        command = [str(DNS), "--n", str(N), "--nu", repr(nu), "--spinup", str(SPINUP), "--frames", str(FRAMES),
                   "--steps-per-frame", str(STEPS_PER_FRAME), "--seed", str(seed), "--out", relative(sequence)]
        log = LOGS / f"dns_{name(nu, seed)}.log"
        with open(log, "w", encoding="utf-8") as f:   # written as the solver runs, so that its progress can be followed
            f.write("# " + " ".join(["dns", *command[1:]]) + "\n")
            f.flush()
            code = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        log.write_text(without_machine_paths(log.read_text(encoding="utf-8", errors="replace")) + "\n", encoding="utf-8", newline="\n")
        h = resolution(sequence) if code == 0 else {}
        print(f"{name(nu, seed)}: exit code {code}" + (f", kmax*eta = {h['kmax_eta']:.3f}, eta = {h['eta_voxels']:.3f} voxel, "
              f"Re_lambda = {h['re_lambda']:.0f}" if h else ""), flush=True)


def already_done():
    if not LOG.is_file():
        return set()
    with open(LOG, encoding="utf-8", newline="") as f:
        return {(r["sequence"], r["codec"], r["bound_over_urms"]) for r in csv.DictReader(f)}


def append(row):
    new = not LOG.is_file()
    with open(LOG, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([row.get(c, "") for c in COLUMNS])


def frame_table(path):
    """Frame table of a .psi file (docs/FORMAT.md, sections 2 and 4): for each frame (bytes, type, relative RMS error of
    the field, largest error over u'), and the size of the file without a seal. None if the table is not there."""
    raw = Path(path).read_bytes()
    n_frames, = struct.unpack_from("<i", raw, 32)
    index_offset, = struct.unpack_from("<Q", raw, 136)
    if index_offset < 208 or index_offset + 48 * n_frames > len(raw):
        return None
    records = [struct.unpack_from("<QQQi3fff", raw, index_offset + 48 * k) for k in range(n_frames)]
    return [(r[1], r[3], r[7], r[8]) for r in records], index_offset + 48 * n_frames


def measure_psi(base, sequence, header, done):
    WORK.mkdir(parents=True, exist_ok=True)
    raw_frame = 12.0 * header["n"] ** 3
    for codec, degree, mf in PSI:
        for bound in ("",) + (BOUNDS if degree == 3 else ()):
            key = (base["sequence"], codec, f"{bound:g}" if bound else "")
            if key in done:
                continue
            output = WORK / f"{base['sequence']}_{codec}_{key[2] or 'none'}.psi"
            command = [str(PACK), "pack", "--data", relative(sequence), "--file", relative(output), "--degree", str(degree), "--mf", str(mf),
                       "--start", str(FIRST), "--frames", str(COUNT), "--no-seal"]
            if bound:
                command += ["--tolerance", repr(bound)]
            run = subprocess.run(command, cwd=ROOT, capture_output=True)
            text = (run.stdout + run.stderr).decode("utf-8", "replace")
            (LOGS / f"pack_{output.stem}.log").write_text("# " + " ".join(["psi_stream", *command[1:]]) + "\n" + without_machine_paths(text) + "\n",
                                                          encoding="utf-8", newline="\n")
            row = dict(base, codec=codec, bound_over_urms=key[2], exit_code=run.returncode)
            table = frame_table(output) if output.is_file() and output.stat().st_size > 208 else None
            if table:
                frames, size = table
                measured = [t for t in frames if t[2] >= 0]
                innovations = [t[0] for t in frames if t[1] == 1]
                row.update(group_compression_ratio=f"{GROUP * raw_frame / sum(t[0] for t in frames[:GROUP]):.3f}",
                           file_compression_ratio=f"{len(frames) * raw_frame / size:.3f}",
                           rms_error_relative=f"{sum(t[2] for t in measured) / max(len(measured), 1):.6e}",
                           largest_error_over_urms=f"{max(t[3] for t in frames):.5e}",
                           note=f"key frame {frames[0][0] / 1048576:.3f} MiB, mean innovation {sum(innovations) / max(len(innovations), 1) / 1048576:.3f} MiB")
            else:
                # "Failure: ..." is what a refusal of the encoder prints (CHK of src/codec_kernels.cuh, and the two
                # refusals at the end of pack in src/codec_stream.cuh).
                failure = next((line.strip() for line in text.splitlines() if "Failure" in line), "")
                row["note"] = failure[:160]
            append(row)
            print(f"   {codec} bound {key[2] or 'none'}: exit code {run.returncode}" + (f", {row['group_compression_ratio']}x, "
                  f"RMS error {100 * float(row['rms_error_relative']):.3f} %, largest error {100 * float(row['largest_error_over_urms']):.2f} % of u'"
                  if table else f" ({row['note'][:80]})"), flush=True)
            if output.is_file():
                output.unlink()


def frame_by_frame(sequence, bounds, frames, kinds, folder):
    """SZ3 and SPERR on single frames, component by component, at the absolute bound `bound x u'`. One record per frame,
    bound and codec: compression ratio, RMS of the error over the RMS of the frame, largest error over u'. The figures
    are rounded as they are written, before they are averaged by the caller."""
    header = cc.read_header(sequence)
    n, u_rms = header["n"], header["u_rms"]
    records = []
    for k in frames:
        u = cc.read_frame(sequence, k, n)
        square_field = float((u.astype(np.float64) ** 2).sum())
        for bound in bounds:
            for kind in kinds:
                try:
                    size, decoded = cc.compress_frame(kind, u, bound * u_rms, folder)
                except Exception as e:  # noqa: BLE001 - a codec that fails does not stop the others
                    print(f"   FAILURE {kind} {bound}: {str(e)[:300]}", flush=True)
                    continue
                error = decoded - u
                rms = float(np.sqrt((error.astype(np.float64) ** 2).sum() / square_field))
                largest = float(np.abs(error).max()) / u_rms
                ratio = 12.0 * n ** 3 / size
                records.append(dict(codec=kind, bound=f"{bound:g}", frames=str(k), ratio=f"{ratio:.3f}", rms=f"{rms:.6e}", largest=f"{largest:.5e}"))
                print(f"frame {k}  {kind:6s} bound {100 * bound:.1f} %: {ratio:8.2f}x  RMS error {100 * rms:.4f} %  largest error {largest:.4f} u'", flush=True)
    return records


def sz3_4d(sequence, first, count, bounds, folder):
    """SZ3 on the 4D block (time, z, y, x) of each component, at the absolute bound `bound x u'`. One record per bound,
    over the three components; none if a run fails."""
    header = cc.read_header(sequence)
    n, u_rms = header["n"], header["u_rms"]
    totals = {bound: dict(bytes=0, largest=0.0, per_frame=np.zeros(count)) for bound in bounds}
    square_field = 0.0
    for c in range(3):  # one component at a time: a block at 512^3 does not fit three times in memory
        block = cc.read_component_block(sequence, first, count, c, n)
        square_field += float((block.astype(np.float64) ** 2).sum()) if n <= 256 else sum(float((block[j].astype(np.float64) ** 2).sum()) for j in range(count))
        for bound in bounds:
            t = totals[bound]
            try:
                size, decoded = cc.sz3_4d_component(block, bound * u_rms, folder)
            except RuntimeError as e:
                print("FAILURE", str(e)[-300:], flush=True)
                return []
            t["bytes"] += size
            for j in range(count):
                error = decoded[j] - block[j]
                t["per_frame"][j] += float((error.astype(np.float64) ** 2).sum())
                t["largest"] = max(t["largest"], float(np.abs(error).max()))
            del decoded
            (folder / "block.dec").unlink()
            print(f"   component {c}, bound {100 * bound:.1f} %: done", flush=True)
        del block
    records = []
    for bound in bounds:
        t = totals[bound]
        ratio = 12.0 * count * n ** 3 / t["bytes"]
        rms = float(np.sqrt(t["per_frame"].sum() / square_field))
        records.append(dict(codec=f"sz3_4d_{count}", bound=f"{bound:g}", frames=f"{first}-{first + count - 1}", ratio=f"{ratio:.3f}", rms=f"{rms:.6e}",
                            largest=f"{t['largest'] / u_rms:.5e}"))
        print(f"SZ3 in 4D, {count} frames {first} to {first + count - 1}, bound {100 * bound:.1f} %: {ratio:.2f}x  RMS error {100 * rms:.4f} % "
              f"(per frame {100 * np.sqrt(t['per_frame'].min() * count / square_field):.4f} to {100 * np.sqrt(t['per_frame'].max() * count / square_field):.4f} %)  "
              f"largest error {t['largest'] / u_rms:.4f} u'", flush=True)
    return records


def measure_comparison(base, sequence, done):
    missing = [b for b in BOUNDS if (base["sequence"], "sperr", f"{b:g}") not in done or (base["sequence"], "sz3", f"{b:g}") not in done]
    missing_4d = [b for b in BOUNDS if (base["sequence"], f"sz3_4d_{GROUP}", f"{b:g}") not in done]
    if not missing and not missing_4d:
        return
    folder = cc.work_folder("validity")
    records = []
    if missing:
        records += frame_by_frame(sequence, missing, list(FRAMES_COMPARED), ["sz3", "sperr"], folder)
    if missing_4d:
        records += sz3_4d(sequence, FIRST, GROUP, missing_4d, folder)
    groups = {}
    for r in records:
        groups.setdefault((r["codec"], r["bound"]), []).append(r)
    for (codec, bound), rows in groups.items():
        ratio = len(rows) / sum(1.0 / float(r["ratio"]) for r in rows)        # harmonic mean: the bytes add up
        append(dict(base, codec=codec, bound_over_urms=bound, exit_code=0, group_compression_ratio=f"{ratio:.3f}", file_compression_ratio="",
                    rms_error_relative=f"{sum(float(r['rms']) for r in rows) / len(rows):.6e}",
                    largest_error_over_urms=f"{max(float(r['largest']) for r in rows):.5e}",
                    note=f"{len(rows)} measurement(s): frames {', '.join(r['frames'] for r in rows)}"))


def measure():
    LOGS.mkdir(parents=True, exist_ok=True)
    for nu, seed, sequence in sequences():
        if not sequence.is_file():
            print(f"{sequence.name}: missing (run `simulate`)", flush=True)
            continue
        h = resolution(sequence)
        base = dict(sequence=sequence.stem, nu=f"{nu:g}", solver_seed=seed, kmax_eta=f"{h['kmax_eta']:.4f}", eta_voxels=f"{h['eta_voxels']:.4f}",
                    re_lambda=f"{h['re_lambda']:.1f}", u_rms=f"{h['u_rms']:.6f}", frame_interval_over_tau_eta=f"{h['frame_interval_over_tau_eta']:.4f}")
        print(f"{sequence.stem}: eta = {h['eta_voxels']:.3f} voxel", flush=True)
        measure_psi(base, sequence, h, already_done())
        measure_comparison(base, sequence, already_done())


def identity():
    """Is the sequence of the sweep at the viscosity and the seed of turb256.seq (0.0028 and 42) the same simulation?
    Compares the SHA-256 fingerprint of three frames: asking for 48 frames instead of 128 must change no frame."""
    a, b = cc.DATA / "turb256.seq", SWEEP / f"{name(0.0028, 42)}.seq"
    lines, same = [], True
    for k in (0, 32, 47):
        digests = []
        for path in (a, b):
            with open(path, "rb") as f:
                f.seek(cc.SEQ_HEADER_BYTES + k * 12 * N ** 3)
                digests.append(hashlib.sha256(f.read(12 * N ** 3)).hexdigest())
        same = same and digests[0] == digests[1]
        lines.append(f"frame {k:2d}: {a.name} {digests[0][:16]}  {b.name} {digests[1][:16]}  {'identical' if digests[0] == digests[1] else 'DIFFERENT'}")
    lines.append("same simulation on these frames" if same else "different simulations")
    print("\n".join(lines))
    (HERE / "identity_turb256.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0 if same else 1


def summary():
    with open(LOG, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_nu = {}
    for r in rows:
        by_nu.setdefault(r["nu"], []).append(r)
    codecs = [(c, b) for c in ("psi_d3_mf2", "psi_d2_mf2", "psi_d3_mf1", "sperr", "sz3", f"sz3_4d_{GROUP}") for b in ("", "0.05", "0.03")]
    lines = []
    for nu in sorted(by_nu, key=float):
        group = by_nu[nu]
        eta = sorted({float(r["eta_voxels"]) for r in group})
        lines.append(f"\nnu = {nu}: eta from {eta[0]:.3f} to {eta[-1]:.3f} voxel, {len({r['sequence'] for r in group})} realisation(s)")
        for codec, bound in codecs:
            v = [r for r in group if r["codec"] == codec and r["bound_over_urms"] == bound]
            if not v:
                continue
            codes = ",".join(sorted({r["exit_code"] for r in v}))
            encoded = [r for r in v if r["group_compression_ratio"]]
            if encoded:
                ratios = [float(r["group_compression_ratio"]) for r in encoded]
                lines.append(f"   {codec:<11} bound {bound or 'none':>6}: exit code {codes:<5} ratio {min(ratios):7.1f} to {max(ratios):7.1f}x  "
                             f"RMS error {100 * min(float(r['rms_error_relative']) for r in encoded):.3f} to "
                             f"{100 * max(float(r['rms_error_relative']) for r in encoded):.3f} %  "
                             f"largest error {100 * max(float(r['largest_error_over_urms']) for r in encoded):.2f} % of u'")
            else:
                lines.append(f"   {codec:<11} bound {bound or 'none':>6}: exit code {codes}")
    print("\n".join(lines))
    (HERE / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "simulate":
        simulate()
    elif action == "measure":
        measure()
    elif action == "summary":
        summary()
    elif action == "identity":
        sys.exit(identity())
    else:
        print(__doc__)
        sys.exit(2)
