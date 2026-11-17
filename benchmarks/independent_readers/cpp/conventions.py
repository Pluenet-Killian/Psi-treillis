#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trials that decide between the conventions that docs/FORMAT.md did not give when the reference reader on CPU was
written (mark "established by trial" of its sections 6.5 and 6.7).

    python benchmarks/independent_readers/cpp/conventions.py [--no-gpu] [--cases FOLDER] [--output FOLDER]

What is measured. Each trial compares several possible readings with what an existing program has produced, without
reading its code, on four test files at 64^3 (d2_base, d3_base, d2_bound, d3_bound) and their probes:

  1. Refinement of the pyramid (6.5): alignment of the masks and rounding of the division, against the sampling
     lattice that `pack --probe` records in its probe. Without a GPU.
         psi_ref F.psi --state J W/state.bin        J: position in the file of the frame of the probe
  2. Position of the nodes of the lattice and of the fine level (6.7), against the velocity field of the probe.
     Without a GPU; no program is run.
  3. Position of the centre of a voxel (6.7), against `psit sample`, which takes its points in grid indices. With a
     GPU. 32 points: five fixed ones, then 27 drawn by a congruential sequence with a fixed seed.
         psit sample F.psi --frame J --at X,Y,Z ...             J: last frame of the file
         psi_ref F.psi --points W/points.txt --frame J

W is build/work/independent_readers/conventions/. The files and the programs are named from the root of the repository,
where the programs are run.

Options.
    --no-gpu          trials 1 and 2 only: psit is not run, and conventions.txt says so
    --cases FOLDER    folder of the test cases at 64^3, from the root (default: build/tests/cases)
    --output FOLDER   where conventions.txt and logs/ are written (default: the folder of this script)

What is written.
    conventions.txt                     the figures of the three trials, one line per file and per reading; also printed
    logs/conventions_state_<file>.log   raw output of psi_ref --state (trial 1)
    logs/conventions_psit_<file>.log    raw output of psit sample (trial 3)
    logs/conventions_psi_ref_<file>_c<c>.log   raw output of psi_ref --points (trial 3), for c = 0.0 and 0.5
No time is read.

Exit code: 0 if every trial asked for is written; 1 if the third trial could not be done or is incomplete (psit
missing, or a run of psit or of psi_ref that fails: conventions.txt says which); 2 if psi_ref or one of the test files
is missing (nothing is written).

Needs Python 3.9 or later and NumPy; build/psi_ref and, for the third trial, build/psit and an NVIDIA GPU
(tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere); the test files at 64^3 left in
build/tests/cases/ by `python tests/test_codec.py`.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
PSI_REF = "build/psi_ref" + EXE  # as written in the logs; the programs are started by their absolute path
PSIT = "build/psit" + EXE
WORK = "build/work/independent_readers/conventions"
CASES = ("d2_base", "d3_base", "d2_bound", "d3_bound")
lines: list = []


def say(text: str = "") -> None:
    lines.append(text)
    print(text, flush=True)


def relative(path: str) -> str:
    """A path under the root, as the commands write it: relative to the root, forward slashes."""
    try:
        return (ROOT / path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        print(f"{path} is not under the root of the repository: the commands name paths relative to the root. Nothing was run.", file=sys.stderr)
        sys.exit(2)


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def run(program: str, arguments: list, log: Path):
    """Runs a program of build/ from the root, writes its raw output (standard output, then standard error) to a log
    file, and returns (exit code, standard output)."""
    p = subprocess.run([str(ROOT / program)] + arguments, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = (stream.decode("utf-8", "replace").replace("\r", "") for stream in (p.stdout, p.stderr))
    log.write_text("".join(part if part.endswith("\n") or not part else part + "\n" for part in (output, error)), encoding="utf-8", newline="\n")
    return p.returncode, output


# --- Reading of the files ---------------------------------------------------------------------------------------------

def read_header(psi: Path) -> dict:
    """Fields of StreamHeader that are useful here (docs/FORMAT.md, section 2)."""
    h = struct.unpack("<8s8i6f", psi.read_bytes()[:64])
    return dict(n=h[1], degree=h[2], mf=h[3], levels=h[4], fb=h[5], frames=h[7], first=h[8], u_rms=h[11])


def read_probe(path: Path) -> dict:
    """Probe file of `pack --probe` (docs/FORMAT.md, section 11)."""
    d = path.read_bytes()
    n, degree, nl0, frame, points, _, fine_present, _ = struct.unpack("<8I", d[8:40])
    scale, fine_scale, _, u_rms, _, mx, my, mz = struct.unpack("<8f", d[40:72])
    p = 72
    lattice = np.frombuffer(d, dtype="<i2", count=3 * nl0 ** 3, offset=p).reshape(nl0, nl0, nl0, 3)   # z, y, x, component
    p += 6 * nl0 ** 3
    fine = None
    if fine_present:
        fine = np.frombuffer(d, dtype="i1", count=3 * n ** 3, offset=p).reshape(n, n, n, 3)
        p += 3 * n ** 3
    positions = np.frombuffer(d, dtype="<f4", count=3 * points, offset=p).reshape(3, points).astype(np.float64)
    p += 12 * points
    velocities = np.frombuffer(d, dtype="<f4", count=3 * points, offset=p).reshape(3, points).astype(np.float64)
    return dict(n=n, degree=degree, nl0=nl0, frame=frame, lattice=lattice, fine=fine, positions=positions, velocities=velocities,
                scale=scale, fine_scale=fine_scale, u_rms=u_rms, mean=np.array([mx, my, mz])[:, None])


def read_state(cases: str, name: str, position: int, e: dict, logs: Path) -> list:
    """Decoded state of a frame by `psi_ref --state`, in the layout of a raw key (section 5)."""
    out = f"{WORK}/state.bin"
    (ROOT / out).unlink(missing_ok=True)
    code, _ = run(PSI_REF, [f"{cases}/{name}.psi", "--state", str(position), out], logs / f"conventions_state_{name}.log")
    if code != 0 or not (ROOT / out).is_file():
        print(f"psi_ref --state failed on {cases}/{name}.psi (exit code {code}): see logs/conventions_state_{name}.log. Nothing is written.",
              file=sys.stderr)
        sys.exit(2)
    d = (ROOT / out).read_bytes()
    states, p = [], 0
    for level in range(e["levels"]):
        nl = (e["n"] // e["mf"]) >> level
        if level == 0:
            a = np.frombuffer(d, dtype="i1", count=3 * nl ** 3, offset=p)
            p += 3 * nl ** 3
        else:
            a = np.frombuffer(d, dtype="<i2", count=3 * nl ** 3, offset=p)
            p += 6 * nl ** 3
        states.append(a.astype(np.int64).reshape(nl, nl, nl, 3))
    return states


# --- Trial 1: refinement ----------------------------------------------------------------------------------------------

def refine_axis(g: np.ndarray, axis: int, degree: int, alignment: str) -> np.ndarray:
    """Doubles the number of nodes along one axis, without a division. `alignment`: "adopted" or "other"."""
    shape = list(g.shape)
    shape[axis] *= 2
    fine = np.zeros(shape, dtype=np.int64)
    even = [slice(None)] * g.ndim
    odd = [slice(None)] * g.ndim
    even[axis], odd[axis] = slice(0, None, 2), slice(1, None, 2)
    before, after = np.roll(g, 1, axis=axis), np.roll(g, -1, axis=axis)   # g[i-1], g[i+1]
    if degree == 2 and alignment == "adopted":
        fine[tuple(even)], fine[tuple(odd)] = 3 * g + before, 3 * g + after
    elif degree == 2:
        fine[tuple(even)], fine[tuple(odd)] = 3 * g + after, g + 3 * after
    elif alignment == "adopted":
        fine[tuple(even)], fine[tuple(odd)] = before + 6 * g + after, 4 * g + 4 * after
    else:
        fine[tuple(even)], fine[tuple(odd)] = 4 * before + 4 * g, before + 6 * g + after
    return fine


def synthesis(states: list, fb: int, degree: int, alignment: str, rounding: str) -> np.ndarray:
    """total = (state << fb) + refinement of the total of the coarser level, from the coarsest level to level 0."""
    bits = 6 if degree == 2 else 9
    total = states[-1] << fb
    for state in reversed(states[:-1]):
        r = total
        for axis in range(3):
            r = refine_axis(r, axis, degree, alignment)
        if rounding == "to nearest":
            r = (r + (1 << (bits - 1))) >> bits
        elif rounding == "down":
            r = r >> bits
        else:   # towards zero
            r = np.sign(r) * (np.abs(r) >> bits)
        total = (state << fb) + r
    return total


def trial_refinement(cases: str, logs: Path) -> None:
    say("1. Refinement of the pyramid: nodes of the sampling lattice that differ from those of the probe")
    say("   (files without an overflow of level 0; frame of the probe)")
    for name in ("d2_base", "d3_base"):
        e, s = read_header(ROOT / cases / f"{name}.psi"), read_probe(ROOT / cases / f"{name}.probe")
        states = read_state(cases, name, s["frame"] - e["first"], e, logs)
        target = s["lattice"].astype(np.int64)
        for alignment in ("adopted", "other"):
            for rounding in ("to nearest", "down", "towards zero"):
                t = synthesis(states, e["fb"], e["degree"], alignment, rounding)
                t = ((t + 32768) % 65536) - 32768   # modulo 2^16, read back as a signed integer
                differing = int(np.count_nonzero((t != target).any(axis=3)))
                what = f"alignment {alignment}, rounding {rounding}"
                say(f"   {name}.psi, degree {e['degree']}, {what:<42} : {differing:>5} nodes out of {target.shape[0] ** 3}")


# --- Trial 2: position of the nodes -----------------------------------------------------------------------------------

def weights(u: np.ndarray, degree: int):
    """Index of the first node of the support, weights and derivatives of the B-spline (section 6.7)."""
    if degree == 2:
        m = np.floor(u + 0.5)
        r = u - m
        return (m.astype(np.int64) - 1, np.stack([0.5 * (0.5 - r) ** 2, 0.75 - r * r, 0.5 * (0.5 + r) ** 2]),
                np.stack([-(0.5 - r), -2 * r, 0.5 + r]))
    i = np.floor(u)
    t = u - i
    return (i.astype(np.int64) - 1,
            np.stack([(1 - t) ** 3 / 6, (3 * t ** 3 - 6 * t * t + 4) / 6, (-3 * t ** 3 + 3 * t * t + 3 * t + 1) / 6, t ** 3 / 6]),
            np.stack([-(1 - t) ** 2 / 2, (3 * t * t - 4 * t) / 2, (-3 * t * t + 2 * t + 1) / 2, t * t / 2]))


def curl(T: np.ndarray, positions: np.ndarray, step: float, d: float, degree: int) -> np.ndarray:
    """Curl of the potential of coefficients T[z, y, x, component], derivatives taken per step of the lattice."""
    N = T.shape[0]
    (ix, wx, dx), (iy, wy, dy), (iz, wz, dz) = (weights(positions[a] / step - d, degree) for a in range(3))
    g = np.zeros((3, 3, positions.shape[1]))   # g[component, axis]
    for a in range(degree + 1):
        for b in range(degree + 1):
            for c in range(degree + 1):
                v = T[(iz + a) % N, (iy + b) % N, (ix + c) % N].T.astype(np.float64)
                g[:, 0] += v * (dx[c] * wy[b] * wz[a])
                g[:, 1] += v * (wx[c] * dy[b] * wz[a])
                g[:, 2] += v * (wx[c] * wy[b] * dz[a])
    return np.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])


def trial_nodes(cases: str) -> None:
    say("2. Position of the nodes: largest velocity difference from the probe, in u', node i being at (i + d) steps")
    for name in CASES:
        s = read_probe(ROOT / cases / f"{name}.probe")
        adopted = 0.5 if s["degree"] == 2 else 0.0
        h0, voxel = 2 * math.pi / s["nl0"], 2 * math.pi / s["n"]
        choices = [(d, None) for d in (0.0, 0.5)] if s["fine"] is None else [(adopted, d) for d in (0.0, 0.5)]
        for d, d_fine in choices:
            u = s["mean"] + s["scale"] * curl(s["lattice"], s["positions"], h0, d, s["degree"])
            if s["fine"] is not None:
                u = u + s["fine_scale"] * curl(s["fine"], s["positions"], voxel, d_fine, s["degree"])
            difference = float(np.abs(u - s["velocities"]).max()) / s["u_rms"]
            what = f"lattice d = {d}" if d_fine is None else f"lattice d = {d}, fine level d = {d_fine}"
            say(f"   {name}.psi, degree {s['degree']}, {what:<38} : {difference:.3e}")


# --- Trial 3: position of the voxels ----------------------------------------------------------------------------------

def trial_voxels(cases: str, gpu: bool, logs: Path) -> bool:
    """Returns True if the trial is complete."""
    say("3. Position of the centre of the voxel of index (i, j, k): largest difference, in u', between `psit sample` (grid")
    say("   indices) and the formulas of section 6.7 at the point ((i + c) voxel, (j + c) voxel, (k + c) voxel)")
    if not gpu:
        say("   not done: --no-gpu")
        return True
    if not (ROOT / PSIT).is_file():
        say(f"   not done: {PSIT} not found")
        return False
    complete = True
    indices = [(0, 0, 0), (12.5, 40, 7.25), (63, 1, 30), (31, 31, 31), (5.75, 60.5, 17.125)]
    x = 20261116
    for _ in range(27):   # points drawn by a congruential sequence with a fixed seed, to a sixteenth of a voxel
        point = []
        for _ in range(3):
            x = (1103515245 * x + 12345) % (1 << 31)
            point.append((x >> 8) % 1024 / 16.0)
        indices.append(tuple(point))
    points_file = f"{WORK}/points.txt"
    for name in CASES:
        psi = f"{cases}/{name}.psi"
        e = read_header(ROOT / psi)
        position = e["frames"] - 1
        arguments = ["sample", psi, "--frame", str(position)]
        for p in indices:
            arguments += ["--at", "%.10g,%.10g,%.10g" % p]
        # src/psit.cpp, cmd_sample, one line per point:
        #   printf("%g %g %g  %.9g %.9g %.9g\n", x[i], y[i], z[i], u[i], v[i], w[i]);
        code, output = run(PSIT, arguments, logs / f"conventions_psit_{name}.log")
        gpu_field = []
        for line in output.splitlines():
            fields = line.split()
            if len(fields) == 6:
                gpu_field.append([float(v) for v in fields[3:]])
        if code != 0 or len(gpu_field) != len(indices):
            say(f"   {name}.psi: `psit sample` failed (exit code {code}, {len(gpu_field)} points returned out of {len(indices)})")
            complete = False
            continue
        voxel = 2 * math.pi / e["n"]
        for c in (0.0, 0.5):
            (ROOT / points_file).write_text("".join("%.17g %.17g %.17g\n" % tuple((v + c) * voxel for v in p) for p in indices))
            # tools/reference_reader.cpp, write_points, one line per point:
            #   std::printf("POINT %.9g %.9g %.9g %.9g %.9g %.9g\n", pos[0], pos[1], pos[2], u[0], u[1], u[2]);
            code, output = run(PSI_REF, [psi, "--points", points_file, "--frame", str(position)], logs / f"conventions_psi_ref_{name}_c{c}.log")
            cpu_field = [[float(v) for v in line.split()[4:7]] for line in output.splitlines() if line.startswith("POINT")]
            if code != 0 or len(cpu_field) != len(indices):
                say(f"   {name}.psi: `psi_ref --points` failed (exit code {code}, {len(cpu_field)} points returned out of {len(indices)})")
                complete = False
                continue
            difference = float(np.abs(np.array(cpu_field) - np.array(gpu_field)).max()) / e["u_rms"]
            say(f"   {name}.psi, degree {e['degree']}, frame at position {position}, {len(indices)} points, c = {c} : {difference:.3e}")
    return complete


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Trials of the conventions of docs/FORMAT.md, sections 6.5 and 6.7.")
    parser.add_argument("--cases", default="build/tests/cases", help="folder of the test cases at 64^3, from the root")
    parser.add_argument("--no-gpu", action="store_true", help="trials 1 and 2 only: psit is not run")
    parser.add_argument("--output", default=str(HERE), help="folder where conventions.txt and logs/ are written (default: the folder of this script)")
    args = parser.parse_args()
    cases, output, gpu = relative(args.cases), Path(args.output).resolve(), not args.no_gpu
    if not (ROOT / PSI_REF).is_file():
        print(f"{PSI_REF} not found: build the instruments first (tools/build.ps1 -Tools). Nothing was run.", file=sys.stderr)
        return 2
    missing = [f"{cases}/{name}{suffix}" for name in CASES for suffix in (".psi", ".probe") if not (ROOT / cases / f"{name}{suffix}").is_file()]
    if missing:
        print(f"Test files at 64^3 not found: {', '.join(missing)}. Run python tests/test_codec.py first. Nothing was run.", file=sys.stderr)
        return 2
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (ROOT / WORK).mkdir(parents=True, exist_ok=True)

    say("Conventions established by trial (docs/FORMAT.md, sections 6.5 and 6.7), on the test files at 64^3")
    say(f"psi_ref: SHA-256 {fingerprint(ROOT / PSI_REF)} (first 16 digits); psit: "
        + (f"SHA-256 {fingerprint(ROOT / PSIT)} (first 16 digits)" if (ROOT / PSIT).is_file() else "absent"))
    say()
    trial_refinement(cases, logs)
    say()
    trial_nodes(cases)
    say()
    complete = trial_voxels(cases, gpu, logs)
    (output / "conventions.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
