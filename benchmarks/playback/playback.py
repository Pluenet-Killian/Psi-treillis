# SPDX-License-Identifier: Apache-2.0
"""Cost of one replayed frame: raw frames, cuZFP at 2 bits and Psi-treillis, measured by the visual comparator.

    python benchmarks/playback/playback.py run [--force]   log playback.csv, outputs of psi_view in logs/
    python benchmarks/playback/playback.py summary         the table, from the log

One protocol for the whole table (fixed before the measurement). For each grid size (256^3 and 512^3) the comparator
build/psi_view is run once per pair of chains, in synchronised mode (both sides advance through the same frames), with
4 million particles per side sorted in Morton order, 2 RK4 steps per frame, on the error view:

    pair 1   left: cuZFP at a fixed rate of 2 bits per value, decoded on the GPU to a half-precision 3D texture
             right: Psi-treillis of degree 3, default settings, no bound
    pair 2   left: raw float32 frames read from disk and uploaded to a half-precision 3D texture
             right: the same Psi-treillis file

The files compared are encoded right here by the current programs, in build/work/playback/: 33 frames from frame 16 of
turb256.seq, 64 frames from frame 0 of turb512.seq. Every figure of the log is measured by psi_view during the run and
printed at its end: data read per frame, disk, bus, decoding and sort times, time of one RK4 step, cost of one frame
(the sum of these), GPU memory of the side, divergence of the field the particles see on a slice (relative to the
velocity gradient), error of the field on a slice against the raw frame.

The times are valid only if the GPU runs nothing else: the script stops if a compute program occupies it, or if its
utilisation is above 5 %; --force overrides this, and the log then says so.

Needs build/ with the instruments (tools/build.ps1 -Tools; psi_view exists on Windows only) and the two sequences.
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
WORK = "build/work/playback"
GRIDS = {256: dict(start=16, frames=33), 512: dict(start=0, frames=64)}
PARTICLES = 4000000
COLUMNS = ["grid", "pair", "side", "chain", "mib_read_per_frame", "disk_ms", "bus_ms", "decoding_ms", "sort_ms", "rk4_steps_per_frame", "rk4_step_ms",
           "frame_cost_ms", "frames_advanced", "gpu_memory_mib", "rendering_ms", "divergence_over_gradient", "slice_error_rms", "gpu_free"]

SIDE = re.compile(r"^(Left|Right) : (.+)$")
PER_FRAME = re.compile(r"^  per frame    : ([\d.]+) MiB read on average, disk ([\d.]+) ms, bus ([\d.]+) ms, decoding ([\d.]+) ms, sort ([\d.]+) ms, (\d+) RK4 steps of ([\d.]+) ms$")
COST = re.compile(r"^  cost         : ([\d.]+) ms per frame, that is [\d.]+ frames per second by its own costs; (\d+) frames advanced$")
MEMORY = re.compile(r"^  GPU memory   : ([\d.]+) MiB; rendering ([\d.]+) ms$")
DIVERGENCE = re.compile(r"^  divergence   : (\S+) of the gradient \(RMS over the slice\)$")
ERROR = re.compile(r"^  error        : (\S+) \(RMS over the slice, against the raw frame\)$")


def call(command, log=None):
    # The program is given by its full path (Windows does not start a relative path with forward slashes); its
    # arguments stay relative to the root, so that the logs name no folder of the machine.
    program = str(ROOT / command[0]) if str(command[0]).startswith("build/") else str(command[0])
    p = subprocess.run([program] + [str(c) for c in command[1:]], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    out = p.stdout.decode("utf-8", "replace").replace("\r", "")
    if log is not None:
        log.write_text(out, encoding="utf-8", newline="\n")
    if p.returncode != 0:
        sys.exit(f"{' '.join(str(c) for c in command)}: exit code {p.returncode}\n{out[-600:]}")
    return out


def gpu_is_free():
    apps = call(["nvidia-smi", "--query-compute-apps=process_name", "--format=csv,noheader"]).split("\n")
    busy = [a for a in apps if re.search(r"(psi_|tracers|dns|python|sz3|cuSZp|zfp)[^\\/]*$", a)]
    load = max(int(x.split()[0]) for x in call(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader"]).split("\n") if x.strip())
    return (not busy and load <= 5), busy, load


def parse(out):
    """Two dictionaries, left then right, from the final report of psi_view. Fails if a line is missing."""
    sides, current = [], None
    for line in out.split("\n"):
        m = SIDE.match(line)
        if m:
            current = dict(side=m.group(1).lower(), chain=m.group(2), divergence_over_gradient="", slice_error_rms="")
            sides.append(current)
            continue
        if current is None:
            continue
        for pattern, names in ((PER_FRAME, ["mib_read_per_frame", "disk_ms", "bus_ms", "decoding_ms", "sort_ms", "rk4_steps_per_frame", "rk4_step_ms"]),
                               (COST, ["frame_cost_ms", "frames_advanced"]), (MEMORY, ["gpu_memory_mib", "rendering_ms"]),
                               (DIVERGENCE, ["divergence_over_gradient"]), (ERROR, ["slice_error_rms"])):
            m = pattern.match(line)
            if m:
                current.update(zip(names, m.groups()))
    required = ["mib_read_per_frame", "frame_cost_ms", "gpu_memory_mib"]
    if len(sides) != 2 or any(k not in s for s in sides for k in required):
        sys.exit("the report of psi_view is not the one expected:\n" + out[-800:])
    return sides


def run(force):
    free, busy, load = gpu_is_free()
    if not free and not force:
        sys.exit(f"the GPU is not free (compute programs: {', '.join(busy) or 'none'}; utilisation {load} %): the times would be wrong. "
                 "Run again later, or with --force.")
    (ROOT / WORK).mkdir(parents=True, exist_ok=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    pack, view = f"build/psi_stream{EXE}", f"build/psi_view{EXE}"
    with open(HERE / "playback.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(COLUMNS)
        for n, g in GRIDS.items():
            sequence, psi, zfs = f"data/turb{n}.seq", f"{WORK}/turb{n}_d3.psi", f"{WORK}/turb{n}.zfs"
            call([pack, "pack", "--data", sequence, "--file", psi, "--degree", 3, "--start", g["start"], "--frames", g["frames"]], logs / f"pack_{n}.log")
            call([view, "packzfp", "--data", sequence, "--file", zfs, "--rate", 2, "--start", g["start"], "--frames", g["frames"]], logs / f"packzfp_{n}.log")
            for pair, left in (("cuzfp_psi", f"zfp:{zfs}"), ("raw_psi", f"raw:{sequence}")):
                out = call([view, "--left", left, "--right", f"psi:{psi}", "--data", sequence, "--view", 3, "--sync", "--sort", "--particles", PARTICLES,
                            "--start", g["start"], "--frames", g["frames"], "--width", 1600, "--height", 900,
                            "--snapshot", f"{WORK}/{pair}_{n}.png"], logs / f"view_{pair}_{n}.log")
                for s in parse(out):
                    w.writerow([n, pair] + [s.get(c, "") for c in COLUMNS[2:-1]] + ["yes" if free else f"no (utilisation {load} %)"])
                fo.flush()
    summary()


def summary():
    rows = list(csv.DictReader(open(HERE / "playback.csv", encoding="utf-8")))
    print("Cost of one replayed frame (4 million sorted particles per side, 2 RK4 steps per frame)")
    print(f"{'grid':>5s} {'pair':10s} {'side':6s} {'chain':46s} {'MiB read':>9s} {'frame ms':>9s} {'GPU MiB':>8s} {'divergence':>11s} {'error':>10s}")
    for r in rows:
        print(f"{r['grid']:>5s} {r['pair']:10s} {r['side']:6s} {r['chain'][:46]:46s} {float(r['mib_read_per_frame']):9.2f} {float(r['frame_cost_ms']):9.3f} "
              f"{float(r['gpu_memory_mib']):8.0f} {r['divergence_over_gradient']:>11s} {r['slice_error_rms']:>10s}")
    if any(r["gpu_free"] != "yes" for r in rows):
        print("WARNING: the GPU was not free during at least one run: the times must not be quoted.")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "run":
        run("--force" in sys.argv[2:])
    elif action == "summary":
        summary()
    else:
        print(__doc__)
        sys.exit(2)
