#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Tests of the public interface: the psitreillis library, its Python module and the psit command line.

    python tests/test_api.py [--build build] [--seq build/tests/turb64.seq] [--only pattern]

They need an NVIDIA GPU, NumPy, and a build folder holding psit, psi_stream, dns and the shared library
(tools/build.ps1 -Tools). The 64^3 test sequence is produced with dns if it is missing. Everything they check is
deterministic: bytes, errors, return codes; no timing is read.

One line per case (ok / FAIL), then a total. Exit code: 0 if no case fails, 1 otherwise.
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FIRST, COUNT = 3, 36          # frames of the sequence used by the cases, as in tests/test_codec.py
TOLERANCE = 0.04              # bound held at 64^3 in degree 3 (tests/README.md)
ROUNDING = 5e-5               # half the certification margin: what two GPU kernels may differ by, in u_rms

cases = []


def case(name):
    def register(function):
        cases.append((name, function))
        return function
    return register


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(*command, expect=0):
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=ROOT)
    codes = expect if isinstance(expect, (tuple, list)) else (expect,)
    require(done.returncode in codes, f"{Path(str(command[0])).name} {' '.join(str(c) for c in command[1:])}: exit code {done.returncode}\n{done.stdout}{done.stderr}")
    return done


class Context:
    def __init__(self, build: Path, seq: Path):
        self.build, self.seq = build, seq
        exe = ".exe" if sys.platform == "win32" else ""
        self.psit, self.pack, self.dns = build / f"psit{exe}", build / f"psi_stream{exe}", build / f"dns{exe}"
        self.work = build / "api_tests"
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        if not seq.is_file():
            seq.parent.mkdir(parents=True, exist_ok=True)
            run(self.dns, "--n", 64, "--nu", 0.02, "--spinup", 12, "--frames", 40, "--steps-per-frame", 4, "--seed", 42, "--out", seq)
        header = seq.read_bytes()[:72]
        _, self.n, self.seq_frames, self.dt, _, self.u_rms, _, _ = struct.unpack("<8sii5f", header[:36])
        self.frames = np.memmap(seq, dtype=np.float32, mode="r", offset=72, shape=(self.seq_frames, 3, self.n, self.n, self.n))
        # The two reference files of every case: without and with an error bound, written by the command line.
        self.base, self.bound = self.work / "base.psi", self.work / "bound.psi"
        run(self.psit, "compress", seq, self.base, "--first", FIRST, "--count", COUNT, "--quiet")
        run(self.psit, "compress", seq, self.bound, "--first", FIRST, "--count", COUNT, "--tolerance", TOLERANCE, "--quiet")

    def original(self, frame: int) -> np.ndarray:
        return np.asarray(self.frames[FIRST + frame])


def errors(decoded: np.ndarray, original: np.ndarray, u_rms: float):
    difference = decoded.astype(np.float64) - original
    rms = float(np.sqrt((difference ** 2).sum() / (original.astype(np.float64) ** 2).sum()))
    return rms, float(np.abs(difference).max() / u_rms)


@case("cli.same_bytes_as_pack")
def _(c, psi):
    """psit compress writes the file psi_stream pack writes, byte for byte, without and with an error bound."""
    for name, extra in (("base", []), ("bound", ["--tolerance", TOLERANCE])):
        reference = c.work / f"pack_{name}.psi"
        run(c.pack, "pack", "--data", c.seq, "--file", reference, "--degree", 3, "--start", FIRST, "--frames", COUNT, *extra)
        require(sha256(reference) == sha256(c.work / f"{name}.psi"), f"{name}.psi differs from the file written by psi_stream pack")


@case("cli.info")
def _(c, psi):
    out = run(c.psit, "info", c.bound).stdout
    require("64^3, 36 frames" in out and "certified on every voxel" in out, f"unexpected output of psit info:\n{out}")


@case("cli.decompress_npy")
def _(c, psi):
    """psit decompress writes a .npy file NumPy reads, equal to what the library decodes."""
    target = c.work / "base.npy"
    run(c.psit, "decompress", c.base, target, "--quiet")
    array = np.load(target, mmap_mode="r")
    require(array.shape == (COUNT, 3, c.n, c.n, c.n) and array.dtype == np.float32, f"shape {array.shape}, dtype {array.dtype}")
    with psi.open(c.base) as f:
        for frame in (0, 17, COUNT - 1):
            require(np.array_equal(array[frame], f.decode(frame)), f"frame {frame} of the .npy file differs from Reader.decode")


@case("api.info")
def _(c, psi):
    with psi.open(c.bound) as f:
        i = f.info
        require((i.n, i.frames, i.degree, i.first_frame) == (64, COUNT, 3, FIRST), f"unexpected info: {i}")
        require(i.sealed and i.has_bound and i.bound_held and abs(i.tolerance - TOLERANCE) < 1e-7, f"unexpected bound: {i}")
        require(abs(i.u_rms - c.u_rms) < 1e-7 and abs(i.dt - c.dt) < 1e-9, "u_rms or dt differ from the sequence header")
        keys = [k for k in range(COUNT) if f.frame_info(k).is_key]
        require(keys == [0, 32], f"key frames at {keys}, expected [0, 32]")


@case("api.errors_match_the_encoder")
def _(c, psi):
    """The error of each decoded frame, measured here with NumPy, is the one the encoder recorded in the file."""
    with psi.open(c.base) as f:
        for frame in range(COUNT):
            rms, worst = errors(f.decode(frame), c.original(frame), c.u_rms)
            recorded = f.frame_info(frame)
            require(abs(rms - recorded.rms_error) <= 2e-4 * recorded.rms_error, f"frame {frame}: rms error {rms:.6e}, recorded {recorded.rms_error:.6e}")
            require(abs(worst - recorded.max_error) <= ROUNDING, f"frame {frame}: largest error {worst:.6f}, recorded {recorded.max_error:.6f}")


@case("api.bound_is_held")
def _(c, psi):
    """With a tolerance, every component of every voxel of every decoded frame is within the tolerance."""
    with psi.open(c.bound) as f:
        worst = max(errors(f.decode(frame), c.original(frame), c.u_rms)[1] for frame in range(COUNT))
    require(worst <= TOLERANCE, f"largest error {worst:.6f} u_rms for a tolerance of {TOLERANCE}")
    return f"largest error {100 * worst:.4f} % of u_rms for a bound of {100 * TOLERANCE:.0f} %"


@case("api.sample_matches_the_grid")
def _(c, psi):
    """Sampling at voxel centres returns the decoded grid; coordinates are periodic."""
    rng = np.random.default_rng(1)
    voxels = rng.integers(0, c.n, size=(20000, 3))
    for path in (c.base, c.bound):
        with psi.open(path) as f:
            grid = f.decode(20)
            sampled = f.sample(20, voxels.astype(np.float32))
            expected = np.stack([grid[component, voxels[:, 2], voxels[:, 1], voxels[:, 0]] for component in range(3)], axis=1)
            require(np.array_equal(sampled, expected), f"{path.name}: sampling at voxel centres differs from the decoded grid")
            shifted = voxels.astype(np.float32) + np.array([c.n, -c.n, 2 * c.n], dtype=np.float32)
            require(np.array_equal(f.sample(20, shifted), sampled), f"{path.name}: sampling is not periodic")


@case("api.divergence_free")
def _(c, psi):
    """The sampled field is divergence-free: by centred differences at random points, off the grid, the divergence is
    the truncation error of the differences, far below the velocity gradient."""
    rng = np.random.default_rng(2)
    points = rng.uniform(0, c.n, size=(50000, 3)).astype(np.float32)
    step = 0.125
    with psi.open(c.bound) as f:
        gradient = np.empty((points.shape[0], 3, 3))
        for axis in range(3):
            offset = np.zeros(3, dtype=np.float32)
            offset[axis] = step
            gradient[:, :, axis] = (f.sample(10, points + offset).astype(np.float64) - f.sample(10, points - offset)) / (2 * step)
    divergence = gradient[:, 0, 0] + gradient[:, 1, 1] + gradient[:, 2, 2]
    ratio = float(np.sqrt((divergence ** 2).mean() / (gradient ** 2).sum(axis=(1, 2)).mean()))
    require(ratio < 2e-3, f"RMS divergence is {ratio:.2e} of the RMS gradient")
    return f"RMS divergence {ratio:.1e} of the RMS gradient, by differences at +-{step} voxel"


@case("api.random_access")
def _(c, psi):
    """A frame read directly is the frame read in sequence, across a key frame and backwards."""
    with psi.open(c.bound) as f:
        sequential = {frame: f.decode(frame).copy() for frame in range(COUNT)}
    with psi.open(c.bound) as f:
        for frame in (35, 5, 33, 32, 31, 0, 20, 20):
            require(np.array_equal(f.decode(frame), sequential[frame]), f"frame {frame} read directly differs from the sequential read")


@case("api.errors_are_reported")
def _(c, psi):
    """Invalid calls raise PsiError and leave the library usable."""
    def refused(call, what):
        try:
            call()
        except psi.PsiError as e:
            return str(e)
        raise AssertionError(f"{what}: no error raised")
    refused(lambda: psi.open(c.work / "missing.psi"), "missing file")
    truncated = c.work / "truncated.psi"
    truncated.write_bytes(c.base.read_bytes()[:5000])
    refused(lambda: psi.open(truncated), "truncated file")
    not_psi = c.work / "not.psi"
    not_psi.write_bytes(b"\x00" * 4096)
    refused(lambda: psi.open(not_psi), "file that is not a .psi file")
    with psi.open(c.base) as f:
        refused(lambda: f.decode(COUNT), "frame out of range")
        refused(lambda: f.sample(0, np.zeros((4, 2), dtype=np.float32)), "points of the wrong shape")
        f.decode(1)
    small = np.zeros((6, 3, 48, 48, 48), dtype=np.float32)
    require("power of two" in refused(lambda: psi.compress(small, c.work / "small.psi"), "grid size that is not a power of two"), "unexpected message")
    require("5 frames" in refused(lambda: psi.compress(np.asarray(c.frames[:3]), c.work / "short.psi"), "too few frames"), "unexpected message")


@case("api.altered_file_is_refused")
def _(c, psi):
    """A bit flipped in a frame is detected when that frame is decoded; the frames before it still decode."""
    altered = c.work / "altered.psi"
    data = bytearray(c.base.read_bytes())
    with psi.open(c.base) as f:
        offset = sum(f.frame_info(k).bytes for k in range(5))
    data[208 + offset + 300] ^= 0x10   # inside frame 5: the frames follow the 208-byte header
    altered.write_bytes(bytes(data))
    try:
        f = psi.open(altered)
    except psi.PsiError:
        return "refused at opening"
    with f:
        f.decode(4)
        try:
            f.decode(5)
        except psi.PsiError as e:
            return f"frame 5 refused: {str(e)[:60]}"
    raise AssertionError("the altered frame was decoded without an error")


@case("py.compress_from_memory")
def _(c, psi):
    """Compressing a NumPy array gives the frames the command line gives from the .seq file."""
    target = c.work / "memory.psi"
    info = psi.compress(c.frames, target, dt=c.dt, first_frame=FIRST, frame_count=COUNT, tolerance=TOLERANCE, u_rms=c.u_rms)
    require(info.bound_held and info.frames == COUNT, f"unexpected info: {info}")
    require(target.stat().st_size == c.bound.stat().st_size, "the file does not have the size of the one written from the .seq file")
    with psi.open(target) as a, psi.open(c.bound) as b:
        for frame in (0, 9, 33, COUNT - 1):
            require(np.array_equal(a.decode(frame), b.decode(frame)), f"frame {frame} differs from the file written from the .seq file")


@case("py.u_rms_is_measured")
def _(c, psi):
    """Without u_rms, the library measures it from the frames: close to the value of the sequence header."""
    info = psi.compress(np.asarray(c.frames[FIRST:FIRST + 8]), c.work / "measured.psi", dt=c.dt)
    require(abs(info.u_rms / c.u_rms - 1) < 0.05, f"measured u_rms {info.u_rms}, header {c.u_rms}")
    return f"measured {info.u_rms:.4f}, sequence header {c.u_rms:.4f}"


@case("cli.npy_and_raw_inputs")
def _(c, psi):
    """A .npy file and a raw float32 file compress to the frames of the .seq file."""
    for extension, extra in ((".npy", []), (".f32", ["--n", c.n])):
        field, target = c.work / f"field{extension}", c.work / f"from{extension}.psi"
        run(c.psit, "convert", c.seq, field, "--first", FIRST, "--count", COUNT, "--quiet")
        run(c.psit, "compress", field, target, *extra, "--dt", c.dt, "--u-rms", repr(c.u_rms), "--quiet")
        with psi.open(target) as a, psi.open(c.base) as b:
            for frame in (0, 20, COUNT - 1):
                require(np.array_equal(a.decode(frame), b.decode(frame)), f"{extension}: frame {frame} differs from the file written from the .seq file")
    require(np.array_equal(np.load(c.work / "field.npy", mmap_mode="r")[7], c.original(7)), "psit convert does not copy the frames")


@case("api.bound_too_tight")
def _(c, psi):
    """A bound the codec cannot reach is refused with a message, and the same output path can be written again."""
    target = c.work / "tight.psi"
    try:
        psi.compress_file(c.seq, target, first_frame=FIRST, frame_count=COUNT, tolerance=0.005)
    except psi.PsiError as e:
        require("too tight" in str(e), f"unexpected message: {e}")
    else:
        raise AssertionError("a tolerance of 0.5 % was accepted")
    info = psi.compress_file(c.seq, target, first_frame=FIRST, frame_count=COUNT, tolerance=TOLERANCE)
    require(info.bound_held and sha256(target) == sha256(c.bound), "the second compression to the same path is not the reference file")


@case("api.memory_is_released")
def _(c, psi):
    """Opening, decoding and closing a file many times does not grow the memory used on the GPU."""
    def used():
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        return int(out.stdout.split()[0])
    growth = None
    for _ in range(3):   # another program may allocate on the card during a measure: keep the smallest growth
        with psi.open(c.bound) as f:
            f.decode(0)
        before = used()
        for _ in range(40):
            with psi.open(c.bound) as f:
                f.decode(3)
        psi.compress_file(c.seq, c.work / "again.psi", first_frame=FIRST, frame_count=8)
        growth = used() - before if growth is None else min(growth, used() - before)
        if growth <= 64:
            break
    require(growth <= 64, f"{growth} MiB more on the GPU after 40 open/decode/close cycles and one compression")
    return f"{growth} MiB after 40 open/decode/close cycles and one compression"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", default="build")
    parser.add_argument("--seq", default="build/tests/turb64.seq")
    parser.add_argument("--only", default="*", help="run the cases whose name matches this pattern")
    args = parser.parse_args()
    build = (ROOT / args.build).resolve()
    os.environ.setdefault("PSITREILLIS_LIBRARY", str(build / ("psitreillis.dll" if sys.platform == "win32" else "libpsitreillis.so")))
    sys.path.insert(0, str(ROOT / "python"))
    import psitreillis as psi

    print(f"psitreillis {psi.version()}, library {Path(os.path.relpath(os.environ['PSITREILLIS_LIBRARY'], ROOT)).as_posix()}")
    context = Context(build, (ROOT / args.seq).resolve())
    failed = 0
    selected = [(name, function) for name, function in cases if fnmatch.fnmatch(name, args.only)]
    for name, function in selected:
        try:
            detail = function(context, psi)
            print(f"ok    {name}" + (f" : {detail}" if detail else ""))
        except Exception as e:  # noqa: BLE001 - a failing case must not stop the others
            failed += 1
            print(f"FAIL  {name} : {type(e).__name__}: {e}")
    print(f"\nTOTAL : {len(selected)} cases, {len(selected) - failed} ok, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
