# SPDX-License-Identifier: Apache-2.0
"""Windowed-cutout protocol: trying the codec on a sub-volume that is not periodic (claim CTRL-1).

    python benchmarks/cutout/cutout.py control   on data/turb512.seq and data/turb512_nu0006.seq (claim CTRL-1)
    python benchmarks/cutout/cutout.py jhtdb     on a cutout of the public JHTDB dataset (claims JHTDB-3, JHTDB-4 and
                                                 REPR-1); prepared, never run to its end: see cutout_jhtdb.py

What is measured. Psi-treillis requires a periodic, divergence-free field; a cutout of a larger field is neither at
its edges. The protocol:
  1. multiply the cutout by a smooth window, equal to 1 at the centre and falling to 0 at the edges (half-cosine);
  2. project the result onto the periodic divergence-free fields (Leray projection, by FFT);
  3. compress that field, and measure the error only on the central cube, where the window is 1.
At the centre, the projection adds to the original field only the gradient of a harmonic potential, hence smooth: the
small scales there remain those of the original field. This is a hypothesis, which the control puts to the test.

The control compares the error at the centre of the windowed cutout with that of the whole field at 512^3, compressed
with the same settings (`psi_stream pack`, degree 3, lattice step of 2 voxels, without a bound and with a bound of 5 %
of u') on the same frames (0 to 7) and measured on the same voxels. Outside the criterion, the windowed cutout is also
compressed at a lattice step of 1 voxel, and by SPERR at the same absolute bound on frames 0, 3 and 7 (compression
ratio only). The .psi files are decoded through the public Python module of the library and compared with the
reference in NumPy, in double precision. No timing is recorded.

What it needs. build/ (tools/build.ps1 -Tools: psi_stream and the library), SPERR in build/codecs/
(benchmarks/codecs/build_codecs.ps1), NumPy, a GPU, and the two sequences at 512^3 of data/.

What it writes, in this folder:
  control.csv       one line per sequence, field (whole field or windowed cutout), setting and bound (columns: COLUMNS
                    below); rewritten from scratch by each run of `control`
  jhtdb.csv         the same lines for the cutout of JHTDB (never written so far)
  logs/pack_*.log   output of each run of `psi_stream pack`, after its command
The windowed sequences and the .psi files are written in build/work/cutout/ and deleted once measured.
"""

import csv
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

WORK = cc.BUILD / "work" / "cutout"
LOGS = HERE / "logs"
PACK = cc.BUILD / f"psi_stream{cc.EXE}"

# The protocol, fixed before any measurement (claim CTRL-1).
SIDE = 256                  # side of the cutout, in voxels
EDGE = 32                   # width of the half-cosine at each edge
CENTRE = slice(48, 208)     # central cube of 160^3 where the error is measured
ORIGIN = (128, 128, 128)    # corner (x, y, z) of the cutout in the field at 512^3
FRAMES = 8                  # frames 0 to 7
BOUND = 0.05                # bound on the largest error, as a fraction of u'
SPERR_FRAMES = (0, 3, 7)
COLUMNS = ["source", "eta_voxels", "u_rms", "field", "setting", "bound_over_urms", "exit_code", "compression_ratio",
           "centre_rms_error_relative", "centre_largest_error_over_urms", "note"]


def relative(path):
    """Path relative to the root, with forward slashes: the programs run from the root, and the logs then name no folder
    of the machine."""
    return Path(path).relative_to(ROOT).as_posix()


def without_machine_paths(text):
    """A versioned log does not carry the path of the machine: the root of the repository is removed from each line, and
    a line that still starts with an absolute path (the source file named by a failed check of a program) is dropped."""
    for root in (str(ROOT) + os.sep, ROOT.as_posix() + "/"):
        text = text.replace(root, "")
    return "\n".join(line for line in text.splitlines() if not re.match(r"[A-Za-z]:[\\/]|/", line.strip()))


def window():
    i = np.arange(SIDE, dtype=np.float64)
    w = np.ones(SIDE)
    ramp = 0.5 * (1.0 - np.cos(math.pi * (i[:EDGE] + 0.5) / EDGE))
    w[:EDGE], w[-EDGE:] = ramp, ramp[::-1]
    return w[:, None, None] * w[None, :, None] * w[None, None, :]


def project(u):
    """Leray projection of a field (3, m, m, m) indexed [component, z, y, x]: removes its part of non-zero divergence."""
    m = u.shape[1]
    k1 = np.fft.fftfreq(m, 1.0 / m)
    k1[m // 2] = 0.0                                   # the Nyquist mode has no defined derivative: left as it is
    kz, ky, kx = k1[:, None, None], k1[None, :, None], np.fft.rfftfreq(m, 1.0 / m)[None, None, :].copy()
    kx[..., -1] = 0.0
    spectrum = [np.fft.rfftn(u[c].astype(np.float64)) for c in range(3)]     # components x, y, z of the velocity
    k2 = kx * kx + ky * ky + kz * kz
    k2[k2 == 0.0] = 1.0                                # mean mode and pure Nyquist modes: nothing to project
    divergence = (kx * spectrum[0] + ky * spectrum[1] + kz * spectrum[2]) / k2
    out = np.empty_like(u, dtype=np.float32)
    for c, k in enumerate((kx, ky, kz)):
        out[c] = np.fft.irfftn(spectrum[c] - k * divergence, s=(m, m, m)).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("the projection produced non-finite values")
    return out


def write_sequence(path, frames, dt, u_rms, nu=0.0, kmax_eta=0.0, re_lambda=0.0):
    """Sequence file (src/sequence.h) of frames (3, SIDE, SIDE, SIDE)."""
    with open(path, "wb") as f:
        f.write(struct.pack("<8sii5f", b"TLSEQ01\0", SIDE, len(frames), dt, nu, u_rms, kmax_eta, re_lambda))
        f.write(b"\0" * 36)
        for t in frames:
            f.write(np.ascontiguousarray(t, dtype=np.float32).tobytes())


def pack(sequence, output, extra, first=0, count=FRAMES):
    """Runs `psi_stream pack` from the root, logs its output and returns its exit code."""
    command = [str(PACK), "pack", "--data", relative(sequence), "--file", relative(output), "--start", str(first), "--frames", str(count), "--no-seal", *extra]
    run = subprocess.run(command, cwd=ROOT, capture_output=True)
    text = (run.stdout + run.stderr).decode("utf-8", "replace")
    LOGS.mkdir(parents=True, exist_ok=True)
    (LOGS / f"pack_{output.stem}.log").write_text("# " + " ".join(["psi_stream", *command[1:]]) + "\n" + without_machine_paths(text) + "\n",
                                                  encoding="utf-8", newline="\n")
    return run.returncode


def centre_error(path, reference, u_rms):
    """Relative RMS error and largest error over u' on the central cube, and compression ratio of the file.
    reference(k) gives the field (3, SIDE, SIDE, SIDE) of frame k on the cutout; for a file of the whole field, the
    library decodes 512^3 and the cutout is taken from it."""
    psi = cc.psitreillis()
    x0, y0, z0 = ORIGIN
    errors = cc.ErrorSum()
    with psi.open(path) as f:
        whole = f.n != SIDE
        for k in range(f.frames):
            decoded = f.decode(k)
            if whole:
                decoded = decoded[:, z0:z0 + SIDE, y0:y0 + SIDE, x0:x0 + SIDE]
            errors.add(reference(k)[:, CENTRE, CENTRE, CENTRE], decoded[:, CENTRE, CENTRE, CENTRE])
        ratio = f.info.raw_bytes / f.info.file_bytes
    return errors.relative_rms(), errors.largest / u_rms, ratio


def sperr_ratio(frames, bound):
    """Compression ratio of SPERR on these frames, at an absolute bound on the largest error."""
    size = 0
    for u in frames:
        size += cc.compress_frame("sperr", u, bound, WORK)[0]
    return len(frames) * 12.0 * SIDE ** 3 / size


def append(log, row):
    new = not log.is_file()
    with open(log, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLUMNS)
        w.writerow([row.get(c, "") for c in COLUMNS])
    print("   " + ", ".join(f"{c} {row[c]}" for c in COLUMNS[3:] if row.get(c, "") != ""), flush=True)


def measure_cutout(log, base, windowed, u_rms, dt, label):
    """Compresses the windowed field with each setting and measures the error at the centre against that same field."""
    WORK.mkdir(parents=True, exist_ok=True)
    sequence = WORK / f"{label}.seq"
    write_sequence(sequence, windowed, dt, u_rms)
    for setting, extra in (("d3_mf2", ["--degree", "3", "--mf", "2"]), ("d3_mf1", ["--degree", "3", "--mf", "1"])):
        for bound in ("", BOUND):
            output = WORK / f"{label}_{setting}_{bound or 'none'}.psi"
            code = pack(sequence, output, extra + (["--tolerance", repr(bound)] if bound else []))
            row = dict(base, field="windowed cutout", setting=setting, bound_over_urms=bound, exit_code=code)
            if output.is_file() and output.stat().st_size > 4096 and code in (0, 3, 4):
                rms, largest, ratio = centre_error(output, lambda k: windowed[k], u_rms)
                row.update(compression_ratio=f"{ratio:.3f}", centre_rms_error_relative=f"{rms:.6e}", centre_largest_error_over_urms=f"{largest:.5e}")
            append(log, row)
            if output.is_file():
                output.unlink()
    ratio = sperr_ratio([windowed[k] for k in SPERR_FRAMES], BOUND * u_rms)
    append(log, dict(base, field="windowed cutout", setting="sperr", bound_over_urms=BOUND, exit_code=0, compression_ratio=f"{ratio:.3f}",
                     note=f"frames {', '.join(str(k) for k in SPERR_FRAMES)}"))
    sequence.unlink()


def control():
    log = HERE / "control.csv"
    if log.is_file():
        log.unlink()
    w = window()
    x0, y0, z0 = ORIGIN
    for name in ("turb512", "turb512_nu0006"):
        source = cc.DATA / f"{name}.seq"
        header = cc.read_header(source)
        eta = cc.eta_in_voxels(header)
        base = dict(source=name, eta_voxels=f"{eta:.4f}", u_rms=f"{header['u_rms']:.6f}")
        print(f"{name}: eta = {eta:.3f} voxel", flush=True)
        raw = np.memmap(source, dtype=np.float32, mode="r", offset=cc.SEQ_HEADER_BYTES, shape=(header["frames"], 3, header["n"], header["n"], header["n"]))
        cutouts = [np.array(raw[k, :, z0:z0 + SIDE, y0:y0 + SIDE, x0:x0 + SIDE]) for k in range(FRAMES)]
        # Whole field at 512^3, same settings, same frames: the error on the voxels of the central cube of the cutout.
        WORK.mkdir(parents=True, exist_ok=True)
        for bound in ("", BOUND):
            whole = WORK / f"{name}_whole_{bound or 'none'}.psi"
            code = pack(source, whole, ["--degree", "3", "--mf", "2"] + (["--tolerance", repr(bound)] if bound else []))
            row = dict(base, field="whole field", setting="d3_mf2", bound_over_urms=bound, exit_code=code)
            if whole.is_file() and whole.stat().st_size > 4096 and code in (0, 3, 4):
                rms, largest, ratio = centre_error(whole, lambda k: cutouts[k], header["u_rms"])
                row.update(compression_ratio=f"{ratio:.3f}", centre_rms_error_relative=f"{rms:.6e}", centre_largest_error_over_urms=f"{largest:.5e}")
            append(log, row)
            if whole.is_file():
                whole.unlink()
        windowed = [project(d * w) for d in cutouts]
        measure_cutout(log, base, windowed, header["u_rms"], header["dt"], f"{name}_cutout")


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "control":
        control()
    elif action == "jhtdb":
        import cutout_jhtdb

        sys.exit(cutout_jhtdb.run(sys.modules[__name__]))
    else:
        print(__doc__)
        sys.exit(2)
