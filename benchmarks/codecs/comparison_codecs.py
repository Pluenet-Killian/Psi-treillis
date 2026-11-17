# SPDX-License-Identifier: Apache-2.0
"""Calls to the comparison compressors and to Psi-treillis, shared by the measurement scripts of benchmarks/.

Every compressor is called on one velocity component at a time, with an absolute bound on the largest error, through
the binaries built by build_codecs.ps1 in build/codecs/:

    sz3     sz3 -f -i IN -z CMP -o DEC -3 n n n -M ABS bound          (command-line tool, default configuration)
    sz3_4d  sz3 -f -i IN -z CMP -o DEC -4 n n n frames -M ABS bound   (a block of frames, time as fourth dimension)
    sperr   sperr_comp_3d(..., chunks of 256^3, mode 3 = point-wise error, quality = bound, 1 thread)   (C interface)
    zfp     zfp -f -3 n n n -i IN -z CMP -o DEC -a bound              (fixed-accuracy mode)
    cuszp   cuSZp -i IN -t f32 -m outlier -d 3 n n n -eb abs bound -x CMP -o DEC   (its example program, GPU)

No option other than the bound is tuned. The errors are never taken from what a compressor reports: the callers
compare the decoded arrays with the original in NumPy.

Psi-treillis is called through its public Python module (python/psitreillis), with the same absolute bound.
"""

import ctypes
import os
import struct
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
BUILD = ROOT / "build"
CODECS = BUILD / "codecs"
DATA = ROOT / "data"
EXE = ".exe" if sys.platform == "win32" else ""
SEQ_HEADER_BYTES = 72


def psitreillis():
    """The public Python module of the library, bound to the library of build/."""
    library = BUILD / ("psitreillis.dll" if sys.platform == "win32" else "libpsitreillis.so")
    os.environ.setdefault("PSITREILLIS_LIBRARY", str(library))
    if str(ROOT / "python") not in sys.path:
        sys.path.insert(0, str(ROOT / "python"))
    import psitreillis as module
    return module


def work_folder(name):
    """A scratch folder of its own for this process: two campaigns can run at the same time."""
    folder = BUILD / "work" / name / str(os.getpid())
    folder.mkdir(parents=True, exist_ok=True)
    return folder


# ---- Sequences (.seq, see src/sequence.h) ---------------------------------------------------------------------------

def read_header(seq):
    """n, frames, dt, nu, u_rms, kmax_eta, re_lambda of a sequence."""
    with open(seq, "rb") as f:
        _, n, frames, dt, nu, u_rms, kmax_eta, re_lambda = struct.unpack("<8sii5f", f.read(36))
    return dict(n=n, frames=frames, dt=dt, nu=nu, u_rms=u_rms, kmax_eta=kmax_eta, re_lambda=re_lambda)


def read_frame(seq, k, n=None):
    """Frame k of a sequence: float32 array (3, n, n, n), components u, v, w indexed [z, y, x]."""
    if n is None:
        n = read_header(seq)["n"]
    with open(seq, "rb") as f:
        f.seek(SEQ_HEADER_BYTES + k * 3 * n ** 3 * 4)
        return np.fromfile(f, dtype=np.float32, count=3 * n ** 3).reshape(3, n, n, n)


def read_component_block(seq, first, count, component, n):
    """Component `component` of `count` consecutive frames: float32 array (count, n, n, n)."""
    block = np.empty((count, n, n, n), dtype=np.float32)
    with open(seq, "rb") as f:
        for j in range(count):
            f.seek(SEQ_HEADER_BYTES + ((first + j) * 3 + component) * n ** 3 * 4)
            block[j] = np.fromfile(f, dtype=np.float32, count=n ** 3).reshape(n, n, n)
    return block


def eta_in_voxels(header):
    """Kolmogorov scale in voxels, from kmax * eta of the header (the solver truncates at kmax = n / 3)."""
    return header["kmax_eta"] * 3.0 / (2.0 * np.pi)


# ---- Command-line compressors ---------------------------------------------------------------------------------------

def _run(command):
    p = subprocess.run([str(c) for c in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout.decode("utf-8", "replace")


def _command(kind, src, cmp_, dec, n, setting):
    if kind == "sz3":
        return [CODECS / "sz3" / "tools" / "sz3" / f"sz3{EXE}", "-f", "-i", src, "-z", cmp_, "-o", dec, "-3", n, n, n, "-M", "ABS", repr(setting)]
    if kind == "zfp":
        return [CODECS / "zfp" / "bin" / f"zfpcmd{EXE}", "-f", "-3", n, n, n, "-i", src, "-z", cmp_, "-o", dec, "-a", repr(setting)]
    if kind == "zfp_rate":
        return [CODECS / "zfp" / "bin" / f"zfpcmd{EXE}", "-f", "-3", n, n, n, "-i", src, "-z", cmp_, "-o", dec, "-r", repr(setting)]
    if kind == "cuszp":
        return [CODECS / "cuszp" / f"cuSZp_cli{EXE}", "-i", src, "-t", "f32", "-m", "outlier", "-d", "3", n, n, n, "-eb", "abs", repr(setting), "-x", cmp_, "-o", dec]
    raise ValueError(kind)


def _cli_component(kind, plane, setting, folder, tag):
    n = plane.shape[0]
    src, cmp_, dec = folder / f"{tag}.f32", folder / f"{tag}.{kind}.cmp", folder / f"{tag}.{kind}.dec"
    plane.tofile(src)
    for p in (cmp_, dec):
        p.unlink(missing_ok=True)
    code, out = _run(_command(kind, src, cmp_, dec, n, setting))
    if code != 0 or not dec.exists() or not cmp_.exists():
        raise RuntimeError(f"{kind} at {setting}, {tag}: exit code {code}\n{out[-400:]}")
    result = (cmp_.stat().st_size, np.fromfile(dec, dtype=np.float32).reshape(n, n, n))
    for p in (src, cmp_, dec):
        p.unlink()
    return result


_sperr = None


def _sperr_component(plane, bound):
    global _sperr
    if _sperr is None:
        lib = ctypes.CDLL(str(CODECS / "sperr" / "sperr_capi.dll"))
        st, vp = ctypes.c_size_t, ctypes.c_void_p
        lib.sperr_comp_3d.argtypes = [vp, ctypes.c_int, st, st, st, st, st, st, ctypes.c_int, ctypes.c_double, st, ctypes.POINTER(vp), ctypes.POINTER(st)]
        lib.sperr_decomp_3d.argtypes = [vp, st, ctypes.c_int, st, ctypes.POINTER(st), ctypes.POINTER(st), ctypes.POINTER(st), ctypes.POINTER(vp)]
        free = ctypes.CDLL("ucrtbase").free
        free.argtypes = [vp]
        _sperr = (lib, free)
    lib, free = _sperr
    n = plane.shape[0]
    src = np.ascontiguousarray(plane, dtype=np.float32)
    dst, size = ctypes.c_void_p(None), ctypes.c_size_t(0)
    code = lib.sperr_comp_3d(src.ctypes.data, 1, n, n, n, 256, 256, 256, 3, float(bound), 1, ctypes.byref(dst), ctypes.byref(size))
    if code != 0:
        raise RuntimeError(f"sperr_comp_3d: code {code}")
    out, dx, dy, dz = ctypes.c_void_p(None), ctypes.c_size_t(0), ctypes.c_size_t(0), ctypes.c_size_t(0)
    code = lib.sperr_decomp_3d(dst, size, 1, 1, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz), ctypes.byref(out))
    if code != 0:
        raise RuntimeError(f"sperr_decomp_3d: code {code}")
    decoded = np.ctypeslib.as_array(ctypes.cast(out, ctypes.POINTER(ctypes.c_float)), shape=(n, n, n)).copy()
    free(dst)
    free(out)
    return size.value, decoded


def compress_frame(kind, u, setting, folder):
    """One frame through one compressor, component by component.

    kind: sz3, sperr, zfp, cuszp (setting = absolute bound on the largest error) or zfp_rate (setting = bits per value).
    u: float32 (3, n, n, n). Returns (compressed bytes of the three components, decoded float32 (3, n, n, n))."""
    with ThreadPoolExecutor(3) as pool:
        if kind == "sperr":
            parts = list(pool.map(lambda c: _sperr_component(u[c], setting), range(3)))
        else:
            parts = list(pool.map(lambda c: _cli_component(kind, u[c], setting, folder, f"c{c}"), range(3)))
    return sum(p[0] for p in parts), np.stack([p[1] for p in parts])


def sz3_4d_component(block, bound, folder, keep=None):
    """A block of frames of one component through SZ3 in 4D (time as the fourth dimension).

    block: float32 (count, n, n, n). Returns (compressed bytes, decoded block as a read-only memory map; the caller
    deletes `folder / "block.dec"` when done). keep: path where the compressed file is left, if wanted."""
    count, n = block.shape[0], block.shape[1]
    src, cmp_, dec = folder / "block.f32", folder / "block.sz4", folder / "block.dec"
    block.tofile(src)
    for p in (cmp_, dec):
        p.unlink(missing_ok=True)
    code, out = _run([CODECS / "sz3" / "tools" / "sz3" / f"sz3{EXE}", "-f", "-i", src, "-z", cmp_, "-o", dec, "-4", n, n, n, count, "-M", "ABS", repr(bound)])
    if code != 0 or not dec.exists():
        raise RuntimeError(f"sz3 in 4D at {bound}: exit code {code}\n{out[-400:]}")
    size = cmp_.stat().st_size
    src.unlink()
    if keep is None:
        cmp_.unlink()
    else:
        os.replace(cmp_, keep)
    return size, np.memmap(dec, dtype=np.float32, mode="r", shape=(count, n, n, n))


# ---- Errors, measured here for every codec in the same way ----------------------------------------------------------

class ErrorSum:
    """Accumulates the error of decoded frames against the originals, in double precision."""

    def __init__(self):
        self.square_error = 0.0
        self.square_field = 0.0
        self.largest = 0.0
        self.low, self.high = np.inf, -np.inf
        self.values = 0

    def add(self, original, decoded):
        """original, decoded: arrays of the same shape (any number of components and frames)."""
        e = decoded.astype(np.float64) - original.astype(np.float64)
        self.square_error += float((e * e).sum())
        self.square_field += float((original.astype(np.float64) ** 2).sum())
        self.largest = max(self.largest, float(np.abs(e).max()))
        self.low, self.high = min(self.low, float(original.min())), max(self.high, float(original.max()))
        self.values += original.size

    def relative_rms(self):
        """RMS of the error over the RMS of the field, all components together."""
        return float(np.sqrt(self.square_error / self.square_field))

    def psnr_db(self):
        """Peak signal-to-noise ratio with the value range of the original as the peak: 20 log10(range / RMSE)."""
        return float(20.0 * np.log10((self.high - self.low) / np.sqrt(self.square_error / self.values)))
