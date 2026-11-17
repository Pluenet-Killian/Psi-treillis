# SPDX-License-Identifier: Apache-2.0
"""psitreillis - Python interface of the Psi-treillis codec.

Psi-treillis compresses time sequences of incompressible, periodic velocity fields on a cubic grid. The decoded field
is divergence-free by construction, and it can be sampled at arbitrary points straight from the compressed file.

    import numpy as np
    import psitreillis as psi

    psi.compress(frames, "flow.psi", dt=0.01, tolerance=0.05)   # frames: float32, shape (F, 3, n, n, n)
    with psi.open("flow.psi") as f:
        print(f.info)
        u = f.decode(10)                     # float32, shape (3, n, n, n): components u, v, w indexed [z, y, x]
        v = f.sample(10, points)             # points: (N, 3) array of (x, y, z) in grid-index units -> (N, 3)

This module is a thin ctypes layer over the C library (include/psitreillis.h). It needs NumPy, an NVIDIA GPU and the
shared library built by the project. The library is looked for, in this order: the path in the environment variable
PSITREILLIS_LIBRARY, the folder of this module, then the build folders of a source checkout.
"""

from __future__ import annotations

import ctypes
import os
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np

__all__ = ["open", "compress", "compress_file", "Reader", "PsiError", "PsiWarning", "version"]

OK, ERROR, CLIPPED, BOUND_MISSED = 0, 1, 3, 4


class PsiError(RuntimeError):
    """A call to the library failed: invalid argument, unreadable or altered file, GPU error."""


class PsiWarning(UserWarning):
    """The file was written, but coefficients were clipped or the requested error bound is not certified."""


class _Info(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int32) for name in
                ("n", "frames", "degree", "first_frame", "key_interval", "format", "sealed", "has_bound", "bound_held")]
    _fields_ += [(name, ctypes.c_float) for name in ("dt", "u_rms", "eps", "tolerance", "max_error", "rms_error")]
    _fields_ += [("file_bytes", ctypes.c_uint64), ("raw_bytes", ctypes.c_uint64)]


class _FrameInfo(ctypes.Structure):
    _fields_ = [("is_key", ctypes.c_int32), ("bytes", ctypes.c_uint64), ("rms_error", ctypes.c_float),
                ("max_error", ctypes.c_float), ("mean", ctypes.c_float * 3)]


class _Options(ctypes.Structure):
    _fields_ = [("degree", ctypes.c_int32), ("eps", ctypes.c_float), ("tolerance", ctypes.c_float),
                ("abs_tolerance", ctypes.c_float), ("key_interval", ctypes.c_int32), ("first_frame", ctypes.c_int32),
                ("frame_count", ctypes.c_int32), ("u_rms", ctypes.c_float), ("seal", ctypes.c_int32),
                ("verbose", ctypes.c_int32), ("reserved", ctypes.c_int32 * 8)]


def _library_name() -> str:
    if sys.platform == "win32":
        return "psitreillis.dll"
    return "libpsitreillis.dylib" if sys.platform == "darwin" else "libpsitreillis.so"


def _load() -> ctypes.CDLL:
    here = Path(__file__).resolve().parent
    candidates = []
    if os.environ.get("PSITREILLIS_LIBRARY"):
        candidates.append(Path(os.environ["PSITREILLIS_LIBRARY"]))
    candidates.append(here / _library_name())
    candidates.append(here.parent.parent / "build" / _library_name())
    if sys.platform == "win32":
        # The CUDA runtime and cuFFT are not on the DLL search path of Python.
        cuda = os.environ.get("CUDA_PATH")
        for sub in ("bin", "bin/x64") if cuda else ():
            if (Path(cuda) / sub).is_dir():
                os.add_dll_directory(str(Path(cuda) / sub))
    for path in candidates:
        if path.is_file():
            if sys.platform == "win32":
                os.add_dll_directory(str(path.parent))
            return ctypes.CDLL(str(path))
    raise ImportError("psitreillis: shared library not found; build it (tools/build.ps1) or set PSITREILLIS_LIBRARY. "
                      "Looked in: " + ", ".join(str(p) for p in candidates))


_lib = _load()
_float_p = ctypes.POINTER(ctypes.c_float)
_lib.psit_version.restype = ctypes.c_char_p
_lib.psit_last_error.restype = ctypes.c_char_p
_lib.psit_open.argtypes = [ctypes.c_char_p, ctypes.POINTER(ctypes.c_void_p)]
_lib.psit_close.argtypes = [ctypes.c_void_p]
_lib.psit_close.restype = None
_lib.psit_get_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(_Info)]
_lib.psit_get_frame_info.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(_FrameInfo)]
_lib.psit_decode.argtypes = [ctypes.c_void_p, ctypes.c_int, _float_p]
_lib.psit_sample.argtypes = [ctypes.c_void_p, ctypes.c_int, _float_p, _float_p, _float_p, ctypes.c_size_t, _float_p, _float_p, _float_p]
_lib.psit_default_options.argtypes = [ctypes.POINTER(_Options)]
_lib.psit_default_options.restype = None
_lib.psit_compress_memory.argtypes = [_float_p, ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_char_p, ctypes.POINTER(_Options)]
_lib.psit_compress_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(_Options)]


def version() -> str:
    """Version of the C library."""
    return _lib.psit_version().decode()


def _message() -> str:
    return _lib.psit_last_error().decode("utf-8", "replace")


def _check(code: int) -> None:
    if code != OK:
        raise PsiError(_message())


def _path(path) -> bytes:
    return os.fsencode(os.fspath(path))


def _pointer(array: np.ndarray):
    return array.ctypes.data_as(_float_p)


def _namespace(structure: ctypes.Structure) -> SimpleNamespace:
    values = {}
    for name, _ in structure._fields_:
        value = getattr(structure, name)
        values[name] = tuple(value) if isinstance(value, ctypes.Array) else value
    return SimpleNamespace(**values)


class Reader:
    """A .psi file opened for reading. Use it as a context manager, or call close()."""

    def __init__(self, path):
        handle = ctypes.c_void_p()
        _check(_lib.psit_open(_path(path), ctypes.byref(handle)))
        self._handle = handle
        raw = _Info()
        _check(_lib.psit_get_info(self._handle, ctypes.byref(raw)))
        info = _namespace(raw)
        for name in ("sealed", "has_bound", "bound_held"):
            setattr(info, name, bool(getattr(info, name)))
        info.ratio = info.raw_bytes / info.file_bytes
        #: n, frames, degree, dt, u_rms, eps, tolerance, max_error, rms_error, has_bound, bound_held, ratio...
        #: Errors are those measured by the encoder; tolerance and max_error are fractions of u_rms.
        self.info = info
        self.n = info.n
        self.frames = info.frames

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.psit_close(self._handle)
            self._handle = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self) -> int:
        return self.frames

    def __getitem__(self, frame: int) -> np.ndarray:
        return self.decode(frame)

    def _require_open(self):
        if not self._handle:
            raise PsiError("the file is closed")

    def frame_info(self, frame: int) -> SimpleNamespace:
        """is_key, bytes, rms_error, max_error and mean velocity of one frame."""
        self._require_open()
        raw = _FrameInfo()
        _check(_lib.psit_get_frame_info(self._handle, int(frame), ctypes.byref(raw)))
        info = _namespace(raw)
        info.is_key = bool(info.is_key)
        return info

    def decode(self, frame: int, out: np.ndarray | None = None) -> np.ndarray:
        """Decodes one frame to the grid: float32 array of shape (3, n, n, n), components u, v, w indexed [z, y, x].

        Frames are cheapest to read in increasing order; a jump restarts from the previous key frame."""
        self._require_open()
        shape = (3, self.n, self.n, self.n)
        if out is None:
            out = np.empty(shape, dtype=np.float32)
        elif out.shape != shape or out.dtype != np.float32 or not out.flags.c_contiguous:
            raise PsiError(f"out must be a C-contiguous float32 array of shape {shape}")
        _check(_lib.psit_decode(self._handle, int(frame), _pointer(out)))
        return out

    def sample(self, frame: int, points) -> np.ndarray:
        """Samples the velocity of one frame at arbitrary points, without decoding the grid.

        points: array of shape (N, 3) holding (x, y, z) in grid-index units: the point (i, j, k) is the centre of voxel
        x = i, y = j, z = k, and coordinates are periodic with period n. Returns a float32 array of shape (N, 3)."""
        self._require_open()
        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise PsiError("points must have the shape (N, 3)")
        xyz = [np.ascontiguousarray(points[:, c]) for c in range(3)]
        uvw = [np.empty(points.shape[0], dtype=np.float32) for _ in range(3)]
        _check(_lib.psit_sample(self._handle, int(frame), *(_pointer(a) for a in xyz), points.shape[0], *(_pointer(a) for a in uvw)))
        return np.stack(uvw, axis=1)


def open(path) -> Reader:
    """Opens a .psi file."""
    return Reader(path)


def _options(degree, eps, tolerance, abs_tolerance, key_interval, first_frame, frame_count, u_rms, seal, verbose) -> _Options:
    options = _Options()
    _lib.psit_default_options(ctypes.byref(options))
    options.degree = int(degree)
    options.eps = float(eps)
    options.tolerance = float(tolerance or 0.0)
    options.abs_tolerance = float(abs_tolerance or 0.0)
    options.key_interval = int(key_interval)
    options.first_frame = int(first_frame)
    options.frame_count = int(frame_count or 0)
    options.u_rms = float(u_rms or 0.0)
    options.seal = 1 if seal else 0
    options.verbose = 1 if verbose else 0
    return options


def _finish(code: int, path) -> SimpleNamespace:
    if code == ERROR:
        raise PsiError(_message())
    if code != OK:
        warnings.warn(_message(), PsiWarning, stacklevel=3)
    with Reader(path) as reader:
        return reader.info


def compress(frames, path, *, dt: float = 1.0, degree: int = 3, eps: float = 0.005, tolerance: float | None = None,
             abs_tolerance: float | None = None, key_interval: int = 32, first_frame: int = 0, frame_count: int | None = None,
             u_rms: float | None = None, seal: bool = True, verbose: bool = False) -> SimpleNamespace:
    """Compresses a sequence of velocity fields into a .psi file and returns the information of the file written.

    frames: float32 array of shape (F, 3, n, n, n): F >= 5 frames, components u, v, w indexed [z, y, x], on a cubic
    periodic grid with n a power of two (>= 64). The field must be divergence-free: the codec keeps only its
    solenoidal part. An np.memmap is read in place.

    eps: target RMS quantisation error, as a fraction of u_rms (the RMS of one velocity component).
    tolerance: certified bound on the largest error of a component at voxel centres, as a fraction of u_rms;
    abs_tolerance: the same bound in the units of the data. With the default eps, bounds down to about 3 % of u_rms were
    certified on the project's datasets; a tighter bound needs a smaller eps.

    Raises PsiError on failure. Emits a PsiWarning if the file is written but the bound is not certified, or if
    coefficients were clipped."""
    data = np.ascontiguousarray(frames, dtype=np.float32)
    if data.ndim != 5 or data.shape[1] != 3 or not (data.shape[2] == data.shape[3] == data.shape[4]):
        raise PsiError(f"frames must have the shape (F, 3, n, n, n), got {data.shape}")
    options = _options(degree, eps, tolerance, abs_tolerance, key_interval, first_frame, frame_count, u_rms, seal, verbose)
    code = _lib.psit_compress_memory(_pointer(data), data.shape[2], data.shape[0], float(dt), _path(path), ctypes.byref(options))
    return _finish(code, path)


def compress_file(seq_path, path, *, degree: int = 3, eps: float = 0.005, tolerance: float | None = None,
                  abs_tolerance: float | None = None, key_interval: int = 32, first_frame: int = 0, frame_count: int | None = None,
                  u_rms: float | None = None, seal: bool = True, verbose: bool = False) -> SimpleNamespace:
    """Same as compress(), from a sequence file in the project's own format (.seq)."""
    options = _options(degree, eps, tolerance, abs_tolerance, key_interval, first_frame, frame_count, u_rms, seal, verbose)
    code = _lib.psit_compress_file(_path(seq_path), _path(path), ctypes.byref(options))
    return _finish(code, path)
