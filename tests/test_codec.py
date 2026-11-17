# SPDX-License-Identifier: Apache-2.0
"""Regression suite of the pack / play chain of Psi-treillis (tools/psi_stream.cu), on a turbulence sequence at 64^3.

    python tests/test_codec.py                        runs every case
    python tests/test_codec.py --build build          build folder holding dns and psi_stream (default: build)
    python tests/test_codec.py --only "d3.*"          runs only the cases whose name matches the pattern (wildcards
                                                      * and ?); several patterns: --only "guard.*" "*.bound_*"
    python tests/test_codec.py --update               regenerates tests/fingerprints.json (see below)
    python tests/test_codec.py --keep-alterations     keeps in <build>/tests/cases/random/ every file of the case
                                                      guard.random_alteration, to read them again with another binary

The suite compiles nothing. It needs an NVIDIA GPU and a build folder holding dns and psi_stream, built with
`tools/build.ps1 -Tools` on Windows or with CMake and -DPSIT_BUILD_TOOLS=ON elsewhere; it says so and exits with code
2 if they are missing. A relative --build is taken from the root of the repository. Python 3.8 or later, standard
library only.

Steps:
  1. produces, if it is missing, the sequence <build>/tests/turb64.seq with dns and a fixed seed;
  2. runs the cases: one line per case, then a total. The files written are in <build>/tests/cases/, the output of
     each command in <build>/tests/logs/.
One line per case:
  OK       the case passes;
  FAIL     the case fails, with the reason and the log to read;
  KNOWN    the case fails on a known and uncorrected defect of the code (described in tests/README.md): it is shown at
           every run, but does not count as a failure;
  UPDATED  with --update, the fingerprint recorded by the case.
Exit code: 0 if no case fails, 1 if one fails, 2 if the prerequisites are missing (binaries, sequence, or an option
that selects nothing).

REFERENCE FINGERPRINTS (tests/fingerprints.json): READ THIS BEFORE CONCLUDING THAT THERE IS A REGRESSION.
The "fingerprint" cases compare the SHA-256 of the sequence and of a few .psi files, and the fingerprint of the final
positions read back by play, with values recorded on ONE toolchain: version of CUDA and of cuFFT, host compiler,
architecture of the GPU (it is described in the file, under "toolchain"). The sequence comes out of a chaotic
simulation in floating point: another version of cuFFT, another compiler or another GPU change its bytes, and with
them every encoded file, without there being any defect.
On another machine, or after a change of toolchain:
  a. first check the integrity: every case other than "fingerprint" must pass (play reads each file back and checks,
     frame by frame, that its state is that of the encoder);
  b. then regenerate the references: python tests/test_codec.py --update. The script refuses to do so if another case
     fails, or if --only restricts the suite.
On the same toolchain, a fingerprint that changes is a regression of the encoder or an intended change of the format:
in the second case, regenerate the file and say so in the commit.

The GPU may be shared: the suite checks only bytes, fingerprints, exit codes and messages. It reads no run time.
"""

from __future__ import annotations

import argparse
import datetime
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import signal
import struct
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FINGERPRINTS_FILE = Path(__file__).resolve().parent / "fingerprints.json"
EXE = ".exe" if sys.platform == "win32" else ""

# ---------------------------------------------------------------------------------------------------------------------
# Settings of the cases
# ---------------------------------------------------------------------------------------------------------------------

# Sequence: 40 frames at 64^3 (120 MiB). 64^3 is accepted everywhere: dns does not constrain the resolution, pack
# requires a power of two and at least 8 nodes at the coarsest level (64 / 2 >> 2 = 8 with three levels, the default),
# the header of a .psi file a power of two of at least 16, the fine level at least 32.
# Viscosity 0.02: kmax * eta = 1.68 and Re_lambda = 32 (measured on 15 November 2026), that is a flow as well resolved
# as turb256.seq (kmax * eta = 1.68). With the default viscosity of dns (0.0035), meant for
# 256^3, the sequence would be under-resolved at 64^3.
SEQUENCE_NAME = "turb64.seq"
SEQUENCE_FRAMES = 40
DNS_OPTIONS = ["--n", "64", "--nu", "0.02", "--spinup", "12", "--frames", str(SEQUENCE_FRAMES), "--steps-per-frame", "4", "--seed", "42"]

# Encoded segment: frames 3 to 38, that is 36 frames. With one key frame every 32 frames (the default), the file
# carries two keys (ranks 0 and 32), two innovations "after a key" (ranks 1 and 33) and 32 ordinary innovations; the
# probe, placed on the last frame, is therefore read back after the second key. A non-zero start exercises first_frame.
FIRST = 3
COUNT = 36
SEGMENT = ["--start", str(FIRST), "--frames", str(COUNT)]

# Tolerances of the fine level, as a fraction of u'. Trials of 15 November 2026 on this sequence (tests/README.md):
#  - without a fine level, the largest error is 0.117 u' in degree 2 and 0.091 u' in degree 3;
#  - with the defaults of pack, the bound is held down to 0.038 u' (degree 2) and 0.034 u' (degree 3). Lower, pack does
#    not exit with code 4 but with code 1: at 64^3 the tolerance then asks to refine more than a quarter of the bricks,
#    and the capacity guard of the fine level stops the encoding (case guard.fine_level_capacity);
#  - bound held: 0.05 (degree 2) and 0.04 (degree 3). The fine level really works there (2.3 % and 3.7 % of the nodes
#    retained, 6 and 4 marking passes) while staying far from the capacity limit;
#  - bound missed (code 4): the same tolerance, with a fine level quantised too coarsely for it (--fine-eps 0.02, the
#    step that pack would choose for a tolerance of 0.17). The tolerance is then below the floor of the fine level:
#    36 frames out of 36 above the threshold, cause "floor";
#  - cap on the passes (code 4 too): the same tolerance with --fine-passes 1, cause "cap".
TOLERANCES = {2: "0.05", 3: "0.04"}
COARSE_FINE_EPS = "0.02"
TOLERANCE_BEYOND_CAPACITY = "0.01"
# Quantisation step of the "exceptions" case. With the default (--eps 0.005), this sequence produces no exception: the
# exception list of the format would never be written nor read back. With a finer step, without clipping (code 0):
# degree 2 at 0.001: 69 innovation escapes and 52,805 overflows of the fine level; degree 3 at 0.002: 176 and 219
# (at 0.001, degree 3 clips the lattice and exits with code 3).
EXCEPTIONS_EPS = {2: "0.001", 3: "0.002"}
PARTICLES = "1000000"         # play --particles ... --sort: the fingerprint of the final positions depends on it
RUN_TIMEOUT = 180             # seconds granted to a command before declaring it hung
GUARD_TIMEOUT = 60            # the same, for an expected refusal
DNS_TIMEOUT = 600             # the same, for the production of the sequence
# Case guard.random_alteration: seed of the draws, altered files per source file (four sources, hence 240 files),
# reads launched at the same time (each one takes a CUDA context and a few tens of MiB at 64^3).
ALTERATION_SEED = 20261115
ALTERATIONS_PER_FILE = 60
SIMULTANEOUS_READS = 6


def f32(value) -> float:
    """The value rounded to float32, as the programs and the files hold it."""
    return struct.unpack("<f", struct.pack("<f", float(value)))[0]


MARGIN = f32(1e-4)            # default of pack --tolerance-margin


# ---------------------------------------------------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------------------------------------------------

class Failure(Exception):
    """Reason why a case fails."""


class Suite:
    """What the cases share: folders, options, reference fingerprints, commands already run."""
    build = cases_dir = logs_dir = sequence = None
    update = False
    keep_alterations = False
    reference = None          # content of tests/fingerprints.json, or None
    context_differences = []  # how the toolchain or the settings differ from those of the reference
    recorded = {}             # fingerprints of this run: key -> value
    note = ""                 # with --update, what a case recorded
    detail = ""               # what a successful case has to add on its "OK" line
    done = {}                 # commands of the cases, run only once each


S = Suite()


def require(condition, message):
    if not condition:
        raise Failure(message)


def shown(path) -> str:
    """Path as the suite passes it to the programs and prints it: relative to the root of the repository if inside."""
    try:
        relative = os.path.relpath(str(path), str(ROOT))
    except ValueError:        # another drive, on Windows
        return str(path)
    return str(path) if relative.startswith("..") else relative


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data) -> str:
    return hashlib.sha256(bytes(data)).hexdigest()


def short(fingerprint: str) -> str:
    return fingerprint[:16] + "..." if len(fingerprint) > 16 else fingerprint


def readable_code(code) -> str:
    if code is None:
        return "none"
    if code < 0:
        return f"signal {-code} (crash)"
    if code > 255:
        return f"0x{code & 0xFFFFFFFF:08X} (crash)"
    return str(code)


def found(pattern: str, text: str, multiline: bool = False):
    """Search of a message of the programs. Case is ignored, as the PowerShell suite this one was ported from did."""
    return re.search(pattern, text, re.IGNORECASE | (re.MULTILINE if multiline else 0))


class Result:
    """Result of a command: exit code (None if it hung), both outputs decoded as UTF-8, the encoding of the sources."""

    def __init__(self, code, output, error, hung, timeout, log):
        self.code, self.output, self.error, self.hung, self.timeout, self.log = code, output, error, hung, timeout, log
        self.file = self.probe = None             # set by pack()
        self.differing = self.positions = None    # set by with_integrity()


class Task:
    """A command to run: name of its log, program of the build folder, arguments, timeout in seconds."""

    def __init__(self, log: str, program: str, arguments, timeout: int):
        self.log, self.program, self.arguments, self.timeout = log, program, [str(a) for a in arguments], timeout


def kill_tree(process):
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(process.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            os.killpg(process.pid, signal.SIGKILL)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        process.kill()
    except OSError:
        pass


def run_task(task: Task) -> Result:
    """Runs a binary of the build folder from the root of the repository, with a timeout, and waits for its end. A
    command that does not end in time is stopped (code None). Everything is written to a log."""
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {"start_new_session": True}
    process = subprocess.Popen([str(S.build / (task.program + EXE))] + task.arguments, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options)
    hung = False
    try:
        out, err = process.communicate(timeout=task.timeout)
    except subprocess.TimeoutExpired:
        hung = True
        kill_tree(process)
        out, err = process.communicate()
    output, error = out.decode("utf-8", errors="replace"), err.decode("utf-8", errors="replace")
    code = None if hung else process.returncode
    path = S.logs_dir / f"{task.log}.log"
    end = f"hung: stopped after {task.timeout} s" if hung else f"code {readable_code(code)}"
    text = f"$ {task.program}{EXE} {' '.join(task.arguments)}\n{output}\n--- error output ---\n{error}\n--- {end} ---\n"
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))
    return Result(code, output, error, hung, task.timeout, shown(path))


def run(log: str, program: str, arguments, timeout: int = RUN_TIMEOUT) -> Result:
    return run_task(Task(log, program, arguments, timeout))


def run_batch(tasks, simultaneous: int):
    """Runs several commands, at most `simultaneous` at a time, each with its own timeout. Results in task order."""
    with ThreadPoolExecutor(max_workers=simultaneous) as pool:
        return list(pool.map(run_task, tasks))


def reason(r: Result) -> str:
    """First line of the error output, to say why a command failed."""
    lines = [line.strip() for line in re.split(r"\r?\n", r.error) if line.strip()]
    return "; " + lines[0] if lines else "; no message"


def require_code(r: Result, expected: int):
    if r.hung:
        raise Failure(f"hang: command stopped after {r.timeout} s (log {r.log})")
    if r.code != expected:
        raise Failure(f"exit code {readable_code(r.code)} instead of {expected}{reason(r)} (log {r.log})")


def require_refusal(r: Result, text: str, what: str):
    """Expected refusal of a guard: code 1 and a message "Failure: ..." that contains the given text; neither a crash
    nor a hang."""
    if r.hung:
        raise Failure(f"{what}: hang, command stopped after {r.timeout} s (log {r.log})")
    if r.code != 1:
        raise Failure(f"{what}: exit code {readable_code(r.code)} instead of 1{reason(r)} (log {r.log})")
    if not found("Failure:", r.error) or text not in r.error:
        raise Failure(f'{what}: expected message "{text}" missing{reason(r)} (log {r.log})')


class Stream:
    """Header, header of the fine level, frame table and seal of a .psi file, read without a GPU (StreamHeader,
    FineInfo, FrameRecord and StreamSeal of src/codec_stream.cuh). Header of 208 bytes: magic (8), then n, degree, mf,
    levels, fb, key_interval, n_frames, first_frame (int32, offsets 8 to 36), index_offset (uint64) at offset 136.
    FineInfo (64 bytes, at offset 208 of a PSISTRM3 file): tolerance, eps, delta, max_error (float32), max_bricks,
    max_nodes, brick, frames_over (32 bits), then the arithmetic margin (float32, offset 240). Table: one record of 48
    bytes per frame, offset, bytes, hash (uint64), type (int32, offset 24), field_err (offset 40) and field_max (offset
    44, largest error relative to u'). Seal (48 bytes after the table, absent with pack --no-seal): "PSISEAL1", three
    fingerprints (header, header of the fine level, table), two zero words."""

    def __init__(self, path):
        data = Path(path).read_bytes()
        name = shown(path)
        require(len(data) >= 208, f"{name}: file shorter than a header")
        self.size = len(data)
        self.magic = data[:8].decode("ascii", errors="replace")
        self.n, self.degree, self.mf, self.levels, self.fb, self.key_interval, self.n_frames, self.first_frame = struct.unpack_from("<8i", data, 8)
        index = struct.unpack_from("<Q", data, 136)[0]
        self.fine = None
        if self.magic == "PSISTRM3":
            self.fine = FineHeader(*struct.unpack_from("<4f2IiIf", data, 208))
        # The file ends with the table (48 bytes per frame), or with the seal of 48 bytes that follows it.
        require(208 <= index < len(data), f"{name}: frame table missing or misplaced")
        count = self.n_frames
        rest = len(data) - index - 48 * count
        require(rest in (0, 48), f"{name}: {rest} byte(s) after the table of {count} frames (0 expected without a seal, 48 with one)")
        self.table_offset = index
        self.table_end = index + 48 * count
        self.seal = None
        if rest == 48:
            self.seal = Seal(data[self.table_end:self.table_end + 8].decode("ascii", errors="replace"), *struct.unpack_from("<5Q", data, self.table_end + 8))
        self.frames = [FrameRecord(*struct.unpack_from("<3Qi", data, index + 48 * j), *struct.unpack_from("<2f", data, index + 48 * j + 40))
                       for j in range(max(count, 0))]
        # Largest error of the table. A value that is not a number never wins, as in the suite this one was ported from.
        errors = [f.max_error for f in self.frames if not math.isnan(f.max_error)]
        self.max_error = max(errors) if errors else float("nan") if self.frames else 0.0


class FineHeader:
    def __init__(self, tolerance, eps, delta, max_error, max_bricks, max_nodes, brick, frames_over, margin):
        self.tolerance, self.eps, self.delta, self.max_error = tolerance, eps, delta, max_error
        self.max_bricks, self.max_nodes, self.brick, self.frames_over, self.margin = max_bricks, max_nodes, brick, frames_over, margin


class Seal:
    def __init__(self, magic, header, fine, table, reserved0, reserved1):
        self.magic, self.header, self.fine, self.table, self.reserved0, self.reserved1 = magic, header, fine, table, reserved0, reserved1


class FrameRecord:
    def __init__(self, offset, size, state_hash, kind, field_error, max_error):
        self.offset, self.bytes, self.hash, self.type, self.field_error, self.max_error = offset, size, state_hash, kind, field_error, max_error


# ---------------------------------------------------------------------------------------------------------------------
# Sequence
# ---------------------------------------------------------------------------------------------------------------------

def sequence_valid() -> bool:
    """The sequence is valid if it has the expected size (header of 72 bytes, src/sequence.h) and a complete header:
    dns writes u' in it only at the end of its run."""
    path = S.sequence
    if not path.is_file() or path.stat().st_size != 72 + SEQUENCE_FRAMES * 3 * 64 * 64 * 64 * 4:
        return False
    with open(path, "rb") as stream:
        header = stream.read(28)
    if len(header) < 28:
        return False
    n, frames = struct.unpack_from("<2i", header, 8)
    u_rms = struct.unpack_from("<f", header, 24)[0]
    return n == 64 and frames == SEQUENCE_FRAMES and u_rms > 0


# ---------------------------------------------------------------------------------------------------------------------
# Toolchain and reference fingerprints
# ---------------------------------------------------------------------------------------------------------------------

def windows_file_version(path: str) -> str:
    """The FileVersion string of the version resource of a DLL."""
    import ctypes
    from ctypes import wintypes
    api = ctypes.WinDLL("version")
    api.GetFileVersionInfoSizeW.argtypes = [wintypes.LPCWSTR, ctypes.c_void_p]
    api.GetFileVersionInfoSizeW.restype = wintypes.DWORD
    api.GetFileVersionInfoW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, ctypes.c_void_p]
    api.GetFileVersionInfoW.restype = wintypes.BOOL
    api.VerQueryValueW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wintypes.UINT)]
    api.VerQueryValueW.restype = wintypes.BOOL
    size = api.GetFileVersionInfoSizeW(path, None)
    if not size:
        return ""
    block = ctypes.create_string_buffer(size)
    if not api.GetFileVersionInfoW(path, 0, size, block):
        return ""
    pointer, length = ctypes.c_void_p(), wintypes.UINT()
    if not api.VerQueryValueW(block, "\\VarFileInfo\\Translation", ctypes.byref(pointer), ctypes.byref(length)) or length.value < 4:
        return ""
    language, codepage = struct.unpack("<2H", ctypes.string_at(pointer.value, 4))
    if not api.VerQueryValueW(block, f"\\StringFileInfo\\{language:04x}{codepage:04x}\\FileVersion", ctypes.byref(pointer), ctypes.byref(length)):
        return ""
    return ctypes.wstring_at(pointer.value, length.value).rstrip("\0")


def cufft_library(build: Path) -> str:
    """The cuFFT library that the binary loads: on Windows the first cufft64_N.dll of the build folder, then of the
    PATH, with its file version; elsewhere the file that the dynamic loader resolves libcufft.so.N to."""
    binary = (build / ("psi_stream" + EXE)).read_bytes()
    dll = re.search(rb"cufft64_\d+\.dll", binary, re.IGNORECASE)
    if dll:
        name = dll.group(0).decode("ascii")
        for folder in [str(build)] + [d for d in os.environ.get("PATH", "").split(os.pathsep) if d]:
            candidate = os.path.join(folder, name)
            if os.path.isfile(candidate):
                version = windows_file_version(candidate) if sys.platform == "win32" else ""
                return f"{name} {version}"
        return f"{name} not found in the PATH"
    shared = re.search(rb"libcufft\.so\.\d+", binary)
    if shared:
        name = shared.group(0).decode("ascii")
        try:
            listing = subprocess.run(["ldd", str(build / "psi_stream")], capture_output=True, text=True).stdout
        except OSError:
            listing = ""
        resolved = re.search(re.escape(name) + r"\s*=>\s*(/\S+)", listing)
        return os.path.basename(os.path.realpath(resolved.group(1))) if resolved else f"{name} not resolved by the dynamic loader"
    return "unknown"


def toolchain(build: Path) -> dict:
    """What fixes the bytes produced: CUDA compiler and libraries (from what CMake recorded in the build folder), host
    compiler, cuFFT library that the binary loads, GPU and driver."""
    c = {"cuda": "unknown", "cuda_architecture": "unknown", "host_compiler": "unknown", "cufft": "unknown", "cufft_library": "unknown",
         "gpu": "unknown", "driver": "unknown"}
    recorded = sorted((build / "CMakeFiles").rglob("CMakeCUDACompiler.cmake")) if (build / "CMakeFiles").is_dir() else []
    if recorded:
        text = recorded[0].read_text(encoding="utf-8", errors="replace")
        version = found(r'set\(CMAKE_CUDA_COMPILER_VERSION "([^"]+)"\)', text)
        if version:
            c["cuda"] = f"nvcc {version.group(1)}"
        architecture = found(r'set\(CMAKE_CUDA_ARCHITECTURES_NATIVE "([^"]+)"\)', text)
        if architecture:
            c["cuda_architecture"] = architecture.group(1)
        host, host_version = found(r'set\(CMAKE_CUDA_HOST_COMPILER_ID "([^"]*)"\)', text), found(r'set\(CMAKE_CUDA_HOST_COMPILER_VERSION "([^"]+)"\)', text)
        if host and host_version:
            c["host_compiler"] = f"{host.group(1)} {host_version.group(1)}"
        toolkit = found(r'set\(CMAKE_CUDA_COMPILER_TOOLKIT_LIBRARY_ROOT "([^"]+)"\)', text)
        header = Path(toolkit.group(1)) / "include" / "cufft.h" if toolkit else None
        if header and header.is_file():
            source = header.read_text(encoding="utf-8", errors="replace")
            parts = [m.group(1) for m in (found(rf"#define CUFFT_VER_{k}\s+(\d+)", source) for k in ("MAJOR", "MINOR", "PATCH", "BUILD")) if m]
            c["cufft"] = f"cuFFT {'.'.join(parts)} (header of the build)"
    try:
        c["cufft_library"] = cufft_library(build)
    except OSError:
        pass
    try:
        listing = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"], capture_output=True, text=True).stdout
        fields = re.split(r",\s*", listing.splitlines()[0]) if listing.strip() else []
        if len(fields) >= 2:
            c["gpu"], c["driver"] = fields[0].strip(), fields[1].strip()
    except (OSError, IndexError):
        pass
    return c


# The settings of the cases that the fingerprints depend on: if they change, the references must be regenerated.
SETTINGS = (f"dns {' '.join(DNS_OPTIONS)} | pack {' '.join(SEGMENT)} | tolerances: degree 2 {TOLERANCES[2]}, degree 3 {TOLERANCES[3]} | "
            f"exceptions: --eps {EXCEPTIONS_EPS[2]} in degree 2, {EXCEPTIONS_EPS[3]} in degree 3 | play --particles {PARTICLES} --sort")


def reference_of(key: str):
    if S.reference and key in S.reference.get("fingerprints", {}):
        return str(S.reference["fingerprints"][key])
    return None


def fingerprint(key: str, value: str):
    """Compares a fingerprint with its reference; with --update, only records it (the file is rewritten at the end)."""
    S.recorded[key] = value
    old = reference_of(key)
    if S.update:
        state = "(new)" if old is None else "(unchanged)" if old == value else f"(CHANGED, was {short(old)})"
        S.note += ("; " if S.note else "") + f"{key} = {short(value)} {state}"
        return
    require(S.reference, f"tests/fingerprints.json missing: no reference for {key}")
    require(old is not None, f"no reference for {key} in tests/fingerprints.json")
    if old != value:
        if S.context_differences:
            hint = "; the toolchain or the settings are not those of the fingerprints: check the integrity, then --update"
        elif key != SEQUENCE_NAME and SEQUENCE_NAME in S.recorded and reference_of(SEQUENCE_NAME) != S.recorded[SEQUENCE_NAME]:
            hint = "; the sequence itself differs from its reference"
        else:
            hint = ""
        raise Failure(f"{key}: fingerprint {short(value)} instead of {short(old)}{hint}")


# ---------------------------------------------------------------------------------------------------------------------
# Commands of the cases, run only once each
# ---------------------------------------------------------------------------------------------------------------------

def with_integrity(r: Result) -> Result:
    """Adds to the result of a play its decoded "Integrity" line: frames whose state differs, fingerprint of the
    positions."""
    line = found(r"Integrity\s*:\s*(\d+) frame\(s\) whose state differs from that of the sender; fingerprint of the final positions ([0-9a-f]{16})", r.output)
    r.differing, r.positions = (int(line.group(1)), line.group(2)) if line else (None, None)
    return r


def pack(name: str, options, probe: bool = False) -> Result:
    """pack: <name>.psi in the folder of the cases; probe=True adds --probe <name>.probe."""
    if f"pack {name}" not in S.done:
        psi, probe_file = S.cases_dir / f"{name}.psi", S.cases_dir / f"{name}.probe"
        for stale in (psi, probe_file):
            if stale.exists():
                stale.unlink()
        arguments = ["pack", "--data", shown(S.sequence), "--file", shown(psi), "--csv"] + list(options)
        if probe:
            arguments += ["--probe", shown(probe_file)]
        r = run(f"pack_{name}", "psi_stream", arguments)
        r.file, r.probe = psi, (probe_file if probe else None)
        S.done[f"pack {name}"] = r
    return S.done[f"pack {name}"]


def play(name: str) -> Result:
    """play of the file of a pack already run, with a million sorted particles; reads its probe back if it has one."""
    if f"play {name}" not in S.done:
        arguments = ["play", "--file", shown(S.cases_dir / f"{name}.psi"), "--particles", PARTICLES, "--sort"]
        probe_file = S.cases_dir / f"{name}.probe"
        if probe_file.exists():
            arguments += ["--probe", shown(probe_file)]
        S.done[f"play {name}"] = with_integrity(run(f"play_{name}", "psi_stream", arguments))
    return S.done[f"play {name}"]


def pack_options(degree: int, variant: str):
    options = ["--degree", str(degree)] + SEGMENT
    if variant == "bound":
        options += ["--tolerance", TOLERANCES[degree]]
    elif variant == "floor":
        options += ["--tolerance", TOLERANCES[degree], "--fine-eps", COARSE_FINE_EPS]
    elif variant == "cap":
        options += ["--tolerance", TOLERANCES[degree], "--fine-passes", "1"]
    elif variant == "old":
        # File with a bound from before the evening of 15 November, byte for byte: 12 passes, zero margin, no seal.
        options += ["--tolerance", TOLERANCES[degree], "--fine-passes", "12", "--tolerance-margin", "0", "--no-seal"]
    elif variant == "raw_keys":
        options += ["--raw-keys"]
    elif variant == "exceptions":
        options += ["--eps", EXCEPTIONS_EPS[degree]]
    elif variant == "no_seal":
        # The "base" and "bound" files without their seal: those that pack wrote before the seal.
        options += ["--no-seal"]
    elif variant == "bound_no_seal":
        options += ["--tolerance", TOLERANCES[degree], "--no-seal"]
    return options


def pack_case(degree: int, variant: str) -> Result:
    """The files of the cases: d<degree>_<variant>.psi. base, bound and exceptions carry a probe; base_again and
    bound_again run the same command a second time (determinism)."""
    plain = re.sub(r"_again$", "", variant)
    return pack(f"d{degree}_{variant}", pack_options(degree, plain), probe=plain in ("base", "bound", "exceptions"))


def require_probe(reading: Result, what: str):
    """"Probe" line of a play: the probed frame is the last of the segment, and the field of the reader is that of the
    encoder."""
    line = found(r"^Probe\s*:\s*frame (\d+), (\d+) points; .*: (\S+)\s*$", reading.output, multiline=True)
    require(line, f'"Probe" line missing or incomplete (log {reading.log})')
    frame, points, error = int(line.group(1)), int(line.group(2)), line.group(3)
    require(frame == FIRST + COUNT - 1 and points > 0, f"probe ({what}): frame {frame}, {points} points (expected: frame {FIRST + COUNT - 1})")
    require(error == "0.000e+00", f"probe ({what}): difference {error} between the field of the reader and that of the encoder (log {reading.log})")


def require_intact(reading: Result):
    """Complete and intact read-back: code 0, every frame read, none whose state differs."""
    require_code(reading, 0)
    require(reading.differing is not None, f'"Integrity" line missing from the output of play (log {reading.log})')
    require(reading.differing == 0, f"{reading.differing} frame(s) whose state differs from that of the sender (log {reading.log})")
    require(found(f"{COUNT} frames read out of {COUNT}", reading.output), f"play did not read the {COUNT} frames (log {reading.log})")


def require_bound_missed(p: Result, tolerance: float, cause: str):
    """Bound missed: code 4, lines "Passes", "Floor" and "Bound", "Failure" message that names the expected cause, file
    written and consistent (the header and the table count the same frames above the certification threshold)."""
    threshold = f32(tolerance - 0.5 * MARGIN)
    require_code(p, 4)
    for label in ("Passes", "Floor", "Bound"):
        require(found(rf"^{label}\s*:", p.output, multiline=True), f'"{label}" line missing from the output of pack (log {p.log})')
    require(found("Failure: bound not held", p.error), f'message "Failure: bound not held" missing (log {p.log})')
    require(cause in p.error, f'the failure message does not name the expected cause ("{cause}") (log {p.log})')
    require(p.file.exists(), "pack exits with code 4 without having written the file")
    f = Stream(p.file)
    over = sum(1 for frame in f.frames if frame.max_error > threshold)
    require(over > 0, f"no frame of the table above the threshold {threshold:.6f} u' whereas pack exits with code 4")
    require(f.fine.frames_over == over, f"the header counts {f.fine.frames_over} frame(s) above the threshold, the table {over}")


def alter(source, name: str, modify) -> str:
    """Copy of a valid file, altered by `modify` (which receives the bytes and returns the bytes to write)."""
    data = modify(bytearray(Path(source).read_bytes()))
    target = S.cases_dir / f"{name}.psi"
    target.write_bytes(bytes(data))
    return shown(target)


def guard_task(log: str, file: str) -> Task:
    """play of an altered file: few particles, short timeout."""
    return Task(log, "psi_stream", ["play", "--file", file, "--particles", "1000", "--sort"], GUARD_TIMEOUT)


def play_guard(log: str, file: str) -> Result:
    return with_integrity(run_task(guard_task(log, file)))


def get_int32(data, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def get_uint32(data, offset: int) -> int:
    return struct.unpack_from("<I", data, offset)[0]


def put_int32(data: bytearray, offset: int, value: int) -> bytearray:
    """Writes a 32-bit integer at the given offset of the bytes."""
    require(0 <= offset <= len(data) - 4, f"alteration outside the file (offset {offset})")
    struct.pack_into("<i", data, offset, value)
    return data


def put_uint64(data: bytearray, offset: int, value: int) -> bytearray:
    require(0 <= offset <= len(data) - 8, f"alteration outside the file (offset {offset})")
    struct.pack_into("<Q", data, offset, value)
    return data


def flip(data: bytearray, offset: int, mask: int = 1) -> bytearray:
    require(0 <= offset < len(data), f"alteration outside the file (offset {offset})")
    data[offset] ^= mask
    return data


def put_byte(data: bytearray, offset: int, value: int) -> bytearray:
    require(0 <= offset < len(data), f"alteration outside the file (offset {offset})")
    data[offset] = value
    return data


def zero(data: bytearray, start: int, end: int) -> bytearray:
    require(0 <= start and end <= len(data), f"alteration outside the file (offsets {start} to {end})")
    for k in range(start, end):
        data[k] = 0
    return data


def alteration_outcome(r: Result) -> str:
    """What an altered file is allowed to give at reading: a refusal (code 1 and message "Failure:"), or a read carried
    through to its end whose "Integrity" line counts the differing frames (code 1 if there are any). Never a CUDA
    error, a crash or a hang. Returns 'refusal', 'counted' (at least one differing frame) or 'read' (code 0, none)."""
    if r.hung:
        return "hang"
    if found("CUDA error", r.error):
        return "CUDA error"
    if r.code == 1 and found("Failure:", r.error):
        return "refusal"
    if r.code == 1 and r.differing is not None and r.differing >= 1:
        return "counted"
    if r.code == 0 and r.differing is not None and r.differing == 0:
        return "read"
    return f"unexpected code {readable_code(r.code)}"


def require_no_damage(r: Result, what: str, allowed=("refusal", "counted")) -> str:
    outcome = alteration_outcome(r)
    if outcome not in allowed:
        raise Failure(f"{what}: {outcome}{reason(r)} (log {r.log})")
    return outcome


# ---------------------------------------------------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------------------------------------------------

CASES = []


def case(name: str, body, *arguments, known: str = ""):
    """`known`: the case fails on a known and uncorrected defect of the code."""
    CASES.append((name, body, arguments, known))


def case_sequence_fingerprint():
    fingerprint(SEQUENCE_NAME, sha256_file(S.sequence))


def case_pack_play(degree: int):
    """pack then play: codes 0, expected header and frame types, no frame whose state differs."""
    p = pack_case(degree, "base")
    require_code(p, 0)
    f = Stream(p.file)
    require(f.magic == "PSISTRM2" and f.n == 64 and f.degree == degree and f.n_frames == COUNT and f.first_frame == FIRST,
            f"unexpected header: {f.magic}, {f.n}^3, degree {f.degree}, {f.n_frames} frames from {f.first_frame}")
    for j in range(COUNT):
        expected = 3 if j % 32 == 0 else 2 if j % 32 == 1 else 1
        require(f.frames[j].type == expected, f"frame {j} of type {f.frames[j].type} instead of {expected} (3: coded key, 2: after a key, 1: innovation)")
    require(found(r"^Clippings\s*:\s*0 in total", p.output, multiline=True), f"pack reports clippings (log {p.log})")
    require_intact(play(f"d{degree}_base"))


def case_determinism(degree: int):
    """Determinism: the same command twice gives the same file and the same probe, without and with a fine level."""
    for variant in ("base", "bound"):
        one = pack_case(degree, variant)
        require_code(one, 0)
        two = pack_case(degree, f"{variant}_again")
        require_code(two, 0)
        require(sha256_file(one.file) == sha256_file(two.file),
                f"two identical pack commands ({variant}) give two different files: {short(sha256_file(one.file))} and {short(sha256_file(two.file))}")
        require(sha256_file(one.probe) == sha256_file(two.probe), f"two identical pack commands ({variant}) give two different probes")


def case_bound_held(degree: int):
    """Certified bound: code 0, fine level present, largest error of the table below the certification threshold
    (tolerance minus half the margin, in float32 as in pack), and intact read-back."""
    tolerance = f32(TOLERANCES[degree])
    threshold = f32(tolerance - 0.5 * MARGIN)
    p = pack_case(degree, "bound")
    require_code(p, 0)
    f = Stream(p.file)
    require(f.magic == "PSISTRM3", f"file {f.magic} instead of PSISTRM3")
    require(f.fine.tolerance == tolerance and f.fine.margin == MARGIN,
            f"header of the fine level: tolerance {f.fine.tolerance:.9g}, margin {f.fine.margin:.9g} instead of {tolerance:.9g} and {MARGIN:.9g}")
    require(f.fine.max_nodes > 0, f"no fine node retained: the tolerance {TOLERANCES[degree]} does not put the fine level to the test")
    require(0 < f.max_error <= threshold, f"largest error of the table {f.max_error:.6f} u' above the certification threshold {threshold:.6f} u'")
    require(f.fine.frames_over == 0, f"the header counts {f.fine.frames_over} frame(s) above the threshold")
    require(f.fine.max_error == f.max_error, f"the header announces a largest error of {f.fine.max_error:.6f} u', the table {f.max_error:.6f} u'")
    reading = play(f"d{degree}_bound")
    require_intact(reading)
    require(found(r"^Certification\s*:\s*bound held", reading.output, multiline=True), f"play does not announce the bound held (log {reading.log})")


def case_bound_missed(degree: int):
    """Tolerance below the floor of the fine level: code 4, cause "floor", file written, consistent and readable."""
    p = pack_case(degree, "floor")
    require_bound_missed(p, f32(TOLERANCES[degree]), "The fine level cannot go below its own floor")
    line = found(r"^Floor\s*:\s*(\d+) frame", p.output, multiline=True)
    require(line, '"Floor" line unreadable')
    require(int(line.group(1)) > 0, f'the "Floor" line counts no frame (log {p.log})')
    reading = play(f"d{degree}_floor")
    require_intact(reading)
    require(found(r"^Warning\s*:\s*the encoder missed the bound", reading.output, multiline=True), f"play does not report the bound missed (log {reading.log})")


def case_pass_cap(degree: int):
    """Cap on the passes: with a single marking pass, the bound is missed and the cause named is the cap."""
    p = pack_case(degree, "cap")
    require_bound_missed(p, f32(TOLERANCES[degree]), "The cap of 1 pass(es) was reached")
    line = found(r"^Passes\s*:\s*cap of 1 pass\(es\) per frame; (\d+) frame", p.output, multiline=True)
    require(line, '"Passes" line unreadable')
    require(int(line.group(1)) > 0, f'the "Passes" line counts no frame stopped on the cap (log {p.log})')


def case_old_behaviour(degree: int):
    """Old behaviour (12 passes, zero margin): marking and certification at the tolerance itself."""
    tolerance = f32(TOLERANCES[degree])
    p = pack_case(degree, "old")
    require_code(p, 0)
    f = Stream(p.file)
    require(f.fine.margin == 0, f"margin {f.fine.margin:.9g} instead of 0")
    require(f.fine.frames_over == 0, f"the header counts {f.fine.frames_over} frame(s) above the tolerance")
    # The table carries the error relative to u', rounded to float32: an error equal to the tolerance may exceed it by
    # a last bit, hence the relative margin of 1e-6 (pack itself compares in units of velocity).
    require(0 < f.max_error <= tolerance * (1 + 1e-6), f"largest error of the table {f.max_error:.7f} u' above the tolerance {tolerance:.7f} u'")
    require(found(r"^Passes\s*:\s*cap of 12 pass\(es\)", p.output, multiline=True), f"pack does not announce a cap of 12 passes (log {p.log})")


def case_probe(degree: int):
    """Probe: the field evaluated by the reader at the points of the probe is that of the encoder, without and with a
    fine level."""
    for variant in ("base", "bound"):
        p = pack_case(degree, variant)
        require_code(p, 0)
        require(p.probe.exists(), f"pack ({variant}) did not write a probe")
        reading = play(f"d{degree}_{variant}")
        require_code(reading, 0)
        require_probe(reading, variant)


def case_exceptions(degree: int):
    """Exception list: with a finer step, innovations leave +-127 (escapes) and targets of the fine level overflow; the
    reader must recover the state and the field of the encoder."""
    p = pack_case(degree, "exceptions")
    require_code(p, 0)
    line = found(r"^Exceptions\s*:\s*(\d+) innovation escapes and (\d+) overflows of the fine level", p.output, multiline=True)
    require(line, f'"Exceptions" line missing (log {p.log})')
    require(int(line.group(1)) > 0 and int(line.group(2)) > 0,
            f"{line.group(1)} escape(s) and {line.group(2)} overflow(s): --eps {EXCEPTIONS_EPS[degree]} no longer exercises both kinds of exception (log {p.log})")
    reading = play(f"d{degree}_exceptions")
    require_intact(reading)
    require_probe(reading, "exceptions")


def case_raw_keys(degree: int):
    """Coded keys against raw keys: same states, same read-back, same final positions."""
    coded = pack_case(degree, "base")
    require_code(coded, 0)
    raw = pack_case(degree, "raw_keys")
    require_code(raw, 0)
    fc, fr = Stream(coded.file), Stream(raw.file)
    for j in (0, 32):
        require(fc.frames[j].type == 3, f"base file: key {j} of type {fc.frames[j].type} instead of 3 (coded)")
        require(fr.frames[j].type == 0, f"--raw-keys file: key {j} of type {fr.frames[j].type} instead of 0 (raw)")
    for j in range(COUNT):
        require(fc.frames[j].hash == fr.frames[j].hash, f"frame {j}: the state of the encoder differs between coded keys and raw keys")
    require(fr.size > fc.size, f"the file with raw keys ({fr.size} bytes) is not longer than the file with coded keys ({fc.size} bytes)")
    reading_coded = play(f"d{degree}_base")
    require_intact(reading_coded)
    reading_raw = play(f"d{degree}_raw_keys")
    require_intact(reading_raw)
    require(reading_coded.positions == reading_raw.positions,
            f"different final positions: {reading_coded.positions} (coded keys) and {reading_raw.positions} (raw keys)")


def case_seal(degree: int):
    """Seal: pack seals by default. --no-seal gives back, byte for byte, the sealed file stripped of its last 48 bytes,
    that is the file that pack wrote before the seal. Both are read back the same way; the reader says which one
    carries a seal, and --require-seal refuses the one that has none. Without and with a fine level."""
    for sealed, bare in (("base", "no_seal"), ("bound", "bound_no_seal")):
        ps = pack_case(degree, sealed)
        require_code(ps, 0)
        pb = pack_case(degree, bare)
        require_code(pb, 0)
        fs, fb = Stream(ps.file), Stream(pb.file)
        require(fs.seal is not None and fs.seal.magic == "PSISEAL1" and fs.seal.reserved0 == 0 and fs.seal.reserved1 == 0,
                f"pack ({sealed}) did not write a valid seal after the table")
        require((fs.seal.fine != 0) == (fs.magic == "PSISTRM3"), f"seal ({sealed}): fingerprint of the header of the fine level {fs.seal.fine} in a {fs.magic} file")
        require(fb.seal is None, f"pack --no-seal ({bare}) wrote a seal")
        require(found(r"^Seal\s*:\s*header and frame table sealed", ps.output, multiline=True), f"pack ({sealed}) does not announce the seal (log {ps.log})")
        require(found(r"^Seal\s*:\s*none \(--no-seal\)", pb.output, multiline=True), f"pack --no-seal ({bare}) does not announce that there is no seal (log {pb.log})")
        data = ps.file.read_bytes()
        require(len(data) == fb.size + 48, f"the sealed file ({sealed}) is {len(data)} bytes long, the file without a seal {fb.size}: a difference of 48 bytes is expected")
        require(sha256_bytes(data[:fb.size]) == sha256_file(pb.file), f"the sealed file ({sealed}), stripped of its last 48 bytes, is not the file written with --no-seal")
        reading_sealed = play(f"d{degree}_{sealed}")
        require_intact(reading_sealed)
        reading_bare = play(f"d{degree}_{bare}")
        require_intact(reading_bare)
        require(found(r"^Seal\s*:\s*present and verified", reading_sealed.output, multiline=True), f"play ({sealed}) does not announce a verified seal (log {reading_sealed.log})")
        require(found(r"^Seal\s*:\s*absent", reading_bare.output, multiline=True), f"play ({bare}) does not say that the file has no seal (log {reading_bare.log})")
        require(reading_sealed.positions == reading_bare.positions,
                f"different final positions with and without a seal: {reading_sealed.positions} and {reading_bare.positions}")
        required = ["play", "--particles", "1000", "--sort", "--require-seal", "--file"]
        require_refusal(run(f"require_seal_d{degree}_{bare}", "psi_stream", required + [shown(S.cases_dir / f"d{degree}_{bare}.psi")], GUARD_TIMEOUT),
                        "file without a seal", f"play --require-seal of a file without a seal ({bare})")
        require_code(run(f"require_seal_d{degree}_{sealed}", "psi_stream", required + [shown(S.cases_dir / f"d{degree}_{sealed}.psi")], GUARD_TIMEOUT), 0)


def case_fingerprint_file(degree: int, variant: str):
    p = pack_case(degree, variant)
    require_code(p, 0)
    fingerprint(f"d{degree}_{variant}.psi", sha256_file(p.file))


def case_fingerprint_positions(degree: int):
    for variant in ("base", "bound"):
        p = pack_case(degree, variant)
        require_code(p, 0)
        reading = play(f"d{degree}_{variant}")
        require_intact(reading)
        fingerprint(f"d{degree}_{variant}.positions", reading.positions)


case("sequence.fingerprint", case_sequence_fingerprint)

for _degree in (2, 3):
    case(f"d{_degree}.pack_play", case_pack_play, _degree)
    case(f"d{_degree}.determinism", case_determinism, _degree)
    case(f"d{_degree}.bound_held", case_bound_held, _degree)
    case(f"d{_degree}.bound_missed", case_bound_missed, _degree)
    case(f"d{_degree}.pass_cap", case_pass_cap, _degree)
    case(f"d{_degree}.old_behaviour", case_old_behaviour, _degree)
    case(f"d{_degree}.probe", case_probe, _degree)
    case(f"d{_degree}.exceptions", case_exceptions, _degree)
    case(f"d{_degree}.raw_keys", case_raw_keys, _degree)
    case(f"d{_degree}.seal", case_seal, _degree)
    # Reference fingerprints: .psi files, then final positions read back (without and with a fine level). "old",
    # "no_seal" and "bound_no_seal" are written without a seal: their fingerprint is that of the files from before the
    # seal.
    for _variant in ("base", "bound", "old", "raw_keys", "exceptions", "no_seal", "bound_no_seal"):
        case(f"d{_degree}.fingerprint.{_variant}", case_fingerprint_file, _degree, _variant)
    case(f"d{_degree}.fingerprint.positions", case_fingerprint_positions, _degree)


# Guards: refusal with code 1 and a message, without a crash or a hang.
def case_guard_fb7_degree3():
    p = pack("guard_fb7_d3", ["--degree", "3", "--fb", "7"] + SEGMENT)
    require_refusal(p, "--fb too large", "pack --degree 3 --fb 7")
    require(not p.file.exists(), "pack refused, but a file was written")


def case_guard_clipping_fb7_degree2():
    """In degree 2, --fb 7 passes the guard but clips the lattice: code 3, file written, read back by differences."""
    p = pack("guard_fb7_d2", ["--degree", "2", "--fb", "7"] + SEGMENT)
    require_code(p, 3)
    require(found(r"Failure: \d+ coefficient\(s\) clipped", p.error), f"clipping message missing (log {p.log})")
    reading = play("guard_fb7_d2")
    require_intact(reading)
    require(found(r"^Warning\s*:\s*the encoder clipped", reading.output, multiline=True), f"play does not report the clipping (log {reading.log})")
    require(found("lattice read by differences", reading.output), f"play does not read this file by differences (log {reading.log})")


def case_guard_fine_level_capacity():
    """Tolerance out of reach at 64^3: the capacity guard of the fine level stops pack (code 1). The file left behind,
    without a table, is that of an interrupted encoding: play must refuse it."""
    for degree in (2, 3):
        p = pack(f"guard_capacity_d{degree}", ["--degree", str(degree), "--tolerance", TOLERANCE_BEYOND_CAPACITY] + SEGMENT)
        require_refusal(p, "fine level: capacity exceeded", f"pack --degree {degree} --tolerance {TOLERANCE_BEYOND_CAPACITY}")
        require(p.file.exists(), f"interrupted pack (degree {degree}): no file left behind, the case cannot check the refusal of play")
        require_refusal(play_guard(f"guard_capacity_d{degree}", shown(S.cases_dir / f"guard_capacity_d{degree}.psi")), "frame table missing",
                        f"play of the file left behind by the interrupted encoding (degree {degree})")


def case_guard_truncated_file():
    base = pack_case(2, "base")
    require_code(base, 0)
    bound = pack_case(3, "bound")
    require_code(bound, 0)
    # Bytes kept (negative: counted from the end). 272 bytes: the header, plus 64 bytes of the first frame (PSISTRM2)
    # or the header of the fine level (PSISTRM3). Both files carry a seal of 48 bytes: cut by 1 to 47 bytes, the file
    # ends in the middle of the seal; by 49 bytes or more, in the middle of the table. Cut by exactly 48 bytes, it is
    # the file without a seal, which is read (case "seal"): the seal protects against an alteration, not against its
    # own removal, which only play --require-seal refuses.
    tail = "unexpected bytes after the frame table"
    trials = [
        (base, 0, "invalid .psi file"),
        (base, 100, "invalid .psi file"),
        (base, 208, "frame table unreadable"),
        (base, 272, "frame table unreadable"),
        (base, -1, tail),
        (base, -49, "frame table unreadable"),
        (bound, 208, "header of the fine level unreadable"),
        (bound, 272, "frame table unreadable"),
        (bound, -20, tail),
        (bound, -96, "frame table unreadable"),
    ]
    for i, (source, size, message) in enumerate(trials, 1):
        total = source.file.stat().st_size
        kept = total + size if size < 0 else size
        file = alter(source.file, f"guard_truncated_{i}", lambda o, kept=kept: o[:kept])
        require_refusal(play_guard(f"guard_truncated_{i}", file), message, f"play of a file reduced to its first {kept} bytes out of {total}")


def case_guard_corrupt_magic():
    base = pack_case(2, "base")
    require_code(base, 0)
    file = alter(base.file, "guard_magic", lambda o: put_byte(o, 0, 0x58))
    require_refusal(play_guard("guard_magic", file), "invalid .psi file", "play, corrupt magic")


def case_guard_invalid_degree():
    base = pack_case(2, "base")
    require_code(base, 0)
    for value in (0, 7):
        file = alter(base.file, f"guard_degree_{value}", lambda o, value=value: put_int32(o, 12, value))
        require_refusal(play_guard(f"guard_degree_{value}", file), "invalid degree", f"play, degree of the header replaced by {value}")


def case_guard_corrupt_frame():
    """One bit flipped in the payload of frame 5: play must see it (code 1) on frames 5 to 31, then resynchronise on
    the key of frame 32. Without this case, "0 frame(s) whose state differs" would prove nothing."""
    base = pack_case(2, "base")
    require_code(base, 0)
    frame = Stream(base.file).frames[5]
    offset = frame.offset + frame.bytes - 10
    file = alter(base.file, "guard_frame", lambda o: flip(o, offset))
    reading = play_guard("guard_frame", file)
    require_code(reading, 1)
    require(reading.differing is not None, f'play exits with code 1 without an "Integrity" line{reason(reading)} (log {reading.log})')
    require(reading.differing == 27, f"{reading.differing} frame(s) whose state differs instead of 27 (frames 5 to 31) (log {reading.log})")


def case_guard_invalid_sequence():
    """Invalid input sequence: a refusal with a message. Until 15 November 2026, an uncaught exception of
    sequence_read_header stopped pack without a message, on a crash code (0xC0000409)."""
    base = pack_case(2, "base")
    require_code(base, 0)
    r = run("guard_sequence", "psi_stream",
            ["pack", "--data", shown(S.cases_dir / "d2_base.psi"), "--file", shown(S.cases_dir / "guard_sequence.psi")] + SEGMENT, GUARD_TIMEOUT)
    require_refusal(r, "Failure:", "pack of a file that is not a sequence")


def case_guard_swapped_degree():
    """Degree of the header changed from 2 to 3 in a file of degree 2. The fingerprint of the frames covers only the
    quantised state, which does not depend on the degree: until 15 November 2026, play exited with 0, announced "0
    frame(s) whose state differs" and returned other final positions. The seal, which pack writes by default, has the
    file refused."""
    base = pack_case(2, "base")
    require_code(base, 0)
    intact = play("d2_base")
    require_intact(intact)
    file = alter(base.file, "guard_degree_3", lambda o: put_int32(o, 12, 3))
    reading = with_integrity(run("guard_degree_3", "psi_stream", ["play", "--file", file, "--particles", PARTICLES, "--sort"], GUARD_TIMEOUT))
    if reading.code == 0:
        raise Failure(f"play accepts a file of degree 2 whose header says 3: code 0, {reading.differing} frame(s) whose state differs, "
                      f"final positions {reading.positions} instead of {intact.positions}")
    require_refusal(reading, "seal of the file: the header has been altered", "play, degree of the header changed from 2 to 3")


def case_guard_sealed_header():
    """Fields of the header, of the header of the fine level, of the table or of the seal altered in a sealed file:
    refusal at opening. The message names the seal and the altered part, except when the alteration moves the expected
    end of the table: the refusal then bears on the size of the file or on the table. Offsets: docs/FORMAT.md, sections
    2 to 4."""
    base = pack_case(2, "base")
    require_code(base, 0)
    bound = pack_case(3, "bound")
    require_code(bound, 0)
    tb, to = Stream(base.file), Stream(bound.file)
    require(tb.seal is not None and to.seal is not None, "the base file and the file with a bound do not carry a seal")
    header = "seal of the file: the header has been altered"
    fine = "seal of the file: the header of the fine level has been altered"
    table = "seal of the file: the frame table has been altered"
    tail = "unexpected bytes after the frame table"
    trials = [
        ("n", base, header, lambda o: put_int32(o, 8, 128)),
        ("mf", base, header, lambda o: put_int32(o, 16, 1)),
        ("levels", base, header, lambda o: put_int32(o, 20, 2)),
        ("fb", base, header, lambda o: put_int32(o, 24, 5)),
        ("n_frames", base, tail, lambda o: put_int32(o, 32, COUNT - 1)),
        ("delta", base, header, lambda o: flip(o, 40)),
        ("dt_frame", base, header, lambda o: flip(o, 44)),
        ("coef_0", base, header, lambda o: put_int32(o, 64, get_int32(o, 64) + 1)),
        ("coef_2b", base, header, lambda o: put_int32(o, 92, get_int32(o, 92) + 1)),
        ("index_offset", base, "corrupt frame table", lambda o: put_uint64(o, 136, tb.table_offset + 48)),
        ("wrapped", base, header, lambda o: put_int32(o, 204, 1)),
        ("mean", base, table, lambda o: flip(o, tb.table_offset + 48 * 3 + 28)),
        ("fingerprint", base, table, lambda o: flip(o, tb.table_offset + 48 * 7 + 16, 0x80)),
        ("type", base, table, lambda o: put_int32(o, tb.table_offset + 48 * 5 + 24, 2)),
        ("seal_value", base, header, lambda o: flip(o, tb.table_end + 8)),
        ("seal_magic", base, tail, lambda o: put_byte(o, tb.table_end, 0x58)),
        ("seal_zero", base, tail, lambda o: put_byte(o, tb.table_end + 44, 1)),
        ("fine_levels", bound, header, lambda o: put_int32(o, 20, 2)),
        ("fine_delta", bound, fine, lambda o: flip(o, 216)),
        ("fine_nodes", bound, fine, lambda o: put_int32(o, 228, get_int32(o, 228) + 1)),
        ("fine_margin", bound, fine, lambda o: flip(o, 240)),
        ("fine_mean", bound, table, lambda o: flip(o, to.table_offset + 48 * 9 + 32)),
    ]
    for name, source, message, modify in trials:
        file = alter(source.file, f"guard_sealed_{name}", modify)
        require_refusal(play_guard(f"guard_sealed_{name}", file), message, f'play of a sealed file, alteration "{name}"')


def case_guard_geometry_no_seal():
    """The same alterations of geometry, and alterations of the table and of the frames, in files WITHOUT a seal:
    nothing reports the altered header, and it is the check of each frame against the geometry (before any decoding
    kernel) that must hold. Required outcome: a refusal, or frames counted on the "Integrity" line. Until 15 November
    2026, n or levels altered gave "CUDA error: an illegal memory access"."""
    p2 = pack_case(2, "no_seal")
    require_code(p2, 0)
    p3 = pack_case(3, "bound_no_seal")
    require_code(p3, 0)
    pk = pack_case(2, "raw_keys")
    require_code(pk, 0)
    t2, t3, tk = Stream(p2.file), Stream(p3.file), Stream(pk.file)
    refusal, both = ("refusal",), ("refusal", "counted")
    # Frame 5 (innovation, type 1): offset of its payload, of its tables (8 bytes of empty exception list, then three
    # tables of 512 bytes), of its block offsets, of its stream area (110 blocks at 64^3); offset of its record.
    c2 = t2.frames[5].offset
    d2 = c2 + 8 + 3 * 512
    z2 = d2 + 4 * 110
    e2 = t2.table_offset + 48 * 5
    c3 = t3.frames[5].offset
    q3 = c3 + t3.frames[5].bytes - 16
    bytes5 = t2.frames[5].bytes
    trials = [
        ("n128", p2, both, lambda o: put_int32(o, 8, 128)),
        ("n32", p2, refusal, lambda o: put_int32(o, 8, 32)),
        ("mf1", p2, both, lambda o: put_int32(o, 16, 1)),
        ("levels2", p2, both, lambda o: put_int32(o, 20, 2)),
        ("levels1", p2, both, lambda o: put_int32(o, 20, 1)),
        ("type0_key", p2, refusal, lambda o: put_int32(o, t2.table_offset + 24, 1)),
        ("type0_raw", p2, refusal, lambda o: put_int32(o, t2.table_offset + 24, 0)),
        ("type5_raw", p2, refusal, lambda o: put_int32(o, e2 + 24, 0)),
        ("type5_coded", p2, both, lambda o: put_int32(o, e2 + 24, 3)),
        ("type5_after", p2, both, lambda o: put_int32(o, e2 + 24, 2)),
        ("type5_9", p2, refusal, lambda o: put_int32(o, e2 + 24, 9)),
        ("bytes_less", p2, both, lambda o: put_uint64(o, e2 + 8, bytes5 - 2000)),
        ("bytes_short", p2, refusal, lambda o: put_uint64(o, e2 + 8, 100)),
        ("bytes_huge", p2, refusal, lambda o: put_uint64(o, e2 + 8, 0x7FFFFFFFFFFF)),
        ("offset_shifted", p2, both, lambda o: put_uint64(o, e2, c2 + 4)),
        ("offset_outside", p2, refusal, lambda o: put_uint64(o, e2, t2.size + 8)),
        ("exceptions3", p2, both, lambda o: put_int32(o, c2, 3)),
        ("exceptions_huge", p2, refusal, lambda o: put_int32(o, c2, 0x7FFFFFFF)),
        ("frequencies", p2, refusal, lambda o: flip(o, c2 + 8 + 512)),
        ("block_offset0", p2, refusal, lambda o: put_int32(o, d2, 100)),
        ("block_offset_huge", p2, refusal, lambda o: put_int32(o, d2 + 4 * 7, -16)),
        ("block_offset_backwards", p2, refusal, lambda o: put_int32(o, d2 + 4 * 50, 4)),
        ("block_offset_end", p2, refusal, lambda o: put_int32(o, d2 + 4 * 109, bytes5)),
        ("zero_state", p2, both, lambda o: zero(o, z2, z2 + 600)),
        ("last_block_zero", p2, both, lambda o: zero(o, z2 + get_uint32(o, d2 + 4 * 109), c2 + bytes5)),
        ("fine_n128", p3, both, lambda o: put_int32(o, 8, 128)),
        ("fine_levels2", p3, both, lambda o: put_int32(o, 20, 2)),
        ("fine_max_bricks", p3, refusal, lambda o: put_int32(o, 224, 1)),
        ("fine_max_nodes", p3, refusal, lambda o: put_int32(o, 228, 1)),
        ("fine_brick", p3, refusal, lambda o: put_int32(o, 232, 8)),
        ("fine_bricks_plus", p3, both, lambda o: put_int32(o, q3, get_int32(o, q3) + 1)),
        ("fine_bricks_minus", p3, both, lambda o: put_int32(o, q3, get_int32(o, q3) - 1)),
        ("fine_bricks_huge", p3, refusal, lambda o: put_int32(o, q3, 0xFFFFFF)),
        ("fine_nodes_plus", p3, both, lambda o: put_int32(o, q3 + 4, get_int32(o, q3 + 4) + 5)),
        ("fine_nodes_zero", p3, both, lambda o: put_int32(o, q3 + 4, 0)),
        ("fine_section_short", p3, both, lambda o: put_int32(o, q3 + 8, get_int32(o, q3 + 8) - 8)),
        ("fine_section_huge", p3, refusal, lambda o: put_int32(o, q3 + 8, 0x7FFFFFF8)),
        ("fine_section_odd", p3, refusal, lambda o: put_int32(o, q3 + 8, get_int32(o, q3 + 8) + 4)),
        ("fine_block_offset", p3, refusal, lambda o: put_int32(o, q3 - get_int32(o, q3 + 8) + 3 * 512 + 4, 0)),
        ("raw_levels2", pk, refusal, lambda o: put_int32(o, 20, 2)),
        ("raw_bytes", pk, refusal, lambda o: put_uint64(o, tk.table_offset + 8, tk.frames[0].bytes - 8)),
        ("raw_exceptions", pk, refusal, lambda o: put_int32(o, tk.frames[0].offset + tk.frames[0].bytes - 8, 2)),
    ]
    for name, source, allowed, modify in trials:
        # The file with raw keys is written with a seal: the three "raw_" alterations are made in a copy stripped of its
        # seal (the 48 bytes that follow the frame table), so that they reach the check of the geometry instead of being
        # refused by the seal. The PowerShell suite this one was ported from meant to do so and did not: its copies kept
        # the seal, and raw_levels2 and raw_bytes were refused by it.
        if name.startswith("raw_"):
            modify = (lambda inner: lambda o: inner(o[:tk.table_end]))(modify)
        file = alter(source.file, f"guard_geo_{name}", modify)
        require_no_damage(play_guard(f"guard_geo_{name}", file), f'play of a file without a seal, alteration "{name}"', allowed)
    # What nothing can see without a seal: fb changed from 2 to 5. The decoded state is the same, the field is not. The
    # reader reads the file (code 0), and it says that the header is covered by no fingerprint.
    file = alter(p2.file, "guard_geo_fb5", lambda o: put_int32(o, 24, 5))
    reading = play_guard("guard_geo_fb5", file)
    require_no_damage(reading, "play, fb changed from 2 to 5 without a seal", ("read",))
    require(found(r"^Seal\s*:\s*absent .*covered by no\s+fingerprint", reading.output, multiline=True),
            f"play reads a file without a seal without saying that its header is not protected (log {reading.log})")


class Generator:
    """Linear congruential generator, modulo 2^31, written here so that the draws depend neither on the language nor
    on a library."""

    def __init__(self, seed: int):
        self.state = seed

    def below(self, bound: int) -> int:
        """Integer in [0, bound[, bound < 2^30: two draws, 15 high-order bits each."""
        value = 0
        for _ in range(2):
            self.state = (self.state * 1103515245 + 12345) % 2147483648
            value = value * 32768 + self.state // 65536
        return value % bound


def random_alterations(stream: Stream, data: bytes, generator: Generator):
    """The draws of one source file: zone, offset, byte before, byte after and kind of each alteration."""
    frames_start = 272 if stream.magic == "PSISTRM3" else 208
    for k in range(ALTERATIONS_PER_FILE):
        # Zone: header (and header of the fine level), table (and seal), a whole frame, or the structure bytes of a
        # frame (its first 2048 bytes: exception list, tables, block offsets; or its last 64).
        zone = ("header", "table", "frame", "structure")[k % 4]
        if zone == "header":
            offset = generator.below(frames_start)
        elif zone == "table":
            offset = stream.table_offset + generator.below(stream.size - stream.table_offset)
        elif zone == "frame":
            frame = stream.frames[generator.below(stream.n_frames)]
            offset = frame.offset + generator.below(frame.bytes)
        else:
            frame = stream.frames[generator.below(stream.n_frames)]
            if generator.below(2) == 0:
                offset = frame.offset + generator.below(min(2048, frame.bytes))
            else:
                offset = frame.offset + frame.bytes - 1 - generator.below(64)
        before = data[offset]
        if generator.below(2) == 0:
            after, kind = before ^ (1 << generator.below(8)), "bit flipped"
        else:
            after, kind = (before + 1 + generator.below(255)) % 256, "byte replaced"
        yield k, zone, offset, before, after, kind


def case_guard_random_alteration():
    """Reproducible random alteration: in four files (without and with a fine level, with and without a seal), one bit
    flipped or one byte replaced, at an offset drawn in the header, in the table (seal included) or in a frame. Each
    altered file must give a refusal, or a complete read that counts the differing frames; or else, if the byte
    touched carries nothing (padding, reserved field, informative field), a read without a differing frame AND the
    final positions of the intact file."""
    sources = [
        ("d2_base", pack_case(2, "base"), True),
        ("d3_bound", pack_case(3, "bound"), True),
        ("d2_no_seal", pack_case(2, "no_seal"), False),
        ("d3_bound_no_seal", pack_case(3, "bound_no_seal"), False),
    ]
    folder = S.cases_dir / "random"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    tasks, draws = [], []
    for rank, (source_name, p, sealed) in enumerate(sources):
        require_code(p, 0)
        stream = Stream(p.file)
        data = p.file.read_bytes()
        generator = Generator(ALTERATION_SEED + rank)      # one generator per source file
        tasks.append(guard_task(f"random_{source_name}_intact", shown(S.cases_dir / f"{source_name}.psi")))
        draws.append({"source": source_name, "sealed": sealed, "intact": True})
        for k, zone, offset, before, after, kind in random_alterations(stream, data, generator):
            copy = bytearray(data)
            copy[offset] = after
            name = f"{source_name}_{k:03d}"
            (folder / f"{name}.psi").write_bytes(bytes(copy))
            tasks.append(guard_task(f"random_{name}", shown(folder / f"{name}.psi")))
            draws.append({"source": source_name, "sealed": sealed, "intact": False, "zone": zone, "offset": offset,
                          "what": f"{kind}, {before:02X} -> {after:02X}", "name": name})
    results = run_batch(tasks, SIMULTANEOUS_READS)
    counts = {"refused": 0, "counted": 0, "without effect": 0, "not seen (no seal)": 0}
    faults = []
    intact = None
    reruns = 0
    for i, r in enumerate(results):
        draw = draws[i]
        # A read that came back without any output (neither a line of play, nor a message) whereas its code is that of
        # a normal read: observed once on 15 November 2026 on 2 reads of this batch, cause not established (see
        # tests/README.md). It is run again alone, once, and counted on the line of the case; empty again, it is a
        # fault.
        if not r.hung and not r.output.strip() and not r.error.strip():
            reruns += 1
            r = run(f"{tasks[i].log}_rerun", tasks[i].program, tasks[i].arguments, tasks[i].timeout)
        r = with_integrity(r)
        if draw["intact"]:
            require_code(r, 0)
            require(r.differing == 0 and r.positions, f"intact file {draw['source']}: read-back not intact (log {r.log})")
            intact = r
            continue
        outcome = alteration_outcome(r)
        fault = None
        if outcome == "refusal":
            verdict = "refused"
        elif outcome == "counted":
            verdict = "counted"
        elif outcome == "read":
            if r.positions == intact.positions:
                verdict = "without effect"
            # Without a seal, nothing covers the header nor the table: an altered field there changes the field
            # rendered without being seen.
            elif not draw["sealed"] and draw["zone"] in ("header", "table"):
                verdict = "not seen (no seal)"
            else:
                fault = "read without a differing frame, different final positions"
        else:
            fault = outcome
        if fault:
            faults.append(f"{draw['name']} ({draw['zone']}, offset {draw['offset']}, {draw['what']}): {fault}{reason(r)}")
        else:
            counts[verdict] += 1
            if not S.keep_alterations:
                (folder / f"{draw['name']}.psi").unlink()
    altered = len(results) - len(sources)
    S.detail = f"{altered} altered files: " + ", ".join(f"{count} {verdict}" for verdict, count in counts.items())
    if reruns:
        S.detail += f"; {reruns} read(s) run again after an empty output"
    if faults:
        raise Failure(f"{len(faults)} altered file(s) out of {altered} outside the allowed outcomes (kept in {shown(folder)}, logs random_*): " + " | ".join(faults))


def case_guard_altered_overflow():
    """Altered overflow of level 0. The fingerprint of a frame covers the quantised state; the value of an overflow
    (ExEntry.val, bit 31 of idx set to 1) touches only the sampling lattice. Altered, it changes the field without
    changing the state: play exits with 0 and announces "0 frame(s) whose state differs". The probe of the last frame,
    for its part, sees the difference. Neither the seal (header and table) nor the check of the geometry covers the
    value of an exception."""
    p = pack_case(2, "exceptions")
    require_code(p, 0)
    stream = Stream(p.file)
    last = stream.frames[COUNT - 1]
    data = p.file.read_bytes()
    count = get_uint32(data, last.offset)
    offset = -1
    for k in range(count):
        entry = last.offset + 8 + 8 * k
        if get_uint32(data, entry) & 0x80000000:
            offset = entry + 4
            break
    require(offset >= 0, f"the last frame of d2_exceptions.psi carries no overflow of level 0 ({count} exception(s)): the case cannot alter it")
    file = alter(p.file, "guard_overflow", lambda o: put_int32(o, offset, get_int32(o, offset) + 3))
    reading = with_integrity(run("guard_overflow", "psi_stream",
                                 ["play", "--file", file, "--particles", "1000", "--sort", "--probe", shown(S.cases_dir / "d2_exceptions.probe")], GUARD_TIMEOUT))
    line = found(r"^Probe\s*:.*: (\S+)\s*$", reading.output, multiline=True)
    probe = line.group(1) if line else "missing"
    if reading.code == 0:
        raise Failure(f"play accepts a frame in which an overflow of level 0 is altered: code 0, {reading.differing} frame(s) whose state differs, "
                      f"whereas the probe measures a difference of {probe} between the field of the reader and that of the encoder")
    require_no_damage(reading, "play, altered overflow of level 0")


case("guard.fb7_degree3", case_guard_fb7_degree3)
case("guard.clipping_fb7_degree2", case_guard_clipping_fb7_degree2)
case("guard.fine_level_capacity", case_guard_fine_level_capacity)
case("guard.truncated_file", case_guard_truncated_file)
case("guard.corrupt_magic", case_guard_corrupt_magic)
case("guard.invalid_degree", case_guard_invalid_degree)
case("guard.corrupt_frame", case_guard_corrupt_frame)
case("guard.invalid_sequence", case_guard_invalid_sequence)
case("guard.swapped_degree", case_guard_swapped_degree)
case("guard.sealed_header", case_guard_sealed_header)
case("guard.geometry_no_seal", case_guard_geometry_no_seal)
case("guard.random_alteration", case_guard_random_alteration)
# Known defects of the code, left as they are (tests/README.md). Each case describes the expected behaviour: it will
# pass the day the defect is corrected, and `known` will then have to be removed.
case("guard.altered_overflow", case_guard_altered_overflow,
     known="the fingerprint of a frame does not cover the value of the overflows of level 0, which touch only the lattice")


# ---------------------------------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------------------------------

def matches(name: str, patterns) -> bool:
    return any(fnmatch.fnmatchcase(name.lower(), pattern.lower()) for pattern in patterns)


def write_fingerprints(context: dict):
    document = {
        "_warning": [
            "Reference fingerprints of tests/test_codec.py: SHA-256 of the sequence and of the .psi files of the cases, and fingerprints of the final positions read back by play.",
            "They are valid only for the toolchain described in \"toolchain\" (version of CUDA and of cuFFT, host compiler, architecture of the GPU) and for the settings of the cases (\"settings\"): the sequence comes out of a chaotic simulation in floating point, whose bytes change with the FFT library, the compiler or the GPU, and every file encoded with it.",
            "On another machine or after a change of toolchain: first check the integrity (every case of tests/test_codec.py other than \"fingerprint\" must pass: play reads each file back), then regenerate this file with python tests/test_codec.py --update.",
            "Do not edit by hand. On the same toolchain, a fingerprint that changes is a regression of the encoder or an intended change of the format: say so in the commit that regenerates this file.",
        ],
        "generated_on": datetime.date.today().isoformat(),
        "toolchain": context,
        "settings": SETTINGS,
        "fingerprints": S.recorded,
    }
    FINGERPRINTS_FILE.write_bytes((json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--build", default="build", help="build folder holding dns and psi_stream (default: build, from the root of the repository)")
    parser.add_argument("--only", nargs="+", action="extend", metavar="PATTERN", help="run only the cases whose name matches one of these patterns (wildcards * and ?)")
    parser.add_argument("--update", action="store_true", help="regenerate tests/fingerprints.json (the complete suite must pass)")
    parser.add_argument("--keep-alterations", action="store_true", help="keep every altered file of the case guard.random_alteration")
    arguments = parser.parse_args()
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    S.build = (ROOT / arguments.build).resolve()
    S.cases_dir = S.build / "tests" / "cases"
    S.logs_dir = S.build / "tests" / "logs"
    S.sequence = S.build / "tests" / SEQUENCE_NAME
    S.update, S.keep_alterations = arguments.update, arguments.keep_alterations
    patterns = [p.strip().strip("'\"") for given in (arguments.only or ["*"]) for p in given.split(",")]
    patterns = [p for p in patterns if p]

    print(f"Psi-treillis: regression suite of the pack / play chain at 64^3 ({datetime.datetime.now():%Y-%m-%d %H:%M})")
    missing = [name + EXE for name in ("dns", "psi_stream") if not (S.build / (name + EXE)).is_file()]
    if missing:
        print(f"PREREQUISITES: {', '.join(shown(S.build / m) for m in missing)} missing. This suite compiles nothing: build dns and psi_stream first "
              "(tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere), or name the build folder with --build.")
        return 2
    S.cases_dir.mkdir(parents=True, exist_ok=True)
    S.logs_dir.mkdir(parents=True, exist_ok=True)

    if not sequence_valid():
        start = time.monotonic()
        options = DNS_OPTIONS + ["--out", shown(S.sequence)]
        r = run("dns", "dns", options, DNS_TIMEOUT)
        if r.code != 0 or not sequence_valid():
            print(f"SEQUENCE: dns failed (code {readable_code(r.code)}){reason(r)}; log {r.log}")
            return 2
        print(f"SEQUENCE: {shown(S.sequence)} produced by dns {' '.join(options)} ({time.monotonic() - start:.0f} s)")
        statistics = [line for line in re.split(r"\r?\n", r.output) if found("^Statistics", line)]
        if statistics:
            print(f"          {statistics[-1].strip()}")

    context = toolchain(S.build)
    if FINGERPRINTS_FILE.is_file():
        S.reference = json.loads(FINGERPRINTS_FILE.read_text(encoding="utf-8"))
    S.context_differences = []
    if S.reference:
        for key, value in context.items():
            known = str(S.reference["toolchain"].get(key, "")) if "toolchain" in S.reference else ""
            if known != str(value):
                S.context_differences.append(f"{key}: {value} (fingerprints: {known})")
        if str(S.reference.get("settings", "")) != SETTINGS:
            S.context_differences.append(f"settings of the cases: {SETTINGS} (fingerprints: {S.reference.get('settings', '')})")
    print(f"Toolchain   : {context['cuda']}, {context['host_compiler']}, {context['cufft_library']}, {context['gpu']} ({context['cuda_architecture']}), driver {context['driver']}")
    if not S.reference:
        print("Fingerprints: tests/fingerprints.json missing" + (": it will be created" if S.update else
              ': the "fingerprint" cases will fail; check the other cases, then --update'))
    elif S.context_differences:
        print('Fingerprints: recorded on ANOTHER toolchain or with other settings; the "fingerprint" cases may fail without a regression.')
        for difference in S.context_differences:
            print(f"              {difference}")
        print("              Check that the other cases pass, then regenerate with --update.")
    else:
        print(f"Fingerprints: recorded on this toolchain, on {S.reference.get('generated_on', '')}")

    if S.update and patterns != ["*"]:
        print("--update requires the complete suite: remove --only.")
        return 2
    selected = [c for c in CASES if matches(c[0], patterns)]
    if not selected:
        print(f"No case matches --only {', '.join(patterns)}. Cases: {', '.join(c[0] for c in CASES)}")
        return 2
    print()
    start = time.monotonic()
    failed, known_defects = [], []
    for name, body, case_arguments, known in selected:
        S.note = S.detail = ""
        why = None
        try:
            body(*case_arguments)
        except Failure as e:
            why = str(e)
        except Exception as e:  # noqa: BLE001 - a failing case must not stop the others
            why = f"{type(e).__name__}: {e}"
        if why is not None and known:
            known_defects.append(name)
            print(f"KNOWN    {name}: {why} [known defect, not corrected: {known}]")
        elif why is not None:
            failed.append(name)
            print(f"FAIL     {name}: {why}")
        elif known:
            print(f"OK       {name}: the known defect no longer shows ({known}); remove `known` from this case")
        elif S.update and S.note:
            print(f"UPDATED  {name}: {S.note}")
        else:
            print(f"OK       {name}" + (f": {S.detail}" if S.detail else ""))
        sys.stdout.flush()
    duration = time.monotonic() - start
    print()
    print(f"TOTAL: {len(selected)} cases, {len(selected) - len(failed) - len(known_defects)} OK, {len(failed)} FAILED, {len(known_defects)} known defect(s)")
    print(f"Duration of the cases: {duration:.0f} s (sequence apart)")
    if failed:
        print(f"Failed cases: {', '.join(failed)}")

    if S.update:
        if failed:
            print(f"tests/fingerprints.json is NOT regenerated: {len(failed)} failed case(s). Correct them first.")
            return 1
        changed = [key for key, value in S.recorded.items() if reference_of(key) != value]
        write_fingerprints(context)
        print(f"tests/fingerprints.json regenerated: {len(S.recorded)} fingerprints, of which {len(changed)} new or changed" + (f" ({', '.join(changed)})" if changed else ""))
        return 0
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
