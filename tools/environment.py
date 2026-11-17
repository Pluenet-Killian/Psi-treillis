# SPDX-License-Identifier: Apache-2.0
"""Records the environment of a measurement campaign, in benchmarks/environment.txt.

    python tools/environment.py              writes benchmarks/environment.txt and prints it
    python tools/environment.py --no-data    without the sequences of data/ (their first frame is not read)
    python tools/environment.py --print      prints only, writes nothing

To be attached to every series of measurements: a figure without this record cannot be compared from one machine to
another. What is recorded:
  - the machine: system, processor, memory;
  - the GPU: model, driver, memory, BIOS, and whether the machine runs on mains;
  - the toolchain: nvcc, host compiler, CMake, Ninja (those of build/ when its CMake cache names them, otherwise those
    of the path), Python and NumPy (those that run this script);
  - the repository: commit, branch, state of the working tree;
  - the comparison compressors cloned in third_party/: revision, date of the commit, address of the remote;
  - the binaries of build/ and of build/codecs/: date and SHA-256;
  - the sequences of data/: the fields of their header, their size, and a SHA-256 of the header and of the first frame
    (not of the whole file: a sequence at 512^3 holds 96 GiB).

It records neither the name of the machine, nor the name of the user, nor any path outside the repository: every path
printed is relative to the root.

Needs: Python 3.8 or later, standard library only (NumPy is looked for to record its version), Windows or Linux.
`git` and `nvidia-smi` are called if they are found; what cannot be read is recorded as "not found". Nothing is run
on the GPU.
"""

import argparse
import datetime
import getpass
import hashlib
import os
import platform
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
OUTPUT = ROOT / "benchmarks" / "environment.txt"
WINDOWS = sys.platform == "win32"
EXE = ".exe" if WINDOWS else ""
SEQ_HEADER_BYTES = 72  # src/sequence.h

# Binaries built by tools/build.ps1 -Tools (or CMake with -DPSIT_BUILD_TOOLS=ON), then the comparison compressors built
# by benchmarks/codecs/build_codecs.ps1, at the paths where benchmarks/codecs/comparison_codecs.py calls them.
BINARIES = [
    "psitreillis.dll" if WINDOWS else "libpsitreillis.so",
    "psit" + EXE, "example_roundtrip" + EXE, "dns" + EXE, "psi_stream" + EXE, "tracers" + EXE, "psi_view" + EXE, "psi_ref" + EXE,
    "codecs/sz3/tools/sz3/sz3" + EXE, "codecs/sperr/sperr_capi.dll", "codecs/zfp/bin/zfpcmd" + EXE, "codecs/cuszp/cuSZp_cli" + EXE,
]


def line(name, value):
    return f"{name + ':':<36} {value}"


def run(*command, cwd=None):
    """Standard output of a command, stripped; None if it cannot be run or fails."""
    try:
        done = subprocess.run([str(c) for c in command], cwd=cwd, capture_output=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if done.returncode != 0:
        return None
    return done.stdout.decode("utf-8", "replace").strip()


def first_line(text):
    return text.splitlines()[0].strip() if text else None


# ---- Machine --------------------------------------------------------------------------------------------------------

def machine_windows():
    import ctypes
    import winreg
    from ctypes import wintypes

    system = f"Windows {platform.release()} {platform.win32_edition() or ''} {platform.version()}".replace("  ", " ")
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            processor = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    except OSError:
        processor = "not found"

    class ProcessorInformation(ctypes.Structure):  # SYSTEM_LOGICAL_PROCESSOR_INFORMATION
        _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int), ("detail", ctypes.c_ulonglong * 2)]

    kernel = ctypes.windll.kernel32
    size = wintypes.DWORD(0)
    kernel.GetLogicalProcessorInformation(None, ctypes.byref(size))
    entries = (ProcessorInformation * (size.value // ctypes.sizeof(ProcessorInformation)))()
    cores = None
    if kernel.GetLogicalProcessorInformation(entries, ctypes.byref(size)):
        cores = sum(1 for entry in entries if entry.relationship == 0)  # RelationProcessorCore

    class MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in ("total", "available", "total_page", "available_page", "total_virtual",
                                                    "available_virtual", "available_extended")]

    memory = MemoryStatus()
    memory.length = ctypes.sizeof(MemoryStatus)
    total = memory.total if kernel.GlobalMemoryStatusEx(ctypes.byref(memory)) else None

    class PowerStatus(ctypes.Structure):  # SYSTEM_POWER_STATUS
        _fields_ = [("ac_line", ctypes.c_ubyte), ("battery_flag", ctypes.c_ubyte), ("battery_percent", ctypes.c_ubyte),
                    ("system_flag", ctypes.c_ubyte), ("battery_time", wintypes.DWORD), ("battery_full_time", wintypes.DWORD)]

    power = PowerStatus()
    mains = None
    if kernel.GetSystemPowerStatus(ctypes.byref(power)) and power.ac_line in (0, 1):
        mains = power.ac_line == 1
    return system, processor, cores, total, mains


def machine_linux():
    system = platform.system() + " " + platform.release()
    try:
        for entry in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            if entry.startswith("PRETTY_NAME="):
                system = entry.split("=", 1)[1].strip().strip('"') + ", kernel " + platform.release()
    except OSError:
        pass
    processor, physical = "not found", set()
    try:
        package = core = None
        for entry in Path("/proc/cpuinfo").read_text(encoding="utf-8", errors="replace").splitlines():
            key, _, value = (part.strip() for part in entry.partition(":"))
            if key == "model name" and processor == "not found":
                processor = value
            elif key == "physical id":
                package = value
            elif key == "core id":
                core = value
            elif not key and package is not None and core is not None:
                physical.add((package, core))
                package = core = None
        if package is not None and core is not None:
            physical.add((package, core))
    except OSError:
        pass
    total = None
    try:
        for entry in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            if entry.startswith("MemTotal:"):
                total = int(entry.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    mains = None
    for supply in sorted(Path("/sys/class/power_supply").glob("*")) if Path("/sys/class/power_supply").is_dir() else []:
        try:
            if (supply / "type").read_text().strip() == "Mains":
                mains = bool(mains) or (supply / "online").read_text().strip() == "1"
        except OSError:
            pass
    return system, processor, len(physical) or None, total, mains


# ---- Toolchain ------------------------------------------------------------------------------------------------------

def cmake_cache():
    """The entries of build/CMakeCache.txt: name -> value."""
    entries = {}
    try:
        for entry in (BUILD / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace").splitlines():
            match = re.match(r"^([A-Za-z0-9_]+):[A-Z]+=(.*)$", entry)
            if match:
                entries[match.group(1)] = match.group(2)
    except OSError:
        pass
    return entries


def tool_version(cache, cache_name, name, *arguments):
    """Output of the tool named by the CMake cache of build/, otherwise of the one found on the path."""
    recorded = cache.get(cache_name, "")
    if recorded and Path(recorded).is_file():
        output = run(recorded, *arguments)
        if output:
            return output, "the one of build/"
    output = run(name, *arguments)
    return (output, "the one of the path") if output else (None, None)


def compilers_of_build():
    """Compilers identified by CMake when build/ was configured: 'C MSVC 19.x; C++ MSVC 19.x; CUDA NVIDIA 13.x (host ...)'."""
    found = []
    for identified in sorted((BUILD / "CMakeFiles").glob("*/CMakeCXXCompiler.cmake")):
        for language, label in (("C", "C"), ("CXX", "C++"), ("CUDA", "CUDA")):
            try:
                text = (identified.parent / f"CMake{language}Compiler.cmake").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            def value(name, text=text):
                match = re.search(rf'^set\({name} "([^"]*)"\)', text, re.M)
                return match.group(1) if match else ""

            entry = f"{label} {value(f'CMAKE_{language}_COMPILER_ID')} {value(f'CMAKE_{language}_COMPILER_VERSION')}"
            if language == "CUDA" and value("CMAKE_CUDA_HOST_COMPILER_ID"):
                entry += f" (host {value('CMAKE_CUDA_HOST_COMPILER_ID')} {value('CMAKE_CUDA_HOST_COMPILER_VERSION')})"
            found.append(entry)
        break
    return "; ".join(found)


def visual_studio():
    vswhere = Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    return first_line(run(vswhere, "-products", "*", "-version", "[17.0,18.0)", "-property", "catalog_productDisplayVersion"))


# ---- Repositories ---------------------------------------------------------------------------------------------------

def git(folder, *arguments):
    return run("git", "-C", folder, *arguments)


def remote_address(url):
    """The address of a remote, without credentials; a local path is not recorded."""
    if not url:
        return "no remote"
    if re.match(r"^[a-z+]+://", url):
        return re.sub(r"^([a-z+]+://)[^/@]*@", r"\1", url)
    if re.match(r"^[\w.-]+@[\w.-]+:", url):  # git@host:owner/name.git
        return url
    return "local remote, not recorded"


# ---- Files ----------------------------------------------------------------------------------------------------------

def sha256(path, limit=None):
    """SHA-256 of a file, or of its first `limit` bytes."""
    digest = hashlib.sha256()
    remaining = path.stat().st_size if limit is None else limit
    with open(path, "rb") as f:
        while remaining > 0:
            block = f.read(min(1 << 22, remaining))
            if not block:
                break
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def minute(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")


def sequence(path):
    """The header of a sequence (src/sequence.h), its size and the fingerprint of its header and first frame."""
    size = path.stat().st_size
    with open(path, "rb") as f:
        header = f.read(36)
    if len(header) < 36 or header[:7] != b"TLSEQ01":
        return "not a sequence file (TLSEQ01 header expected)"
    _, n, frames, dt_frame, nu, u_rms, kmax_eta, re_lambda = struct.unpack("<8sii5f", header)
    fingerprint = sha256(path, min(size, SEQ_HEADER_BYTES + 3 * n ** 3 * 4))
    return (f"{n}^3, {frames} frames, dt_frame {dt_frame:.6g}, nu {nu:.6g}, u' {u_rms:.6g}, kmax*eta {kmax_eta:.5g}, "
            f"Re_lambda {re_lambda:.5g}, {size / 2 ** 30:.1f} GiB, SHA-256 {fingerprint}")


# ---- Record ---------------------------------------------------------------------------------------------------------

def record(with_data):
    out = [f"# Measurement environment, recorded on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}", ""]

    system, processor, cores, memory, mains = machine_windows() if WINDOWS else machine_linux()
    threads = os.cpu_count()
    out += ["## Machine",
            line("System", system),
            line("Processor", f"{processor} ({cores if cores else 'unknown number of'} cores, {threads} threads)"),
            line("Memory", f"{memory / 2 ** 30:.0f} GiB" if memory else "not found"),
            ""]

    gpu = run("nvidia-smi", "--query-gpu=name,driver_version,memory.total,vbios_version", "--format=csv,noheader")
    out += ["## GPU",
            line("Model, driver, memory, BIOS", " | ".join(gpu.splitlines()) if gpu else "nvidia-smi not found"),
            line("Power supply", "not recorded" if mains is None else "mains" if mains
                 else "battery: the clocks of the GPU may be throttled"),
            ""]

    cache = cmake_cache()
    nvcc, nvcc_origin = tool_version(cache, "CMAKE_CUDA_COMPILER", "nvcc", "--version")
    nvcc = next((entry.strip() for entry in (nvcc or "").splitlines() if "release" in entry), None)
    cmake, cmake_origin = tool_version(cache, "CMAKE_COMMAND", "cmake", "--version")
    ninja, ninja_origin = tool_version(cache, "CMAKE_MAKE_PROGRAM", "ninja", "--version")
    try:
        import numpy
        numpy_version = numpy.__version__
    except ImportError:
        numpy_version = "absent"
    out += ["## Toolchain",
            line("nvcc", f"{nvcc} ({nvcc_origin})" if nvcc else "not found"),
            line("Compilers of build/", compilers_of_build() or "build/ is not configured")]
    if WINDOWS:
        out.append(line("Visual Studio", visual_studio() or "not found"))
    else:
        out.append(line("GCC", first_line(run("gcc", "--version")) or "not found"))
    out += [line("CMake", f"{first_line(cmake).replace('cmake version ', '')} ({cmake_origin})" if cmake else "not found"),
            line("Ninja", f"{first_line(ninja)} ({ninja_origin})" if ninja else "not found"),
            line("Python", platform.python_version()),
            line("NumPy", numpy_version),
            ""]

    out.append("## Repository")
    if git(ROOT, "rev-parse", "--is-inside-work-tree") == "true":
        changed = [entry for entry in (git(ROOT, "status", "--porcelain", "--", ".") or "").splitlines() if entry.strip()]
        out += [line("Commit", git(ROOT, "log", "-1", "--format=%h %ad %s", "--date=short") or "not found"),
                line("Branch", git(ROOT, "branch", "--show-current") or "none (detached commit)"),
                line("Working tree", f"{len(changed)} file(s) modified or untracked: the binaries may not match the commit"
                     if changed else "clean")]
    else:
        out.append(line("Commit", "not a git checkout, or git not found"))
    out.append("")

    out.append("## Comparison compressors (third_party/: the revisions checked out)")
    folders = sorted((p for p in (ROOT / "third_party").glob("*") if p.is_dir()), key=lambda p: p.name.lower())
    for folder in folders:
        revision = git(folder, "rev-parse", "HEAD") if (folder / ".git").exists() else None
        if not revision:
            out.append(line(folder.name, "not a git checkout: revision unknown"))
            continue
        date = git(folder, "log", "-1", "--format=%ad", "--date=short") or "date unknown"
        modified = [entry for entry in (git(folder, "status", "--porcelain", "--untracked-files=no") or "").splitlines() if entry.strip()]
        state = f"{len(modified)} tracked file(s) modified" if modified else "tracked files unmodified"
        out.append(line(folder.name, f"{revision} {date}  {remote_address(git(folder, 'remote', 'get-url', 'origin'))}  ({state})"))
    if not folders:
        out.append("third_party/ is empty or absent")
    out.append("")

    out.append("## Binaries (build/)")
    for name in BINARIES:
        path = BUILD / name
        out.append(line(f"build/{name}", f"{minute(path.stat().st_mtime)}  SHA-256 {sha256(path)}" if path.is_file() else "absent"))

    if with_data:
        out += ["", "## Sequences (data/, not versioned: they can be produced again by dns)",
                "The fingerprint covers the header and the first frame, not the whole file."]
        sequences = sorted((ROOT / "data").rglob("*.seq"))
        for path in sequences:
            out.append(line(path.relative_to(ROOT).as_posix(), sequence(path)))
        if not sequences:
            out.append("no sequence in data/")
    return scrub("\n".join(out) + "\n")


def scrub(text):
    """Last guard: no absolute path of this machine, no user name, no machine name in what is recorded."""
    home = Path.home()
    for path, replacement in ((str(ROOT), "."), (ROOT.as_posix(), "."), (str(home), "[home]"), (home.as_posix(), "[home]")):
        text = text.replace(path, replacement)
    names = {platform.node(), os.environ.get("USERNAME", ""), os.environ.get("USER", "")}
    try:
        names.add(getpass.getuser())
    except (OSError, KeyError, ImportError):
        pass
    for name in names:
        if len(name) >= 3:
            text = re.sub(rf"(?i)(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])", "[removed]", text)
    return text


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-data", action="store_true", help="do not record the sequences of data/")
    parser.add_argument("--print", action="store_true", dest="print_only", help="print the record without writing it")
    arguments = parser.parse_args()
    text = record(not arguments.no_data)
    sys.stdout.write(text)
    if not arguments.print_only:
        with open(OUTPUT, "w", newline="\n", encoding="utf-8") as f:
            f.write(text)
        print(f"\nwritten: {OUTPUT.relative_to(ROOT).as_posix()}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
