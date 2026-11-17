# SPDX-License-Identifier: Apache-2.0
"""Claim REF-1 (docs/CLAIMS.md): the reference reader on CPU, psi_ref, written from docs/FORMAT.md without reading the
sources of the codec, reads back the twelve reference files.

    python benchmarks/independent_readers/cpp/reference_cpp.py [--no-gpu] [--files FOLDER] [--cases FOLDER] [--output FOLDER]

What is measured.

  Step 1, without a GPU: for each frame of each reference file, the state fingerprint recomputed by psi_ref is compared
          with that of the frame table of the file.
  Step 2: at the frame of the probe, the sampling lattice and the fine level rebuilt by psi_ref are compared with those
          recorded by `psi_stream pack --probe`, and its velocity field with that of the probe (points off the grid).
          `psi_stream play --probe` gives the difference between the GPU reader and this same probe: the sum of the two
          differences bounds the difference between the CPU reader and the GPU reader.

    psi_ref F.psi --csv --probe F.probe
    psi_stream play --file F.psi --particles 4096 --probe F.probe

  The ten reference files with a bound have their probe beside them, written at the encoding by
  benchmarks/compression/reference_files/reference_files.py. The two files without a bound (z_t512_d2, z_t512_d3) have
  none: the script encodes them again with the command of reference_files.py plus --probe,

    psi_stream pack --data data/turb512.seq --file W/z.psi --probe W/z.probe --csv --no-seal --degree D --start 0 --frames 55

  in W = build/work/independent_readers/cpp/, accepts the probe only if the re-encoded file is identical, byte for byte,
  to the reference file (SHA-256), keeps the probe with the SHA-256 beside it and deletes the re-encoded file. A probe
  kept by an earlier run is used again if the SHA-256 beside it is still that of the reference file.

  Second part, coverage at 64^3: every .psi file of the folder of the test cases (written by tests/test_codec.py),
  valid or altered, goes through the two readers; the outcome of each is recorded:
      read       exit code 0 and, for psi_ref, the line "CSV,ref"; for play, the line "Integrity" with no differing frame
      counted    exit code 1, the same line, at least one frame whose fingerprint or state differs
      refused    exit code 1 and a message "Failure"
      anomaly (code N)   anything else
  psi_ref gets the probe of the file when there is one beside it (same name, extension .probe); play is run without a
  probe (`psi_stream play --file F.psi --particles 4096`). The two readers agree on a file when they have the same
  outcome and the same number of differing frames.

Options.
    --files FOLDER    folder of the twelve reference files, from the root (default: build/work/reference_files)
    --cases FOLDER    folder of the test cases at 64^3, from the root (default: build/tests/cases)
    --no-gpu          step 1 only, plus the comparison with the probes already present: psi_stream is not run
    --output FOLDER   where the logs are written (default: the folder of this script)

What is written.
    reference_cpp.csv        one line per reference file: grid, degree, version, frames of the file, frames read,
                             fingerprints that differ, escapes, overflows, frame and points of the probe, nodes of the
                             lattice and of the fine level that differ from those of the encoder, largest difference
                             between the CPU reader and the probe, between the GPU reader and the probe, their sum
                             (all three as a fraction of u'), frames that differ for the GPU reader, exit code of
                             psi_ref, origin of the probe
    coverage_64.csv          one line per test file at 64^3: size, outcome and differing frames of each reader,
                             agreement, and with a probe the difference of the CPU reader and the differing nodes
    logs/reference_cpp.log   what the script printed: SHA-256 of the programs, state of the GPU, output of psi_ref for
                             each reference file, the lines "Probe", "Integrity" and "Failure" of play, the summary
                             of the coverage
    logs/psi_ref_<file>.log, logs/play_<file>.log, logs/pack_<file>.log
                             raw output of each program run on a reference file
    logs/coverage_64.log     raw output of every run of the second part, each under a line "== program file: exit code"
No time is read: only bytes, fingerprints, differences and exit codes. The GPU may be shared.

The "Probe" line of play is in velocity units; the script divides it by u', read in the header of the file (float32
at byte 48, docs/FORMAT.md, section 2). The verdict of REF-1 is given by tools/claims.py, which reads reference_cpp.csv.

Exit code: 0 if the twelve files are read by psi_ref with exit code 0 and every expected line was found; 1 if a
reference file is missing, if psi_ref ends with another exit code on one of them, if a line that a figure comes from
is missing or has another shape, or if the second part could not be done (the list is printed at the end); 2 if
psi_ref is missing (nothing was run).

Needs build/psi_ref and, unless --no-gpu, build/psi_stream and an NVIDIA GPU (tools/build.ps1 -Tools on Windows, CMake
with -DPSIT_BUILD_TOOLS=ON elsewhere; this script compiles nothing); the twelve reference files and their probes
(benchmarks/compression/reference_files/reference_files.py); data/turb512.seq for the probes of the two files without
a bound; the test cases (python tests/test_codec.py). Python 3.8 or later, standard library only. Windows and Linux.
"""

import argparse
import collections
import csv
import re
import struct
import subprocess
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent / "compression"))
import psi_stream_runs as ps  # noqa: E402

ROOT = ps.ROOT
PSI_REF = "build/psi_ref" + ps.EXE  # as written in the logs; the program is started by its absolute path
SOURCE = "tools/reference_reader.cpp"
WORK = "build/work/independent_readers/cpp"
PARTICLES = "4096"

D2 = ["--degree", "2", "--start", "0", "--frames", "55"]
D3 = ["--degree", "3", "--start", "0", "--frames", "55"]
# (file, options of pack that give it back, sequence): the last two only for the files that have no probe beside them.
FILES = [
    ("f_t256_d3_t05", None, None), ("f_t256_d3_t04", None, None), ("f_t256_d3_t03", None, None),
    ("f_t512_d3_t05", None, None), ("f_t512_d3_t04", None, None), ("f_t512_d3_t03", None, None),
    ("f_t256_d2_t06", None, None), ("f_t256_d2_t05", None, None), ("f_t512_d2_t07", None, None), ("f_t512_d2_t06", None, None),
    ("z_t512_d2", D2, "data/turb512.seq"), ("z_t512_d3", D3, "data/turb512.seq"),
]

COLUMNS = ["file", "n", "degree", "version", "frames", "frames_read", "fingerprints_differing", "escapes", "overflows", "probe_frame",
           "probe_points", "lattice_nodes_differing", "fine_nodes_differing", "cpu_probe_difference_over_u_rms",
           "gpu_probe_difference_over_u_rms", "cpu_gpu_bound_over_u_rms", "gpu_frames_differing", "cpu_exit_code", "probe_origin"]
COVERAGE_COLUMNS = ["file", "bytes", "cpu_outcome", "cpu_frames_differing", "gpu_outcome", "gpu_frames_differing", "agreement",
                    "cpu_probe_difference_over_u_rms", "lattice_nodes_differing", "fine_nodes_differing"]

# Lines of psi_ref that the script reads. tools/reference_reader.cpp, end of main (only with --csv):
#   std::printf("CSV,ref,%d,%d,%d,%d,%d,%d,%d,%llu,%llu\n", e.n, e.degree, e.version, to_read, e.n_frames, f.sealed ? 1 : 0, differing,
#               (unsigned long long)c.escapes, (unsigned long long)c.overflows);
#     fields after "CSV,ref": 0 n   1 degree   2 version   3 frames read   4 frames of the file   5 seal   6 fingerprints that
#     differ   7 escapes   8 overflows of level 0
#   std::printf("CSV,probe,%u,%u,%llu,%llu,%.4e,%.4e,%d\n", s.frame, s.points, (unsigned long long)report.lattice_nodes,
#               (unsigned long long)report.fine_nodes, report.max_difference / double(e.u_rms), report.rms_difference / double(e.u_rms), direct ? 1 : 0);
#     (only with --probe)   0 frame, counted in the source sequence   1 points   2 nodes of the lattice that differ   3 nodes of the
#     fine level that differ   4 largest difference of the field, as a fraction of u'   5 RMS difference   6 lattice read directly
CSV_FIELDS = {"ref": 9, "probe": 7}
# Standard error of psi_ref. tools/reference_reader.cpp, the two catch clauses at the end of main (exit code 1):
#   std::fprintf(stderr, "Failure: %s\n", x.what());
CPU_FAILURE = re.compile(r"^Failure", re.M)
# Lines of play that the script reads: ps.INTEGRITY and ps.PROBE (tools/psi_stream.cu, quoted in psi_stream_runs.py), and
# the messages "Failure: ..." of the macro CHK (src/codec_kernels.cuh) and of sequence_fail (src/sequence.h), on standard error.
SHOWN_LINES_OF_PLAY = re.compile(r"^(Probe|Integrity|Failure)")

console = []


def say(text=""):
    console.append(text)
    print(text, flush=True)


def sci(x):
    return f"{x:.3e}"


def u_rms(file):
    """u': float32 at byte 48 of the header (docs/FORMAT.md, section 2)."""
    with open(ROOT / file, "rb") as f:
        return float(struct.unpack_from("<f", f.read(208), 48)[0])


def csv_fields(output, kind):
    """Fields of the first line "CSV,<kind>," of an output of psi_ref, without "CSV" and the kind; None if there is no such
    line or if it does not have the expected number of fields."""
    for line in output.split("\n"):
        if line.startswith(f"CSV,{kind},"):
            fields = line.split(",")[2:]
            return fields if len(fields) == CSV_FIELDS[kind] else None
    return None


def read_cpu(file, probe):
    """Runs psi_ref on a file, from the root, and reads its outcome."""
    arguments = [file, "--csv"] + (["--probe", probe] if probe else [])
    p = subprocess.run([str(ROOT / PSI_REF)] + arguments, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = (stream.decode("utf-8", "replace").replace("\r", "") for stream in (p.stdout, p.stderr))
    r = types.SimpleNamespace(code=p.returncode, command=" ".join(["psi_ref"] + arguments), output=output, error=error,
                              csv=csv_fields(output, "ref"), probe=csv_fields(output, "probe"), differing="")
    r.text = "".join(part if part.endswith("\n") or not part else part + "\n" for part in (output, error))
    if r.csv:
        r.differing = int(r.csv[6])
    if r.code == 0 and r.csv:
        r.outcome = "read"
    elif r.code == 1 and r.csv and r.differing > 0:
        r.outcome = "counted"
    elif r.code == 1 and CPU_FAILURE.search(r.text):
        r.outcome = "refused"
    else:
        r.outcome = f"anomaly (code {r.code})"
    return r


def read_gpu(file, probe, log=None):
    """Runs `psi_stream play` on a file and reads its outcome. difference: the largest difference at the probe, in
    velocity units, or None if play does not give it."""
    r = ps.run(["play", "--file", file, "--particles", PARTICLES] + (["--probe", probe] if probe else []), log)
    integrity = ps.INTEGRITY.search(r.output)
    at_probe = ps.PROBE.search(r.output)
    r.differing = int(integrity.group(1)) if integrity else ""
    r.frame_not_read = bool(at_probe and at_probe.group(4))
    r.difference = float(at_probe.group(3)) if at_probe and not r.frame_not_read else None
    if r.code == 0 and integrity and r.differing == 0:
        r.outcome = "read"
    elif r.code == 1 and integrity and r.differing > 0:
        r.outcome = "counted"
    elif r.code == 1 and "Failure" in r.text():
        r.outcome = "refused"
    else:
        r.outcome = f"anomaly (code {r.code})"
    return r


def probe_of(name, folder, pack, sequence, gpu, logs):
    """(probe of a reference file or None, where it comes from)."""
    file, beside = f"{folder}/{name}.psi", f"{folder}/{name}.probe"
    if (ROOT / beside).is_file():
        return beside, "written at encoding"
    if pack is None:
        return None, "absent"
    # File without a probe: encoded again with --probe, accepted only if this gives back the reference file byte for byte.
    probe, witness, copy = f"{WORK}/{name}.probe", ROOT / WORK / f"{name}.sha256", f"{WORK}/{name}.psi"
    fingerprint = ps.sha256(ROOT / file)
    if (ROOT / probe).is_file() and witness.is_file() and witness.read_text(encoding="utf-8").strip() == fingerprint:
        return probe, "identical re-encoding (SHA-256), already done"
    if not gpu:
        return None, "absent (re-encoding not done: no GPU)"
    if not (ROOT / sequence).is_file():
        return None, f"absent (sequence {sequence} not found)"
    (ROOT / WORK).mkdir(parents=True, exist_ok=True)
    (ROOT / copy).unlink(missing_ok=True)  # a file left by an interrupted run must not be compared in place of this one
    r = ps.run(["pack", "--data", sequence, "--file", copy, "--probe", probe, "--csv", "--no-seal"] + pack, logs / f"pack_{name}.log")
    identical = (ROOT / copy).is_file() and ps.sha256(ROOT / copy) == fingerprint
    (ROOT / copy).unlink(missing_ok=True)
    if r.code != 0 or not identical:
        (ROOT / probe).unlink(missing_ok=True)
        witness.unlink(missing_ok=True)
        return None, f"absent (re-encoding: exit code {r.code}, file {'identical' if identical else 'DIFFERENT from the reference file'})"
    witness.write_text(fingerprint + "\n", encoding="utf-8", newline="\n")
    return probe, "identical re-encoding (SHA-256)"


def main():
    parser = argparse.ArgumentParser(description="Claim REF-1: the reference reader on CPU against the twelve reference files and the test files at 64^3.")
    parser.add_argument("--files", default="build/work/reference_files", help="folder of the twelve reference files, from the root")
    parser.add_argument("--cases", default="build/tests/cases", help="folder of the test cases at 64^3, from the root")
    parser.add_argument("--no-gpu", action="store_true", help="step 1 only, plus the comparison with the probes already present: psi_stream is not run")
    parser.add_argument("--output", default=str(HERE), help="folder where the logs are written (default: the folder of this script)")
    arguments = parser.parse_args()

    ps.require([PSI_REF], "Build the instruments first (tools/build.ps1 -Tools): this script compiles nothing.")
    folder, cases = ps.relative(arguments.files), ps.relative(arguments.cases)
    output = Path(arguments.output).resolve()
    logs = output / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    problems = []

    gpu = not arguments.no_gpu
    if gpu and not (ROOT / ps.PSI_STREAM).is_file():
        say(f"{ps.PSI_STREAM} not found: step 2 limited to the comparison with the probes.")
        gpu = False
    say("Claim REF-1: reference reader on CPU against the twelve reference files")
    say(f"psi_ref    : {PSI_REF}, SHA-256 {ps.sha256(ROOT / PSI_REF)[:16]} (first 16 digits); source {SOURCE}, SHA-256 "
        f"{ps.sha256(ROOT / SOURCE)[:16]} (first 16 digits)")
    if gpu:
        say(f"psi_stream : {ps.PSI_STREAM}, SHA-256 {ps.sha256(ROOT / ps.PSI_STREAM)[:16]} (first 16 digits)")
        say(f"GPU at the start: {ps.gpu_line(ps.gpu_state())}")
    else:
        say("psi_stream : not run")
    say()

    # ---- First part: the twelve reference files ----
    with open(output / "reference_cpp.csv", "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        writer.writerow(COLUMNS)
        for name, pack, sequence in FILES:
            file = f"{folder}/{name}.psi"
            if not (ROOT / file).is_file():
                problems.append(f"{file}: not found")
                say(f"== {name}.psi: NOT FOUND")
                writer.writerow([f"{name}.psi"] + [""] * (len(COLUMNS) - 2) + ["not found"])
                continue
            probe, origin = probe_of(name, folder, pack, sequence, gpu, logs)
            if gpu and probe is None:
                problems.append(f"{file}: no probe ({origin}): step 2 was not done for this file")
            cpu = read_cpu(file, probe)
            (logs / f"psi_ref_{name}.log").write_text(cpu.text, encoding="utf-8", newline="\n")
            if cpu.code != 0:
                problems.append(f"{file}: psi_ref ended with exit code {cpu.code} (logs/psi_ref_{name}.log)")
            elif cpu.csv is None or (probe and cpu.probe is None):
                problems.append(f'{file}: line "CSV,ref" or "CSV,probe" missing from the output of psi_ref, or of another shape (logs/psi_ref_{name}.log)')
            say(f"== {name}.psi ({(ROOT / file).stat().st_size} bytes); probe: {origin}")
            for line in cpu.text.splitlines():
                say(f"    {line}")
            say(f"    psi_ref: exit code {cpu.code}")
            c, s = cpu.csv, cpu.probe
            cpu_difference = gpu_difference = bound = None  # None: not measured (a difference of zero is a measurement)
            gpu_differing = ""
            if s:
                cpu_difference = float(s[4])
            if gpu and probe:
                g = read_gpu(file, probe, logs / f"play_{name}.log")
                for line in g.text().splitlines():
                    if SHOWN_LINES_OF_PLAY.match(line):
                        say(f"    psi_stream play: {line}")
                say(f"    psi_stream play: exit code {g.code}")
                gpu_differing = g.differing
                if g.difference is not None:
                    gpu_difference = g.difference / u_rms(file)  # the "Probe" line of play is in velocity units
                if g.outcome not in ("read", "counted") or g.difference is None:
                    problems.append(f"{file}: play gave no difference at the probe (outcome: {g.outcome}"
                                    + (", probed frame not read" if g.frame_not_read else "") + f"; logs/play_{name}.log)")
            if cpu_difference is not None and gpu_difference is not None:
                bound = cpu_difference + gpu_difference
            row = [f"{name}.psi"]
            row += [c[0], c[1], c[2], c[4], c[3], c[6], c[7], c[8]] if c else [""] * 8
            row += [s[0], s[1], s[2], s[3]] if s else [""] * 4
            row += ["" if x is None else sci(x) for x in (cpu_difference, gpu_difference, bound)]
            row += [gpu_differing, cpu.code, origin]
            writer.writerow(row)
            fo.flush()
            say()

    # ---- Second part: coverage at 64^3 ----
    files = sorted(f.name for f in (ROOT / cases).glob("*.psi") if f.is_file())
    if files:
        say(f"Coverage at 64^3: files of {cases}/ (tests/test_codec.py), read by " + ("the two readers" if gpu else "the CPU reader only"))
        counts, disagreements, worst, with_probe = collections.Counter(), 0, 0.0, 0
        with open(output / "coverage_64.csv", "w", newline="", encoding="utf-8") as fo, \
             open(logs / "coverage_64.log", "w", encoding="utf-8", newline="\n") as raw:
            writer = csv.writer(fo, lineterminator="\n")
            writer.writerow(COVERAGE_COLUMNS)
            for name in files:
                file, probe = f"{cases}/{name}", f"{cases}/{name[:-4]}.probe"
                if not (ROOT / probe).is_file():
                    probe = None
                cpu = read_cpu(file, probe)
                raw.write(f"== psi_ref {name}: exit code {cpu.code}, {cpu.outcome}\n{cpu.text}")
                gpu_outcome = gpu_differing = agreement = ""
                if gpu:
                    g = read_gpu(file, None)
                    raw.write(f"== psi_stream play {name}: exit code {g.code}, {g.outcome}\n{g.text()}")
                    gpu_outcome, gpu_differing = g.outcome, g.differing
                    agreement = "yes" if cpu.outcome == g.outcome and str(cpu.differing) == str(g.differing) else "no"
                    if agreement == "no":
                        disagreements += 1
                        say(f"    disagreement on {name}: CPU {cpu.outcome} {cpu.differing}, GPU {g.outcome} {g.differing}")
                raw.flush()
                difference, lattice_nodes, fine_nodes = None, "", ""
                if cpu.probe:
                    difference, lattice_nodes, fine_nodes = float(cpu.probe[4]), cpu.probe[2], cpu.probe[3]
                    worst = max(worst, difference)
                    with_probe += 1
                counts[cpu.outcome] += 1
                writer.writerow([name, (ROOT / file).stat().st_size, cpu.outcome, cpu.differing, gpu_outcome, gpu_differing, agreement,
                                 "" if difference is None else sci(difference), lattice_nodes, fine_nodes])
                fo.flush()
        say(f"    {len(files)} files; outcomes of the CPU reader: " + ", ".join(f"{counts[o]} {o}" for o in sorted(counts)))
        if gpu:
            say(f"    disagreements of outcome or of number of differing frames between the CPU reader and the GPU reader: {disagreements}")
        say(f"    probes at 64^3: {with_probe}, largest difference of the CPU reader {sci(worst)} u'")
    else:
        problems.append(f"no .psi file in {cases}/: the coverage at 64^3 was not done (run python tests/test_codec.py)")
        say(f"Coverage at 64^3: no .psi file in {cases}/ (run python tests/test_codec.py), part not done")

    if gpu:
        say(f"GPU at the end: {ps.gpu_line(ps.gpu_state())}")
    (logs / "reference_cpp.log").write_text("\n".join(console) + "\n", encoding="utf-8", newline="\n")
    if problems:
        print(f"{len(problems)} PROBLEM(S): the log is incomplete or cannot be trusted for these files:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
