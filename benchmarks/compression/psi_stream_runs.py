# SPDX-License-Identifier: Apache-2.0
"""Runs of build/psi_stream and reading of what it prints: shared by the measurement scripts of benchmarks/compression/,
benchmarks/speed/ and benchmarks/robustness/.

This module measures nothing by itself and is not run directly. It holds, in one place:

  - how psi_stream is run: from the root of the repository, with paths relative to the root and forward slashes, so
    that no log names a folder of the machine. Standard output and standard error are captured apart; a log file holds
    the first followed by the second, with nothing changed but the line ends (LF);
  - every pattern that the scripts match in the output of psi_stream, each with the source line that prints it. A
    pattern that finds nothing raises UnexpectedOutput: the scripts then report it and exit with a non-zero code,
    instead of writing an empty figure;
  - pack_then_play, the encoding of one file followed by its read-back, with the figures that the claims rest on;
  - the state of the GPU as nvidia-smi gives it, for the scripts whose times need a free GPU.

Python 3.8 or later, standard library only. Windows and Linux.
"""

import hashlib
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
EXE = ".exe" if sys.platform == "win32" else ""
PSI_STREAM = "build/psi_stream" + EXE  # as written in the logs; the program is started by its absolute path


class UnexpectedOutput(Exception):
    """A line that a measurement relies on is missing from the output of a program, or does not have the expected shape."""


# ---- Patterns, each with the source line that prints the text it matches --------------------------------------------

# tools/psi_stream.cu, play<D>, last lines of the report:
#   printf("Integrity      : %d frame(s) whose state differs from that of the sender; fingerprint of the final positions %016llx\n", bad_hash, pos_hash);
INTEGRITY = re.compile(r"^Integrity\s*:\s*(\d+) frame\(s\) whose state differs from that of the sender; fingerprint of the final positions ([0-9a-f]{16})$", re.M)

# tools/psi_stream.cu, play<D> (only with --probe):
#   printf("Probe          : frame %d, %u points; largest difference between the field of the reader and that of the encoder: %.3e%s\n",
#          probe_frame + sh.first_frame, probe_count, probe_diff, probe_diff < 0.0 ? " (frame not read)" : "");
PROBE = re.compile(r"^Probe\s*:\s*frame (\d+), (\d+) points; largest difference between the field of the reader and that of the encoder: "
                   r"(-?[0-9.]+e[+-]\d+)( \(frame not read\))?$", re.M)

# tools/psi_stream.cu, play<D>:
#   printf("GPU memory     : %.1f MiB allocated by the reader, of which %.1f MiB of receiver state; lattice at %p (remainder modulo 2 MiB: %llu KiB)\n", ...);
GPU_MEMORY = re.compile(r"^GPU memory\s*:\s*([0-9.]+) MiB allocated by the reader, of which ([0-9.]+) MiB of receiver state; ", re.M)

# tools/playback_kernels.cuh, print_seal, called by play<D>:
#   printf("Seal           : present and verified at opening: header%s and frame table unchanged since writing\n", ...);
#   printf("Seal           : absent (file from before the seal, or pack --no-seal): the header and the frame table are covered by no fingerprint\n");
SEAL = re.compile(r"^Seal\s*:\s*(present and verified at opening|absent) ?[:(]", re.M)

# Standard error. src/codec_kernels.cuh, macro CHK (every refusal of the encoder and of the reader; exit code 1):
#   fprintf(stderr, "Failure: %s\n  %s:%d\n", msg, source_name(__FILE__), __LINE__);
# src/sequence.h, sequence_fail (a sequence file that cannot be read; exit code 1):
#   fprintf(stderr, "Failure: %s\n", msg);
# src/codec_stream.cuh, end of pack<D>(o, src) (the file is written; exit code 4 for the first message, 3 for the second):
#   fprintf(stderr, "Failure: bound not held. Largest error %.6f u' for a certification threshold of %.6f u' (tolerance %.4f u', margin %.1e u'), on %u frame(s). The file is written.\n", ...);
#   fprintf(stderr, "Failure: %lld coefficient(s) clipped. The file is written, but it does not reproduce the requested projection.\n  ...", clip_all);
FAILURE = re.compile(r"^Failure: .*$", re.M)

# Standard error. src/codec_kernels.cuh, macro CK (a CUDA call that fails; exit code 1):
#   fprintf(stderr, "CUDA error: %s\n  %s:%d\n", cudaGetErrorString(e_), source_name(__FILE__), __LINE__);
CUDA_ERROR = re.compile(r"^CUDA error: .*$", re.M)

# The "CSV," lines, printed only with --csv. Number of fields after the kind, and the fields that the scripts read.
#
# src/codec_stream.cuh, end of pack<D>(o, src):
#   printf("CSV,pack,%d,%d,%d,%d,%.4g,%d,%.4f,%.4f,%.2f,%.4f,%.4f,%.3f,%.4e,%.4e,%lld,%.5f,%lld,%lld,%lld,%lld,%lld,%lld\n",
#          n, D, g.L, o.frames, o.eps, o.key_interval, pos / mib, raw, raw / (pos / mib), <MiB per key>, <MiB per innovation>,
#          1e3 * t_enc / o.frames, err_n ? err_sum / err_n : 0.0, err_max, snd.clipped(), max_worst, <six clipping counts>);
#     0 n   1 degree   3 frames   8 raw bytes / file bytes   11 encoding, ms per frame   12 relative field error, mean over the checked
#     frames   13 the same, worst frame   15 largest error of a component, as a fraction of u'
#   printf("CSV,fine,%d,%d,%d,%.5g,%.5g,%.4f,%.6f,%.5f,%.6f,%u,%u,%u,%d,%.2f\n",
#          n, D, o.frames, o.tolerance, fine_eps, <MiB per frame>, <fine bytes / other bytes>, max_worst, <fraction of nodes>, max_bricks,
#          max_nodes, frames_over, max_passes, 1e3 * t_fine / o.frames);          (only with --tolerance)
#     3 tolerance, as a fraction of u'   11 frames above the certification threshold
#   printf("CSV,passes,%d,%d,%d,%d,%d,%d,%.3g,%.6f,%d,%u\n",
#          n, D, o.frames, o.fine_passes, max_passes, frames_capped, margin, certified, frames_floor, floor_voxels);   (only with --tolerance)
#     7 certification threshold, as a fraction of u'
# tools/psi_stream.cu, end of play<D>:
#   printf("CSV,play,%d,%d,%u,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.1f,%.1f,%d,%016llx,%d,%.4f,%.4f,%.4f,%.4f\n",
#          sh.n, D, np, o.substeps, o.sort0 ? 1 : 0, o.resort, <MiB read per frame>, disk_ms, bus_ms, dec_ms, sort_ms, step_ms, ray_ms,
#          frame_ms, 1e3 * wall / nf, <MiB allocated>, <MiB of receiver state>, bad_hash, pos_hash, n_key_read, <MiB per key>,
#          <disk, ms per key>, <bus, ms per key>, <decoding, ms per key>);
#     7 disk   8 bus   9 decoding   11 one RK4 step   13 full budget of a frame   14 wall clock per frame (all in ms)
#     17 frames whose state differs   18 fingerprint of the final positions   19 key frames read in steady state   23 decoding of a key frame, ms
CSV_FIELDS = {"pack": 22, "fine": 14, "passes": 10, "play": 24}


def csv_line(output, kind):
    """The fields of the line "CSV,<kind>,..." of an output, without "CSV" and the kind. Exactly one such line is expected."""
    lines = [line for line in output.split("\n") if line.startswith(f"CSV,{kind},")]
    if len(lines) != 1:
        raise UnexpectedOutput(f'{len(lines)} line(s) "CSV,{kind}," in the output, 1 expected')
    fields = lines[0].split(",")[2:]
    if len(fields) != CSV_FIELDS[kind]:
        raise UnexpectedOutput(f'the line "CSV,{kind}," has {len(fields)} fields, {CSV_FIELDS[kind]} expected: {lines[0]}')
    return fields


def integrity(output):
    """(frames whose state differs, fingerprint of the final positions) from the "Integrity" line of play."""
    m = INTEGRITY.search(output)
    if not m:
        raise UnexpectedOutput('line "Integrity      : N frame(s) whose state differs ..." missing from the output of play')
    return int(m.group(1)), m.group(2)


def probe(output):
    """(frame, points, largest difference, as printed) from the "Probe" line of play. The probed frame must have been read."""
    m = PROBE.search(output)
    if not m:
        raise UnexpectedOutput('line "Probe          : frame N, N points; largest difference ..." missing from the output of play')
    if m.group(4):
        raise UnexpectedOutput(f"the probed frame was not read by play: {m.group(0)}")
    return m.group(1), m.group(2), m.group(3)


def gpu_memory(output):
    """(MiB allocated by the reader, MiB of receiver state), as printed, from the "GPU memory" line of play."""
    m = GPU_MEMORY.search(output)
    if not m:
        raise UnexpectedOutput('line "GPU memory     : N MiB allocated by the reader, of which N MiB of receiver state" missing from the output of play')
    return m.group(1), m.group(2)


def seal(output):
    """"present" or "absent": what play says of the seal of the file it has opened."""
    m = SEAL.search(output)
    if not m:
        raise UnexpectedOutput('line "Seal           : present and verified ..." or "Seal           : absent ..." missing from the output of play')
    return "present" if m.group(1).startswith("present") else "absent"


def failure(error):
    """The first line "Failure: ..." or "CUDA error: ..." of a standard error, or "" if there is none."""
    found = [m for m in (FAILURE.search(error), CUDA_ERROR.search(error)) if m]
    return min(found, key=lambda m: m.start()).group(0).strip() if found else ""


def outcome(run):
    """Outcome of one play, as benchmarks/robustness/ records it. In this order:
      hang                              stopped after the watchdog timeout;
      CUDA error                        a message "CUDA error" on standard error;
      refused                           exit code 1 and a message "Failure:" on standard error;
      counted                           complete read, at least one frame whose state differs (exit code 1);
      read                              complete read, no differing frame (exit code 0);
      crash                             exit code other than 0 and 1;
      code N without an expected line   none of the above: the script that sees it exits with a non-zero code."""
    if run.hung:
        return "hang"
    if "CUDA error" in run.error:
        return "CUDA error"
    if run.code == 1 and "Failure:" in run.error:
        return "refused"
    if run.code == 1 and re.search(r"Integrity\s*:\s*[1-9]\d* frame", run.output):
        return "counted"
    if run.code == 0 and re.search(r"Integrity\s*:\s*0 frame", run.output):
        return "read"
    if run.code not in (0, 1):
        return "crash"
    return f"code {run.code} without an expected line"


# ---- Running psi_stream ---------------------------------------------------------------------------------------------

def relative(path):
    """A path under the root, as the commands and the logs write it: relative to the root, forward slashes."""
    try:
        return (ROOT / path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        stop(f"{path} is not under the root of the repository: the commands and the logs name paths relative to the root. Nothing was run.")


def stop(message, code=2):
    """Prints a message on standard error and exits (code 2: nothing was measured)."""
    print(message, file=sys.stderr, flush=True)
    sys.exit(code)


def require(paths, hint):
    """Stops with exit code 2 if a prerequisite (path relative to the root) is missing."""
    missing = [str(p) for p in paths if not (ROOT / p).exists()]
    if missing:
        stop(f"Missing: {', '.join(missing)}. {hint} Nothing was run.")


class Run:
    """One run of psi_stream, started at once. arguments: what follows the name of the program, paths relative to the
    root. After wait(): code (exit code), output and error (the two streams, as text), hung (stopped by the timeout),
    command (as the logs write it)."""

    def __init__(self, arguments):
        self.arguments = [str(a) for a in arguments]
        self.command = " ".join(["psi_stream"] + self.arguments)
        self.code, self.output, self.error, self.hung = None, "", "", False
        self._streams = (tempfile.TemporaryFile(), tempfile.TemporaryFile())  # files, not pipes: nothing to drain while it runs
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
        self._started = time.monotonic()
        self._process = subprocess.Popen([str(ROOT / PSI_STREAM)] + self.arguments, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                                         stdout=self._streams[0], stderr=self._streams[1], **options)

    def finished(self, timeout=None):
        """True once the program has exited; with a timeout in seconds, the program is stopped when it is exceeded."""
        if self._process.poll() is None:
            if timeout is None or time.monotonic() - self._started <= timeout:
                return False
            self.hung = True
            self._process.kill()
        return True

    def wait(self):
        self.code = self._process.wait()
        texts = []
        for stream in self._streams:
            stream.seek(0)
            texts.append(stream.read().decode("utf-8", "replace").replace("\r", ""))
            stream.close()
        self.output, self.error = texts
        return self

    def text(self):
        """What a log file holds: standard output, then standard error."""
        return "".join(part if part.endswith("\n") or not part else part + "\n" for part in (self.output, self.error))


def run(arguments, log=None):
    """Runs psi_stream to its end; log: file that receives its output."""
    r = Run(arguments).wait()
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(r.text(), encoding="utf-8", newline="\n")
    return r


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def option(arguments, name):
    """Value that follows an option in a list of arguments, or None."""
    arguments = [str(a) for a in arguments]
    return arguments[arguments.index(name) + 1] if name in arguments else None


# ---- One file encoded, then read back -------------------------------------------------------------------------------

COLUMNS = ["name", "pack_command", "pack_exit_code", "pack_failure", "file_bytes", "sha256", "grid", "degree", "frames", "compression_ratio",
           "encoding_ms_per_frame", "field_error_mean", "field_error_worst", "largest_error_over_u_rms", "tolerance_over_u_rms",
           "certification_threshold_over_u_rms", "frames_above_threshold", "play_command", "play_exit_code", "play_failure", "frames_differing",
           "final_positions", "probe_frame", "probe_points", "probe_largest_difference", "reader_allocated_mib", "receiver_state_mib",
           "gpu_free_at_start"]


def pack_then_play(name, pack_arguments, play_arguments, logs):
    """Encodes one file with `psi_stream pack`, reads it back with `psi_stream play` (always, as a refused encoding leaves
    a file that the reader must refuse too), and returns (row of COLUMNS, list of problems). The raw outputs go to
    logs/pack_<name>.log and logs/play_<name>.log. The figures are copied as the programs print them, never recomputed.

    What is expected of pack:
      exit code 0, 3 or 4   the file is complete (3: coefficients clipped; 4: bound missed): the line "CSV,pack" is
                            required, and with --tolerance the lines "CSV,fine" and "CSV,passes"; with 3 or 4, a
                            message "Failure:";
      exit code 1           refusal: a message "Failure:" or "CUDA error:" is required; no figure;
      anything else         a problem.
    What is expected of play:
      exit code 0           the "Integrity" line with no differing frame, the "GPU memory" line, and with --probe the
                            "Probe" line;
      exit code 1           either a refusal (message "Failure:" or "CUDA error:"), or the same lines with at least one
                            differing frame;
      anything else         a problem.
    A problem is a missing line, a line of another shape, or an exit code outside these cases."""
    row = dict.fromkeys(COLUMNS, "")
    problems = []
    file = option(pack_arguments, "--file")
    probe_file = option(pack_arguments, "--probe")
    for stale in (file, probe_file):  # a file left by an earlier run must not be read back in place of this one
        if stale is not None:
            (ROOT / stale).unlink(missing_ok=True)
    (ROOT / file).parent.mkdir(parents=True, exist_ok=True)

    p = run(pack_arguments, logs / f"pack_{name}.log")
    row.update(name=name, pack_command=p.command, pack_exit_code=p.code, pack_failure=failure(p.error))
    try:
        if p.code in (0, 3, 4):
            c = csv_line(p.output, "pack")
            row.update(grid=c[0], degree=c[1], frames=c[3], compression_ratio=c[8], encoding_ms_per_frame=c[11], field_error_mean=c[12],
                       field_error_worst=c[13], largest_error_over_u_rms=c[15])
            for asked, read, what in ((option(pack_arguments, "--degree"), c[1], "degree"), (option(pack_arguments, "--frames"), c[3], "number of frames")):
                if asked is not None and int(asked) != int(read):
                    raise UnexpectedOutput(f'the line "CSV,pack" gives a {what} of {read}, {asked} was asked: its fields are not the expected ones')
            tolerance = option(pack_arguments, "--tolerance")
            if tolerance is not None:
                fine, passes = csv_line(p.output, "fine"), csv_line(p.output, "passes")
                if float(fine[3]) != float(tolerance):
                    raise UnexpectedOutput(f'the line "CSV,fine" gives a tolerance of {fine[3]}, {tolerance} was asked: its fields are not the expected ones')
                row.update(tolerance_over_u_rms=fine[3], frames_above_threshold=fine[11], certification_threshold_over_u_rms=passes[7])
            if p.code != 0 and not row["pack_failure"]:
                raise UnexpectedOutput(f'exit code {p.code} without a message "Failure:"')
        elif p.code == 1:
            if not row["pack_failure"]:
                raise UnexpectedOutput('exit code 1 without a message "Failure:" or "CUDA error:"')
        else:
            raise UnexpectedOutput(f"exit code {p.code}: neither an encoding (0, 3, 4) nor a refusal (1)")
    except UnexpectedOutput as e:
        problems.append(f"{name}, pack: {e} (logs/pack_{name}.log)")
    if (ROOT / file).is_file():
        row.update(file_bytes=(ROOT / file).stat().st_size, sha256=sha256(ROOT / file))
    else:
        problems.append(f"{name}, pack: {file} was not written (logs/pack_{name}.log)")

    q = run(play_arguments, logs / f"play_{name}.log")
    row.update(play_command=q.command, play_exit_code=q.code, play_failure=failure(q.error))
    try:
        if q.code == 0 or (q.code == 1 and not row["play_failure"]):
            differing, positions = integrity(q.output)
            if (differing == 0) != (q.code == 0):
                raise UnexpectedOutput(f"exit code {q.code} with {differing} differing frame(s)")
            allocated, state = gpu_memory(q.output)
            row.update(frames_differing=differing, final_positions=positions, reader_allocated_mib=allocated, receiver_state_mib=state)
            if option(play_arguments, "--probe") is not None:
                frame, points, difference = probe(q.output)
                row.update(probe_frame=frame, probe_points=points, probe_largest_difference=difference)
        elif q.code != 1:
            raise UnexpectedOutput(f"exit code {q.code}: neither a read (0, or 1 with differing frames) nor a refusal (1)")
    except UnexpectedOutput as e:
        problems.append(f"{name}, play: {e} (logs/play_{name}.log)")
    return row, problems


def shown(row):
    """One line for the console."""
    line = f"{row['name']}: pack exit code {row['pack_exit_code']}, play exit code {row['play_exit_code']}, {row['file_bytes'] or 'no'} bytes"
    if row["compression_ratio"]:
        line += f"; ratio {row['compression_ratio']}, field error {row['field_error_mean']}, largest error {row['largest_error_over_u_rms']} u'"
    if row["frames_above_threshold"] != "":
        line += f", {row['frames_above_threshold']} frame(s) above the threshold of {row['certification_threshold_over_u_rms']} u'"
    if row["frames_differing"] != "":
        line += f"; {row['frames_differing']} frame(s) differing at read-back"
    if row["probe_largest_difference"]:
        line += f", probe {row['probe_largest_difference']}"
    for key in ("pack_failure", "play_failure"):
        if row[key]:
            line += f"\n    {key.split('_')[0]}: {row[key][:200]}"
    return line


# ---- State of the GPU -----------------------------------------------------------------------------------------------

BUSY_ABOVE_PERCENT = 5  # utilisation above which the GPU is taken as busy
# Programs of this repository that compute on the GPU (the Python scripts of benchmarks/ call the library and the comparison codecs).
REPOSITORY_PROGRAMS = ("psi_stream", "psi_view", "psit", "tracers", "dns", "python", "sz3", "cuszp", "zfp")


def _nvidia_smi(*arguments):
    """Lines printed by nvidia-smi, or None if it cannot be run."""
    try:
        p = subprocess.run(["nvidia-smi"] + list(arguments), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return None
    if p.returncode != 0:
        return None
    return [line.strip() for line in p.stdout.decode("utf-8", "replace").split("\n") if line.strip()]


def gpu_utilisation():
    """Utilisation of the GPU in percent (the busiest one if there are several), or None if nvidia-smi does not give it.
    nvidia-smi gives a mean over its last sampling period: just after a computation, it is the load of that computation."""
    lines = _nvidia_smi("--query-gpu=utilization.gpu", "--format=csv,noheader")
    values = [re.fullmatch(r"(\d+) %", line) for line in lines or []]
    if not values or not all(values):
        return None
    return max(int(m.group(1)) for m in values)


def gpu_memory_used():
    """Memory in use on the GPU in MiB (the fullest one if there are several), or None."""
    lines = _nvidia_smi("--query-gpu=memory.used", "--format=csv,noheader,nounits")
    if not lines or not all(line.isdigit() for line in lines):
        return None
    return max(int(line) for line in lines)


def gpu_programs():
    """(programs of this repository that compute on the GPU, number of other compute programs), or (None, None). Only the
    programs of the repository are named: the others belong to the machine, not to the measurement."""
    lines = _nvidia_smi("--query-compute-apps=process_name", "--format=csv,noheader")
    if lines is None:
        return None, None
    names = [re.split(r"[\\/]", line)[-1] for line in lines]
    ours = sorted(n for n in names if n.lower().startswith(REPOSITORY_PROGRAMS))
    return ours, len(names) - len(ours)


def gpu_state():
    """State of the GPU now: dictionary with utilisation_percent, memory_used_mib, repository_programs, other_programs, free.
    free: no program of the repository computes on it and its utilisation is at most BUSY_ABOVE_PERCENT; False if nvidia-smi
    cannot tell."""
    utilisation, memory = gpu_utilisation(), gpu_memory_used()
    ours, others = gpu_programs()
    free = utilisation is not None and ours is not None and utilisation <= BUSY_ABOVE_PERCENT and not ours
    return dict(utilisation_percent=utilisation, memory_used_mib=memory, repository_programs=ours, other_programs=others, free=free)


def gpu_line(state):
    """The state of the GPU on one line, for a log."""
    def text(value, unit=""):
        return "unknown" if value is None else f"{value}{unit}"
    ours = state["repository_programs"]
    return (f"utilisation {text(state['utilisation_percent'], ' %')}, {text(state['memory_used_mib'], ' MiB')} in use; compute programs of this repository: "
            f"{'unknown' if ours is None else ', '.join(ours) or 'none'}; other compute programs: {text(state['other_programs'])}")


def require_free_gpu(force):
    """State of the GPU at the start of a script that reads times. Stops with exit code 2 if the GPU is busy, unless force."""
    state = gpu_state()
    print(f"GPU at the start: {gpu_line(state)}", flush=True)
    if not state["free"] and not force:
        stop(f"The GPU is not free (utilisation above {BUSY_ABOVE_PERCENT} %, a program of this repository computing on it, or nvidia-smi silent): "
             "the times would be wrong. Nothing was run. Run again later, or with --force: the log then says that the GPU was not free.")
    return state
