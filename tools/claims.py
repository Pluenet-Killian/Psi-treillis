#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Claims sheet of Psi-treillis: registry of the claims, and verdicts.
#
# Each claim is a statement, a numerical criterion and a function that reads again, in the repository, the log that
# carries it. This script runs no computation: it needs no GPU and nothing but the standard library of Python 3.8 or
# later.
#
#   python tools/claims.py           prints one verdict per claim, then checks that docs/CLAIMS.md is up to date
#   python tools/claims.py --write   rewrites the generated part of docs/CLAIMS.md
#
# Exit code: 0 if each verdict is the one that the registry expects, if every log is there and if the sheet is up to
# date; 1 otherwise (verdict changed, log missing or unreadable for a claim registered as measured, log that does not
# have the scope that its criterion states, sheet out of date). The last line printed is "N claims, M anomalies".
#
# To add a claim: an entry in REGISTRY, with the verdict expected. A criterion is "fixed before the measurement" only
# if the commit that introduces it precedes the log. A criterion changed afterwards takes a new identifier; the old
# one stays, with its verdict. A threshold is never adjusted to make this script pass.

from __future__ import annotations

import csv
import io
import json
import math
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Union

ROOT = Path(__file__).resolve().parent.parent
SHEET = "docs/CLAIMS.md"
START = "<!-- verdicts: start of the part written by tools/claims.py, do not edit by hand -->"
END = "<!-- verdicts: end -->"

# The logs, as the scripts of benchmarks/ write them. Each reading function below names the script it follows.
REFERENCE = "benchmarks/compression/reference_files/reference_files.csv"
UNDER_RESOLVED = "benchmarks/compression/under_resolved/under_resolved.csv"
SPEED = "benchmarks/speed/read_timing.csv"
ALTERED = "benchmarks/robustness/altered_files.csv"
TEST_CODEC = "benchmarks/interface/test_logs/test_codec_64.log"
TEST_API = "benchmarks/interface/test_logs/test_api_64.log"
BOUNDS_512 = "benchmarks/interface/bounds_512/verify_bounds_512.csv"
READER_CPP = "benchmarks/independent_readers/cpp/reference_cpp.csv"
READER_PYTHON = "benchmarks/independent_readers/python/reference_python.csv"
GAPS = "benchmarks/independent_readers/python/gaps.csv"
EQUAL_BOUND = "benchmarks/equal_bound/equal_bound_{n}.csv"
VALIDITY = "benchmarks/validity/validity.csv"
CONTROL = "benchmarks/cutout/control.csv"
TRACERS = "benchmarks/tracers"
DIVERGENCE = "benchmarks/divergence/divergence.csv"
SPECTRA = "benchmarks/spectra/spectra.csv"
PLAYBACK = "benchmarks/playback/playback.csv"
WORKFLOW = "benchmarks/linux_build/workflow_run.json"

HOLDS, DOES_NOT_HOLD, PENDING = "holds", "does not hold", "pending"
NO_LOG, UNREADABLE = "log missing", "log unreadable"
UNREGISTERED = "none yet"     # expected verdict of a claim that has never been measured with the present protocol

AFTER = "Fixed after the measurement."
BEFORE_TRACERS = ("Fixed before the analysis (at the top of `benchmarks/tracers/tracers.py`), except the number of tracers per "
                  "voxel at 256³, raised from 1 to 8 after a first trial.")
BEFORE_16 = "Fixed on 16 November 2026, before any measurement."
BEFORE_17 = "Fixed on 17 November 2026, before any measurement."
AFTER_16_PROTOCOL_BEFORE = "Fixed after the measurement of 16 November 2026; protocol fixed before the re-measurement."


class Evidence(Exception):
    """A log is missing, or is not what the script that writes it would have written."""


class LogMissing(Evidence):
    """The file of a log is not in the repository."""


@dataclass
class Reading:
    holds: bool                                      # the criterion, evaluated on the log
    text: str                                        # what is read in the log
    notes: List[str] = field(default_factory=list)   # where the log does not have the scope that the criterion states


# --- Reading the logs ------------------------------------------------------------------------------------------------

def read_log(path: str) -> str:
    file = ROOT / path
    if not file.is_file():
        raise LogMissing(f"log missing: {path}")
    return file.read_text(encoding="utf-8-sig", errors="replace")


def read_table(path: str, columns: list) -> list:
    """Lines of a CSV log, as dictionaries. The columns that the caller reads must all be in the header: a log written
    by an earlier version of a script, under other column names, is refused and never read by guessing."""
    reader = csv.DictReader(io.StringIO(read_log(path), newline=""))
    if not reader.fieldnames:
        raise Evidence(f"{path} is empty")
    absent = [c for c in columns if c not in reader.fieldnames]
    if absent:
        raise Evidence(f"{path} is not the log that the present script writes: column(s) {', '.join(absent)} absent from its header")
    rows = list(reader)
    if not rows:
        raise Evidence(f"{path} holds no line")
    return rows


def number(row: dict, column: str, path: str, line: str) -> float:
    text = (row.get(column) or "").strip()
    try:
        value = float(text)
    except ValueError:
        raise Evidence(f"{path}, {line}: the column {column} holds \"{text}\" where a number is expected") from None
    if not math.isfinite(value):
        raise Evidence(f"{path}, {line}: the column {column} holds \"{text}\"")
    return value


def one(rows: list, path: str, what: str, **equal) -> dict:
    """The only line of a log whose columns have these values."""
    found = [r for r in rows if all(r[k] == v for k, v in equal.items())]
    if len(found) != 1:
        raise Evidence(f"{path}: {len(found)} line(s) for {what}, 1 expected")
    return found[0]


# Lines written by pack_then_play (benchmarks/compression/psi_stream_runs.py): one file encoded by `psi_stream pack`, then
# read back by `psi_stream play`. The figures are those that the programs print, copied as text.
PACK_COLUMNS = ["name", "pack_exit_code", "pack_failure", "grid", "frames", "compression_ratio", "encoding_ms_per_frame", "field_error_mean",
                "largest_error_over_u_rms", "tolerance_over_u_rms", "certification_threshold_over_u_rms", "frames_above_threshold",
                "play_exit_code", "play_failure", "frames_differing", "probe_points", "probe_largest_difference", "gpu_free_at_start"]


@dataclass
class Encoded:
    n: int            # side of the grid
    frames: int
    ratio: float      # raw bytes / bytes of the file
    ms: float         # encoding, per frame
    error: float      # relative field error, mean over the frames checked
    largest: float    # largest error of a component, as a fraction of u'
    row: dict


def packed(path: str, name: str) -> dict:
    row = {r["name"]: r for r in read_table(path, PACK_COLUMNS)}.get(name)
    if row is None:
        raise Evidence(f"line {name} absent from {path}")
    return row


def encoded(path: str, name: str) -> Encoded:
    """A file that pack wrote to its end (exit code 0, 3 or 4). A refusal of pack leaves no figure: it is not a zero."""
    r = packed(path, name)
    if r["compression_ratio"] == "":
        raise Evidence(f"{path}: {name} was not encoded (pack exit code {r['pack_exit_code']}"
                       + (f": {r['pack_failure'][:120]}" if r["pack_failure"] else "") + ")")
    line = f"line {name}"
    return Encoded(int(r["grid"]), int(r["frames"]), number(r, "compression_ratio", path, line), number(r, "encoding_ms_per_frame", path, line),
                   number(r, "field_error_mean", path, line), number(r, "largest_error_over_u_rms", path, line), r)


def bounded(name: str):
    """Reference file with a bound: (n, tolerance, largest error, certification threshold, frames above it, ratio)."""
    p = encoded(REFERENCE, f"f_{name}")
    line = f"line f_{name}"
    if p.row["frames_above_threshold"] == "":
        raise Evidence(f"{REFERENCE}, {line}: no fine level in this file (column frames_above_threshold empty)")
    return (p.n, number(p.row, "tolerance_over_u_rms", REFERENCE, line), p.largest,
            number(p.row, "certification_threshold_over_u_rms", REFERENCE, line), int(p.row["frames_above_threshold"]), p.ratio)


def bench(n: int):
    """Equal-bound bench (benchmarks/equal_bound/equal_bound.py): (codec, bound) -> line. One line per codec and bound,
    all on the same 32 frames."""
    path = EQUAL_BOUND.format(n=n)
    frames = {256: "16-47", 512: "0-31"}[n]
    points = {}
    for r in read_table(path, ["grid", "codec", "bound_over_urms", "frames", "block_frames", "exit_code", "compression_ratio",
                               "rms_error_relative", "largest_error_over_urms"]):
        key = (r["codec"], r["bound_over_urms"])
        if r["grid"] != str(n) or r["frames"] != frames:
            raise Evidence(f"{path}: {r['codec']} at a bound of {r['bound_over_urms']} is measured at {r['grid']}³ on frames {r['frames']}, "
                           f"{n}³ and frames {frames} expected")
        if key in points:
            raise Evidence(f"{path}: two lines for {r['codec']} at a bound of {r['bound_over_urms']}")
        points[key] = r
    return points, path


def bench_point(points: dict, path: str, codec: str, bound: str):
    """(compression ratio, field error, line) of a codec at a bound. A codec that did not encode has no point."""
    r = points.get((codec, bound))
    if r is None:
        raise Evidence(f"{path}: no line for {codec} at a bound of {bound}")
    if r["compression_ratio"] == "":
        raise Evidence(f"{path}: {codec} was not encoded at a bound of {bound} (exit code {r['exit_code']})")
    line = f"{codec} at {bound}"
    return number(r, "compression_ratio", path, line), number(r, "rms_error_relative", path, line), r


def sweep():
    """Log of benchmarks/validity/validity.py: ((sequence, codec, bound) -> line, [(eta in voxels, viscosity, sequence)])."""
    rows = read_table(VALIDITY, ["sequence", "nu", "eta_voxels", "codec", "bound_over_urms", "exit_code", "group_compression_ratio",
                                 "rms_error_relative"])
    lines = {(r["sequence"], r["codec"], r["bound_over_urms"]): r for r in rows}
    sequences = sorted({(float(r["eta_voxels"]), r["nu"], r["sequence"]) for r in rows})
    if len(sequences) != 18:
        raise Evidence(f"{len(sequences)} sequence(s) in {VALIDITY}, 18 expected")
    return lines, sequences


def tracer_summary(campaign: str):
    """Summary of a campaign (analyse() of benchmarks/tracers/tracers.py): path -> line. One line per path: the growth
    of var(D) per Kolmogorov time, and the figures of the criterion (bins of 1 voxel, 10 Kolmogorov times, seeds 1 to 5)."""
    path = f"{TRACERS}/{campaign}_summary.csv"
    lines = {}
    for r in read_table(path, ["campaign", "grid", "path", "fd", "seeds", "last_t_tau", "var_d_growth_per_tau", "criterion_seeds",
                               "criterion_standard_errors"]):
        if r["campaign"] != campaign:
            raise Evidence(f"{path}: line of the campaign {r['campaign']}")
        if r["path"] in lines:
            raise Evidence(f"{path}: two lines for the path {r['path']}")
        lines[r["path"]] = r
    return lines, path


def criterion_excess(lines: dict, path: str) -> dict:
    """Excess dispersion in standard errors, for the paths measured on the five seeds of the criterion."""
    return {p: number(r, "criterion_standard_errors", path, f"path {p}") for p, r in lines.items() if r["criterion_seeds"] == "5"}


def growth(lines: dict, path: str, name: str) -> float:
    if name not in lines:
        raise Evidence(f"path {name} absent from {path}")
    return number(lines[name], "var_d_growth_per_tau", path, f"path {name}")


def excess_over_seeds(campaigns: list, path_name: str, seeds: list) -> float:
    """Excess dispersion of one path, in standard errors, on tracer seeds taken from several campaigns: the arithmetic of
    analyse() in benchmarks/tracers/tracers.py (bins of 1 voxel, 10 Kolmogorov times; standard error: the larger of the
    one between seeds and that of a uniform draw, sqrt(2 / bins) / sqrt(seeds))."""
    values, side = {}, 0
    for campaign in campaigns:
        file = f"{TRACERS}/{campaign}.csv"
        for r in read_table(file, ["grid", "path", "seed", "t_tau", "bin", "excess1"]):
            if r["path"] == path_name and r["seed"] in seeds and round(float(r["t_tau"])) == 10 and float(r["bin"]) == 1.0:
                if r["seed"] in values:
                    raise Evidence(f"{file}: seed {r['seed']} of the path {path_name} appears twice")
                values[r["seed"]] = number(r, "excess1", file, f"path {path_name}, seed {r['seed']}")
                side = int(r["grid"])
    if sorted(values) != sorted(seeds):
        raise Evidence(f"path {path_name}: {len(values)} of the {len(seeds)} tracer seeds found in the campaigns {', '.join(campaigns)}")
    count = len(values)
    mean = statistics.fmean(values.values())
    between = math.sqrt(sum((x - mean) ** 2 for x in values.values()) / (count - 1) / count)
    return mean / max(between, math.sqrt(2.0 / side ** 3) / math.sqrt(count))


SOLVER_SEEDS = ("42", "7", "11")              # the three realisations of the criteria of 17 November 2026, in this order
WITHOUT_BOUND = ("raw", "cuzfp_2bits", "psi_free")


def chain_lines(path: str, columns: list) -> Callable:
    """Logs of benchmarks/divergence/divergence.py and benchmarks/spectra/spectra.py: one line per realisation and per
    chain, frame 32 of data/sweep/nu00280_seed<seed>.seq, at a bound of 5 % of u' for the chains that take one."""
    rows = read_table(path, ["sequence", "solver_seed", "frame", "chain", "bound_over_urms"] + columns)

    def line(seed: str, chain: str) -> dict:
        r = one(rows, path, f"the chain {chain} of the realisation of seed {seed}", solver_seed=seed, chain=chain,
                bound_over_urms="" if chain in WITHOUT_BOUND else "0.05")
        if r["sequence"] != f"nu00280_seed{seed}.seq" or r["frame"] != "32":
            raise Evidence(f"{path}: the chain {chain} of seed {seed} is measured on frame {r['frame']} of {r['sequence']}, "
                           f"frame 32 of nu00280_seed{seed}.seq expected")
        return r
    return line


def by_realisation(path: str, column: str, chain: str) -> list:
    line = chain_lines(path, [column])
    return [number(line(seed, chain), column, path, f"chain {chain}, seed {seed}") for seed in SOLVER_SEEDS]


# Sides of the report of psi_view, as benchmarks/playback/playback.py copies them (titles set in tools/viewer/psi_view.cu).
PSI_SIDE = "Psi-treillis, degree 3"
CUZFP_SIDE = "cuZFP 2 bits + half-precision 3D texture"
RAW_SIDE = "Raw frames + half-precision 3D texture"


def playback():
    return read_table(PLAYBACK, ["grid", "pair", "side", "chain", "mib_read_per_frame", "frame_cost_ms", "gpu_memory_mib",
                                 "divergence_over_gradient", "gpu_free"])


# --- Writing the numbers: decimal point, comma between thousands -----------------------------------------------------

EXPONENTS = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def fixed(x: float, decimals: int = 1) -> str:
    return f"{x:.{decimals}f}"


def pct(x: float, decimals: int = 3) -> str:
    return fixed(100 * x, decimals) + " %"


def sci(x: float) -> str:
    if x == 0:
        return "0"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"{mantissa}·10{str(int(exponent)).translate(EXPONENTS)}"


def whole(x: float) -> str:
    return f"{round(x):,}"


def case(n: int, tolerance: float) -> str:
    return f"{n}³ at {fixed(100 * tolerance, 0)} %"


def times(a: float, b: float, decimals: Optional[int] = None) -> str:
    """a / b, as a text; the comparisons themselves are made by multiplication, never on this quotient."""
    if b == 0:
        return "infinitely many"
    return whole(a / b) if decimals is None else fixed(a / b, decimals)


def three(values: list, write: Callable = sci) -> str:
    return ", ".join(write(x) for x in values)


# Reconstruction paths as benchmarks/tracers/tracers.py names them in its logs, and as the sheet shows them.
PATH_NAMES = {"zfp2": "cuZFP 2 bits + texture", "tex": "trilinear texture", "cubic": "tricubic", "psi3": "Ψ degree 3", "psi2": "Ψ degree 2",
              "psi3b": "Ψ degree 3, bound 5 %"}


def path_name(path: str) -> str:
    if path.startswith("ref"):
        return f"Ψ reference (step 1 voxel, eps {path[3:]})"
    return PATH_NAMES.get(path, path)


def is_psi(path: str) -> bool:
    return path.startswith("psi") or path.startswith("ref")


# --- Measurements: each returns a Reading (criterion held?, what is read in the log, notes on the scope) -------------

DEGREE3 = ["t256_d3_t05", "t256_d3_t04", "t256_d3_t03", "t512_d3_t05", "t512_d3_t04", "t512_d3_t03"]
DEGREE2_HIGH = ["t256_d2_t06", "t512_d2_t07"]
DEGREE2_LOW = ["t256_d2_t05", "t512_d2_t06"]
PSI = "psi_d3"
GPU_NOT_FREE = "the log says that the GPU was not free during the run: its times must not be quoted"


def m_compression(degree: int, min_ratio: float, max_error: float) -> Callable:
    def measure():
        p = encoded(REFERENCE, f"z_t512_d{degree}")
        return Reading(p.ratio >= min_ratio and p.error <= max_error,
                       f"{fixed(p.ratio)}× at a field error of {pct(p.error)}, over {p.frames} frames")
    return measure


def m_bound(names: list) -> Callable:
    def measure():
        held, detail = 0, []
        for name in names:
            n, tolerance, largest, threshold, above, _ = bounded(name)
            ok = above == 0 and largest <= threshold < tolerance
            held += ok
            detail.append(f"{case(n, tolerance)}: {pct(largest)}" + ("" if ok else f", {above} frame(s) above the threshold"))
        return Reading(held == len(names),
                       f"{held} of {len(names)} files below the certification threshold. Largest error, as a fraction of u′: " + "; ".join(detail))
    return measure


def m_cost_of_the_bound(names: list, floors: dict) -> Callable:
    def measure():
        ok, detail = True, []
        for name in names:
            n, tolerance, _, _, _, ratio = bounded(name)
            ok = ok and ratio >= floors[round(100 * tolerance)]
            detail.append(f"{case(n, tolerance)}: {fixed(ratio)}×")
        return Reading(ok, "; ".join(detail))
    return measure


def m_readback():
    names = [f"f_{name}" for name in DEGREE3 + DEGREE2_HIGH + DEGREE2_LOW] + ["z_t512_d2", "z_t512_d3"]
    differing, probes, worst, points = 0, 0, 0.0, set()
    for name in names:
        r = packed(REFERENCE, name)
        if r["frames_differing"] == "":
            raise Evidence(f"{REFERENCE}: {name} was not read back to its end (play exit code {r['play_exit_code']}"
                           + (f": {r['play_failure'][:120]}" if r["play_failure"] else "") + ")")
        differing += int(r["frames_differing"])
        if r["probe_largest_difference"] != "":
            difference = number(r, "probe_largest_difference", REFERENCE, f"line {name}")
            if difference < 0:        # play prints a negative difference for a probed frame that it did not read:
                continue              # that is not a comparison, and never a zero difference
            probes += 1
            worst = max(worst, difference)
            points.add(int(r["probe_points"]))
    return Reading(differing == 0 and probes == 10 and worst == 0.0,
                   f"{len(names)} files read back, {differing} frames whose state differs; off-grid probe on {probes} files"
                   + (f" ({' or '.join(whole(p) for p in sorted(points))} points)" if points else "") + f", largest difference {sci(worst)}")


def m_divergence():
    rows = [r for r in playback() if r["chain"].startswith("Psi-treillis")]
    if not rows:
        raise Evidence(f"{PLAYBACK}: no side shows Psi-treillis")
    values = []
    for r in rows:
        if r["divergence_over_gradient"] == "":
            raise Evidence(f"{PLAYBACK}: no divergence for the side \"{r['chain']}\" at {r['grid']}³, pair {r['pair']}")
        values.append(number(r, "divergence_over_gradient", PLAYBACK, f"{r['grid']}³, pair {r['pair']}"))
    grids = sorted({int(r["grid"]) for r in rows})
    return Reading(max(values) <= 1e-12,
                   f"{sci(max(values))} of the RMS gradient at most, on the {len(values)} sides of the log that show Psi-treillis "
                   f"({' and '.join(f'{n}³' for n in grids)})")


def m_speed(max_decoding: float, max_step: float) -> Callable:
    def measure():
        rows = read_table(SPEED, ["repetition", "file", "gpu_free_at_start", "decoding_ms", "rk4_step_ms", "frames_differing"])
        files = sorted({r["file"] for r in rows})
        if len(files) != 2:
            raise Evidence(f"{SPEED}: {len(files)} file(s), two expected")
        ok, detail, notes = True, [], []
        for file in files:
            series = [r for r in rows if r["file"] == file]
            decoding = statistics.median(number(r, "decoding_ms", SPEED, file) for r in series)
            step = statistics.median(number(r, "rk4_step_ms", SPEED, file) for r in series)
            ok = ok and decoding <= max_decoding and step <= max_step and all(r["frames_differing"] == "0" for r in series)
            detail.append(f"`{file}`: decoding {fixed(decoding, 2)} ms, RK4 step {fixed(step, 2)} ms (medians of {len(series)} repetitions)")
            if len(series) != 15:
                notes.append(f"{len(series)} repetitions of {file} in the log, the criterion states 15")
        if any(r["gpu_free_at_start"] != "yes" for r in rows):
            notes.append(GPU_NOT_FREE)
        return Reading(ok, "; ".join(detail), notes)
    return measure


def m_encoding(ceilings: dict) -> Callable:
    def measure():
        ok, detail, notes = True, [], []
        for name, ceiling in ceilings.items():
            n, tolerance, _, _, _, _ = bounded(name)
            p = encoded(REFERENCE, f"f_{name}")
            ok = ok and p.ms <= ceiling
            detail.append(f"{case(n, tolerance)}: {fixed(p.ms, 0)} ms per frame")
            if p.row["gpu_free_at_start"] != "yes" and GPU_NOT_FREE not in notes:
                notes.append(GPU_NOT_FREE)
        return Reading(ok, "; ".join(detail), notes)
    return measure


def m_dispersion(selected: Callable, held: Callable, expected: int) -> Callable:
    def measure():
        ok, detail = True, []
        for size, campaign in (("256", "resolved_256"), ("512", "resolved_512")):
            lines, path = tracer_summary(campaign)
            for name, excess in criterion_excess(lines, path).items():
                if selected(name):
                    ok = ok and held(excess)
                    detail.append(f"{path_name(name)}, {size}³: " + f"{excess:+.1f}")
        if len(detail) != expected:
            raise Evidence(f"{len(detail)} five-seed line(s) found in the summaries of the tracer campaigns, {expected} expected")
        return Reading(ok, "In standard errors. " + "; ".join(detail))
    return measure


def m_volume_drift(min_ratio: float) -> Callable:
    def measure():
        ok, detail = True, []
        for size, campaign in (("256", "resolved_256"), ("512", "resolved_512")):
            lines, path = tracer_summary(campaign)
            psi, texture, zfp = growth(lines, path, "psi3"), growth(lines, path, "tex"), growth(lines, path, "zfp2")
            ok = ok and texture >= min_ratio * psi and zfp >= min_ratio * psi
            detail.append(f"{size}³: Ψ {sci(psi)}, texture {sci(texture)} ({times(texture, psi)} times), cuZFP {sci(zfp)} ({times(zfp, psi)} times)")
        return Reading(ok, "Per Kolmogorov time. " + "; ".join(detail))
    return measure


def m_confirmation(min_ratio: float) -> Callable:
    """The four criteria of LAGR-1 to LAGR-4 on another realisation (campaign confirmation_256, five seeds)."""
    def measure():
        lines, path = tracer_summary("confirmation_256")
        table = criterion_excess(lines, path)
        try:
            zfp, texture = table["zfp2"], table["tex"]
        except KeyError as e:
            raise Evidence(f"path {e} absent from {path}, or measured on fewer than five seeds") from None
        texture_growth, psi_growth = growth(lines, path, "tex"), growth(lines, path, "psi3")
        psi = {name: excess for name, excess in table.items() if is_psi(name)}
        if len(psi) < 4:
            raise Evidence(f"{len(psi)} Ψ path(s) with five seeds in {path}, 4 expected")
        return Reading(zfp > 5 and all(abs(e) < 2 for e in psi.values()) and texture < 5 and texture_growth >= min_ratio * psi_growth,
                       f"In standard errors: cuZFP at 2 bits {zfp:+.1f}; trilinear texture {texture:+.1f}; "
                       + "; ".join(f"{path_name(name)} {e:+.1f}" for name, e in psi.items())
                       + f". Growth of var(D) per Kolmogorov time: texture {sci(texture_growth)}, Ψ degree 3 {sci(psi_growth)} "
                       f"({times(texture_growth, psi_growth)} times)")
    return measure


def m_altered():
    rows = read_table(ALTERED, ["batch", "file", "exit_code", "outcome"])
    outcomes: dict = {}
    for r in rows:
        outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
    notes = [] if len(rows) == 283 else [f"{len(rows)} altered files in the log, the criterion states 283"]
    return Reading(set(outcomes) <= {"refused", "counted", "read"},
                   f"{len(rows)} altered files read back: " + ", ".join(f"{count} {outcome}" for outcome, count in sorted(outcomes.items())), notes)


def m_tests():
    text = read_log(TEST_CODEC)
    total = re.search(r"^TOTAL: (\d+) cases, (\d+) OK, (\d+) FAILED, (\d+) known defect\(s\)", text, re.M)
    if not total:
        raise Evidence(f'line "TOTAL: N cases, N OK, N FAILED, N known defect(s)" absent from {TEST_CODEC}')
    cases, passed_count, failed, known = (int(x) for x in total.groups())
    passed = set(re.findall(r"^OK\s+([^\s:]+)", text, re.M))
    required = {"d2.determinism", "d3.determinism", "sequence.fingerprint"}
    fingerprints = sorted(c for c in passed if ".fingerprint." in c)
    return Reading(failed == 0 and required <= passed and len(fingerprints) >= 2,
                   f"{cases} cases, {passed_count} OK, {failed} failed, {known} known defect{'' if known == 1 else 's'}; determinism in degree 2 and 3: "
                   f"{'OK' if required <= passed else 'no'}; {len(fingerprints)} fingerprint cases OK")


def m_api():
    text = read_log(TEST_API)
    total = re.search(r"^TOTAL : (\d+) cases, (\d+) ok, (\d+) failed", text, re.M)
    if not total:
        raise Evidence(f'line "TOTAL : N cases, N ok, N failed" absent from {TEST_API}')
    cases, passed_count, failed = (int(x) for x in total.groups())
    passed = set(re.findall(r"^ok\s+(\S+)", text, re.M))
    required = {"cli.same_bytes_as_pack", "api.bound_is_held", "api.sample_matches_the_grid", "api.divergence_free", "api.memory_is_released"}
    bound = re.search(r"largest error ([0-9.]+) % of u_rms for a bound of ([0-9.]+) %", text)
    divergence = re.search(r"RMS divergence ([0-9.eE+-]+) of the RMS gradient", text)
    detail = f"{cases} cases, {passed_count} passed, {failed} failed; the five required cases: {'passed' if required <= passed else 'no'}"
    if bound:
        detail += f"; largest error {bound.group(1)} % of u′ for a bound of {bound.group(2)} %"
    if divergence:
        detail += f"; divergence {sci(float(divergence.group(1)))} of the gradient, by differences"
    return Reading(failed == 0 and required <= passed, detail)


def m_api_512():
    rows = read_table(BOUNDS_512, ["file", "frame", "bound_held", "tolerance", "largest_error", "verdict"])
    if any(r["verdict"] not in ("consistent", "MISMATCH") for r in rows):
        raise Evidence(f"{BOUNDS_512}: the column verdict holds something other than consistent or MISMATCH")
    files = {r["file"] for r in rows}
    consistent = sum(r["verdict"] == "consistent" for r in rows)
    # As before the translation of the script: every file with a tolerance, except the one whose bound is known to be
    # missed (f_t512_d2_t06, claim BOUND-4), which the script does not read.
    ratios = [number(r, "largest_error", BOUNDS_512, f"{r['file']}, frame {r['frame']}") / float(r["tolerance"])
              for r in rows if float(r["tolerance"]) > 0 and r["file"] != "f_t512_d2_t06"]
    if not ratios:
        raise Evidence(f"{BOUNDS_512}: no frame of a file with a bound")
    notes = [] if (len(rows), len(files)) == (24, 6) else [f"{len(rows)} frames of {len(files)} files in the log, the criterion states four frames "
                                                            "of six files"]
    return Reading(consistent == len(rows) and max(ratios) <= 1.0,
                   f"{len(rows)} frames of {len(files)} files, {len(rows) - consistent} non-conforming; with a bound, largest error at "
                   f"{fixed(100 * max(ratios), 2)} % of the tolerance at most", notes)


def m_against_sperr(min_ratio: float, max_error: float) -> Callable:
    def measure():
        ok, detail, notes = True, [], []
        for n, bounds in ((256, ("0.05", "0.04", "0.03")), (512, ("0.05", "0.04"))):
            points, path = bench(n)
            for b in bounds:
                ratio, error, line = bench_point(points, path, PSI, b)
                ratio_s, error_s, _ = bench_point(points, path, "sperr", b)
                ok = ok and ratio >= min_ratio * ratio_s and error <= max_error * error_s
                detail.append(f"{case(n, float(b))}: {fixed(ratio)}× against {fixed(ratio_s)}× ({times(ratio, ratio_s, 2)} times), "
                              f"field error {pct(error)} against {pct(error_s)}")
                if line["exit_code"] != "0":
                    notes.append(f"Psi-treillis ends with code {line['exit_code']} at {case(n, float(b))}: it is not at the bound requested")
        return Reading(ok, "; ".join(detail), notes)
    return measure


def m_pareto_front():
    points, path = bench(256)
    others = [(number(r, "compression_ratio", path, f"{codec} at {bound}"), number(r, "rms_error_relative", path, f"{codec} at {bound}"))
              for (codec, bound), r in points.items() if not codec.startswith("psi") and r["compression_ratio"] != ""]
    ours = [bench_point(points, path, PSI, b)[:2] for b in ("0.05", "0.04", "0.03")]
    free = sum(not any(ratio_a >= ratio and error_a <= error for ratio_a, error_a in others) for ratio, error in ours)
    return Reading(free == len(ours), f"{free} of {len(ours)} points not dominated, among {len(others)} points of the other codecs")


def m_against_sz3_4d():
    points, path = bench(256)
    ok, detail = True, []
    for b in ("0.05", "0.04", "0.03"):
        ratio, error, _ = bench_point(points, path, PSI, b)
        ratio_s, error_s, line = bench_point(points, path, "sz3_4d", b)
        if line["block_frames"] != "32":
            raise Evidence(f"{path}: SZ3 in 4D is measured by blocks of {line['block_frames']} frames at a bound of {b}, one block of 32 expected")
        ok = ok and ratio >= ratio_s
        detail.append(f"{case(256, float(b))}: {fixed(ratio)}× against {fixed(ratio_s)}× (field error {pct(error)} against {pct(error_s)})")
    return Reading(ok, "; ".join(detail))


def m_reference_cpp(threshold: float) -> Callable:
    """Log of benchmarks/independent_readers/cpp/reference_cpp.py: one line per reference file, read by psi_ref (CPU)."""
    def measure():
        rows = read_table(READER_CPP, ["file", "frames", "frames_read", "fingerprints_differing", "probe_points", "lattice_nodes_differing",
                                       "fine_nodes_differing", "cpu_gpu_bound_over_u_rms", "gpu_frames_differing", "probe_origin"])
        lines = {r["file"]: r for r in rows}
        files = [f"f_{name}.psi" for name in DEGREE3 + DEGREE2_HIGH + DEGREE2_LOW] + ["z_t512_d2.psi", "z_t512_d3.psi"]
        frames = read = differing = nodes = gpu_differing = 0
        worst = 0.0
        for file in files:
            r = lines.get(file)
            if r is None or not r["frames"]:
                raise Evidence(f"{file} was not read in {READER_CPP}")
            if not r["cpu_gpu_bound_over_u_rms"]:
                raise Evidence(f"{file} was not compared with the GPU reader at the probe in {READER_CPP} ({r['probe_origin']})")
            frames, read, differing = frames + int(r["frames"]), read + int(r["frames_read"]), differing + int(r["fingerprints_differing"])
            nodes += int(r["lattice_nodes_differing"]) + int(r["fine_nodes_differing"])
            gpu_differing += int(r["gpu_frames_differing"])
            worst = max(worst, number(r, "cpu_gpu_bound_over_u_rms", READER_CPP, file))
        points = {int(lines[f]["probe_points"]) for f in files}
        return Reading(read == frames and differing == 0 and gpu_differing == 0 and worst < threshold,
                       f"{len(files)} files, {read} of {frames} frames read back, {differing} fingerprints different from the one in the table; at the "
                       f"probe of each file ({' or '.join(whole(p) for p in sorted(points))} points, one frame): {nodes} nodes of the lattice or of "
                       f"the fine level different from those of the encoder, and a velocity field at most {sci(worst)} u′ from that of the GPU reader")
    return measure


def m_reference_python(threshold: float) -> Callable:
    """Log of benchmarks/independent_readers/python/reference_python.py: one line per file read by
    tools/reference_reader.py; and the list of the passages of docs/FORMAT.md noted while that reader was written
    (gaps.csv, kept by hand)."""
    def measure():
        rows = read_table(READER_PYTHON, ["file", "group", "frames", "frames_read", "fingerprints_differing", "probe_read_on",
                                          "lattice_nodes_differing", "fine_nodes_differing", "field_difference_over_u_rms"])
        lines = {r["file"]: r for r in rows}
        expected = [f"f_{name}.psi" for name in ("t256_d3_t05", "t256_d3_t04", "t256_d3_t03", "t256_d2_t06", "t256_d2_t05")]
        expected += [f"d{d}_{c}.psi" for d in (2, 3) for c in ("base", "bound", "raw_keys", "exceptions")]
        frames = read = differing = nodes = elsewhere = 0
        worst = 0.0
        for file in expected:
            if file not in lines:
                raise Evidence(f"{file} was not read in {READER_PYTHON}")
        for r in lines.values():
            if not r["field_difference_over_u_rms"]:
                raise Evidence(f"{r['file']} was not compared with a probe in {READER_PYTHON}")
            frames, read, differing = frames + int(r["frames"]), read + int(r["frames_read"]), differing + int(r["fingerprints_differing"])
            nodes += int(r["lattice_nodes_differing"]) + int(r["fine_nodes_differing"])
            elsewhere += r["probe_read_on"] != "file"
            worst = max(worst, number(r, "field_difference_over_u_rms", READER_PYTHON, r["file"]))
        gaps = read_table(GAPS, ["id", "section", "counts"])
        if any(g["counts"] not in ("yes", "no") for g in gaps):
            raise Evidence(f"{GAPS}: the column counts holds something other than yes or no")
        counted = [g for g in gaps if g["counts"] == "yes"]
        references = sum(1 for r in lines.values() if r["group"] == "reference_256")
        listed = (": " + "; ".join(f"{g['id']}, section {g['section']}" for g in counted)) if counted else ""
        return Reading(read == frames and differing == 0 and worst < threshold and not counted,
                       f"{references} reference files at 256³ and {len(lines) - references} test files at 64³, {whole(read)} frames read back "
                       f"out of {whole(frames)}, {differing} fingerprint different from that of the table; at the probe of each file, {nodes} node "
                       f"of the lattice or of the fine level different from those of the encoder, and a velocity field at {sci(worst)} u′ at most"
                       + (f" ({elsewhere} files probed on their re-encoding, their bytes not having been found again)" if elsewhere else "")
                       + f"; {len(gaps)} passages noted during the writing, of which {len(counted)} counted as concerning the state or the "
                       f"field{listed}")
    return measure


def m_under_resolved_fidelity(min_ratio: float, max_error: float) -> Callable:
    def measure():
        p = encoded(UNDER_RESOLVED, "d3")
        return Reading(p.ratio >= min_ratio and p.error <= max_error,
                       f"{fixed(p.ratio)}× at a field error of {pct(p.error)}; largest error {fixed(p.largest, 3)} u′")
    return measure


def m_under_resolved_bound():
    codes = {}
    for name in ("d3_t05", "d3_t04", "d3_t03", "d2_t07"):
        codes[name] = packed(UNDER_RESOLVED, name)["pack_exit_code"]
        if not codes[name].isdigit():
            raise Evidence(f"{UNDER_RESOLVED}, line {name}: the column pack_exit_code holds \"{codes[name]}\"")
    degree3 = [codes[c] for c in ("d3_t05", "d3_t04", "d3_t03")]
    return Reading(all(c == "0" for c in degree3),
                   f"Degree 3: `pack` exits with code {degree3[0]} at 5 %, {degree3[1]} at 4 % and {degree3[2]} at 3 % (0: bound held; 1: refusal, "
                   f"capacity of the fine level exceeded; 4: bound missed). Degree 2 at 7 %: code {codes['d2_t07']}")


def sweep_number(lines: dict, key: tuple, column: str) -> float:
    return number(lines[key], column, VALIDITY, f"{key[0]}, {key[1]}, bound {key[2] or 'none'}")


def m_bound_threshold(low: float, high: float) -> Callable:
    def measure():
        lines, sequences = sweep()
        held = [e for e, _, s in sequences if lines[(s, "psi_d3_mf2", "0.05")]["exit_code"] == "0"]
        missed = [e for e, _, s in sequences if lines[(s, "psi_d3_mf2", "0.05")]["exit_code"] != "0"]
        if not held or not missed:
            return Reading(False, f"Bound held on {len(held)} sequence(s), missed or refused on {len(missed)}: the threshold is not within "
                                  f"the sweep ({fixed(sequences[0][0], 3)} to {fixed(sequences[-1][0], 3)} voxel per Kolmogorov scale)")
        detail = (f"Bound held on {len(held)} sequences, from {fixed(min(held), 3)} to {fixed(max(held), 3)} voxel per Kolmogorov scale; "
                  f"missed or refused on {len(missed)}, from {fixed(min(missed), 3)} to {fixed(max(missed), 3)} voxel")
        if max(missed) >= min(held):
            return Reading(False, detail + ": the two sets overlap, there is no clear threshold")
        threshold = (max(missed) + min(held)) / 2
        return Reading(low <= threshold <= high, detail + f"; η* = {fixed(threshold, 3)} voxel")
    return measure


def m_ahead_of_sperr_under_bound():
    lines, sequences = sweep()
    held = []
    for e, _, s in sequences:
        if lines[(s, "psi_d3_mf2", "0.05")]["exit_code"] == "0":
            ratio = sweep_number(lines, (s, "psi_d3_mf2", "0.05"), "group_compression_ratio")
            ratio_s = sweep_number(lines, (s, "sperr", "0.05"), "group_compression_ratio")
            held.append((ratio / ratio_s, e, ratio, ratio_s))
    if not held:
        return Reading(False, "No sequence of the sweep holds the bound of 5 % at a step of 2 voxels")
    ahead = sum(c[0] >= 1 for c in held)
    (r0, e0, t0, s0), (r1, e1, t1, s1) = min(held), max(held)
    return Reading(ahead == len(held),
                   f"Ahead of SPERR on {ahead} of the {len(held)} sequences where the bound holds: from {fixed(r0, 2)} times its compression ratio "
                   f"({fixed(t0)}× against {fixed(s0)}×, at {fixed(e0, 3)} voxel per Kolmogorov scale) to {fixed(r1, 2)} times ({fixed(t1)}× against "
                   f"{fixed(s1)}×, at {fixed(e1, 3)} voxel)")


def m_step_of_one_voxel():
    lines, sequences = sweep()
    held = sum(lines[(s, "psi_d3_mf1", "0.05")]["exit_code"] == "0" for _, _, s in sequences)
    ratios = []
    for e, _, s in sequences:
        if lines[(s, "psi_d3_mf1", "0.05")]["group_compression_ratio"]:
            ratio = sweep_number(lines, (s, "psi_d3_mf1", "0.05"), "group_compression_ratio")
            ratio_s = sweep_number(lines, (s, "sperr", "0.05"), "group_compression_ratio")
            ratios.append((ratio / ratio_s, ratio, ratio_s, e))
    if not ratios:
        return Reading(False, f"Bound held on {held} of {len(sequences)} sequence(s), no file written")
    (r0, t0, s0, e0), (r1, t1, s1, e1) = min(ratios), max(ratios)
    return Reading(held == len(sequences) and len(ratios) == len(sequences) and r1 < 1,
                   f"Bound held on {held} of {len(sequences)} sequences; compression ratio from {fixed(r0, 2)} times that of SPERR ({fixed(t0)}× "
                   f"against {fixed(s0)}×, at {fixed(e0, 3)} voxel per Kolmogorov scale) to {fixed(r1, 2)} times ({fixed(t1)}× against {fixed(s1)}×, "
                   f"at {fixed(e1, 3)} voxel)")


def m_error_against_resolution(eta_resolved: float, max_error: float, eta_under_resolved: float, min_error: float) -> Callable:
    def measure():
        lines, sequences = sweep()
        errors = [(e, sweep_number(lines, (s, "psi_d3_mf2", ""), "rms_error_relative")) for e, _, s in sequences]
        resolved = [x for e, x in errors if e >= eta_resolved]
        under_resolved = [x for e, x in errors if e <= eta_under_resolved]
        if not resolved or not under_resolved:
            return Reading(False, f"{len(resolved)} sequence(s) at {fixed(eta_resolved, 2)} voxel or more, {len(under_resolved)} at "
                                  f"{fixed(eta_under_resolved, 2)} voxel or less: at least one is needed in each range")
        return Reading(max(resolved) <= max_error and min(under_resolved) > min_error,
                       f"{len(resolved)} sequences at {fixed(eta_resolved, 2)} voxel or more: field error from {pct(min(resolved))} to "
                       f"{pct(max(resolved))}; {len(under_resolved)} sequences at {fixed(eta_under_resolved, 2)} voxel or less: from "
                       f"{pct(min(under_resolved))} to {pct(max(under_resolved))}")
    return measure


def m_spread_between_realisations(threshold: float) -> Callable:
    def measure():
        lines, sequences = sweep()
        worst = {"group_compression_ratio": (0.0, ""), "rms_error_relative": (0.0, "")}
        viscosities = sorted({nu for _, nu, _ in sequences}, key=float)
        for nu in viscosities:
            group = [s for _, v, s in sequences if v == nu]
            if len(group) != 3:
                raise Evidence(f"{len(group)} realisation(s) for viscosity {nu} in {VALIDITY}, 3 expected")
            for column in worst:
                values = [sweep_number(lines, (s, "psi_d3_mf2", ""), column) for s in group]
                worst[column] = max(worst[column], ((max(values) - min(values)) / statistics.fmean(values), nu))
        (ratio, nu_ratio), (error, nu_error) = worst["group_compression_ratio"], worst["rms_error_relative"]
        return Reading(ratio < threshold and error < threshold,
                       f"Over the {len(viscosities)} viscosities, between the three realisations: the compression ratio varies by {pct(ratio, 1)} at "
                       f"most (viscosity {nu_ratio}), the field error by {pct(error, 1)} at most (viscosity {nu_error})")
    return measure


def m_cutout_control(margin: float) -> Callable:
    def measure():
        rows = read_table(CONTROL, ["source", "field", "setting", "bound_over_urms", "centre_rms_error_relative"])
        ok, detail = True, []
        for source in ("turb512", "turb512_nu0006"):
            errors = {}
            for kind in ("whole field", "windowed cutout"):
                found = [r for r in rows if r["source"] == source and r["field"] == kind and r["setting"] == "d3_mf2" and not r["bound_over_urms"]]
                if len(found) != 1 or not found[0]["centre_rms_error_relative"]:
                    raise Evidence(f'measurement without a bound of "{kind}" absent for {source} in {CONTROL}')
                errors[kind] = number(found[0], "centre_rms_error_relative", CONTROL, f"{source}, {kind}")
            ratio = errors["windowed cutout"] / errors["whole field"]
            ok = ok and abs(ratio - 1.0) <= margin
            detail.append(f"`{source}.seq`: {pct(errors['windowed cutout'])} at the centre of the windowed cutout against "
                          f"{pct(errors['whole field'])} on the same voxels of the whole field, that is {fixed(ratio, 3)} times")
        return Reading(ok, "; ".join(detail))
    return measure


def m_workflow():
    """Record of benchmarks/linux_build/record.py: the last completed run of the GitHub workflow that compiles the
    project under Linux."""
    r = json.loads(read_log(WORKFLOW))
    absent = [k for k in ("run", "commit", "conclusion", "steps", "compilers", "linked_targets", "claims_sheet") if k not in r]
    if absent:
        raise Evidence(f"{WORKFLOW} is not the record that the present script writes: field(s) {', '.join(absent)} absent")
    steps = r["steps"]
    succeeded = sum(s["conclusion"] == "success" for s in steps)
    missing = [t for t in ("libpsitreillis.so", "psit", "psi_stream", "dns") if not any(x.startswith(t) for x in r["linked_targets"])]
    reader = any("reference reader" in s["name"] and s["conclusion"] == "success" for s in steps)
    versions = sorted({m.group(1) + " " + m.group(2) for c in r["compilers"] for m in [re.search(r"is (GNU|NVIDIA) (\S+)", c)] if m})
    return Reading(r["conclusion"] == "success" and succeeded == len(steps) and not missing and reader,
                   f"Run {r['run']} of the workflow on commit `{r['commit'][:7]}`: {succeeded} of {len(steps)} steps succeeded; compilers "
                   f"{' and '.join(versions)}; linked targets: {', '.join('`' + t + '`' for t in r['linked_targets'])}"
                   + (f"; missing target(s): {', '.join(missing)}" if missing else "")
                   + f"; `psi_ref` {'compiled' if reader else 'not compiled'} by g++; claims sheet replayed on the workflow machine: "
                   f"\"{r['claims_sheet']}\"")


COMPRESSORS = ("sz3", "sz3_4d", "sperr")      # the error-bounded chains that the criteria of 17 November 2026 judge
IN_THIS_ORDER = "realisations of solver seeds 42, 7 and 11, in this order"


def m_grid_divergence(decoded_min: float, original_max: float) -> Callable:
    def measure():
        original = by_realisation(DIVERGENCE, "grid_divergence_over_gradient", "raw")
        decoded = {c: by_realisation(DIVERGENCE, "grid_divergence_over_gradient", c) for c in COMPRESSORS}
        return Reading(all(x >= decoded_min for v in decoded.values() for x in v) and all(x <= original_max for x in original),
                       f"M1 of the original grid: {three(original)}; of the decoded grid: "
                       + "; ".join(f"`{c}` {three(v)}" for c, v in decoded.items()) + f" ({IN_THIS_ORDER})")
    return measure


def m_read_divergence(original_min: float) -> Callable:
    def measure():
        original = by_realisation(DIVERGENCE, "read_divergence_over_gradient", "raw")
        return Reading(all(x >= original_min for x in original), f"M2 of the original grid: {three(original)} ({IN_THIS_ORDER})")
    return measure


def m_read_divergence_added(min_ratio: float) -> Callable:
    def measure():
        original = by_realisation(DIVERGENCE, "read_divergence_over_gradient", "raw")
        ok, detail = True, []
        for c in COMPRESSORS:
            decoded = by_realisation(DIVERGENCE, "read_divergence_over_gradient", c)
            ok = ok and all(d >= min_ratio * o for d, o in zip(decoded, original))
            detail.append(f"`{c}` {three(decoded)}, that is {', '.join(times(d, o, 2) for d, o in zip(decoded, original))} times")
        return Reading(ok, f"M2 of the original grid: {three(original)}; of the decoded grid: " + "; ".join(detail) + f" ({IN_THIS_ORDER})")
    return measure


def compressor_campaign():
    """Summary of the campaign compressors_256 as M3 defines it: five tracer seeds, 10 Kolmogorov times, 256³."""
    lines, path = tracer_summary("compressors_256")
    for name in ("tex", "psi3b", "zfp2") + COMPRESSORS:
        r = lines.get(name)
        if r is None:
            raise Evidence(f"path {name} absent from {path}")
        if (r["grid"], r["seeds"], r["last_t_tau"], r["criterion_seeds"]) != ("256", "5", "10", "5"):
            raise Evidence(f"{path}: the path {name} is measured at {r['grid']}³ on {r['seeds']} seed(s) up to {r['last_t_tau']} Kolmogorov times, "
                           "256³, 5 seeds and 10 Kolmogorov times expected")
    return lines, path


def m_tracers_after_compressors(min_ratio: float) -> Callable:
    def measure():
        lines, path = compressor_campaign()
        raw, psi = growth(lines, path, "tex"), growth(lines, path, "psi3b")
        ok, detail = True, []
        for c in COMPRESSORS:
            g = growth(lines, path, c)
            ok = ok and g >= raw and g >= min_ratio * psi
            detail.append(f"`{c}` {sci(g)} ({times(g, raw, 1)} times that of `raw`, {times(g, psi)} times that of `psi`)")
        return Reading(ok, f"Growth of the variance of D per Kolmogorov time: `raw` {sci(raw)}, `psi` {sci(psi)}; " + "; ".join(detail))
    return measure


def m_spectrum(enstrophy_margin: float) -> Callable:
    def measure():
        line = chain_lines(SPECTRA, ["truncation_wavenumber", "enstrophy_over_original", "k10"])
        ok, enstrophies, shells, truncations = True, [], [], set()
        for seed in SOLVER_SEEDS:
            r = line(seed, "psi_free")
            enstrophy = number(r, "enstrophy_over_original", SPECTRA, f"chain psi_free, seed {seed}")
            truncation = int(r["truncation_wavenumber"])
            if r["k10"] != "none" and not r["k10"].isdigit():
                raise Evidence(f"{SPECTRA}: the column k10 holds \"{r['k10']}\" for the chain psi_free of seed {seed}")
            # "none": no shell is more than 10 % away up to the truncation wavenumber, so k10 is beyond it.
            ok = ok and abs(enstrophy - 1.0) <= enstrophy_margin and (r["k10"] == "none" or 2 * int(r["k10"]) >= truncation)
            enstrophies.append(enstrophy)
            shells.append(r["k10"])
            truncations.add(truncation)
        return Reading(ok, f"Enstrophy of the decoded grid over that of the original: {three(enstrophies, lambda x: fixed(x, 4))}; k10: "
                           f"{', '.join(shells)}, for a truncation wavenumber of {' or '.join(str(t) for t in sorted(truncations))} "
                           f"({IN_THIS_ORDER})")
    return measure


def m_playback(min_data_ratio: float, min_memory_ratio: float) -> Callable:
    def measure():
        rows = playback()

        def side(grid: str, pair: str, chain: str) -> dict:
            return one(rows, PLAYBACK, f"the side \"{chain}\" at {grid}³, pair {pair}", grid=grid, pair=pair, chain=chain)

        def value(r: dict, column: str) -> float:
            return number(r, column, PLAYBACK, f"{r['grid']}³, pair {r['pair']}, side {r['side']}")

        psi_256, zfp_256 = side("256", "cuzfp_psi", PSI_SIDE), side("256", "cuzfp_psi", CUZFP_SIDE)
        psi_512, raw_512 = side("512", "raw_psi", PSI_SIDE), side("512", "raw_psi", RAW_SIDE)
        cost, cost_zfp = value(psi_256, "frame_cost_ms"), value(zfp_256, "frame_cost_ms")
        data, data_raw = value(psi_512, "mib_read_per_frame"), value(raw_512, "mib_read_per_frame")
        memory, memory_raw = value(psi_512, "gpu_memory_mib"), value(raw_512, "gpu_memory_mib")
        notes = [GPU_NOT_FREE] if any(r["gpu_free"] != "yes" for r in (psi_256, zfp_256, psi_512, raw_512)) else []
        return Reading(cost > cost_zfp and data_raw >= min_data_ratio * data and memory_raw >= min_memory_ratio * memory,
                       f"256³: one frame costs {fixed(cost, 2)} ms with Psi-treillis against {fixed(cost_zfp, 2)} ms with cuZFP at 2 bits; 512³: "
                       f"{fixed(data, 2)} MiB read per frame against {fixed(data_raw, 2)} MiB for the raw chain ({times(data_raw, data, 1)} times less), "
                       f"{fixed(memory, 0)} MiB of GPU memory against {fixed(memory_raw, 0)} MiB ({times(memory_raw, memory, 2)} times less)", notes)
    return measure


# --- Figures that a caveat or a basis quotes: read in the logs too, so that a replay cannot leave them stale ---------

class Filled:
    """A text of the sheet with figures read in the logs: values() gives what replaces each {name} of the text. A figure
    whose log cannot be read is given as the Evidence that says why; it is then written "[not available]", never guessed."""

    def __init__(self, text: str, values: Callable):
        self.text, self.values = text, values

    def render(self):
        """(text, reason why some of its figures are missing or None)."""
        try:
            values = self.values()
        except Evidence as e:
            reasons = [str(e)]
        except (KeyError, IndexError, ValueError, ZeroDivisionError, OverflowError) as e:
            reasons = [f"log unreadable: {e!r}"]
        else:
            reasons = []
            for v in values.values():
                if isinstance(v, Evidence) and str(v) not in reasons:
                    reasons.append(str(v))
            text = self.text.format(**{k: "[not available]" if isinstance(v, Evidence) else v for k, v in values.items()})
            if not reasons:
                return text, None
            return text + f" [Figures not available: {'; '.join(reasons)}.]", "; ".join(reasons)
        return self.text.format_map(_NotAvailable()) + f" [Figures not available: {reasons[0]}.]", reasons[0]


class _NotAvailable(dict):
    def __missing__(self, key):
        return "[not available]"


def available(values: Callable, names: list) -> dict:
    """The figures that values() gives, or, for each of these names, the Evidence that says why its log cannot be read."""
    try:
        return values()
    except Evidence as e:
        return {name: e for name in names}


def v_under_resolved_probe() -> dict:
    found = []
    for name in ("d2_t07", "probe_nu0006_d2"):
        r = packed(UNDER_RESOLVED, name)
        if r["probe_largest_difference"] == "":
            raise Evidence(f"{UNDER_RESOLVED}: no difference at the probe for {name}")
        found.append(f"{sci(number(r, 'probe_largest_difference', UNDER_RESOLVED, name))} for `{name}`")
    return {"differences": " and ".join(found)}


def v_under_resolved() -> dict:
    p = encoded(UNDER_RESOLVED, "d3")
    code = packed(UNDER_RESOLVED, "d3_t05")["pack_exit_code"]
    return {"error": pct(p.error), "outcome": {"0": "held", "1": "refused", "4": "missed"}.get(code, f"ended with code {code}")}


def v_volume_drift() -> dict:
    def offsets() -> dict:
        file = f"{TRACERS}/offsets_256_summary.csv"
        rows = [r for r in read_table(file, ["path", "fd", "var_d_growth_per_tau"]) if r["path"] == "tex"]
        if sorted((r["fd"] for r in rows), key=float) != ["0.0625", "0.25"]:
            raise Evidence(f"{file}: the texture is expected once at each of the offsets 0.25 and 0.0625")
        rates = [number(r, "var_d_growth_per_tau", file, f"texture, offset {r['fd']}") for r in rows]
        return {"low": sci(min(rates)), "high": sci(max(rates))}

    def long_run() -> dict:
        lines, path = tracer_summary("long_256")
        psi, texture, zfp = growth(lines, path, "psi3"), growth(lines, path, "tex"), growth(lines, path, "zfp2")
        return {"long_times": lines["psi3"]["last_t_tau"], "long_psi": sci(psi), "long_texture": sci(texture),
                "long_texture_times": times(texture, psi), "long_zfp": sci(zfp), "long_zfp_times": times(zfp, psi)}
    values = available(offsets, ["low", "high"])
    values.update(available(long_run, ["long_times", "long_psi", "long_texture", "long_texture_times", "long_zfp", "long_zfp_times"]))
    return values


def v_confirmation() -> dict:
    lines, path = tracer_summary("confirmation_256")
    table = criterion_excess(lines, path)
    reference = [e for name, e in table.items() if name.startswith("ref")]
    codec = [abs(table[name]) for name in ("psi3", "psi2", "psi3b") if name in table]
    if len(reference) != 1 or len(codec) != 3:
        raise Evidence(f"{path}: one reference path and the three paths of the codec are expected on five seeds")
    name = next(n for n in table if n.startswith("ref"))
    values = {"reference": f"{reference[0]:+.1f}", "codec_paths": fixed(max(codec))}

    def ten_seeds() -> dict:
        excess = excess_over_seeds(["confirmation_256", "confirmation_256_more_seeds"], name, [str(s) for s in range(1, 11)])
        return {"ten_seeds": f"{excess:+.1f}"}
    values.update(available(ten_seeds, ["ten_seeds"]))
    return values


def v_bench_256() -> dict:
    points, path = bench(256)
    ours = [bench_point(points, path, PSI, b)[:2] for b in ("0.05", "0.04", "0.03")]
    theirs = [bench_point(points, path, "sz3_4d", b)[:2] for b in ("0.05", "0.04", "0.03")]
    tight_ratio, tight_error, _ = bench_point(points, path, "sz3_4d", "0.015")
    more = [s[0] / p[0] for p, s in zip(ours, theirs)]
    higher = [s[1] / p[1] for p, s in zip(ours, theirs)]
    zfp = [number(bench_point(points, path, "zfp", b)[2], "largest_error_over_urms", path, f"zfp at {b}") for b in ("0.05", "0.04", "0.03")]
    return {"window_low": fixed(100 * min(e for _, e in ours), 2), "window_high": fixed(100 * max(e for _, e in ours), 2) + " %",
            "tight_error": pct(tight_error, 2), "tight_ratio": fixed(tight_ratio), "more_low": fixed(min(more)), "more_high": fixed(max(more)),
            "more_at_5": fixed(more[0]),
            "higher_low": fixed(min(higher)), "higher_high": fixed(max(higher)), "zfp_largest": three(zfp, lambda x: fixed(100 * x, 2)) + " %"}


def v_bench_512() -> dict:
    def at_512() -> dict:
        points, path = bench(512)
        ratio, _, _ = bench_point(points, path, PSI, "0.05")
        ratio_s, _, line = bench_point(points, path, "sz3_4d", "0.05")
        return {"block": line["block_frames"], "sz3_512": fixed(ratio_s), "psi_512": fixed(ratio), "times_512": times(ratio_s, ratio, 2)}
    values = available(v_bench_256, ["more_at_5"])
    values.update(available(at_512, ["block", "sz3_512", "psi_512", "times_512"]))
    return values


def v_grid_divergence() -> dict:
    def m1(chain: str) -> str:
        return three(by_realisation(DIVERGENCE, "grid_divergence_over_gradient", chain))
    return {"zfp": m1("zfp"), "cuszp": m1("cuszp"), "cuzfp": m1("cuzfp_2bits"), "psi": m1("psi"), "psi_free": m1("psi_free")}


def v_read_divergence() -> dict:
    def m2(chain: str) -> str:
        return three(by_realisation(DIVERGENCE, "read_divergence_over_gradient", chain))
    return {"zfp": m2("zfp"), "cuszp": m2("cuszp"), "cuzfp": m2("cuzfp_2bits")}


def v_tracers_after_compressors() -> dict:
    lines, path = compressor_campaign()
    table = criterion_excess(lines, path)
    values = {name.replace("3b", ""): f"{table[name]:+.1f}" for name in ("sz3", "sz3_4d", "sperr", "zfp2", "tex", "psi3b")}
    values["zfp_growth"] = sci(growth(lines, path, "zfp2"))
    return values


def v_spectrum() -> dict:
    line = chain_lines(SPECTRA, ["enstrophy_over_original", "k10"])
    chains = ("sz3", "sz3_4d", "sperr", "zfp", "cuszp", "psi", "cuzfp_2bits")
    return {"others": "; ".join(f"`{c}` " + ", ".join(f"{fixed(number(line(s, c), 'enstrophy_over_original', SPECTRA, c), 4)} and {line(s, c)['k10']}"
                                                      for s in SOLVER_SEEDS) for c in chains)}


# --- Registry --------------------------------------------------------------------------------------------------------

Text = Union[str, Filled]


@dataclass
class Claim:
    ident: str
    title: str
    statement: str
    criterion: str
    fixed: str
    expected: str
    evidence: str = ""                    # the log (measured claim)
    replay: str = ""                      # the command that writes the log again
    caveat: Text = ""
    measure: Optional[Callable] = None
    basis: Text = ""                      # pending claim: what the hypothesis rests on
    needed: Text = ""                     # pending claim: what is needed first


REFERENCE_FOLDER = "benchmarks/compression/reference_files"
REPLAY_REFERENCE = f"`python {REFERENCE_FOLDER}/reference_files.py`"
REPLAY_UNDER_RESOLVED = "`python benchmarks/compression/under_resolved/under_resolved.py`"
REPLAY_TRACERS = ("`python benchmarks/tracers/tracers.py run resolved_256 resolved_512` (GPU; it runs the campaigns, then writes "
                  "`<campaign>_analysis.txt` and `<campaign>_summary.csv`)")
EVIDENCE_TRACERS = (f"`{TRACERS}/resolved_256_summary.csv`, `resolved_512_summary.csv` (lines of the tracer program: `resolved_256.csv`, "
                    "`resolved_512.csv`; tables: `resolved_256_analysis.txt`, `resolved_512_analysis.txt`)")
EVIDENCE_VALIDITY = f"`{VALIDITY}` (output of each run in `benchmarks/validity/logs/`)"
REPLAY_VALIDITY = "`python benchmarks/validity/validity.py simulate`, then `measure` (GPU; 18 sequences of 9.0 GiB each, not versioned)"
REPLAY_BENCH = "`python benchmarks/equal_bound/equal_bound.py 256` and `512` (GPU; needs the comparison codecs of `build/codecs/`)"
REPLAY_DIVERGENCE = ("`python benchmarks/divergence/divergence.py run` (GPU for the chains decoded there; needs the comparison codecs of "
                     "`build/codecs/` and the three sequences `data/sweep/nu00280_seed<seed>.seq`)")
SWEEP = ("Sweep of `benchmarks/validity/validity.py`: 18 sequences at 256³ (6 viscosities from 0.0011 to 0.0028; seeds 42, 7 and 11), "
         "frames 8 to 43. ")
ONE_VOXEL_BINS = (" The campaigns replayed on 17 November 2026 count bins of 1 voxel directly at 256³; the first campaign counted bins of "
                  "half a voxel and merged them: the count per bin of 1 voxel is the same quantity.")
PROTOCOL = (" Protocol corrected before the re-measurement of 17 November 2026, the criterion being kept as it was: every codec is measured "
            "on the same 32 frames (frames 16 to 47 of `turb256.seq`, frames 0 to 31 of `turb512.seq`), where the frame-by-frame codecs "
            "were measured on four frames at 256³ and two at 512³, one of them outside the group of Psi-treillis; their compression ratio "
            "is the total raw size over the total compressed size, where it was the mean of the ratios of the frames measured; the error "
            "of Psi-treillis is no longer the one its encoder reports: its file is decoded through the public interface and compared with "
            "the raw frames by the same code as the other codecs; at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames (63 GiB "
            "of RAM do not hold a block of 32); zfp is reported apart, its fixed-accuracy mode leaving the largest error well under the "
            "bound requested.")
RENAMED_17 = " The files of the three realisations were named `v_nu00280_g<seed>.seq` when the criterion was written."
DEFINED_ABOVE = (" The realisations, the bound, the chains and the instruments M1 to M4 are those of the section \"Data, settings and "
                 "instruments of the claims fixed on 17 November 2026\" of this sheet.")
JHTDB_ACCESS = (" Access to the database was obtained on 16 November 2026, but the transfers of the cutout broke repeatedly and no complete "
                "frame was retrieved; the scripts are in `benchmarks/cutout/`.")

REGISTRY = [
    Claim(
        "COMP-1", "Compression without a bound, degree 3",
        "At 512³, with the default settings in degree 3, Psi-treillis compresses at least 140 times for a field error of at most 0.65 %.",
        "Compression ratio ≥ 140 and mean field error ≤ 0.65 %, on frames 0 to 54 of `turb512.seq`.", AFTER, HOLDS,
        f"`{REFERENCE}`, line `z_t512_d3` (output of the encoder: `{REFERENCE_FOLDER}/logs/pack_z_t512_d3.log`)", REPLAY_REFERENCE,
        "The compression ratio is that of a file of 55 frames with a key frame every 32 frames: it is inter-frame.",
        m_compression(3, 140, 0.0065)),
    Claim(
        "COMP-2", "Compression without a bound, degree 2",
        "At 512³, with the default settings in degree 2, Psi-treillis compresses at least 115 times for a field error of at most 0.70 %.",
        "Compression ratio ≥ 115 and mean field error ≤ 0.70 %, on frames 0 to 54 of `turb512.seq`.", AFTER, HOLDS,
        f"`{REFERENCE}`, line `z_t512_d2` (output of the encoder: `{REFERENCE_FOLDER}/logs/pack_z_t512_d2.log`)", REPLAY_REFERENCE, "",
        m_compression(2, 115, 0.0070)),
    Claim(
        "BOUND-1", "Certified bound, degree 3",
        "In degree 3, `pack --tolerance` holds a bound of 5, 4 and 3 % of u′ on the largest error of a component, at 256³ and at 512³.",
        "For each of the six files: no frame above the certification threshold, checked on all the voxels of all the frames.",
        AFTER, HOLDS, f"`{REFERENCE}`, lines `f_t256_d3_t0*` and `f_t512_d3_t0*` (outputs of the encoder: `{REFERENCE_FOLDER}/logs/pack_f_t*_d3_t0*.log`)",
        REPLAY_REFERENCE,
        "The bound holds at voxel centres and at the times of the frames; between frames and off the grid, nothing is guaranteed. "
        "The certification margin is empirical.", m_bound(DEGREE3)),
    Claim(
        "BOUND-2", "Cost of the bound, degree 3",
        "With this bound, the compression ratio remains at least 130 times at 5 %, 120 times at 4 % and 95 times at 3 %.",
        "Compression ratio of the whole file ≥ 130, 120 and 95 according to the tolerance, at 256³ and at 512³.", AFTER, HOLDS,
        f"`{REFERENCE}`, lines `f_t256_d3_t0*` and `f_t512_d3_t0*` (outputs of the encoder: `{REFERENCE_FOLDER}/logs/pack_f_t*_d3_t0*.log`)",
        REPLAY_REFERENCE, "", m_cost_of_the_bound(DEGREE3, {5: 130, 4: 120, 3: 95})),
    Claim(
        "BOUND-3", "Certified bound, degree 2",
        "In degree 2, the bound is held at 6 % of u′ at 256³ and at 7 % at 512³.",
        "No frame above the certification threshold in the two files.", AFTER, HOLDS,
        f"`{REFERENCE}`, lines `f_t256_d2_t06` and `f_t512_d2_t07` (outputs of the encoder: `{REFERENCE_FOLDER}/logs/pack_f_t256_d2_t06.log`, "
        "`pack_f_t512_d2_t07.log`)", REPLAY_REFERENCE,
        "See BOUND-4: one point lower, it is missed.", m_bound(DEGREE2_HIGH)),
    Claim(
        "READBACK-1", "Read-back consistent with the encoding",
        "The reader gives back the state of the encoder: no frame differs, and the same field off the grid.",
        "On the twelve reference files: 0 frames whose state differs; on the ten files with a bound: zero difference at the off-grid probe (114,688 points).",
        AFTER, HOLDS, f"`{REFERENCE}`, columns `frames_differing` and `probe_largest_difference` (outputs of the reader: `{REFERENCE_FOLDER}/logs/play_*.log`)",
        REPLAY_REFERENCE,
        Filled("A rounding difference between reader and encoder exists on the less resolved sequence, in degree 2: {differences}, in "
               f"velocity units, u′ being 0.850 on that sequence (`{UNDER_RESOLVED}`, column `probe_largest_difference`). It is not "
               "explained.", v_under_resolved_probe), m_readback),
    Claim(
        "DIV-1", "Zero divergence",
        "The field read back has zero divergence, up to rounding.",
        "RMS divergence of the analytic gradient, in double precision, ≤ 10⁻¹² of the RMS gradient.", AFTER, HOLDS,
        f"`{PLAYBACK}`, column `divergence_over_gradient` of the sides that show Psi-treillis (outputs of the comparator: "
        "`benchmarks/playback/logs/view_*.log`)", "`python benchmarks/playback/playback.py run` (GPU running nothing else; `psi_view` exists on Windows only)",
        "Measured on a slice of one frame. By finite differences, one measures the truncation of the scheme, not the field. Since "
        "17 November 2026 the figure is read in the log of the playback measurement, on files of degree 3 without a bound, at 256³ "
        "and at 512³; the earlier log was one capture of the visual comparator, which showed a file without a bound and a file with a "
        "bound of 3 %: a file with a bound is no longer among those measured here.", m_divergence),
    Claim(
        "SPEED-1", "Reading speed",
        "At 512³ in degree 3, on the measurement machine, a frame is decoded in less than 4 ms and one RK4 step of 4 million sorted "
        "particles takes less than 4.5 ms, without a bound and with a bound of 5 %.",
        "Medians of 15 repetitions of the current reader: decoding ≤ 4.0 ms and RK4 step ≤ 4.5 ms, on both files.", AFTER, HOLDS,
        f"`{SPEED}`", "`python benchmarks/speed/read_timing.py` (see `benchmarks/speed/README.md`)",
        "A single machine (RTX 5090 Laptop), GPU running no other computation, system cache warm. Degree 2 has not been measured again with the current reader.",
        m_speed(4.0, 4.5)),
    Claim(
        "ENC-1", "Encoding speed",
        "At 512³ in degree 3, on the measurement machine, encoding takes less than 300 ms per frame with a bound of 5 % and less than 450 ms with a bound of 3 %.",
        "Encoding time per frame ≤ 300 ms and ≤ 450 ms, excluding the reading of the raw frames from disk.", AFTER, HOLDS,
        f"`{REFERENCE}`, lines `f_t512_d3_t05` and `f_t512_d3_t03`, column `encoding_ms_per_frame` (outputs of the encoder: "
        f"`{REFERENCE_FOLDER}/logs/pack_f_t512_d3_t05.log`, `pack_f_t512_d3_t03.log`)", REPLAY_REFERENCE,
        "A single run per case, a single machine.", m_encoding({"t512_d3_t05": 300, "t512_d3_t03": 450})),
    Claim(
        "LAGR-1", "Clustering of tracers under cuZFP",
        "Tracers advected in a frozen frame seen through cuZFP at 2 bits cluster measurably.",
        "Excess dispersion of the count in bins of 1 voxel, after 10 Kolmogorov times, seeds 1 to 5: more than 5 standard errors, at 256³ and at 512³.",
        BEFORE_TRACERS, HOLDS, EVIDENCE_TRACERS, REPLAY_TRACERS,
        "Frozen field, one frame per resolution, tracers without inertia." + ONE_VOXEL_BINS,
        m_dispersion(lambda name: name == "zfp2", lambda e: e > 5, 2)),
    Claim(
        "LAGR-2", "No clustering under Psi-treillis",
        "In the same protocol, no path of Psi-treillis shows clustering.",
        "Excess dispersion below 2 standard errors for each Ψ path measured on five seeds.", BEFORE_TRACERS, HOLDS,
        EVIDENCE_TRACERS, REPLAY_TRACERS,
        "See LAGR-4: the trilinear texture does not pass the threshold of 5 either, so this protocol does not separate Psi-treillis from the texture."
        + ONE_VOXEL_BINS, m_dispersion(is_psi, lambda e: abs(e) < 2, 5)),
    Claim(
        "LAGR-3", "Volume carried by a tracer",
        "Along the trajectories, the variance of D (integral of the divergence, that is the logarithm of the volume carried) grows at least "
        "1,000 times more slowly with Psi-treillis of degree 3 than with the trilinear texture or cuZFP at 2 bits.",
        "Growth of var(D) per Kolmogorov time: that of the texture and that of cuZFP ≥ 1,000 times that of Psi-treillis, at 256³ and at 512³.",
        AFTER, HOLDS, EVIDENCE_TRACERS, REPLAY_TRACERS,
        Filled("For Psi-treillis, D measures only the truncation of the probe by differences (±1/8 voxel); for the texture, the value "
               f"depends on this offset ({{low}} to {{high}} at ±1/4 and ±1/16 voxel, campaign `offsets_256`, `{TRACERS}/offsets_256_summary.csv`). "
               "Descriptive, on one seed, 1 tracer per voxel and {long_times} Kolmogorov times (campaign `long_256`): Ψ {long_psi}, texture "
               "{long_texture} ({long_texture_times} times), cuZFP {long_zfp} ({long_zfp_times} times).", v_volume_drift),
        m_volume_drift(1000)),
    Claim(
        "ROBUST-1", "Reader faced with an altered file",
        "An altered file is refused, or read while counting the frames that differ; it causes neither a CUDA error, nor a crash, nor a hang.",
        'On the 283 altered files at 64³, the only outcomes of the current reader are "refused", "counted" and "read".', AFTER, HOLDS,
        f"`{ALTERED}` (outputs of the reader: `benchmarks/robustness/logs/altered_files_*.log`)",
        '`python tests/test_codec.py --keep-alterations --only "guard.geometry_no_seal" "guard.random_alteration"`, then '
        "`python benchmarks/robustness/altered_files.py`",
        "64³ only, one alteration at a time. Four alterations of the header or of the table of a file without a seal are not seen, "
        "nor is the altered value of an overflow of level 0 (case `guard.altered_overflow`).", m_altered),
    Claim(
        "TEST-1", "Determinism and non-regression",
        "The same options give the same bytes, and the regression suite at 64³ passes.",
        "`tests/test_codec.py`: no `FAIL`; cases `d2.determinism`, `d3.determinism` and `sequence.fingerprint` marked `OK`.", AFTER, HOLDS,
        f"`{TEST_CODEC}`", f"`python tests/test_codec.py > {TEST_CODEC}`",
        "The fingerprints are valid only for the toolchain described in `tests/fingerprints.json`. See PORT-1.", m_tests),
    Claim(
        "API-1", "Public interface, round trip at 64³",
        "Through the public interface (library, Python module, `psit` tool), the compressed file is that of `pack` byte for byte, "
        "the decoded field respects the requested bound, off-grid sampling returns a divergence-free field and the GPU memory "
        "is released on closing.",
        "`tests/test_api.py`: no failed case; cases `cli.same_bytes_as_pack`, `api.bound_is_held`, `api.sample_matches_the_grid`, "
        "`api.divergence_free` and `api.memory_is_released` passed.", AFTER, HOLDS,
        f"`{TEST_API}`", f"`python tests/test_api.py > {TEST_API}`",
        "64³ and Windows only. The errors are measured by the test in NumPy, outside the code of the project. The \"bound missed\" exit "
        "(code 4) cannot be reached at 64³ through the public options: it is not covered.", m_api),
    Claim(
        "API-2", "Errors read back at 512³ through another code path",
        "At 512³, the field decoded to the grid by the library, compared in NumPy with the raw frames, has the error that the encoder "
        "recorded in the file, and it is within the tolerance of each file whose bound is held.",
        "On four frames of six reference files: largest error less than 5·10⁻⁵ u′ from the recorded value, field error "
        "to within 0.02 %, and largest error ≤ tolerance.", AFTER, HOLDS,
        f"`{BOUNDS_512}` (the same lines, with the version of the library: `verify_bounds_512.log`)",
        "`python benchmarks/interface/bounds_512/verify_bounds_512.py`",
        "Four frames per file out of 55; the encoder, for its part, checks all the frames (BOUND-1). The kernel that decodes to the grid "
        "is that of the library; the comparison is made on the host in double precision.", m_api_512),
    Claim(
        "BENCH-1", "Against SPERR, at an equal bound",
        "At an equal bound on the largest error, Psi-treillis of degree 3 compresses at least 15 % more than SPERR, with no higher field error.",
        "Compression ratio ≥ 1.15 times that of SPERR and field error ≤ 1.01 times that of SPERR, at 5, 4 and 3 % (256³) and at 5 and 4 % (512³).", AFTER, HOLDS,
        f"`{EQUAL_BOUND.format(n=256)}`, `equal_bound_512.csv`", REPLAY_BENCH,
        "Psi-treillis is measured on a group of 32 frames, with temporal prediction; SPERR is measured frame by frame: the lead is that "
        "of temporal prediction." + PROTOCOL, m_against_sperr(1.15, 1.01)),
    Claim(
        "BENCH-2", "Pareto front at 256³",
        "At 256³, the three default points of Psi-treillis of degree 3 (bounds of 5, 4 and 3 %) are not dominated by any measured point of "
        "SZ3, SZ3 in 4D, SPERR, zfp or cuSZp.",
        "No point of another codec has both a compression ratio at least equal and a field error at most equal.", AFTER, HOLDS,
        f"`{EQUAL_BOUND.format(n=256)}`", REPLAY_BENCH,
        Filled("The window is narrow: field error from {window_low} to {window_high}. The largest error does not enter this criterion: with a "
               "bound of 1.5 %, SZ3 in 4D has a field error of {tight_error}, for a compression ratio of {tight_ratio}×. The points of zfp "
               "are among those of the other codecs, as the statement names it, although it is not at an equal attained bound: asked for "
               "5, 4 and 3 %, its largest error is {zfp_largest} of u′." + PROTOCOL, v_bench_256), m_pareto_front),
    Claim(
        "REF-1", "A reader independent of the code",
        "A `.psi` file can be read from its specification alone.",
        "A CPU reader written from `docs/FORMAT.md`, without reading `src/`, gives back for each frame of the twelve reference files "
        "the state fingerprint recorded in the table of the file, and a velocity field less than 5·10⁻⁵ u′ from that of the GPU reader at the "
        "points of the probe.", BEFORE_16, HOLDS,
        f"`{READER_CPP}` (reader: `tools/reference_reader.cpp`, program `psi_ref`)", "`python benchmarks/independent_readers/cpp/reference_cpp.py`",
        "The statement holds in full only for the state of the frames, decoded from the specification alone. For the velocity field, the "
        "specification of the time was not sufficient: the masks and the rounding of the refinement, the position of the nodes and the format of the probe "
        "were established by trial on files, without reading the code (list of the gaps: `benchmarks/independent_readers/cpp/README.md`). They "
        "were then carried over into `docs/FORMAT.md` (sections 6.5, 6.7 and 11), then cross-checked against the code, with which they agree; "
        "no reader had yet been written from the text thus completed when this claim was first measured: that reader is the one of REF-2. "
        "A single frame probed per file. The GPU reader is compared only through the probe of the encoder, from which it is at 0. The difference grows with the "
        "size of the grid and is more than half the threshold at 512³; it is attributed to the single precision of the GPU reader, without "
        "proof. The threshold is half the certification margin; the largest rounding disagreement recorded between two GPU kernels "
        "is 2.3·10⁻⁵ u′.", m_reference_cpp(5e-5)),
    Claim(
        "LAGR-4", "Clustering under the trilinear texture",
        "In the protocol of LAGR-1, the trilinear texture causes a clustering that Psi-treillis avoids.",
        "Excess dispersion of the trilinear texture above 5 standard errors, at 256³ and at 512³.", BEFORE_TRACERS, DOES_NOT_HOLD,
        EVIDENCE_TRACERS, REPLAY_TRACERS,
        "On resolved data, this protocol sees no effect of the divergence of the texture at the scale of the voxel. Only LAGR-3 "
        "separates Psi-treillis from the texture." + ONE_VOXEL_BINS, m_dispersion(lambda name: name == "tex", lambda e: e > 5, 2)),
    Claim(
        "BOUND-4", "Certified bound, degree 2, one point lower",
        "In degree 2, the bound is held at 5 % of u′ at 256³ and at 6 % at 512³.",
        "No frame above the certification threshold in the two files.", AFTER, DOES_NOT_HOLD,
        f"`{REFERENCE}`, lines `f_t256_d2_t05` and `f_t512_d2_t06` (outputs of the encoder: `{REFERENCE_FOLDER}/logs/pack_f_t256_d2_t05.log`, "
        "`pack_f_t512_d2_t06.log`)", REPLAY_REFERENCE,
        "Floor of the fine level: one voxel, in one frame, that it does not bring back below the threshold once all its nodes are retained.",
        m_bound(DEGREE2_LOW)),
    Claim(
        "RESO-1", "Fidelity on a less resolved turbulence",
        "The result of COMP-1 holds on a turbulence whose Kolmogorov scale is 0.45 voxel instead of 0.74.",
        "On `turb512_nu0006.seq`, same settings, same frames: compression ratio ≥ 140 and field error ≤ 0.65 %.", AFTER, DOES_NOT_HOLD,
        f"`{UNDER_RESOLVED}`, line `d3` (output of the encoder: `benchmarks/compression/under_resolved/logs/pack_d3.log`)", REPLAY_UNDER_RESOLVED,
        "This is the resolution, in voxels, of the public JHTDB dataset at 1,024³ (0.47 voxel, calculated from its description).",
        m_under_resolved_fidelity(140, 0.0065)),
    Claim(
        "RESO-2", "Certified bound on a less resolved turbulence",
        "On this same sequence, the bound of BOUND-1 remains attainable in degree 3.",
        "`pack --tolerance` exits with code 0 at 5, 4 and 3 %.", AFTER, DOES_NOT_HOLD,
        f"`{UNDER_RESOLVED}`, column `pack_exit_code` of the lines `d3_t05`, `d3_t04`, `d3_t03` and `d2_t07`", REPLAY_UNDER_RESOLVED,
        "The lattice step of 2 voxels is 4.4 Kolmogorov scales on this sequence, against 2.7 on the other.", m_under_resolved_bound),
    Claim(
        "BENCH-3", "Against SZ3 in 4D, at an equal bound",
        "At an equal bound, Psi-treillis compresses at least as much as SZ3 in 4D on 32 frames.",
        "Compression ratio of Psi-treillis of degree 3 ≥ that of SZ3 in 4D, at 5, 4 and 3 %, at 256³.", AFTER, DOES_NOT_HOLD,
        f"`{EQUAL_BOUND.format(n=256)}`", REPLAY_BENCH,
        Filled("SZ3 in 4D compresses {more_low} to {more_high} times more, with a field error {higher_low} to {higher_high} times higher. At "
               "512³, it is measured by blocks of 8 frames only, which this criterion does not read." + PROTOCOL, v_bench_256),
        m_against_sz3_4d),
    Claim(
        "JHTDB-1", "Expected degradation on JHTDB",
        "On the public JHTDB dataset of isotropic turbulence at 1,024³, with the default settings in degree 3, the field error is more than "
        "twice that of `turb512.seq` and the bound of 5 % of u′ is not held.",
        "On at least 8 consecutive frames: mean field error > 1.28 % (twice the 0.640 % of COMP-1) and `pack --tolerance 0.05` "
        "exits with a non-zero code. The hypothesis is refuted if either of the two conditions is missing.", BEFORE_16, PENDING,
        basis=Filled("RESO-1 and RESO-2: at the same resolution in voxels, the field error is {error} and the bound is {outcome}.", v_under_resolved),
        needed="The encoder does not handle 1,024³ (about 31 GiB of GPU memory, calculation, for a GPU of 24 GiB), and a sub-cube is not "
        "periodic whereas the encoder relies on a periodic FFT. Since 16 November 2026, not testable under the access conditions obtained: the database "
        "limits a cutout to 3 GB from a local machine and reserves its use for small subsets, whereas 8 whole frames weigh about a hundred GB "
        "(calculation). JHTDB-3 replaces it with a cutout." + JHTDB_ACCESS),
    Claim(
        "JHTDB-2", "Usefulness for compression on JHTDB",
        "On this same dataset, there is a setting of Psi-treillis that holds a certified bound of 5 % of u′ or less and compresses at least as much as SPERR at the same bound.",
        "For one tolerance among 5, 4 and 3 %: `pack` exits with code 0 and its compression ratio is ≥ that of SPERR set to the same absolute bound, "
        "on the same frames. All the settings tried are reported, not only the best.", BEFORE_16, PENDING,
        basis="BENCH-1 on resolved data; RESO-2 leads one to expect the opposite with the lattice of step 2 voxels. I do not know which verdict will come out.",
        needed="The prerequisites of JHTDB-1, and a lattice of step 1 voxel (`--mf 1`) that fits in memory at this size. If the claim "
        "does not hold, what remains to Psi-treillis, on these data, is zero divergence and decoding on the GPU, not the compression ratio. Not testable "
        "under the access conditions obtained, for the same reason as JHTDB-1: JHTDB-4 replaces it with a cutout." + JHTDB_ACCESS),
    Claim(
        "PORT-1", "Another machine",
        "The chain gives the same test verdicts on another GPU.",
        "On an NVIDIA GPU of an architecture other than that of `tests/fingerprints.json`: `tests/test_codec.py` counts no `FAIL` "
        "outside the fingerprint cases (`*.fingerprint.*`, `sequence.fingerprint`).", BEFORE_16, PENDING,
        basis="`tests/README.md` describes the procedure; it has never been followed anywhere other than on the measurement machine.",
        needed="A second machine. Under Linux, a build on a machine with a GPU is needed first: the project has been measured only under Windows "
        "with MSVC, and compiled under Linux only on a machine without a GPU (LINUX-1). The CPU reference reader was compiled with GCC 13.3 "
        "and run on Linux on 17 November 2026 on one 64³ test file (0 fingerprint differing), which is not this criterion."),
    Claim(
        "LAGR-5", "Lagrangian confirmation on an independent realisation",
        "The verdicts of LAGR-1 to LAGR-4 are found again on another realisation of the turbulence, all the parameters being fixed in advance.",
        "256³ sequence produced by `dns --seed 7` (other options as for `turb256.seq`), frame 32, 8 tracers per voxel, bins of 1 voxel, "
        "10 Kolmogorov times, tracer seeds 1 to 5, time step of half a frame: cuZFP at 2 bits above 5 standard errors; "
        "each Ψ path below 2; trilinear texture below 5; growth of var(D) of the texture ≥ 1,000 times that of Psi-treillis of degree 3.",
        BEFORE_16, DOES_NOT_HOLD,
        f"`{TRACERS}/confirmation_256_summary.csv` (lines of the tracer program: `confirmation_256.csv`; tables: `confirmation_256_analysis.txt`)",
        "`python benchmarks/validity/validity.py simulate` for the sequence `data/sweep/nu00280_seed7.seq`, then "
        "`python benchmarks/tracers/tracers.py run confirmation_256 confirmation_256_more_seeds` (GPU)",
        Filled("Three of the four conditions are met as predicted: cuZFP clusters, the trilinear texture stays under the threshold, and the "
               "volume drift separates the texture from Psi-treillis. The fourth fails on one path, read strictly. The criterion says "
               "\"below 2\", the analysis script that defines it says \"in absolute value\", and the reference path at a step of 1 voxel is "
               "at {reference}: an under-dispersion, not a clustering. Read as a signed threshold, the claim holds. On ten seeds instead of five, "
               "run after seeing this result and therefore descriptive only, the same path is at {ten_seeds} (seeds 6 to 10: campaign "
               "`confirmation_256_more_seeds`; calculation of `tools/claims.py` on the lines of the two campaigns, with the arithmetic of the "
               "analysis): the five seeds of the criterion look like a fluctuation, which the criterion does not allow us to discount. The "
               "three paths of the codec itself (degree 3, degree 2, degree 3 with a bound) are within {codec_paths}. Bins of 1 voxel were counted "
               "directly, where the first campaign counted bins of half a voxel and merged them. The sequence has 48 frames instead of 128, "
               "which does not change a frame: with the default seed, frames 0, 32 and 47 of the 48-frame run are byte-identical to those of "
               "`turb256.seq` (`benchmarks/validity/identity_turb256.txt`, written by `python benchmarks/validity/validity.py identity`). The "
               "sequence was named `v_nu00280_g7.seq` when the criterion was written.", v_confirmation), m_confirmation(1000)),
    Claim(
        "BENCH-4", "SZ3 in 4D on 32 frames at 512³",
        "At 512³, SZ3 in 4D on 32 frames compresses at least 1.5 times more than Psi-treillis of degree 3 at an equal bound of 5 %.",
        "Compression ratio of SZ3 in 4D on frames 0 to 31 of `turb512.seq` ≥ 1.5 times that of Psi-treillis on the same frames, both measured "
        "with the current encoder.", BEFORE_16, PENDING,
        basis=Filled("BENCH-3 at 256³ ({more_at_5} times at 5 %); at 512³, by blocks of {block} frames over frames 0 to 31, SZ3 in 4D gives "
                     "{sz3_512}× against {psi_512}× for Psi-treillis on one group of 32 frames, that is {times_512} times "
                     f"(`{EQUAL_BOUND.format(n=512)}`).", v_bench_512),
        needed="More than 63 GiB of RAM, or a variant of SZ3 that processes the block out of core. The bench of BENCH-1 to BENCH-3 was "
        "measured again on 17 November 2026 with the corrected protocol: at 512³, SZ3 in 4D takes four consecutive blocks of 8 frames, which "
        "is not the single block of 32 frames of this criterion."),
    Claim(
        "VALID-1", "Resolution threshold of the bound of 5 %",
        "At a step of 2 voxels in degree 3, holding a bound of 5 % of u′ depends only on the resolution of the data: the bound is held "
        "above a threshold of the Kolmogorov scale, missed or refused below it, and this threshold is between 0.45 and 0.74 voxel.",
        SWEEP + "There is a resolution η* such that `pack --degree 3 --tolerance 0.05` exits with code 0 on all the "
        "sequences with η > η* and with a non-zero code on all those with η < η*; η*, the midpoint between the largest resolution that fails and "
        "the smallest that succeeds, is between 0.45 and 0.74 voxel. Refuted if the two sets overlap, if one of them is empty, or if "
        "η* falls outside the interval.", BEFORE_16, HOLDS,
        EVIDENCE_VALIDITY, REPLAY_VALIDITY,
        "The threshold is that of this sweep: 256³, one solver, lattice step of 2 voxels, bound of 5 %. The three realisations of one "
        "viscosity (0.0019) fall on both sides of it: the bound is refused at 0.530 voxel and held at 0.543 and 0.552. For a bound of "
        "3 %, the same sweep puts the threshold between 0.643 and 0.702 voxel: refused on 15 sequences, held on 3. The Kolmogorov "
        "scale is that of the header of each sequence, computed by the solver from the dissipation of the run.",
        m_bound_threshold(0.45, 0.74)),
    Claim(
        "VALID-2", "Ahead of SPERR where the bound holds",
        "On each sequence where the bound of 5 % is held (degree 3, step of 2 voxels), Psi-treillis compresses at least as much as SPERR "
        "at the same bound.",
        SWEEP + "For each sequence where `pack --degree 3 --tolerance 0.05` exits with code 0: compression ratio of the group of 32 frames ≥ compression ratio "
        "of SPERR at the same absolute bound (harmonic mean over frames 8, 24 and 39). Does not hold if no sequence holds the bound.",
        BEFORE_16, DOES_NOT_HOLD, EVIDENCE_VALIDITY, REPLAY_VALIDITY,
        "Psi-treillis is ahead of SPERR on the 6 sequences at 0.61 voxel per Kolmogorov scale or more, and behind it on the two just "
        "above the threshold of VALID-1 (0.543 and 0.552 voxel). There the bound is held at the price of a fine level that costs "
        "almost half of the file: the compression ratio falls from 105 to 107 times without the bound to 56 to 60 times with it. "
        "SPERR is applied frame by frame; Psi-treillis uses the temporal prediction.", m_ahead_of_sperr_under_bound),
    Claim(
        "VALID-3", "The step of 1 voxel holds the bound everywhere, without catching up on the compression ratio",
        "At a step of 1 voxel in degree 3, the bound of 5 % is held whatever the resolution of the sweep, but the compression ratio then remains below "
        "that of SPERR at the same bound.",
        SWEEP + "On the 18 sequences: `pack --degree 3 --mf 1 --tolerance 0.05` exits with code 0, and its compression ratio for the group of 32 "
        "frames is lower than that of SPERR at the same bound. Refuted if one sequence misses the bound, or if the compression ratio reaches that of "
        "SPERR on one sequence.", BEFORE_16, HOLDS,
        EVIDENCE_VALIDITY, REPLAY_VALIDITY,
        "This claim holds, and it is not good news for the codec: a step of 1 voxel keeps the bound on every sequence, down to "
        "0.36 voxel per Kolmogorov scale, but at about half the compression ratio of SPERR. Its field error is then 0.39 to 0.55 %, "
        "lower than that of SPERR at the same bound.", m_step_of_one_voxel),
    Claim(
        "VALID-4", "Field error as a function of the resolution",
        "Without a bound, the field error of degree 3 at a step of 2 voxels remains below 0.70 % on resolved data and exceeds twice "
        "the value of COMP-1 on under-resolved data.",
        SWEEP + "Field error ≤ 0.70 % for every sequence with η ≥ 0.69 voxel, and > 1.28 % for every sequence with η ≤ 0.45 voxel. "
        "At least one sequence is needed in each range.", BEFORE_16, DOES_NOT_HOLD,
        EVIDENCE_VALIDITY, REPLAY_VALIDITY,
        "The second half is met, the first is not: one of the three sequences at 0.69 voxel or more is at 0.76 % (0.702 voxel). The "
        "threshold of 0.70 % came from the two sequences at 512³; at 256³ and at an equal resolution in voxels the field error is "
        "higher. Over the sweep it falls steadily, from 2.8 % at 0.36 voxel per Kolmogorov scale to 0.63 % at 0.80.",
        m_error_against_resolution(0.69, 0.0070, 0.45, 0.0128)),
    Claim(
        "VALID-5", "Spread between realisations",
        "The compression measurements depend on the realisation of the turbulence only to within a few per cent.",
        SWEEP + "For each of the 6 viscosities, in degree 3 at a step of 2 voxels and without a bound: the compression ratio of the group of 32 frames and "
        "the field error each vary by less than 5 % between the three realisations (difference between the largest and the smallest "
        "value, relative to their mean).", BEFORE_16, DOES_NOT_HOLD,
        EVIDENCE_VALIDITY, REPLAY_VALIDITY,
        "The three realisations of one viscosity do not have the same resolution: the Kolmogorov scale depends on the dissipation of "
        "each run and itself varies by up to 13 % between seeds (0.702 to 0.795 voxel at a viscosity of 0.0028). Part of the spread "
        "measured here is therefore a spread of resolution, which this criterion did not separate. What it does establish: a single "
        "sequence per viscosity, as in every earlier measurement of this project, fixes a compression ratio to within 5 to 13 % only.",
        m_spread_between_realisations(0.05)),
    Claim(
        "CTRL-1", "Windowed cutout protocol",
        "The error of the codec measured at the centre of a windowed cutout, not periodic originally, is the one it makes on the whole field.",
        "On `turb512.seq` and `turb512_nu0006.seq`: the 256³ cutout of frames 0 to 7, multiplied by a smooth window (1 at the centre, "
        "half-cosine over 32 voxels at each edge) then projected onto the divergence-free fields, is compressed in degree 3 at a step of "
        "2 voxels with the u′ of the sequence. Its relative RMS error on the central cube of 160³ equals, to within 10 %, that of the whole field "
        "at 512³ compressed with the same settings on the same frames, measured on the same voxels. True for both sequences.",
        BEFORE_16, HOLDS,
        f"`{CONTROL}` (output of each encoding in `benchmarks/cutout/logs/`)", "`python benchmarks/cutout/cutout.py control` (GPU, the two sequences at 512³)",
        "The protocol overestimates the error, by 6 % on one sequence and by 9 % on the other: the correction made by the projection is "
        "smooth at the centre, but not zero. The bound of 5 % is held or refused in the same way on the whole field and on the cutout, "
        "for both sequences. The compression ratio, for its part, does not carry over: the windowed cutout compresses 150 times against 126 for the "
        "whole field at 0.74 voxel per Kolmogorov scale, its edges being almost zero.", m_cutout_control(0.10)),
    Claim(
        "JHTDB-3", "Degradation on a JHTDB cutout",
        "On a 256³ cutout of the public JHTDB dataset `isotropic1024coarse`, processed by the protocol of CTRL-1, degree 3 at a step of "
        "2 voxels makes more than twice the error of COMP-1 and does not hold the bound of 5 %.",
        "8 consecutive frames: relative RMS error at the centre (cube of 160³) > 1.28 %, and `--tolerance 0.05` exits with a non-zero code. "
        "Not applicable if CTRL-1 does not hold.", BEFORE_16, PENDING,
        basis="RESO-1 and RESO-2, on our sequence of the same resolution in voxels.",
        needed="The access token placed in the environment variable `JHTDB_TOKEN`, then `benchmarks/cutout/download_jhtdb.py`, run with the "
        "Python of an environment that holds the client `givernylocal` of the database, and `python benchmarks/cutout/cutout.py jhtdb`: "
        "8 cutouts of 0.2 GB, one request at a time (usage limits of the database: 3 GB per cutout, small subsets)." + JHTDB_ACCESS),
    Claim(
        "JHTDB-4", "Step of 1 voxel on the JHTDB cutout",
        "On this same cutout, the step of 1 voxel holds the bound of 5 % but compresses less than SPERR.",
        "`--mf 1 --tolerance 0.05` exits with code 0, and its compression ratio on the 8 frames is lower than that of SPERR applied to the same "
        "windowed cutout, at the same absolute bound. Not applicable if CTRL-1 does not hold.", BEFORE_16, PENDING,
        basis="VALID-3, if it holds on our data.", needed="The prerequisites of JHTDB-3." + JHTDB_ACCESS),
    Claim(
        "REPR-1", "Does our under-resolved sequence stand in for JHTDB?",
        "`turb512_nu0006.seq`, with the same Kolmogorov scale in voxels, predicts the behaviour of the codec on JHTDB.",
        "The relative RMS error at the centre of the windowed cutout of JHTDB (degree 3, step of 2 voxels, 8 frames) equals that of the "
        "windowed cutout of `turb512_nu0006.seq` to within 20 %.", BEFORE_16, PENDING,
        basis="Same resolution in voxels (0.45 and 0.47). Against: our simulation truncates its spectrum at n/3, I do not know where the database "
        "truncates its own, and its interval between frames is not verified.", needed="The prerequisites of JHTDB-3." + JHTDB_ACCESS),
    Claim(
        "REF-2", "A second reader from the completed specification",
        "The completed specification is sufficient, on its own, to write a reader: the state of the frames and the velocity field, without any trial to "
        "decide between readings.",
        "A second reader, written from `docs/FORMAT.md` by someone who has read neither `src/`, nor the first reader (`tools/reference_reader.cpp`, "
        "which was in `src/` at the time), nor `benchmarks/independent_readers/cpp/`, gives back the fingerprint of each frame of the five reference "
        "files at 256³ and of the test files at 64³, and the field of their probe to less than 5·10⁻⁵ u′. The list of the passages where the text was "
        "not sufficient, kept during the writing, is empty for everything that concerns the state and the field.", BEFORE_16, DOES_NOT_HOLD,
        f"`{READER_PYTHON}` and `gaps.csv` (reader: `tools/reference_reader.py`; notice: `benchmarks/independent_readers/python/README.md`)",
        "`python benchmarks/independent_readers/python/reference_python.py`",
        "The criterion fails only on its last clause, read strictly. The first two are met: the reader, run as it was written, gave "
        "back every fingerprint and the field at its first run, and its decoding has not been touched since. The passage counted is a "
        "silence of the text with no effect on the files read: the prediction formula does not say on how many bits the sum is "
        "computed, and no limit bounds the predictors; an exact sum and a sum on 32 bits differ only for predictors at least 21 times "
        "larger than those of these files (calculation). Whoever does not count this passage reads \"holds\". The five other passages "
        "noted are not counted: two readings kept as they were and then confirmed on files produced for that purpose (tables of the "
        "empty segments, reading by differences), one reading that is the only one consistent with the text, one case that the text "
        "itself calls undefined, and the fate of malformed exceptions in an altered file. A single frame probed per file, the last. "
        "The field is evaluated in double precision and compared with the probe of the encoder, not with the GPU reader. `PSISTRM1` "
        "is not supported, and nothing is read at 512³. The text was completed on the six passages on 17 November 2026: whether it is now "
        "sufficient is REF-3.", m_reference_python(5e-5)),
    Claim(
        "LINUX-1", "Compilation under Linux",
        "The project compiles under Linux.",
        "On GitHub, at each commit of `main`, an Ubuntu machine without a GPU compiles the library, `psit`, `psi_stream`, `dns` and "
        "`psi_ref` with GCC and `nvcc`, then replays `python tools/claims.py`: the last run is green. Nothing is executed there "
        "on a GPU.", BEFORE_16, PENDING,
        f"`{WORKFLOW}`", "`python benchmarks/linux_build/record.py` after a commit on `main` (`gh` tool)",
        f"`{WORKFLOW}` does not exist when this repository is first published: the workflow can only run after the publication, so the "
        "verdict registered is \"pending\". The same build was green in the development repository on 16 November 2026 (run 43806060947, on "
        "the previous layout of the repository), after a first run that had failed at the linking of the C example (mathematics library absent "
        "from the link line), corrected in the following commit. Compilation only: the workflow machine has no GPU, no kernel is executed "
        "there. Whether the chain gives under Linux the same files and the same test verdicts remains to be established: that is PORT-1.",
        m_workflow,
        basis="The same build was green in the development repository on 16 November 2026 (run 43806060947, on the previous layout of the "
        "repository).",
        needed=f"The publication of this repository, a completed run of `.github/workflows/linux.yml` on `main`, then "
        f"`python benchmarks/linux_build/record.py`, which writes `{WORKFLOW}`. This script then evaluates the record, and says that the "
        "verdict registered must be updated."),
    Claim(
        "GRID-1", "Compression alone breaks the law",
        "Compression alone breaks the law.",
        "For each of `sz3`, `sz3_4d` and `sperr`, on each of the three realisations: M1 of the decoded grid ≥ 1·10⁻², while M1 of the "
        "original grid ≤ 1·10⁻⁵. Refuted if one of the nine decoded grids is below 1·10⁻², or if one original grid is above 1·10⁻⁵. `zfp`, "
        "`cuszp` and `cuzfp_2bits` are measured and reported without a criterion: I do not know where they fall.", BEFORE_17, HOLDS,
        f"`{DIVERGENCE}`, column `grid_divergence_over_gradient`", REPLAY_DIVERGENCE,
        Filled("The law is the zero divergence of the velocity field." + DEFINED_ABOVE + " Measured without a criterion, M1 of the decoded "
               "grid: `zfp` {zfp}; `cuszp` {cuszp}; `cuzfp_2bits` {cuzfp}. GRID-2 of the same pre-registration makes no prediction and is not a "
               "claim: the grid decoded by the library samples a divergence-free field at voxel centres, and its spectral divergence M1 is "
               "not zero by construction; measured, it is {psi} for Psi-treillis with the bound (`psi`) and {psi_free} without a bound "
               "(`psi_free` in the log). One frame per realisation, one viscosity, 256³." + RENAMED_17, v_grid_divergence),
        m_grid_divergence(1e-2, 1e-5)),
    Claim(
        "READ-1", "Reading alone breaks the law",
        "Reading alone breaks the law.",
        "M2 of the original grid ≥ 1·10⁻² on each of the three realisations.", BEFORE_17, HOLDS,
        f"`{DIVERGENCE}`, column `read_divergence_over_gradient` of the chain `raw`", REPLAY_DIVERGENCE,
        "The law is the zero divergence of the velocity field." + DEFINED_ABOVE + " M2 is computed in double precision on the trilinear "
        "interpolant: it leaves out the rounding of a half-precision texture. For Psi-treillis the field read is the analytic curl of the "
        "potential: its divergence is zero by identity, and DIV-1 measures it. One frame per realisation, one viscosity, 256³." + RENAMED_17,
        m_read_divergence(1e-2)),
    Claim(
        "READ-2", "Compression adds to it",
        "Compression adds to it.",
        "For each of `sz3`, `sz3_4d` and `sperr`, on each of the three realisations: M2 of the decoded grid ≥ 1.1 times M2 of the "
        "original grid.", BEFORE_17, HOLDS,
        f"`{DIVERGENCE}`, column `read_divergence_over_gradient`", REPLAY_DIVERGENCE,
        Filled("\"It\" is the divergence that reading alone gives (READ-1)." + DEFINED_ABOVE + " Measured without a criterion, M2 of the "
               "decoded grid: `zfp` {zfp}; `cuszp` {cuszp}; `cuzfp_2bits` {cuzfp}. One frame per realisation, one viscosity, 256³."
               + RENAMED_17, v_read_divergence),
        m_read_divergence_added(1.1)),
    Claim(
        "TRAC-1", "Tracers after an error-bounded compressor",
        "Tracers after an error-bounded compressor.",
        "On the realisation of seed 42, for each of `sz3`, `sz3_4d` and `sperr`: the growth of the variance of D (M3) is at least that "
        "of `raw`, and at least 1,000 times that of `psi`. The excess dispersion of these three chains is measured and reported without a "
        "criterion: I do not know whether it passes the threshold of 5 standard errors that `cuzfp_2bits` passes and `raw` does not.",
        BEFORE_17, HOLDS,
        f"`{TRACERS}/compressors_256_summary.csv` (lines of the tracer program: `compressors_256.csv`; tables: `compressors_256_analysis.txt`)",
        "`python benchmarks/tracers/tracers.py run compressors_256` (GPU; needs the comparison codecs of `build/codecs/` and "
        "`data/sweep/nu00280_seed42.seq`)",
        Filled("The label of the pre-registration is not a sentence; the hypothesis is the first sentence of the criterion." + DEFINED_ABOVE
               + " In the log, `raw` is the path `tex`, `psi` the path `psi3b` and `cuzfp_2bits` the path `zfp2`. The path `psi3b` is not the "
               "chain `psi` exactly as the pre-registration defines it: the tracer program encodes frame 32 alone, as a key frame, with the "
               "bound of 5 %, and samples the potential analytically; it does not extract frame 32 from the group of frames 16 to 47. "
               "Measured without a criterion, the excess dispersion in bins of 1 voxel after 10 Kolmogorov times, in standard errors over "
               "tracer seeds 1 to 5: `sz3` {sz3}; `sz3_4d` {sz3_4d}; `sperr` {sperr}; `cuzfp_2bits` {zfp2}; `raw` {tex}; `psi` {psi}. The growth "
               "of the variance of D of `cuzfp_2bits`, outside the criterion: {zfp_growth}. Frozen field, one frame, one realisation, tracers "
               "without inertia." + RENAMED_17, v_tracers_after_compressors),
        m_tracers_after_compressors(1000)),
    Claim(
        "SPEC-1", "Spectrum of Psi-treillis",
        "Spectrum of Psi-treillis.",
        "On each of the three realisations, for `psi` without a bound (degree 3, default settings): the enstrophy of the decoded grid "
        "is within 5 % of that of the original, and k10 is at least half the truncation wavenumber of the solver. The other chains are "
        "measured and reported without a criterion.", BEFORE_17, HOLDS,
        f"`{SPECTRA}`, lines of the chain `psi_free` (energy of every shell: `benchmarks/spectra/shells.csv`)",
        "`python benchmarks/spectra/spectra.py run` (needs the comparison codecs of `build/codecs/` and the three sequences "
        "`data/sweep/nu00280_seed<seed>.seq`)",
        Filled("The label of the pre-registration is not a sentence; the hypothesis is the first sentence of the criterion." + DEFINED_ABOVE
               + " In the log, `psi` without a bound is the chain `psi_free`, and the truncation wavenumber is the integer part of n/3. "
               "Measured without a criterion, enstrophy of the decoded grid over that of the original and k10, for each realisation: "
               "{others}. One frame per realisation, one viscosity, 256³: the spectrum of the grid decoded at voxel centres, not that of "
               "the field sampled off the grid." + RENAMED_17, v_spectrum),
        m_spectrum(0.05)),
    Claim(
        "PLAY-1", "Cost of one replayed frame",
        "Replayed frame after frame on the GPU, Psi-treillis costs more time per frame than cuZFP at 256³; at 512³ it reads far less data "
        "than raw frames and holds less GPU memory.",
        "One protocol and one run of the visual comparator per grid size and per pair of chains, sorted particles, GPU running no other "
        "computation: data read per frame, GPU memory, cost of one frame, for raw frames, cuZFP at 2 bits and Psi-treillis of degree 3 "
        "without a bound. At 256³ the frame of Psi-treillis costs more time than that of cuZFP; at 512³ Psi-treillis reads at least "
        "100 times less data than the raw chain and uses at least 5 times less GPU memory.", AFTER_16_PROTOCOL_BEFORE, HOLDS,
        f"`{PLAYBACK}` (outputs of the comparator: `benchmarks/playback/logs/view_*.log`)",
        "`python benchmarks/playback/playback.py run` (GPU running nothing else; `psi_view` exists on Windows only)",
        "The statement is written for this sheet; the criterion is that of the pre-registration of 17 November 2026, which kept the one of "
        "the measurement of 16 November. The two times at 256³ are those of the run that shows cuZFP and Psi-treillis side by side (pair "
        "`cuzfp_psi`); the data and the memory at 512³ are those of the run that shows the raw frames and Psi-treillis (pair `raw_psi`), "
        "the memory of Psi-treillis being compared with that of the raw chain. Timings of one machine, one run per case; in synchronised "
        "mode both sides share the GPU while they are measured. cuZFP at 2 bits is at a fixed rate, not at a bound.", m_playback(100, 5)),
    Claim(
        "REF-3", "A third reader from the specification completed on 17 November 2026",
        "The specification completed on 17 November 2026 on the six gaps of REF-2 is sufficient, on its own, to write a reader.",
        "A third reader, written from `docs/FORMAT.md` by someone who has read neither `src/` nor the two existing readers, gives back the "
        "fingerprint of each frame of the five reference files at 256³ and of the test files at 64³, and the field of their probe to less "
        "than 5·10⁻⁵ u′; the list of the passages where the text was not sufficient, kept during the writing, is empty for everything "
        "that concerns the state and the field.", BEFORE_17, PENDING,
        basis="REF-2: the second reader gave back every fingerprint and the field, and noted six passages where the text left a doubt, one of "
        "them counted; `docs/FORMAT.md` was completed on each of them on 17 November 2026 (marks \"completed on 17 November 2026, after the "
        "second reader\").",
        needed="A working session that has read neither `src/` nor the two existing readers (`tools/reference_reader.cpp`, "
        "`tools/reference_reader.py`) nor their logs, and that keeps the list of its doubts before any trial."),
]


# --- Verdicts and sheet ----------------------------------------------------------------------------------------------

@dataclass
class Judgement:
    verdict: str                 # HOLDS, DOES_NOT_HOLD, PENDING, NO_LOG or UNREADABLE
    measured: str                # what is read in the log, or why nothing is
    anomalies: List[str]
    caveat: str
    basis: str
    needed: str


def rendered(text: Text, what: str, anomalies: list) -> str:
    if isinstance(text, Filled):
        text, reason = text.render()
        if reason:
            anomalies.append(f"{what}: {reason}")
    return text


def judge(c: Claim) -> Judgement:
    anomalies: list = []
    verdict, measured = PENDING, ""
    if c.measure is None:
        if c.expected != PENDING:
            anomalies.append("no measurement is defined")
    else:
        try:
            reading = c.measure()
        except LogMissing as e:
            if c.expected != PENDING:        # a pending claim whose log is not there yet stays pending, without an anomaly
                verdict, measured = NO_LOG, str(e)
                anomalies.append(str(e))
        except Evidence as e:
            verdict, measured = UNREADABLE, str(e)
            anomalies.append(f"log unreadable: {e}")
        except (KeyError, IndexError, ValueError, ZeroDivisionError, OverflowError) as e:
            verdict, measured = UNREADABLE, f"{e!r}"
            anomalies.append(f"log unreadable: {e!r}")
        else:
            verdict = HOLDS if reading.holds else DOES_NOT_HOLD
            measured = reading.text
            if reading.notes:
                measured += " [The log does not have the scope that the criterion states: " + "; ".join(reading.notes) + ".]"
                anomalies += [f"scope: {note}" for note in reading.notes]
            if c.expected == UNREGISTERED:
                anomalies.append(f'no verdict is registered yet, the log gives "{verdict}": write it in the registry')
            elif c.expected == PENDING:
                anomalies.append(f'the registry says "pending", the log is there and gives "{verdict}": the verdict registered must be updated')
            elif verdict != c.expected:
                anomalies.append(f'the registry expects "{c.expected}", the logs give "{verdict}"')
    return Judgement(verdict, measured, anomalies, rendered(c.caveat, "caveat", anomalies), rendered(c.basis, "basis", anomalies),
                     rendered(c.needed, "what is needed first", anomalies))


def ended(text: str) -> str:
    return text if text.endswith((".", ".]")) else text + "."


def generated_part(judgements: list) -> str:
    lines = ["## Summary", "", "| Identifier | Claim | Verdict |", "|---|---|---|"]
    for c, j in judgements:
        lines.append(f"| {c.ident} | {c.title} | {'**' + j.verdict + '**' if j.verdict != HOLDS else j.verdict} |")
    count = {v: sum(1 for _, j in judgements if j.verdict == v) for v in (HOLDS, DOES_NOT_HOLD, PENDING)}
    without = len(judgements) - sum(count.values())
    lines += ["", f"{len(judgements)} claims: {count[HOLDS]} hold, {count[DOES_NOT_HOLD]} do not hold, {count[PENDING]} pending"
              + (f", {without} without a readable log" if without else "") + "."]
    sections = (
        (HOLDS, "Claims that hold", "Each one is verified by replaying the command given, then this script."),
        (DOES_NOT_HOLD, "Claims that do not hold",
         "What the project would have liked to be able to assert and that its own measurements refute. They bound the domain of validity."),
        (PENDING, "Pending claims",
         "Hypotheses whose criterion is written before any measurement. Each one states the test that settles it and what is needed first."),
        (NO_LOG, "Claims whose log is missing or unreadable",
         "The registry holds these claims as measured, but the log that carries them is not in the repository, or is not the one that the "
         "present script writes: no verdict is given. Each one states the verdict that the registry expects; the log confirms it, or not, "
         "once it is there."),
    )
    for section, title, heading in sections:
        members = [(c, j) for c, j in judgements if (j.verdict if j.verdict in (HOLDS, DOES_NOT_HOLD, PENDING) else NO_LOG) == section]
        if not members and section == NO_LOG:
            continue
        lines += ["", f"## {title}", "", heading]
        for c, j in members:
            hypothesis = c.expected == PENDING
            lines += ["", f"### {c.ident}. {c.title}", "",
                      f"- **{'Hypothesis' if hypothesis else 'Claim'}.** {c.statement}",
                      f"- **Criterion.** {c.criterion} *{c.fixed}*"]
            if section == NO_LOG:
                lines.append(f"- **Measured.** Nothing: {ended(j.measured)}")
                lines.append(f"- **Verdict: none ({j.verdict}).** Verdict that the registry expects: {c.expected}.")
            else:
                if j.measured:
                    lines.append(f"- **Measured.** {ended(j.measured)}")
                lines.append(f"- **Verdict: {j.verdict}.**")
            if j.verdict == PENDING:
                lines += [f"- **Basis.** {j.basis}", f"- **What is needed first.** {j.needed}"]
            else:
                lines.append(f"- **Evidence.** {c.evidence}; to reproduce: {c.replay}.")
            if j.caveat:
                lines.append(f"- **Caveat.** {j.caveat}")
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    write = "--write" in sys.argv[1:]
    if any(a != "--write" for a in sys.argv[1:]):
        print("usage: python tools/claims.py [--write]", file=sys.stderr)
        return 2

    judgements = [(c, judge(c)) for c in REGISTRY]
    anomalies = 0
    for c, j in judgements:
        print(f"{j.verdict:<14} {c.ident:<10} {c.title}" + (f": {j.measured}" if j.measured else ""))
        for anomaly in j.anomalies:
            anomalies += 1
            print(f"   ANOMALY {c.ident}: {anomaly}", file=sys.stderr)
    if len({c.ident for c in REGISTRY}) != len(REGISTRY):
        anomalies += 1
        print("   ANOMALY: two claims carry the same identifier", file=sys.stderr)

    file = ROOT / SHEET
    raw = file.read_bytes().decode("utf-8") if file.is_file() else ""
    line_end = "\r\n" if "\r\n" in raw else "\n"
    text = raw.replace("\r\n", "\n")
    if START not in text or END not in text:
        print(f"   ANOMALY: {SHEET} does not contain the two markers of the generated part", file=sys.stderr)
        print(f"\n{len(REGISTRY)} claims, {anomalies + 1} anomal{'y' if anomalies + 1 == 1 else 'ies'}")
        return 1
    before, rest = text.split(START, 1)
    old, after = rest.split(END, 1)
    new = "\n\n" + generated_part(judgements) + "\n\n"
    if write:
        file.write_bytes((before + START + new + END + after).replace("\n", line_end).encode("utf-8"))
        print(f"\n{SHEET}: generated part " + ("rewritten" if new != old else "already up to date"))
    elif new != old:
        anomalies += 1
        print(f"   ANOMALY: {SHEET} is no longer up to date; run again with --write, read it over, then record the change in CHANGELOG.md",
              file=sys.stderr)
    print(f"\n{len(REGISTRY)} claims, {anomalies} anomal{'y' if anomalies == 1 else 'ies'}")
    return 1 if anomalies else 0


if __name__ == "__main__":
    sys.exit(main())
