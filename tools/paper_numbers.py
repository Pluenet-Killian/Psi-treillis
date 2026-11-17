# SPDX-License-Identifier: Apache-2.0
"""Reads the logs of benchmarks/ and writes every number quoted by the article and by README.md.

    python tools/paper_numbers.py            writes paper/numbers.tex, paper/tables/*.tex and README.md
    python tools/paper_numbers.py --list     prints every value with its key, writes nothing

No number of the article or of README.md is typed by hand. paper/paper.tex calls a value by its key, \\val{key}, and
README.md is generated from tools/README.template.md, where a value is written {{key}}. This script computes each key
from the logs, with the rounding stated next to it, and fails if a log, a row or a key is missing. It runs no
measurement and needs neither a GPU nor a library: only the standard library.
"""

import csv
import math
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
B = ROOT / "benchmarks"


class Missing(Exception):
    pass


def rows(relative):
    path = B / relative
    if not path.is_file():
        raise Missing(f"log missing: benchmarks/{relative}")
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def one(table, **where):
    found = [r for r in table if all(r[k] == v for k, v in where.items())]
    if len(found) != 1:
        raise Missing(f"{len(found)} row(s) where {where}, one expected")
    return found[0]


# ---- Formats: each returns (LaTeX, Markdown) ------------------------------------------------------------------------

def fixed(x, decimals):
    text = f"{x:.{decimals}f}"
    if text.startswith("-"):
        return "$-$" + text[1:], "−" + text[1:]
    return text, text


def integer(x):
    text = f"{int(round(x)):,}"
    return text, text


def percent(x, decimals):
    return fixed(100.0 * x, decimals)


def significant(x, digits=2):
    if x == 0:
        return "0", "0"
    decimals = max(0, digits - 1 - int(math.floor(math.log10(abs(x)))))
    return fixed(x, decimals)


def scientific(x, decimals=1):
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / 10.0 ** exponent
    if round(mantissa, decimals) >= 10.0:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    superscript = str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"${mantissa:.{decimals}f}\\times10^{{{exponent}}}$", f"{mantissa:.{decimals}f} × 10{superscript}"


def times(x):
    """A ratio 'x times': no decimal from 10 up, one below."""
    return fixed(x, 0 if abs(x) >= 9.95 else 1)


def errors(x):
    """A number of standard errors: no decimal from 100 up, one below."""
    return fixed(x, 0 if abs(x) >= 99.95 else 1)


# ---- Values ---------------------------------------------------------------------------------------------------------

def values():
    v, missing = {}, []

    def group(function):
        try:
            function()
        except Missing as e:
            missing.append(f"{function.__name__}: {e}")
        except KeyError as e:
            missing.append(f"{function.__name__}: column {e} missing from a log")
        return function

    @group
    def divergence():
        table = rows("divergence/divergence.csv")
        seeds = sorted({r["solver_seed"] for r in table})
        if len(seeds) != 3:
            raise Missing(f"divergence.csv has {len(seeds)} realisation(s), 3 expected")

        def column(chain, name):
            return [float(one(table, chain=chain, solver_seed=s)[name]) for s in seeds]

        raw_read = column("raw", "read_divergence_over_gradient")
        for chain in ("sz3", "sz3_4d", "sperr", "psi", "cuszp", "zfp", "cuzfp_2bits"):
            grid, read = column(chain, "grid_divergence_over_gradient"), column(chain, "read_divergence_over_gradient")
            v[f"grid.{chain}.min"], v[f"grid.{chain}.max"] = percent(min(grid), 1), percent(max(grid), 1)   # per cent, one decimal
            v[f"read.{chain}.min"], v[f"read.{chain}.max"] = percent(min(read), 1), percent(max(read), 1)
            ratio = [a / b for a, b in zip(read, raw_read)]
            v[f"read.{chain}.ratio.min"], v[f"read.{chain}.ratio.max"] = fixed(min(ratio), 2), fixed(max(ratio), 2)
        v["grid.raw.max"] = scientific(max(column("raw", "grid_divergence_over_gradient")))
        v["read.raw.min"], v["read.raw.max"] = percent(min(raw_read), 1), percent(max(raw_read), 1)

    @group
    def tracers():
        s = rows("tracers/compressors_256_summary.csv")

        def row(path):
            return one(s, path=path)

        growth = {p: float(row(p)["var_d_growth_per_tau"]) for p in ("tex", "zfp2", "sz3", "sz3_4d", "sperr", "psi3b")}
        for path, key in (("tex", "raw"), ("zfp2", "zfp2"), ("sz3", "sz3"), ("sz3_4d", "sz3_4d"), ("sperr", "sperr"), ("psi3b", "psi3b")):
            v[f"trac.{key}.z"] = errors(float(row(path)["criterion_standard_errors"]))
        for path in ("sz3", "sz3_4d", "sperr", "zfp2"):
            v[f"trac.{path}.over.raw"] = times(growth[path] / growth["tex"])
        v["trac.raw.over.psi"] = integer(growth["tex"] / growth["psi3b"])
        for campaign, key in (("resolved_256", "256"), ("resolved_512", "512"), ("confirmation_256", "conf")):
            summary = rows(f"tracers/{campaign}_summary.csv")
            for path, name in (("tex", "tex"), ("zfp2", "zfp2"), ("psi3", "psi3")):
                v[f"lagr.{key}.{name}.z"] = errors(float(one(summary, path=path)["criterion_standard_errors"]))
        v["lagr.conf.ref.z"] = errors(float(one(rows("tracers/confirmation_256_summary.csv"), path="ref0.002")["criterion_standard_errors"]))
        # The reference path on ten tracer seeds: the two campaigns together, with the arithmetic of the analysis.
        ten = [r for name in ("confirmation_256", "confirmation_256_more_seeds") for r in rows(f"tracers/{name}.csv")
               if r["path"] == "ref0.002" and round(float(r["t_tau"])) == 10]
        if len({r["seed"] for r in ten}) != 10:
            raise Missing(f"reference path of the confirmation: {len(ten)} run(s) at 10 Kolmogorov times, 10 seeds expected")
        excess = [float(r["excess1"]) for r in ten]
        bins = (int(ten[0]["grid"]) / float(ten[0]["bin"])) ** 3
        expected = math.sqrt(2.0 / bins) / math.sqrt(len(excess))
        between = statistics.stdev(excess) / math.sqrt(len(excess))
        v["lagr.conf.ref.z.ten"] = errors(statistics.mean(excess) / max(between, expected))
        long = rows("tracers/long_256_summary.csv")
        for path in ("zfp2", "tex", "cubic", "psi3"):
            r = one(long, path=path)
            if r["last_t_tau"] != "100":
                raise Missing("long_256 does not reach 100 Kolmogorov times")
            v[f"long.{path}.std"] = significant(float(r["d_standard_deviation_at_last_t"]), 2)

    @group
    def playback():
        table = rows("playback/playback.csv")
        if any(r["gpu_free"] != "yes" for r in table):
            raise Missing("playback.csv was measured with a GPU that was not free")

        def side(grid, pair, which):
            return one(table, grid=str(grid), pair=pair, side=which)

        raw, psi = side(512, "raw_psi", "left"), side(512, "raw_psi", "right")
        v["play.512.raw.over.psi.read"] = times(float(raw["mib_read_per_frame"]) / float(psi["mib_read_per_frame"]))
        v["play.512.raw.over.psi.mem"] = fixed(float(raw["gpu_memory_mib"]) / float(psi["gpu_memory_mib"]), 1)
        v["play.512.raw.over.psi.time"] = times(float(raw["frame_cost_ms"]) / float(psi["frame_cost_ms"]))
        v["play.512.psi.time.raw"] = fixed(float(psi["frame_cost_ms"]), 1)
        for grid in (256, 512):
            zfp, psi = side(grid, "cuzfp_psi", "left"), side(grid, "cuzfp_psi", "right")
            v[f"play.{grid}.zfp.over.psi.read"] = fixed(float(zfp["mib_read_per_frame"]) / float(psi["mib_read_per_frame"]), 1)
            v[f"play.{grid}.zfp.over.psi.mem"] = fixed(float(zfp["gpu_memory_mib"]) / float(psi["gpu_memory_mib"]), 1)
            v[f"play.{grid}.psi.time"], v[f"play.{grid}.zfp.time"] = fixed(float(psi["frame_cost_ms"]), 1), fixed(float(zfp["frame_cost_ms"]), 1)
            v[f"play.{grid}.psi.rk4"], v[f"play.{grid}.zfp.rk4"] = fixed(float(psi["rk4_step_ms"]), 1), fixed(float(zfp["rk4_step_ms"]), 1)
            v[f"play.{grid}.zfp.disk"], v[f"play.{grid}.zfp.read"] = fixed(float(zfp["disk_ms"]), 0), fixed(float(zfp["mib_read_per_frame"]), 0)
        divergences = [float(r["divergence_over_gradient"]) for r in table if r["side"] == "right" and r["divergence_over_gradient"]]
        if not divergences:
            raise Missing("playback.csv gives no divergence for Psi-treillis")
        v["div.psi.max"] = scientific(max(divergences))

    @group
    def equal_bound():
        t256, t512 = rows("equal_bound/equal_bound_256.csv"), rows("equal_bound/equal_bound_512.csv")

        def cell(table, codec, bound):
            r = one(table, codec=codec, bound_over_urms=bound)
            if not r["compression_ratio"]:
                raise Missing(f"{codec} at {bound} was not encoded")
            return float(r["compression_ratio"]), float(r["rms_error_relative"]), float(r["largest_error_over_urms"])

        bounds = ("0.05", "0.04", "0.03")
        ratio = [cell(t256, "sz3_4d", b)[0] / cell(t256, "psi_d3", b)[0] for b in bounds]
        error = [cell(t256, "sz3_4d", b)[1] / cell(t256, "psi_d3", b)[1] for b in bounds]
        v["eq.256.sz4d.over.psi.min"], v["eq.256.sz4d.over.psi.max"] = fixed(min(ratio), 1), fixed(max(ratio), 1)
        v["eq.256.sz4d.rms.over.psi.min"], v["eq.256.sz4d.rms.over.psi.max"] = fixed(min(error), 1), fixed(max(error), 1)
        sperr = [cell(t, "psi_d3", b)[0] / cell(t, "sperr", b)[0] for t, bs in ((t256, bounds), (t512, ("0.05", "0.04"))) for b in bs]
        v["eq.sperr.under.psi.min"], v["eq.sperr.under.psi.max"] = fixed(min(sperr), 2), fixed(max(sperr), 2)
        v["eq.256.other.points"] = integer(sum(1 for r in t256 if r["codec"] != "psi_d3" and r["compression_ratio"]))
        v["eq.256.sz4d.t015.ratio"], v["eq.256.sz4d.t015.rms"] = fixed(cell(t256, "sz3_4d", "0.015")[0], 0), percent(cell(t256, "sz3_4d", "0.015")[1], 2)
        v["eq.256.psi.t05.ratio"], v["eq.256.psi.t05.rms"] = fixed(cell(t256, "psi_d3", "0.05")[0], 0), percent(cell(t256, "psi_d3", "0.05")[1], 2)
        psi_rms = [cell(t256, "psi_d3", b)[1] for b in bounds]
        v["eq.256.psi.rms.min"], v["eq.256.psi.rms.max"] = percent(min(psi_rms), 2), percent(max(psi_rms), 2)
        v["eq.256.zfp.t05.max"] = percent(cell(t256, "zfp", "0.05")[2], 2)

    @group
    def validity():
        table = rows("validity/validity.csv")
        names = sorted({r["sequence"] for r in table})
        if len(names) != 18:
            raise Missing(f"validity.csv has {len(names)} sequence(s), 18 expected")

        def cell(name, codec, bound):
            return one(table, sequence=name, codec=codec, bound_over_urms=bound)

        eta = {n: float(cell(n, "psi_d3_mf2", "")["eta_voxels"]) for n in names}
        held = [n for n in names if cell(n, "psi_d3_mf2", "0.05")["exit_code"] == "0"]
        refused = [n for n in names if n not in held]
        held3 = [n for n in names if cell(n, "psi_d3_mf2", "0.03")["exit_code"] == "0"]
        v["valid.eta.min"], v["valid.eta.max"] = fixed(min(eta.values()), 2), fixed(max(eta.values()), 2)
        v["valid.held"], v["valid.refused"], v["valid.held.t03"] = integer(len(held)), integer(len(refused)), integer(len(held3))
        low, high = max(eta[n] for n in refused), min(eta[n] for n in held)
        if low >= high:
            raise Missing("the sequences that hold the bound and those that do not overlap in resolution")
        v["valid.eta.held.min"], v["valid.eta.refused.max"] = fixed(high, 2), fixed(low, 2)
        v["valid.eta.star"] = fixed(0.5 * (low + high), 2)
        v["valid.eta.held.t03.min"] = fixed(min(eta[n] for n in held3), 2)
        free = {n: float(cell(n, "psi_d3_mf2", "")["rms_error_relative"]) for n in names}
        v["valid.rms.best"], v["valid.rms.worst"] = percent(min(free.values()), 2), percent(max(free.values()), 1)

        def ratio(name, codec):
            return float(cell(name, codec, "0.05")["group_compression_ratio"])

        against = [ratio(n, "psi_d3_mf2") / ratio(n, "sperr") for n in held]
        v["valid.ahead"] = integer(sum(1 for x in against if x >= 1.0))
        v["valid.sperr.best"], v["valid.sperr.worst"] = fixed(max(against), 2), fixed(min(against), 2)
        fine = [ratio(n, "psi_d3_mf1") / ratio(n, "sperr") for n in names]
        if any(cell(n, "psi_d3_mf1", "0.05")["exit_code"] != "0" for n in names):
            raise Missing("the lattice step of 1 voxel does not hold the bound on every sequence")
        v["valid.mf1.sperr.min"], v["valid.mf1.sperr.max"] = fixed(min(fine), 2), fixed(max(fine), 2)
        spread_ratio, spread_rms = [], []
        for nu in sorted({cell(n, "psi_d3_mf2", "")["nu"] for n in names}):
            same = [cell(n, "psi_d3_mf2", "") for n in names if cell(n, "psi_d3_mf2", "")["nu"] == nu]
            for column, into in (("group_compression_ratio", spread_ratio), ("rms_error_relative", spread_rms)):
                x = [float(r[column]) for r in same]
                into.append((max(x) - min(x)) / statistics.mean(x))
        v["valid.spread.ratio"], v["valid.spread.rms"] = percent(max(spread_ratio), 0), percent(max(spread_rms), 0)

    @group
    def reference_files():
        table = rows("compression/reference_files/reference_files.csv")

        def file(name):
            return one(table, name=name)

        for name, key in (("z_t512_d3", "comp.512.d3"), ("z_t512_d2", "comp.512.d2")):
            v[f"{key}.ratio"], v[f"{key}.rms"] = fixed(float(file(name)["compression_ratio"]), 0), percent(float(file(name)["field_error_mean"]), 2)
        for grid in ("256", "512"):
            for bound in ("05", "04", "03"):
                r = file(f"f_t{grid}_d3_t{bound}")
                if r["pack_exit_code"] != "0" or r["frames_above_threshold"] != "0":
                    raise Missing(f"f_t{grid}_d3_t{bound} does not hold its bound")
                v[f"comp.{grid}.d3.t{bound}.ratio"] = fixed(float(r["compression_ratio"]), 1)
        points = {r["probe_points"] for r in table if r["probe_points"]}
        if len(points) != 1:
            raise Missing(f"the probes of the reference files have {sorted(points)} points")
        v["readback.points"] = integer(int(points.pop()))
        if any(r["frames_differing"] != "0" for r in table):
            raise Missing("a reference file is read back with a state that differs")
        for bound in ("05", "03"):
            r = file(f"f_t512_d3_t{bound}")
            if r["gpu_free_at_start"] != "yes":
                raise Missing("the reference files were encoded with a GPU that was not free: the encoding times cannot be quoted")
            v[f"enc.512.t{bound}"] = fixed(float(r["encoding_ms_per_frame"]), 0)

    @group
    def under_resolved():
        r = one(rows("compression/under_resolved/under_resolved.csv"), name="d3")
        v["reso.ratio"], v["reso.rms"] = fixed(float(r["compression_ratio"]), 0), percent(float(r["field_error_mean"]), 1)
        v["reso.eta"] = fixed(float(one(rows("data/sequences.csv"), sequence="turb512_nu0006.seq")["eta_voxels"]), 2)

    @group
    def speed():
        table = rows("speed/read_timing.csv")
        if any(r["gpu_free_at_start"] != "yes" for r in table):
            raise Missing("read_timing.csv was measured with a GPU that was not free")
        counts = set()
        for name, key in (("z_t512_d3", "z"), ("f_t512_d3_t05", "f")):
            mine = [r for r in table if Path(r["file"]).stem == name]
            if not mine:
                raise Missing(f"read_timing.csv has no row for {name}")
            counts.add(len(mine))
            v[f"speed.{key}.decode"] = fixed(statistics.median(float(r["decoding_ms"]) for r in mine), 2)
            v[f"speed.{key}.rk4"] = fixed(statistics.median(float(r["rk4_step_ms"]) for r in mine), 2)
        if len(counts) != 1:
            raise Missing("the two files of read_timing.csv do not have the same number of repetitions")
        v["speed.repetitions"] = integer(counts.pop())

    @group
    def spectra():
        table = [r for r in rows("spectra/spectra.csv") if r["chain"] == "psi_free"]
        if len(table) != 3:
            raise Missing("spectra.csv does not have three realisations of psi_free")
        v["spec.psi.enstrophy.max"] = percent(max(abs(float(r["enstrophy_over_original"]) - 1.0) for r in table), 1)
        k10 = [int(r["k10"]) for r in table]
        v["spec.psi.k10.min"], v["spec.psi.k10.max"] = integer(min(k10)), integer(max(k10))
        v["spec.truncation"] = integer(int(table[0]["truncation_wavenumber"]))

    @group
    def cutout():
        table = rows("cutout/control.csv")
        ratios = []
        for source in sorted({r["source"] for r in table}):
            whole = one(table, source=source, field="whole field", setting="d3_mf2", bound_over_urms="")
            cut = one(table, source=source, field="windowed cutout", setting="d3_mf2", bound_over_urms="")
            ratios.append(float(cut["centre_rms_error_relative"]) / float(whole["centre_rms_error_relative"]))
        v["ctrl.ratio.min"], v["ctrl.ratio.max"] = fixed(min(ratios), 2), fixed(max(ratios), 2)

    @group
    def readers():
        cpp = [r for r in rows("independent_readers/cpp/reference_cpp.csv")]
        if any(r["fingerprints_differing"] != "0" or r["frames"] != r["frames_read"] for r in cpp):
            raise Missing("the C++ reader does not give back every fingerprint")
        v["ref.cpp.frames"] = integer(sum(int(r["frames_read"]) for r in cpp))
        v["ref.cpp.field"] = scientific(max(float(r["cpu_probe_difference_over_u_rms"]) for r in cpp if r["cpu_probe_difference_over_u_rms"]))
        py = [r for r in rows("independent_readers/python/reference_python.csv")]
        if any(r["fingerprints_differing"] != "0" or r["frames"] != r["frames_read"] for r in py):
            raise Missing("the Python reader does not give back every fingerprint")
        v["ref.py.files"], v["ref.py.frames"] = integer(len(py)), integer(sum(int(r["frames_read"]) for r in py))
        v["ref.py.field"] = scientific(max(float(r["field_difference_over_u_rms"]) for r in py if r["field_difference_over_u_rms"]))
        v["ref.py.margin"] = integer(math.floor(min(float(r["prediction_margin_32_bits"]) for r in py if r["prediction_margin_32_bits"])))

    @group
    def claims_and_tests():
        sheet = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")
        m = re.search(r"^(\d+) claims: (\d+) hold, (\d+) do not hold, (\d+) pending\.$", sheet, re.MULTILINE)
        if not m:
            raise Missing("docs/CLAIMS.md has no line of totals")
        v["claims.total"], v["claims.hold"], v["claims.fail"], v["claims.pending"] = (integer(int(x)) for x in m.groups())
        log = B / "interface" / "test_logs" / "test_codec_64.log"
        if not log.is_file():
            raise Missing("log missing: benchmarks/interface/test_logs/test_codec_64.log")
        m = re.search(r"^TOTAL: (\d+) cases, (\d+) OK, 0 FAILED, (\d+) known defect\(s\)$", log.read_text(encoding="utf-8"), re.MULTILINE)
        if not m:
            raise Missing("test_codec_64.log has no clean total line")
        v["test.cases"] = integer(int(m.group(1)))

    return v, missing


# ---- Tables of the article ------------------------------------------------------------------------------------------

def table_data():
    s = rows("data/sequences.csv")
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Sequence & Grid & Frames & $Re_\\lambda$ & $\\eta$ (voxels) \\\\", "\\midrule"]
    for name, label in (("turb256.seq", "\\texttt{turb256}"), ("turb512.seq", "\\texttt{turb512}"), ("turb512_nu0006.seq", "\\texttt{turb512\\_nu0006}")):
        r = one(s, sequence=name)
        lines.append(f"{label} & ${r['grid']}^3$ & {r['frames']} & {float(r['re_lambda']):.0f} & {float(r['eta_voxels']):.2f} \\\\")
    sweep = [r for r in s if r["sequence"].startswith("sweep/")]
    if len(sweep) != 18 or len({(r["grid"], r["frames"]) for r in sweep}) != 1:
        raise Missing("the sweep of data/sequences.csv is not 18 sequences of one size")
    re_l, eta = [float(r["re_lambda"]) for r in sweep], [float(r["eta_voxels"]) for r in sweep]
    lines.append(f"resolution sweep, 18 sequences & ${sweep[0]['grid']}^3$ & {sweep[0]['frames']} each & {min(re_l):.0f} to {max(re_l):.0f} & "
                 f"{min(eta):.2f} to {max(eta):.2f} \\\\")
    return lines + ["\\bottomrule", "\\end{tabular}"]


def table_divergence():
    t = rows("divergence/divergence.csv")
    seeds = sorted({r["solver_seed"] for r in t})
    names = [("raw", "original grid"), ("zfp", "zfp"), ("cuzfp_2bits", "cuZFP, 2 bits"), ("sperr", "SPERR"), ("sz3", "SZ3"), ("sz3_4d", "SZ3 in 4D"),
             ("cuszp", "cuSZp"), ("psi", "\\codec, decoded grid")]
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule",
             " & Compression & RMS error & Divergence & Divergence \\\\", "Chain & ratio & (\\%) & on the grid (\\%) & as read (\\%) \\\\", "\\midrule"]

    def span(values, small):
        low, high = min(values), max(values)
        if small:
            return f"{scientific(low)[0]} to {scientific(high)[0]}".replace("to", "to ")  # relative, not per cent
        return f"{100 * low:.1f} to {100 * high:.1f}"

    for chain, label in names:
        mine = [one(t, chain=chain, solver_seed=s) for s in seeds]
        first = one(t, chain=chain, solver_seed="42")
        grid = [float(r["grid_divergence_over_gradient"]) for r in mine]
        read = [float(r["read_divergence_over_gradient"]) for r in mine]
        ratio = "" if chain == "raw" else f"{float(first['compression_ratio']):.0f}"
        rms = "" if chain == "raw" else f"{100 * float(first['rms_error_relative']):.2f}"
        grid_text = f"$<${100 * max(grid):.4f}" if chain == "raw" else span(grid, False)
        lines.append(f"{label} & {ratio} & {rms} & {grid_text} & {span(read, False)} \\\\")
    return lines + ["\\bottomrule", "\\end{tabular}"]


def table_tracers():
    s = rows("tracers/compressors_256_summary.csv")
    names = [("tex", "original grid"), ("zfp2", "cuZFP, 2 bits"), ("sperr", "SPERR"), ("sz3", "SZ3"), ("sz3_4d", "SZ3 in 4D"), ("psi3b", "\\codec, bound 5\\,\\%")]
    base = float(one(s, path="tex")["var_d_growth_per_tau"])
    lines = ["\\begin{tabular}{lrrr}", "\\toprule", " & Clustering & Growth of var($D$) & Relative to the \\\\",
             "Chain & (standard errors) & per Kolmogorov time & original grid \\\\", "\\midrule"]
    for path, label in names:
        r = one(s, path=path)
        if r["seeds"] != "5" or r["last_t_tau"] != "10":
            raise Missing(f"compressors_256: {path} is not on 5 seeds and 10 Kolmogorov times")
        growth = float(r["var_d_growth_per_tau"])
        relative = growth / base
        relative_text = "1" if path == "tex" else (f"{relative:.0f}" if relative >= 9.95 else (f"{relative:.1f}" if relative >= 0.1 else f"1/{base / growth:,.0f}"))
        lines.append(f"{label} & {errors(float(r['criterion_standard_errors']))[0]} & {scientific(growth)[0]} & {relative_text} \\\\")
    return lines + ["\\bottomrule", "\\end{tabular}"]


def table_playback():
    t = rows("playback/playback.csv")
    lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule", " & & Read & GPU memory & Disk & Decoding & RK4 step & Frame \\\\",
             "Grid & Chain & (MiB) & (MiB) & (ms) & (ms) & (ms) & (ms) \\\\", "\\midrule"]
    label = {("raw_psi", "left"): "raw frames", ("raw_psi", "right"): "\\codec", ("cuzfp_psi", "left"): "cuZFP, 2 bits", ("cuzfp_psi", "right"): "\\codec"}
    for grid in ("256", "512"):
        for pair in ("cuzfp_psi", "raw_psi"):
            for side in ("left", "right"):
                r = one(t, grid=grid, pair=pair, side=side)
                lines.append(f"${grid}^3$ & {label[(pair, side)]} & {float(r['mib_read_per_frame']):.2f} & {float(r['gpu_memory_mib']):.0f} & "
                             f"{float(r['disk_ms']):.2f} & {float(r['decoding_ms']):.2f} & {float(r['rk4_step_ms']):.2f} & {float(r['frame_cost_ms']):.1f} \\\\")
            lines.append("\\addlinespace")
    lines.pop()
    return lines + ["\\bottomrule", "\\end{tabular}"]


def table_equal_bound():
    lines = ["\\begin{tabular}{llrrrrrr}", "\\toprule", "Grid & Bound & \\codec & SZ3 in 4D & SZ3 & SPERR & cuSZp & zfp \\\\", "\\midrule"]
    for grid in ("256", "512"):
        t = rows(f"equal_bound/equal_bound_{grid}.csv")
        bounds = sorted({r["bound_over_urms"] for r in t if r["bound_over_urms"] != "none"}, key=float, reverse=True)
        for bound in bounds:
            cells = []
            for codec in ("psi_d3", "sz3_4d", "sz3", "sperr", "cuszp", "zfp"):
                found = [r for r in t if r["codec"] == codec and r["bound_over_urms"] == bound]
                if not found:
                    cells.append("")
                elif not found[0]["compression_ratio"]:
                    cells.append("not encoded")
                else:
                    cells.append(f"{float(found[0]['compression_ratio']):.1f} / {100 * float(found[0]['rms_error_relative']):.3f}")
            lines.append(f"${grid}^3$ & {100 * float(bound):g}\\,\\% & " + " & ".join(cells) + " \\\\")
        lines.append("\\addlinespace")
    lines.pop()
    return lines + ["\\bottomrule", "\\end{tabular}"]


def table_claims():
    sheet = (ROOT / "docs" / "CLAIMS.md").read_text(encoding="utf-8")
    found = re.findall(r"^\| ([A-Z]+-\d+) \| (.+?) \| (\**(?:holds|does not hold|pending)\**) \|$", sheet, re.MULTILINE)
    if not found:
        raise Missing("docs/CLAIMS.md has no summary table")
    escape = str.maketrans({"%": "\\%", "&": "\\&", "_": "\\_", "#": "\\#", "³": "$^3$", "′": "'", "η": "$\\eta$", "×": "$\\times$"})
    lines = ["\\begin{longtable}{@{}lp{10.6cm}l@{}}", "\\caption{The claims of the project and their verdicts.\\label{tab:claims}}\\\\", "\\toprule",
             "Identifier & Claim & Verdict \\\\", "\\midrule", "\\endfirsthead", "\\toprule", "Identifier & Claim & Verdict \\\\", "\\midrule", "\\endhead",
             "\\bottomrule", "\\endfoot"]
    for identifier, claim, verdict in found:
        verdict = verdict.strip("*")
        lines.append(f"{identifier} & {claim.translate(escape)} & {'\\textbf{' + verdict + '}' if verdict != 'holds' else verdict} \\\\")
    return lines + ["\\end{longtable}"]


TABLES = {"data": table_data, "divergence": table_divergence, "tracers": table_tracers, "playback": table_playback, "equal_bound": table_equal_bound,
          "claims": table_claims}


# ---- Outputs --------------------------------------------------------------------------------------------------------

def main():
    v, missing = values()
    tables = {}
    for name, function in TABLES.items():
        try:
            tables[name] = function()
        except Missing as e:
            missing.append(f"table {name}: {e}")
    if "--list" in sys.argv[1:]:
        for key in sorted(v):
            print(f"{key:36s} {v[key][1]}")
    paper = (ROOT / "paper" / "paper.tex").read_text(encoding="utf-8")
    template = (ROOT / "tools" / "README.template.md").read_text(encoding="utf-8")
    wanted = set(re.findall(r"\\val\{([^}]+)\}", paper)) | set(re.findall(r"\{\{([^}]+)\}\}", template))
    for key in sorted(wanted - set(v)):
        missing.append(f"key without a value: {key}")
    if missing:
        print(f"{len(missing)} problem(s):", file=sys.stderr)
        for m in missing:
            print("  " + m, file=sys.stderr)
        return 1
    if "--list" in sys.argv[1:]:
        return 0
    lines = ["% Written by tools/paper_numbers.py from the logs of benchmarks/. Do not edit: run the script again."]
    lines += [f"\\expandafter\\def\\csname val@{key}\\endcsname{{{v[key][0]}}}" for key in sorted(v)]
    (ROOT / "paper" / "numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    (ROOT / "paper" / "tables").mkdir(exist_ok=True)
    for name, body in tables.items():
        text = "% Written by tools/paper_numbers.py from the logs of benchmarks/. Do not edit: run the script again.\n" + "\n".join(body) + "\n"
        (ROOT / "paper" / "tables" / f"{name}.tex").write_text(text, encoding="utf-8", newline="\n")
    readme = re.sub(r"\{\{([^}]+)\}\}", lambda m: v[m.group(1)][1], template)
    readme = readme.rstrip("\n") + "\n"
    (ROOT / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    print(f"{len(v)} values, {len(tables)} tables; paper/numbers.tex, paper/tables/ and README.md written")
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
