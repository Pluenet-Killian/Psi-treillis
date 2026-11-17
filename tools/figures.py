# SPDX-License-Identifier: Apache-2.0
"""Draws the figures of the article and of README.md from the logs kept in benchmarks/. No GPU, no measurement: it
only reads CSV files, and every mark and every number in a figure comes from a log.

    python tools/figures.py                   draws everything
    python tools/figures.py NAME [NAME ...]   draws only the named figures
    python tools/figures.py --preview DIR     also writes a PNG rendering of each article figure in DIR, to inspect it

divergence   benchmarks/divergence/divergence.csv: divergence of one frame as each chain gives it back, on the grid
             and as read by trilinear interpolation, one marker per realisation;
tracers      benchmarks/tracers/compressors_256.csv: variance of D against time, mean over the tracer seeds;
equal_bound  benchmarks/equal_bound/equal_bound_256.csv: compression ratio against relative RMS error, one point per
             bound on the largest error;
validity     benchmarks/validity/validity.csv: the resolution sweep, error against the Kolmogorov scale of the data and
             compression ratio at a bound of 5 %.

Each figure is written three times:
  paper/figures/NAME.pdf                 for the article: vector, white background, no title (the article has captions),
                                         at its printed size on an A4 page with margins of 2.5 cm (text 16 cm wide;
                                         tracers and equal_bound are included at 0.78 of that width);
  docs/figures/NAME_{light,dark}.png     for README.md: title and subtitle inside the image, in the light and the dark
                                         theme of the page that shows it.

The script stops with a message if a log is missing or does not hold what a figure needs. Needs matplotlib and NumPy.

Visual system. The palette is the validated categorical order blue, orange, aqua, yellow. Psi-treillis is orange in
every figure and the original grid is in the neutral ink. Where a figure has more chains than the palette has colours
(divergence, tracers), colour gives the role: every general-purpose compressor is blue. Identity never rests on colour
alone: each chain has its marker shape and a label next to its marks, and text is always in the ink colours.
"""

import argparse
import csv
import logging
import math
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)  # the fallback font has no semibold weight
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper" / "figures"
README = ROOT / "docs" / "figures"
LOGS = {
    "divergence": ROOT / "benchmarks" / "divergence" / "divergence.csv",
    "tracers": ROOT / "benchmarks" / "tracers" / "compressors_256.csv",
    "equal_bound": ROOT / "benchmarks" / "equal_bound" / "equal_bound_256.csv",
    "validity": ROOT / "benchmarks" / "validity" / "validity.csv",
}

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", secondary="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
                  band="#f0efec", series=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", secondary="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
                 band="#2c2c2a", series=["#3987e5", "#d95926", "#199e70", "#c98500"]),
}

TEXT_WIDTH = 16.0 / 2.54  # inches: A4 with margins of 2.5 cm
NARROW = 0.78 * TEXT_WIDTH  # the width at which the article includes tracers and equal_bound

# Index in the series colours. Psi-treillis keeps its colour in every figure; None is the neutral ink.
ROLE_COLOUR = {"original": None, "compressor": 0, "psi": 1}
CODEC_COLOUR = {"sz3_4d": 0, "psi": 1, "sperr": 2, "sz3": 3}
MARKER = {"original": "P", "cuzfp": "v", "sperr": "^", "sz3": "D", "sz3_4d": "s", "psi": "o", "psi_step1": "X"}
SEED_MARKERS = ["o", "s", "^"]  # divergence: one shape per realisation
BOUND = "0.05"  # validity: the bound of the right panel and of the filled markers
# Two facts of the protocols that the logs do not record, used in the subtitles of the README figures only.
SWEEP_GRID = 256  # the sequences of the resolution sweep, also used by divergence (benchmarks/validity/validity.py)
TRACER_BOUND = 0.05  # bound given to SZ3, SZ3 in 4D and SPERR in the campaign compressors_256 (benchmarks/tracers/tracers.py)

SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def stop(message):
    sys.exit(f"figures.py: {message}")


# ---------------------------------------------------------------------------------------------------------------------
# Looks: the article and the two themes of README.md
# ---------------------------------------------------------------------------------------------------------------------

def look(kind, theme="light"):
    """Colours and sizes (in points) of one output. `paper` is the light theme on a white page, at its printed size."""
    L = dict(THEMES[theme], kind=kind, theme=theme)
    if kind == "paper":
        L.update(surface="#ffffff", font=8.5, small=8.0, line=1.5, marker=6.0, ring=1.0, hair=0.6)
    else:
        L.update(font=10.5, small=9.5, line=2.0, marker=8.0, ring=1.6, hair=0.8)
    return L


def style(L):
    plt.rcParams.update({
        "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": L["font"], "axes.labelsize": L["font"],
        "xtick.labelsize": L["small"], "ytick.labelsize": L["small"], "legend.fontsize": L["small"],
        "legend.title_fontsize": L["small"],
        "figure.facecolor": L["surface"], "axes.facecolor": L["surface"], "savefig.facecolor": L["surface"],
        "text.color": L["ink"], "axes.labelcolor": L["secondary"], "xtick.color": L["muted"], "ytick.color": L["muted"],
        "xtick.labelcolor": L["secondary"], "ytick.labelcolor": L["secondary"],
        "axes.edgecolor": L["axis"], "axes.linewidth": L["hair"], "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": L["hair"], "ytick.major.width": L["hair"], "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "axes.grid": True, "grid.color": L["grid"], "grid.linewidth": L["hair"], "grid.linestyle": "-", "axes.axisbelow": True,
        "legend.frameon": False, "legend.labelcolor": L["secondary"], "lines.linewidth": L["line"],
        "lines.solid_capstyle": "round", "lines.dash_capstyle": "round", "pdf.fonttype": 42,
    })


def colour(L, slot):
    return L["ink"] if slot is None else L["series"][slot]


def filled(L, c, marker, **more):
    """A filled marker with a thin ring of the surface colour, so that overlapping markers stay distinct."""
    size = L["marker"] * (1.15 if marker in "PX^v" else 1.0)  # these shapes have less ink than a disc of the same size
    return {**dict(marker=marker, ms=size, mfc=c, mec=L["surface"], mew=L["ring"]), **more}


def hollow(L, c, marker, **more):
    return {**dict(marker=marker, ms=L["marker"] - 0.5 * L["line"], mfc=L["surface"], mec=c, mew=L["line"]), **more}


# ---------------------------------------------------------------------------------------------------------------------
# Layout: every length is in inches, so that text keeps its size whatever the size of the figure
# ---------------------------------------------------------------------------------------------------------------------

def panel(fig, left, bottom, width, height):
    w, h = fig.get_size_inches()
    return fig.add_axes([left / w, bottom / h, width / w, height / h])


def note(fig, x, y_from_top, text, **more):
    """Text at x inches from the left edge and y inches below the top edge of the figure."""
    w, h = fig.get_size_inches()
    return fig.text(x / w, 1.0 - y_from_top / h, text, **more)


def heading_height(lines):
    """Height in inches of a title and a subtitle of that many lines."""
    return 0.45 + 0.19 * lines


def heading(fig, L, x, head, lines):
    """Title and subtitle of a README figure, from the top edge down. Returns the height used, in inches."""
    note(fig, x, 0.16, head, fontsize=12, fontweight="semibold", color=L["ink"], va="top")
    for k, line in enumerate(lines):
        note(fig, x, 0.45 + 0.19 * k, line, fontsize=L["small"], color=L["secondary"], va="top")
    return heading_height(len(lines))


def plain(v, _=None):
    return f"{v:g}"


def power(v, _=None):
    e = round(math.log10(v))
    return "1" if e == 0 else "10" if e == 1 else "10" + str(e).translate(SUPERSCRIPT)


def log_axis(axis, ticks, formatter):
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(formatter))


def decades(values):
    """Exponents of the powers of ten that enclose the values."""
    return math.floor(math.log10(min(values))), math.ceil(math.log10(max(values)))


def end_labels(ax, L, items, clear=None):
    """Direct labels right of the last marker of each line, in text ink. `items` holds (x, y, text) in data
    coordinates. Labels whose lines end at the same place are merged into one; the others are moved apart vertically
    by the least amount that keeps them from touching. `clear` is the distance in points from the last point to its
    label (default: past a marker of the usual size)."""
    to_px, to_data = ax.transData, ax.transData.inverted()
    px = ax.figure.dpi / 72.0
    gap = 1.3 * L["small"] * px
    marks = sorted((to_px.transform((x, y))[1], to_px.transform((x, y))[0], k, text) for k, (x, y, text) in enumerate(items))
    groups = []
    for y, x, k, text in marks:
        if groups and y - groups[-1]["y"][-1] < 0.3 * gap:
            groups[-1]["y"].append(y)
            groups[-1]["x"] = max(groups[-1]["x"], x)
            groups[-1]["texts"].append((k, text))
        else:
            groups.append(dict(y=[y], x=x, texts=[(k, text)]))
    at = [statistics.fmean(g["y"]) for g in groups]
    for _ in range(200):
        for i in range(1, len(at)):
            lack = gap - (at[i] - at[i - 1])
            if lack > 0:
                at[i - 1] -= 0.5 * lack
                at[i] += 0.5 * lack
    for g, y in zip(groups, at):
        text = ", ".join(t for _, t in sorted(g["texts"]))
        ax.annotate(text, to_data.transform((g["x"], y)), xytext=(clear or 0.5 * L["marker"] + 5.0, 0), textcoords="offset points", va="center",
                    fontsize=L["small"], color=L["secondary"], annotation_clip=False)


def save(fig, L, name, preview):
    if L["kind"] == "paper":
        PAPER.mkdir(parents=True, exist_ok=True)
        out = PAPER / f"{name}.pdf"
        fig.savefig(out, metadata={"Creator": "tools/figures.py", "CreationDate": None, "ModDate": None})
        if preview:
            preview.mkdir(parents=True, exist_ok=True)
            fig.savefig(preview / f"{name}_paper.png", dpi=300)
    else:
        README.mkdir(parents=True, exist_ok=True)
        out = README / f"{name}_{L['theme']}.png"
        fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Logs
# ---------------------------------------------------------------------------------------------------------------------

def read_log(name, columns):
    path = LOGS[name]
    where = path.relative_to(ROOT).as_posix()
    if not path.is_file():
        stop(f"the log {where} is missing: the figure '{name}' cannot be drawn (see the README.md of its folder to replay the measurement)")
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        stop(f"the log {where} is empty")
    absent = [c for c in columns if c not in rows[0]]
    if absent:
        stop(f"the log {where} has no column {', '.join(absent)}")
    return rows, where


def need(table, key, where):
    if key not in table:
        stop(f"the log {where} has no row for {' / '.join(str(k) or 'no bound' for k in key)}")
    return table[key]


# ---------------------------------------------------------------------------------------------------------------------
# divergence
# ---------------------------------------------------------------------------------------------------------------------

# (chain in the log, label of the row, role); `psi_free` of the log is not drawn.
CHAINS = [("raw", "original grid", "original"), ("zfp", "zfp", "compressor"), ("cuzfp_2bits", "cuZFP, 2 bits", "compressor"),
          ("sperr", "SPERR", "compressor"), ("sz3", "SZ3", "compressor"), ("sz3_4d", "SZ3 in 4D", "compressor"),
          ("cuszp", "cuSZp", "compressor"), ("psi", "Psi-treillis, decoded grid", "psi")]
DIVERGENCE_PANELS = [("grid_divergence_over_gradient", "On the grid"),
                     ("read_divergence_over_gradient", "As read by trilinear\ninterpolation")]


def read_divergence():
    rows, where = read_log("divergence", ["solver_seed", "chain", "frame", "bound_over_urms", "exit_code"] + [c for c, _ in DIVERGENCE_PANELS])
    seeds = list(dict.fromkeys(r["solver_seed"] for r in rows))
    if len(seeds) > len(SEED_MARKERS):
        stop(f"the log {where} holds {len(seeds)} realisations and the figure has {len(SEED_MARKERS)} marker shapes")
    table = {(r["chain"], r["solver_seed"]): r for r in rows}
    values = {}
    for chain, _, _ in CHAINS:
        for seed in seeds:
            r = need(table, (chain, seed), where)
            if r["exit_code"] != "0":
                stop(f"the log {where} has an exit code {r['exit_code']} for {chain}, seed {seed}")
            values[(chain, seed)] = [float(r[c]) for c, _ in DIVERGENCE_PANELS]
    bounds = sorted({r["bound_over_urms"] for r in rows if r["bound_over_urms"] and r["chain"] in [c for c, _, _ in CHAINS]})
    if len(bounds) != 1:
        stop(f"the log {where} holds the bounds {', '.join(bounds)}: the figure expects one")
    return dict(seeds=seeds, values=values, bound=float(bounds[0]), frame=rows[0]["frame"])


def draw_divergence(L, d):
    paper = L["kind"] == "paper"
    seeds, values = d["seeds"], d["values"]
    spans = [decades([v[k] for v in values.values()]) for k in range(2)]
    count = [hi - lo for lo, hi in spans]
    if paper:
        width, left, right, between, row, top, bottom = TEXT_WIDTH, 1.50, 0.16, 0.26, 0.36, 0.42, 0.56
    else:
        width, left, right, between, row, bottom = 10.0, 1.95, 0.30, 0.42, 0.47, 0.70
    rows = len(CHAINS)
    lines = [f"Frame {d['frame']} of {len(seeds)} realisations at {SWEEP_GRID}³, one marker each; divergence relative to the RMS velocity gradient.",
             f"Bound of {100 * d['bound']:g} % of u' on the largest error; cuZFP at a fixed rate. Psi-treillis reads the curl of its potential, "
             "not this grid."]
    if not paper:
        top = heading_height(len(lines)) + 0.56
    fig = plt.figure(figsize=(width, top + rows * row + bottom))
    if not paper:
        heading(fig, L, 0.22, "Divergence of one frame as each chain gives it back", lines)
    decade = (width - left - right - between) / sum(count)  # the same length in both panels
    step = 0.29  # vertical distance between two realisations of a row, in rows
    x0 = left
    for k, (_, header) in enumerate(DIVERGENCE_PANELS):
        ax = panel(fig, x0, bottom, decade * count[k], rows * row)
        for i, (chain, _, role) in enumerate(CHAINS):
            for j, seed in enumerate(seeds):
                ax.plot([values[(chain, seed)][k]], [i + step * (j - 0.5 * (len(seeds) - 1))], ls="none", zorder=3, clip_on=False,
                        **filled(L, colour(L, ROLE_COLOUR[role]), SEED_MARKERS[j]))
        ax.set_xscale("log")
        ax.set_xlim(10.0 ** spans[k][0], 10.0 ** spans[k][1])
        log_axis(ax.xaxis, [10.0 ** e for e in range(spans[k][0], spans[k][1] + 1)], power)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_yticks(range(rows))
        ax.set_yticklabels([label for _, label, _ in CHAINS] if k == 0 else [])
        ax.tick_params(axis="y", length=0, pad=6)
        ax.spines["left"].set_visible(False)
        ax.text(0.0, 1.0, header, transform=ax.transAxes, va="bottom", ha="left", fontsize=L["font"], color=L["ink"], linespacing=1.15)
        x0 += decade * count[k] + between
    w, h = fig.get_size_inches()
    centre = left + 0.5 * (width - left - right)
    low = 0.08 if paper else 0.12
    fig.text(centre / w, low / h, "RMS divergence over the RMS velocity gradient", ha="center", va="bottom", fontsize=L["font"],
             color=L["secondary"])
    # Key of the marker shapes, in the margin under the names of the rows. Within a row the realisations are drawn one
    # under the other, in the order of the key, so that close values do not hide each other.
    keys = [Line2D([], [], ls="none", label="Solver seed")]
    keys += [Line2D([], [], ls="none", label=seed, **filled(L, L["muted"], SEED_MARKERS[j])) for j, seed in enumerate(seeds)]
    fig.legend(handles=keys, ncol=len(keys), loc="lower left", bbox_to_anchor=((0.02 if paper else 0.10) / w, low / h), handlelength=1.0,
               handletextpad=0.1, columnspacing=0.7, borderpad=0.0, borderaxespad=0.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# tracers
# ---------------------------------------------------------------------------------------------------------------------

# (path in the log, label, role, marker, line style); SZ3 in 4D is drawn hollow because it lies on SZ3.
PATHS = [("tex", "original grid", "original", MARKER["original"], "-"),
         ("zfp2", "cuZFP, 2 bits", "compressor", MARKER["cuzfp"], (0, (5, 2.2))),
         ("sperr", "SPERR", "compressor", MARKER["sperr"], (0, (5, 2, 1, 2))),
         ("sz3", "SZ3", "compressor", MARKER["sz3"], "-"),
         ("sz3_4d", "SZ3 in 4D", "compressor", MARKER["sz3_4d"], (0, (1, 2))),
         ("psi3b", "Psi-treillis", "psi", MARKER["psi"], "-")]


def read_tracers():
    rows, where = read_log("tracers", ["path", "seed", "t_tau", "d_variance", "grid", "frame"])
    samples = {}
    for r in rows:
        if float(r["t_tau"]) > 0.0:
            samples.setdefault((r["path"], round(float(r["t_tau"]))), []).append((r["seed"], float(r["t_tau"]), float(r["d_variance"])))
    instants = sorted({k[1] for k in samples})
    seeds = sorted({s for v in samples.values() for s, _, _ in v}, key=int)
    lines = {}
    for path, *_ in PATHS:
        points = []
        for instant in instants:
            v = need(samples, (path, instant), where)
            if sorted((s for s, _, _ in v), key=int) != seeds:
                stop(f"the log {where} does not hold every seed for {path} at {instant} Kolmogorov times")
            points.append((statistics.fmean(t for _, t, _ in v), statistics.fmean(x for _, _, x in v)))
        lines[path] = points
    return dict(lines=lines, instants=instants, seeds=len(seeds), grid=rows[0]["grid"], frame=rows[0]["frame"])


def draw_tracers(L, d):
    paper = L["kind"] == "paper"
    if paper:
        width, height, left, right, top, bottom = NARROW, 3.30, 0.62, 1.12, 0.10, 0.48
    else:
        width, height, left, right, bottom = 7.4, 5.1, 0.82, 1.70, 0.62
    fig = plt.figure(figsize=(width, height))
    if not paper:
        top = 0.22 + heading(fig, L, 0.22, "Variance of D, the logarithm of the volume carried by a tracer", [
            f"Frame {d['frame']} of a {d['grid']}³ sequence, held frozen; mean of {d['seeds']} tracer seeds. "
            f"SZ3 and SPERR at a bound of {100 * TRACER_BOUND:g} % of u'.",
            "For Psi-treillis, D holds only the truncation error of the differences that measure it."])
    ax = panel(fig, left, bottom, width - left - right, height - top - bottom)
    ends = []
    for path, label, role, marker, dashes in PATHS:
        c = colour(L, ROLE_COLOUR[role])
        t, v = zip(*d["lines"][path])
        ax.plot(t, v, color=c, ls=dashes, zorder=2)
        look_of = hollow(L, c, marker, ms=1.65 * L["marker"], mew=0.7 * L["line"]) if path == "sz3_4d" else filled(L, c, marker)
        ax.plot(t, v, ls="none", zorder=2.5 if path == "sz3_4d" else 3, clip_on=False, **look_of)
        ends.append((t[-1], v[-1], label))
    every = [v for line in d["lines"].values() for _, v in line]
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.84 * d["instants"][0], 1.19 * d["instants"][-1])
    ax.set_ylim(min(every) / 2.5, max(every) * 2.5)
    log_axis(ax.xaxis, d["instants"], plain)
    lo, hi = decades(every)
    log_axis(ax.yaxis, [10.0 ** e for e in range(lo, hi + 1)], power)
    ax.set_xlabel("Time (Kolmogorov times)")
    ax.set_ylabel("Variance of D")
    end_labels(ax, L, ends, clear=L["marker"] + 5.0)
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# equal_bound
# ---------------------------------------------------------------------------------------------------------------------

# (codec in the log, label, colour and marker key, where the label goes: point, offset in marker sizes, alignment)
CODECS = [("psi_d3", "Psi-treillis", "psi", (-1, (-1.3, 0.35), "right")),
          ("sz3_4d", "SZ3 in 4D", "sz3_4d", (-1, (-1.3, 0.35), "right")),
          ("sz3", "SZ3", "sz3", (-1, (-1.3, 0.35), "right")),
          ("sperr", "SPERR", "sperr", (-1, (1.3, -0.35), "left"))]
# Where the bound is written next to the two end points of each curve: (offset in marker sizes, horizontal, vertical).
BOUND_NOTES = {"psi_d3": [((-1.0, 0.5), "right", "center"), ((0.0, 1.0), "center", "bottom")],
               "sz3_4d": [((0.25, -0.95), "center", "top"), ((0.0, 1.0), "center", "bottom")],
               "sz3": [((0.0, -1.0), "center", "top"), ((0.0, 1.0), "center", "bottom")],
               "sperr": [((0.0, -1.0), "center", "top"), ((0.0, 1.0), "center", "bottom")]}
NOT_DRAWN = [("cuszp", "cuSZp"), ("zfp", "zfp")]


def read_equal_bound():
    rows, where = read_log("equal_bound", ["grid", "codec", "bound_over_urms", "exit_code", "compression_ratio", "rms_error_relative", "frames"])
    curves, refused, low = {}, [], []
    for r in rows:
        if r["bound_over_urms"] == "none":
            continue
        if r["exit_code"] != "0" or not r["compression_ratio"]:
            refused.append((r["codec"], float(r["bound_over_urms"])))
            continue
        point = (float(r["bound_over_urms"]), 100.0 * float(r["rms_error_relative"]), float(r["compression_ratio"]))
        if r["codec"] in [c for c, _ in NOT_DRAWN]:
            low.append(point[2])
        curves.setdefault(r["codec"], []).append(point)
    for codec, *_ in CODECS:
        if len(curves.get(codec, [])) < 2:
            stop(f"the log {where} has fewer than two encoded bounds for {codec}")
        curves[codec].sort()
    first, last = rows[0]["frames"].split("-")
    return dict(curves=curves, refused=refused, low=low, grid=rows[0]["grid"], frames=int(last) - int(first) + 1)


def draw_equal_bound(L, d):
    paper = L["kind"] == "paper"
    if paper:
        width, height, left, right, top, bottom = NARROW, 3.30, 0.60, 0.30, 0.16, 0.48
    else:
        width, height, left, right, bottom = 7.4, 5.1, 0.80, 0.50, 0.62
    fig = plt.figure(figsize=(width, height))
    if not paper:
        bounds = sorted({100.0 * p[0] for c, *_ in CODECS for p in d["curves"][c]}, reverse=True)
        listed = ", ".join(f"{b:g}" for b in bounds[:-1]) + f" and {bounds[-1]:g}"
        lines = [f"{d['grid']}³, {d['frames']} frames. Along each curve the bound on the largest error is {listed} % of u'."]
        second = [f"Psi-treillis could not be encoded at {100.0 * b:g} %." for c, b in d["refused"] if c == "psi_d3"]
        if d["low"]:
            names = " and ".join(label for _, label in NOT_DRAWN)
            second.append(f"Not drawn: {names}, at ratios of {min(d['low']):.0f} to {max(d['low']):.0f}.")
        top = 0.22 + heading(fig, L, 0.22, "Compression ratio against RMS error, at equal bounds", lines + [" ".join(second)] * bool(second))
    ax = panel(fig, left, bottom, width - left - right, height - top - bottom)
    m = L["marker"]
    for codec, label, key, (at, offset, align) in CODECS:
        c = colour(L, CODEC_COLOUR[key])
        points = d["curves"][codec]
        x, y = [p[1] for p in points], [p[2] for p in points]
        ax.plot(x, y, color=c, zorder=2)
        ax.plot(x, y, ls="none", zorder=3, **filled(L, c, MARKER[key]))
        ax.annotate(label, (x[at], y[at]), xytext=(offset[0] * m, offset[1] * m), textcoords="offset points", ha=align, va="center",
                    fontsize=L["small"], color=L["secondary"])
        for end, (shift, ha, va) in zip((0, -1), BOUND_NOTES[codec]):
            ax.annotate(f"{100.0 * points[end][0]:g} %", (x[end], y[end]), xytext=(shift[0] * m, shift[1] * m), textcoords="offset points",
                        ha=ha, va=va, fontsize=L["small"] - 0.5, color=L["muted"])
    ax.set_xscale("log")
    ax.set_yscale("log")
    xs = [p[1] for c, *_ in CODECS for p in d["curves"][c]]
    ys = [p[2] for c, *_ in CODECS for p in d["curves"][c]]
    ax.set_xlim(min(xs) / 1.22, max(xs) * 1.22)
    ax.set_ylim(min(ys) / 1.25, max(ys) * 1.3)
    log_axis(ax.xaxis, [t for t in (0.1, 0.2, 0.3, 0.5, 1, 2, 3, 5, 10) if min(xs) / 1.22 <= t <= max(xs) * 1.22], plain)
    log_axis(ax.yaxis, [t for t in (10, 20, 50, 100, 200, 300, 500, 1000) if min(ys) / 1.25 <= t <= max(ys) * 1.3], plain)
    ax.set_xlabel("Relative RMS error (%)")
    ax.set_ylabel("Compression ratio")
    return fig


# ---------------------------------------------------------------------------------------------------------------------
# validity
# ---------------------------------------------------------------------------------------------------------------------

# Right panel: (codec in the log, label, colour key, marker key, line style).
SWEEP = [("psi_d3_mf2", "Psi-treillis, step 2", "psi", "psi", "-"), ("psi_d3_mf1", "Psi-treillis, step 1", "psi", "psi_step1", (0, (4, 2))),
         ("sperr", "SPERR", "sperr", "sperr", "-"), ("sz3_4d_32", "SZ3 in 4D", "sz3_4d", "sz3_4d", "-")]


def read_validity():
    """One entry per sequence of the sweep. The compression ratio is that of the group of the first 32 frames."""
    rows, where = read_log("validity", ["sequence", "eta_voxels", "codec", "bound_over_urms", "exit_code", "group_compression_ratio",
                                        "rms_error_relative"])
    table = {(r["sequence"], r["codec"], r["bound_over_urms"]): r for r in rows}
    out = []
    for name in sorted({r["sequence"] for r in rows}):
        free = need(table, (name, "psi_d3_mf2", ""), where)
        if free["exit_code"] != "0":
            stop(f"the log {where} has no error without a bound for {name}")
        ratios = []
        for codec, *_ in SWEEP:
            r = need(table, (name, codec, BOUND), where)
            if r["exit_code"] != "0" and not codec.startswith("psi"):
                stop(f"the log {where} has an exit code {r['exit_code']} for {codec} on {name}")
            ratios.append(float(r["group_compression_ratio"]) if r["exit_code"] == "0" else None)
        out.append(dict(eta=float(free["eta_voxels"]), error=100.0 * float(free["rms_error_relative"]),
                        held=ratios[0] is not None, ratios=ratios))  # ratios[0]: Psi-treillis at the default step, with the bound
    return sorted(out, key=lambda s: s["eta"])


def draw_validity(L, seqs):
    paper = L["kind"] == "paper"
    if paper:
        width, height, left, wide, between, right, top, bottom = TEXT_WIDTH, 2.75, 0.47, 2.05, 0.66, 1.33, 0.10, 0.48
    else:
        width, height, left, wide, between, right, bottom = 11.2, 4.75, 0.72, 3.95, 1.02, 1.56, 0.62
    fig = plt.figure(figsize=(width, height))
    if not paper:
        top = 0.22 + max(
            heading(fig, L, 0.22, "Error against the resolution of the data", [
                "Psi-treillis of degree 3, lattice step of 2 voxels, no bound.",
                f"One marker per sequence ({len(seqs)} sequences at {SWEEP_GRID}³)."]),
            heading(fig, L, left + wide + between - 0.50, f"Compression ratio at an equal bound of {100 * float(BOUND):g} %", [
                f"Largest error at most {100 * float(BOUND):g} % of u'. One marker per sequence, joined in the order of the scale.",
                "Step: lattice step in voxels. Psi-treillis at step 2 is drawn where it held the bound."]))
    tall = height - top - bottom
    a = panel(fig, left, bottom, wide, tall)
    b = panel(fig, left + wide + between, bottom, wide, tall)
    orange = colour(L, CODEC_COLOUR["psi"])
    span = (min(s["eta"] for s in seqs) - 0.03, max(s["eta"] for s in seqs) + 0.03)

    # Left: error without a bound; the fill of the marker says whether the bound could be held.
    for held, how in ((True, filled), (False, hollow)):
        chosen = [s for s in seqs if s["held"] == held]
        a.plot([s["eta"] for s in chosen], [s["error"] for s in chosen], ls="none", zorder=3 if held else 2.5, **how(L, orange, MARKER["psi"]))
    a.set_xlim(*span)
    a.set_ylim(0, 1.08 * max(s["error"] for s in seqs))
    a.set_xlabel("Kolmogorov scale of the data (voxels)")
    a.set_ylabel("Relative RMS error without a bound (%)")
    percent = f"{100 * float(BOUND):g} %"
    a.legend(handles=[Line2D([], [], ls="none", label=f"bound of {percent} held", **filled(L, orange, MARKER["psi"])),
                      Line2D([], [], ls="none", label=f"bound of {percent} refused", **hollow(L, orange, MARKER["psi"]))],
             loc="upper right", handletextpad=0.2, borderaxespad=0.3, labelspacing=0.45)

    # Right: compression ratio at the bound; one marker per sequence. The line joins the sequences in the order of their
    # Kolmogorov scale and is broken where a sequence could not be encoded.
    ends = []
    for k, (_, label, colour_key, marker_key, dashes) in enumerate(SWEEP):
        c = colour(L, CODEC_COLOUR[colour_key])
        ratios = [math.nan if s["ratios"][k] is None else s["ratios"][k] for s in seqs]
        b.plot([s["eta"] for s in seqs], ratios, color=c, ls=dashes, zorder=2)
        # Psi-treillis at step 2 has the fewest points and two of them lie on those of SPERR: it is drawn on top.
        b.plot([s["eta"] for s in seqs], ratios, ls="none", zorder=4 if k == 0 else 3, **filled(L, c, MARKER[marker_key]))
        last = max(i for i, r in enumerate(ratios) if not math.isnan(r))
        ends.append((seqs[last]["eta"], ratios[last], label))
    every = [r for s in seqs for r in s["ratios"] if r is not None]
    b.set_yscale("log")
    b.set_xlim(*span)
    b.set_ylim(min(every) / 1.25, max(every) * 1.25)
    log_axis(b.yaxis, [t for t in (10, 20, 50, 100, 200, 500, 1000) if min(every) / 1.25 <= t <= max(every) * 1.25], plain)
    b.set_xlabel("Kolmogorov scale of the data (voxels)")
    b.set_ylabel(f"Compression ratio at the bound of {percent}")
    end_labels(b, L, ends)
    return fig


# ---------------------------------------------------------------------------------------------------------------------

FIGURES = {"divergence": (read_divergence, draw_divergence), "tracers": (read_tracers, draw_tracers),
           "equal_bound": (read_equal_bound, draw_equal_bound), "validity": (read_validity, draw_validity)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Draws the figures of the article and of README.md from the logs of benchmarks/.")
    parser.add_argument("names", nargs="*", metavar="NAME", help="figures to draw (default: all): " + ", ".join(FIGURES))
    parser.add_argument("--preview", metavar="DIR", type=Path, help="also write a PNG rendering of each article figure in DIR")
    args = parser.parse_args()
    unknown = [n for n in args.names if n not in FIGURES]
    if unknown:
        stop(f"no figure named {', '.join(unknown)}; the figures are {', '.join(FIGURES)}")
    for name in args.names or list(FIGURES):
        read, draw = FIGURES[name]
        data = read()
        for L in (look("paper"), look("readme", "light"), look("readme", "dark")):
            style(L)
            out = save(draw(L, data), L, name, args.preview)
            print(out.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    sys.exit(main())
