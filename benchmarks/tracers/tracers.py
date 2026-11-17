# SPDX-License-Identifier: Apache-2.0
"""Tracer experiment: campaigns of the program build/tracers and their analysis (claims LAGR-1 to LAGR-5 and TRAC-1).

    python benchmarks/tracers/tracers.py run CAMPAIGN [CAMPAIGN ...]   runs campaigns; log CAMPAIGN.csv, outputs in logs/
    python benchmarks/tracers/tracers.py analyse [CAMPAIGN ...]        writes CAMPAIGN_analysis.txt and CAMPAIGN_summary.csv
    python benchmarks/tracers/tracers.py list                          the campaigns

Tracers drawn uniformly are advected by RK4 in ONE frame of the velocity field, held frozen, seen through a
reconstruction path (tools/tracers.cu). In a stationary divergence-free field the flow preserves volume: the tracers
stay uniform and their count per bin follows Poisson's law.

CRITERION, fixed before the first analysis and unchanged since:
  statistic       excess of the dispersion index (variance / mean - 1 of the number of tracers per bin);
  bins            1 voxel on a side;
  instant         10 Kolmogorov times, in a frozen frame;
  seeds           1 to 5, the same for every path;
  standard error  the larger of the standard error between seeds and that of a uniform draw, sqrt(2 / bins) / sqrt(seeds);
  thresholds      more than 5 standard errors for the trilinear texture and for cuZFP, less than 2 (in absolute value)
                  for Psi-treillis.
Everything else (other bin sizes, other instants, other seeds, statistics of D, the paths read from a decoded grid) is
descriptive: these figures do not decide the criterion, and none is corrected for multiple comparisons.

For each campaign, each path and each instant, the analysis gives the excess dispersion for three bin sizes (mean over
the seeds, standard error between seeds, expected standard error, number of standard errors), and D, the integral of
the divergence along the trajectories: standard deviation, mean, mean over variance (-1/2 if the total volume is
conserved and D is Gaussian), and the divergence seen by the tracers as a fraction of the RMS gradient. Then the growth
of var(D) per Kolmogorov time: var(D) at the last instant divided by that instant, mean over the seeds.

Needs build/tracers (tools/build.ps1 -Tools), the sequences of data/, and for the campaign `compressors` the comparison
codecs of build/codecs/.
"""

import collections
import csv
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "codecs"))

COLUMNS = ["grid", "frame", "path", "seed", "substeps", "tracers", "fd", "t_tau", "step", "gradient_rms", "tau_eta", "bin", "mean_count",
           "excess1", "excess2", "excess4", "empty1", "empty2", "empty4", "d_mean", "d_variance", "divergence_over_gradient",
           "empty_poisson", "reserved", "seconds"]

EIGHT_PER_VOXEL_256 = 134217728  # 8 tracers per voxel at 256^3, 1 per voxel at 512^3
SIXTEEN_MILLION = 16777216

# sequence: under data/. paths: (label in the campaign, options of the program). seeds: tracer seeds.
CAMPAIGNS = {
    "resolved_256": dict(
        about="turb256.seq, frame 32, 8 tracers per voxel, bins of 1 voxel, 10 Kolmogorov times (claims LAGR-1 to LAGR-4)",
        sequence="turb256.seq", frame=32, seeds=[1, 2, 3, 4, 5], options=["--particles", EIGHT_PER_VOXEL_256, "--ref-eps", 0.002],
        paths=["zfp", "tex", "cubic", "psi3", "psi2", "psi3b", "ref"]),
    "resolved_512": dict(
        about="turb512.seq, frame 32, 1 tracer per voxel, bins of 1 voxel, 10 Kolmogorov times (claims LAGR-1 to LAGR-4)",
        sequence="turb512.seq", frame=32, seeds=[1, 2, 3, 4, 5], options=["--particles", EIGHT_PER_VOXEL_256],
        paths=["zfp", "tex", "psi3"]),
    "confirmation_256": dict(
        about="sweep/nu00280_seed7.seq (another realisation), frame 32, same protocol as resolved_256 (claim LAGR-5)",
        sequence="sweep/nu00280_seed7.seq", frame=32, seeds=[1, 2, 3, 4, 5], options=["--particles", EIGHT_PER_VOXEL_256, "--ref-eps", 0.002],
        paths=["zfp", "tex", "psi3", "psi2", "psi3b", "ref"]),
    "confirmation_256_more_seeds": dict(
        about="the reference path of confirmation_256 on tracer seeds 6 to 10, run after seeing its result: descriptive only",
        sequence="sweep/nu00280_seed7.seq", frame=32, seeds=[6, 7, 8, 9, 10], options=["--particles", EIGHT_PER_VOXEL_256, "--ref-eps", 0.002],
        paths=["ref"]),
    "long_256": dict(
        about="turb256.seq, frame 32, 1 tracer per voxel, up to 100 Kolmogorov times, one seed (claim LAGR-3, descriptive)",
        sequence="turb256.seq", frame=32, seeds=[1], options=["--particles", SIXTEEN_MILLION, "--tau", "0,1,2,5,10,20,50,100"],
        paths=["zfp", "tex", "cubic", "psi3"]),
    "offsets_256": dict(
        about="turb256.seq, frame 32, 1 tracer per voxel, 20 Kolmogorov times, one seed: D with differences at +-1/4 and +-1/16 voxel",
        sequence="turb256.seq", frame=32, seeds=[1], options=["--particles", SIXTEEN_MILLION, "--tau", "0,1,2,5,10,20"],
        paths=["tex", "psi3"], variants=[["--fd", 0.25], ["--fd", 0.0625]]),
    "compressors_256": dict(
        about="sweep/nu00280_seed42.seq, frame 32, 8 tracers per voxel: grids decoded by SZ3, SZ3 in 4D and SPERR at a bound of 5 % (claim TRAC-1)",
        sequence="sweep/nu00280_seed42.seq", frame=32, seeds=[1, 2, 3, 4, 5], options=["--particles", EIGHT_PER_VOXEL_256],
        paths=["tex", "zfp", "grid:sz3", "grid:sz3_4d", "grid:sperr", "psi3b"], bound=0.05),
}

NAMES = {"psi2": "Psi-treillis, degree 2", "psi3": "Psi-treillis, degree 3", "psi3b": "Psi-treillis, degree 3, bound 5 %", "cubic": "tricubic",
         "tex": "trilinear texture", "zfp2": "cuZFP 2 bits + texture", "sz3": "SZ3 + texture", "sz3_4d": "SZ3 in 4D + texture",
         "sperr": "SPERR + texture"}
CRITERION_T, CRITERION_SIDE, CRITERION_SEEDS = 10, 1.0, {"1", "2", "3", "4", "5"}


def name(path):
    if path.startswith("ref"):
        return f"Psi-treillis reference (step 1 voxel, eps {path[3:]})"
    return NAMES.get(path, path)


# ---- Running --------------------------------------------------------------------------------------------------------

def run(campaign):
    import comparison_codecs as cc
    c = CAMPAIGNS[campaign]
    sequence = cc.DATA / c["sequence"]
    log = HERE / f"{campaign}.csv"
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)

    def relative(path):  # paths relative to the root: the logs then name no folder of the machine
        return path.relative_to(cc.ROOT).as_posix()

    with open(log, "w", newline="", encoding="utf-8") as fo:
        fo.write(",".join(COLUMNS) + "\n")
        for variant in c.get("variants", [[]]):
            for seed in c["seeds"]:
                for path in c["paths"]:
                    options = ["--path", path]
                    if path.startswith("grid:"):
                        import decoded_grids as dg
                        chain = path[5:]
                        grid, _ = dg.decoded(chain, sequence, c["bound"])
                        grid_file = cc.BUILD / "work" / "tracers" / f"{chain}.f32"
                        grid_file.parent.mkdir(parents=True, exist_ok=True)
                        grid.tofile(grid_file)
                        options = ["--path", "grid", "--grid", relative(grid_file), "--label", chain]
                    command = [cc.BUILD / f"tracers{cc.EXE}", "--data", relative(sequence), "--frame", c["frame"], "--seed", seed] + options + ["--csv"] + c["options"] + variant
                    tag = path.replace(":", "_") + "".join(f"_{str(v).lstrip('-')}" for v in variant)
                    p = subprocess.run([str(x) for x in command], cwd=cc.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                    out = p.stdout.decode("utf-8", "replace").replace("\r", "")
                    (logs / f"{campaign}_{tag}_s{seed}.log").write_text(out, encoding="utf-8", newline="\n")
                    for line in out.split("\n"):
                        if line.startswith("CSV,lagr,"):
                            fo.write(line[len("CSV,lagr,"):] + "\n")
                    fo.flush()
                    last = [line for line in out.split("\n") if line.startswith("t = ")]
                    print(f"{campaign}: {tag} seed {seed}: exit code {p.returncode}; {last[-1][:180] if last else out[-200:]}", flush=True)


# ---- Analysis -------------------------------------------------------------------------------------------------------

def stats(v):
    k = len(v)
    m = sum(v) / k
    se = math.sqrt(sum((x - m) ** 2 for x in v) / (k - 1) / k) if k > 1 else float("nan")
    return m, se


def analyse(campaign):
    path = HERE / f"{campaign}.csv"
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    lines = [f"== {campaign}: {CAMPAIGNS[campaign]['about']}"]
    summary = []
    g = collections.OrderedDict()
    for r in rows:
        key = (r["grid"], r["substeps"], r["tracers"], r["bin"], r["fd"], r["path"])
        g.setdefault(key, collections.OrderedDict()).setdefault(round(float(r["t_tau"])), []).append(r)

    lines.append("\n   CRITERION: excess dispersion, bins of 1 voxel, 10 Kolmogorov times, seeds 1 to 5")
    lines.append("      path                                 | seeds | mean excess | standard error between seeds | expected | standard errors | density variance | verdict")
    criterion = {}
    for (n, sp, np_, side_text, fd, p), by_t in g.items():
        side, tracers = float(side_text), int(np_)
        level = round(math.log2(CRITERION_SIDE / side))
        if level not in (0, 1, 2) or CRITERION_T not in by_t or p.startswith("grad"):
            continue
        v = [x for x in by_t[CRITERION_T] if x["seed"] in CRITERION_SEEDS]
        if not v:
            continue
        col = ("excess1", "excess2", "excess4")[level]
        m, se = stats([float(x[col]) for x in v])
        nb = round(int(n) / side)
        bins = nb ** 3 / 8 ** level
        per_bin = tracers / bins
        expected = math.sqrt(2.0 / bins) / math.sqrt(len(v))
        use = max(se, expected) if len(v) > 1 else expected
        z = m / use
        if len(v) < len(CRITERION_SEEDS):
            verdict = f"incomplete ({len(v)} seed(s) out of {len(CRITERION_SEEDS)})"
        elif p in ("tex", "zfp2"):
            verdict = "criterion met (more than 5)" if z > 5 else "criterion NOT met (more than 5 is needed)"
        elif p.startswith("psi") or p.startswith("ref"):
            verdict = "criterion met (less than 2)" if abs(z) < 2 else "criterion NOT met (less than 2 is needed)"
        else:
            verdict = "outside the criterion"
        criterion[(n, sp, np_, fd, p)] = (len(v), m, se, expected, z, per_bin)
        lines.append(f"      {name(p):<36} | {len(v):>5} | {m:+.5f}    | {se:.5f}                      | {expected:.5f}  | {z:+15.1f} | {m / per_bin:+.2e}         | {verdict}"
                     f"   [dt_frame/{sp}, {per_bin:g} tracer(s) per bin]")

    for (n, sp, np_, side_text, fd, p), by_t in g.items():
        tracers, side = int(np_), float(side_text)
        nb = round(int(n) / side)
        seeds = max(len(v) for v in by_t.values())
        lines.append(f"\n   {name(p)}: {n}^3, {tracers / int(n) ** 3:g} tracer(s) per voxel, time step dt_frame/{sp}, {seeds} seed(s), differences at +-{fd} voxel")
        lines.append("      t/tau_eta | excess dispersion per bin: mean +- standard error between seeds [expected] (in standard errors)                    | standard deviation of D | mean of D | mean/var | div/grad")
        for t, v in by_t.items():
            cells = []
            for level, col in enumerate(("excess1", "excess2", "excess4")):
                m, se = stats([float(x[col]) for x in v])
                bins = nb ** 3 / 8 ** level
                expected = math.sqrt(2.0 / bins) / math.sqrt(len(v))  # draw with a fixed total: (bins - 1) x index follows a chi-square with bins - 1 degrees
                use = max(se, expected) if len(v) > 1 else expected
                cells.append(f"{side * 2 ** level:g} vox: {m:+.5f} +- {se:.5f} [{expected:.5f}] ({m / use:+.1f})")
            dv = sum(float(x["d_variance"]) for x in v) / len(v)
            dm = sum(float(x["d_mean"]) for x in v) / len(v)
            dr = sum(float(x["divergence_over_gradient"]) for x in v) / len(v)
            lines.append(f"      {t:>9} | " + "; ".join(cells) + f" | {math.sqrt(max(dv, 0)):.4e} | {dm:+.3e} | {dm / dv if dv > 0 else float('nan'):+.3f} | {dr:.3e}")
        t_last = max(by_t)
        if t_last > 0:
            v = by_t[t_last]
            rate, rate_se = stats([float(x["d_variance"]) / float(x["t_tau"]) for x in v])
            dr = sum(float(x["divergence_over_gradient"]) for x in v) / len(v)
            d_std = math.sqrt(max(sum(float(x["d_variance"]) for x in v) / len(v), 0.0))
            lines.append(f"      growth of var(D): {rate:.4e} +- {rate_se:.1e} per Kolmogorov time (standard error between {len(v)} seed(s), over {t_last} times); "
                         f"standard deviation of D at {t_last} Kolmogorov times: {d_std:.4e}")
            c = criterion.get((n, sp, np_, fd, p))
            summary.append([campaign, n, p, fd, sp, np_, len(v), t_last, f"{rate:.6e}", f"{rate_se:.3e}", f"{d_std:.6e}", f"{dr:.6e}"]
                           + ([c[0], f"{c[1]:+.6e}", f"{c[2]:.6e}", f"{c[3]:.6e}", f"{c[4]:+.4f}", f"{c[5]:g}"] if c else ["", "", "", "", "", ""]))

    (HERE / f"{campaign}_analysis.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    with open(HERE / f"{campaign}_summary.csv", "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(["campaign", "grid", "path", "fd", "substeps", "tracers", "seeds", "last_t_tau", "var_d_growth_per_tau", "var_d_growth_standard_error",
                    "d_standard_deviation_at_last_t", "divergence_over_gradient", "criterion_seeds", "criterion_mean_excess",
                    "criterion_standard_error_between_seeds", "criterion_expected_standard_error", "criterion_standard_errors", "tracers_per_bin"])
        w.writerows(summary)
    print("\n".join(lines))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action, names = (sys.argv[1] if len(sys.argv) > 1 else ""), sys.argv[2:]
    unknown = [c for c in names if c not in CAMPAIGNS]
    if action == "list":
        for k, c in CAMPAIGNS.items():
            print(f"{k:28s} {c['about']}")
    elif action == "run" and names and not unknown:
        for c in names:
            run(c)
            analyse(c)
    elif action == "analyse" and not unknown:
        for c in names or [k for k in CAMPAIGNS if (HERE / f"{k}.csv").is_file()]:
            analyse(c)
    else:
        print(__doc__)
        sys.exit(2)
