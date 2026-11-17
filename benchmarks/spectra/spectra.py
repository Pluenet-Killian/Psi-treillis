# SPDX-License-Identifier: Apache-2.0
"""Energy spectrum of one frame as each chain gives it back on the grid (claim SPEC-1).

    python benchmarks/spectra/spectra.py run       measures the three realisations; logs spectra.csv and shells.csv
    python benchmarks/spectra/spectra.py summary   the table, from spectra.csv

Instrument, in double precision: the energy spectrum by integer shells of the wavenumber, E(k) = half the sum of
|u_hat|^2 over the modes whose wavenumber rounds to k, on a periodic domain of length 2 pi (u_hat normalised so that the
sum of E(k) over the shells is the kinetic energy per unit mass, (u^2 + v^2 + w^2) / 2 averaged over the grid).

    energy     sum of E(k) over the shells
    enstrophy  sum of k^2 E(k) over the shells
    k10        the smallest shell k >= 1 where |E_decoded(k) / E_original(k) - 1| exceeds 10 %, searched up to the
               truncation wavenumber of the solver, n / 3; "none" if no shell does

spectra.csv gives these three figures for each chain, relative to the original frame; shells.csv gives E(k) for every
chain and every shell, for the figure.

Data, chains and bound: benchmarks/codecs/decoded_grids.py.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import scipy.fft as sfft

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "codecs"))
import comparison_codecs as cc  # noqa: E402
import decoded_grids as dg  # noqa: E402

SEEDS = [42, 7, 11]
BOUND = 0.05
THRESHOLD = 0.10
COLUMNS = ["sequence", "solver_seed", "eta_voxels", "frame", "chain", "bound_over_urms", "truncation_wavenumber", "energy_over_original",
           "enstrophy_over_original", "k10"]


def shell_spectrum(grid):
    """E(k) for every integer shell k, from 0 to the largest wavenumber of the grid, of a grid (3, n, n, n)."""
    n = grid.shape[1]
    k = sfft.fftfreq(n, 1.0 / n)
    shell = np.rint(np.sqrt(k[:, None, None] ** 2 + k[None, :, None] ** 2 + k[None, None, :] ** 2)).astype(np.int64)
    energy = np.zeros(int(shell.max()) + 1)
    for c in range(3):
        hat = sfft.fftn(grid[c].astype(np.float64), workers=-1)
        energy += np.bincount(shell.ravel(), weights=(hat.real ** 2 + hat.imag ** 2).ravel(), minlength=energy.size)
    return 0.5 * energy / float(n) ** 6


def k10(original, decoded, truncation):
    for k in range(1, truncation + 1):
        if original[k] > 0.0 and abs(decoded[k] / original[k] - 1.0) > THRESHOLD:
            return k
    return None


def run():
    with open(HERE / "spectra.csv", "w", newline="", encoding="utf-8") as fo, open(HERE / "shells.csv", "w", newline="", encoding="utf-8") as fs:
        w, ws = csv.writer(fo), csv.writer(fs)
        w.writerow(COLUMNS)
        ws.writerow(["sequence", "solver_seed", "chain", "bound_over_urms", "k", "energy"])
        for seed in SEEDS:
            sequence = dg.realisation(seed)
            header = cc.read_header(sequence)
            n = header["n"]
            truncation = n // 3
            original = shell_spectrum(dg.decoded("raw", sequence)[0])
            shells = np.arange(original.size, dtype=np.float64)
            for chain, bound in dg.chains(BOUND):
                grid, _ = dg.decoded(chain, sequence, bound)
                spectrum = shell_spectrum(grid)
                first = k10(original, spectrum, truncation)
                tag = "" if bound is None else f"{bound:g}"
                row = [sequence.name, seed, f"{cc.eta_in_voxels(header):.4f}", dg.FRAME, chain, tag, truncation,
                       f"{spectrum.sum() / original.sum():.6f}", f"{(shells ** 2 * spectrum).sum() / (shells ** 2 * original).sum():.6f}",
                       "none" if first is None else first]
                w.writerow(row)
                fo.flush()
                for k in range(spectrum.size):
                    ws.writerow([sequence.name, seed, chain, tag, k, f"{spectrum[k]:.9e}"])
                print("  ".join(str(x) for x in row), flush=True)
    summary()


def summary():
    rows = list(csv.DictReader(open(HERE / "spectra.csv", encoding="utf-8")))
    seeds = sorted({r["solver_seed"] for r in rows}, key=lambda s: SEEDS.index(int(s)))
    names = []
    for r in rows:
        if r["chain"] not in names:
            names.append(r["chain"])
    print(f"\nSpectrum of the decoded grid relative to the original, frame {rows[0]['frame']}: enstrophy ratio / k10 (truncation at {rows[0]['truncation_wavenumber']})")
    print(f"{'chain':14s}" + "".join(f"  seed {s:>3s}: {'enstrophy':>10s} {'k10':>5s}" for s in seeds))
    for c in names:
        cells = []
        for s in seeds:
            r = next(x for x in rows if x["chain"] == c and x["solver_seed"] == s)
            cells.append(f"            {float(r['enstrophy_over_original']):10.4f} {r['k10']:>5s}")
        print(f"{c:14s}" + "".join(cells))


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    action = sys.argv[1] if len(sys.argv) > 1 else ""
    if action == "run":
        run()
    elif action == "summary":
        summary()
    else:
        print(__doc__)
        sys.exit(2)
