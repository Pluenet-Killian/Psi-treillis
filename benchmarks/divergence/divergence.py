# SPDX-License-Identifier: Apache-2.0
"""Divergence of one frame as each chain gives it back (claims GRID-1, GRID-2, READ-1, READ-2).

    python benchmarks/divergence/divergence.py run       measures the three realisations; log divergence.csv
    python benchmarks/divergence/divergence.py summary   the table, from the log

Two instruments, both in double precision, both divided by G, the RMS of the velocity gradient of the original frame
(sqrt of the mean over the grid of the sum over i, j of (du_i/dx_j)^2, computed spectrally):

  M1, divergence on the grid. Spectral divergence of the grid (periodic FFT; the Nyquist wavenumber of each axis is
      set to zero, as for any odd derivative on an even grid), RMS over the grid. No interpolation is involved: it
      measures what compression alone does to the frame.
  M2, divergence as read. Analytic divergence of the trilinear interpolant of the grid (values at voxel centres,
      periodic), at 4,194,304 points drawn uniformly (NumPy default_rng(20261117)), RMS. It is what a tracer sees when
      the grid is read by trilinear interpolation, without the rounding of a half-precision texture.

For Psi-treillis the field that is read is not this grid but the analytic curl of the potential, whose divergence is
zero by identity (claim DIV-1 measures it in double precision). The grid decoded by the library is measured here like
the others, for what it is: a sampling of that field at voxel centres.

Data, chains and bound: benchmarks/codecs/decoded_grids.py. Needs build/ (tools/build.ps1 -Tools), build/codecs/ and the
three sequences data/sweep/nu00280_seed{42,7,11}.seq.
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
POINTS = 4_194_304
POINT_SEED = 20261117
LENGTH = 2.0 * np.pi
COLUMNS = ["sequence", "solver_seed", "eta_voxels", "frame", "chain", "bound_over_urms", "exit_code", "compression_ratio",
           "rms_error_relative", "largest_error_over_urms", "gradient_rms", "grid_divergence_over_gradient",
           "read_divergence_over_gradient"]


def wavenumbers(n):
    """Integer wavenumbers of an axis, with the Nyquist mode set to zero."""
    k = sfft.fftfreq(n, 1.0 / n)
    k[n // 2] = 0.0
    return k


def spectral(grid):
    """(RMS of the velocity gradient, RMS of the spectral divergence) of a grid (3, n, n, n) indexed [z, y, x], on a
    periodic domain of length 2 pi."""
    n = grid.shape[1]
    k = wavenumbers(n)
    kz, ky, kx = k[:, None, None], k[None, :, None], k[None, None, :]
    k2 = kx * kx + ky * ky + kz * kz
    gradient_square, div = 0.0, np.zeros((n, n, n), dtype=np.complex128)
    for c, kc in enumerate((kx, ky, kz)):  # components u, v, w go with x, y, z
        hat = sfft.fftn(grid[c].astype(np.float64), workers=-1)
        gradient_square += float((k2 * (hat.real ** 2 + hat.imag ** 2)).sum())
        div += kc * hat
    total = float(n) ** 3
    return np.sqrt(gradient_square) / total, float(np.sqrt((div.real ** 2 + div.imag ** 2).sum())) / total


def trilinear_divergence(grid, points):
    """Analytic divergence of the trilinear interpolant at points (N, 3) of (x, y, z) in grid-index units: the point
    (i, j, k) is the centre of voxel x = i, y = j, z = k. Domain of length 2 pi: the voxel is 2 pi / n wide."""
    n = grid.shape[1]
    base = np.floor(points).astype(np.int64)
    f = points - base
    i0, j0, k0 = base[:, 0] % n, base[:, 1] % n, base[:, 2] % n
    i1, j1, k1 = (i0 + 1) % n, (j0 + 1) % n, (k0 + 1) % n
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    u, v, w = (grid[c].astype(np.float64) for c in range(3))

    def bilinear(a00, a10, a01, a11, s, t):
        return (1.0 - s) * (1.0 - t) * a00 + s * (1.0 - t) * a10 + (1.0 - s) * t * a01 + s * t * a11

    # du/dx: difference along x, bilinear in (y, z); arrays are indexed [z, y, x].
    dudx = bilinear(u[k0, j0, i1] - u[k0, j0, i0], u[k0, j1, i1] - u[k0, j1, i0], u[k1, j0, i1] - u[k1, j0, i0], u[k1, j1, i1] - u[k1, j1, i0], fy, fz)
    dvdy = bilinear(v[k0, j1, i0] - v[k0, j0, i0], v[k0, j1, i1] - v[k0, j0, i1], v[k1, j1, i0] - v[k1, j0, i0], v[k1, j1, i1] - v[k1, j0, i1], fx, fz)
    dwdz = bilinear(w[k1, j0, i0] - w[k0, j0, i0], w[k1, j0, i1] - w[k0, j0, i1], w[k1, j1, i0] - w[k0, j1, i0], w[k1, j1, i1] - w[k0, j1, i1], fx, fy)
    return (dudx + dvdy + dwdz) * (n / LENGTH)


def run():
    log = HERE / "divergence.csv"
    with open(log, "w", newline="", encoding="utf-8") as fo:
        w = csv.writer(fo)
        w.writerow(COLUMNS)
        for seed in SEEDS:
            sequence = dg.realisation(seed)
            header = cc.read_header(sequence)
            n, u_rms = header["n"], header["u_rms"]
            points = np.random.default_rng(POINT_SEED).random((POINTS, 3)) * n
            raw, _ = dg.decoded("raw", sequence)
            gradient, _ = spectral(raw)
            for chain, bound in dg.chains(BOUND):
                grid, record = dg.decoded(chain, sequence, bound)
                _, grid_div = spectral(grid)
                read_div = float(np.sqrt(np.mean(trilinear_divergence(grid, points) ** 2)))
                errors = cc.ErrorSum()
                errors.add(raw, grid)
                raw_bytes = record["frames_covered"] * 3 * n ** 3 * 4
                row = [sequence.name, seed, f"{cc.eta_in_voxels(header):.4f}", dg.FRAME, chain, "" if bound is None else f"{bound:g}",
                       record["exit_code"], f"{raw_bytes / record['compressed_bytes']:.3f}", f"{errors.relative_rms():.6e}",
                       f"{errors.largest / u_rms:.5e}", f"{gradient:.6e}", f"{grid_div / gradient:.4e}", f"{read_div / gradient:.4e}"]
                w.writerow(row)
                fo.flush()
                print("  ".join(str(x) for x in row), flush=True)
    summary()


def summary():
    rows = list(csv.DictReader(open(HERE / "divergence.csv", encoding="utf-8")))
    seeds = sorted({r["solver_seed"] for r in rows}, key=lambda s: SEEDS.index(int(s)))
    names = []
    for r in rows:
        if r["chain"] not in names:
            names.append(r["chain"])
    print(f"\nDivergence over the RMS velocity gradient, frame {rows[0]['frame']}: on the grid (M1) / as read by trilinear interpolation (M2)")
    print(f"{'chain':14s}" + "".join(f"  seed {s:>3s}: {'M1':>9s} {'M2':>9s}" for s in seeds))
    for c in names:
        cells = []
        for s in seeds:
            r = next(x for x in rows if x["chain"] == c and x["solver_seed"] == s)
            cells.append(f"            {float(r['grid_divergence_over_gradient']):9.2e} {float(r['read_divergence_over_gradient']):9.2e}")
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
