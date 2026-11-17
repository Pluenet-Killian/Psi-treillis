# SPDX-License-Identifier: Apache-2.0
"""Cutout of the public JHTDB dataset: the protocol of cutout.py applied to the frames downloaded by download_jhtdb.py
(claims JHTDB-3, JHTDB-4 and REPR-1).

Status: prepared, never run on data. Access to the database was obtained, but the transfers of the cutout broke
repeatedly and no complete frame was retrieved (see download_jhtdb.py and README.md): this measurement is not done.

What it measures. The 256^3 cutout of the dataset `isotropic1024coarse`, on 8 consecutive stored instants, windowed and
projected as in the control, then compressed by `psi_stream pack` (degree 3, lattice steps of 2 voxels and of 1 voxel,
without a bound and with a bound of 5 % of u') and by SPERR at the same absolute bound; the error is measured on the
central cube against the windowed field. u' is not that of a header: it is measured on the raw cutout.

How to run it: `python benchmarks/cutout/cutout.py jhtdb`, once download_jhtdb.py has written the 8 frames.
What it needs: what cutout.py needs, and the frames data/cutout/jhtdb/t<instant>.f32.
What it writes: jhtdb.csv in this folder (same columns as control.csv), and the output of each run of `psi_stream pack`
in logs/.
"""

import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
FOLDER = ROOT / "data" / "cutout" / "jhtdb"     # where download_jhtdb.py writes the frames

# Interval between two stored instants of `isotropic1024coarse`: 5,028 instants for t from 0 to 10.056, according to the
# page of the dataset (https://turbulence.idies.jhu.edu/datasets/homogeneousTurbulence/isotropic), that is 0.002
# (computed).
DT = 10.056 / 5028
INSTANTS = tuple(range(1001, 1009))


def run(c):
    """c: the module cutout (window, projection, measurement)."""
    paths = [FOLDER / f"t{t}.f32" for t in INSTANTS]
    missing = [p.name for p in paths if not p.is_file() or p.stat().st_size != 12 * c.SIDE ** 3]
    if missing:
        print("frames missing:", ", ".join(missing))
        print("download them first: benchmarks/cutout/download_jhtdb.py, run with the Python of the environment where the client of the "
              "database is installed (token in JHTDB_TOKEN)")
        return 2
    log = Path(c.HERE) / "jhtdb.csv"
    if log.is_file():
        log.unlink()
    raw = [np.fromfile(p, dtype=np.float32).reshape(3, c.SIDE, c.SIDE, c.SIDE) for p in paths]
    # u': the mean, over the frames, of the RMS of the three components about their mean, measured on the raw cutout.
    u_rms = float(np.mean([math.sqrt(sum(float(b[i].astype(np.float64).var()) for i in range(3)) / 3.0) for b in raw]))
    base = dict(source="jhtdb_isotropic1024coarse", eta_voxels="", u_rms=f"{u_rms:.6f}")
    print(f"JHTDB isotropic1024coarse, cutout of {c.SIDE}^3, instants {INSTANTS[0]} to {INSTANTS[-1]}: u' = {u_rms:.4f} (measured on the cutout)", flush=True)
    w = c.window()
    windowed = [c.project(b * w) for b in raw]
    c.measure_cutout(log, base, windowed, u_rms, DT, "jhtdb_cutout")
    return 0
