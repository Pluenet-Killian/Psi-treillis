# SPDX-License-Identifier: Apache-2.0
"""roundtrip.py - compress a synthetic velocity field from NumPy, read it back, sample it at arbitrary points.

    python examples/roundtrip.py

The field is the curl of a potential made of a few Fourier modes: periodic and divergence-free, which is what the
codec expects. From a source checkout the module is found in python/ and the library in build/ (tools/build.ps1).
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import psitreillis as psi  # noqa: E402

n, frames, dt = 64, 8, 0.05

# Voxel centres, in radians: the arrays are indexed [z, y, x].
centre = 2 * np.pi * (np.arange(n) + 0.5) / n
z, y, x = np.meshgrid(centre, centre, centre, indexing="ij")
field = np.empty((frames, 3, n, n, n), dtype=np.float32)
for f in range(frames):
    t = dt * f
    field[f, 0] = -np.cos(z + t) + 0.9 * np.cos(3 * y - t)   # u
    field[f, 1] = -np.cos(x + t) + 0.9 * np.cos(3 * z - t)   # v
    field[f, 2] = -np.cos(y + t) + 0.9 * np.cos(3 * x - t)   # w

# Compress with a certified bound of 5 % of u_rms on the largest error of a component.
info = psi.compress(field, "roundtrip.psi", dt=dt, tolerance=0.05)
print(f"{info.n}^3, {info.frames} frames, {info.ratio:.1f} times smaller than raw; bound certified: {info.bound_held}")

with psi.open("roundtrip.psi") as f:
    # Decode a frame to the grid and measure the largest error ourselves.
    decoded = f.decode(frames - 1)
    worst = np.abs(decoded - field[-1]).max() / f.info.u_rms
    print(f"largest error on the last frame: {100 * worst:.4f} % of u_rms (bound: {100 * f.info.tolerance:.1f} %)")

    # Sample the velocity anywhere, straight from the compressed file: here along a line through the box.
    s = np.linspace(0, n, 5, endpoint=False, dtype=np.float32)
    points = np.stack([s, 0.5 * s + 3.25, np.full_like(s, 10.5)], axis=1)   # (x, y, z) in grid-index units
    for point, velocity in zip(points, f.sample(frames - 1, points)):
        print(f"velocity at {point}: {velocity}")
