# SPDX-License-Identifier: Apache-2.0
"""tracers.py - follow particles through a compressed sequence, without decoding a single frame to the grid.

    python examples/tracers.py flow.psi [--particles 100000] [--seed 1]

Particles are drawn uniformly in the periodic box and advected from the first frame of the file to the last. The
velocity is sampled at their positions straight from the compressed file (Reader.sample): the grid of velocities is
never rebuilt. Between two frames the field is held at the earlier frame, and the positions advance by one classical
fourth-order Runge-Kutta step per frame; this is the simplest scheme that shows the use of the interface, not an
accurate integrator (the measurement program tools/tracers.cu and the visual comparator interpolate in time).

The file is assumed to hold a flow in a periodic box of side 2 pi sampled on n^3 voxels, as the sequences of this
repository are: a velocity u moves a particle by u * dt * n / (2 pi) voxels during the time dt between two frames.

The script prints how far the particles travelled, and the dispersion index of their count per bin of 4 voxels at the
end (variance / mean of the count; 1 for particles that stayed uniform).

Needs NumPy, an NVIDIA GPU, and the library built in build/ (tools/build.ps1, or CMake).
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import psitreillis as psi  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Follow particles through a .psi file.")
    parser.add_argument("file")
    parser.add_argument("--particles", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    with psi.open(args.file) as flow:
        n, frames, dt = flow.n, flow.frames, flow.info.dt
        voxels_per_unit = n / (2.0 * math.pi)        # the box has a side of 2 pi
        step = dt * voxels_per_unit                  # a velocity of 1 moves a particle by this many voxels per frame
        start = np.random.default_rng(args.seed).random((args.particles, 3)) * n
        x = start.copy()                             # (x, y, z) in grid-index units; not wrapped, to measure the travel

        def velocity(frame, positions):
            return flow.sample(frame, np.mod(positions, n)).astype(np.float64)

        for frame in range(frames - 1):
            k1 = velocity(frame, x)
            k2 = velocity(frame, x + 0.5 * step * k1)
            k3 = velocity(frame, x + 0.5 * step * k2)
            k4 = velocity(frame, x + step * k3)
            x += step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    travel = np.sqrt(((x - start) ** 2).sum(axis=1))
    side = 4
    bins = n // side
    index = (np.mod(x, n) // side).astype(np.int64)
    count = np.bincount((index[:, 2] * bins + index[:, 1]) * bins + index[:, 0], minlength=bins ** 3)
    print(f"{args.file}: {n}^3, {frames} frames, time between frames {dt:g}")
    print(f"{args.particles} particles followed over {frames - 1} frame intervals")
    print(f"distance travelled: mean {travel.mean():.3f} voxels, largest {travel.max():.3f} voxels")
    print(f"dispersion index of the count per bin of {side} voxels: {count.var() / count.mean():.4f} "
          f"(mean count {count.mean():.2f}; 1 for a uniform draw)")


if __name__ == "__main__":
    main()
