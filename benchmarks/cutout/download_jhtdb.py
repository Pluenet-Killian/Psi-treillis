# SPDX-License-Identifier: Apache-2.0
"""Downloads a 256^3 cutout of the public JHTDB dataset `isotropic1024coarse`, on 8 consecutive stored instants.

Status: prepared, never run to its end. Access to the database was obtained, but the transfers broke repeatedly and no
complete frame was retrieved: the frames that cutout_jhtdb.py needs do not exist, and the trial on JHTDB is not done.

How to run it. With the Python of an environment where the official client of the database is installed
(`pip install givernylocal`):

    python benchmarks/cutout/download_jhtdb.py

What it needs. NumPy, the client `givernylocal`, a network link to the database, and a personal access token of the
database. The token is read from the environment variable JHTDB_TOKEN and from nowhere else. It is written nowhere:
neither in a file nor in the output of this script (the client of the database puts it in the address of each request;
any error message is therefore redacted before it is printed).

Usage limits of the database, respected here: one request at a time; small, targeted cutouts. The client receives each
cutout as JSON text, about 21 bytes per value: a block of 128^3 weighs 135 MB on the network, for 25 MB once decoded.
Observed on 16 November 2026: of four blocks of 128^3 requested, a single one arrived whole, the three other
connections broke after 25 MB, 6 MB and 2 MB; one block of 64^3 (17 MB of text) arrived in 78 s.

Each frame is therefore requested in 64 blocks of 64^3, one after the other, each with four attempts spaced out, and a
pause separates two requests. That makes 512 requests for 1.6 GB of decoded data, and of the order of eleven hours at
the pace observed (computed). BLOCK can be set back to 128 if the link allows it: a block of 128^3 that fails twice is
then requested again as eight blocks of 64^3.

The script resumes where it stopped: a block received is written to disk at once, a complete frame is not requested
again. It can be stopped at any time.

What it writes, not versioned: data/cutout/jhtdb/t<instant>.f32, a field (3, 256, 256, 256) of 32-bit floats,
components x, y, z of the velocity, indexed [component, z, y, x]; and, while a frame is incomplete, its blocks in
data/cutout/jhtdb/blocks/.
"""

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUT = ROOT / "data" / "cutout" / "jhtdb"
BLOCKS = OUTPUT / "blocks"
DATASET = "isotropic1024coarse"
SIDE, BLOCK, SMALL = 256, 64, 64
ORIGIN = (384, 384, 384)             # corner (x, y, z) of the cutout in the box of 1,024^3, counted from 0
INSTANTS = tuple(range(1001, 1009))  # eight consecutive stored instants, in the middle of the database (indices from 1 to 5,028)
PAUSE = 2.0                          # seconds between two requests
WAITS = {BLOCK: (15.0,), SMALL: (15.0, 60.0, 180.0)}     # seconds before each new attempt, by size of the block


class Database:
    def __init__(self, token):
        from givernylocal.turbulence_dataset import turb_dataset
        from givernylocal.turbulence_toolkit import getCutout

        self.token, self.request = token, getCutout
        self.cube = turb_dataset(dataset_title=DATASET, output_path=str(OUTPUT), auth_token=token)
        self.requests = 0

    def block(self, t, bx, by, bz, side):
        """The block (3, side, side, side) of corner (bx, by, bz) in the cutout, or None if it could not be obtained."""
        path = BLOCKS / f"t{t}_c{side}_{bz:03d}_{by:03d}_{bx:03d}.npy"
        if path.is_file():
            return np.load(path)
        x0, y0, z0 = ORIGIN
        # The database counts its indices from 1, bounds included.
        axes = np.array([[x0 + bx + 1, x0 + bx + side], [y0 + by + 1, y0 + by + side], [z0 + bz + 1, z0 + bz + side], [t, t]])
        waits = WAITS[side]
        for attempt in range(len(waits) + 1):
            start = time.perf_counter()
            self.requests += 1
            try:
                result = self.request(self.cube, "velocity", axes, np.array([1, 1, 1, 1]), verbose=False)
                raw = np.asarray(result[list(result.data_vars)[0]])           # (z, y, x, component)
            except Exception as e:  # noqa: BLE001 - the original message may contain the address of the request, hence the token
                message = f"{type(e).__name__}: {str(e).replace(self.token, '***')[:160]}"
                print(f"   instant {t}, block of {side}^3 at ({bx}, {by}, {bz}): attempt {attempt + 1} failed after "
                      f"{time.perf_counter() - start:.0f} s ({message})", flush=True)
                if attempt < len(waits):
                    time.sleep(waits[attempt])
                continue
            if raw.shape != (side, side, side, 3):
                print(f"   instant {t}: unexpected shape {raw.shape}", flush=True)
                return None
            block = np.ascontiguousarray(np.moveaxis(raw, -1, 0), dtype=np.float32)
            partial = path.with_suffix(".partial")
            with open(partial, "wb") as f:
                np.save(f, block)
            partial.replace(path)
            print(f"   instant {t}, block of {side}^3 at ({bx}, {by}, {bz}): received in {time.perf_counter() - start:.0f} s", flush=True)
            time.sleep(PAUSE)
            return block
        if side == SMALL:
            return None
        # The large block does not get through: request it again as eight small ones.
        block = np.empty((3, side, side, side), dtype=np.float32)
        for dz in range(0, side, SMALL):
            for dy in range(0, side, SMALL):
                for dx in range(0, side, SMALL):
                    small = self.block(t, bx + dx, by + dy, bz + dz, SMALL)
                    if small is None:
                        return None
                    block[:, dz:dz + SMALL, dy:dy + SMALL, dx:dx + SMALL] = small
        with open(path.with_suffix(".partial"), "wb") as f:
            np.save(f, block)
        path.with_suffix(".partial").replace(path)
        return block


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    token = os.environ.get("JHTDB_TOKEN", "").strip()
    if not token:
        print("JHTDB_TOKEN is not set: nothing is requested from the database.")
        return 2
    warnings.filterwarnings("ignore")
    BLOCKS.mkdir(parents=True, exist_ok=True)
    database = Database(token)
    for t in INSTANTS:
        target = OUTPUT / f"t{t}.f32"
        if target.is_file() and target.stat().st_size == 12 * SIDE ** 3:
            print(f"instant {t}: already on disk", flush=True)
            continue
        field = np.empty((3, SIDE, SIDE, SIDE), dtype=np.float32)
        start = time.perf_counter()
        for bz in range(0, SIDE, BLOCK):
            for by in range(0, SIDE, BLOCK):
                for bx in range(0, SIDE, BLOCK):
                    block = database.block(t, bx, by, bz, BLOCK)
                    if block is None:
                        print(f"instant {t}: given up, a block could not be obtained; running the script again will resume here", flush=True)
                        return 1
                    field[:, bz:bz + BLOCK, by:by + BLOCK, bx:bx + BLOCK] = block
        partial = target.with_suffix(".partial")
        field.tofile(partial)
        partial.replace(target)
        for path in BLOCKS.glob(f"t{t}_*.npy"):
            path.unlink()
        print(f"instant {t}: written in {(time.perf_counter() - start) / 60:.0f} min, RMS of the components "
              f"{[round(float(np.sqrt((field[c].astype(np.float64) ** 2).mean())), 4) for c in range(3)]}", flush=True)
    print(f"cutout complete: {len(INSTANTS)} instants, {database.requests} requests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
