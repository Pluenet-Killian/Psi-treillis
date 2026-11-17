# SPDX-License-Identifier: Apache-2.0
"""One frame of a sequence, as each chain gives it back on the grid. Shared by benchmarks/divergence, benchmarks/spectra
and benchmarks/tracers, so that the three measure the same decoded grids.

Chains (the frame is frame FRAME of the sequence; the bound is a fraction of u', the RMS velocity of its header):

    raw          the original float32 frame
    sz3          SZ3, frame by frame, at the bound
    sz3_4d       SZ3 in 4D on the block of frames FIRST to FIRST + FRAMES - 1, at the bound; frame FRAME extracted
    sperr        SPERR, frame by frame, at the bound
    zfp          zfp, fixed-accuracy mode, at the bound
    cuszp        cuSZp, at the bound
    cuzfp_2bits  cuZFP on the GPU at a fixed rate of 2 bits per value: the grid that the tracer experiment decodes
                 (written by `tracers --path zfp --dump-grid`)
    psi          Psi-treillis of degree 3, default settings, group of frames FIRST to FIRST + FRAMES - 1, at the bound;
                 frame FRAME decoded to the grid by the library
    psi_free     the same without a bound

A decoded grid is cached in build/work/decoded/ together with the compressed size it stands for; delete that folder
to recompute everything.
"""

import json
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comparison_codecs as cc  # noqa: E402

FIRST, FRAMES, FRAME = 16, 32, 32
BOUNDED = ["sz3", "sz3_4d", "sperr", "zfp", "cuszp", "psi"]
CACHE = cc.BUILD / "work" / "decoded"


def realisation(seed):
    """Sequence of the resolution sweep at viscosity 0.0028 produced with this solver seed."""
    return cc.DATA / "sweep" / f"nu00280_seed{seed}.seq"


def decoded(chain, sequence, bound=None):
    """(grid float32 (3, n, n, n), record) for one chain. record: compressed_bytes of what was compressed, the number
    of frames it covers, and the exit code of the encoder where there is one."""
    sequence = Path(sequence)
    tag = f"{sequence.stem}_f{FRAME}_{chain}" + ("" if bound is None else f"_{bound:g}")
    grid_path, record_path = CACHE / f"{tag}.npy", CACHE / f"{tag}.json"
    if grid_path.is_file() and record_path.is_file():
        return np.load(grid_path), json.loads(record_path.read_text(encoding="utf-8"))
    CACHE.mkdir(parents=True, exist_ok=True)
    header = cc.read_header(sequence)
    n, u_rms = header["n"], header["u_rms"]
    folder = cc.work_folder("decoded")
    record = dict(chain=chain, sequence=sequence.name, frame=FRAME, bound_over_urms=bound, frames_covered=1, exit_code=0)
    if chain == "raw":
        grid = cc.read_frame(sequence, FRAME, n).copy()
        record["compressed_bytes"] = grid.nbytes
    elif chain in ("sz3", "sperr", "zfp", "cuszp"):
        size, grid = cc.compress_frame(chain, cc.read_frame(sequence, FRAME, n), bound * u_rms, folder)
        record["compressed_bytes"] = size
    elif chain == "sz3_4d":
        grid, total = np.empty((3, n, n, n), dtype=np.float32), 0
        for c in range(3):
            size, block = cc.sz3_4d_component(cc.read_component_block(sequence, FIRST, FRAMES, c, n), bound * u_rms, folder)
            grid[c] = block[FRAME - FIRST]
            total += size
            del block
            (folder / "block.dec").unlink()
        record.update(compressed_bytes=total, frames_covered=FRAMES)
    elif chain in ("psi", "psi_free"):
        psi = cc.psitreillis()
        path = folder / "group.psi"
        path.unlink(missing_ok=True)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            psi.compress_file(sequence, path, degree=3, abs_tolerance=None if chain == "psi_free" else bound * u_rms, u_rms=u_rms,
                              first_frame=FIRST, frame_count=FRAMES, seal=False)
        with psi.open(path) as f:
            grid = f.decode(FRAME - FIRST)
            if caught:
                record["exit_code"] = psi.BOUND_MISSED if (f.info.has_bound and not f.info.bound_held) else psi.CLIPPED
                record["warning"] = str(caught[-1].message)
        record.update(compressed_bytes=path.stat().st_size, frames_covered=FRAMES)
        path.unlink()
    elif chain == "cuzfp_2bits":
        dump = folder / "cuzfp.f32"
        command = [cc.BUILD / f"tracers{cc.EXE}", "--data", sequence, "--frame", FRAME, "--path", "zfp", "--zfp-rate", 2, "--dump-grid", dump]
        p = subprocess.run([str(x) for x in command], cwd=cc.ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        if p.returncode != 0 or not dump.is_file():
            raise RuntimeError(f"tracers --dump-grid: exit code {p.returncode}\n{p.stdout.decode('utf-8', 'replace')[-400:]}")
        grid = np.fromfile(dump, dtype=np.float32).reshape(3, n, n, n)
        dump.unlink()
        record["compressed_bytes"] = 3 * n ** 3 * 2 // 8  # fixed rate: 2 bits per value
    else:
        raise ValueError(chain)
    np.save(grid_path, grid)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    return grid, record


def chains(bound):
    """(chain, bound) pairs measured for one bound, in the order of the logs."""
    return [("raw", None)] + [(c, bound) for c in BOUNDED] + [("cuzfp_2bits", None), ("psi_free", None)]
