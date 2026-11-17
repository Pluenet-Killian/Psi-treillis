# SPDX-License-Identifier: Apache-2.0
"""The seal at 256^3: what `psi_stream pack` adds to a file when it seals it, and how the reader reads the file with
and without it.

    python benchmarks/robustness/seal_256.py

What is measured. Two encodings of data/turb256.seq, those of two reference files, are written twice by pack: sealed
(its default) and with --no-seal.

    z_t256_d2        --degree 2 --start 16 --frames 33
    f_t256_d3_t05    --degree 3 --start 16 --frames 57 --tolerance 0.05

    psi_stream pack --data data/turb256.seq --file F_sealed.psi OPTIONS
    psi_stream pack --data data/turb256.seq --file F_unsealed.psi OPTIONS --no-seal
    psi_stream play --file F.psi --particles 1000000 --sort          (each of the two files)

For each encoding the script records:
  1. the size of the two files and their difference (the seal is 48 bytes written after the frame table,
     docs/FORMAT.md);
  2. whether the sealed file, without its last 48 bytes, has the SHA-256 of the file written with --no-seal;
  3. for each of the two files read by play: the exit code, the number of frames whose state differs from that of the
     sender, the fingerprint of the final positions of one million sorted particles, and what the reader says of the
     seal ("present" or "absent"); and whether the two final positions are the same.

What is written.
    seal_256.csv                       one line per encoding, with the figures above
    logs/seal_256_<encoding>_<run>.log raw output of each run: pack_sealed, pack_unsealed, play_sealed, play_unsealed
The four files go to build/work/seal_256/.

Exit code: 0 if every run gave the lines the script reads (the answers themselves are in the log, whatever they are);
1 if a file is not written, if a read ends otherwise than by a complete read, or if a line is missing or has another
shape (the list is printed at the end); 2 if nothing was run.

The GPU may be shared: no time is read.

Needs build/psi_stream (tools/build.ps1 -Tools on Windows, CMake with -DPSIT_BUILD_TOOLS=ON elsewhere),
data/turb256.seq, an NVIDIA GPU. Python 3.8 or later, standard library only.
"""

import csv
import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "compression"))
import psi_stream_runs as ps  # noqa: E402

WORK = "build/work/seal_256"
SEQUENCE = "data/turb256.seq"
SEAL_BYTES = 48
PARTICLES = "1000000"
ENCODINGS = [
    ("z_t256_d2", ["--degree", "2", "--start", "16", "--frames", "33"]),
    ("f_t256_d3_t05", ["--degree", "3", "--start", "16", "--frames", "57", "--tolerance", "0.05"]),
]
COLUMNS = ["encoding", "pack_options", "sealed_pack_exit_code", "unsealed_pack_exit_code", "sealed_bytes", "unsealed_bytes", "extra_bytes",
           "unsealed_sha256", "sealed_sha256_without_last_48_bytes", "same_bytes_before_the_seal",
           "sealed_play_exit_code", "sealed_frames_differing", "sealed_final_positions", "sealed_seal_seen_by_reader",
           "unsealed_play_exit_code", "unsealed_frames_differing", "unsealed_final_positions", "unsealed_seal_seen_by_reader",
           "same_final_positions"]


def measure(name, options, logs):
    """One encoding: (row of COLUMNS, list of problems)."""
    row = dict.fromkeys(COLUMNS, "")
    row.update(encoding=name, pack_options=" ".join(options))
    problems = []
    files = {"sealed": f"{WORK}/{name}_sealed.psi", "unsealed": f"{WORK}/{name}_unsealed.psi"}
    for kind, file in files.items():
        (ps.ROOT / file).unlink(missing_ok=True)  # a file left by an earlier run must not stand for this one
        pack = ["pack", "--data", SEQUENCE, "--file", file] + options + (["--no-seal"] if kind == "unsealed" else [])
        row[f"{kind}_pack_exit_code"] = ps.run(pack, logs / f"seal_256_{name}_pack_{kind}.log").code
        if not (ps.ROOT / file).is_file():
            problems.append(f"{name}: {file} was not written, pack exit code {row[f'{kind}_pack_exit_code']} (logs/seal_256_{name}_pack_{kind}.log)")
    if problems:
        return row, problems

    sealed, unsealed = (ps.ROOT / files["sealed"]).read_bytes(), (ps.ROOT / files["unsealed"]).read_bytes()
    row.update(sealed_bytes=len(sealed), unsealed_bytes=len(unsealed), extra_bytes=len(sealed) - len(unsealed),
               unsealed_sha256=hashlib.sha256(unsealed).hexdigest(),
               sealed_sha256_without_last_48_bytes=hashlib.sha256(sealed[:len(sealed) - SEAL_BYTES]).hexdigest())
    row["same_bytes_before_the_seal"] = "yes" if row["unsealed_sha256"] == row["sealed_sha256_without_last_48_bytes"] else "no"
    del sealed, unsealed

    for kind, file in files.items():
        log = logs / f"seal_256_{name}_play_{kind}.log"
        r = ps.run(["play", "--file", file, "--particles", PARTICLES, "--sort"], log)
        row[f"{kind}_play_exit_code"] = r.code
        try:
            differing, positions = ps.integrity(r.output)
            row.update({f"{kind}_frames_differing": differing, f"{kind}_final_positions": positions, f"{kind}_seal_seen_by_reader": ps.seal(r.output)})
        except ps.UnexpectedOutput as e:
            message = ps.failure(r.error)
            problems.append(f"{name}, play of the {kind} file: exit code {r.code}; {e}" + (f"; {message}" if message else "") + f" (logs/{log.name})")
    if row["sealed_final_positions"] and row["unsealed_final_positions"]:
        row["same_final_positions"] = "yes" if row["sealed_final_positions"] == row["unsealed_final_positions"] else "no"
    return row, problems


def main():
    if len(sys.argv) > 1:
        print(__doc__)
        return 2
    ps.require([ps.PSI_STREAM, SEQUENCE], "Build the instruments (tools/build.ps1 -Tools) and produce the sequences of data/ first.")
    (ps.ROOT / WORK).mkdir(parents=True, exist_ok=True)
    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    problems = []
    with open(HERE / "seal_256.csv", "w", newline="", encoding="utf-8") as fo:
        writer = csv.DictWriter(fo, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        for name, options in ENCODINGS:
            row, found = measure(name, options, logs)
            writer.writerow(row)
            fo.flush()
            problems += found
            print(f"{name}: pack {' '.join(options)}")
            print(f"   sealed {row['sealed_bytes']} bytes (pack exit code {row['sealed_pack_exit_code']}), without a seal {row['unsealed_bytes']} bytes "
                  f"(pack exit code {row['unsealed_pack_exit_code']}): {row['extra_bytes']} more; same bytes before the seal: {row['same_bytes_before_the_seal']}")
            for kind in ("sealed", "unsealed"):
                print(f"   {kind} file read: exit code {row[f'{kind}_play_exit_code']}, {row[f'{kind}_frames_differing']} frame(s) differing, "
                      f"final positions {row[f'{kind}_final_positions']}, seal {row[f'{kind}_seal_seen_by_reader']}")
            print(f"   same final positions: {row['same_final_positions']}", flush=True)
    print(f"Log: {ps.relative(HERE / 'seal_256.csv')}; raw outputs in {ps.relative(logs)}/")
    if problems:
        print(f"{len(problems)} PROBLEM(S): the log is incomplete for these runs:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
