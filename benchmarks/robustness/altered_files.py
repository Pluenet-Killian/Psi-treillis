# SPDX-License-Identifier: Apache-2.0
"""The reader faced with altered files: outcome of `psi_stream play` on each altered file of the test suite (claim ROBUST-1).

    python tests/test_codec.py --keep-alterations --only "guard.geometry_no_seal" "guard.random_alteration"
    python benchmarks/robustness/altered_files.py [--cases build/tests/cases] [--simultaneous 6] [--timeout 60]

What is measured. The first command leaves, under build/tests/cases/, the altered files at 64^3 that these two cases of
the suite write; the second reads each of them back and records how the read ends. Two batches:

    chosen    build/tests/cases/guard_geo_*.psi: chosen alterations of the header, of the frame table and of the frames
              of files without a seal (case guard.geometry_no_seal);
    random    build/tests/cases/random/*.psi: one bit flipped or one byte replaced at a drawn offset, with a fixed seed,
              in four files of which two are sealed (case guard.random_alteration; kept only with --keep-alterations).

Each file is read by

    psi_stream play --file F.psi --particles 1000 --sort

several reads at a time, each with a watchdog timeout. Outcomes (benchmarks/compression/psi_stream_runs.py, outcome):

    refused       exit code 1 and a message "Failure:";
    counted       complete read, at least one frame whose state differs (exit code 1);
    read          complete read, no differing frame (exit code 0): alteration without effect, or not seen;
    CUDA error    a message "CUDA error";
    crash         exit code other than 0 and 1;
    hang          read stopped after the watchdog timeout.

What is written.
    altered_files.csv              one line per file: batch, file, exit code, outcome, message of the reader if any
    logs/altered_files_<batch>.log the raw output of every read of the batch, each under a line "== file: exit code, outcome"

Exit code: 0 if every read has one of the six outcomes above (a CUDA error, a crash or a hang is a result: it is
recorded and shown, and tools/claims.py judges it); 1 if a read ends in none of them (an exit code 0 or 1 without the
line that goes with it: the outcome then reads "code N without an expected line"); 2 if nothing was read (a batch
without files, the reader missing).

The GPU may be shared: no time is read.

Needs build/psi_stream and the altered files (first command above), an NVIDIA GPU. Python 3.8 or later, standard
library only.
"""

import argparse
import collections
import csv
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "compression"))
import psi_stream_runs as ps  # noqa: E402

BATCHES = [("chosen", "guard_geo_*.psi"), ("random", "random/*.psi")]  # (batch, pattern under the folder of the cases)
SUITE = 'python tests/test_codec.py --keep-alterations --only "guard.geometry_no_seal" "guard.random_alteration"'
OUTCOMES = ("refused", "counted", "read", "CUDA error", "crash", "hang")


def read_all(files, simultaneous, timeout):
    """Reads every file with play, `simultaneous` at a time; returns file -> finished run."""
    waiting, running, runs = list(files), [], {}
    while waiting or running:
        while waiting and len(running) < simultaneous:
            file = waiting.pop(0)
            running.append((file, ps.Run(["play", "--file", file, "--particles", "1000", "--sort"])))
        time.sleep(0.01)
        for item in list(running):
            file, run = item
            if run.finished(timeout):
                runs[file] = run.wait()
                running.remove(item)
    return runs


def main():
    parser = argparse.ArgumentParser(description="Outcome of psi_stream play on each altered file of the test suite.")
    parser.add_argument("--cases", default="build/tests/cases", help="folder of the cases of the test suite, from the root (default: build/tests/cases)")
    parser.add_argument("--simultaneous", type=int, default=6, help="reads at a time (default: 6)")
    parser.add_argument("--timeout", type=int, default=60, help="watchdog timeout of one read, in seconds (default: 60)")
    arguments = parser.parse_args()
    if arguments.simultaneous < 1 or arguments.timeout < 1:
        parser.error("--simultaneous and --timeout: at least 1")

    ps.require([ps.PSI_STREAM], "Build the instruments first (tools/build.ps1 -Tools).")
    cases = ps.ROOT / arguments.cases
    batches = [(batch, sorted(ps.relative(f) for f in cases.glob(pattern))) for batch, pattern in BATCHES]
    empty = [f"{batch} ({ps.relative(cases)}/{pattern})" for (batch, files), (_, pattern) in zip(batches, BATCHES) if not files]
    if empty:
        ps.stop(f"No altered file in the batch {' nor in the batch '.join(empty)}. Run the test suite first, keeping its altered files: {SUITE}. "
                "Nothing was read.")

    logs = HERE / "logs"
    logs.mkdir(exist_ok=True)
    unexpected = []
    with open(HERE / "altered_files.csv", "w", newline="", encoding="utf-8") as fo:
        writer = csv.writer(fo, lineterminator="\n")
        writer.writerow(["batch", "file", "exit_code", "outcome", "message"])
        for batch, files in batches:
            runs = read_all(files, arguments.simultaneous, arguments.timeout)
            counts = collections.Counter()
            with open(logs / f"altered_files_{batch}.log", "w", encoding="utf-8", newline="\n") as raw:
                for file in files:
                    run = runs[file]
                    outcome = ps.outcome(run)
                    counts[outcome] += 1
                    if outcome not in OUTCOMES:
                        unexpected.append(f"{file}: {outcome}")
                    writer.writerow([batch, file, run.code, outcome, ps.failure(run.error)])
                    raw.write(f"== {file}: exit code {run.code}, {outcome}\n{run.text()}")
            print(f"{batch}: {len(files)} files: " + ", ".join(f"{counts[o]} {o}" for o in sorted(counts)), flush=True)
    print(f"Log: {ps.relative(HERE / 'altered_files.csv')}; raw outputs in {ps.relative(logs)}/")
    if unexpected:
        print(f"{len(unexpected)} READ(S) WITHOUT A KNOWN OUTCOME (the line that goes with the exit code is missing from the output):", file=sys.stderr)
        for line in unexpected:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    sys.exit(main())
