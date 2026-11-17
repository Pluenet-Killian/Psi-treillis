# SPDX-License-Identifier: Apache-2.0
"""Records the last completed run of the workflow "Linux build" on the branch main (claim LINUX-1).

    python benchmarks/linux_build/record.py                      the published repository
    python benchmarks/linux_build/record.py --repo OWNER/NAME    another repository on GitHub (a fork)

The workflow (.github/workflows/linux.yml) compiles the project on an Ubuntu machine without a GPU at every commit of
main, then replays the claims sheet and the check of the language. This script reads the result of its last completed
run with the GitHub command-line tool and writes it next to itself. It measures nothing on this machine, and what it
records is a proof of compilation: nothing runs on a GPU in the workflow.

Needs: the tool `gh` (https://cli.github.com), signed in to an account that can read the repository. Python 3.8 or
later, standard library only.

Writes:
    workflow_run.json       identifier of the run, commit, conclusion of the run and of each step, versions of the
                            compilers and of the CUDA packages read in the log, targets linked, last line of
                            tools/claims.py and of tools/check_english.py
    logs/workflow_run.log   the log of the run, as `gh run view --log` gives it

The record describes one commit. The commit that stores it in the repository triggers the workflow again: the state of
the day is read on the Actions page of the repository, or by running this script again.

Exit code: 0 if a run was recorded (whatever its conclusion), 2 if there is nothing to record (no `gh`, workflow absent
from the repository, no completed run on main).
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPOSITORY = "Pluenet-Killian/Psi-treillis"
WORKFLOW = "linux.yml"


class NothingToRecord(Exception):
    pass


def gh(repository, *arguments):
    """Standard output of a read-only `gh` command on the repository."""
    try:
        done = subprocess.run(["gh", *arguments, "--repo", repository], capture_output=True)
    except FileNotFoundError:
        raise NothingToRecord("the tool `gh` is not installed, or not on the path (https://cli.github.com)") from None
    if done.returncode != 0:
        message = done.stderr.decode("utf-8", "replace").strip()[:300]
        if "404" in message and "workflow" in message:
            raise NothingToRecord(f"the workflow {WORKFLOW} is not on the default branch of {repository}: there is no run to read. "
                                  "Nothing is recorded; push the repository with .github/workflows/linux.yml, wait for the run "
                                  f"to complete, then run this script again. [gh: {message}]")
        raise NothingToRecord("gh " + " ".join(arguments[:3]) + ": " + message)
    return done.stdout.decode("utf-8", "replace")


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", default=REPOSITORY, help="repository on GitHub, OWNER/NAME (default: %(default)s)")
    repository = parser.parse_args().repo

    runs = json.loads(gh(repository, "run", "list", "--workflow", WORKFLOW, "--branch", "main", "--status", "completed", "--limit", "1",
                         "--json", "databaseId"))
    if not runs:
        raise NothingToRecord(f"no completed run of the workflow {WORKFLOW} on the branch main of {repository}: nothing is recorded")
    run_id = str(runs[0]["databaseId"])
    view = json.loads(gh(repository, "run", "view", run_id, "--json", "databaseId,headSha,conclusion,createdAt,updatedAt,workflowName,event,url,jobs"))
    log = gh(repository, "run", "view", run_id, "--log")
    # A line of the log: job, step, then the time and the text.
    lines = [re.sub(r"^\S+Z ", "", line.split("\t", 2)[-1]) for line in log.splitlines()]

    def find(pattern):
        return [m.group(1) for line in lines for m in [re.search(pattern, line)] if m]

    record = {
        "run": view["databaseId"],
        "workflow": view["workflowName"],
        "event": view["event"],
        "commit": view["headSha"],
        "created": view["createdAt"],
        "finished": view["updatedAt"],
        "conclusion": view["conclusion"],
        "url": view["url"],
        "steps": [{"name": step["name"], "conclusion": step["conclusion"]} for job in view["jobs"] for step in job["steps"]],
        # printed by CMake at the configuration
        "compilers": find(r"-- (The \w+ compiler identification is .+)$"),
        # printed by apt at the installation of the CUDA compiler and libraries
        "packages": find(r"Setting up ((?:cuda-nvcc|cuda-cudart-dev|libcufft-dev|cuda-cccl)\S* \([^)]+\))"),
        # printed by Ninja
        "linked_targets": find(r"^\[\d+/\d+\] Linking \w+ (?:executable|shared library) (\S+)"),
        # last line of tools/claims.py
        "claims_sheet": (find(r"^(\d+ claims, \d+ anomal\S+)") or [""])[-1],
        # last line of tools/check_english.py
        "english_check": (find(r"^(no French found|\d+ line\(s\) in \d+ file\(s\))$") or [""])[-1],
    }
    for field in ("compilers", "packages", "linked_targets", "claims_sheet", "english_check"):
        if not record[field]:
            print(f"record.py: warning: nothing in the log of the run for \"{field}\"", file=sys.stderr)
    (HERE / "logs").mkdir(exist_ok=True)
    with open(HERE / "logs" / "workflow_run.log", "w", newline="\n", encoding="utf-8") as f:
        f.write("\n".join(log.splitlines()) + "\n")
    with open(HERE / "workflow_run.json", "w", newline="\n", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(f"run {run_id} on {record['commit'][:7]}: {record['conclusion']}; " + ", ".join(record["linked_targets"]))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except NothingToRecord as nothing:
        print(f"record.py: {nothing}", file=sys.stderr)
        sys.exit(2)
