# Linux build (claim LINUX-1)

## What is recorded, and why

The project is developed and measured on Windows, with MSVC as the host compiler of `nvcc`. A GitHub Actions workflow,
`.github/workflows/linux.yml`, compiles it on Ubuntu with GCC and `nvcc` at every commit of `main`, on a machine
without a GPU: the library, `psit`, the C example, `psi_stream`, `dns` and the CPU reference reader `psi_ref`. It then
starts `psit` and `psi_ref` without a file, replays the claims sheet (`tools/claims.py`) and checks the language of the
repository (`tools/check_english.py`).

`record.py` stores the outcome of the last completed run of this workflow, so that the claim "the project compiles
under Linux" rests on a file of the repository and not on the state of a web page.

**This proves that the project builds on Linux, not that it runs there.** The workflow machine has no GPU: no kernel
is executed, no file is encoded or decoded.

## How to replay

After a commit on `main` has been pushed and its run has completed:

```
python benchmarks/linux_build/record.py
```

It needs the GitHub command-line tool `gh`, signed in to an account that can read the repository
(`Pluenet-Killian/Psi-treillis`; `--repo OWNER/NAME` for a fork). Nothing is compiled or measured on the local
machine. If the workflow is absent from the repository, or has no completed run on `main`, the script says so, writes
nothing and exits with code 2.

To rebuild on a Linux machine of your own, follow the commands of the workflow file: the packages installed, the CMake
configuration (`-DPSIT_BUILD_TOOLS=ON -DPSIT_CUDA_ARCHITECTURES=86`, the architecture being given because the machine
has no GPU to detect) and the targets built.

## What is written

| File | Content |
|---|---|
| `workflow_run.json` | `run` (identifier), `workflow`, `event`, `commit`, `created`, `finished`, `conclusion`, `url`; `steps`: the name and the conclusion of each step; `compilers`: the lines "compiler identification" of CMake; `packages`: the CUDA packages installed and their versions; `linked_targets`: the targets linked by Ninja; `claims_sheet`: the last line of `tools/claims.py` on the workflow machine; `english_check`: the last line of `tools/check_english.py` |
| `logs/workflow_run.log` | The log of the run, as `gh run view --log` returns it; the fields above from `compilers` on are read in it |

## Caveats

- **Compilation only.** Whether the encoder writes the same bytes on Linux as on Windows, and whether the tests give
  the same verdicts there, is not established by this record: it needs a Linux machine with a GPU (claim PORT-1).
- **What is started is not a test of the codec.** `psit --help`, `psit version` and `psi_ref` without arguments show
  that the binaries load and print their usage, nothing more.
- **The tracer experiment and the visual comparator are not built.** `tracers` needs `third_party/zfp`, which the
  workflow does not check out; `psi_view` is a Windows program.
- **One distribution, one compiler pair, one architecture number**: those of the workflow file.
- **The claims sheet replayed on the workflow machine reads the logs kept in the repository**; it measures nothing
  there.
- **The record describes one commit.** The commit that stores `workflow_run.json` triggers the workflow again; its
  current state is on the Actions page of the repository, or in a fresh run of `record.py`.
- **The fields read in the log depend on the wording of CMake, apt and Ninja.** If one of them comes back empty,
  `record.py` warns; read `logs/workflow_run.log` before concluding that something was not built.
