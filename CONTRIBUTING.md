# Contributing

Thank you for looking at Psi-treillis. This file says how to build the project, what to run before proposing a change,
how to report a problem, and the few rules that keep its results verifiable.

## Build

Requirements: an NVIDIA GPU, CUDA 13 with `nvcc`, CMake 3.24 or later, Ninja, and a C++17 compiler. The project is
developed and tested on Windows with Visual Studio 2022. It also builds on Ubuntu 24.04 with GCC 13 (checked at every
commit on a machine without a GPU); it has not been run there, apart from the CPU reference reader.

```powershell
.\tools\build.ps1                # library, psit, example, in build\
.\tools\build.ps1 -Tools         # also the measurement instruments of tools\
```

or, with CMake directly:

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DPSIT_BUILD_TOOLS=ON
cmake --build build
```

The tracer experiment and the visual comparator need zfp: `git submodule update --init third_party/zfp`.

## Before proposing a change

Run the four checks. They are deterministic: none of them reads a timing.

```powershell
python tests\test_api.py         # public interface: library, Python module, psit (64^3, about a minute)
python tests\test_codec.py       # codec chain pack / play: fingerprints of the files written, bound, guards
python tools\claims.py           # claims sheet: every verdict is still the one recorded, no GPU needed
python tools\check_english.py    # the repository is in English
```

## Reporting a problem, asking a question

Open an issue on the repository (https://github.com/Pluenet-Killian/Psi-treillis/issues). For a problem, give the
command, its full output, the GPU, the version of CUDA and of the host compiler, and the output of
`python tools/environment.py --print --no-data`. Questions about the method, the measurements or a result that you
could not reproduce are welcome in the same place: a measurement that cannot be reproduced is a problem of the
project.

## Rules

- **Files are reproducible.** With the same options the encoder writes the same bytes, and `tests/fingerprints.json`
  holds the fingerprints of reference files. A change that alters the bytes written is marked **[bytes]** in
  `CHANGELOG.md`, with the option that restores the previous files if there is one.
- **The format is specified.** `docs/FORMAT.md` describes the `.psi` file. A change of format changes that document in
  the same commit.
- **Claims are recorded with their verdict.** `docs/CLAIMS.md` lists what the project claims, the criterion, the
  measurement and the verdict, including the claims that do not hold. A new claim is added to the registry in
  `tools/claims.py`; its criterion is committed before the measurement is made. A claim that fails stays in the sheet.
- **Measurements come with their script.** A number written in a document is in a log of `benchmarks/`, next to the
  script that produces it and a notice that says how to replay it. The numbers of the article and of `README.md` are
  not typed: `python tools/paper_numbers.py` reads them from the logs (`README.md` is generated from
  `tools/README.template.md`).
- **Everything is in English**: documents, comments, messages of the programs, identifiers, names of files and
  columns. `tools/check_english.py` checks it.
- **A log names no folder of the machine.** Scripts run the programs from the root of the repository with relative
  paths.
