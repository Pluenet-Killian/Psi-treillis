# The article

`paper.pdf` is the article; `paper.tex` is its source.

| File | What it is | Written by |
|---|---|---|
| `paper.tex` | The text | hand |
| `numbers.tex` | Every number the text quotes, as `\val{key}` | `python tools/paper_numbers.py`, from the logs of `benchmarks/` |
| `tables/` | The tables | `python tools/paper_numbers.py`, from the same logs |
| `figures/` | The figures | `python tools/figures.py`, from the same logs |
| `references.bib` | The bibliography | hand; each entry was checked on the page named in its `verified` field |
| `references_verification.md`, `references_verification_addendum.md` | What was read to check each reference, the searches for prior work, and what could not be verified | hand |

No number of the article is typed by hand: `paper.tex` calls each value by a key, and the build fails if a key is not
in `numbers.tex`.

## Building

```
python tools/paper_numbers.py
python tools/figures.py
cd paper
pdflatex paper && bibtex paper && pdflatex paper && pdflatex paper
```

Any LaTeX distribution with the packages `lm`, `geometry`, `amsmath`, `amsfonts`, `graphics`, `booktabs`, `microtype`,
`xcolor`, `natbib`, `hyperref`, `xurl`, `caption`, `longtable` (in `tools`) and `pgf` (TikZ) will do. `paper.pdf` was built
with MiKTeX (pdfTeX 1.40.28).

The text of the article and its figures are released under the Creative Commons Attribution 4.0 licence (CC BY 4.0).
