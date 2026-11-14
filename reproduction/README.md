# Reproduce the RITRO paper figures

This directory is self-contained. It reproduces the numerical figures in
manuscript v3.9 with no optimizer, simulator or research repository. Python
3.10+, NumPy, Matplotlib and PyMuPDF are required. A TeX installation gives
the original Computer Modern typography.

```sh
python -m pip install numpy matplotlib pymupdf
python plot_figures.py --output output
```

Run here, or pass the full script path from any directory. `--check-only`
validates the data and numerical arrays without saving figures. `--no-tex`
uses Matplotlib fonts when TeX is unavailable. The script writes nine PDFs
and PNGs: selected/default performance and data profiles, three descent
grids, and two robot figures. Figure 1 is an illustration in the manuscript.

## Data and provenance

| Input | Bytes | Retained observations |
|---|---:|---|
| `data/metrics.json` | 1,301,405 | 4,860 first-hit records: 270 instances, 3 methods, 3 targets and 2 parameter views |
| `data/descent_records.json.gz` | 6,250,134 | 1,620 trajectories with exact incumbent change points |
| `data/panda_results.json` | 711,666 | 30 robot runs, scalar costs, controllers and representative traces |

Total input size is **8,263,205 bytes**. The manifest gives original
research-relative paths, source hashes and distribution hashes. Numerical
values are preserved; JSON whitespace is compressed and machine-local path
prefixes become `research_results/`. These labels identify the original
records; the script does not open those external paths.

Descent records reconstruct the incumbent at every evaluation count. They
omit non-improving query values, query locations and solver internal state.
Robot display traces retain every tenth simulation sample (20 ms). Full
raw records remain in the research repository. Failed/stopped runs and
unachieved targets are retained according to the paper's conventions.

The input budget is 10 MiB total, 7 MiB per input and 128 MiB Python allocation
for reading any one input. Independent reading measured peaks of 5,442,572,
56,009,364 and 2,067,445 bytes respectively, using `tracemalloc`. This measures
JSON reading, excluding imported libraries and complete plotting-process RSS.

## Verification

`expected_figures.json` contains profile counts, quantile hashes and all 218
line/band numerical-array hashes from the original generator. Every run
checks these and the input hashes. `validated_results.json` records the
completed isolated-directory check. All nine resulting PDFs were
byte-identical to the original numerical figures in the recorded environment
(NumPy 2.3.2, Matplotlib 3.10.9). Rendering may vary with library or TeX
versions; the numerical checks remain applicable.

The export source is `scripts/prepare_portable_figures.py` in the research
repository. Plot functions come from `scripts/figures_paper_v3_5.py`, retained
by manuscript v3.9. The style is included locally; no user skill path, prior
manuscript generator, hidden cache or simulator is needed.

Publication input check: 8,263,205 bytes of numerical input; largest input 6,250,134 bytes. Peak measured Python allocation for per-file reading was 55,994,081 bytes (CSV streamed; NPZ byte-stream check; imported libraries and rendering excluded). Budgets: 20 MiB total, 8 MiB per file, 128 MiB reading allocation.

