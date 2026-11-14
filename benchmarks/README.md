# Nine-family benchmark

Run from the distribution root after installing the solver:

```sh
python benchmarks/run_benchmark.py --problems small --budget 40 --output outputs/nine_family_check
python benchmarks/run_benchmark.py --problems grassmann_5_2 --methods ritro int_rfd rdse_sb --seeds 2201 2202 --parameters selected --budget-factor 200 --output outputs/grassmann
```

`small` selects one configuration per family. `all` selects all 27
configurations. `--budget` overrides `--budget-factor`; otherwise the charged
budget is `factor*(d+1)`. The small check validates execution and is not the
paper's performance experiment.

| Family | Configurations (intrinsic dimensions) |
|---|---|
| Dictionary learning | `dictionary_3_6_1` (8), `dictionary_3_5_2` (14), `dictionary_4_6_3` (27) |
| Grassmann estimation | `grassmann_5_2` (6), `grassmann_9_2` (14), `grassmann_11_3` (24) |
| Hyperbolic fitting | `hyperbolic_6`, `hyperbolic_14`, `hyperbolic_24` |
| Rectangular Procrustes | `procrustes_6_2` (9), `procrustes_10_2` (17), `procrustes_10_3` (24) |
| Rotation synchronization | `rotation_3` (9), `rotation_6` (18), `rotation_10` (30) |
| Singular subspace | `singular_3_3_2` (6), `singular_5_5_2` (14), `singular_7_7_2` (22) |
| SPD mean | `spd_mean_3` (6), `spd_mean_5` (15), `spd_mean_7` (28) |
| Sphere energy | `sphere_8`, `sphere_16`, `sphere_24` |
| Phase synchronization | `torus_8`, `torus_16`, `torus_24` |

The objective factories retain the research implementation's draws and
formulas. Dictionary data are conditioned on nonzero data without reference
to solver outcomes. Available derivative diagnostics are never passed to an
optimizer. All methods start at the same physical point for a given seed.
As in the recorded benchmark runner, the Int-RFD direction generator uses
its default seed 0; the problem instance seed is a separate random stream.

The paper uses ten held-out seeds, 2201 through 2210, at budget `200*(d+1)`.
Run both parameter views in separate directories if reconstructing the pooled
empirical references used in the figures:

```sh
python benchmarks/run_benchmark.py --problems all --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --parameters selected --budget-factor 200 --output outputs/selected
python benchmarks/run_benchmark.py --problems all --seeds 2201 2202 2203 2204 2205 2206 2207 2208 2209 2210 --parameters default --budget-factor 200 --output outputs/default
```

Selected candidate indices are the saved development-set choices, not choices
made using these test runs. They are explicitly listed in `run_benchmark.py`.
The runner is serial and writes one JSON file per run, with all queried points,
values, error status, options and charged evaluations. It stops before any
over-budget query. No uncharged final function evaluation is used in selection.
The saved-data reproduction directory supplies the exact records used by the
paper; fresh runs can vary numerically across BLAS and library versions.

The comparison implementations retain their upstream attribution and
GPL-3.0-or-later license under [baselines](baselines/NOTICE.md).
