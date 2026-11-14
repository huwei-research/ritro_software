# RITRO

RITRO solves smooth derivative-free optimization problems on Riemannian
manifolds using intrinsic quadratic interpolation models and trust-region steps.
The solver reuses evaluated points, transports curvature information, and
controls the inherited model through a regional bound.

This distribution accompanies *An intrinsic interpolation trust-region method
for Riemannian derivative-free optimization* by Wei Hu, Ya-Xiang Yuan and
Pengcheng Xie (manuscript v4.0; numerical results unchanged from v3.9). Version 0.2.0 packages the frozen
`main_20261104` implementation used by the manuscript. The earlier 0.1.0
research package has a different implementation.

## Install and solve

Python 3.10 or later, NumPy and SciPy are required. From this directory:

```sh
python -m pip install .
python examples/sphere.py
python examples/spd.py
```

```python
import numpy as np
from ritro import minimize, default_options
from ritro.manifolds import Sphere

x0 = np.ones(3) / np.sqrt(3)
options = default_options(2, max_evaluations=180)
result = minimize(lambda x: float(np.array([1., 2., 4.]) @ x**2),
                  Sphere(2), x0, options=options)
print(result["best_value"], result["best_point"], result["evaluations"])
```

For an editable installation use `python -m pip install -e ".[test]"`.
Run commands from this distribution or an unrelated working directory;
the historical research repository's top-level `ritro/` package can shadow
an installed package when that repository is the current directory.
Check the loaded version with:

```sh
python -c "import ritro; print(ritro.__version__, ritro.__file__)"
```

## Interface and parameters

- `minimize(objective, manifold, initial, options=...)` requests scalar function
  values only. It returns the frozen solver record plus the best evaluated
  point and value. Geometry evaluations and rejected trials count towards
  the function-evaluation budget.
- `default_options(d, ...)` selects the manuscript's frozen algorithm with
  `2*d+1` model points, tCG, persistent sampling scale, transported Hessian
  updates, regional protection and incremental weighted reuse. Use
  `dataclasses.replace(options, initial_radius=...)` to change parameters.
- `solve(objective, manifold, initial, options)` exposes the original solver
  record: `point` is the accepted centre, while the archive contains all queries.
- Supported geometries include Euclidean space, spheres, products, Stiefel,
  Grassmann, affine-invariant SPD matrices and hyperbolic space. See
  [the manifold interface](docs/manifolds.md) for custom problems.

Without a supplied valid `lipschitz_bound`, termination uses the evaluation
budget, transition limit, or numerical sampling resolution. A
`sampling_resolution_limit` status is reported separately from a certified
gradient-norm stop. [Parameters and outputs](docs/solver.md) give the details.

## Experiments

- [Benchmarks](benchmarks/README.md): nine families, 27 configurations,
  RITRO, Int-RFD and RDSE-SB; charged evaluations, paired seeds and the
  manuscript's selected parameter indices.
- [Panda impedance task](applications/panda/README.md): optional MuJoCo
  application comparing RITRO, Int-RFD and adapted GaBO.
- [Saved-data reproduction](reproduction/README.md): compact records and
  publication figures, without new optimizer or simulation calls.

Small benchmark check:

```sh
python benchmarks/run_benchmark.py --problems sphere_8 --budget 40 --output outputs/sphere_check
```

Each experiment requires a new output directory. Existing output is never
silently overwritten. The full experimental settings are documented alongside
the corresponding runners.

## Validation and source history

```sh
python -m pip install ".[test]"
python -m pytest
```

Tests cover stationarity certification on a quadratic with a known bound,
complete evaluation accounting, feasibility, local coordinate inverses,
isometric transport and the two radius scales. Packaging checks compare
the numerical source and short trajectories against the frozen research
implementation. See [MANIFEST.md](MANIFEST.md) and `source_manifest.json`.

## Citation and licenses

Citation metadata are provided in `CITATION.cff`. No publication DOI,
arXiv identifier or public repository URL is assigned in this package.

The RITRO solver installed from `src/ritro/` is licensed under the
[MIT License](LICENSE). Comparison ports in `benchmarks/baselines/` retain
[GPL-3.0-or-later](benchmarks/baselines/LICENSE); they are outside the
installed solver wheel. The optional GaBO kernel retains its upstream MIT
notice and the Panda model its Apache-2.0 license.
[THIRD_PARTY.md](THIRD_PARTY.md) lists sources, modifications and attribution.

Repository: https://github.com/huwei-research/ritro_software
