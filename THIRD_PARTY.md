# Third-party material and attribution

The installed solver package `src/ritro/` contains the RITRO authors' solver and
geometry code under MIT. Its import dependency closure contains NumPy, SciPy
and Python standard-library modules. It does not import the comparison ports,
MuJoCo, GaBOtorch or their assets.

| Material | Location | License and source |
|---|---|---|
| RDSE-SB and accompanying comparison ports | `benchmarks/baselines/` | GPL-3.0-or-later; full text and provenance in that directory |
| SPD Gaussian kernel and tensor utilities | `applications/panda/vendor/gabo/` | MIT, copyright 2020 Noémie Jaquier; GaBOtorch commit `884f64a989699e4ead65cd7714892da79c223f82` |
| Panda rigid-body XML model | `applications/panda/assets/` | Apache-2.0; MuJoCo Menagerie Franka Emika Panda model; original license preserved |

The GaBO kernel's internal import was made package-relative. Its author
headers and license are retained. The local compatibility adapter replaces
distance evaluation with current PyTorch linear algebra; the experiment uses
LogEI and unconstrained multistart BFGS in matrix-log coordinates. It is the
paper's adapted GaBO comparison.

The Panda XML is retained byte-for-byte from the research experiment. The
rollout code removes visual and collision geometry at runtime for the stated
free-space task, retains inertial and kinematic parameters, and adds a TCP
site. No mesh files are required by that rollout. The model source is
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda).

Dependencies installed from PyPI retain their respective licenses. The MIT
license selected for the RITRO release applies to the authors' contribution;
the listed upstream notices remain in force for the included third-party files.
