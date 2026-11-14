# Panda impedance tuning

This task optimizes a full SPD(3) Cartesian stiffness matrix. One function
evaluation runs a four-second MuJoCo rollout with step 0.002 s, constant target
`(0.66, -0.01, 0.69)` m and external force `(0, 20, -20)` N. The objective is
the time-mean squared position error, plus `1e-13*det(K)` and
`1e-4*cond(K)`. These are the same settings as the manuscript application.

From the distribution root:

```sh
python -m pip install ".[robot]"
python applications/panda/run.py --methods ritro int_rfd --budget 20 --output outputs/panda_check
```

To include the adapted GaBO comparison:

```sh
python -m pip install ".[robot,gabo]"
python applications/panda/run.py --methods ritro int_rfd gabo --budget 300 --seeds 4201 4202 4203 4204 4205 4206 4207 4208 4209 4210 --output outputs/panda_full
```

All three methods receive the same five initial SPD matrices for each seed.
All five rollout evaluations count towards the budget. RITRO and Int-RFD
start from the best initial observation; its already known value is reused
once without a duplicate simulation. GaBO starts with all five observations.
The default selected parameters are initial radius 0.3 for RITRO,
initial regularization 0.01 for Int-RFD, and the base GaBO candidate. These
choices were selected using the paper's separate development seeds.

Each JSON result contains the queried stiffness matrices, their component
objectives and physical metrics, objective-call counts, rollout counts and
errors. Every evaluation is checked against the remaining budget before
simulation. The best observation and its task metrics are read from existing
records, without an additional uncharged rollout. Task success uses final-
second position RMS at most 20 mm and the stated joint-limit tolerance.

The GaBO code uses the author's AIRM Gaussian kernel with the manuscript's
LogEI, fit-recovery and unconstrained acquisition-optimization adaptations.
Its vendor code retains MIT notices. The Panda XML retains its Apache-2.0
license. Int-RFD's comparison module retains the license specified under
`benchmarks/baselines/`. All optional software is outside the installed MIT
solver wheel.

Fresh simulations depend on the MuJoCo and numerical-library versions;
the runner records installed versions. Exact published curves and figures
are reproduced from the compact records in `reproduction/`.
