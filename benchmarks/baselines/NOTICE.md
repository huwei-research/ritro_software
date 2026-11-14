# Comparison implementations

`report_baselines.py` and `spd_metric_baselines.py` are the Python comparison
ports used in the RITRO experiments. They are distributed under
**GPL-3.0-or-later**, with the complete license in `LICENSE`. These files are
outside the installed MIT `ritro` solver wheel and are imported only by the
comparison runners.

- RDSE-SB derives from `smooth/linesearchext.m` in
  [DamianoZeffiro/riemannian-ds](https://github.com/DamianoZeffiro/riemannian-ds),
  commit `3b850ff913202a7de64441364076976e5320d168`.
  Upstream authors: Vyacheslav Kungurtsev, Francesco Rinaldi and Damiano Zeffiro.
  Their licensing notice is preserved in `UPSTREAM_README.md`.
- Int-RFD implements the algorithms of Taminiau, Massart and Grapiglia;
  comparison against their MATLAB source used
  [taminiaut/Extrinsic-Intrinsic-Riemannian-Finite-Difference-Method](https://github.com/taminiaut/Extrinsic-Intrinsic-Riemannian-Finite-Difference-Method),
  commit `d351efedb5ed1f599cb94144408df3fa22623d05`.
- The Python ports, metric adaptations and oracle-budget integration were
  prepared for the RITRO experiments. The embedded version uses projected
  ambient directions; the Grassmann, SPD and hyperbolic adaptations use
  orthonormal tangent coordinates.

The two source files are unchanged from the recorded research copies.
`upstream_manifest.json` records both checked source revisions. The MIT license
at the distribution root does not replace the license of these comparison files.
