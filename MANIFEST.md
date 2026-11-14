# Release source manifest

Software version: **0.2.0**. Manuscript: **v3.9**. Algorithm source:
**main_20261104**.

`source_manifest.json` records research-relative source paths, source SHA256,
delivered paths, delivered SHA256 and the edits needed to make the modules
importable without a research checkout. Source paths identify provenance;
they are not runtime dependencies.

The frozen algorithm modules were copied from
`prototypes/transported_mfn/releases/main_20261104/`. Only local imports were
changed to package-relative imports. The solver's dataclasses, numerical
branches, constants, order of operations and stopping rules were retained.
The optional vectorized coefficient functions are an existing research
acceleration, supplied separately; the core API does not enable them implicitly.

The geometric modules and problem factories come from the manuscript
experiment implementation. The four original problem factories were extracted
from `run_seed_benchmark.py`; their function bodies are retained. The benchmark
and robot command-line interfaces are new portable entry points around those
same objective and optimizer routines.

Source mappings include the robot objective, acquisition implementation,
GaBO kernel and Panda XML. Licenses remain attached as described in
`THIRD_PARTY.md`. The current release's MIT license was selected by the author
on 2026-11-14; historical research snapshots retain their recorded files.

Validation records belong to the research review
`reviews/project_cleanup_20261114/software_review.md`. The test commands in
this distribution run independently of that review and the research repository.
