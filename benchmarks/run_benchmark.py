"""Run paired manuscript benchmark instances with charged function evaluations."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time
from dataclasses import asdict, replace

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
import ritro
from original_problems import get_problem, SIZES as ORIGINAL_SIZES
from additional_problems import problem as additional_problem, SIZES as EXTRA_SIZES

SIZES = {**ORIGINAL_SIZES, **EXTRA_SIZES}
CANDIDATES = {
    "ritro": [{}, {"initial_radius": .05}, {"initial_radius": .1},
              {"initial_radius": .3}, {"acceptance": .2}],
    "int_rfd": [{}, {"sigma0": .25}, {"sigma0": 4.}, {"tau0": 10.}, {"tau0": 1000.}],
    "rdse_sb": [{}, {"alpha0": .25}, {"alpha0": 2.},
                {"sufficient_decrease": .01}, {"sufficient_decrease": .05}],
}
FAMILIES = {
    "dictionary": ["dictionary_3_6_1", "dictionary_3_5_2", "dictionary_4_6_3"],
    "grassmann": ["grassmann_5_2", "grassmann_9_2", "grassmann_11_3"],
    "hyperbolic": ["hyperbolic_6", "hyperbolic_14", "hyperbolic_24"],
    "procrustes": ["procrustes_6_2", "procrustes_10_2", "procrustes_10_3"],
    "rotation": ["rotation_3", "rotation_6", "rotation_10"],
    "singular": ["singular_3_3_2", "singular_5_5_2", "singular_7_7_2"],
    "spd_mean": ["spd_mean_3", "spd_mean_5", "spd_mean_7"],
    "sphere": ["sphere_8", "sphere_16", "sphere_24"],
    "torus": ["torus_8", "torus_16", "torus_24"],
}
SELECTED = {
    "dictionary": [[3, 4, 3], [2, 3, 4], [0, 1, 1]],
    "grassmann": [[3, 0, 0], [3, 4, 4], [1, 0, 1]],
    "hyperbolic": [[0, 4, 4], [2, 4, 2], [3, 1, 1]],
    "procrustes": [[4, 0, 4], [2, 4, 4], [1, 1, 1]],
    "rotation": [[3, 3, 3], [2, 0, 4], [1, 1, 1]],
    "singular": [[3, 4, 4], [3, 2, 3], [0, 1, 1]],
    "spd_mean": [[2, 4, 3], [0, 2, 3], [1, 1, 1]],
    "sphere": [[0, 2, 3], [2, 3, 3], [1, 1, 4]],
    "torus": [[3, 3, 3], [1, 1, 1], [3, 3, 3]],
}


class BudgetExhausted(BaseException):
    """Stop before querying the oracle beyond its budget."""


def instance(name, seed):
    if name not in SIZES:
        raise ValueError(f"Unknown problem {name}")
    for attempt in range(100):
        used = seed + 100000 * attempt
        factory = additional_problem if name in EXTRA_SIZES else get_problem
        manifold, point, objective, diagnostic_gradient, data = factory(name, used)
        if not name.startswith("dictionary") or np.linalg.norm(data["Y"]) > 0:
            data.update(requested_seed=seed, generation_seed=used, generation_attempts=attempt + 1)
            return manifold, point, objective, data
    raise RuntimeError("Unable to generate nonzero dictionary data")


def parameters(name, method, view):
    if view == "default":
        return {}
    for family, names in FAMILIES.items():
        if name in names:
            index = SELECTED[family][list(CANDIDATES).index(method)][names.index(name)]
            return CANDIDATES[method][index].copy()
    raise ValueError(name)


def run(name, seed, method, budget, view="default"):
    manifold, initial, objective, data = instance(name, seed)
    params = parameters(name, method, view)
    records = []
    started = time.perf_counter()

    def oracle(point, feasible=True):
        if len(records) >= budget:
            raise BudgetExhausted()
        row = {"point": np.asarray(point).tolist(), "value": None,
               "feasible": bool(feasible)}
        records.append(row)
        tick = time.perf_counter()
        try:
            row["value"] = float(objective(point))
            if not np.isfinite(row["value"]):
                raise FloatingPointError("Nonfinite objective value")
            return row["value"]
        finally:
            row["objective_seconds"] = time.perf_counter() - tick

    oracle.retractions = 0
    result, opt, error = None, None, None
    try:
        if method == "ritro":
            opt = replace(ritro.default_options(manifold.dimension,
                          max_evaluations=budget, max_transitions=10000), **params)
            result = ritro.solve(oracle, manifold, initial.copy(), opt)
            status, error = result["status"], result["failure"]
        else:
            from baselines import report_baselines, spd_metric_baselines
            adapted = name.startswith(("grassmann_", "spd_mean_", "hyperbolic_"))
            module = spd_metric_baselines if adapted else report_baselines
            if method == "int_rfd":
                module.rfd(manifold, initial.copy(), oracle, **params)
            elif method == "rdse_sb":
                module.rdse(manifold, initial.copy(), oracle, **params)
            else:
                raise ValueError(method)
            status = "returned"
    except BudgetExhausted:
        status = "evaluation_limit"
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    valid = [r for r in records if r["value"] is not None and r["feasible"]]
    best = min(valid, key=lambda r: r["value"]) if valid else None
    return {
        "summary": {"problem": name, "seed": seed,
                    "generation_seed": data["generation_seed"],
                    "method": method, "view": view, "parameters": params,
                    "dimension": manifold.dimension, "budget": budget,
                    "evaluations": len(records), "status": status, "error": error,
                    "initial": records[0]["value"] if records else None,
                    "best": best["value"] if best else None,
                    "best_point": best["point"] if best else None,
                    "seconds": time.perf_counter() - started},
        "options": asdict(opt) if opt else None,
        "oracle": records,
        "evaluation_sources": result["evaluation_sources"] if result else None,
        "map_calls": result["map_calls"] if result else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--problems", nargs="+", default=["sphere_8"],
                        help="Problem names, small (nine families), or all (27 configurations)")
    parser.add_argument("--methods", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2201])
    parser.add_argument("--parameters", choices=["default", "selected"], default="default")
    parser.add_argument("--budget", type=int, help="Absolute function-evaluation budget")
    parser.add_argument("--budget-factor", type=int, default=10,
                        help="Budget = factor * (d+1), unless --budget is supplied")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = args.problems
    if names == ["all"]:
        names = [n for group in FAMILIES.values() for n in group]
    elif names == ["small"]:
        names = [group[0] for group in FAMILIES.values()]
    if any(name not in SIZES for name in names):
        parser.error("Unknown problem name")
    if args.budget is not None and args.budget < 1 or args.budget_factor < 1:
        parser.error("The budget must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"ritro_version": ritro.__version__, "parameters": args.parameters,
                "problems": names, "seeds": args.seeds, "methods": args.methods,
                "numpy": importlib.metadata.version("numpy"),
                "scipy": importlib.metadata.version("scipy"),
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "budget": args.budget, "budget_factor": args.budget_factor}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for name in names:
        for seed in args.seeds:
            for method in args.methods:
                budget = args.budget or args.budget_factor * (SIZES[name] + 1)
                record = run(name, seed, method, budget, args.parameters)
                path = args.output / f"{name}_s{seed}_{method}.json"
                path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
                print(json.dumps(record["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
