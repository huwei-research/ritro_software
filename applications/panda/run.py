"""Run the paper's value-only Panda stiffness task with explicit rollout accounting."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
from dataclasses import asdict, replace

for key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np
from ritro import default_options, solve
from ritro.manifolds import SPD
from ritro._frozen.vectorized_coefficients import install
from panda_paper_task import Objective, CONFIG


class BudgetExhausted(BaseException):
    pass


def initial_points(seed):
    rng = np.random.default_rng(seed)
    result = []
    for i in range(5):
        Q = np.linalg.qr(rng.normal(size=(3, 3)))[0]
        eigenvalues = (np.array([100., 300., 900.]) if i == 0 else
                       np.exp(rng.uniform(np.log(100), np.log(900), 3)))
        result.append(((Q*eigenvalues) @ Q.T).ravel())
    return result


def run(seed, method, budget, selected=True):
    if budget < 5:
        raise ValueError("The common initial design uses five charged evaluations")
    objective = Objective()
    records = []
    opt, result, acquisition, error = None, None, None, None
    parameters = ({"initial_radius": .3} if method == "ritro" else
                  {"sigma0": .01} if method == "int_rfd" else {}) if selected else {}
    started = time.perf_counter()

    def oracle(point, *args, **kwargs):
        if len(records) >= budget:
            raise BudgetExhausted()
        row = {"point": np.asarray(point).ravel().tolist(), "value": None}
        records.append(row)
        before, tick = objective.sim.calls, time.perf_counter()
        try:
            row["value"] = float(objective(point))
            row["physical"] = objective.records[-1]
            return row["value"]
        finally:
            row["rollouts"] = objective.sim.calls-before
            row["objective_seconds"] = time.perf_counter()-tick

    restore = install()
    try:
        for point in initial_points(seed):
            oracle(point)
        best_initial = min(records, key=lambda row: row["value"])
        initial = np.asarray(best_initial["point"])
        cached = [True]

        def local_oracle(point, *args, **kwargs):
            if cached[0] and np.array_equal(point, initial):
                cached[0] = False
                return best_initial["value"]
            return oracle(point, *args, **kwargs)

        local_oracle.retractions = 0
        if method == "ritro":
            opt = replace(default_options(6, max_evaluations=budget-4,
                          max_transitions=10000), **parameters)
            result = solve(local_oracle, SPD(), initial.copy(), opt)
            status, error = result["status"], result["failure"]
        elif method == "int_rfd":
            # The comparison code is outside the MIT solver wheel and retains GPL.
            sys.path.insert(0, str(Path(__file__).resolve().parents[2]/"benchmarks"))
            from baselines.spd_metric_baselines import rfd
            rfd(SPD(), initial.copy(), local_oracle, random_seed=seed, **parameters)
            status = "returned"
        elif method == "gabo":
            from gabo_acquisition import Acquisition
            acquisition = Acquisition(seed, parameters)
            while len(records) < budget:
                point = acquisition.propose([r["point"] for r in records],
                                            [r["value"] for r in records])
                oracle(point)
            status = "evaluation_limit"
        else:
            raise ValueError(method)
    except BudgetExhausted:
        status = "evaluation_limit"
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        restore()
    valid = [row for row in records if row["value"] is not None]
    best = min(valid, key=lambda row: row["value"]) if valid else None
    assert objective.sim.calls == sum(row["rollouts"] for row in records)
    return {
        "summary": {"method": method, "seed": seed, "budget": budget,
                    "parameters": parameters, "evaluations": len(records),
                    "rollouts": objective.sim.calls, "status": status, "error": error,
                    "initial": records[0]["value"] if records else None,
                    "initial_design_best": min(row["value"] for row in valid[:5]) if valid else None,
                    "best": best["value"] if best else None,
                    "best_point": best["point"] if best else None,
                    "physical": best.get("physical") if best else None,
                    "seconds": time.perf_counter()-started},
        "oracle": records, "config": CONFIG, "options": asdict(opt) if opt else None,
        "evaluation_sources": result["evaluation_sources"] if result else None,
        "acquisition": acquisition.records if acquisition else None,
        "fit_records": acquisition.fit_records if acquisition else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", choices=["ritro", "int_rfd", "gabo"],
                        default=["ritro", "int_rfd"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[4201])
    parser.add_argument("--budget", type=int, default=20)
    parser.add_argument("--parameters", choices=["selected", "default"], default="selected")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.budget < 5:
        parser.error("--budget must be at least five")
    args.output.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ["numpy", "scipy", "mujoco", "torch", "gpytorch", "botorch"]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    (args.output/"manifest.json").write_text(json.dumps({"config": CONFIG, "versions": versions,
        "methods": args.methods, "seeds": args.seeds, "budget": args.budget,
        "parameters": args.parameters}, indent=2)+"\n")
    for seed in args.seeds:
        for method in args.methods:
            record = run(seed, method, args.budget, args.parameters == "selected")
            (args.output/f"s{seed}_{method}.json").write_text(json.dumps(record, indent=2)+"\n")
            print(json.dumps(record["summary"]), flush=True)


if __name__ == "__main__":
    main()
