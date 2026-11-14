"""Check release provenance and trajectories against an explicit research checkout."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import ritro
from ritro.manifolds import Euclidean, Sphere, SPD


class NormalizeImports(ast.NodeTransformer):
    def visit_ImportFrom(self, node):
        if node.module is None:
            return ast.Import(names=node.names)
        node.module = node.module.split(".")[-1]
        node.level = 0
        return node


def objects(tree, names):
    selected = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in names:
            selected.append(node)
        elif isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in names for target in node.targets):
            selected.append(node)
    return ast.Module(body=selected, type_ignores=[])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distribution = Path(__file__).resolve().parents[1]
    mapping = json.loads((distribution/"source_manifest.json").read_text())
    checks = []
    for item in mapping["files"]:
        original = args.research_root/item["source"]
        released = distribution/item["target"]
        assert hashlib.sha256(original.read_bytes()).hexdigest() == item["source_sha256"]
        assert hashlib.sha256(released.read_bytes()).hexdigest() == item["sha256"]
        row = {"target": item["target"], "hashes_match": True}
        if original.suffix == ".py":
            original_text = original.read_text(encoding="utf-8")
            changes = item.get("changes", [])
            if isinstance(changes, list):
                for change in changes:
                    if change["old"].startswith("Path(__file__).parent/"):
                        original_text = original_text.replace(change["old"], change["new"])
                        row["asset_path_adaptation"] = change
                    else:
                        assert change["old"].lstrip().startswith(("from ", "import "))
                        original_text = original_text.replace(change["old"], change["new"])
            old = ast.parse(original_text)
            new = ast.parse(released.read_text(encoding="utf-8"))
            if "extracted_objects" in item:
                old = objects(old, item["extracted_objects"])
                new = objects(new, item["extracted_objects"])
            old = NormalizeImports().visit(old)
            new = NormalizeImports().visit(new)
            assert ast.dump(old) == ast.dump(new), item["target"]
            row["arithmetic_ast_unchanged"] = True
        checks.append(row)
    frozen = args.research_root/"prototypes/transported_mfn/releases/main_20261104"
    sys.path.insert(0, str(frozen))
    import selected_regional as original_solver
    spd = SPD(2)
    target_chart = spd.chart(np.diag([1., 2.]).ravel())
    cases = [
        ("euclidean", Euclidean(2), np.array([.6, -.4]),
         lambda x: .5*(x[0]**2+2*x[1]**2)),
        ("sphere", Sphere(2), np.ones(3)/np.sqrt(3),
         lambda x: float(np.array([1., 2., 4.]) @ x**2)),
        ("spd", spd, np.diag([2., 1.]).ravel(),
         lambda x: float(target_chart.log(x) @ target_chart.log(x))),
    ]
    trajectories = []
    for name, manifold, initial, objective in cases:
        kwargs = {"max_evaluations": 80, "max_transitions": 200}
        old = original_solver.solve(objective, manifold, initial.copy(),
                                    original_solver.options(manifold.dimension, **kwargs))
        new = ritro.solve(objective, manifold, initial.copy(),
                          ritro.default_options(manifold.dimension, **kwargs))
        for key in ["status", "evaluations", "evaluation_sources", "map_calls", "transport_calls"]:
            assert old[key] == new[key], (name, key)
        np.testing.assert_array_equal(old["archive_points"], new["archive_points"])
        np.testing.assert_array_equal(old["archive_values"], new["archive_values"])
        trajectories.append({"problem": name, "evaluations": new["evaluations"],
                             "points_and_values_bitwise_identical": True})
    result = {"ritro_version": ritro.__version__, "imported_package": ritro.__file__,
              "source_checks": checks, "trajectory_checks": trajectories}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"source_files": len(checks), "trajectories": trajectories}, indent=2))


if __name__ == "__main__":
    main()
