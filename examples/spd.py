"""A value-only affine-invariant squared-distance objective on SPD(2)."""
import numpy as np
from ritro import minimize, default_options
from ritro.manifolds import SPD

manifold = SPD(2)
target = np.diag([1.0, 2.0]).ravel()
chart = manifold.chart(target)
initial = np.diag([2.0, 1.0]).ravel()
result = minimize(lambda x: float(chart.log(x) @ chart.log(x)), manifold, initial,
                  options=default_options(3, max_evaluations=140))
print({key: result[key] for key in ["status", "evaluations", "best_value"]})
assert np.linalg.eigvalsh(result["best_point"].reshape(2, 2)).min() > 0
assert result["best_value"] < 0.02
