"""A value-only Rayleigh-quotient problem on the two-dimensional sphere."""
import numpy as np
from ritro import minimize, default_options
from ritro.manifolds import Sphere

diagonal = np.array([1.0, 2.0, 4.0])
initial = np.array([1.0, 1.0, 1.0]) / np.sqrt(3.0)
result = minimize(lambda x: float(diagonal @ x**2), Sphere(2), initial,
                  options=default_options(2, max_evaluations=180))
print({key: result[key] for key in ["status", "evaluations", "best_value"]})
print("point:", result["best_point"])
assert abs(np.linalg.norm(result["best_point"]) - 1.0) < 1e-10
assert result["best_value"] < 1.01
