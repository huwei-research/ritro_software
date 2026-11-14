from dataclasses import replace
import numpy as np
from ritro import minimize, solve, default_options
from ritro.manifolds import Euclidean, Sphere, SPD


def test_quadratic_certificate():
    result = minimize(lambda x: .5 * (x[0]**2 + 2*x[1]**2), Euclidean(2), [.6, -.4],
                      options=default_options(2, max_evaluations=2000,
                                              lipschitz_bound=2., epsilon=.05))
    assert result["status"] == "certificate"
    assert np.linalg.norm(result["point"] * [1, 2]) <= .05
    assert result["history"][-1]["stationarity_bound"] <= .05


def test_budget_covers_initialization_and_geometry():
    points = []
    def objective(x):
        points.append(x.copy())
        return float(x @ x)
    result = minimize(objective, Euclidean(2), [.6, -.4],
                      options=default_options(2, max_evaluations=4))
    assert result["status"] == "evaluation_limit"
    assert result["evaluations"] == len(points) == 4
    assert result["best_value"] == min(x @ x for x in points)


def test_sphere_feasibility_and_descent():
    x = np.ones(3) / np.sqrt(3)
    result = minimize(lambda z: float(np.array([1, 2, 4]) @ z**2), Sphere(2), x,
                      options=default_options(2, max_evaluations=120))
    assert result["best_value"] < 1.01
    for point in result["archive_points"]:
        assert abs(np.linalg.norm(point) - 1.) < 1e-10


def test_chart_roundtrip_and_isometric_transport():
    for manifold, point in [(Sphere(2), np.ones(3)/np.sqrt(3)),
                            (SPD(2), np.diag([1., 2.]).ravel())]:
        chart = manifold.chart(point)
        step = np.linspace(.01, .03, manifold.dimension)
        target = chart.exp(step)
        np.testing.assert_allclose(chart.log(target), step, atol=1e-12)
        transport = chart.transport_to(manifold.chart(target))
        np.testing.assert_allclose(transport.T @ transport, np.eye(manifold.dimension), atol=1e-11)


def test_sampling_resolution_is_separate_from_certificate():
    options = replace(default_options(1), initial_radius=1e-16)
    result = solve(lambda x: x[0]**2, Euclidean(1), np.array([1.]), options)
    assert result["status"] == "sampling_resolution_limit"
    assert result["evaluations"] == 1


def test_sampling_scale_and_trial_scale_invariants():
    options = replace(default_options(2, max_evaluations=120, max_transitions=40),
                      trial_scale_limit=4.)
    result = solve(lambda x: .0005 * (x @ x), Euclidean(2), np.array([1., .5]), options)
    history = result["history"]
    for before, after in zip(history, history[1:]):
        assert after["model_scale"] <= before["model_scale"]
    for row in history:
        if "trial_id" in row:
            point = result["archive_points"][row["trial_id"]]
            centre = result["archive_points"][row["center_id"]]
            assert np.linalg.norm(point-centre) <= 4*row["model_scale"]*(1+1e-9)
            assert row["model_scale"] <= row["trial_radius"]
