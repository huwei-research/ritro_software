from dataclasses import replace
import numpy as np
import pytest
from ritro import minimize, solve, default_options
from ritro.manifolds import Euclidean
from ritro._frozen.selected_regional import solve as frozen_solve


@pytest.mark.parametrize("budget", [1.5, np.nan, np.inf, True, 0, -1])
def test_invalid_budget_is_rejected_before_any_query(budget):
    calls = []
    options = replace(default_options(2), max_evaluations=budget)
    with pytest.raises(ValueError, match="max_evaluations"):
        solve(lambda x: calls.append(x), Euclidean(2), [1., 2.], options)
    assert calls == []


@pytest.mark.parametrize("budget", [1, 2, 4, 7])
def test_valid_budget_is_a_hard_query_limit(budget):
    calls = []
    def objective(x):
        calls.append(x.copy())
        return float(x @ x)
    r = minimize(objective, Euclidean(2), [1., 2.],
                 options=default_options(2, max_evaluations=np.int64(budget)))
    assert r["evaluations"] == len(calls) <= budget


def test_public_wrapper_preserves_frozen_query_trajectory():
    options = default_options(2, max_evaluations=40)
    f = lambda x: float(x @ np.diag([1., 10.]) @ x)
    a = solve(f, Euclidean(2), [1., 2.], options)
    b = frozen_solve(f, Euclidean(2), np.array([1., 2.]), options)
    np.testing.assert_array_equal(a["archive_points"], b["archive_points"])
    np.testing.assert_array_equal(a["archive_values"], b["archive_values"])
    assert a["status"] == b["status"]


@pytest.mark.parametrize("value", [np.array([1.]), 1.+2.j])
def test_objective_must_be_a_real_scalar(value):
    with pytest.raises(ValueError, match="real scalar"):
        minimize(lambda x: value, Euclidean(1), [1.])
