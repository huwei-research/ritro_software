"""Public interface to the frozen solver used in the paper."""
import numbers
import numpy as np
from ._frozen.selected_regional import options as _default_options, solve as _solve


def _integer(name, value, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def default_options(dimension, **kwargs):
    """Return the manuscript options with validated integer sizes and budgets."""
    _integer("dimension", dimension, 1)
    options = _default_options(dimension, **kwargs)
    _validate_options(dimension, options)
    return options


def _validate_options(dimension, options):
    _integer("dimension", dimension, 1)
    _integer("nodes", options.nodes, dimension + 1)
    _integer("max_evaluations", options.max_evaluations, 1)
    _integer("max_transitions", options.max_transitions, 0)


def solve(objective, manifold, initial, options, stop_observer=None):
    """Validate the public call, then run the unchanged frozen numerical core."""
    if not callable(objective):
        raise TypeError("objective must be callable")
    if stop_observer is not None and not callable(stop_observer):
        raise TypeError("stop_observer must be callable")
    _validate_options(manifold.dimension, options)
    if np.iscomplexobj(initial):
        raise ValueError("initial must be a finite real vector")
    point = np.array(initial, dtype=float, copy=True)
    if point.ndim != 1 or not np.all(np.isfinite(point)):
        raise ValueError("initial must be a finite real vector")

    def scalar_objective(x):
        value = np.asarray(objective(x))
        if value.ndim != 0 or np.iscomplexobj(value):
            raise ValueError("objective must return a real scalar")
        return float(value)

    return _solve(scalar_objective, manifold, point, options, stop_observer=stop_observer)


def minimize(objective, manifold, initial, *, options=None):
    """Minimize a deterministic scalar objective on ``manifold``.

    ``manifold.dimension`` is the intrinsic dimension. ``manifold.chart(x)``
    supplies ``exp(s)``, ``log(y)``, and ``transport_to(other_chart)`` in
    orthonormal tangent coordinates. See ``docs/manifolds.md``.

    The returned dictionary retains the frozen solver's accepted ``point``
    and ``value``. ``best_point`` and ``best_value`` also include all evaluated
    geometry and rejected trial points. All objective evaluations count
    towards ``options.max_evaluations``. No derivatives are requested.
    """
    if options is None:
        options = default_options(manifold.dimension)
    result = solve(objective, manifold, initial, options)
    index = int(np.argmin(result["archive_values"]))
    result["best_point"] = result["archive_points"][index].copy()
    result["best_value"] = float(result["archive_values"][index])
    return result
