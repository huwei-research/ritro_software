"""Independent intrinsic MFN trust-region research prototype.

Manifold interface: dimension and chart(x), with exp(s), log(y) in orthonormal
tangent coordinates. No analytic gradient is used. This floating-point solver
does not certify the exact-arithmetic geometry subroutine.
"""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .core import Geometry, Model, maximum_absolute
from .maintenance import insert_center, maintain, maintain_with_trial
from .steps import truncated_cg


@dataclass(frozen=True)
class Options:
    nodes: int
    initial_radius: float = .2
    maximum_radius: float = .3
    poisedness: float = 2.
    criticality: float = 1.
    acceptance: float = .1
    contraction: float = .5
    expansion: float = 2.
    epsilon: float = .05
    lipschitz_bound: float | None = None
    max_transitions: int = 500
    max_evaluations: int = 5000
    reuse_points: bool = True
    expansion_acceptance: float | None = None
    model_scale_ratio: float = 1.
    reuse_rejected_trial: bool = False
    same_center_hessian_prior: bool = False
    transported_hessian_prior: bool = False
    expansion_boundary_fraction: float | None = None
    memory_region_multiplier: float | None = None
    memory_curvature_floor: float = 1.


class _BudgetReached(Exception):
    pass


def solve(objective, manifold, initial, options, stop_observer=None):
    started = perf_counter()
    d, o = manifold.dimension, options
    scalars = [o.initial_radius, o.maximum_radius, o.poisedness, o.criticality,
               o.acceptance, o.contraction, o.expansion, o.epsilon]
    if (not np.all(np.isfinite(scalars)) or not 0 < o.initial_radius <= o.maximum_radius
            or o.poisedness <= 1 or o.criticality <= 0 or not 0 < o.acceptance < 1
            or not 0 < o.contraction < 1 or o.expansion <= 1 or o.epsilon <= 0
            or not d + 1 <= o.nodes <= (d + 1) * (d + 2) // 2
            or o.max_evaluations < 1 or o.max_transitions < 0):
        raise ValueError("Invalid solver options")
    if o.lipschitz_bound is not None and (not np.isfinite(o.lipschitz_bound)
                                        or o.lipschitz_bound < 0):
        raise ValueError("Invalid supplied Lipschitz bound")
    if o.expansion_acceptance is not None and not o.acceptance <= o.expansion_acceptance < 1:
        raise ValueError("Expansion threshold must be between acceptance and one")
    if not np.isfinite(o.model_scale_ratio) or o.model_scale_ratio < 1:
        raise ValueError("Model scale ratio must be at least one")
    if (o.lipschitz_bound is not None and
            (o.same_center_hessian_prior or o.transported_hessian_prior) and
            o.memory_region_multiplier is None):
        raise ValueError("Zero-prior stationarity certificate is invalid with a Hessian prior")
    if o.expansion_boundary_fraction is not None and not 0 < o.expansion_boundary_fraction <= 1:
        raise ValueError("Expansion boundary fraction must be in (0,1]")
    if o.memory_region_multiplier is not None:
        if (not np.isfinite(o.memory_region_multiplier) or o.memory_region_multiplier <= 0
                or not np.isfinite(o.memory_curvature_floor) or o.memory_curvature_floor <= 0
                or not o.transported_hessian_prior):
            raise ValueError("Memory control requires transport and positive finite parameters")
        if o.model_scale_ratio != 1.:
            raise ValueError("Regional memory theory requires model scale equal to trial radius")
    points, values, sources, history = [], [], [], []
    map_calls = {"exp": 0, "log": 0}
    transport_calls = 0

    def exp(chart, vector):
        map_calls["exp"] += 1
        return chart.exp(vector)

    def log(chart, point):
        map_calls["log"] += 1
        return chart.log(point)

    def evaluate(point, source):
        point = np.asarray(point, dtype=float)
        if not np.all(np.isfinite(point)):
            raise ValueError("Nonfinite physical point")
        for i, old in enumerate(points):
            if np.array_equal(point, old):
                return i
        if len(values) == o.max_evaluations:
            raise _BudgetReached()
        value = float(objective(point.copy()))
        if not np.isfinite(value):
            raise ValueError("Objective returned a nonfinite value")
        points.append(point.copy())
        values.append(value)
        sources.append(source)
        return len(values) - 1

    manifold.chart(initial)  # validate before any objective query
    center = evaluate(initial, "initial")
    active = []
    radius = o.initial_radius
    transitions = 0
    status = "transition_limit"
    failure = None
    reuse_model = False
    model_scale = radius
    pending_trial = None
    model_center = None
    try:
        while True:
            previous_active = set(active)
            before = len(values)
            if not reuse_model:
                if not o.reuse_points:
                    active = []
                chart = manifold.chart(points[center])
                coords = (np.array([log(chart, points[i]) / radius for i in active])
                          if active else np.empty((0, d)))
                if pending_trial is not None and active:
                    selected = maintain_with_trial(coords, active,
                                log(chart, points[pending_trial])/radius,
                                pending_trial, d, o.nodes, o.poisedness)
                else:
                    selected = maintain(coords, active or [center], d, o.nodes, o.poisedness)
                pending_trial = None
                active = []
                for v, node_id in zip(selected.points, selected.ids):
                    active.append(evaluate(exp(chart, radius * v), "geometry")
                                  if node_id is None else node_id)
                # Fit in coordinates of actual physical points, not idealized exp/log pairs.
                actual = np.array([log(chart, points[i]) / radius for i in active])
                physical_error = radius * np.max(np.linalg.norm(actual - selected.points, axis=1))
                roundoff = 256 * np.finfo(float).eps * max(1., radius, np.linalg.norm(points[center]))
                if physical_error > roundoff:
                    raise ArithmeticError("Exp/Log round-trip changed selected geometry")
                geometry = Geometry(actual)
                maximum = max(maximum_absolute(m)[0] for m in geometry.lagrange()[1:])
                if maximum > o.poisedness + 1e-8:
                    raise ArithmeticError("Physical interpolation set failed poisedness check")
                prior = None
                if o.transported_hessian_prior and model_center is not None:
                    if model_center == center:
                        prior = B * radius**2
                    else:
                        T = model_chart.transport_to(chart)
                        transport_calls += 1
                        prior = (T @ B @ T.T) * radius**2
                elif o.same_center_hessian_prior and model_center == center:
                    prior = B * radius**2
                model = geometry.fit(np.array([values[i] for i in active]), prior=prior)
                memory_control = None
                if o.memory_region_multiplier is not None:
                    zero = geometry.fit(np.array([values[i] for i in active]))
                    difference = Model(0., (model.gradient-zero.gradient)/radius**2,
                                       (model.hessian-zero.hessian)/radius**2)
                    defect = maximum_absolute(difference)[0]
                    threshold = o.memory_region_multiplier * (o.memory_curvature_floor
                                    + np.linalg.norm(zero.hessian/radius**2, 2))
                    alpha = min(1., threshold/defect) if defect > 0 else 1.
                    if alpha < 1.:
                        model = Model(zero.constant, zero.gradient+alpha*(model.gradient-zero.gradient),
                                      zero.hessian+alpha*(model.hessian-zero.hessian))
                        if prior is not None:
                            prior = alpha*prior
                    memory_control = dict(defect=float(defect), threshold=float(threshold),
                                          alpha=float(alpha))
                model_center = center
                model_chart = chart
                model_scale = radius
                g, B = model.gradient / model_scale, model.hessian / model_scale**2
            norm_g = np.linalg.norm(g)
            record = {"center_id": center, "value": values[center], "radius": radius,
                      "active_ids": list(active), "new_geometry": len(values) - before,
                      "model_gradient_norm": float(norm_g),
                      "model_gradient": g.tolist(), "model_hessian": B.tolist(),
                      "prior_hessian": None if prior is None else (prior/model_scale**2).tolist(),
                      "hessian_norm": float(np.linalg.norm(B, 2)),
                      "geometry_events": [] if reuse_model else selected.events, "poisedness": maximum,
                      "memory_control": memory_control,
                      "model_reused": reuse_model, "model_scale": model_scale}
            record.update(evaluations=len(values), map_calls=dict(map_calls),
                          transport_calls=transport_calls,
                          elapsed=perf_counter() - started,
                          retained_nodes=len(previous_active.intersection(active)))
            history.append(record)
            # An experiment observer may end a run; it supplies no model or step data.
            if stop_observer is not None and stop_observer(points[center].copy()):
                status = "observer_stop"
                record["action"] = status
                break
            if o.lipschitz_bound is not None:
                reference = 2 * (o.nodes - 1) * o.poisedness * o.lipschitz_bound
                allowance = (0. if o.memory_region_multiplier is None else
                             o.memory_region_multiplier * (o.memory_curvature_floor + 2 * reference))
                A = max(1., reference + allowance)
                record["stationarity_bound"] = float(norm_g + A * model_scale)
                if record["stationarity_bound"] <= o.epsilon:
                    status = "certificate"
                    record["action"] = status
                    break
            if transitions >= o.max_transitions:
                record["action"] = "transition_limit"
                break
            if radius > o.criticality * norm_g:
                reuse_model = False
                record["action"] = "criticality"
                radius *= o.contraction
                transitions += 1
                if radius == 0:
                    raise ArithmeticError("Radius underflow")
                continue
            step, step_info = truncated_cg(g, B, radius)
            record["subproblem"] = step_info
            predicted = float(-g @ step - .5 * step @ B @ step)
            if not np.isfinite(predicted) or predicted <= 0:
                raise ArithmeticError("Nonpositive trust-region predicted decrease")
            before_trial = len(values)
            trial = evaluate(exp(chart, step), "trial")
            ratio = (values[center] - values[trial]) / predicted
            record.update(trial_id=trial, new_trial=len(values) - before_trial,
                          ratio=float(ratio), predicted=predicted)
            if ratio >= o.acceptance:
                inserted = insert_center(actual, active, log(chart, points[trial]) / model_scale, trial)
                active = inserted.ids
                record["insertion_events"] = inserted.events
                record["action"] = "accept"
                reuse_model = False
                center = trial
                pending_trial = None
                if ((o.expansion_acceptance is None or ratio >= o.expansion_acceptance)
                        and (o.expansion_boundary_fraction is None
                             or np.linalg.norm(step) >= o.expansion_boundary_fraction * radius)):
                    radius = min(o.maximum_radius, o.expansion * radius)
                record["next_radius"] = radius
            else:
                record["action"] = "reject"
                if o.reuse_rejected_trial:
                    pending_trial = trial
                radius *= o.contraction
                reuse_model = model_scale <= o.model_scale_ratio * radius
            transitions += 1
    except _BudgetReached:
        status = "evaluation_limit"
        if history and "action" not in history[-1]:
            history[-1]["action"] = status
    except (ArithmeticError, np.linalg.LinAlgError) as error:
        status = "numerical_error"
        failure = dict(message=str(error), radius=radius, transitions=transitions)
    return {"point": points[center].copy(), "value": values[center], "status": status,
            "evaluations": len(values), "transitions": transitions, "history": history,
            "archive_points": points, "archive_values": values, "evaluation_sources": sources,
            "map_calls": map_calls, "transport_calls": transport_calls, "failure": failure}
