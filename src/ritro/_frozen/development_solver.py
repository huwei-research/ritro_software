"""Development two-scale MFN solver; frozen solver remains unchanged.

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
    persistent_sampling: bool = False
    fast_algebra: bool = False
    growing_set: bool = False
    trial_scale_limit: float | None = None
    roundoff_guard: bool = False
    fast_extrema: bool = False
    weighted_reuse_radius: float | None = None
    reuse_poisedness: float | None = None
    incremental_reuse: bool = False
    systematic_repair: bool = False
    repair_directions: bool = True
    repair_score: str = 'prediction'
    insertion_rule: str = 'original'


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
    if o.trial_scale_limit is not None and (not np.isfinite(o.trial_scale_limit) or o.trial_scale_limit < 1):
        raise ValueError("Trial scale limit must be finite and at least one")
    if o.weighted_reuse_radius is not None and (not np.isfinite(o.weighted_reuse_radius) or o.weighted_reuse_radius < 2):
        raise ValueError('Weighted reuse outer radius must be finite and at least two')
    if o.systematic_repair and (o.weighted_reuse_radius is None or not o.incremental_reuse):
        raise ValueError('Systematic repair requires incremental weighted reuse')
    if o.repair_score not in ('prediction','curvature'):
        raise ValueError('Unknown repair score')
    if o.insertion_rule not in ('original','far','balanced','lookahead'):
        raise ValueError('Unknown insertion rule')
    if o.insertion_rule=='lookahead' and o.weighted_reuse_radius is None:
        raise ValueError('Insertion lookahead requires weighted reuse')
    if o.reuse_poisedness is not None and (not np.isfinite(o.reuse_poisedness) or o.reuse_poisedness<o.poisedness or o.weighted_reuse_radius is None):
        raise ValueError('Reuse poisedness requires weighted reuse and cannot be below base threshold')
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
    ball_maximum = maximum_absolute
    if o.fast_extrema:
        from .spectral_extrema import maximum_absolute as ball_maximum
    geometry_factory = Geometry
    if o.fast_algebra:
        from .fast_core import FastGeometry
        geometry_factory = FastGeometry
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
    sampling_radius = radius
    pending_trial = None
    model_center = None
    try:
        while True:
            if o.roundoff_guard and sampling_radius <= 256*np.finfo(float).eps*max(1.,np.linalg.norm(points[center])):
                status = 'sampling_resolution_limit'
                break
            previous_active = set(active)
            before = len(values)
            if not reuse_model:
                if not o.reuse_points:
                    active = []
                chart = manifold.chart(points[center])
                coords = (np.array([log(chart, points[i]) / sampling_radius for i in active])
                          if active else np.empty((0, d)))
                if pending_trial is not None and active:
                    selected = maintain_with_trial(coords, active,
                                log(chart, points[pending_trial])/sampling_radius,
                                pending_trial, d, o.nodes, o.poisedness)
                else:
                    count = max(d+1, len(active)) if o.growing_set else o.nodes
                    if o.weighted_reuse_radius is None:
                        selected = maintain(coords, active or [center], d, count, o.poisedness, geometry_factory=geometry_factory, ball_maximum=ball_maximum)
                    else:
                        from .weighted_reuse import select
                        if o.systematic_repair and len(active)==count:
                            from .systematic_repair import repair
                            archive=[]
                            # Bounded recent archive; chart.log rejects points outside its domain.
                            for key in range(len(points)-1,max(-1,len(points)-1-2*count),-1):
                                if key in active:continue
                                try:u=log(chart,points[key])/sampling_radius
                                except (ValueError,ArithmeticError):continue
                                if np.all(np.isfinite(u)) and np.linalg.norm(u)<=o.weighted_reuse_radius:
                                    archive.append((key,u))
                                if len(archive)==12:break
                            predictor=None
                            try:
                                preview_geometry=geometry_factory(coords)
                                preview_values=np.array([values[i] for i in active])
                                preview_prior=None
                                if o.transported_hessian_prior and model_center is not None:
                                    if model_center==center:preview_prior=B*sampling_radius**2
                                    else:
                                        preview_T=model_chart.transport_to(chart)
                                        transport_calls+=1
                                        preview_prior=(preview_T@B@preview_T.T)*sampling_radius**2
                                predictor=preview_geometry.fit(preview_values,prior=preview_prior)
                                if o.memory_region_multiplier is not None:
                                    preview_zero=preview_geometry.fit(preview_values)
                                    preview_diff=Model(0.,(predictor.gradient-preview_zero.gradient)/sampling_radius**2,
                                        (predictor.hessian-preview_zero.hessian)/sampling_radius**2)
                                    preview_defect=ball_maximum(preview_diff)[0]
                                    preview_bound=o.memory_region_multiplier*(o.memory_curvature_floor+
                                        np.linalg.norm(preview_zero.hessian/sampling_radius**2,2))
                                    preview_alpha=min(1.,preview_bound/preview_defect) if preview_defect>0 else 1.
                                    predictor=Model(preview_zero.constant,
                                        preview_zero.gradient+preview_alpha*(predictor.gradient-preview_zero.gradient),
                                        preview_zero.hessian+preview_alpha*(predictor.hessian-preview_zero.hessian))
                            except np.linalg.LinAlgError:pass
                            selected=repair(coords,active,d,count,o.poisedness,
                                o.weighted_reuse_radius,geometry_factory,ball_maximum,
                                o.reuse_poisedness,archive,predictor,o.repair_directions,o.repair_score)
                        else:
                            selected = select(coords,active or [center],d,count,o.poisedness,
                                          o.weighted_reuse_radius,geometry_factory,ball_maximum,
                                          o.reuse_poisedness,o.incremental_reuse)
                pending_trial = None
                active = []
                for v, node_id in zip(selected.points, selected.ids):
                    active.append(evaluate(exp(chart, sampling_radius * v), "geometry")
                                  if node_id is None else node_id)
                # Fit in coordinates of actual physical points, not idealized exp/log pairs.
                actual = np.array([log(chart, points[i]) / sampling_radius for i in active])
                physical_error = sampling_radius * np.max(np.linalg.norm(actual - selected.points, axis=1))
                roundoff = 256 * np.finfo(float).eps * max(1., sampling_radius, np.linalg.norm(points[center]))
                if physical_error > roundoff:
                    raise ArithmeticError("Exp/Log round-trip changed selected geometry")
                geometry = geometry_factory(actual)
                maximum = max(ball_maximum(m)[0] for m in geometry.lagrange()[1:])
                allowed_poisedness=o.reuse_poisedness or o.poisedness
                if maximum > allowed_poisedness + 1e-8:
                    raise ArithmeticError("Physical interpolation set failed poisedness check")
                if o.weighted_reuse_radius is not None:
                    from .weighted_reuse import measures
                    accuracy=measures(actual,geometry_factory,ball_maximum)
                    if (accuracy['radius']>o.weighted_reuse_radius+1e-9 or
                        accuracy['weighted_gradient']>4*(len(active)-1)*o.poisedness+1e-8 or
                        accuracy['weighted_hessian']>8*(len(active)-1)*o.poisedness+1e-8):
                        raise ArithmeticError('Physical weighted accuracy check failed')
                prior = None
                if o.transported_hessian_prior and model_center is not None:
                    if model_center == center:
                        prior = B * sampling_radius**2
                    else:
                        T = model_chart.transport_to(chart)
                        transport_calls += 1
                        prior = (T @ B @ T.T) * sampling_radius**2
                elif o.same_center_hessian_prior and model_center == center:
                    prior = B * sampling_radius**2
                model = geometry.fit(np.array([values[i] for i in active]), prior=prior)
                memory_control = None
                if o.memory_region_multiplier is not None:
                    zero = geometry.fit(np.array([values[i] for i in active]))
                    difference = Model(0., (model.gradient-zero.gradient)/sampling_radius**2,
                                       (model.hessian-zero.hessian)/sampling_radius**2)
                    defect = ball_maximum(difference)[0]
                    threshold = o.memory_region_multiplier * (o.memory_curvature_floor
                                    + np.linalg.norm(zero.hessian/sampling_radius**2, 2))
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
                model_scale = sampling_radius
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
            if sampling_radius > o.criticality * norm_g:
                reuse_model = False
                record["action"] = "criticality"
                sampling_radius *= o.contraction
                record["next_sampling_radius"] = sampling_radius
                transitions += 1
                if sampling_radius == 0:
                    raise ArithmeticError("Radius underflow")
                continue
            trial_radius = radius if o.trial_scale_limit is None else min(radius, o.trial_scale_limit*sampling_radius)
            record['trial_radius'] = trial_radius
            step, step_info = truncated_cg(g, B, trial_radius)
            record["subproblem"] = step_info
            predicted = float(-g @ step - .5 * step @ B @ step)
            if not np.isfinite(predicted) or predicted <= 0:
                raise ArithmeticError("Nonpositive trust-region predicted decrease")
            before_trial = len(values)
            trial = evaluate(exp(chart, step), "trial")
            ratio = (values[center] - values[trial]) / predicted
            record.update(trial_id=trial, new_trial=len(values) - before_trial,
                          ratio=float(ratio), predicted=predicted)
            augmented = False
            if o.growing_set and len(active) < o.nodes and trial not in active:
                trial_coords = log(chart, points[trial]) / model_scale
                candidate = np.vstack([actual, trial_coords])
                try:
                    geometry_factory(candidate)
                    active = active + [trial]
                    actual = candidate
                    augmented = True
                except np.linalg.LinAlgError:
                    pass
            record['set_augmented'] = augmented
            if ratio >= o.acceptance:
                new_coords=None;preview=None
                if o.insertion_rule!='original':
                    new_chart=manifold.chart(points[trial])
                    new_coords=np.array([log(new_chart,points[i])/model_scale for i in active])
                    if o.insertion_rule=='lookahead':
                        from .weighted_reuse import select
                        preview=lambda z,keys:select(z,keys,d,len(keys),o.poisedness,
                            o.weighted_reuse_radius,geometry_factory,ball_maximum,
                            o.reuse_poisedness,o.incremental_reuse)
                inserted = insert_center(actual, active, log(chart, points[trial]) / model_scale, trial,
                    geometry_factory=geometry_factory,new_center_coordinates=new_coords,
                    insertion_rule=o.insertion_rule,repair_preview=preview)
                active = inserted.ids
                record["insertion_events"] = inserted.events
                record["action"] = "accept"
                reuse_model = False
                center = trial
                pending_trial = None
                if ((o.expansion_acceptance is None or ratio >= o.expansion_acceptance)
                        and (o.expansion_boundary_fraction is None
                             or np.linalg.norm(step) >= o.expansion_boundary_fraction * trial_radius)):
                    radius = min(o.maximum_radius, o.expansion * radius)
                record["next_radius"] = radius
                sampling_radius = min(sampling_radius, radius) if o.persistent_sampling else radius
            else:
                record["action"] = "reject"
                if o.reuse_rejected_trial:
                    pending_trial = trial
                radius *= o.contraction
                sampling_radius = min(sampling_radius, radius)
                reuse_model = o.persistent_sampling and sampling_radius == model_scale and not augmented
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
