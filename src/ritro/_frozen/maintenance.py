"""Current-center maintenance from current_center_mfn_complete.md.

Pure geometry: no objective callback or values. Coordinates are normalized by
the current radius. Stable integer IDs identify evaluated physical points;
None identifies a virtual node. Floating-point rank checks are diagnostics.
"""
from dataclasses import dataclass
import numpy as np
from .core import Geometry, canonical, maximum_absolute


@dataclass
class Selection:
    points: np.ndarray
    ids: list
    events: list


def potential(geometry):
    radii = np.linalg.norm(geometry.points, axis=1)
    return float(geometry.logdet - 4 * np.log(np.maximum(1., radii)).sum())


def maintain_with_trial(points, ids, trial_point, trial_id, d, n, threshold):
    """Compare virtual repairs; protect center and bound insertion volume loss."""
    baseline = maintain(points, ids, d, n, threshold)
    event = dict(kind="rejected_trial_candidate", id=trial_id, chosen=False,
                 baseline_unknown=baseline.ids.count(None))
    if trial_id in ids:
        event['reason'] = 'already_active'
        baseline.events.append(event)
        return baseline
    geometry = Geometry(points)
    cardinal = [abs(m.values(np.asarray(trial_point)[None, :])[0])
                for m in geometry.lagrange()[1:]]
    index = 1 + int(np.argmax(cardinal))
    candidate = np.asarray(points).copy()
    candidate[index] = trial_point
    try:
        proposed = Geometry(candidate)
    except np.linalg.LinAlgError:
        event['reason'] = 'singular_candidate'
        baseline.events.append(event)
        return baseline
    change = proposed.logdet - geometry.logdet
    event.update(logdet_change=float(change), removed_id=ids[index])
    if change < -2*np.log(n):
        event['reason'] = 'volume_loss'
        baseline.events.append(event)
        return baseline
    candidate_ids = list(ids)
    candidate_ids[index] = trial_id
    repaired = maintain(candidate, candidate_ids, d, n, threshold)
    event['candidate_unknown'] = repaired.ids.count(None)
    event['trial_survives'] = trial_id in repaired.ids
    # At equal paid cost, favor incorporating the observed failed direction.
    chosen = (trial_id in repaired.ids and
              repaired.ids.count(None) <= baseline.ids.count(None))
    event['chosen'] = chosen
    result = repaired if chosen else baseline
    result.events.append(event)
    return result


def insert_center(points, ids, new_point, new_id, geometry_factory=Geometry,
                  new_center_coordinates=None, insertion_rule='original', repair_preview=None):
    """Insert an already evaluated point in OLD normalized coordinates.

    Return the new center first, without translating coordinates. A manifold
    caller must recompute Log coordinates at the new center before maintain().
    """
    points = np.asarray(points, dtype=float)
    new_point = np.asarray(new_point, dtype=float)
    if len(ids) != len(points) or None in ids or len(set(ids)) != len(ids):
        raise ValueError("Insertion needs distinct evaluated physical IDs")
    if new_id is None or new_point.shape != (points.shape[1],):
        raise ValueError("Invalid new center")
    if not np.all(np.isfinite(new_point)):
        raise ValueError("Nonfinite center coordinates")
    if new_id in ids:
        index = ids.index(new_id)
        # The existing physical node and its cached value are authoritative.
        order = [index] + [i for i in range(len(ids)) if i != index]
        return Selection(points[order].copy(), [ids[i] for i in order],
                         [{"kind": "existing_center", "id": new_id}])
    old = geometry_factory(points)
    values = np.array([m.values(new_point[None, :])[0] for m in old.lagrange()])
    index = int(np.argmax(np.abs(values)))
    original_index=index
    preview_costs=[]
    if insertion_rule!='original':
        if insertion_rule not in ('far','balanced','lookahead'):
            raise ValueError('Unknown insertion rule')
        z=np.asarray(new_center_coordinates,dtype=float)
        if z.shape!=points.shape or not np.all(np.isfinite(z)):
            raise ValueError('New-center coordinates required')
        floor=1/len(points)
        if insertion_rule=='balanced':floor=max(floor,.5*abs(values[index]))
        eligible=np.flatnonzero(np.abs(values)>=floor)
        if len(eligible):
            radii=np.linalg.norm(z,axis=1)
            index=int(max(eligible,key=lambda j:(radii[j],abs(values[j]))))
            if insertion_rule=='lookahead':
                if repair_preview is None:raise ValueError('Repair preview required')
                shortlist=list(dict.fromkeys([original_index]+list(sorted(eligible,key=lambda j:-radii[j])[:3])))
                scored=[]
                for j in shortlist:
                    keep=[i for i in range(len(ids)) if i!=j]
                    coordinates=np.vstack([np.zeros(points.shape[1]),z[keep]])
                    trial_ids=[new_id]+[ids[i] for i in keep]
                    try:cost=repair_preview(coordinates,trial_ids).ids.count(None)
                    except np.linalg.LinAlgError:continue
                    preview_costs.append(dict(index=int(j),unknown=cost))
                    scored.append(((cost,-abs(values[j]),-radii[j]),int(j)))
                if scored:index=min(scored)[1]
    if abs(values[index]) < 1 / len(points) - 1e-10:
        raise ArithmeticError("Lagrange partition of unity check failed")
    trial = points.copy()
    trial[index] = new_point
    new = geometry_factory(trial)
    lower = 2 * np.log(abs(values[index]))
    if new.logdet - old.logdet < lower - 1e-8:
        raise ArithmeticError("Center insertion determinant bound failed")
    updated_ids = list(ids)
    removed = updated_ids[index]
    updated_ids[index] = new_id
    order = [index] + [i for i in range(len(ids)) if i != index]
    return Selection(trial[order], [updated_ids[i] for i in order], [{
        "kind": "center_insert", "removed_id": removed, "id": new_id,
        "pivot": float(values[index]),
        "insertion_rule": insertion_rule,"original_index": original_index,
        "selected_index": index,"preview_costs": preview_costs,
        "logdet_change": float(new.logdet - old.logdet)}])


def maintain(candidate, ids, d, n, threshold=2., archive_points=None,
             archive_ids=None, geometry_factory=Geometry, ball_maximum=maximum_absolute):
    """Repair locality and poisedness, then resolve exact cache matches.

    Empty candidate initializes with the canonical set; ids[0] then supplies
    the evaluated center ID. Otherwise candidate[0] must be exactly zero.
    Numerical singularity triggers a separately recorded reset, which is not
    claimed equivalent to exact-arithmetic singularity in the theorem.
    """
    if not np.isfinite(threshold) or threshold <= 1:
        raise ValueError("Poisedness threshold must be finite and exceed one")
    canonical_points = canonical(d, n)  # validates dimensions and node count
    points = np.asarray(candidate, dtype=float).reshape(-1, d)
    if not ids or ids[0] is None:
        raise ValueError("An evaluated center ID is required")
    center_id = ids[0]
    events = []
    if len(points) == 0:
        geometry = geometry_factory(canonical_points)
        current_ids = [center_id] + [None] * (n - 1)
        events.append({"kind": "initial_seed"})
    else:
        if len(points) != n or len(ids) != n or not np.array_equal(points[0], np.zeros(d)):
            raise ValueError("Candidate must contain n nodes with center first")
        known_ids = [i for i in ids if i is not None]
        if len(set(known_ids)) != len(known_ids):
            raise ValueError("Duplicate physical IDs")
        if not np.all(np.isfinite(points)):
            raise ValueError("Nonfinite Log coordinates")
        current_ids = list(ids)
        try:
            geometry = geometry_factory(points)
        except np.linalg.LinAlgError:
            geometry = geometry_factory(canonical_points)
            current_ids = [center_id] + [None] * (n - 1)
            events.append({"kind": "numerical_rank_reset"})

    initial_potential = potential(geometry)
    ceiling = n / 2 * np.log(n / 16 + 2) + (d + 1) / 2 * np.log(n)
    max_improvements = max(0, int(np.ceil(
        (ceiling - initial_potential) / (2 * np.log(threshold))))) + 2

    def replace(index, point, kind, maximum):
        nonlocal geometry
        before = geometry
        trial = before.points.copy()
        trial[index] = point
        geometry = geometry_factory(trial)
        gain = potential(geometry) - potential(before)
        required = 0 if kind == "far_replace" else 2 * np.log(threshold)
        if gain < required - 1e-8:
            raise ArithmeticError("Repair potential progress check failed")
        events.append({"kind": kind, "index": index,
                       "removed_id": current_ids[index], "maximum": float(maximum),
                       "old_radius": float(np.linalg.norm(before.points[index])),
                       "potential_gain": float(gain)})
        current_ids[index] = None

    for _ in range(n - 1):
        far = np.flatnonzero(np.linalg.norm(geometry.points, axis=1) > 2)
        if not len(far):
            break
        index = int(far[0])
        value, point = ball_maximum(geometry.lagrange()[index])
        if index == 0 or value <= 0:
            raise ArithmeticError("Invalid far-node replacement")
        replace(index, point, "far_replace", value)
    if np.max(np.linalg.norm(geometry.points, axis=1)) > 2:
        raise ArithmeticError("Locality repair did not finish")

    for _ in range(max_improvements + 1):
        maxima = [ball_maximum(m) for m in geometry.lagrange()[1:]]
        offset = int(np.argmax([v for v, _ in maxima]))
        value, point = maxima[offset]
        if value <= threshold:
            break
        replace(offset + 1, point, "poised_replace", value)
    else:
        raise ArithmeticError("Poisedness exchange bound exceeded")

    if (archive_points is None) != (archive_ids is None):
        raise ValueError("Cache coordinates and IDs must be supplied together")
    if archive_points is not None:
        archive = np.asarray(archive_points, dtype=float).reshape(-1, d)
        if len(archive) != len(archive_ids) or None in archive_ids:
            raise ValueError("Invalid evaluated archive")
        for i, point in enumerate(geometry.points):
            if current_ids[i] is None:
                matches = [j for j, old in enumerate(archive) if np.array_equal(point, old)]
                if matches:
                    current_ids[i] = archive_ids[matches[0]]
    events.append({"kind": "ready", "unknown": current_ids.count(None),
                   "maximum": float(max(v for v, _ in maxima))})
    return Selection(geometry.points.copy(), current_ids, events)
