"""Fixed regional-memory algorithm, version 1.1; no tuning switches."""
from .solver import Options


def options(dimension, *, max_evaluations=1000, max_transitions=300,
            lipschitz_bound=None, epsilon=.01):
    """Fixed-b experimental configuration; dimension theorem uses b=b0/(n-1)."""
    return Options(nodes=2*dimension+1, initial_radius=.2, maximum_radius=.3,
                   poisedness=2., criticality=1., acceptance=.1,
                   contraction=.5, expansion=2., expansion_acceptance=.7,
                   expansion_boundary_fraction=.8, transported_hessian_prior=True,
                   memory_region_multiplier=4., memory_curvature_floor=1.,
                   model_scale_ratio=1., reuse_points=True,
                   reuse_rejected_trial=False, same_center_hessian_prior=False,
                   max_evaluations=max_evaluations, max_transitions=max_transitions,
                   lipschitz_bound=lipschitz_bound, epsilon=epsilon)
