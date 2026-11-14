"""Selected regional MFN with persistent scale and incremental weighted reuse."""
from dataclasses import asdict
from .frozen_regional import options as fixed_options
from .development_solver import Options, solve

def options(dimension, *, max_evaluations=1000, max_transitions=5000,
            lipschitz_bound=None, epsilon=.01, roundoff_guard=True, fast_extrema=True):
    base=asdict(fixed_options(dimension,max_evaluations=max_evaluations,
              max_transitions=max_transitions,lipschitz_bound=lipschitz_bound,epsilon=epsilon))
    return Options(**base,persistent_sampling=True,fast_algebra=True,
                   growing_set=False,trial_scale_limit=1024.,roundoff_guard=roundoff_guard,
                   fast_extrema=fast_extrema,weighted_reuse_radius=4.,
                   reuse_poisedness=4.,incremental_reuse=True,
                   systematic_repair=False,insertion_rule='original')
