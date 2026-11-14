# Solver settings and records

`default_options(d)` returns the frozen options used by the manuscript's
default RITRO configuration:

| Setting | Value |
|---|---:|
| Interpolation points | `2*d+1` |
| Initial / maximum trial radius | `0.2 / 0.3` |
| Base / reuse poisedness thresholds | `2 / 4` |
| Outer reuse radius multiplier | `4` |
| Regional model multiplier | `4` |
| Criticality multiplier | `1` |
| Trial-to-sampling scale limit | `1024` |
| Acceptance / expansion ratio threshold | `0.1 / 0.7` |
| Contraction / expansion factor | `0.5 / 2` |
| Expansion boundary fraction | `0.8` |

The paper's selected-parameter benchmark changes one parameter at a time
using independent development seeds. The defaults above remain the public
solver defaults. The theoretical dimension-scaling discussion also considers
a dimension-dependent criticality multiplier; that setting differs from the
fixed default tested in the paper.

The `Options` dataclass retains the original source fields for reproducibility.
The release entry `default_options` selects the frozen method. Changing fields
that select experimental branches produces a different algorithm configuration.

Every distinct objective query in the internal archive is counted. An exact
array-equality cache avoids duplicate queries; it has no geometric tolerance.
The result includes:

| Key | Meaning |
|---|---|
| `point`, `value` | Last accepted centre and its known value |
| `best_point`, `best_value` | Best of all observations, added by `minimize` |
| `evaluations` | Number of objective calls |
| `status`, `failure` | Termination reason and numerical-error information |
| `history` | Per-reconstruction / trial diagnostics |
| `archive_points`, `archive_values` | All observed physical points and values |
| `evaluation_sources` | Initial, geometry and trial query classification |
| `map_calls`, `transport_calls` | Counted geometric operations |

`certificate` requires the user to supply a valid pullback remainder bound
through `lipschitz_bound`. In ordinary black-box use this argument is `None`.
`evaluation_limit`, `transition_limit`, `sampling_resolution_limit` and
`numerical_error` remain distinct statuses. The numerical resolution rule
does not require an extra objective query.

Public entry points validate dimensions, model-point counts and budgets as integers before any objective call; bool, NaN and fractional budgets are rejected. Initial coordinates must be finite real vectors, and objective values must be real scalars. These checks wrap the unchanged frozen solver.
