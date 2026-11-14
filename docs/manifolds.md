# Manifold interface

The solver uses intrinsic dimension `d = manifold.dimension`. A call to
`manifold.chart(x)` returns an object representing orthonormal coordinates
on the tangent space at the physical point `x`.

| Operation | Required result |
|---|---|
| `chart.exp(s)` | Physical point reached by the exponential map, for an array `s` of shape `(d,)` |
| `chart.log(y)` | Local logarithm of the physical point `y`, in shape `(d,)` |
| `chart.transport_to(other)` | A `(d,d)` matrix mapping coordinate vectors at this centre to the target chart along the connecting geodesic |

The transport matrix is orthogonal for orthonormal coordinates. The solver
transports a Hessian prior as `P @ B @ P.T`. All required point pairs must lie
in the chosen normal domains. A custom chart should raise an informative
exception at an invalid point or a nonunique logarithm.

Physical points are NumPy arrays; the provided matrix geometries use flattened
row-major arrays. Their intrinsic dimensions are:

| Class | Physical representation | Dimension |
|---|---|---|
| `Euclidean(d)` | Length `d` | `d` |
| `Sphere(d)` | Unit vector of length `d+1` | `d` |
| `ProductSpheres(ds)` | Concatenated unit vectors | `sum(ds)` |
| `Stiefel(n,q)` | Flattened `n` by `q` matrix with orthonormal columns | `n*q-q*(q+1)/2` |
| `Grassmann(n,r)` | Flattened rank-`r` orthogonal projector | `r*(n-r)` |
| `SPD(n)` | Flattened symmetric positive definite matrix | `n*(n+1)/2` |
| `Hyperbolic(d)` | Lorentz hyperboloid vector `(x0,...,xd)`, `x0>0` | `d` |

`Product` and `Factor` reproduce the product representations used by the
benchmark. Stiefel operations use the Euclidean induced metric and numerical
geodesic integration/shooting; SPD operations use the affine-invariant metric.
Grassmann projectors use half the Frobenius inner product. These choices
determine the meaning of radii, norms and transported model coefficients.

The comparison methods additionally call `chart.retr`. Embedded-coordinate
comparison routines use `chart.frame` and `manifold.ambient_order`; the
intrinsic metric adaptations use tangent coordinates directly. These extra
members are unnecessary for the RITRO solver interface.
