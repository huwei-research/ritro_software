"""Independent normalized-coordinate kernels for the transported MFN draft.

No objective callbacks occur in geometry selection. Floating-point checks are
diagnostics, not an implementation of the exact-arithmetic proof certificates.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import brentq


def features(points):
    points = np.asarray(points, dtype=float)
    d = points.shape[1]
    return np.column_stack(
        [0.5 * points[:, i] ** 2 for i in range(d)]
        + [points[:, i] * points[:, j] / np.sqrt(2)
           for i in range(d) for j in range(i + 1, d)]
    )


def pack(matrix):
    d = len(matrix)
    return np.array(list(np.diag(matrix)) +
                    [np.sqrt(2) * matrix[i, j]
                     for i in range(d) for j in range(i + 1, d)])


def unpack(vector, d):
    matrix = np.diag(vector[:d]).copy()
    k = d
    for i in range(d):
        for j in range(i + 1, d):
            matrix[i, j] = matrix[j, i] = vector[k] / np.sqrt(2)
            k += 1
    return matrix


@dataclass
class Model:
    constant: float
    gradient: np.ndarray
    hessian: np.ndarray

    def values(self, points):
        return (self.constant + np.asarray(points) @ self.gradient
                + features(points) @ pack(self.hessian))


class Geometry:
    def __init__(self, points):
        self.points = np.asarray(points, dtype=float).copy()
        self.n, self.d = self.points.shape
        if not np.all(np.isfinite(self.points)):
            raise ValueError("Nonfinite coordinates")
        self.Q = features(self.points)
        self.L = np.column_stack([np.ones(self.n), self.points])
        self.K = np.block([[self.Q @ self.Q.T, self.L],
                           [self.L.T, np.zeros((self.d + 1, self.d + 1))]])
        if np.linalg.matrix_rank(self.K) != len(self.K):
            raise np.linalg.LinAlgError("Numerically singular MFN geometry")
        sign, self.logdet = np.linalg.slogdet(self.K)
        if sign == 0:
            raise np.linalg.LinAlgError("Singular MFN geometry")

    def fit(self, values, prior=None):
        prior = np.zeros((self.d, self.d)) if prior is None else np.asarray(prior)
        if prior.shape != (self.d, self.d) or not np.allclose(prior, prior.T):
            raise ValueError("Prior must be symmetric")
        rhs = np.r_[np.asarray(values) - self.Q @ pack(prior),
                    np.zeros(self.d + 1)]
        solution = np.linalg.solve(self.K, rhs)
        correction = unpack(self.Q.T @ solution[:self.n], self.d)
        return Model(float(solution[self.n]), solution[self.n + 1:],
                     prior + correction)

    def lagrange(self):
        rhs = np.vstack([np.eye(self.n), np.zeros((self.d + 1, self.n))])
        solution = np.linalg.solve(self.K, rhs)
        return [Model(float(solution[self.n, i]), solution[self.n + 1:, i],
                      unpack(self.Q.T @ solution[:self.n, i], self.d))
                for i in range(self.n)]


def minimum_on_ball(gradient, hessian):
    """Global quadratic unit-ball solver, including the spectral hard case.

    Returns a candidate only after approximate KKT/PSD verification. No ridge
    is added to the model. The multiplier belongs to H + multiplier * I.
    """
    gradient = np.asarray(gradient, dtype=float)
    hessian = np.asarray(hessian, dtype=float)
    scale = max(1., np.linalg.norm(hessian, 2), np.linalg.norm(gradient))
    eig, vectors = np.linalg.eigh(hessian / scale)
    a = vectors.T @ (gradient / scale)
    lower = max(0., -eig[0])
    denominator = eig + lower
    singular = denominator <= 1e-13
    x = np.zeros_like(a)
    x[~singular] = -a[~singular] / denominator[~singular]
    if np.linalg.norm(a[singular]) <= 1e-13 and np.linalg.norm(x) <= 1:
        multiplier = lower
        if lower > 0:
            x[np.flatnonzero(singular)[0]] = np.sqrt(max(0., 1 - x @ x))
    else:
        # Solve for the distance above the spectral pole, avoiding cancellation
        # in eig[0] + multiplier when the solution is close to the hard case.
        def secular(shift):
            den = denominator + shift
            if np.any((den <= 0) & (np.abs(a) > 0)):
                return np.inf
            quotient = np.divide(a, den, out=np.zeros_like(a), where=den > 0)
            return np.linalg.norm(quotient) - 1

        upper = 1.
        while secular(upper) > 0:
            upper *= 2
        shift = brentq(secular, 0., upper, xtol=np.nextafter(0., 1.),
                       rtol=4*np.finfo(float).eps, maxiter=1000)
        multiplier = lower + shift
        x = -a / (denominator + shift)
    point = vectors @ x
    residual = np.linalg.norm((hessian / scale) @ point
                              + gradient / scale + multiplier * point)
    if (residual > 1e-8 or np.linalg.norm(point) > 1 + 1e-8
            or eig[0] + multiplier < -1e-10
            or abs(multiplier * (point @ point - 1)) > 1e-8):
        raise ArithmeticError("Unit-ball KKT verification failed")
    return point


def maximum_absolute(model):
    candidates = [minimum_on_ball(model.gradient, model.hessian),
                  minimum_on_ball(-model.gradient, -model.hessian)]
    values = [abs(model.values(s[None, :])[0]) for s in candidates]
    best = int(np.argmax(values))
    return float(values[best]), candidates[best]


def canonical(d, n):
    points = [np.zeros(d)] + list(np.eye(d)) + list(-np.eye(d))
    points += [(np.eye(d)[i] + np.eye(d)[j]) / np.sqrt(2)
               for i in range(d) for j in range(i + 1, d)]
    if not d + 1 <= n <= len(points):
        raise ValueError("Invalid interpolation size")
    return np.array(points[:n])


def is_cached(point, archive):
    # Approximate equality must not silently substitute an objective value.
    return any(np.array_equal(point, old) for old in archive)


def select_geometry(candidate, archive, d, n, threshold=2.):
    """Select normalized coordinates; return points and an event trace.

    Archive coordinates represent known values. No objective or values are
    accepted here. Exact coordinate matching is intentionally conservative.
    """
    if threshold <= 1:
        raise ValueError("Poisedness threshold must exceed one")
    archive = np.asarray(archive, dtype=float).reshape(-1, d)
    candidate = np.asarray(candidate, dtype=float).reshape(-1, d)
    trace = []
    geometry = None
    if (len(candidate) == n and np.array_equal(candidate[0], np.zeros(d))
            and np.max(np.linalg.norm(candidate, axis=1)) <= 2):
        try:
            geometry = Geometry(candidate)
            if max(maximum_absolute(m)[0] for m in geometry.lagrange()[1:]) <= threshold:
                return candidate.copy(), ["reuse"]
        except np.linalg.LinAlgError:
            geometry = None
    log_floor = -max(0, n - 2 * d - 1) * np.log(8.)
    if (geometry is None or np.max(np.linalg.norm(candidate, axis=1)) > 1
            or geometry.logdet < log_floor):
        geometry = Geometry(canonical(d, n))
        trace.append("canonical_seed")
    local_archive = [v for v in archive if np.linalg.norm(v) <= 1]
    while True:
        changed = False
        for i in range(1, n):
            if is_cached(geometry.points[i], archive):
                continue
            for point in local_archive:
                if is_cached(point, geometry.points):
                    continue
                trial = geometry.points.copy()
                trial[i] = point
                try:
                    replacement = Geometry(trial)
                except np.linalg.LinAlgError:
                    continue
                if replacement.logdet >= geometry.logdet:
                    geometry = replacement
                    trace.append("archive_exchange")
                    changed = True
                    break
            if changed:
                break
        if not changed:
            break
    inner = np.sqrt(threshold)
    log_ceiling = (4 * n * np.log(2) + n / 2 * np.log(n / 16 + 2)
                   + (d + 1) / 2 * np.log(n))
    limit = int(np.ceil((log_ceiling - log_floor) / (2 * np.log(inner)))) + 5
    for _ in range(limit):
        maxima = [maximum_absolute(m) for m in geometry.lagrange()[1:]]
        i = int(np.argmax([item[0] for item in maxima]))
        value, point = maxima[i]
        if value <= inner:
            return geometry.points.copy(), trace
        trial = geometry.points.copy()
        trial[i + 1] = point
        replacement = Geometry(trial)
        if replacement.logdet <= geometry.logdet:
            raise ArithmeticError("Geometry exchange failed volume increase")
        geometry = replacement
        trace.append("lagrange_exchange")
    raise ArithmeticError("Geometry exchange budget exceeded")
