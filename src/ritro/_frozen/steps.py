"""Steihaug truncated CG with an exact directional Cauchy safeguard."""
import numpy as np


def truncated_cg(gradient, hessian, radius, relative_tolerance=.01):
    g = np.asarray(gradient, dtype=float)
    B = np.asarray(hessian, dtype=float)
    if (g.ndim != 1 or not len(g) or B.shape != (len(g), len(g))
            or not np.all(np.isfinite(g)) or not np.all(np.isfinite(B))
            or not np.isfinite(radius) or radius <= 0
            or not 0 < relative_tolerance < 1 or not np.allclose(B, B.T)):
        raise ValueError("Invalid truncated-CG inputs")
    norm_g = np.linalg.norm(g)
    if norm_g == 0:
        return np.zeros_like(g), {"reason": "zero_gradient", "iterations": 0,
                                "cauchy_safeguard": False}
    direction = -g / norm_g
    curvature = float(direction @ B @ direction)
    length = min(radius, norm_g / curvature) if curvature > 0 else radius
    cauchy = length * direction

    def decrease(s):
        return float(-g @ s - .5 * s @ B @ s)

    def boundary(s, p):
        pp = float(p @ p)
        sp = float(s @ p)
        remaining = max(0., radius**2 - float(s @ s))
        root = np.sqrt(sp**2 + pp * remaining)
        tau = (remaining / (sp + root) if sp >= 0 and sp + root > 0
               else (-sp + root) / pp)
        return s + tau * p

    s = np.zeros_like(g)
    residual = g.copy()
    p = -residual
    rr = float(residual @ residual)
    reason = "iteration_limit"
    for iteration in range(1, len(g) + 1):
        Bp = B @ p
        pBp = float(p @ Bp)
        if pBp <= 0:
            s = boundary(s, p)
            reason = "negative_curvature"
            break
        alpha = rr / pBp
        candidate = s + alpha * p
        if np.linalg.norm(candidate) >= radius:
            s = boundary(s, p)
            reason = "boundary"
            break
        s = candidate
        residual = residual + alpha * Bp
        new_rr = float(residual @ residual)
        if np.sqrt(new_rr) <= relative_tolerance * norm_g:
            reason = "residual"
            break
        p = -residual + (new_rr / rr) * p
        rr = new_rr
    if np.linalg.norm(s) > radius:
        s = s * (radius / np.linalg.norm(s))
    safe_decrease = decrease(cauchy)
    predicted = decrease(s)
    safeguard = not np.isfinite(predicted) or predicted < safe_decrease
    if safeguard:
        s = cauchy
    if not np.all(np.isfinite(s)) or decrease(s) <= 0:
        raise ArithmeticError("No finite positive model decrease")
    return s, {"reason": reason, "iterations": iteration,
               "cauchy_safeguard": safeguard,
               "cauchy_decrease": safe_decrease, "predicted": decrease(s)}
