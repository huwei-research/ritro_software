"""Equivalent global quadratic extrema using one symmetric eigendecomposition."""
import numpy as np
from scipy.optimize import brentq


def _minimum_spectral(gradient, hessian, eig, vectors):
    a = vectors.T @ gradient
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
    residual = np.linalg.norm((hessian) @ point
                              + gradient + multiplier * point)
    if (residual > 1e-8 or np.linalg.norm(point) > 1 + 1e-8
            or eig[0] + multiplier < -1e-10
            or abs(multiplier * (point @ point - 1)) > 1e-8):
        raise ArithmeticError("Unit-ball KKT verification failed")
    return point


def maximum_absolute(model):
    eig, vectors = np.linalg.eigh(model.hessian)
    scale = max(1., np.max(np.abs(eig)), np.linalg.norm(model.gradient))
    H, g, e = model.hessian/scale, model.gradient/scale, eig/scale
    points = [_minimum_spectral(g,H,e,vectors),
              _minimum_spectral(-g,-H,-e[::-1],vectors[:,::-1])]
    values = [abs(model.constant + model.gradient@s + .5*s@model.hessian@s) for s in points]
    best = int(np.argmax(values))
    return float(values[best]), points[best]
