"""Explicit, reversible acceleration of frozen coefficient packing kernels.

Ordering and scalar operations match core.py: diagonal first, then strict
upper triangle in row-major order. No optimizer parameter or rule changes.
"""
from functools import lru_cache
import numpy as np


@lru_cache(maxsize=32)
def indices(d):
    return np.triu_indices(d, 1)


def features(points):
    points = np.asarray(points, dtype=float)
    i, j = indices(points.shape[1])
    return np.column_stack((0.5 * points ** 2,
                            points[:, i] * points[:, j] / np.sqrt(2)))


def pack(matrix):
    i, j = indices(len(matrix))
    return np.concatenate((np.diag(matrix), np.sqrt(2) * matrix[i, j]))


def unpack(vector, d):
    matrix = np.diag(vector[:d]).copy()
    i, j = indices(d)
    matrix[i, j] = matrix[j, i] = vector[d:] / np.sqrt(2)
    return matrix


def install():
    """Call after frozen imports; return a restoration callback for checks."""
    from . import core
    from . import fast_core
    saved = []
    for module, names in [(core, ('features', 'pack', 'unpack')),
                          (fast_core, ('pack', 'unpack'))]:
        for name in names:
            saved.append((module, name, getattr(module, name)))
            setattr(module, name, globals()[name])
    def restore():
        for module, name, value in saved:
            setattr(module, name, value)
    return restore
