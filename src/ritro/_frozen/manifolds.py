"""Orthonormal tangent-coordinate charts for the isolated prototype."""
import numpy as np


class Euclidean:
    def __init__(self, dimension):
        self.dimension = dimension

    def chart(self, point):
        center = np.asarray(point, dtype=float).copy()
        if center.shape != (self.dimension,):
            raise ValueError("Invalid Euclidean point")
        return _EuclideanChart(center)


class _EuclideanChart:
    def __init__(self, center):
        self.center = center

    def transport_to(self, target):
        return np.eye(len(self.center))

    def exp(self, vector):
        return self.center + vector

    def log(self, point):
        return np.asarray(point) - self.center


class Sphere:
    def __init__(self, dimension):
        if dimension < 1:
            raise ValueError("Sphere dimension must be positive")
        self.dimension = dimension

    def chart(self, point):
        point = np.asarray(point, dtype=float)
        if point.shape != (self.dimension + 1,):
            raise ValueError("Invalid sphere point dimension")
        if not np.isclose(np.linalg.norm(point), 1., atol=1e-12, rtol=0):
            raise ValueError("Point is not on the unit sphere")
        return _SphereChart(point)


class _SphereChart:
    def __init__(self, center):
        self.center = center.copy()
        self.frame = np.linalg.svd(center[None, :], full_matrices=True)[2][1:].T

    def transport_to(self, target):
        denominator = 1. + self.center @ target.center
        if denominator < 1e-12:
            raise ValueError("Parallel transport ambiguous near antipodes")
        transported = self.frame - np.outer(self.center + target.center,
                                            target.center @ self.frame)/denominator
        return target.frame.T @ transported

    def exp(self, vector):
        radius = np.linalg.norm(vector)
        if radius >= np.pi:
            raise ValueError("Exponential input outside chosen normal domain")
        if radius == 0:
            return self.center.copy()
        return np.cos(radius) * self.center + np.sinc(radius / np.pi) * (self.frame @ vector)

    def log(self, point):
        point = np.asarray(point, dtype=float)
        if np.array_equal(point, self.center):
            return np.zeros(self.frame.shape[1])
        if not np.isclose(np.linalg.norm(point), 1., atol=1e-12, rtol=0):
            raise ValueError("Log target is not on the sphere")
        cosine = float(np.clip(self.center @ point, -1., 1.))
        tangent = point - cosine * self.center
        sine = np.linalg.norm(tangent)
        if cosine < 0 and sine < 1e-12:
            raise ValueError("Sphere Log is ambiguous at the antipode")
        if sine == 0:
            return np.zeros(self.frame.shape[1])
        return (np.arctan2(sine, cosine) / sine) * (self.frame.T @ tangent)


class ProductSpheres:
    """Product metric; flattened physical and tangent coordinates."""
    def __init__(self, dimensions):
        self.factors = [Sphere(d) for d in dimensions]
        self.dimension = sum(dimensions)
        self.physical_offsets = np.cumsum([0] + [d + 1 for d in dimensions])
        self.tangent_offsets = np.cumsum([0] + list(dimensions))

    def chart(self, point):
        point = np.asarray(point)
        if point.shape != (self.physical_offsets[-1],):
            raise ValueError("Invalid product point")
        charts = [factor.chart(point[a:b]) for factor, a, b in zip(
            self.factors, self.physical_offsets[:-1], self.physical_offsets[1:])]
        return _ProductChart(charts, self.physical_offsets, self.tangent_offsets)


class _ProductChart:
    def __init__(self, charts, physical_offsets, tangent_offsets):
        self.charts, self.p, self.t = charts, physical_offsets, tangent_offsets

    def transport_to(self, target):
        result = np.zeros((self.t[-1], self.t[-1]))
        for source, dest, a, b in zip(self.charts, target.charts, self.t[:-1], self.t[1:]):
            result[a:b, a:b] = source.transport_to(dest)
        return result

    def exp(self, vector):
        return np.concatenate([c.exp(vector[a:b]) for c, a, b in zip(
            self.charts, self.t[:-1], self.t[1:])])

    def log(self, point):
        return np.concatenate([c.log(point[a:b]) for c, a, b in zip(
            self.charts, self.p[:-1], self.p[1:])])
