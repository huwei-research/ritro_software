"""Factorization reuse for immutable MFN interpolation geometries."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from .core import Geometry, Model, pack, unpack


class FastGeometry(Geometry):
    def __init__(self, points):
        super().__init__(points)
        self._factor = lu_factor(self.K)
        self._cardinals = None

    def fit(self, values, prior=None):
        prior = np.zeros((self.d, self.d)) if prior is None else np.asarray(prior)
        if prior.shape != (self.d, self.d) or not np.allclose(prior, prior.T):
            raise ValueError('Prior must be symmetric')
        rhs = np.r_[np.asarray(values)-self.Q@pack(prior), np.zeros(self.d+1)]
        s = lu_solve(self._factor, rhs)
        return Model(float(s[self.n]), s[self.n+1:],
                     prior+unpack(self.Q.T@s[:self.n], self.d))

    def lagrange(self):
        if self._cardinals is None:
            s = lu_solve(self._factor, np.vstack([np.eye(self.n),np.zeros((self.d+1,self.n))]))
            self._cardinals = [Model(float(s[self.n,i]),s[self.n+1:,i],
                              unpack(self.Q.T@s[:self.n,i],self.d)) for i in range(self.n)]
        return self._cardinals
