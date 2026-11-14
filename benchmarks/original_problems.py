"""The four original benchmark families, extracted without arithmetic changes."""
import numpy as np
from ritro._geometry.report_problems import Product
from ritro.manifolds import Stiefel
from dictionary import problem as dictionary

SIZES = {'singular_3_3_2': 6, 'dictionary_3_6_1': 8,
         'rotation_3': 9, 'procrustes_6_2': 9,
         'singular_5_5_2': 14, 'dictionary_3_5_2': 14,
         'procrustes_10_2': 17, 'rotation_6': 18,
         'singular_7_7_2': 22, 'procrustes_10_3': 24,
         'dictionary_4_6_3': 27, 'rotation_10': 30}

def get_problem(name, seed):
    if name.startswith('dictionary_'):
        return dictionary(*map(int, name.split('_')[1:]), seed)
    rng = np.random.default_rng(seed)
    def point(n, q):
        Q = np.linalg.qr(rng.normal(size=(n, q)))[0]
        if n == q:
            Q[:, -1] *= np.linalg.det(Q)
        return Q
    if name.startswith('singular_'):
        n, m, q = map(int, name.split('_')[1:])
        M = Product([Stiefel(n, q), Stiefel(m, q)])
        A = rng.normal(size=(n, m))
        x = np.r_[point(n,q).ravel(), point(m,q).ravel()]
        def f(z):
            return float(-np.trace(z[:n*q].reshape(n,q).T@A@z[n*q:].reshape(m,q)))
        def eg(z):
            return np.r_[(-A@z[n*q:].reshape(m,q)).ravel(), (-A.T@z[:n*q].reshape(n,q)).ravel()]
        data = dict(A=A, x0=x)
    elif name.startswith('procrustes_'):
        n, q = map(int, name.split('_')[1:])
        M = Product([Stiefel(n,q)])
        A = rng.normal(size=(2*n,n)); B = rng.normal(size=(2*n,q))
        x = point(n,q).ravel()
        def f(z): return float(np.sum((A@z.reshape(n,q)-B)**2))
        def eg(z): return (2*A.T@(A@z.reshape(n,q)-B)).ravel()
        data = dict(A=A, B=B, x0=x)
    elif name.startswith('rotation_'):
        count = int(name.split('_')[1])
        M = Product([Stiefel(3,3) for _ in range(count)])
        truth = np.array([point(3,3) for _ in range(count)])
        edges = np.array([(i,j) for i in range(count) for j in range(i)])
        H = np.array([truth[i]@truth[j].T+.01*rng.normal(size=(3,3)) for i,j in edges])
        x = np.concatenate([point(3,3).ravel() for _ in range(count)])
        def f(z):
            Z = z.reshape(count,3,3)
            return float(sum(np.sum((Z[i]-h@Z[j])**2) for (i,j),h in zip(edges,H)))
        def eg(z):
            Z=z.reshape(count,3,3); G=np.zeros_like(Z)
            for (i,j),h in zip(edges,H):
                E=Z[i]-h@Z[j]; G[i]+=2*E; G[j]-=2*h.T@E
            return G.ravel()
        data=dict(truth=truth, edges=edges, H=H, x0=x)
    else:
        raise ValueError(name)
    def gn(z): return float(np.linalg.norm(M.chart(z).frame.T@eg(z)))
    return M,x,f,gn,data
