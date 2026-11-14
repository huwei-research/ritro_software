"""Multi-atom oblique dictionary instances; exact product geometry.

Objective follows RFD section 5.2; sparse noiseless generation follows the
pinned RDSE mainsolver.m. NumPy draws are new instances, not MATLAB replicas.
"""
import numpy as np
from ritro._geometry.report_problems import Product, Factor


def problem(a, b, q, seed, density=.1):
    if min(a, b, q) < 1 or a < 2 or not 0 <= density <= 1:
        raise ValueError('Require a >= 2, positive sizes and density in [0,1]')
    rng = np.random.default_rng(seed)
    def dictionary():
        D = rng.normal(size=(a, q))
        return D / np.linalg.norm(D, axis=0)
    D0 = dictionary()
    C0 = rng.uniform(size=(q, b)) * (rng.uniform(size=(q, b)) < density)
    Y = D0 @ C0
    x = np.r_[dictionary().ravel(order='F'), rng.normal(size=q*b)]
    M = Product([Factor('sphere', a) for _ in range(q)] + [Factor('euclidean', q*b)])
    def unpack(z):
        return z[:a*q].reshape(a, q, order='F'), z[a*q:].reshape(q, b)
    def f(z):
        D, C = unpack(np.asarray(z))
        return float(np.sum((D@C-Y)**2) + .01*np.sum(np.sqrt(C*C+.001**2)))
    def gn(z):
        D, C = unpack(np.asarray(z)); E = D@C-Y
        g = np.r_[(2*E@C.T).ravel(order='F'),
                  (2*D.T@E+.01*C/np.sqrt(C*C+.001**2)).ravel()]
        return float(np.linalg.norm(M.chart(z).frame.T@g))
    return M, x, f, gn, dict(x0=x, D0=D0, C0=C0, Y=Y, density=density)


if __name__ == '__main__':
    for a, b, q in [(3,5,2), (4,6,3), (3,20,12)]:
        M,x,f,gn,data = problem(a,b,q,31)
        assert M.dimension == q*(a-1+b)
        chart=M.chart(x); eye=np.eye(M.dimension)
        fd=np.array([(f(chart.exp(1e-6*v))-f(chart.exp(-1e-6*v)))/2e-6 for v in eye])
        np.testing.assert_allclose(np.linalg.norm(fd),gn(x),rtol=2e-7)
        s=np.linspace(-.01,.01,M.dimension);z=chart.exp(s)
        np.testing.assert_allclose(chart.log(z),s,atol=1e-12)
        T=chart.transport_to(M.chart(z))
        np.testing.assert_allclose(T.T@T,eye,atol=1e-12)
        print(f'{a},{b},{q}: d={M.dimension}, gradient and geometry verified')
