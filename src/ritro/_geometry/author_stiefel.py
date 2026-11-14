"""Euclidean-metric Stiefel geometry via numerical geodesic shooting.

Reference implementation for small tests, with explicit residual checks.
Exp integrates X'=V, V'=-X(V.T V). Parallel fields satisfy
W'=-X sym(V.T W). Log solves the endpoint equation in an orthonormal frame.
"""
import numpy as np
from scipy.linalg import null_space
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares


def sym(A):
    return (A+A.T)/2


class Stiefel:
    def __init__(self,n,q):
        if not 1 <= q <= n:
            raise ValueError('Require 1 <= q <= n')
        self.n,self.q=n,q
        self.size=n*q
        self.dimension=n*q-q*(q+1)//2
    def chart(self,x):
        return Chart(self,x)


class Chart:
    def __init__(self,m,x):
        self.m=m;self.center=np.asarray(x).copy();X=self.center.reshape(m.n,m.q)
        if not np.allclose(X.T@X,np.eye(m.q),atol=2e-8):
            raise ValueError('Invalid Stiefel point')
        self.X=X
        constraints=[]
        for i in range(m.q):
            for j in range(i,m.q):
                S=np.zeros((m.q,m.q));S[i,j]+=1;S[j,i]+=1
                constraints.append((X@S).ravel())
        self.frame=null_space(np.array(constraints))
        self.cache={}
    def integrate(self,s,transport=False):
        n,q=self.m.n,self.m.q;N=n*q;d=self.m.dimension
        V=(self.frame@s).reshape(n,q)
        initial=np.r_[self.center,V.ravel(),self.frame.ravel() if transport else []]
        def rhs(t,u):
            X=u[:N].reshape(n,q);V=u[N:2*N].reshape(n,q)
            parts=[V.ravel(),(-X@(V.T@V)).ravel()]
            if transport:
                W=u[2*N:].reshape(N,d)
                parts.append(np.column_stack([(-X@sym(V.T@W[:,j].reshape(n,q))).ravel() for j in range(d)]).ravel())
            return np.concatenate(parts)
        sol=solve_ivp(rhs,(0,1),initial,method='DOP853',rtol=3e-14,atol=1e-15)
        if not sol.success:
            raise ValueError('Geodesic integration failed')
        return sol.y[:,-1]
    def exp(self,s):
        return self.integrate(s)[:self.m.size]
    def retr(self,s):
        U,_,Vt=np.linalg.svd(self.X+(self.frame@s).reshape(self.X.shape),full_matrices=False)
        return (U@Vt).ravel()
    def log(self,y):
        y=np.asarray(y);key=y.tobytes()
        if np.array_equal(y,self.center):return np.zeros(self.m.dimension)
        if key in self.cache:return self.cache[key].copy()
        fit=least_squares(lambda s:self.exp(s)-y,self.frame.T@(y-self.center),
                          xtol=1e-15,ftol=1e-15,gtol=1e-15,max_nfev=60)
        if not fit.success or np.linalg.norm(fit.fun)>2e-9:
            raise ValueError('Stiefel Log endpoint residual exceeds tolerance')
        self.cache[key]=fit.x.copy()
        return fit.x
    def transport_to(self,target):
        s=self.log(target.center);u=self.integrate(s,transport=True)
        W=u[2*self.m.size:].reshape(self.m.size,self.m.dimension)
        T=target.frame.T@W
        if np.linalg.norm(T.T@T-np.eye(self.m.dimension))>2e-8:
            raise ValueError('Parallel transport isometry residual exceeds tolerance')
        return T
