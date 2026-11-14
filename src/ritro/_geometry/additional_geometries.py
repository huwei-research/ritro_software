"""Exact Grassmann projector and Lorentz hyperbolic geometries.

Grassmann metric: one half Frobenius; Bendokat et al., arXiv:2011.13699.
Hyperbolic metric: diag(-1,1,...,1), sectional curvature -1.
All coordinate frames are orthonormal in their respective metrics.
"""
import numpy as np
from scipy.linalg import expm

class Grassmann:
    def __init__(self,n,r):
        self.n=n;self.r=r;self.dimension=r*(n-r);self.size=n*n
    def chart(self,x):return GrassmannChart(self,x)

class GrassmannChart:
    def __init__(self,m,x):
        self.m=m;self.center=np.asarray(x).copy();self.P=self.center.reshape(m.n,m.n)
        np.testing.assert_allclose(self.P,self.P.T,atol=1e-10)
        w,V=np.linalg.eigh(self.P)
        np.testing.assert_allclose(w,np.r_[np.zeros(m.n-m.r),np.ones(m.r)],atol=1e-9)
        self.X=V[:,-m.r:];self.N=V[:,:-m.r]
        self.basis=np.array([np.outer(self.N[:,i],self.X[:,j])+np.outer(self.X[:,j],self.N[:,i])
                            for i in range(m.n-m.r) for j in range(m.r)])
        self.frame=self.basis.reshape(m.dimension,-1).T
    def generator(self,s):
        D=np.einsum('i,ijk->jk',s,self.basis)
        return D@self.P-self.P@D
    def exp(self,s):
        Q=expm(self.generator(s));P=Q@self.P@Q.T
        return ((P+P.T)/2).ravel()
    retr=exp
    def log(self,y):
        if np.array_equal(y,self.center):return np.zeros(self.m.dimension)
        _,V=np.linalg.eigh(np.asarray(y).reshape(self.m.n,self.m.n));Y=V[:,-self.m.r:]
        A=self.X.T@Y
        if np.linalg.svd(A,compute_uv=False).min()<1e-10:raise ValueError('Grassmann cut locus')
        W=np.linalg.solve(A.T,(self.N.T@Y).T).T
        U,w,Vt=np.linalg.svd(W,full_matrices=False)
        return ((U*np.arctan(w))@Vt).ravel()
    def transport_to(self,target):
        Q=expm(self.generator(self.log(target.center)))
        return np.column_stack([.5*target.frame.T@(Q@B@Q.T).ravel() for B in self.basis])

def lorentz(x,y):return -x[0]*y[0]+np.dot(x[1:],y[1:])

class Hyperbolic:
    def __init__(self,d):self.dimension=d;self.size=d+1
    def chart(self,x):return HyperbolicChart(self,x)

class HyperbolicChart:
    def __init__(self,m,x):
        self.m=m;self.center=np.asarray(x).copy();x=self.center
        if not np.all(np.isfinite(x)) or x[0]<=0:raise ValueError('Invalid hyperboloid point')
        if abs(lorentz(x,x)+1)>1e-9*max(1,np.dot(x,x)):raise ValueError('Hyperboloid constraint violation')
        self.frame=np.vstack([x[1:],np.eye(m.dimension)+np.outer(x[1:],x[1:])/(1+x[0])])
    def coordinates(self,v):return self.frame[1:].T@v[1:]-self.frame[0]*v[0]
    def exp(self,s):
        r=np.linalg.norm(s)
        with np.errstate(over='raise',invalid='raise'):
            return np.cosh(r)*self.center+(np.sinh(r)/r if r>1e-8 else 1+r*r/6)*(self.frame@s)
    retr=exp
    def log(self,y):
        if np.array_equal(y,self.center):return np.zeros(self.m.dimension)
        a=-lorentz(self.center,y);v=np.asarray(y)-a*self.center
        coords=self.coordinates(v);norm=np.linalg.norm(coords)
        return coords*(np.arcsinh(norm)/norm if norm>1e-10 else 1.)
    def transport_to(self,target):
        x,y=self.center,target.center
        return np.column_stack([target.coordinates(v+lorentz(y,v)/(1-lorentz(x,y))*(x+y)) for v in self.frame.T])
