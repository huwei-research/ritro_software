"""Small report problem instances with exact intrinsic product geometries.

Objective/generator provenance: vendor/report_sources/rdse/smooth/mainsolver.m
and Taminiau et al. section 5.2. NumPy seeds are not MATLAB instances.
"""
import numpy as np
from scipy.linalg import block_diag
from .._frozen.manifolds import Sphere, Euclidean


class Factor:
    def __init__(self, kind, size):
        self.kind,self.size=kind,size
        self.dimension=size-1 if kind=='sphere' else (1 if kind=='so2' else size)
    def chart(self,x):return FactorChart(self,x)

class FactorChart:
    def __init__(self,m,x):
        self.m=m;self.center=np.array(x)
        if m.kind=='sphere':
            self.base=Sphere(m.dimension).chart(x);self.frame=self.base.frame
        elif m.kind=='euclidean':
            self.base=Euclidean(m.dimension).chart(x);self.frame=np.eye(m.dimension)
        else:
            R=self.center.reshape(2,2)
            if not np.allclose(R.T@R,np.eye(2),atol=1e-10) or np.linalg.det(R)<0:raise ValueError('Invalid SO(2) point')
            self.frame=(R@np.array([[0.,-1.],[1.,0.]])).reshape(-1,1)/np.sqrt(2)
    def exp(self,s):
        if self.m.kind!='so2':return self.base.exp(s)
        t=s[0]/np.sqrt(2);Q=np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        return (self.center.reshape(2,2)@Q).ravel()
    def log(self,y):
        if np.array_equal(np.asarray(y),self.center):return np.zeros(self.m.dimension)
        if self.m.kind!='so2':return self.base.log(y)
        Q=self.center.reshape(2,2).T@np.asarray(y).reshape(2,2)
        t=np.arctan2(Q[1,0],Q[0,0])
        if abs(t)>np.pi-1e-10:raise ValueError('SO(2) cut locus')
        return np.array([np.sqrt(2)*t])
    def retr(self,s):
        y=self.center+self.frame@s
        if self.m.kind=='sphere':return y/np.linalg.norm(y)
        if self.m.kind=='euclidean':return y
        return self.exp(np.array([np.sqrt(2)*np.arctan(s[0]/np.sqrt(2))]))
    def transport_to(self,target):
        if self.m.kind=='so2':return np.eye(1)
        return self.base.transport_to(target.base)

class Product:
    def __init__(self,factors,ambient_order=None):
        self.factors=factors;self.dimension=sum(f.dimension for f in factors)
        self.p=np.cumsum([0]+[f.size for f in factors]);self.t=np.cumsum([0]+[f.dimension for f in factors])
        self.ambient_order=list(range(self.p[-1])) if ambient_order is None else ambient_order
    def chart(self,x):return ProductChart(self,np.asarray(x))

class ProductChart:
    def __init__(self,m,x):
        self.m=m;self.center=x.copy()
        self.charts=[f.chart(x[a:b]) for f,a,b in zip(m.factors,m.p[:-1],m.p[1:])]
        self.frame=block_diag(*[c.frame for c in self.charts])
    def exp(self,s):return np.concatenate([c.exp(s[a:b]) for c,a,b in zip(self.charts,self.m.t[:-1],self.m.t[1:])])
    def retr(self,s):return np.concatenate([c.retr(s[a:b]) for c,a,b in zip(self.charts,self.m.t[:-1],self.m.t[1:])])
    def log(self,y):return np.concatenate([c.log(y[a:b]) for c,a,b in zip(self.charts,self.m.p[:-1],self.m.p[1:])])
    def transport_to(self,target):return block_diag(*[a.transport_to(b) for a,b in zip(self.charts,target.charts)])

def problem(name,seed):
    rng=np.random.default_rng(seed)
    def unit(n):
        v=rng.normal(size=n);return v/np.linalg.norm(v)
    if name=='top_singular_2_2_1':
        M=Product([Factor('sphere',2),Factor('sphere',2)])
        A=rng.normal(size=(2,2));shift=np.linalg.svd(A,compute_uv=False)[0]
        x=np.r_[unit(2),unit(2)]
        def f(z):return float(-z[:2]@A@z[2:]+shift)
        def eg(z):return np.r_[-A@z[2:],-A.T@z[:2]]
        data=dict(A=A,x0=x,shift=shift)
    elif name=='procrustes_5_3_1':
        M=Product([Factor('sphere',3)])
        A=rng.normal(size=(5,3));solution=unit(3);B=A@solution
        chart=M.chart(solution);x=chart.retr(chart.frame.T@rng.normal(size=3))
        def f(z):return float(np.sum((A@z-B)**2))
        def eg(z):return 2*A.T@(A@z-B)
        data=dict(A=A,B=B,solution=solution,x0=x)
    elif name=='dictionary_2_3_1':
        M=Product([Factor('sphere',2),Factor('euclidean',3)])
        x=np.r_[unit(2),rng.normal(size=3)];D=unit(2)
        C=rng.uniform(size=3)*(rng.uniform(size=3)<.1);Y=np.outer(D,C)
        reg=.01;delta=.001;shift=reg*np.sum(np.sqrt(C*C+delta*delta))
        def f(z):return float(np.sum((Y-np.outer(z[:2],z[2:]))**2)+reg*np.sum(np.sqrt(z[2:]**2+delta**2))-shift)
        def eg(z):
            E=np.outer(z[:2],z[2:])-Y
            return np.r_[2*E@z[2:],2*E.T@z[:2]+reg*z[2:]/np.sqrt(z[2:]**2+delta**2)]
        data=dict(Y=Y,C0=C,D0=D,x0=x,reg=reg,delta=delta,shift=shift)
    elif name=='rotation_2_2':
        M=Product([Factor('so2',4),Factor('so2',4)],ambient_order=[0,4,1,5,2,6,3,7])
        def rotation():
            t=rng.uniform(-np.pi,np.pi);return np.array([[np.cos(t),-np.sin(t)],[np.sin(t),np.cos(t)]])
        R1,R2=rotation(),rotation();H=R1@R2.T+.01*rng.normal(size=(2,2));x=np.r_[rotation().ravel(),rotation().ravel()]
        def f(z):return float(np.sum((z[:4].reshape(2,2)-H@z[4:].reshape(2,2))**2))
        def eg(z):
            E=z[:4].reshape(2,2)-H@z[4:].reshape(2,2)
            return np.r_[(2*E).ravel(),(-2*H.T@E).ravel()]
        data=dict(H=H,x0=x)
    else:raise ValueError(name)
    def gn(z):return float(np.linalg.norm(M.chart(z).frame.T@eg(z)))
    return M,x,f,gn,data

NAMES=['top_singular_2_2_1','dictionary_2_3_1','rotation_2_2','procrustes_5_3_1']
