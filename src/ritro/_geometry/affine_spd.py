"""Affine-invariant SPD geometry in orthonormal whitened coordinates.

g_X(U,V)=tr(X^-1 U X^-1 V). No spectral clipping or ridge.
"""
import numpy as np

def sym(A):return (A+A.T)/2
def spectral(A,fn,positive=True):
    w,Q=np.linalg.eigh(sym(A))
    if positive and np.min(w)<=0:raise ValueError('SPD domain violated')
    result=sym((Q*fn(w))@Q.T)
    if not np.all(np.isfinite(result)):raise FloatingPointError('Nonfinite matrix function')
    return result

class SPD:
    def __init__(self,n=3):
        self.n=n;self.size=n*n;self.dimension=n*(n+1)//2
        B=[]
        for i in range(n):
            A=np.zeros((n,n));A[i,i]=1;B.append(A)
        for i in range(n):
            for j in range(i+1,n):
                A=np.zeros((n,n));A[i,j]=A[j,i]=1/np.sqrt(2);B.append(A)
        self.basis=np.array(B)
    def chart(self,x):return SPDChart(self,x)

class SPDChart:
    def __init__(self,m,x):
        self.m=m;self.center=np.asarray(x).copy();X=self.center.reshape(m.n,m.n)
        np.testing.assert_allclose(X,X.T,atol=1e-10,rtol=1e-12)
        self.half=spectral(X,np.sqrt);self.inv=spectral(X,lambda w:1/np.sqrt(w))
    def matrix(self,s):return np.einsum('i,ijk->jk',s,self.m.basis)
    def coordinates(self,A):return np.einsum('ijk,jk->i',self.m.basis,A)
    def exp(self,s):
        return sym(self.half@spectral(self.matrix(s),np.exp,False)@self.half).ravel()
    def log(self,y):
        if np.array_equal(y,self.center):return np.zeros(self.m.dimension)
        return self.coordinates(spectral(self.inv@np.asarray(y).reshape(self.m.n,self.m.n)@self.inv,np.log))
    def retr(self,s):return self.exp(s)
    def transport_to(self,target):
        # E = X^1/2 (X^-1/2 Y X^-1/2)^1/2 X^-1/2.
        # Whitened representation at Y is Q S Q^T, Q orthogonal.
        C=self.inv@target.center.reshape(self.m.n,self.m.n)@self.inv
        Q=target.inv@self.half@spectral(C,np.sqrt)
        return np.column_stack([target.coordinates(Q@B@Q.T) for B in self.m.basis])
