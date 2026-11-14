"""Metric-consistent adaptations of the existing RFD and RDSE rules.

RDSE rule port is GPL-3.0-or-later, derived from report_baselines.py and
DamianoZeffiro/riemannian-ds smooth/linesearchext.m (see vendor source).

Use affine-orthonormal coordinates; no Euclidean projection masquerading as
an affine-metric projection. RDSE uses signed whitened symmetric basis polling.
"""
import numpy as np

def rfd(M,x,oracle,tol=1e-5,*,sigma0=1.,tau0=100.,random_seed=0):
    if min(tol,sigma0,tau0)<=0:raise ValueError('Positive parameters required')
    rng=np.random.default_rng(random_seed);sigma=float(sigma0);tau=float(tau0);d=M.dimension
    f0=oracle(x,True)
    while True:
        chart=M.chart(x);E=np.linalg.qr(rng.normal(size=(d,d)))[0]
        h=2*tol/(5*tau*np.sqrt(d));refresh=True
        for _ in range(500):
            if refresh:
                g=np.zeros(d)
                for j in range(d):g+=(oracle(chart.retr(h*E[:,j]),True)-f0)/h*E[:,j]
            if not refresh or np.linalg.norm(g)>.8*tol:
                y=chart.retr(-g/sigma);fy=oracle(y,True)
                if f0-fy>=g@g/(4*sigma):x,f0=y,fy;sigma/=2;break
                sigma*=2
                if tau<sigma:tau*=2;h/=2;refresh=True
                else:refresh=False
            else:tau*=2;h/=2;refresh=True

def rdse(M,x,oracle,*,alpha0=1.,sufficient_decrease=.11,contraction=.81,extrapolation=.32):
    if alpha0<=0 or sufficient_decrease<=0 or not 0<contraction<1 or not 0<extrapolation<1:
        raise ValueError('Invalid parameters')
    fbest=oracle(x,True);alpha=np.full(M.dimension,float(alpha0))
    while True:
        for index in range(M.dimension):
            chart=M.chart(x)
            for sign in [1.,-1.]:
                v=np.eye(M.dimension)[index]*sign
                fy=oracle(chart.retr(alpha[index]*v),True)
                if fy<fbest-sufficient_decrease*alpha[index]**2:
                    previous=fy
                    while True:
                        alpha[index]/=extrapolation
                        fy=oracle(chart.retr(alpha[index]*v),True)
                        if fy>=fbest-sufficient_decrease*alpha[index]**2 or fy>=previous:alpha[index]*=extrapolation;break
                        previous=fy
                    x=chart.retr(alpha[index]*v);fbest=previous;break
                alpha[index]*=contraction
