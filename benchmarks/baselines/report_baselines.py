"""Python algorithm ports checked against pinned author source.

RFD: Algorithms 1/2, sigma0=1,tau0=100,tol=1e-5, inner cap 500.
RDSE-SB: projected ambient signed coordinate polling, author extrapolation
constants. Source: DamianoZeffiro/riemannian-ds (GPL-3.0-or-later),
smooth/linesearchext.m; this derivative port is GPL-3.0-or-later.
Strict external ledger stops BEFORE an over-budget objective query.
"""
import numpy as np

def rfd(M,x,oracle,extrinsic=False,tol=1e-5,*,sigma0=1.,tau0=100.,random_seed=0):
    if min(sigma0,tau0,tol)<=0:raise ValueError('Positive RFD parameters required')
    rng=np.random.default_rng(random_seed);sigma=float(sigma0);tau=float(tau0);d=M.dimension
    x=x.copy();f0=oracle(x,True);retr=0
    while True:
        chart=M.chart(x)
        # Project ambient Gaussian vectors then orthonormalize, as the authors.
        E=np.linalg.qr(chart.frame.T@rng.normal(size=(len(x),d)))[0]
        h=2*tol/(5*tau*np.sqrt(d));refresh=True
        for _ in range(500):
            if refresh:
                g=np.zeros(d)
                for j in range(d):
                    if extrinsic:y=x+chart.frame@(h*E[:,j])
                    else:
                        y=chart.retr(h*E[:,j]);retr+=1
                    oracle.retractions=retr
                    g+=(oracle(y,not extrinsic)-f0)/h*E[:,j]
            if not refresh or np.linalg.norm(g)>.8*tol:
                y=chart.retr(-g/sigma);retr+=1;oracle.retractions=retr
                fy=oracle(y,True)
                if f0-fy>=g@g/(4*sigma):
                    x,f0=y,fy;sigma/=2;break
                sigma*=2
                if tau<sigma:
                    tau*=2;h/=2;refresh=True
                else:refresh=False
            else:tau*=2;h/=2;refresh=True

def rdse(M,x,oracle,*,alpha0=1.,sufficient_decrease=.11,contraction=.81,extrapolation=.32):
    if alpha0<=0 or sufficient_decrease<=0 or not 0<contraction<1 or not 0<extrapolation<1:
        raise ValueError('Invalid RDSE parameters')
    x=x.copy();fbest=oracle(x,True);alpha=np.full(len(x),float(alpha0));retr=0
    while True:
        for index in M.ambient_order:
            chart=M.chart(x);direction=chart.frame[index,:].copy()
            for sign in [1.,-1.]:
                v=sign*direction;y=chart.retr(alpha[index]*v);retr+=1;oracle.retractions=retr
                fy=oracle(y,True)
                if fy<fbest-sufficient_decrease*alpha[index]**2:
                    previous=fy
                    while True:
                        alpha[index]/=extrapolation
                        y=chart.retr(alpha[index]*v);retr+=1;oracle.retractions=retr
                        fy=oracle(y,True)
                        if fy>=fbest-sufficient_decrease*alpha[index]**2 or fy>=previous:
                            alpha[index]*=extrapolation;break
                        previous=fy
                    x=chart.retr(alpha[index]*v);retr+=1;oracle.retractions=retr
                    fbest=previous;break
                alpha[index]*=contraction
