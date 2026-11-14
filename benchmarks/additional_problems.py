"""Five additional benchmark families with independent derivative observers."""
import numpy as np
from ritro._geometry.report_problems import Product,Factor
from ritro._geometry.affine_spd import SPD,spectral
from ritro._geometry.additional_geometries import Grassmann,Hyperbolic

SIZES={**{f'sphere_{d}':d for d in [8,16,24]},
       'grassmann_5_2':6,'grassmann_9_2':14,'grassmann_11_3':24,
       'spd_mean_3':6,'spd_mean_5':15,'spd_mean_7':28,
       **{f'hyperbolic_{d}':d for d in [6,14,24]},
       **{f'torus_{d}':d for d in [8,16,24]}}

def problem(name,seed):
    rng=np.random.default_rng(seed)
    if name.startswith('sphere_'):
        d=int(name.split('_')[1]);M=Product([Factor('sphere',d+1)])
        Q=np.linalg.qr(rng.normal(size=(d+1,d+1)))[0];A=(Q*np.linspace(-1,2,d+1))@Q.T
        b=rng.normal(size=d+1)/np.sqrt(d+1);x=rng.normal(size=d+1);x/=np.linalg.norm(x);lam=2.
        def f(z):return float(z@A@z+lam*np.sum(z**4)+b@z)
        def gradient(z):return M.chart(z).frame.T@(2*A@z+4*lam*z**3+b)
        data=dict(A=A,b=b,lam=lam)
    elif name.startswith('grassmann_'):
        n,r=map(int,name.split('_')[1:]);M=Grassmann(n,r)
        Q=np.linalg.qr(rng.normal(size=(n,n)))[0];A=(Q*np.linspace(-1,2,n))@Q.T
        X=np.linalg.qr(rng.normal(size=(n,r)))[0];x=(X@X.T).ravel();c=np.full(n,r/n);lam=.7
        def f(z):
            P=z.reshape(n,n);return float(-np.sum(P*A)+lam*np.sum((np.diag(P)-c)**2))
        def gradient(z):
            G=-A+2*lam*np.diag(np.diag(z.reshape(n,n))-c)
            return M.chart(z).frame.T@G.ravel()
        data=dict(A=A,c=c,lam=lam)
    elif name.startswith('spd_mean_'):
        n=int(name.split('_')[-1]);M=SPD(n)
        samples=[]
        for _ in range(8):
            Q=np.linalg.qr(rng.normal(size=(n,n)))[0];samples.append(((Q*np.exp(rng.uniform(-1.5,1.5,n)))@Q.T).ravel())
        samples=np.array(samples);x=np.eye(n).ravel()
        def f(z):
            chart=M.chart(z);return float(np.mean([np.sum(chart.log(a)**2) for a in samples]))
        def gradient(z):return -2*np.mean([M.chart(z).log(a) for a in samples],axis=0)
        data=dict(samples=samples)
    elif name.startswith('hyperbolic_'):
        d=int(name.split('_')[1]);M=Hyperbolic(d);origin=np.r_[1.,np.zeros(d)];chart=M.chart(origin)
        anchors=np.array([chart.exp(rng.normal(size=d)/np.sqrt(d)) for _ in range(2*d+3)])
        truth=chart.exp(rng.normal(size=d)*.4/np.sqrt(d));x=chart.exp(rng.normal(size=d)*.7/np.sqrt(d))
        # Squared-distance residuals are smooth even at an anchor.
        targets=np.array([np.sum(M.chart(truth).log(a)**2) for a in anchors])+.02*rng.normal(size=len(anchors))
        def f(z):
            logs=np.array([M.chart(z).log(a) for a in anchors]);res=np.sum(logs**2,axis=1)-targets
            return float(np.mean(res**2))
        def gradient(z):
            logs=np.array([M.chart(z).log(a) for a in anchors]);res=np.sum(logs**2,axis=1)-targets
            return -4*np.mean(res[:,None]*logs,axis=0)
        data=dict(anchors=anchors,targets=targets,truth=truth)
    elif name.startswith('torus_'):
        d=int(name.split('_')[1]);M=Product([Factor('sphere',2) for _ in range(d)])
        truth=rng.uniform(-np.pi,np.pi,d);angles=rng.uniform(-np.pi,np.pi,d);x=np.column_stack([np.cos(angles),np.sin(angles)]).ravel()
        edges=np.array([(i,(i+1)%d) for i in range(d)]+[(i,(i+3)%d) for i in range(d)])
        phase=truth[edges[:,0]]-truth[edges[:,1]]+.1*rng.normal(size=len(edges));anchor=truth[0]
        def angle(z):return np.arctan2(z.reshape(d,2)[:,1],z.reshape(d,2)[:,0])
        def f(z):
            theta=angle(z);v=theta[edges[:,0]]-theta[edges[:,1]]-phase
            return float(np.mean(1-np.cos(v))+.1*(1-np.cos(theta[0]-anchor)))
        def gradient(z):
            theta=angle(z);v=np.sin(theta[edges[:,0]]-theta[edges[:,1]]-phase)/len(edges);g=np.zeros(d)
            np.add.at(g,edges[:,0],v);np.add.at(g,edges[:,1],-v);g[0]+=.1*np.sin(theta[0]-anchor)
            ambient=(g[:,None]*np.column_stack([-np.sin(theta),np.cos(theta)])).ravel()
            return M.chart(z).frame.T@ambient
        data=dict(edges=edges,phase=phase,anchor=anchor,truth=truth)
    else:raise ValueError(name)
    data.update(x0=x)
    def gn(z):return float(np.linalg.norm(gradient(z)))
    gn.coordinates=gradient
    return M,x,f,gn,data
