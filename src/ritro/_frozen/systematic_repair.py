"""Finite candidate repair with potential progress and whole-path cost comparison."""
import numpy as np
from .maintenance import potential
from .weighted_reuse import select


def curvature_gain(geometry):
    """Smallest observable curvature singular value, after affine elimination."""
    U,_,_=np.linalg.svd(geometry.L,full_matrices=True)
    A=U[:,geometry.d+1:].T@geometry.Q
    return float(np.linalg.svd(A,compute_uv=False)[-1]) if len(A) else 0.


def repair(points, ids, d, n, threshold, outer_radius, factory, maximum,
           reuse_threshold, archive, predictor=None, use_directions=True,score_rule='prediction'):
    if score_rule not in ('prediction','curvature'):raise ValueError('Unknown repair score')
    args=(points,ids,d,n,threshold,outer_radius,factory,maximum,reuse_threshold,True)
    baseline=select(*args)
    if len(points)!=n or baseline.ids.count(None)==0:
        return baseline
    def policy(current, current_ids, index, reference):
        old=factory(current)
        h0=float(np.log(np.maximum(1.,np.linalg.norm(current,axis=1))).sum())
        v0=potential(old)
        far=np.linalg.norm(current[index])>np.sqrt(2.)
        candidates=[(np.asarray(u),key,'archive') for key,u in archive if key not in current_ids]
        if use_directions:
            directions=[current[index]]
            if predictor is not None:directions.append(predictor.gradient)
            card=old.lagrange()[index]
            for direction in directions:
                length=np.linalg.norm(direction)
                if length==0:continue
                direction=direction/length
                aa=.5*direction@card.hessian@direction
                bb=card.gradient@direction
                scalars=[-1.,1.]
                if abs(aa)>1e-15 and abs(bb/(2*aa))<1:scalars.append(-bb/(2*aa))
                candidates.extend((t*direction,None,'direction') for t in scalars)
        reference_score=float(predictor.values(np.asarray(reference)[None,:])[0]) if predictor is not None else 0.
        if score_rule=='curvature':
            ref_points=current.copy();ref_points[index]=reference
            reference_score=-curvature_gain(factory(ref_points))
        # The original geometry point competes with all unpaid alternatives.
        feasible=[((True,reference_score,0.),reference,None,'reference',None,None)]
        for u,key,kind in candidates:
            if not np.all(np.isfinite(u)) or np.linalg.norm(u)>outer_radius:continue
            proposed=current.copy();proposed[index]=u
            try:new=factory(proposed)
            except np.linalg.LinAlgError:continue
            h=float(np.log(np.maximum(1.,np.linalg.norm(proposed,axis=1))).sum())
            dv=potential(new)-v0;dh=h0-h
            # Quarter of the original guaranteed progress; both potentials monotone.
            sufficient=(dh>=.25*np.log(np.sqrt(2.)) if far else dv>=.5*np.log(threshold))
            if dh < 0 or dv < 0 or not sufficient:continue
            score=float(predictor.values(u[None,:])[0]) if predictor is not None else -dv
            if score_rule=='curvature':score=-curvature_gain(new)
            feasible.append(((key is None,score,-dv),u,key,kind,dh,dv))
        if feasible:
            _,u,key,kind,dh,dv=min(feasible,key=lambda item:item[0])
            if kind=='reference':return u,key,dict(kind='systematic_exchange',source='reference')
            return u,key,dict(kind='systematic_exchange',source=kind,delta_h=dh,delta_v=dv,far=bool(far))
        return reference,None,dict(kind='systematic_exchange',source='reference')
    candidate=select(*args,exchange_policy=policy)
    # No extra objective calls: both repair paths are virtual.
    chosen=candidate.ids.count(None)<=baseline.ids.count(None)
    result=candidate if chosen else baseline
    result.events.append(dict(kind='systematic_path',chosen=chosen,
        baseline_unknown=baseline.ids.count(None),candidate_unknown=candidate.ids.count(None),
        archive_candidates=len(archive)))
    return result
