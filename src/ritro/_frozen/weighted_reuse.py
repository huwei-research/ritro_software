"""Optional all-paid-node retention under explicit accuracy coefficients."""
import numpy as np
from .maintenance import Selection, maintain


def _geometry_data(points, geometry_factory, ball_maximum):
    geometry=geometry_factory(points)
    cards=geometry.lagrange()[1:]
    maxima=[ball_maximum(m) for m in cards]
    rr=np.sum(np.asarray(points)[1:]**2,axis=1)
    metrics=dict(radius=float(np.max(np.linalg.norm(points,axis=1))),
                maximum=float(max(v for v,_ in maxima)),
                weighted_gradient=float(sum(r*np.linalg.norm(m.gradient) for r,m in zip(rr,cards))),
                weighted_hessian=float(sum(r*np.linalg.norm(m.hessian,2) for r,m in zip(rr,cards))))
    return metrics,cards,maxima


def measures(points, geometry_factory, ball_maximum):
    return _geometry_data(points,geometry_factory,ball_maximum)[0]


def select(points,ids,d,n,threshold,outer_radius,geometry_factory,ball_maximum,
           reuse_threshold=None,incremental=False,exchange_policy=None):
    points=np.asarray(points,dtype=float).reshape(-1,d)
    reuse_threshold=threshold if reuse_threshold is None else reuse_threshold
    def passed(a):
        return (a['radius']<=outer_radius and a['maximum']<=reuse_threshold and
                a['weighted_gradient']<=4*(n-1)*threshold and
                a['weighted_hessian']<=8*(n-1)*threshold)
    event=None
    if (len(points)==n and len(ids)==n and None not in ids and
            len(set(ids))==n and np.array_equal(points[0],np.zeros(d)) and
            np.all(np.isfinite(points))):
        event=dict(kind='weighted_reuse_check',accepted=False)
        try:
            metrics,cards,maxima=_geometry_data(points,geometry_factory,ball_maximum)
            event.update(metrics)
            event['accepted']=bool(passed(metrics))
            if event['accepted']:
                return Selection(points.copy(),list(ids),[event,dict(kind='ready',unknown=0,maximum=metrics['maximum'])])
            if incremental:
                candidate=points.copy();candidate_ids=list(ids);steps=[]
                # Bounded virtual exchanges, before paying for any new values.
                for _ in range(n-1):
                    radii=np.linalg.norm(candidate,axis=1)
                    if max(radii)>outer_radius:
                        index=int(np.argmax(radii))
                    elif max(v for v,_ in maxima)>reuse_threshold:
                        index=1+int(np.argmax([v for v,_ in maxima]))
                    else:
                        scores=[r*r*(np.linalg.norm(m.gradient)/(4*(n-1)*threshold)+
                                     np.linalg.norm(m.hessian,2)/(8*(n-1)*threshold))
                                for r,m in zip(radii[1:],cards)]
                        index=1+int(np.argmax(scores))
                    old_id=candidate_ids[index]
                    node,node_id,extra=maxima[index-1][1],None,None
                    if exchange_policy is not None:
                        node,node_id,extra=exchange_policy(candidate,candidate_ids,index,node)
                    candidate[index]=node;candidate_ids[index]=node_id
                    metrics,cards,maxima=_geometry_data(candidate,geometry_factory,ball_maximum)
                    steps.append(dict(kind='weighted_virtual_exchange',index=index,removed_id=old_id))
                    if extra is not None:steps.append(extra)
                    if passed(metrics):
                        return Selection(candidate,candidate_ids,[event]+steps+[
                            dict(kind='weighted_partial_reuse',accepted=True,**metrics),
                            dict(kind='ready',unknown=candidate_ids.count(None),maximum=metrics['maximum'])])
                event['virtual_exchanges_abandoned']=len(steps)
        except np.linalg.LinAlgError:
            event['reason']='singular_candidate'
    result=maintain(points,ids,d,n,threshold,geometry_factory=geometry_factory,ball_maximum=ball_maximum)
    if event is not None:result.events.insert(0,event)
    return result
