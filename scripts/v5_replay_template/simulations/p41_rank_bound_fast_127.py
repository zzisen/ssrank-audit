"""Vectorized finite-sample rank envelope. Synthetic development only."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata

def envelope(z,w,mask,groups,y,assume_no_ties,unknown_ratio=0.):
    z=np.asarray(z,float);w=np.asarray(w,float);mask=np.asarray(mask);a=np.asarray(groups,float);y=np.asarray(y,float)
    if mask.dtype!=np.bool_ or w.ndim!=1 or mask.shape!=w.shape or z.ndim!=2 or z.shape[1]!=mask.sum() or a.ndim!=2 or a.shape[1]!=z.shape[0] or y.shape!=(a.shape[0],) or len(y)<2 or z.shape[0]<2:raise ValueError('shape')
    if not all(np.isfinite(v).all() for v in (z,w,a,y)) or not np.isfinite(unknown_ratio) or unknown_ratio<0:raise ValueError('nonfinite or invalid budget')
    if not np.allclose(a.sum(axis=1),1,rtol=0,atol=1e-10):raise ValueError('condition weights')
    if z.shape[1] and (not np.allclose(z.mean(0),0,rtol=0,atol=1e-9) or not np.allclose(z.std(0,ddof=1),1,rtol=0,atol=1e-9)):raise ValueError('standardization')
    total=float(np.abs(w).sum())
    if total<=0:raise ValueError('zero known weights')
    m=len(y)
    if len(set(y.tolist()))!=m:return {'status':'unavailable_function_ties','lower':None,'upper':None}
    if not assume_no_ties:return {'status':'unavailable_no_tie_assumption','lower':-1.,'upper':1.}
    missing=float(np.abs(w[~mask]).sum());u=unknown_ratio*total
    v=a@(z@w[mask]);diff=v[:,None]-v[None,:]
    c=a[:,None,:]-a[None,:,:];c=c-c.mean(axis=2,keepdims=True)
    bound=np.sqrt(z.shape[0]-1)*np.linalg.norm(c,axis=2)
    gt=(diff-(missing+u)*bound)/(total+u)>1e-12
    if np.any(np.diag(gt)) or np.any(gt & gt.T):raise ValueError('contradictory pair')
    closure=gt.copy()
    for k in range(m):closure |= closure[:,k,None]&closure[None,k,:]
    if np.any(np.diag(closure)):raise ValueError('contradictory certified order')
    lo_rank=1+gt.sum(axis=1);hi_rank=m-gt.sum(axis=0)
    rank=np.arange(1,m+1);yr=rankdata(y);allowed=(rank[None,:]>=lo_rank[:,None])&(rank[None,:]<=hi_rank[:,None])
    costs=(rank[None,:]-yr[:,None])**2
    x1,x2=linear_sum_assignment(np.where(allowed,costs,1e9));y1,y2=linear_sum_assignment(np.where(allowed,costs,-1e9),maximize=True)
    if not allowed[x1,x2].all() or not allowed[y1,y2].all():raise ValueError('no rank assignment')
    factor=6/(m*(m*m-1))
    return {'status':'conditional_strict_ranks','lower':max(-1.,float(1-factor*costs[y1,y2].sum())),
            'upper':min(1.,float(1-factor*costs[x1,x2].sum())),
            'rank_lower':lo_rank.tolist(),'rank_upper':hi_rank.tolist(),'forced_pairs':int(gt.sum()),'force_order':gt.tolist()}
