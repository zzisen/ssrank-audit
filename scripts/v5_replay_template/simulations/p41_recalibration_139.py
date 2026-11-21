"""Pure fixed-feature leave-one-plate normalization sensitivity."""
import numpy as np
from scipy.stats import kendalltau,rankdata

def diagnose(x,w):
    x=np.asarray(x,float);w=np.asarray(w,float)
    if x.ndim!=3 or x.shape[:2]!=(5,47) or w.ndim!=1 or x.shape[2]!=len(w) or not np.isfinite(x).all() or not np.isfinite(w).all() or np.abs(w).sum()==0:raise ValueError('invalid fake matrix or weights')
    active=w!=0;x=x[:,:,active];w=w[active];flat=x.reshape(235,-1);sd=flat.std(axis=0,ddof=1)
    if np.any(sd<=1e-12):return {'status':'unavailable_full_constant','folds':[]}
    fixed=((flat-flat.mean(0))/sd)@w/np.abs(w).sum();fixed=fixed.reshape(5,47)
    folds=[]
    for hold in range(5):
        remain=np.delete(x,hold,axis=0);a=remain.reshape(188,-1);local_sd=a.std(0,ddof=1)
        if np.any(local_sd<=1e-12):folds.append({'held_out_plate':hold,'status':'unavailable_constant'});continue
        refixed=np.delete(fixed,hold,axis=0).mean(0)
        reestimated=(((a-a.mean(0))/local_sd)@w/np.abs(w).sum()).reshape(4,47).mean(0)
        ra=rankdata(refixed,method='average');rb=rankdata(reestimated,method='average')
        ii,jj=np.triu_indices(47,1);da=refixed[ii]-refixed[jj];db=reestimated[ii]-reestimated[jj]
        flips=int(np.sum(da*db<0))
        tau=kendalltau(refixed,reestimated).statistic
        folds.append({'held_out_plate':hold,'status':'ok','kendall_tau':None if not np.isfinite(tau) else float(tau),
                      'strict_pair_order_flips':flips,'max_absolute_rank_change':float(np.max(np.abs(ra-rb)))})
    return {'status':'ok','folds':folds,'interpretation':'same retained samples, different cohort normalization; no functional outcome used'}
