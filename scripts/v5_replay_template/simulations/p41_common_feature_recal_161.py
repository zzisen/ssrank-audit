"""Fixed common-feature recalibration sensitivity; no outcome input or file I/O."""
import importlib.util
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('recal139_for161',ROOT/'simulations/p41_recalibration_139.py');recal=importlib.util.module_from_spec(s);s.loader.exec_module(recal)

def diagnose(x,weights,mask,threshold=.8):
    x=np.asarray(x,float);w=np.asarray(weights,float);mask=np.asarray(mask)
    if x.ndim!=3 or x.shape[:2]!=(5,47) or mask.dtype!=np.bool_ or mask.ndim!=1 or w.shape!=mask.shape or x.shape[2]!=mask.sum() or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(threshold) or not 0<=threshold<=1 or np.abs(w).sum()==0:raise ValueError('invalid input')
    observed=w[mask];full_sd=x.reshape(235,-1).std(0,ddof=1);full_usable=(observed!=0)&(full_sd>1e-12)
    if not full_usable.any():return {'status':'unavailable_full_score','strict':None,'common':None}
    strict=recal.diagnose(x[:,:,full_usable],observed[full_usable])
    common=full_usable.copy()
    for hold in range(5):common &= np.delete(x,hold,axis=0).reshape(188,-1).std(0,ddof=1)>1e-12
    observed_positions=np.flatnonzero(mask);global_mask=np.zeros(len(w),dtype=bool);global_mask[observed_positions[common]]=True
    def cv(which):
        total=np.abs(w[which]).sum();return None if total==0 else float(np.abs(w[which&global_mask]).sum()/total)
    coverage={'total':cv(w!=0),'positive':cv(w>0),'negative':cv(w<0)}
    if not common.any() or any(v is not None and v+1e-15<threshold for v in coverage.values()):
        return {'status':'unavailable_common_coverage','strict':strict,'common':None,'coverage_known_weights':coverage,'common_gene_count':int(common.sum())}
    result=recal.diagnose(x[:,:,common],observed[common])
    return {'status':'ok','strict':strict,'common':result,'coverage_known_weights':coverage,'common_gene_count':int(common.sum()),
            'limits':['common-feature sensitivity is not the original main score','no outcome used','not external validation']}
