"""Synthetic-development candidate; no real-data file access."""
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
bound=load('bound127_for134','simulations/p41_rank_bound_fast_127.py')
effect=load('effect131_for134','simulations/p41_effect_size_131.py')

def evaluate(x,w,mask,groups,y,threshold=.8):
    x=np.asarray(x,float);w=np.asarray(w,float);mask=np.asarray(mask);groups=np.asarray(groups,float);y=np.asarray(y,float)
    if x.ndim!=2 or mask.dtype!=np.bool_ or w.ndim!=1 or mask.shape!=w.shape or x.shape[1]!=mask.sum() or x.shape[0]<2 or groups.ndim!=2 or groups.shape[1]!=x.shape[0] or y.shape!=(groups.shape[0],):raise ValueError('input shape')
    if not all(np.isfinite(v).all() for v in (x,w,groups,y)) or np.any(y<=0) or not np.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('invalid values')
    if np.abs(w).sum()==0:raise ValueError('zero weights')
    if not np.allclose(groups.sum(1),1,rtol=0,atol=1e-10):raise ValueError('condition weights')
    sd=x.std(0,ddof=1);is_constant=sd<=1e-12
    positions=np.flatnonzero(mask);usable=mask.copy();usable[positions[is_constant]]=False
    def coverage(which):
        total=float(np.abs(w[which]).sum())
        return None if total==0 else float(np.abs(w[which&usable]).sum()/total)
    cov={'total':coverage(w!=0),'positive':coverage(w>0),'negative':coverage(w<0)}
    audit={'coverage_known_weights':cov,'constant_observed_nonzero_weight_count':int(np.sum(is_constant & (w[positions]!=0))),
           'unknown_weight_mass':'unidentified','threshold':threshold}
    if any(v is not None and v+1e-15<threshold for v in cov.values()):return {'status':'unavailable_coverage','audit':audit}
    usable_x=x[:,~is_constant];sd=sd[~is_constant];z=(usable_x-usable_x.mean(0))/sd
    if z.shape[1]==0 or np.abs(w[usable]).sum()==0:return {'status':'unavailable_no_usable_weight','audit':audit}
    score=z@w[usable]/np.abs(w[usable]).sum();means=groups@score
    rho=float(spearmanr(means,y).statistic)
    if not np.isfinite(rho):return {'status':'unavailable_constant_rank','audit':audit}
    eff=effect.descriptive_effect(means,y)
    if audit['constant_observed_nonzero_weight_count']:
        sensitivity={'status':'unavailable_observed_constant_features','lower':None,'upper':None}
    else:
        sensitivity=bound.envelope(z,w,usable,groups,y,True)
    return {'status':'ok','audit':audit,'score_means':means.tolist(),'rho_observed_feature_target':rho,
            'effect':eff,'conditional_known_weight_sensitivity':{'status':sensitivity['status'],'lower':sensitivity['lower'],'upper':sensitivity['upper']},
            'interpretation':'noncausal fixed-sample and observed-feature description; bounds conditional on no complete-score ties'}
