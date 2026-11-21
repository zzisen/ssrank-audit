"""Pure V3 primary computations. No file loading, entry point or authorization."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

ROOT=Path(__file__).resolve().parents[1]
def _module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
effect=_module('effect114','simulations/p41_effect_diagnostics_84.py')
early_mod=_module('early114','simulations/p41_early_sensitivity_90.py')
within_mod=_module('within114','simulations/p41_within_between_94.py')
burden_mod=_module('burden114','simulations/p41_burden_100.py')

def rank_correlation(x,y):
    a=rankdata(x);b=rankdata(y)
    if a.size<2 or np.std(a)==0 or np.std(b)==0:return None
    return float(np.corrcoef(a,b)[0,1])

def log_cpm(counts):
    x=np.asarray(counts,float)
    if x.ndim!=3 or x.shape[:2]!=(5,47) or not np.isfinite(x).all() or np.any(x<0):raise ValueError('invalid main count matrix')
    flat=x.reshape(235,-1);libraries=flat.sum(axis=1)
    if np.any(libraries<=0) or not np.isfinite(libraries).all():raise ValueError('invalid main libraries')
    return np.log2(1+1e6*flat/libraries[:,None])

def score_program(log,weights,indices,unknown_count=0,threshold=.8):
    z=np.asarray(log,float);w=np.asarray(weights,float)
    if z.ndim!=2 or z.shape[0]!=235 or not np.isfinite(z).all():raise ValueError('invalid log-CPM')
    if w.ndim!=1 or len(w)!=len(indices) or not np.isfinite(w).all() or not np.any(w):raise ValueError('invalid known weights')
    if not isinstance(unknown_count,int) or unknown_count<0:raise ValueError('invalid unknown count')
    mapped=[]
    for i in indices:
        if i is None:mapped.append(None);continue
        if isinstance(i,bool) or not isinstance(i,(int,np.integer)) or not 0<=i<z.shape[1]:raise ValueError('invalid feature mapping')
        mapped.append(int(i))
    used_indices=[i for i in mapped if i is not None]
    if len(set(used_indices))!=len(used_indices):raise ValueError('many-to-one feature mapping')
    usable=np.zeros(len(w),bool);out=np.zeros(235);zero_sd=0
    for j,i in enumerate(mapped):
        if i is None or w[j]==0:continue
        values=z[:,i];sd=values.std(ddof=1)
        if not np.isfinite(sd) or sd<=1e-12:zero_sd+=1;continue
        usable[j]=True;out+=(values-values.mean())/sd*w[j]
    def coverage(mask):
        den=np.abs(w[mask]).sum()
        return None if den==0 else float(np.abs(w[mask&usable]).sum()/den)
    cov={'total':coverage(w!=0),'positive':coverage(w>0),'negative':coverage(w<0)}
    report={'coverage_known_finite_weights':cov,'unknown_weight_count':unknown_count,
            'original_weight_mass_coverage':'unidentifiable' if unknown_count else 'known',
            'known_genes':len(w),'mapped_count':sum(i is not None for i in mapped),
            'usable_nonzero_count':int(usable.sum()),'zero_sd_count':zero_sd}
    if any(v is not None and v<threshold for v in cov.values()):return {'status':'unavailable_coverage','audit':report,'scores':None}
    denom=np.abs(w[usable]).sum()
    if denom<=0:return {'status':'unavailable_no_weight','audit':report,'scores':None}
    return {'status':'ok','audit':report,'scores':(out/denom).reshape(5,47)}

def describe_program(scores,late,early=None,counts=None,wm_controls=None,repeat_by_treatment=None):
    y=np.asarray(late,float);s=np.asarray(scores,float)
    if s.shape!=(5,47) or y.shape!=(5,47) or not np.isfinite(s).all() or not np.isfinite(y).all():
        return {'status':'unavailable_incomplete'}
    diagnostics=effect.diagnostics(s,y)
    sm=s.mean(axis=0);mean=y.mean(axis=0);median=np.median(y,axis=0)
    diagnostics['median_rho']=rank_correlation(sm,median)
    if np.all(y>0):
        diagnostics['geometric_status']='ok';diagnostics['geometric_rho']=rank_correlation(sm,np.exp(np.log(y).mean(axis=0)))
    else:diagnostics['geometric_status']='unavailable_nonpositive';diagnostics['geometric_rho']=None
    result={'status':'ok','effect':diagnostics,'within_between':within_mod.calculate(s,y),
            'early':early_mod.calculate(s,early,y) if early is not None else {'status':'unavailable_missing'},
            'wm_burden':{'status':'unavailable_missing'},'repeat':{'status':'unavailable_missing'},
            'interpretation':'fixed conditions and normalized function; descriptive only; no population inference'}
    if counts is not None and wm_controls is not None:
        result['wm_burden']=burden_mod.calculate(counts,wm_controls,s,y)[0]
    if repeat_by_treatment is not None:
        if len(repeat_by_treatment)!=47 or any(not v or not np.isfinite(np.asarray(v,float)).all() for v in repeat_by_treatment):
            result['repeat']={'status':'unavailable_incomplete'}
        else:
            result['repeat']={'status':'ok','rho':rank_correlation(sm,[float(np.mean(v)) for v in repeat_by_treatment]),
                              'n_by_treatment':[len(v) for v in repeat_by_treatment],
                              'interpretation':'same-label functional repeat only; no independent RNA pairing'}
    return result
