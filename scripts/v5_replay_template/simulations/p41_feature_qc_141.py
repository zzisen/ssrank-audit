"""Pure expression feature-variance QA; no outcome or score computations."""
import numpy as np

def feature_qc(log,weights,indices,threshold=.8):
    x=np.asarray(log,float);w=np.asarray(weights,float)
    if x.ndim!=3 or x.shape[:2]!=(5,47) or w.ndim!=1 or len(w)!=len(indices) or not np.isfinite(x).all() or not np.isfinite(w).all() or not np.isfinite(threshold) or not 0<=threshold<=1 or np.abs(w).sum()==0:raise ValueError('invalid inputs')
    mapped=[]
    for i in indices:
        if i is None:mapped.append(None)
        elif isinstance(i,(int,np.integer)) and not isinstance(i,bool) and 0<=i<x.shape[2]:mapped.append(int(i))
        else:raise ValueError('invalid index')
    real=[i for i in mapped if i is not None]
    if len(real)!=len(set(real)):raise ValueError('ambiguous many-to-one mapping')
    present=np.array([i is not None for i in mapped],bool)
    def assess(data,label):
        a=data.reshape(-1,x.shape[2]);sd=a.std(axis=0,ddof=1);usable=np.zeros(len(w),bool)
        for j,i in enumerate(mapped):
            if i is not None and sd[i]>1e-12 and w[j]!=0:usable[j]=True
        def cv(which):
            total=float(np.abs(w[which]).sum());return None if total==0 else float(np.abs(w[which&usable]).sum()/total)
        coverage={'total':cv(w!=0),'positive':cv(w>0),'negative':cv(w<0)}
        return {'sample_scope':label,'samples':len(a),'coverage_known_weights':coverage,
                'mapped_nonzero_constant_count':int(np.sum(present & (w!=0) & ~usable)),
                'status':'pass_engineering_coverage' if all(v is None or v+1e-15>=threshold for v in coverage.values()) else 'unavailable_coverage'}
    full=assess(x,'full');folds=[assess(np.delete(x,p,axis=0),f'excluding_plate_{p}') for p in range(5)]
    return {'status':'completed_qc_only','full':full,'folds':folds,'threshold':threshold,
            'known_weights_only':True,'score_or_function_evaluated':False}
