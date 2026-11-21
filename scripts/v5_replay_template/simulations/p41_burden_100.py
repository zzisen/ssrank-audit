import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
def rho(x,y):
 a=rankdata(x);b=rankdata(y)
 return None if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
def calculate(counts,controls,score,force):
 x=np.asarray(counts,float);s=np.asarray(score,float);y=np.asarray(force,float)
 if x.ndim!=3 or x.shape[:2]!=(5,47) or s.shape!=(5,47) or y.shape!=(5,47) or len(controls)!=5 or not np.isfinite(x).all() or (x<0).any() or not np.isfinite(s).all() or not np.isfinite(y).all() or np.any(x.sum(axis=2)<=0):return {'status':'unavailable_input'},None
 lg=np.log2(1+1e6*x/x.sum(axis=2,keepdims=True));burden=np.empty((5,47))
 for p,c in enumerate(controls):
  c=np.asarray(c,float)
  if c.ndim!=2 or c.shape[1]!=x.shape[2] or c.shape[0]<2 or not np.isfinite(c).all() or (c<0).any() or (c.sum(axis=1)<=0).any():return {'status':'unavailable_controls','plate_index':p},None
  lc=np.log2(1+1e6*c/c.sum(axis=1,keepdims=True));burden[p]=np.median(np.abs(lg[p]-np.median(lc,axis=0)),axis=1)
 sm=s.mean(axis=0);ym=y.mean(axis=0);bm=burden.mean(axis=0);rs=rankdata(sm);ry=rankdata(ym);rb=rankdata(bm)
 if any(len(np.unique(v))<3 for v in (rs,ry,rb)):return {'status':'unavailable_constant_rank'},burden
 z=(rb-rb.mean())/rb.std(ddof=1);design=np.column_stack([np.ones(47),z]);cond=np.linalg.cond(design)
 if not np.isfinite(cond) or cond>30:return {'status':'unavailable_collinear'},burden
 sx=rs-design@np.linalg.lstsq(design,rs,rcond=None)[0];yy=ry-design@np.linalg.lstsq(design,ry,rcond=None)[0]
 if sx.std()<=1e-12 or yy.std()<=1e-12:return {'status':'unavailable_residual_constant'},burden
 return {'status':'ok','score_burden_rho':rho(sm,bm),'force_burden_rho':rho(ym,bm),'score_force_rho':rho(sm,ym),'partial_rank_correlation':float(np.corrcoef(sx,yy)[0,1]),'interpretation':'WM-labelled relative expression proxy only; conditional association is exploratory'},burden
def main(seed,reps,out):
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise RuntimeError('guard')
 out=out.resolve()
 if not out.is_relative_to((ROOT/'reports/runs').resolve()):raise ValueError('synthetic only')
 out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);results=[]
 for i in range(reps):
  h=rng.normal(size=47);s=h+rng.normal(0,.7,size=(5,47));y=h+rng.normal(0,.7,size=(5,47));gene_sign=np.r_[np.ones(250),-np.ones(250)];x=np.exp(4+np.abs(h)[None,:,None]*.3*gene_sign[None,None,:]+rng.normal(0,.4,size=(5,47,500)));ctrl=[np.exp(4+rng.normal(0,.4,size=(3,500))) for _ in range(5)]
  result,burden=calculate(x,ctrl,s,y)
  if result['status']!='ok':raise AssertionError(result)
  results.append([result['score_burden_rho'],result['force_burden_rho'],result['score_force_rho'],result['partial_rank_correlation']])
  if i==0:np.savez_compressed(out/'example_SYNTHETIC.npz',counts=x,controls=np.array(ctrl),score=s,force=y,burden=burden)
 bad=[c.copy() for c in ctrl];bad[0]=bad[0][:1];insufficient=calculate(x,bad,s,y)[0]['status'];bad=[c.copy() for c in ctrl];bad[0][0,0]=np.nan;invalid=calculate(x,bad,s,y)[0]['status']
 if insufficient!='unavailable_controls' or invalid!='unavailable_controls':raise AssertionError('controls not protected')
 values=np.asarray(results);np.save(out/'replicates_SYNTHETIC.npy',values);output={'synthetic_only':True,'seed':seed,'replicates':reps,'median_correlations':np.median(values,axis=0).tolist(),'columns':['score_burden','force_burden','score_force','conditional_rank'],'insufficient_control_status':insufficient,'invalid_control_status':invalid,'correct_vehicle_established':False}
 (out/'metrics_SYNTHETIC.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--replicates',type=int,default=200);p.add_argument('--out',type=Path,required=True);z=p.parse_args();main(z.seed,z.replicates,z.out)
