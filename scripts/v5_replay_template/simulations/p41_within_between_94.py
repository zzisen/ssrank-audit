import argparse,json,os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
def corr(x,y):
 x=rankdata(x);y=rankdata(y)
 return None if x.std()==0 or y.std()==0 else float(np.corrcoef(x,y)[0,1])
def calculate(score,force):
 s=np.asarray(score,float);y=np.asarray(force,float)
 if s.shape!=(5,47) or y.shape!=(5,47) or not np.isfinite(s).all() or not np.isfinite(y).all():return {'status':'unavailable_incomplete'}
 between=corr(s.mean(axis=0),y.mean(axis=0));within=corr((s-s.mean(axis=0)).ravel(),(y-y.mean(axis=0)).ravel());each=[corr(s[:,i],y[:,i]) for i in range(47)];valid=[v for v in each if v is not None]
 return {'status':'ok' if between is not None and within is not None else 'unavailable_constant_rank','between_treatment_rho':between,'within_centered_rho':within,'within_treatment_rho':each,'within_treatment_available':len(valid),'within_treatment_median':float(np.median(valid)) if valid else None}
def main(seed,repetitions,out):
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise RuntimeError('guard')
 out=out.resolve()
 if not out.is_relative_to((ROOT/'reports/runs').resolve()):raise ValueError('synthetic only')
 out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);results={};representatives={}
 for world in ('independent_within','shared_within'):
  rows=[]
  for i in range(repetitions):
   h=rng.normal(size=47);u=rng.normal(0,.5,size=(5,47));v=rng.normal(0,.5,size=(5,47));s=h+u;y=h+(v if world=='independent_within' else .8*u+v)
   res=calculate(s,y);rows.append([res['between_treatment_rho'],res['within_centered_rho'],res['within_treatment_median']])
   if i==0:representatives[world+'_score']=s;representatives[world+'_force']=y
  mat=np.asarray(rows);np.save(out/(world+'_replicates_SYNTHETIC.npy'),mat);results[world]={'median_between_rho':float(np.median(mat[:,0])),'median_centered_within_rho':float(np.median(mat[:,1])),'median_of_treatment_median_rho':float(np.median(mat[:,2]))}
 np.savez_compressed(out/'examples_SYNTHETIC.npz',**representatives)
 baseline=np.load(ROOT/'reports/runs/p41_fullsize_controls_91/fullsize_expected_SYNTHETIC.npz');integrated=[calculate(score,baseline['main']) for score in baseline['scores']]
 if any(r['status']!='ok' for r in integrated):raise AssertionError('synthetic full-scale integration')
 for fake in (np.full((5,47),np.nan),np.ones((5,47))):
  if calculate(fake,np.ones((5,47)))['status']=='ok':raise AssertionError('invalid matrix accepted')
 output={'synthetic_only':True,'seed':seed,'replicates_per_world':repetitions,'worlds':results,'stage91_reused_fake_program_results':integrated,'cell_unit_warning':'235 centered cells are not independent patients or independent treatments'}
 (out/'metrics_SYNTHETIC.json').write_text(json.dumps(output,indent=2));print(json.dumps({k:output[k] for k in ('synthetic_only','seed','replicates_per_world','worlds','cell_unit_warning')},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--replicates',type=int,default=300);p.add_argument('--out',type=Path,required=True);z=p.parse_args();main(z.seed,z.replicates,z.out)
