import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
def corr(a,b):
 x=rankdata(a);y=rankdata(b)
 if x.std()==0 or y.std()==0:return None
 return float(np.corrcoef(x,y)[0,1])
def diagnostics(score,force):
 score=np.asarray(score,dtype=float);force=np.asarray(force,dtype=float)
 if score.shape!=(5,47) or force.shape!=(5,47) or not np.isfinite(score).all() or not np.isfinite(force).all():raise ValueError('invalid arrays')
 sm=score.mean(axis=0);fm=force.mean(axis=0);order=np.argsort(sm,kind='stable');q=11
 tied=sm[order[q-1]]==sm[order[q]] or sm[order[-q-1]]==sm[order[-q]]
 diff=None if tied else float(np.median(fm[order[-q:]])-np.median(fm[order[:q]]))
 return {'primary_rho':corr(sm,fm),'plate_rho':[corr(score[i],force[i]) for i in range(5)],'leave_one_plate_out_rho':[corr(np.delete(score,i,axis=0).mean(axis=0),np.delete(force,i,axis=0).mean(axis=0)) for i in range(5)],'force_treatment_mean_quantiles':np.quantile(fm,[0,.25,.5,.75,1]).tolist(),'high_minus_low_median_force':diff,'extreme_groups_n_each':q,'extreme_group_status':'unavailable_tied_boundary' if tied else 'ok'}
def main(seed,out):
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise RuntimeError('guard')
 out=out.resolve()
 if not out.is_relative_to((ROOT/'reports/runs').resolve()):raise ValueError('synthetic runs only')
 out.mkdir(parents=True,exist_ok=True);rng=np.random.default_rng(seed);s=rng.normal(size=(5,47));y=rng.normal(size=(5,47));d=diagnostics(s,y)
 doubled=diagnostics(s,2*y);reversed_score=diagnostics(-s,y);constant=diagnostics(np.ones((5,47)),y)
 if not np.isclose(doubled['high_minus_low_median_force'],2*d['high_minus_low_median_force']) or not np.isclose(reversed_score['high_minus_low_median_force'],-d['high_minus_low_median_force']):raise AssertionError('scale or sign')
 if constant['primary_rho'] is not None or constant['extreme_group_status']!='unavailable_tied_boundary':raise AssertionError('constant ranks')
 tied=s.copy();tied[:,1]=tied[:,0];means=tied.mean(axis=0);order=np.argsort(means);tied[:,order[11]]=tied[:,order[10]]
 if diagnostics(tied,y)['extreme_group_status']!='unavailable_tied_boundary':raise AssertionError('boundary tie')
 try:diagnostics(s,np.full((5,47),np.nan))
 except ValueError:pass
 else:raise AssertionError('nonfinite accepted')
 f=out/'effect_diagnostics_SYNTHETIC.npz';np.savez_compressed(f,score=s,force=y)
 result={'synthetic_only':True,'seed':seed,'diagnostics':d,'constant_score_status':constant['extreme_group_status'],'boundary_tie_rejected':True,'nonfinite_rejected':True,'data_sha256':hashlib.sha256(f.read_bytes()).hexdigest()};(out/'metrics_SYNTHETIC.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.seed,a.out)
