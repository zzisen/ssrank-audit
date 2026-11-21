import argparse,importlib.util,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def calculate(score,early,late):
 s=np.asarray(score,float);a=np.asarray(early,float);b=np.asarray(late,float)
 if any(x.shape!=(5,47) or not np.isfinite(x).all() for x in (s,a,b)):return {'status':'unavailable_incomplete'}
 ss=s.mean(axis=0);aa=np.median(a,axis=0);bb=np.median(b,axis=0);sd_s=ss.std(ddof=1);sd_a=aa.std(ddof=1)
 if min(sd_s,sd_a)<=1e-12:return {'status':'unavailable_constant'}
 z_s=(ss-ss.mean())/sd_s;z_a=(aa-aa.mean())/sd_a;design=np.column_stack([np.ones(47),z_a,z_s]);condition=float(np.linalg.cond(design))
 if not np.isfinite(condition) or condition>30 or np.linalg.matrix_rank(design)!=3:return {'status':'unavailable_collinear','design_condition':condition if np.isfinite(condition) else None}
 beta=np.linalg.lstsq(design,bb,rcond=None)[0]
 return {'status':'ok','score_beta_per_sample_SD':float(beta[2]),'early_beta_per_sample_SD':float(beta[1]),'condition_number':condition,'n_treatments':47,'time_interpretation':'24h score and 24h function are concurrent; no future prediction or causal claim'}
def main(source,out):
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise RuntimeError('guard')
 source=source.resolve();out=out.resolve();root=(ROOT/'reports/runs').resolve()
 if not source.is_relative_to(root) or not out.is_relative_to(root):raise ValueError('synthetic only')
 out.mkdir(parents=True,exist_ok=True);a=np.load(source/'expected_SYNTHETIC.npz');s=np.load(source/'gate86_SYNTHETIC.npz')['score'];early=a['early'];late=a['main'];res=calculate(s,early,late)
 if res['status']!='ok':raise AssertionError('unexpected fake input status')
 perm=np.arange(46,-1,-1);rp=calculate(s[:,perm],early[:,perm],late[:,perm])
 if not np.isclose(res['score_beta_per_sample_SD'],rp['score_beta_per_sample_SD']):raise AssertionError('condition permutation')
 missing=early.copy();missing[0,0]=np.nan;equal=np.tile(s.mean(axis=0),(5,1));const=np.ones((5,47))
 checks={'missing_early':calculate(s,missing,late)['status'],'collinear_early':calculate(s,equal,late)['status'],'constant_early':calculate(s,const,late)['status']}
 if checks!={'missing_early':'unavailable_incomplete','collinear_early':'unavailable_collinear','constant_early':'unavailable_constant'}:raise AssertionError(checks)
 output={'synthetic_only':True,'reuses_stage_86_inputs':True,'analysis':res,'negative_checks':checks,'condition_permutation_invariant':True};(out/'metrics_SYNTHETIC.json').write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.source,a.out)
