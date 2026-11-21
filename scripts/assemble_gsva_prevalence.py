"""Assemble matched 360-draw GSVA comparison and a replacement figure-input table."""
from pathlib import Path
import argparse,hashlib,json
import numpy as np
import pandas as pd

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(description="Assemble complete matched historical/new GSVA draws; never overwrites existing files.")
 p.add_argument('--batch-dirs',type=Path,nargs='+',required=True)
 p.add_argument('--reference',type=Path,required=True,help='historical 9-row archived CSV')
 p.add_argument('--allow-missing-summary',action='store_true',help='allow an interrupted batch directory without summary; record the provenance limitation')
 p.add_argument('--output-dir',type=Path,required=True)
 a=p.parse_args()
 frames=[pd.read_csv(d/'matched_batch_comparison.csv') for d in a.batch_dirs]
 all_rows=pd.concat(frames,ignore_index=True)
 # Identical overlap between split runs is allowed; disagreement is an error.
 columns=[c for c in all_rows.columns if c not in ('prev_dcm_pct','replicate')]
 for key,g in all_rows.groupby(['prev_dcm_pct','replicate']):
  if len(g)>1 and not g[columns].eq(g[columns].iloc[0]).all(axis=None):
   raise ValueError('Conflicting duplicate matched batch '+str(key))
 all_rows=all_rows.drop_duplicates(['prev_dcm_pct','replicate']).reset_index(drop=True)
 if len(all_rows)!=360 or set(all_rows.prev_dcm_pct.unique())!=set(range(10,100,10)) or any(len(g)!=40 or set(g.replicate)!=set(range(1,41)) for _,g in all_rows.groupby('prev_dcm_pct')):
  raise ValueError('Need all 360 unique matched draws, 40 per prevalence')
 missing_summary=[str(d) for d in a.batch_dirs if not (d/'summary.json').is_file()]
 if missing_summary and not a.allow_missing_summary:raise ValueError('Missing summary files: '+str(missing_summary))
 summaries=[json.loads((d/'summary.json').read_text(encoding='utf8')) for d in a.batch_dirs if (d/'summary.json').is_file()]
 if not summaries:raise ValueError('At least one completed batch summary is required')
 if any(x.get('seed') != 20261120 for x in summaries):raise ValueError('Unexpected or inconsistent sampling seed')
 for key in ['historic_full_sd','tieavg_full_sd','historic_threshold','tieavg_threshold']:
  if not np.allclose([x[key] for x in summaries],summaries[0][key],atol=0,rtol=0): raise ValueError('Inconsistent baseline '+key)
 s=summaries[0]; sds={'old':s['historic_full_sd'],'new':s['tieavg_full_sd']}
 baseline={k:float(all_rows.loc[all_rows.prev_dcm_pct==50,f'{k}_probe_mean'].mean()) for k in ('old','new')}
 rows={k:[] for k in ('old','new')}
 for pct,g in all_rows.groupby('prev_dcm_pct',sort=True):
  for k in ('old','new'):
   drift=(g[f'{k}_probe_mean'].to_numpy()-baseline[k])/sds[k]
   rows[k].append({'prev_dcm_pct':int(pct),'n_draws':len(g),'gsva_mean_drift_sd':float(drift.mean()),'gsva_ci_lo':float(np.quantile(drift,.025)),'gsva_ci_hi':float(np.quantile(drift,.975)),'gsva_sens':float(g[f'{k}_sens'].mean()),'gsva_spec':float(g[f'{k}_spec'].mean())})
 old=pd.DataFrame(rows['old']);new=pd.DataFrame(rows['new']);saved=pd.read_csv(a.reference)
 if old.shape!=saved.shape or old.columns.tolist()!=saved.columns.tolist():raise ValueError('Historic reference schema mismatch')
 diffs={col:float(np.max(np.abs(old[col]-saved[col]))) for col in old.columns}
 a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'v6_4_table_clinical_prevalence_gsva_tieavg.csv';report=a.output_dir/'gsva_matched_recomputation.json'
 if out.exists() or report.exists():raise FileExistsError('Refusing to overwrite')
 new.to_csv(out,index=False)
 def metric(tab,k):return float(tab.loc[tab.prev_dcm_pct==k,'gsva_mean_drift_sd'].iloc[0])
 record={'n_batches':len(all_rows),'matched_n_per_prevalence':40,'random_seed':20261120,'directories_missing_summary':missing_summary,'old_reference_table_max_abs_differences':diffs,
  'old_total_drift_10_minus_90':metric(old,10)-metric(old,90),'new_total_drift_10_minus_90':metric(new,10)-metric(new,90),
  'old_spec_10':float(old.loc[old.prev_dcm_pct==10,'gsva_spec'].iloc[0]),'new_spec_10':float(new.loc[new.prev_dcm_pct==10,'gsva_spec'].iloc[0]),
  'old_sens_90':float(old.loc[old.prev_dcm_pct==90,'gsva_sens'].iloc[0]),'new_sens_90':float(new.loc[new.prev_dcm_pct==90,'gsva_sens'].iloc[0]),
  'mean_absolute_probe_difference_old_vs_new':float(all_rows.probe_absdiff_mean.mean()),'max_absolute_probe_difference_old_vs_new':float(all_rows.probe_absdiff_max.max()),
  'mean_absolute_probe_difference_ties_only':float(all_rows.tieonly_probe_absdiff_mean.mean()),'max_absolute_probe_difference_ties_only':float(all_rows.tieonly_probe_absdiff_max.max()),
  'mean_fraction_genes_in_ties_float32':float(all_rows.old_tie_fraction_mean.mean()),'new_table_sha256':sha(out),
  'important':'Old versus new changes both tie processing and float32/float64 kernel; tie-only holds the old transformed matrix fixed. Values are descriptive matched sensitivity results.'}
 report.write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
