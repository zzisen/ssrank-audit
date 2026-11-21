"""Export four V5 benchmark seed tables from source-derived V5 replay JSON files.

Run the independently configured V5 3D gate, B6 gate, and repeated-function
repair first. See seed origin audit. Writes to a new output directory only.
"""
from pathlib import Path
import argparse,csv,json
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--replay-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 names=['3d_treatment_summary.csv','3d_plate_treatment_values.csv','b6_all_program_condition_statuses.csv','b6_drug_summary.csv']
 if any((out/n).exists() for n in names):raise FileExistsError('Refusing to overwrite existing seed files')
 def read(n):return json.loads((a.replay_dir/n).read_text(encoding='utf8'))
 d=read('V5_3D_BIO_DESCRIPTIVE_2026-11-18.json')['original'];b=read('V5_B6_BIO_DESCRIPTIVE_2026-11-18.json')['original'];r=read('V5_REPEAT_COMPLETE_PAIRS_EXPLORATORY_2026-11-18.json')
 labels=d['labels'];x=np.asarray(d['main']['scores_by_plate_and_treatment']);y=np.asarray(d['late_function_by_plate_and_treatment']);early=np.asarray(d['early_function_by_plate_and_treatment']);repeats={v['label']:v for v in r['by_treatment']}
 with (out/names[0]).open('x',encoding='utf-8-sig',newline='') as f:
  wr=csv.writer(f);wr.writerow(['treatment','mean_24h_score','mean_24h_normalized_function','mean_04h_normalized_function','repeat_24h_mean_postqc','repeat_plate_count'])
  for i,l in enumerate(labels):wr.writerow([l,float(x[:,i].mean()),float(y[:,i].mean()),float(early[:,i].mean()),repeats[l]['repeat_function_mean_24h'],repeats[l]['n_repeat_plates']])
 with (out/names[1]).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['plate','treatment','score','function_04h_normalized','function_24h_normalized'])
  for j,plate in enumerate(d['plates']):
   for i,label in enumerate(labels):w.writerow([plate,label,d['main']['scores_by_plate_and_treatment'][j][i],d['early_function_by_plate_and_treatment'][j][i],d['late_function_by_plate_and_treatment'][j][i]])
 with (out/names[2]).open('x',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['program','assay','drug','concentration_rank','status','reasons','mean_score','assay_value'])
  for name,v in b['programs'].items():
   if v['status']!='ok':w.writerow([name,'','','',v['status']]);continue
   for c in v['conditions']:w.writerow([name,c['assay'],c['drug'],c['rank'],c['status'],';'.join(c['reasons']),c.get('mean_score'),c.get('assay_value')])
 with (out/names[3]).open('x',encoding='utf-8-sig',newline='') as f:
  wr=csv.writer(f);wr.writerow(['endpoint','drug','eligible_ranks','score_mean','assay_mean','within_drug_spearman_rho'])
  for assay,v in b['programs']['main']['endpoints'].items():
   for drug in v['drugs']:wr.writerow([assay,drug['drug'],len(drug['ranks']),drug['score_mean'],drug['assay_mean'],drug['within_rank_rho']])
 print('Exported',len(names),'tables to',out)
if __name__=='__main__':main()
