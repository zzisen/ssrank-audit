"""QC-only staged entry with traceable sample order; no association analysis."""
import argparse,csv,importlib.util,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('p102',ROOT/'simulations/p41_qc_entry_102.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);q=m.q
def enrich(result,rows,mode):
 ids=[row['SampleID'] for row in rows]
 if len(set(ids))!=len(ids) or len(ids)!=len(result['b6']['library_totals']):raise ValueError('2D QC sample identifier mismatch')
 result['b6']['sample_ids']=ids
 cardio=result['cardiopedia']
 if 'library_totals' in cardio:
  if mode=='real':labels=tuple(sorted(json.loads((ROOT/'prior_art/cardiopedia_treatment_plate_incidence_2026-11-16.json').read_text(encoding='utf8'))['noncontrol_labels_on_all_five']))
  else:labels=tuple(f'FAKE_L{i:02d}' for i in range(47))
  order=[{'plate':p,'treatment':label} for p in ('P01','P03','P04','P05','P06') for label in labels]
  if len(order)!=len(cardio['library_totals']):raise ValueError('3D QC sample identifier mismatch')
  cardio['main_sample_order']=order
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--paths-json',type=Path);p.add_argument('--output',type=Path,required=True);z=p.parse_args()
 if z.mode=='real':
  q.real_gate() # No real path or output opened before QC-scope authorization.
  out=z.output.resolve()
  if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise ValueError('real output outside QC directory')
 else:
  if z.paths_json is None:raise ValueError('fake paths required')
  out=q.a.safe(z.output)
 try:
  if z.mode=='real':paths=q.checked_real_paths()
  else:paths=q.synthetic_paths(json.loads(q.a.safe(z.paths_json).read_text(encoding='utf8')))
  q.stream_qc=m.p99.strict_stream_qc;result=q.run(paths,z.mode)
  if z.mode=='real':
   with paths['b6_samples'].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
  else:rows=json.loads(paths['b6_samples'].read_text(encoding='utf8'))
  result['b6']['author_excluded_flag_counts']=m.exclusion_counts(rows);result['b6']['author_excluded_flag_interpretation']='Counts only; no category automatically treated as included or excluded.';result['status']='completed_qc';result=enrich(result,rows,z.mode)
 except Exception as exc:
  m.safe_error(out,z.mode,exc);raise SystemExit(2)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'scope':result['scope'],'mode':z.mode,'status':result['status'],'scientific_association_performed':False,'cardio_main_status':result['cardiopedia']['audit']['main_status'],'b6_samples':result['b6']['sample_count']},indent=2))
if __name__=='__main__':main()
