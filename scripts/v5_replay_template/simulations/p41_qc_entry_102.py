"""Final staged QC-only entry: strict parser, no unapproved real inputs."""
import argparse,collections,csv,importlib.util,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('p99',ROOT/'simulations/p41_qc_entry_99.py');p99=importlib.util.module_from_spec(sp);sp.loader.exec_module(p99)
q=p99.q
def exclusion_counts(rows):
 result=collections.Counter()
 for row in rows:
  if 'Excluded' not in row:result['<field absent>']+=1
  elif row['Excluded'] is None or not str(row['Excluded']).strip():result['<blank>']+=1
  else:result[str(row['Excluded']).strip()]+=1
 return dict(result)
def safe_error(out,mode,exc):
 message=str(exc)
 if not re.fullmatch(r'(invalid numeric text at row |count field count mismatch at row |invalid numeric count at row |invalid or duplicate gene ID at row )\d+',message):message='QC could not complete; input or processing details withheld from this report.'
 payload={'scope':q.SCOPE,'mode':mode,'status':'unavailable_preflight_error','scientific_association_performed':False,'error_type':type(exc).__name__,'error_message':message}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2),encoding='utf8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--paths-json',type=Path);p.add_argument('--output',type=Path,required=True);z=p.parse_args()
 if z.mode=='real':
  q.real_gate() # No real path or output opened before the user-scope gate.
  out=z.output.resolve()
  if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise ValueError('real output outside QC directory')
 else:
  if z.paths_json is None:raise ValueError('fake paths required')
  out=q.a.safe(z.output)
 try:
  if z.mode=='real':paths=q.checked_real_paths()
  else:paths=q.synthetic_paths(json.loads(q.a.safe(z.paths_json).read_text(encoding='utf8')))
  q.stream_qc=p99.strict_stream_qc;result=q.run(paths,z.mode)
  if z.mode=='real':
   with paths['b6_samples'].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
  else:rows=json.loads(paths['b6_samples'].read_text(encoding='utf8'))
  result['b6']['author_excluded_flag_counts']=exclusion_counts(rows);result['b6']['author_excluded_flag_interpretation']='Counts only; no category automatically treated as included or excluded.';result['status']='completed_qc'
 except Exception as exc:
  safe_error(out,z.mode,exc);raise SystemExit(2)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'scope':result['scope'],'mode':z.mode,'status':result['status'],'scientific_association_performed':False,'cardio_main_status':result['cardiopedia']['audit']['main_status'],'b6_samples':result['b6']['sample_count']},indent=2))
if __name__=='__main__':main()
