"""QC-only entry, superseding p41_qc_preflight_98's loose numeric parser."""
import argparse,gzip,importlib.util,json,os,warnings
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('p98',ROOT/'simulations/p41_qc_preflight_98.py');q=importlib.util.module_from_spec(sp);sp.loader.exec_module(q)
def strict_stream_qc(path,sample_ids):
 sample_ids=list(sample_ids);n=len(sample_ids)
 if not n or len(set(sample_ids))!=n:raise ValueError('invalid sample IDs')
 libs=np.zeros(n);detected=np.zeros(n,dtype=np.int64);seen=set();count=0
 with gzip.open(path,'rt',encoding='utf-8-sig') as f:
  h=f.readline().rstrip('\r\n').split('\t');observed=h[1:]
  if h[0]!='Ensembl' or len(observed)!=n or len(set(observed))!=n or set(observed)!=set(sample_ids):raise ValueError('2D sample header mismatch')
  lookup={name:i for i,name in enumerate(observed)};order=np.array([lookup[name] for name in sample_ids])
  for row,line in enumerate(f,2):
   fields=line.rstrip('\r\n').split('\t',1)
   if len(fields)!=2 or not fields[0] or fields[0] in seen:raise ValueError('invalid or duplicate gene ID at row '+str(row))
   gid,raw=fields;seen.add(gid)
   if raw.count('\t')!=n-1:raise ValueError('count field count mismatch at row '+str(row))
   try:
    with warnings.catch_warnings():
     warnings.simplefilter('error',DeprecationWarning)
     values=np.fromstring(raw,sep='\t')
   except (DeprecationWarning,ValueError):raise ValueError('invalid numeric text at row '+str(row))
   if len(values)!=n or not np.isfinite(values).all() or (values<0).any():raise ValueError('invalid numeric count at row '+str(row))
   libs+=values;detected+=(values>0);count+=1
 if not count or not np.isfinite(libs).all() or (libs<=0).any():raise ValueError('invalid 2D library')
 return {'gene_rows':count,'sample_count':n,'sample_header_order_differs_from_metadata':observed!=sample_ids,'library_totals':libs[order].tolist(),'detected_genes':detected[order].tolist()},seen
def write_failure(out,mode,error):
 report={'scope':q.SCOPE,'mode':mode,'status':'unavailable_preflight_error','scientific_association_performed':False,'error_type':type(error).__name__,'error_message':str(error)}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf8')
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--paths-json',type=Path);p.add_argument('--output',type=Path,required=True);z=p.parse_args()
 if z.mode=='real':
  q.real_gate() # Must succeed before any real path is opened or output is written.
  out=z.output.resolve()
  if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise ValueError('real output outside QC directory')
 else:
  if z.paths_json is None:raise ValueError('fake paths required')
  out=q.a.safe(z.output)
 try:
  if z.mode=='real':paths=q.checked_real_paths()
  else:paths=q.synthetic_paths(json.loads(q.a.safe(z.paths_json).read_text(encoding='utf8')))
  q.stream_qc=strict_stream_qc;result=q.run(paths,z.mode);result['status']='completed_qc'
 except (ValueError,PermissionError,OSError,KeyError) as exc:
  write_failure(out,z.mode,exc);raise SystemExit(2)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'scope':result['scope'],'mode':z.mode,'status':result['status'],'scientific_association_performed':False,'cardio_main_status':result['cardiopedia']['audit']['main_status'],'b6_samples':result['b6']['sample_count']},indent=2))
if __name__=='__main__':main()
