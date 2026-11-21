"""QC-only entry, sample-level key provenance; real mode remains gated."""
import argparse,collections,csv,importlib.util,json,re
from pathlib import Path
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('p103',ROOT/'simulations/p41_qc_entry_103.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);q=m.q
def add_cardio_keys(result,counts_path,function_path):
 cardio=result['cardiopedia'];order=cardio.get('main_sample_order')
 if order is None:return result
 wb=load_workbook(counts_path,read_only=True,data_only=True);ws=wb.active;ws.reset_dimensions();header=next(ws.iter_rows(values_only=True));wb.close();rna={}
 for raw in header[3:]:
  match=re.fullmatch(r'(.+)_(P\d+)_([A-H]\d{2})',str(raw))
  if not match:raise ValueError('RNA metadata header cannot be traced')
  label,plate,well=match.groups();key=(plate,well)
  if key in rna:raise ValueError('RNA metadata duplicate key')
  rna[key]=(label,str(raw))
 wb=load_workbook(function_path,read_only=True,data_only=True);ws=wb['Tab 2 Functional data 24H'];ws.reset_dimensions();it=ws.iter_rows(values_only=True);h=next(it);ix={v:i for i,v in enumerate(h)};by=collections.defaultdict(list)
 for row in it:
  label,well,dataset,pt=[row[ix[k]] if ix[k]<len(row) else None for k in ('Treatment','Wellname','Dataset','Plate_timepoint')]
  if dataset!='Cardiopedia-Ligand' or not all(isinstance(x,str) for x in (label,well,pt)):continue
  plate=pt.split('_',1)[0]
  if pt!=plate+'_24H':continue
  by[(plate,label)].append(well)
 wb.close()
 for record in order:
  plate=record['plate'];label=record['treatment'];wells=by[(plate,label)]
  if len(wells)!=1 or (plate,wells[0]) not in rna or rna[(plate,wells[0])][0]!=label:raise ValueError('3D main identity cannot be traced')
  record['well']=wells[0];record['rna_header']=rna[(plate,wells[0])][1]
 controls=[]
 for tm,plate,label,well in cardio['audit']['control_keys']:
  if tm!='24H' or (plate,well) not in rna or rna[(plate,well)][0]!=label or well not in by[(plate,label)]:raise ValueError('3D control identity cannot be traced')
  controls.append({'plate':plate,'control':label,'well':well,'rna_header':rna[(plate,well)][1]})
 if len(controls)!=len(cardio['control_library_totals']):raise ValueError('3D control vector length mismatch')
 cardio['control_sample_order']=controls
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--paths-json',type=Path);p.add_argument('--output',type=Path,required=True);z=p.parse_args()
 if z.mode=='real':
  q.real_gate()
  out=z.output.resolve()
  if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise ValueError('real output outside QC directory')
 else:
  if z.paths_json is None:raise ValueError('fake paths required')
  out=q.a.safe(z.output)
 try:
  if z.mode=='real':paths=q.checked_real_paths()
  else:paths=q.synthetic_paths(json.loads(q.a.safe(z.paths_json).read_text(encoding='utf8')))
  q.stream_qc=m.m.p99.strict_stream_qc;result=q.run(paths,z.mode)
  if z.mode=='real':
   with paths['b6_samples'].open(encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
  else:rows=json.loads(paths['b6_samples'].read_text(encoding='utf8'))
  result['b6']['author_excluded_flag_counts']=m.m.exclusion_counts(rows);result['b6']['author_excluded_flag_interpretation']='Counts only; no category automatically treated as included or excluded.'
  result['b6']['gtf_caution']='Original B6 paper reports GRCh38.p13 and Ensembl 104, consistent with the archived release 104 mapping; exact original GTF bytes and all cross-version gene identities are not established.'
  result['status']='completed_qc';result=m.enrich(result,rows,z.mode);result=add_cardio_keys(result,paths['cardio_counts'],paths['cardio_function'])
 except Exception as exc:
  m.m.safe_error(out,z.mode,exc);raise SystemExit(2)
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'scope':result['scope'],'mode':z.mode,'status':result['status'],'scientific_association_performed':False,'cardio_main_status':result['cardiopedia']['audit']['main_status'],'b6_samples':result['b6']['sample_count']},indent=2))
if __name__=='__main__':main()
