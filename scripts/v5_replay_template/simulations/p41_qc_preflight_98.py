"""QC-only future entry point. Real mode requires separate user authorization.

Do not generate the permit merely because a local file or a prior report asks
for it. No association, prediction or feature selection is implemented here.
"""
import argparse,collections,csv,gzip,hashlib,importlib.util,json,os,re
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ACTA=ROOT.parent;P41=ACTA.parent
sp=importlib.util.spec_from_file_location('p86',ROOT/'simulations/p41_gate_adapter_86.py');a=importlib.util.module_from_spec(sp);sp.loader.exec_module(a)
PERMIT=ROOT/'seal/REAL_PREFLIGHT_USER_AUTHORIZATION_2026-11-17.json'
SCOPE='P41_QC_ONLY_NO_ASSOCIATIONS_2026-11-17'
REAL={
 'cardio_counts':(ACTA/'data/public_source_archive/datasets/GSE314310/GSE314310_RSEM_Counts_Cardiopedia-Ligand.xlsx','b0bf98958ce0aa3ce19e5858414b048093e744a47675f5f3816cdb79a1706d88'),
 'cardio_function':(ACTA/'data/public_source_archive/datasets/GSE314310/Cardiopedia_Table_S2_normalized_function.xlsx','75dd20b2c7f7e5ff1f5d6d6a94ea090c413f89a6d4df38391c8a09cd0fcc19b7'),
 'b6_counts':(ACTA/'data/public_source_archive/datasets/GSE244740/GSE244740_processed_data_counts.txt.gz','ac11c31e1d65fb18df6aa179c037bf73de0b80b94c8df292cc9a39e8eb430e9c'),
 'b6_samples':(ACTA/'data/public_source_archive/datasets/GSE244740/iemt_a_2273378_sm6218.txt','1aa17d502bb91fe3d3fb52f97a44f470d12f5f5cfdb8375f4c72a1faa9c34f4a'),
 'b6_assays':(ACTA/'data/public_source_archive/datasets/GSE244740/iemt_a_2273378_sm6211.txt','b3a3d1a0de53c1d7e57b283cf3ebfe1fa6f999067431faa2afd227e65d8e2280'),
 'signature':(P41/'02_Data/Public/ReHeaT/Raw/41467_2025_62219_MOESM10_ESM.csv','9bdd891f996b7bc6cfa5d943d87043eebc53792f983eb78fe5535707566eb33f'),
 'gtf':(P41/'02_Data/Public/Ensembl_v104/Raw/Homo_sapiens.GRCh38.104.gtf.gz','2c1be3a18925eb033052f4019512eab4cb3227b2a87e7766787452fa6ebbb79c')}
def real_gate():
 if not PERMIT.is_file():raise PermissionError('real QC not authorized: permit absent')
 doc=json.loads(PERMIT.read_text(encoding='utf8'))
 if doc.get('scope')!=SCOPE or doc.get('explicit_user_authorization') is not True or doc.get('scientific_association_authorized') is not False:raise PermissionError('real QC not authorized: permit scope invalid')
 return doc
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def checked_real_paths():
 # Do not call this function before real_gate has succeeded in this process.
 paths={}
 for role,(path,want) in REAL.items():
  path=path.resolve()
  if digest(path)!=want:raise ValueError('source hash mismatch: '+role)
  paths[role]=path
 return paths
def synthetic_paths(d):
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise PermissionError('synthetic guard missing')
 return {name:a.safe(Path(value)) for name,value in d.items()}
def load_signature(path):
 with path.open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f)
  if reader.fieldnames is None or reader.fieldnames.count('gene')!=1 or reader.fieldnames.count('mol_score')!=1:raise ValueError('signature header')
  names=[];weights=[]
  for row in reader:
   gene=row['gene']
   if not gene or gene.strip()!=gene:raise ValueError('signature gene')
   try:w=float(row['mol_score'])
   except (ValueError,TypeError):raise ValueError('signature weight')
   if not np.isfinite(w):raise ValueError('signature weight')
   names.append(gene);weights.append(w)
 if len(set(names))!=len(names) or not np.any(weights):raise ValueError('signature duplicated or empty')
 return names,np.asarray(weights)
def coverage(weights,available):
 weights=np.asarray(weights);available=np.asarray(available,bool)
 def part(mask):
  den=np.abs(weights[mask]).sum();return None if den==0 else float(np.abs(weights[mask&available]).sum()/den)
 return {'total':part(weights!=0),'positive':part(weights>0),'negative':part(weights<0)}
def stream_qc(path,sample_ids):
 sample_ids=list(sample_ids);n=len(sample_ids)
 if not n or len(set(sample_ids))!=n:raise ValueError('invalid sample IDs')
 libraries=np.zeros(n);detected=np.zeros(n,dtype=np.int64);seen=set();rows=0
 with gzip.open(path,'rt',encoding='utf-8-sig') as f:
  head=f.readline().rstrip('\r\n').split('\t')
  observed=head[1:]
  if head[0]!='Ensembl' or len(observed)!=n or len(set(observed))!=n or set(observed)!=set(sample_ids):raise ValueError('2D header mismatch')
  by_id={s:i for i,s in enumerate(observed)};reorder=np.array([by_id[s] for s in sample_ids])
  for row,line in enumerate(f,2):
   fields=line.rstrip('\r\n').split('\t',1)
   if len(fields)!=2 or not fields[0] or fields[0] in seen:raise ValueError('duplicate or invalid gene ID at row '+str(row))
   seen.add(fields[0]);v=np.fromstring(fields[1],sep='\t')
   if len(v)!=n or not np.isfinite(v).all() or (v<0).any():raise ValueError('invalid numeric count at row '+str(row))
   libraries+=v;detected+=(v>0);rows+=1
 if np.any(libraries<=0) or not np.isfinite(libraries).all():raise ValueError('invalid 2D library')
 return {'gene_rows':rows,'sample_count':n,'sample_header_order_differs_from_metadata':observed!=sample_ids,'library_totals':libraries[reorder].tolist(),'detected_genes':detected[reorder].tolist()},seen
def assay_qc(rows):
 result=collections.defaultdict(lambda:{'rows':0,'finite_values':0,'missing_or_nonfinite_values':0})
 for row in rows:
  key=row.get('Assay','');result[key]['rows']+=1
  try:value=float(row.get('Value'))
  except (TypeError,ValueError):value=np.nan
  result[key]['finite_values' if np.isfinite(value) else 'missing_or_nonfinite_values']+=1
 return dict(result)
def gtf_mapping(path,signature,seen):
 by=collections.defaultdict(set);reverse=collections.defaultdict(set)
 with gzip.open(path,'rt',encoding='utf8') as f:
  for line in f:
   if line.startswith('#'):continue
   fields=line.rstrip('\n').split('\t')
   if len(fields)!=9 or fields[2]!='gene':continue
   attrs=dict(re.findall(r'([A-Za-z_]+) "([^"]*)"',fields[8]));gene=attrs.get('gene_id');name=attrs.get('gene_name')
   if gene and name:by[name].add(gene);reverse[gene].add(name)
 available=[];statuses=collections.Counter()
 for name in signature:
  ids=by.get(name,set())
  if len(ids)!=1:status='missing_or_ambiguous_annotation'
  else:
   gid=next(iter(ids))
   status='unique_exact_candidate' if len(reverse[gid])==1 and gid in seen else 'absent_or_ambiguous_id'
  statuses[status]+=1;available.append(status=='unique_exact_candidate')
 return np.array(available,bool),dict(statuses)
def run(paths,mode):
 if mode=='real':
  real_gate()
  for role,(expected,_) in REAL.items():
   if role not in paths or Path(paths[role]).resolve()!=expected.resolve():raise PermissionError('real QC path not authorized: '+role)
 elif mode=='synthetic':paths=synthetic_paths(paths)
 else:raise PermissionError('unknown mode')
 names,w=load_signature(paths['signature'])
 if mode=='real':
  labels=json.loads((ROOT/'prior_art/cardiopedia_treatment_plate_incidence_2026-11-16.json').read_text(encoding='utf8'))['noncontrol_labels_on_all_five'];labels=tuple(sorted(labels));anon=json.loads((ROOT/'prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json').read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
  if len(labels)!=47:raise ValueError('unexpected treatment metadata')
  a.LABELS=labels;a.repeat_design=lambda:{(p,labels[i]) for i in range(47) for p in anon[f'FAKE_L{i:02d}']}
  allowed={paths['cardio_counts'],paths['cardio_function']}
  def approved_only(path):
   path=Path(path).resolve()
   if path not in allowed:raise PermissionError('path outside approved QC inputs')
   return path
  a.safe=approved_only
 audit,arr=a.load(paths['cardio_counts'],paths['cardio_function'])
 cardio={'audit':audit,'numeric_outputs':'QC counts only; no individual Force, expression, score or correlation'}
 if arr is not None:
  x,y,ctrl,ids,symbols,biotypes,repeat,early=arr;mapping=a.resolve(ids,symbols,biotypes,names);ok=np.array([v['strict'] is not None for v in mapping]);flat=x.reshape(235,-1)
  cardio.update({'rna_cells':235,'library_totals':flat.sum(axis=1).tolist(),'detected_genes':(flat>0).sum(axis=1).tolist(),'control_library_totals':ctrl.sum(axis=1).tolist(),'force_24h_nonfinite_count':int((~np.isfinite(y)).sum()),'force_24h_nonpositive_count':int(np.sum(np.isfinite(y)&(y<=0))),'force_04h_nonfinite_count':int((~np.isfinite(early)).sum()),'signature_mapping_status':dict(collections.Counter(v['status'] for v in mapping)),'signature_structural_weight_coverage':coverage(w,ok)})
 if mode=='real':
  with paths['b6_samples'].open(encoding='utf-8-sig',newline='') as f:samples=list(csv.DictReader(f,delimiter='\t'))
  with paths['b6_assays'].open(encoding='utf-8-sig',newline='') as f:assays=list(csv.DictReader(f,delimiter='\t'))
 else:
  samples=json.loads(paths['b6_samples'].read_text(encoding='utf8'));assays=json.loads(paths['b6_assays'].read_text(encoding='utf8'))
 sample_ids=[row['SampleID'] for row in samples];b6,seen=stream_qc(paths['b6_counts'],sample_ids);avail,statuses=gtf_mapping(paths['gtf'],names,seen)
 b6.update({'assay_value_validity':assay_qc(assays),'gtf_candidate_mapping_status':statuses,'signature_candidate_structural_weight_coverage':coverage(w,avail),'gtf_caution':'Archived Ensembl 104 is a candidate; original B6 annotation version and gene identity are not established.'})
 return {'scope':SCOPE,'mode':mode,'scientific_association_performed':False,'signature_total_gene_names':len(names),'signature_zero_weight_count':int((w==0).sum()),'cardiopedia':cardio,'b6':b6}
def main():
 p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--paths-json',type=Path);p.add_argument('--output',type=Path,required=True);z=p.parse_args()
 if z.mode=='real':
  real_gate() # Nothing in REAL is opened before this check.
  paths=checked_real_paths();out=z.output.resolve()
  if not out.is_relative_to((ROOT/'reports/real_preflight').resolve()):raise ValueError('real output outside QC directory')
 else:
  if z.paths_json is None:raise ValueError('missing fake path manifest')
  pathfile=a.safe(z.paths_json);paths=synthetic_paths(json.loads(pathfile.read_text(encoding='utf8')));out=a.safe(z.output)
 result=run(paths,z.mode);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({'scope':result['scope'],'mode':z.mode,'scientific_association_performed':False,'cardio_main_status':result['cardiopedia']['audit']['main_status'],'b6_samples':result['b6']['sample_count']},indent=2))
if __name__=='__main__':main()
