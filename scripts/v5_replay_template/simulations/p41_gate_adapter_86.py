"""Synthetic-only audit-first adapter and complete-support diagnostics."""
import collections,importlib.util,json,os,re
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1]
PLATES=('P01','P03','P04','P05','P06');REPEAT_PLATES=('P01','P02','P03','P04','P06')
LABELS=tuple(f'FAKE_L{i:02d}' for i in range(47));CONTROLS=('CTRL-WM','CTRL-DMSO','CTRL-Ethanol')
def safe(path):
 path=Path(path).resolve()
 if os.getenv('P41_PYTHON_SEAL_GUARD')!='active' or not path.is_relative_to((ROOT/'reports/runs').resolve()):raise ValueError('synthetic run path and guard required')
 return path
def parse_gmt(path,names):
 path=safe(path);names=list(names)
 if not names or len(set(names))!=len(names):raise ValueError('bad requested names')
 found={}
 for line in path.read_text(encoding='utf-8-sig').splitlines():
  fields=line.split('\t')
  if fields[0] not in names:continue
  if fields[0] in found:raise ValueError('duplicate set')
  while len(fields)>2 and fields[-1]=='':fields.pop()
  genes=fields[2:]
  if not genes or any(not x or x.strip()!=x for x in genes) or len(set(genes))!=len(genes):raise ValueError('invalid set members')
  found[fields[0]]=genes
 if set(found)!=set(names):raise ValueError('set missing')
 return found
def resolve(ids,symbols,biotypes,signature):
 if len(ids)!=len(symbols) or len(ids)!=len(biotypes) or len(set(ids))!=len(ids) or len(set(signature))!=len(signature):raise ValueError('invalid identifier structure')
 by=collections.defaultdict(list)
 for g,s,b in zip(ids,symbols,biotypes):
  if s is not None and str(s).strip():by[str(s).strip()].append((g,str(b or '')))
 rows=[]
 for symbol in signature:
  choices=by.get(symbol,[]);coding=[g for g,b in choices if b.casefold().replace('-','_')=='protein_coding']
  rows.append({'symbol':symbol,'strict':choices[0][0] if len(choices)==1 else None,'coding_sensitivity':choices[0][0] if len(choices)==1 else coding[0] if len(coding)==1 else None,'status':'unique' if len(choices)==1 else 'missing' if not choices else 'ambiguous','candidate_count':len(choices)})
 return rows
def repeat_design():
 data=json.loads((ROOT/'prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json').read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
 return {(p,l) for p in REPEAT_PLATES for l in LABELS if p in data[l]}
def num(v):
 try:
  if v is None or isinstance(v,bool):return None
  x=float(v)
  return x if np.isfinite(x) else None
 except (ValueError,TypeError):return None
def load(cp,fp):
 cp=safe(cp);fp=safe(fp);issues=[];fatal=[]
 w=load_workbook(cp,read_only=True,data_only=True);sh=w.active;sh.reset_dimensions();it=sh.iter_rows(values_only=True);header=next(it)
 if list(header[:3])!=['gene_id','gene_symbol','gene_biotype']:raise ValueError('bad fake gene headers')
 rna={}
 for j,raw in enumerate(header[3:],3):
  match=re.fullmatch(r'(.+)_(P\d+)_([A-H]\d{2})',str(raw))
  if not match:raise ValueError('bad fake RNA key')
  label,p,well=match.groups();key=(p,well)
  if key in rna:raise ValueError('duplicate fake RNA key')
  rna[key]=(j,label)
 funcs={};book=load_workbook(fp,read_only=True,data_only=True)
 for tm,sheet in [('04H','Tab 1 Functional data 04H'),('24H','Tab 2 Functional data 24H')]:
  ws=book[sheet];ws.reset_dimensions();rows=ws.iter_rows(values_only=True);hh=next(rows);ix={v:i for i,v in enumerate(hh)}
  if not {'Treatment','Wellname','Dataset','Plate_timepoint','Force'}<=set(ix):raise ValueError('fake function headers')
  for rowno,row in enumerate(rows,2):
   label,well,ds,pt,val=[row[ix[k]] if ix[k]<len(row) else None for k in ('Treatment','Wellname','Dataset','Plate_timepoint','Force')]
   if ds not in ('Cardiopedia-Ligand','Independent_Experiment'):continue
   if not all(isinstance(x,str) and x.strip()==x and x for x in (label,well,pt)):
    issues.append({'kind':'invalid_metadata','dataset':ds,'time':tm,'row':rowno});continue
   plate=pt.split('_',1)[0];valid_label=(label in LABELS and ((ds=='Cardiopedia-Ligand' and plate in PLATES) or (ds=='Independent_Experiment' and plate in REPEAT_PLATES))) or (label in CONTROLS and ds=='Cardiopedia-Ligand' and plate in PLATES)
   if not valid_label:continue
   if pt!=plate+'_'+tm:
    issues.append({'kind':'time_mismatch','dataset':ds,'time':tm,'plate':plate,'label':label,'well':well});continue
   key=(ds,tm,plate,well);funcs.setdefault(key,[]).append((label,num(val)))
 book.close()
 selected={};controls={};repeat={};expected=repeat_design()
 for ds,tm,p,well in list(funcs):
  entries=funcs[(ds,tm,p,well)]
  if len(entries)!=1:
   labels={v[0] for v in entries}
   issues.append({'kind':'duplicate_key','dataset':ds,'time':tm,'plate':p,'well':well,'label':next(iter(labels)) if len(labels)==1 else None});continue
  label,value=entries[0]
  if value is None:issues.append({'kind':'invalid_force','dataset':ds,'time':tm,'plate':p,'label':label,'well':well})
  key=(p,label)
  if ds=='Cardiopedia-Ligand':
   if (p,well) not in rna or rna[(p,well)][1]!=label:
    issues.append({'kind':'rna_label_or_key_mismatch','dataset':ds,'time':tm,'plate':p,'label':label,'well':well});continue
   dest=controls if label in CONTROLS else selected
   unique=(tm,p,label,well) if label in CONTROLS else (tm,p,label)
   if unique in dest:issues.append({'kind':'duplicate_cell','dataset':ds,'time':tm,'plate':p,'label':label,'well':well});continue
   dest[unique]=(rna[(p,well)][0],value,well)
  else:
   if key not in expected:issues.append({'kind':'unexpected_repeat_cell','dataset':ds,'time':tm,'plate':p,'label':label,'well':well});continue
   unique=(tm,p,label)
   if unique in repeat:issues.append({'kind':'duplicate_cell','dataset':ds,'time':tm,'plate':p,'label':label,'well':well});continue
   repeat[unique]=(value,well)
 for tm in ('04H','24H'):
  for p in PLATES:
   for label in LABELS:
    if (tm,p,label) not in selected:issues.append({'kind':'missing_main_cell','dataset':'Cardiopedia-Ligand','time':tm,'plate':p,'label':label})
  for p,label in expected:
   if (tm,p,label) not in repeat:issues.append({'kind':'missing_repeat_cell','dataset':'Independent_Experiment','time':tm,'plate':p,'label':label})
 for p in PLATES:
  for label in LABELS:
   a=selected.get(('04H',p,label));b=selected.get(('24H',p,label))
   if a and b and (a[0],a[2])!=(b[0],b[2]):issues.append({'kind':'main_time_key_mismatch','plate':p,'label':label})
 for p,label in expected:
  a=repeat.get(('04H',p,label));b=repeat.get(('24H',p,label))
  if a and b and a[1]!=b[1]:issues.append({'kind':'repeat_time_key_mismatch','plate':p,'label':label})
 for tm,p,label,well in list(controls):
  other=('24H' if tm=='04H' else '04H',p,label,well)
  if other not in controls:issues.append({'kind':'control_time_key_mismatch','dataset':'Cardiopedia-Ligand','time':tm,'plate':p,'label':label,'well':well})
 mainkeys=[(p,l) for p in PLATES for l in LABELS];controlkeys=sorted(k for k in controls if k[0]=='24H')
 if len([1 for p,l in mainkeys if ('24H',p,l) in selected])!=235:
  w.close();return {'issues':issues,'main_status':'unavailable_identity','repeat_status':'unavailable' if any(i.get('dataset')=='Independent_Experiment' for i in issues) else 'ok','control_count':len(controlkeys)},None
 cols=[selected[('24H',p,l)][0] for p,l in mainkeys]+[controls[k][0] for k in controlkeys]
 ids=[];symbols=[];types=[];matrix=[];control_bad=[]
 for rowno,row in enumerate(it,2):
  gene,symbol,kind=row[:3]
  if not isinstance(gene,str) or not gene or gene.strip()!=gene:raise ValueError('invalid fake gene ID')
  ids.append(gene);symbols.append(symbol);types.append(kind);vals=[]
  for colindex,col in enumerate(cols):
   v=num(row[col] if col<len(row) else None)
   if v is None or v<0:
    problem={'kind':'invalid_count','gene_id':gene,'column':col+1,'row':rowno,'branch':'main' if colindex<235 else 'control'}
    (fatal if colindex<235 else control_bad).append(problem);v=None
   vals.append(v if v is not None else np.nan)
  matrix.append(vals)
 w.close()
 if len(set(ids))!=len(ids) or not ids:raise ValueError('duplicate fake gene ID')
 if fatal:issues.extend(fatal+control_bad);return {'issues':issues,'main_status':'unavailable_counts','repeat_status':'unavailable' if any(i.get('dataset')=='Independent_Experiment' for i in issues) else 'ok','control_count':len(controlkeys)},None
 issues.extend(control_bad)
 x=np.asarray(matrix,dtype=float).T;main=x[:235].reshape(5,47,-1);ctrl=x[235:];y=np.array([selected[('24H',p,l)][1] if selected[('24H',p,l)][1] is not None else np.nan for p,l in mainkeys]).reshape(5,47)
 main_issues=[i for i in issues if (i.get('dataset')=='Cardiopedia-Ligand' and i.get('label') not in CONTROLS and not(i.get('kind')=='invalid_force' and i.get('time')=='04H')) or i.get('kind')=='main_time_key_mismatch']
 rep_issues=[i for i in issues if (i.get('dataset')=='Independent_Experiment' and not(i.get('kind')=='invalid_force' and i.get('time')=='04H')) or i.get('kind')=='repeat_time_key_mismatch']
 early=np.array([selected[('04H',p,l)][1] if ('04H',p,l) in selected and selected[('04H',p,l)][1] is not None else np.nan for p,l in mainkeys]).reshape(5,47)
 control_issues=control_bad or any(i.get('label') in CONTROLS for i in issues)
 return {'issues':issues,'main_status':'unavailable_main_qc' if main_issues or not np.isfinite(y).all() else 'ok','early_status':'ok' if np.isfinite(early).all() else 'unavailable_early_values','repeat_status':'unavailable_repeat_qc' if rep_issues else 'ok','control_status':'unavailable_control_qc' if control_issues else 'ok','control_count':len(controlkeys),'main_cells':len(mainkeys),'repeat_expected_cells':len(expected),'control_keys':[list(k) for k in controlkeys]},(main,y,ctrl,ids,symbols,types,repeat,early)
def rho(x,y):
 a=rankdata(x);b=rankdata(y)
 return None if a.std()==0 or b.std()==0 else float(np.corrcoef(a,b)[0,1])
def diagnostics(score,force):
 s=np.asarray(score,dtype=float);y=np.asarray(force,dtype=float)
 if s.shape!=(5,47) or y.shape!=(5,47) or not np.isfinite(s).all() or not np.isfinite(y).all():return {'status':'unavailable_incomplete'}
 sm=s.mean(axis=0);mean=y.mean(axis=0);med=np.median(y,axis=0);geo=np.exp(np.log(y).mean(axis=0)) if np.all(y>0) else None
 order=np.argsort(sm,kind='stable');q=11;tied=sm[order[q-1]]==sm[order[q]] or sm[order[-q-1]]==sm[order[-q]]
 return {'status':'ok','mean_rho':rho(sm,mean),'median_rho':rho(sm,med),'geometric_rho':rho(sm,geo) if geo is not None else None,'geometric_status':'ok' if geo is not None else 'unavailable_nonpositive','plate_rho':[rho(s[i],y[i]) for i in range(5)],'leave_one_plate_out_rho':[rho(np.delete(s,i,0).mean(axis=0),np.delete(y,i,0).mean(axis=0)) for i in range(5)],'high_minus_low':None if tied else float(np.median(mean[order[-q:]])-np.median(mean[order[:q]])),'high_low_status':'unavailable_tied_boundary' if tied else 'ok','force_quantiles':np.quantile(mean,[0,.25,.5,.75,1]).tolist()}
