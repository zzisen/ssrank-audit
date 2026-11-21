"""V3 3D descriptive runner. Real mode remains denied without a separate user permit.

This is not authorization or registration. Do not create the permit without the
user's explicit subsequent instruction for the concrete manifest and scope.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import numpy as np
from openpyxl import load_workbook

ROOT=Path(__file__).resolve().parents[1]
PERMIT=ROOT/'seal/REAL_SCIENTIFIC_USER_AUTHORIZATION_V3_3D.json'
MANIFEST=ROOT/'theory/V3_3D_RUNNER_118_REVIEW_MANIFEST.json'
SCOPE='P41_V3_3D_DESCRIPTIVE_ONLY'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
adapter=module('runner115_adapter','simulations/p41_gate_adapter_86.py')
core=module('runner115_core','simulations/p41_v3_primary_core_114.py')
sig=module('runner115_signature','simulations/p41_qc_entry_105_known_weights.py')
PLATES=('P01','P03','P04','P05','P06')
REAL_PROGRAMS=('structural_exclusive','conduction_exclusive','ecm','hypoxia','oxphos','upr','fibroblast_marker','structural_original','conduction_original')

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def real_gate():
    # No real inputs or output directories are opened before this check.
    if not PERMIT.is_file():raise PermissionError('real scientific analysis not authorized: permit absent')
    permit=json.loads(PERMIT.read_text(encoding='utf8'))
    if (permit.get('scope')!=SCOPE or permit.get('explicit_user_authorization') is not True or
        permit.get('manifest_sha256')!=digest(MANIFEST)):
        raise PermissionError('real scientific analysis not authorized: permit invalid')
    return permit

def real_paths():
    real_gate()
    doc=json.loads(MANIFEST.read_text(encoding='utf8'));paths={}
    for role,item in doc['inputs'].items():
        p=Path(item['path']).resolve()
        if digest(p)!=item['sha256']:raise ValueError('input hash mismatch: '+role)
        paths[role]=p
    for item in doc['implementation']:
        p=(ROOT/item['path']).resolve()
        if not p.is_relative_to(ROOT) or digest(p)!=item['sha256']:raise ValueError('implementation hash mismatch')
    return paths

def synthetic_path(p):
    if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise PermissionError('synthetic guard absent')
    p=Path(p).resolve()
    if not p.is_relative_to((ROOT/'reports/runs').resolve()):raise PermissionError('synthetic path outside runs')
    return p

def normalized_workbook(path,labels):
    path=Path(path).resolve()
    if os.getenv('P41_PYTHON_SEAL_GUARD')=='active':synthetic_path(path)
    else:
        real_gate()
        if path!=real_paths()['function']:raise PermissionError('function path not approved')
    book=load_workbook(path);changed=0
    for time,sheet in [('04H','Tab 1 Functional data 04H'),('24H','Tab 2 Functional data 24H')]:
        ws=book[sheet];h={c.value:c.column for c in ws[1]};wells=set();conditions=set()
        if not {'Treatment','Wellname','Dataset','Plate_timepoint','Force'}<=set(h):raise ValueError('function columns')
        for row in range(2,ws.max_row+1):
            def get(k):return ws.cell(row,h[k]).value
            ds,name,well,pt=(get(k) for k in ('Dataset','Treatment','Wellname','Plate_timepoint'))
            if not all(isinstance(v,str) and v for v in (ds,name,well,pt)):continue
            clean=name.strip();plate=pt.split('_',1)[0]
            if clean!=name:ws.cell(row,h['Treatment']).value=clean;changed+=1
            if ds not in ('Cardiopedia-Ligand','Independent_Experiment') or clean not in set(labels)|set(adapter.CONTROLS):continue
            wk=(ds,time,plate,well)
            if wk in wells:raise ValueError('duplicate selected function well')
            wells.add(wk)
            if clean not in adapter.CONTROLS:
                ck=(ds,time,plate,clean)
                if ck in conditions:raise ValueError('duplicate normalized selected condition')
                conditions.add(ck)
    data=io.BytesIO();book.save(data);book.close();return data.getvalue(),changed

def read_programs(path,requested):
    path=Path(path).resolve()
    fixed=(ROOT/'theory/fixed_programs_2026-11-17/fixed_programs.gmt').resolve()
    if path!=fixed:synthetic_path(path)
    found={}
    for line in path.read_text(encoding='utf-8-sig').splitlines():
        fields=line.split('\t')
        while len(fields)>2 and fields[-1]=='':fields.pop()
        if fields[0] not in requested:continue
        if fields[0] in found or len(fields)<3 or any(not g or g.strip()!=g for g in fields[2:]) or len(set(fields[2:]))!=len(fields[2:]):
            raise ValueError('invalid program definition')
        found[fields[0]]=fields[2:]
    if set(found)!=set(requested):raise ValueError('missing program definition')
    return found

def evaluate(paths,labels,repeat_map,program_names,mode):
    if mode=='synthetic':
        paths={k:synthetic_path(paths[k]) for k in ('counts','function','signature','programs')}
    elif mode=='real':
        real_gate();approved=real_paths()
        if any(Path(paths[k]).resolve()!=approved[k] for k in ('counts','function','signature','programs')):
            raise PermissionError('real inputs not approved')
        expected_labels=sorted(json.loads(approved['incidence'].read_text(encoding='utf8'))['noncontrol_labels_on_all_five'])
        expected_repeat=json.loads(approved['repeat_metadata'].read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
        if list(labels)!=expected_labels or repeat_map!=expected_repeat or tuple(program_names)!=REAL_PROGRAMS:
            raise PermissionError('real design not approved')
    else:raise PermissionError('unknown analysis mode')
    if len(labels)!=47 or len(set(labels))!=47 or tuple(labels)!=tuple(sorted(labels)):raise ValueError('candidate label list')
    normalized,changed=normalized_workbook(paths['function'],labels)
    adapter.LABELS=tuple(labels)
    expected={(p,labels[i]) for i in range(47) for p in repeat_map[f'FAKE_L{i:02d}']}
    adapter.repeat_design=lambda:expected
    allowed={paths['counts'].resolve(),paths['function'].resolve()}
    def checked(p):
        p=Path(p).resolve()
        if p not in allowed:raise PermissionError('unexpected adapter input')
        return p
    def book(p,*args,**kwargs):
        return load_workbook(io.BytesIO(normalized) if Path(p).resolve()==paths['function'].resolve() else p,*args,**kwargs)
    oldsafe,oldloader=adapter.safe,adapter.load_workbook
    adapter.safe,adapter.load_workbook=checked,book
    try:audit,arr=adapter.load(paths['counts'],paths['function'])
    finally:adapter.safe,adapter.load_workbook=oldsafe,oldloader
    if arr is None or audit['main_status']!='ok':
        return {'status':'unavailable_main_input','audit':audit,'normalized_treatment_cells':changed}
    counts,late,controls,ids,symbols,biotypes,repeat,early=arr
    log=core.log_cpm(counts);known,weights=sig.load_known_weights(paths['signature']);unknown=sig.signature_audit['explicit_NA_rows']
    programs=read_programs(paths['programs'],program_names)
    by_id={g:i for i,g in enumerate(ids)}
    def score(names,w,unknown_count=0):
        mapping=adapter.resolve(ids,symbols,biotypes,names)
        index=[None if x['strict'] is None else by_id[x['strict']] for x in mapping]
        result=core.score_program(log,w,index,unknown_count=unknown_count)
        result['mapping_status']={k:sum(x['status']==k for x in mapping) for k in ('unique','missing','ambiguous')}
        return result
    keys=audit['control_keys']
    if len(keys)!=len(controls):raise ValueError('control vector/key mismatch')
    wm=[np.asarray([controls[i] for i,k in enumerate(keys) if k[1]==p and k[2]=='CTRL-WM']).reshape(-1,len(ids)) for p in PLATES]
    repeat_values=None
    if audit['repeat_status']=='ok':
        candidate=[[v for (tm,p,l),(v,well) in repeat.items() if tm=='24H' and l==label] for label in labels]
        if all(group and all(v is not None and np.isfinite(v) for v in group) for group in candidate):repeat_values=candidate
    def report_score(result):
        values=result.pop('scores')
        if values is not None:
            result['description']=core.describe_program(values,late,early,counts,wm,repeat_values)
            result['scores_by_plate_and_treatment']=values.tolist()
        else:result['description']={'status':'unavailable_score'}
        return result
    main=report_score(score(known,weights,unknown))
    comparisons={}
    for name in program_names:
        genes=programs[name]
        comparisons[name]=report_score(score(genes,np.ones(len(genes))))
        comparisons[name]['overlap_with_known_signature_names']=len(set(genes)&set(known))
    return {'status':'completed_descriptive','mode':mode,'scope':SCOPE,'labels':list(labels),
            'plates':list(PLATES),'normalized_treatment_cells':changed,'signature_format':dict(sig.signature_audit),
            'audit':audit,'main':main,'comparators':comparisons,
            'late_function_by_plate_and_treatment':late.tolist(),
            'early_function_by_plate_and_treatment':[[None if not np.isfinite(v) else float(v) for v in row] for row in early],
            'limits':['retrospective, transductive, fixed conditions','normalized function and unknown reference-pool scope',
                      'no causal claim, independent donor claim, new-drug generalization, p value or confidence interval',
                      'candidate published finite-weight score is not the original complete MOFA model'],
            'scientific_association_performed':True}

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--counts',type=Path);p.add_argument('--function',type=Path);p.add_argument('--signature',type=Path);p.add_argument('--programs',type=Path)
    z=p.parse_args()
    if z.mode=='real':
        real_gate();paths=real_paths();out=z.output.resolve()
        if not out.is_relative_to((ROOT/'reports/real_analysis').resolve()):raise PermissionError('real output outside analysis directory')
        labels=sorted(json.loads(paths['incidence'].read_text(encoding='utf8'))['noncontrol_labels_on_all_five'])
        repeat_map=json.loads(paths['repeat_metadata'].read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
        program_names=REAL_PROGRAMS
    else:
        if any(v is None for v in (z.counts,z.function,z.signature,z.programs)):raise ValueError('synthetic paths required')
        paths={k:synthetic_path(getattr(z,k)) for k in ('counts','function','signature','programs')};out=synthetic_path(z.output)
        labels=[f'FAKE_L{i:02d}' for i in range(47)]
        repeat_map=json.loads((ROOT/'prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json').read_text(encoding='utf8'))['anonymous_repeat_plate_incidence']
        program_names=tuple(f'FAKE_PROGRAM_{i}' for i in range(7))
    if out.exists():raise FileExistsError('refusing to overwrite analysis output')
    result=evaluate(paths,labels,repeat_map,program_names,z.mode)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'mode':z.mode,'status':result['status'],'scope':SCOPE,'scientific_association_performed':result.get('scientific_association_performed',False)}))

if __name__=='__main__':main()
