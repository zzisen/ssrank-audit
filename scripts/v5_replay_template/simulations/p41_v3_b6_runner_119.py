"""V3 B6 condition-level runner. Real mode needs a separate explicit permit."""
from __future__ import annotations
import argparse
import collections
import csv
import gzip
import hashlib
import importlib.util
import json
import os
import re
import warnings
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PERMIT=ROOT/'seal/REAL_SCIENTIFIC_USER_AUTHORIZATION_V3_B6.json'
MANIFEST=ROOT/'theory/V3_B6_RUNNER_119_REVIEW_MANIFEST.json'
SCOPE='P41_V3_B6_CONDITION_DESCRIPTIVE_ONLY'
def module(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
planmod=module('b6plan116','simulations/p41_v3_b6_plan_116.py')
primary=module('primary118for119','simulations/p41_v3_primary_runner_118.py')
sig=primary.sig
REAL_PROGRAMS=primary.REAL_PROGRAMS

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def real_gate():
    if not PERMIT.is_file():raise PermissionError('real B6 scientific analysis not authorized: permit absent')
    permit=json.loads(PERMIT.read_text(encoding='utf8'))
    if (permit.get('scope')!=SCOPE or permit.get('explicit_user_authorization') is not True or
        permit.get('manifest_sha256')!=digest(MANIFEST)):
        raise PermissionError('real B6 scientific analysis not authorized: permit invalid')
def real_paths():
    real_gate()
    doc=json.loads(MANIFEST.read_text(encoding='utf8'));paths={}
    for role,item in doc['inputs'].items():
        path=Path(item['path']).resolve()
        if digest(path)!=item['sha256']:raise ValueError('input hash mismatch: '+role)
        paths[role]=path
    for item in doc['implementation']:
        p=(ROOT/item['path']).resolve()
        if not p.is_relative_to(ROOT) or digest(p)!=item['sha256']:raise ValueError('implementation hash mismatch')
    return paths
def synthetic_path(path):
    if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise PermissionError('synthetic guard absent')
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'reports/runs').resolve()):raise PermissionError('synthetic path outside runs')
    return path
def parse_row(raw,n):
    if raw.count('\t')!=n-1:raise ValueError('invalid count field count')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',DeprecationWarning);v=np.fromstring(raw,sep='\t')
    except (ValueError,DeprecationWarning):raise ValueError('invalid numeric count text') from None
    if len(v)!=n or not np.isfinite(v).all() or np.any(v<0):raise ValueError('invalid numeric count')
    return v
_ACTIVE_INPUTS=None
def require_active(path,role):
    if _ACTIVE_INPUTS is None or Path(path).resolve()!=_ACTIVE_INPUTS.get(role):
        raise PermissionError('B6 file helper requires approved execution context')
def annotation(path):
    require_active(path,'gtf')
    by=collections.defaultdict(set);reverse=collections.defaultdict(set)
    with gzip.open(path,'rt',encoding='utf8') as f:
        for line in f:
            if line.startswith('#'):continue
            fields=line.rstrip('\n').split('\t')
            if len(fields)!=9 or fields[2]!='gene':continue
            attrs=dict(re.findall(r'([A-Za-z_]+) "([^"]*)"',fields[8]));gid=attrs.get('gene_id');name=attrs.get('gene_name')
            if gid and name:by[name].add(gid);reverse[gid].add(name)
    def resolve(name):
        ids=by.get(name,set())
        if len(ids)!=1:return None
        gid=next(iter(ids));return gid if len(reverse[gid])==1 else None
    return resolve

def build_metadata_plan(samples,assays):
    transformed=[]
    for row in samples:
        if row.get('Excluded') not in ('TRUE','FALSE'):raise ValueError('invalid excluded flag')
        if row.get('Treatment') in ('DMSO','NoTreatment'):continue
        transformed.append({'LibraryID':row.get('LibraryID'),'Line':row.get('iPSC Line'),'Batch':row.get('iPSC Batch'),
                            'Treatment':row.get('Treatment'),'Rank':row.get('Concentration Rank'),
                            'Time':row.get('Timepoint'),'Block':row.get('Triplicate Block'),
                            'sample_id':row.get('SampleID'),'Excluded':row['Excluded']})
    selected=[r for r in assays if r.get('Assay') in ('Amplitude','DNA structure')]
    metadata=[{'assay':r.get('Assay'),'treatment':r.get('Treatment'),'label':r.get('Label'),
               'rank':r.get('Concentration Rank')} for r in selected]
    return planmod.enforce_min_ranks(planmod.build_plan(transformed,metadata)),selected

def score_stream(path,samples,programs,resolver):
    require_active(path,'counts')
    sample_ids=[r.get('SampleID') for r in samples]
    if not sample_ids or any(not x for x in sample_ids) or len(set(sample_ids))!=len(sample_ids):raise ValueError('sample ID structure')
    n=len(sample_ids);metadata_keep=[i for i,r in enumerate(samples) if r['Excluded']=='FALSE']
    network=collections.defaultdict(list);denoms=[];mapped=[]
    for j,(_,names,weights,_) in enumerate(programs):
        w=np.asarray(weights,float)
        if len(names)!=len(w) or len(set(names))!=len(names) or not np.isfinite(w).all() or not np.any(w):raise ValueError('invalid known program')
        denoms.append({'total':float(np.abs(w).sum()),'positive':float(w[w>0].sum()),'negative':float(-w[w<0].sum())})
        count=0;seen_gid=set()
        for name,weight in zip(names,w):
            gid=resolver(name)
            if gid is None or weight==0:continue
            if gid in seen_gid:raise ValueError('many-to-one program annotation')
            seen_gid.add(gid);network[gid].append((j,float(weight)));count+=1
        mapped.append(count)
    libs=np.zeros(n);seen=set();rows=0
    with gzip.open(path,'rt',encoding='utf-8-sig') as f:
        h=f.readline().rstrip('\r\n').split('\t');observed=h[1:]
        if h[0]!='Ensembl' or len(observed)!=n or len(set(observed))!=n or set(observed)!=set(sample_ids):raise ValueError('count sample header mismatch')
        source_index={s:i for i,s in enumerate(observed)}
        order=np.asarray([source_index[sample_ids[i]] for i in metadata_keep],dtype=int)
        for line in f:
            fields=line.rstrip('\r\n').split('\t',1)
            if len(fields)!=2 or not fields[0] or fields[0] in seen:raise ValueError('invalid or duplicate gene ID')
            seen.add(fields[0]);libs+=parse_row(fields[1],n);rows+=1
    if not rows or not np.isfinite(libs).all() or np.any(libs<=0) or len(order)<2:raise ValueError('invalid count libraries or retained sample count')
    acc=np.zeros((len(order),len(programs)));mass=np.zeros((len(programs),3));usable=np.zeros(len(programs),int)
    with gzip.open(path,'rt',encoding='utf-8-sig') as f:
        f.readline()
        for line in f:
            gene,raw=line.rstrip('\r\n').split('\t',1)
            if gene not in network:continue
            vals=parse_row(raw,n)[order];v=np.log2(1+1e6*vals/libs[order]);sd=v.std(ddof=1)
            if not np.isfinite(sd) or sd<=1e-12:continue
            z=(v-v.mean())/sd
            for j,w in network[gene]:
                acc[:,j]+=w*z;mass[j,0]+=abs(w);mass[j,1]+=max(w,0);mass[j,2]+=max(-w,0);usable[j]+=1
    results=[]
    for j,(name,names,weights,unknown) in enumerate(programs):
        cov={k:(float(mass[j,i]/denoms[j][k]) if denoms[j][k]>0 else None) for i,k in enumerate(('total','positive','negative'))}
        status='unavailable_coverage' if any(v is not None and v<.8 for v in cov.values()) or mass[j,0]<=0 else 'ok'
        results.append({'name':name,'status':status,'coverage_known_weights':cov,'known_genes':len(names),
                        'unknown_weight_count':unknown,'unique_annotation_candidates':mapped[j],
                        'usable_genes':int(usable[j]),'scores':(acc[:,j]/mass[j,0] if status=='ok' else None)})
    return results,[sample_ids[i] for i in metadata_keep],{'gene_rows':rows,'samples':n,'retained':len(metadata_keep),'header_order_matches_metadata':observed==sample_ids}

def _evaluate_impl(paths,program_names,mode):
    for role in ('counts','samples','assays','signature','programs','gtf'):require_active(paths[role],role)
    with paths['samples'].open(encoding='utf-8-sig',newline='') as f:samples=list(csv.DictReader(f,delimiter='\t'))
    with paths['assays'].open(encoding='utf-8-sig',newline='') as f:assays=list(csv.DictReader(f,delimiter='\t'))
    plan,selected_assays=build_metadata_plan(samples,assays) # No Value or count used in the plan.
    known,weights=sig.load_known_weights(paths['signature']);unknown=sig.signature_audit['explicit_NA_rows']
    gene_sets=primary.read_programs(paths['programs'],program_names)
    programs=[('main',known,weights,unknown)]+[(name,gene_sets[name],np.ones(len(gene_sets[name])),0) for name in program_names]
    streams,retained_ids,stream_audit=score_stream(paths['counts'],samples,programs,annotation(paths['gtf']))
    records={};values=[r.get('Value') for r in selected_assays]
    for stream in streams:
        scores=stream.pop('scores');name=stream['name']
        if scores is None:
            stream['conditions']=[];stream['endpoints']={'status':'unavailable_score'}
        else:
            score_map={sid:float(v) for sid,v in zip(retained_ids,scores)}
            stream['conditions']=planmod.summarize(plan,score_map,values)
            stream['endpoints']=planmod.endpoint_summary(stream['conditions'])
            stream['scores_by_retained_sample']=scores.tolist()
        records[name]=stream
    return {'status':'completed_descriptive','mode':mode,'scope':SCOPE,'signature_format':dict(sig.signature_audit),
            'plan':plan,'stream_audit':stream_audit,'retained_sample_ids':retained_ids,'programs':records,
            'limits':['retrospective transductive summary','assay time/line/physical dose may not be identifiable',
                      'one functional value per condition, not per RNA replicate','no donor or treatment population inference'],
            'scientific_association_performed':True}

def evaluate(paths,program_names,mode):
    global _ACTIVE_INPUTS
    if _ACTIVE_INPUTS is not None:raise PermissionError('nested B6 execution is not allowed')
    roles=('counts','samples','assays','signature','programs','gtf')
    if mode=='synthetic':
        paths={role:synthetic_path(paths[role]) for role in roles}
    elif mode=='real':
        real_gate();approved=real_paths()
        if any(Path(paths[role]).resolve()!=approved[role] for role in roles) or tuple(program_names)!=REAL_PROGRAMS:
            raise PermissionError('real B6 inputs or design not approved')
        paths={role:approved[role] for role in roles}
    else:raise PermissionError('unknown analysis mode')
    _ACTIVE_INPUTS={role:Path(paths[role]).resolve() for role in roles}
    try:return _evaluate_impl(paths,program_names,mode)
    finally:_ACTIVE_INPUTS=None

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('synthetic','real'),required=True);p.add_argument('--output',type=Path,required=True)
    for name in ('counts','samples','assays','signature','programs','gtf'):p.add_argument('--'+name,type=Path)
    z=p.parse_args()
    if z.mode=='real':
        real_gate();paths=real_paths();out=z.output.resolve()
        if not out.is_relative_to((ROOT/'reports/real_analysis').resolve()):raise PermissionError('real output outside analysis directory')
        names=REAL_PROGRAMS
    else:
        if any(getattr(z,name) is None for name in ('counts','samples','assays','signature','programs','gtf')):raise ValueError('synthetic paths required')
        paths={name:synthetic_path(getattr(z,name)) for name in ('counts','samples','assays','signature','programs','gtf')}
        out=synthetic_path(z.output);names=tuple(f'FAKE_PROGRAM_{i}' for i in range(7))
    if out.exists():raise FileExistsError('refusing to overwrite output')
    result=evaluate(paths,names,z.mode)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({'mode':z.mode,'scope':SCOPE,'status':result['status'],'scientific_association_performed':True}))

if __name__=='__main__':main()
