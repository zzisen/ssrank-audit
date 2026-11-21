"""Synthetic-only streaming RNA feature QA. No function data, scores or associations."""
import argparse,collections,csv,gzip,importlib.util,json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
PROGRAM_NAMES=tuple(f'FAKE_PROGRAM_{i}' for i in range(7))
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
b=load('b119_for_qc143','simulations/p41_v3_b6_runner_119.py')

def safe(path):
    if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise PermissionError('synthetic guard')
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'reports/runs').resolve()):raise PermissionError('synthetic paths only')
    return path

def inspect(paths):
    paths={k:safe(paths[k]) for k in ('counts','samples','signature','programs','gtf')}
    with paths['samples'].open(encoding='utf-8-sig',newline='') as f:samples=list(csv.DictReader(f,delimiter='\t'))
    ids=[r.get('SampleID') for r in samples]
    if not ids or any(not x for x in ids) or len(set(ids))!=len(ids):raise ValueError('sample IDs')
    if any(r.get('Excluded') not in ('TRUE','FALSE') for r in samples):raise ValueError('author QC flags')
    keep=[i for i,r in enumerate(samples) if r['Excluded']=='FALSE']
    if len(keep)<2:raise ValueError('too few retained samples')
    names,w=b.sig.load_known_weights(paths['signature']);unknown=b.sig.signature_audit['explicit_NA_rows']
    requested=PROGRAM_NAMES;sets=b.primary.read_programs(paths['programs'],requested)
    programs=[('main',names,w,unknown)]+[(name,sets[name],np.ones(len(sets[name])),0) for name in requested]
    b._ACTIVE_INPUTS={'gtf':paths['gtf']}
    try:resolve=b.annotation(paths['gtf'])
    finally:b._ACTIVE_INPUTS=None
    network=collections.defaultdict(list);den=[];mapped=[]
    for j,(_,genes,weights,_) in enumerate(programs):
        weights=np.asarray(weights,float)
        if len(set(genes))!=len(genes) or not np.isfinite(weights).all() or np.abs(weights).sum()==0:raise ValueError('invalid program')
        den.append([float(np.abs(weights).sum()),float(weights[weights>0].sum()),float(-weights[weights<0].sum())])
        seen=set()
        for gene,weight in zip(genes,weights):
            gid=resolve(gene)
            if gid is None or weight==0:continue
            if gid in seen:raise ValueError('many-to-one program mapping')
            seen.add(gid);network[gid].append((j,float(weight)))
        mapped.append(len(seen))
    n=len(ids);libraries=np.zeros(n);seen=set();row_count=0
    with gzip.open(paths['counts'],'rt',encoding='utf-8-sig') as f:
        h=f.readline().rstrip('\r\n').split('\t');observed=h[1:]
        if h[0]!='Ensembl' or len(observed)!=n or len(set(observed))!=n or set(observed)!=set(ids):raise ValueError('count header')
        source={sid:i for i,sid in enumerate(observed)};order=np.array([source[ids[i]] for i in keep])
        for line in f:
            parts=line.rstrip('\r\n').split('\t',1)
            if len(parts)!=2 or not parts[0] or parts[0] in seen:raise ValueError('gene rows')
            seen.add(parts[0]);libraries+=b.parse_row(parts[1],n);row_count+=1
    if not row_count or np.any(libraries<=0) or not np.isfinite(libraries).all():raise ValueError('library sizes')
    masses=np.zeros((len(programs),3));usable=np.zeros(len(programs),int)
    with gzip.open(paths['counts'],'rt',encoding='utf-8-sig') as f:
        f.readline()
        for line in f:
            gene,raw=line.rstrip('\r\n').split('\t',1)
            if gene not in network:continue
            vals=b.parse_row(raw,n)[order];v=np.log2(1+1e6*vals/libraries[order]);sd=v.std(ddof=1)
            if not np.isfinite(sd) or sd<=1e-12:continue
            for j,weight in network[gene]:
                masses[j]+=[abs(weight),max(weight,0),max(-weight,0)];usable[j]+=1
    out={}
    for j,(name,genes,weights,unknown) in enumerate(programs):
        coverage={k:(None if den[j][i]==0 else float(masses[j,i]/den[j][i])) for i,k in enumerate(('total','positive','negative'))}
        out[name]={'coverage_known_weights':coverage,'status':'unavailable_coverage' if any(v is not None and v<.8 for v in coverage.values()) or masses[j,0]<=0 else 'pass_engineering_coverage',
                   'known_gene_rows':len(genes),'unknown_weight_rows':unknown,'unique_annotation_candidates':mapped[j],'usable_gene_rows':int(usable[j])}
    return {'status':'completed_synthetic_qc','synthetic_only':True,'gene_rows':row_count,'sample_columns':n,'retained':len(keep),
            'signature_format':dict(b.sig.signature_audit),'programs':out,'score_or_function_evaluated':False}

def main():
    p=argparse.ArgumentParser()
    for k in ('counts','samples','signature','programs','gtf','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();out=safe(a.output)
    if out.exists():raise FileExistsError('no overwrite')
    result=inspect({k:getattr(a,k) for k in ('counts','samples','signature','programs','gtf')})
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'score_or_function_evaluated':False}))
if __name__=='__main__':main()
