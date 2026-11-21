"""Synthetic-only RNA variance/coverage QC, with no functional data input."""
import argparse,importlib.util,json,os,re
from pathlib import Path
import numpy as np
from openpyxl import load_workbook
ROOT=Path(__file__).resolve().parents[1]
def load(name,path):
    s=importlib.util.spec_from_file_location(name,ROOT/path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
adapter=load('adapter86_for142','simulations/p41_gate_adapter_86.py')
signature=load('sig105_for142','simulations/p41_qc_entry_105_known_weights.py')
qc=load('qc141_for142','simulations/p41_feature_qc_141.py')
PLATES=('P01','P03','P04','P05','P06');LABELS=tuple(f'FAKE_L{i:02d}' for i in range(47))

def safe(path):
    if os.getenv('P41_PYTHON_SEAL_GUARD')!='active':raise PermissionError('synthetic guard required')
    path=Path(path).resolve()
    if not path.is_relative_to((ROOT/'reports/runs').resolve()):raise PermissionError('synthetic paths only')
    return path

def inspect(counts,signature_path):
    counts=safe(counts);signature_path=safe(signature_path)
    book=load_workbook(counts,read_only=True,data_only=True);ws=book.active;ws.reset_dimensions()
    it=ws.iter_rows(values_only=True);header=next(it)
    if list(header[:3])!=['gene_id','gene_symbol','gene_biotype']:raise ValueError('header')
    columns={}
    for col,raw in enumerate(header[3:],3):
        match=re.fullmatch(r'(.+)_(P\d+)_([A-H]\d{2})',str(raw))
        if not match:raise ValueError('invalid RNA header')
        label,plate,well=match.groups()
        if label not in LABELS or plate not in PLATES:continue
        key=(plate,label)
        if key in columns:raise ValueError('duplicate RNA label and plate')
        columns[key]=col
    if set(columns)!={(p,l) for p in PLATES for l in LABELS}:raise ValueError('missing RNA label and plate')
    order=[columns[(p,l)] for p in PLATES for l in LABELS];ids=[];symbols=[];biotypes=[];rows=[]
    for row in it:
        gid,sym,bio=row[:3]
        if not isinstance(gid,str) or not gid or gid.strip()!=gid:raise ValueError('invalid gene id')
        ids.append(gid);symbols.append(sym);biotypes.append(bio)
        vals=[]
        for j in order:
            v=row[j] if j<len(row) else None
            if v is None or isinstance(v,bool):raise ValueError('invalid count')
            try:v=float(v)
            except (ValueError,TypeError):raise ValueError('invalid count') from None
            if not np.isfinite(v) or v<0:raise ValueError('invalid count')
            vals.append(v)
        rows.append(vals)
    book.close()
    if not ids or len(set(ids))!=len(ids):raise ValueError('duplicate or missing gene ids')
    mat=np.asarray(rows,float).T;libs=mat.sum(1)
    if np.any(libs<=0) or not np.isfinite(libs).all():raise ValueError('invalid library')
    log=np.log2(1+1e6*mat/libs[:,None]).reshape(5,47,-1)
    names,weights=signature.load_known_weights(signature_path);mapping=adapter.resolve(ids,symbols,biotypes,names);lookup={v:i for i,v in enumerate(ids)}
    indices=[None if r['strict'] is None else lookup[r['strict']] for r in mapping]
    result=qc.feature_qc(log,weights,indices)
    return {'status':'completed_synthetic_qc','synthetic_only':True,'gene_rows':len(ids),'sample_columns':len(order),
            'signature_format':dict(signature.signature_audit),
            'mapping_status':{k:sum(r['status']==k for r in mapping) for k in ('unique','missing','ambiguous')},
            'feature_qc':result,'score_or_function_evaluated':False}

def main():
    p=argparse.ArgumentParser();p.add_argument('--counts',type=Path,required=True);p.add_argument('--signature',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    dest=safe(a.output)
    if dest.exists():raise FileExistsError('no overwrite')
    result=inspect(a.counts,a.signature)
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps({'status':result['status'],'synthetic_only':True,'score_or_function_evaluated':False}))
if __name__=='__main__':main()
