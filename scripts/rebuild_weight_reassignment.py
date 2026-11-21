"""Rebuild the post-hoc 2,000-weight reference from public inputs.

Draft: run in a disposable directory; requires the packaged run_3d_cardiopedia_benchmark
module and four seed tables. CSV is a conditional descriptive reference, not a p value.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr

def sha256(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''): h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--release-dir', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    a = parser.parse_args()
    output = a.output.resolve()
    if output.exists(): raise FileExistsError(output)
    os.environ['CARDIAC_DATA_DIR'] = str(a.data_dir.resolve())
    module_path = a.release_dir.resolve() / 'scripts/run_3d_cardiopedia_benchmark.py'
    spec = importlib.util.spec_from_file_location('run_3d_cardiopedia_benchmark', module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    cp = mod.load_cardiopedia_strict()
    # Matrix and gene order are those of the original workbook, with plates first.
    mat = np.asarray(cp['counts_235'], float).T
    syms = cp['gene_syms']
    with (a.data_dir/'datasets/ReHeaT2_zenodo_15261569/mol_consistency.csv').open(encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    weights={}
    for r in rows:
        if r['mol_score'] in ('NA',''): continue
        if r['gene'] in weights: raise ValueError('duplicate signature symbol')
        weights[r['gene']]=float(r['mol_score'])
    if not np.isfinite(mat).all() or (mat < 0).any(): raise ValueError('invalid counts')
    libs = mat.sum(axis=0)
    if (libs <= 0).any(): raise ValueError('zero library')
    logged=np.log2(1+mat*1e6/libs[None,:])
    means=logged.mean(axis=1); sd=logged.std(axis=1,ddof=1)
    freq=Counter(s for s in syms if s)
    use=np.array([s in weights and freq[s]==1 and weights[s]!=0 and sd[i]>1e-12 for i,s in enumerate(syms)])
    w=np.array([weights[s] for i,s in enumerate(syms) if use[i]])
    if len(w)!=13256: raise ValueError(f'Unexpected usable genes: {len(w)}')
    z=(logged[use]-means[use,None])/sd[use,None]
    n=len(w)
    a_bin=np.minimum(4,((rankdata(means[use],method='average')-1)*5/n).astype(int))
    b_bin=np.minimum(4,((rankdata(sd[use],method='average')-1)*5/n).astype(int))
    bins=[np.flatnonzero(a_bin*5+b_bin==k) for k in range(25)]
    bins=[g for g in bins if len(g)]
    zm=z.reshape(n,5,47).mean(axis=1)
    y=cp['endpoints_24h']['Force'].mean(axis=0)
    yr=rankdata(y);yr=yr-yr.mean(); yn=np.linalg.norm(yr)
    actual=float(spearmanr(zm.T@w/np.abs(w).sum(),y).statistic)
    rng=np.random.default_rng(2026111801); all_r=[]
    for start in range(0,2000,100):
        # Preserve original draw order: one replicate at a time, then 25 strata.
        wp=np.repeat(w[:,None],min(100,2000-start),axis=1)
        for col in range(wp.shape[1]):
            for ix in bins: wp[ix,col]=rng.permutation(w[ix])
        scores=zm.T@wp/np.abs(w).sum()
        ranks=rankdata(scores,axis=0);ranks=ranks-ranks.mean(axis=0,keepdims=True)
        norms=np.linalg.norm(ranks,axis=0)
        if (norms==0).any():raise ValueError('constant shuffled score')
        all_r.extend(((yr@ranks)/(yn*norms)).tolist())
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8',newline='') as f:
        writer=csv.writer(f);writer.writerow(['replicate','spearman_rho']);writer.writerows(enumerate(all_r,1))
    result={'output':str(output),'sha256':sha256(output),'n':len(all_r),'n_genes':len(w),
            'seed':2026111801,'observed_rho':actual,'quantiles':np.quantile(all_r,[.025,.5,.975]).tolist(),
            'extreme_n':int(np.sum(np.abs(all_r)>=abs(actual)))}
    if a.compare:
        with a.compare.open(encoding='utf-8',newline='') as f: old=np.array([float(r['spearman_rho']) for r in csv.DictReader(f)])
        if len(old)!=len(all_r):raise ValueError('reference length differs')
        result['reference_sha256']=sha256(a.compare)
        result['max_abs_diff']=float(np.max(np.abs(old-np.asarray(all_r))))
        result['identical_values']=bool(np.array_equal(old,np.asarray(all_r)))
    print(json.dumps(result,indent=2))

if __name__=='__main__': main()
