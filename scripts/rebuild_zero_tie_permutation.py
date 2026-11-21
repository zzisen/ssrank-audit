"""Reproduce the retrospective 10-permutation zero-boundary sensitivity audit.

The permutations reapply the historical stable-sort perturbation in different
column orders and then restore gene identity before tie-invariant scoring. The
reported ranges are exploratory, conditional on these seeds and data order, not
confidence intervals. Never overwrites the specified output.
"""
import argparse
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path(os.environ.get('CARDIAC_DATA_DIR',repo/'data')))
    p.add_argument('--reference',type=Path,default=repo/'results/v6_4_table_zero_inflation_ties_stress.csv')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError('Refusing to overwrite existing output')
    os.environ['CARDIAC_DATA_DIR']=str(a.data_dir.resolve())
    sys.path[:0]=[str(repo/'scripts'),str(repo)]
    import run_3d_cardiopedia_benchmark as base
    from benchmark_utils import apply_zero_inflation
    import ssrank_audit as sa
    cp=base.load_cardiopedia_strict()
    X=base.log2_cpm(cp['counts_235']); syms=cp['gene_syms']
    force=cp['endpoints_24h']['Force'].mean(axis=0)
    mc=pd.read_csv(a.data_dir/'datasets/ReHeaT2_zenodo_15261569/mol_consistency.csv')
    mv=mc.loc[mc['mol_score'].notna() & (mc['mol_score']!=0)]
    rawidx=base.resolve_strict_symbols(syms,list(mv['gene'].astype(str)))
    valid=np.array([i is not None for i in rawidx]);idx=np.array([i for i in rawidx if i is not None],int)
    w=mv['mol_score'].to_numpy(float)[valid];n=X.shape[1]
    seeds=list(range(2026112101,2026112111));perms=[np.random.default_rng(seed).permutation(n) for seed in seeds]
    reference=pd.read_csv(a.reference).set_index('zero_inflation_pct')
    methods=[('SS-Rank-Internal','hco_3d_rho_ssri',sa.score_ss_rank_internal),('singscore-Internal','hco_3d_rho_sgi',sa.score_singscore_internal)]
    rows=[]
    for pct in (70,20,40,60):
        k=int(round(n*pct/100));cutoff=np.partition(X,k-1,axis=1)[:,k-1]
        lower=(X<cutoff[:,None]).sum(axis=1);ties=(X==cutoff[:,None]).sum(axis=1)
        boundary=int(((lower<k)&(lower+ties>k)&(cutoff!=0)).sum())
        orig=apply_zero_inflation(X,pct)
        base_scores=[fn(orig,idx,w) for _,_,fn in methods]
        permuted=[]
        for perm in perms:
            xp=apply_zero_inflation(X[:,perm],pct)
            z=np.empty_like(X);z[:,perm]=xp
            permuted.append([fn(z,idx,w) for _,_,fn in methods])
        for method_i,(name,col,fn) in enumerate(methods):
            baseline=float(spearmanr(base_scores[method_i].reshape(5,47).mean(axis=0),force).statistic)
            for j,seed in enumerate(seeds):
                score=permuted[j][method_i]
                rho=float(spearmanr(score.reshape(5,47).mean(axis=0),force).statistic)
                rows.append({'zero_pct':pct,'method':name,'seed':seed,'boundary_tied_samples':boundary,'original_rho':baseline,'published_csv_rho':float(reference.loc[pct,col]),'permuted_rho':rho,'permuted_minus_original':rho-baseline,'max_sample_score_change':float(np.max(np.abs(score-base_scores[method_i]))),'changed_sample_scores':int(np.sum(np.abs(score-base_scores[method_i])>1e-12))})
        print('completed zero percentage',pct,flush=True)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create prevents accidental replacement of prior analysis.
    with a.output.open('x',encoding='utf-8',newline='') as f:pd.DataFrame(rows).to_csv(f,index=False)
    print('wrote',a.output,flush=True)

if __name__ == '__main__':main()
