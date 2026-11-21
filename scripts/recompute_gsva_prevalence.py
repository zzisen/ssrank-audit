"""Re-run identical prevalence batches using historic versus tie-averaged GSVA.

The historic implementation is copied from v6_4_complete_ceiling_audit.py;
only tie fractions and return metadata were added. Writes new files only.
"""
from pathlib import Path
import argparse,csv,importlib.util,json,math,os,re,sys,time
import numpy as np
import pandas as pd
from scipy import special

def legacy_gsva(expr_2d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """
    Faithful transductive bidirectional GSVA (Hanzelmann et al., BMC Bioinformatics 2013):
      1. Across-sample Gaussian kernel ECDF per gene (bandwidth h_g = sd_g / 4, kcdf='Gaussian'),
         transformed to log-odds z_{i,g} = log(F / (1 - F)). [TRANSDUCTIVE across cohort samples!]
      2. Within each sample i, rank genes descending by z_{i,g}, center ranks symmetrically around zero,
         and compute KS random-walk max-deviation (es.diff = max_pos + min_neg) for S+ minus S-.
    """
    X = np.asarray(expr_2d, dtype=np.float32)
    N, G = X.shape
    w = np.asarray(weights, float)
    valid = [(idx, wj) for idx, wj in zip(gene_indices, w) if idx is not None and math.isfinite(wj) and wj != 0]
    up_cols = np.array([idx for idx, wj in valid if wj > 0], dtype=int)
    dn_cols = np.array([idx for idx, wj in valid if wj < 0], dtype=int)
    if len(up_cols) == 0 or len(dn_cols) == 0:
        raise ValueError("Bidirectional GSVA requires both positive and negative weights.")

    Z = np.empty((N, G), dtype=np.float32)
    chunk = 1024
    eps = np.float32(1e-6)
    for start in range(0, G, chunk):
        end = min(start + chunk, G)
        Xc = X[:, start:end]
        sd = Xc.std(axis=0, ddof=1)
        h = np.where(sd > 1e-8, sd * np.float32(0.25), np.float32(1.0))
        diff = (Xc[:, None, :] - Xc[None, :, :]) / h[None, None, :]
        F = special.ndtr(diff).mean(axis=1, dtype=np.float32)
        const_mask = sd <= 1e-8
        if np.any(const_mask):
            F[:, const_mask] = np.float32(0.5)
        F = np.clip(F, eps, np.float32(1.0) - eps)
        Z[:, start:end] = np.log(F / (np.float32(1.0) - F))

    tie_fracs=[]
    for row in Z:
        zz=np.sort(row)
        diff=np.diff(zz)
        tied=diff==0
        tie_fracs.append(float((np.count_nonzero(tied[:-1] | tied[1:])+int(tied[0])+int(tied[-1]))/G) if G>2 else 0.0)
    up_mask_full = np.zeros(G, dtype=bool)
    dn_mask_full = np.zeros(G, dtype=bool)
    up_mask_full[up_cols] = True
    dn_mask_full[dn_cols] = True
    n_miss_up = G - int(up_mask_full.sum())
    n_miss_dn = G - int(dn_mask_full.sum())
    sym_r = np.abs(G / 2.0 - np.arange(1, G + 1, dtype=float))
    if tau != 1.0:
        sym_r = sym_r ** tau

    out = np.empty(N, dtype=float)
    tied_out = np.empty(N, dtype=float)
    for i in range(N):
        order = np.argsort(-Z[i], kind="mergesort")
        in_up = up_mask_full[order]
        in_dn = dn_mask_full[order]

        w_up = np.where(in_up, sym_r, 0.0)
        s_up = w_up.sum()
        step_up = (w_up / s_up if s_up > 0 else 0.0) - np.where(in_up, 0.0, 1.0 / max(n_miss_up, 1))
        walk_up = np.cumsum(step_up)
        es_up = float(max(0.0, walk_up.max()) + min(0.0, walk_up.min()))

        w_dn = np.where(in_dn, sym_r, 0.0)
        s_dn = w_dn.sum()
        step_dn = (w_dn / s_dn if s_dn > 0 else 0.0) - np.where(in_dn, 0.0, 1.0 / max(n_miss_dn, 1))
        walk_dn = np.cumsum(step_dn)
        es_dn = float(max(0.0, walk_dn.max()) + min(0.0, walk_dn.min()))

        out[i] = es_up - es_dn
        tu = np.cumsum(sa._tie_averaged_walk_steps(Z[i], up_mask_full, "gsva", tau))
        td = np.cumsum(sa._tie_averaged_walk_steps(Z[i], dn_mask_full, "gsva", tau))
        tied_out[i] = float(max(0.0,tu.max())+min(0.0,tu.min())-max(0.0,td.max())-min(0.0,td.min()))
    return out, tied_out, float(np.mean(tie_fracs)), float(np.max(tie_fracs))


def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser();p.add_argument('--release-dir',type=Path,default=repo);p.add_argument('--data-dir',type=Path,default=Path(os.environ.get('CARDIAC_DATA_DIR',repo/'data')))
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-batches',type=int,default=360);p.add_argument('--first-pct',type=int,default=10);p.add_argument('--last-pct',type=int,default=90)
    a=p.parse_args()
    if a.max_batches < 1 or not (10 <= a.first_pct <= a.last_pct <= 90) or a.first_pct % 10 or a.last_pct % 10:
        p.error('max-batches must be positive and prevalence bounds multiples of 10 within 10..90')
    os.environ['CARDIAC_DATA_DIR']=str(a.data_dir.resolve());sys.path[:0]=[str(a.release_dir.resolve()/'scripts'),str(a.release_dir.resolve())]
    import run_3d_cardiopedia_benchmark as base
    global sa
    import ssrank_audit as sa
    from benchmark_utils import youden_threshold,eval_threshold
    a.output_dir.mkdir(parents=True,exist_ok=True);out=a.output_dir/'matched_batch_comparison.csv';summary=a.output_dir/'summary.json'
    if out.exists() or summary.exists():raise FileExistsError('Refusing overwrite')
    ds=a.data_dir/'datasets';mc=pd.read_csv(ds/'ReHeaT2_zenodo_15261569/mol_consistency.csv')
    m=mc.loc[mc.mol_score.notna() & (mc.mol_score!=0)].copy();syms=list(m.gene.astype(str));w=m.mol_score.to_numpy(float)
    df=pd.read_csv(ds/'GSE165303/GSE165303_RNA-Seq_master_table.txt.gz',sep='\t')
    nf=sorted([c for c in df.columns if re.fullmatch(r'NF-\d+',c)],key=lambda x:int(x.split('-')[1]))
    dm=sorted([c for c in df.columns if re.fullmatch(r'DCM-\d+',c)],key=lambda x:int(x.split('-')[1]))
    X=df[nf+dm].apply(pd.to_numeric,errors='coerce').to_numpy(float).T
    finite=np.all(np.isfinite(X),axis=0);X=X[:,finite]
    xsyms=[str(s).strip() for s,ok in zip(df['Symbol'].fillna(''),finite) if ok]
    idx=base.resolve_strict_symbols(xsyms,syms); y=np.array([0]*51+[1]*50,int)
    old_full,tie_full,full_tf,full_tm=legacy_gsva(X,idx,w);new_full=sa.score_gsva_bidirectional(X,idx,w)
    sd_old=float(np.std(old_full,ddof=1));sd_tie=float(np.std(tie_full,ddof=1));sd_new=float(np.std(new_full,ddof=1))
    threshold_old=youden_threshold(y,old_full);threshold_tie=youden_threshold(y,tie_full);threshold_new=youden_threshold(y,new_full)
    nf_idx=np.where(y==0)[0];dm_idx=np.where(y==1)[0]
    probes=np.r_[nf_idx[:4],dm_idx[:4]];rn=nf_idx[4:];rd=dm_idx[4:];rng=np.random.default_rng(20261120)
    fields=['prev_dcm_pct','replicate','old_probe_mean','new_probe_mean','old_sens','new_sens','old_spec','new_spec','probe_absdiff_mean','probe_absdiff_max','tieold_probe_mean','tieold_sens','tieold_spec','tieonly_probe_absdiff_mean','tieonly_probe_absdiff_max','old_tie_fraction_mean','old_tie_fraction_max']
    n=0;t0=time.perf_counter()
    with out.open('x',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();f.flush()
        for pct in range(10,100,10):
            nd=int(round(40*pct/100.));addd=nd-4;addn=(40-nd)-4
            for rep in range(40):
                bgd=rng.choice(rd,size=addd,replace=False) if addd>0 else np.array([],int)
                bgn=rng.choice(rn,size=addn,replace=False) if addn>0 else np.array([],int)
                if pct<a.first_pct or pct>a.last_pct: continue
                sel=np.r_[probes,bgn,bgd];yb=np.r_[np.zeros(4,int),np.ones(4,int),np.zeros(addn,int),np.ones(addd,int)]
                old,tieold,tf,tm=legacy_gsva(X[sel],idx,w);new=sa.score_gsva_bidirectional(X[sel],idx,w)
                eo=eval_threshold(yb,old,threshold_old);et=eval_threshold(yb,tieold,threshold_tie);en=eval_threshold(yb,new,threshold_new)
                row={'prev_dcm_pct':pct,'replicate':rep+1,'old_probe_mean':old[:8].mean(),'new_probe_mean':new[:8].mean(),
                     'old_sens':eo['sens'],'new_sens':en['sens'],'old_spec':eo['spec'],'new_spec':en['spec'],'tieold_probe_mean':tieold[:8].mean(),'tieold_sens':et['sens'],'tieold_spec':et['spec'],'tieonly_probe_absdiff_mean':np.abs(old[:8]-tieold[:8]).mean(),'tieonly_probe_absdiff_max':np.abs(old[:8]-tieold[:8]).max(),
                     'probe_absdiff_mean':np.abs(old[:8]-new[:8]).mean(),'probe_absdiff_max':np.abs(old[:8]-new[:8]).max(),
                     'old_tie_fraction_mean':tf,'old_tie_fraction_max':tm}
                writer.writerow(row);f.flush();n+=1
                if n%10==0 or n==1:print('completed',n,'seconds',round(time.perf_counter()-t0,1),flush=True)
                if n>=a.max_batches:break
            if n>=a.max_batches:break
    r=pd.read_csv(out);records=[]
    if (r.prev_dcm_pct==50).any():
        baselines={k:r.loc[r.prev_dcm_pct==50,f'{k}_probe_mean'].mean() for k in ('old','tieold','new')}
        for pct,g in r.groupby('prev_dcm_pct',sort=True):
            records.append({'pct':int(pct),'n':len(g),**{f'{k}_drift_mean_sd':float((g[f'{k}_probe_mean'].mean()-baselines[k])/sd) for k,sd in [('old',sd_old),('tieold',sd_tie),('new',sd_new)]},'old_spec':float(g.old_spec.mean()),'tieold_spec':float(g.tieold_spec.mean()),'new_spec':float(g.new_spec.mean()),'old_sens':float(g.old_sens.mean()),'tieold_sens':float(g.tieold_sens.mean()),'new_sens':float(g.new_sens.mean())})
    result={'n_batches':n,'requested':a.max_batches,'first_pct':a.first_pct,'last_pct':a.last_pct,'seed':20261120,'historic_full_sd':sd_old,'tieavg_full_sd':sd_new,'historic_threshold':threshold_old,'tieavg_threshold':threshold_new,'old_full_tie_fraction_mean':full_tf,'mean_abs_score_difference':float(r.probe_absdiff_mean.mean()),'mean_abs_score_difference_tieonly':float(r.tieonly_probe_absdiff_mean.mean()),'max_abs_score_difference_tieonly':float(r.tieonly_probe_absdiff_max.max()),'max_abs_score_difference':float(r.probe_absdiff_max.max()),'batch_tie_fraction_mean':float(r.old_tie_fraction_mean.mean()),'levels':records}
    summary.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()

