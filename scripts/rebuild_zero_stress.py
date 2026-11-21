"""Rebuild the historical fixed-column-order zero-inflation stress table.

The perturbation sorts each sample using stable column order; if nonzero ties
straddle the cutoff, which equal-valued genes are zeroed can depend on column
order. This is distinct from the scoring methods' tie handling. The script
reproduces the original matrix order and never overwrites an existing table.
"""
from pathlib import Path
import argparse, os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path(os.environ.get('CARDIAC_DATA_DIR',repo/'data')))
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    os.environ['CARDIAC_DATA_DIR']=str(a.data_dir.resolve())
    sys.path[:0]=[str(repo/'scripts'),str(repo)]
    import run_3d_cardiopedia_benchmark as base
    from run_cross_system_benchmark import load_gtf_1to1
    from benchmark_utils import load_gse57338,youden_threshold,eval_threshold,apply_zero_inflation
    import ssrank_audit as sa
    global DS,OUT_DIR,load_cardiopedia_strict,log2_cpm,resolve_strict_symbols,spearman_rho,score_single_sample_rank,score_ss_rank_internal,score_singscore_internal
    DS=base.DS;OUT_DIR=a.output_dir.resolve();OUT_DIR.mkdir(parents=True,exist_ok=True)
    dest=OUT_DIR/'v6_4_table_zero_inflation_ties_stress.csv'
    if dest.exists():raise FileExistsError('Refusing to overwrite existing output')
    load_cardiopedia_strict=base.load_cardiopedia_strict;log2_cpm=base.log2_cpm
    resolve_strict_symbols=base.resolve_strict_symbols;spearman_rho=base.spearman_rho
    score_single_sample_rank=base.score_single_sample_rank
    score_ss_rank_internal=lambda X,idx,w:(sa.score_ss_rank_internal(X,idx,w),None)
    score_singscore_internal=sa.score_singscore_internal
    mc=pd.read_csv(DS/'ReHeaT2_zenodo_15261569/mol_consistency.csv')
    mv=mc.loc[mc.mol_score.notna() & (mc.mol_score!=0)]
    syms=list(mv.gene.astype(str));w=mv.mol_score.to_numpy(float)
    _,ids=load_gtf_1to1()
    zero_test(ids,syms,w,load_gse57338,youden_threshold,eval_threshold,apply_zero_inflation)
    print('written',dest,flush=True)

def zero_test(id2sym_1to1,sig_syms,sig_w,load_gse57338,youden_threshold,eval_threshold,apply_zero_inflation):
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph141_sub = ph141[ph141["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    raw141 = cnt141[list(ph141_sub["sample_name"].astype(str))].to_numpy(float).T
    lmat_141 = log2_cpm(raw141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    y_141 = (ph141_sub["etiology"] == "DCM").astype(int).to_numpy()

    g57 = load_gse57338()
    cp=load_cardiopedia_strict();log_cpm_235=log2_cpm(cp['counts_235'])
    fm=cp['endpoints_24h']['Force'].mean(axis=0);sig_idx=resolve_strict_symbols(cp['gene_syms'],sig_syms)
    idx_141 = resolve_strict_symbols(syms_141, sig_syms)
    idx_57 = resolve_strict_symbols(g57["syms"], sig_syms)
    thr_zi = {
        "SS-Rank-Internal": youden_threshold(y_141, score_ss_rank_internal(lmat_141, idx_141, sig_w)[0]),
        "singscore-Internal": youden_threshold(y_141, score_singscore_internal(lmat_141, idx_141, sig_w)),
        "SS-Rank": youden_threshold(y_141, score_single_sample_rank(lmat_141, idx_141, sig_w)[0]),
    }
    zi_rows = []
    for pct in [0, 20, 40, 60, 70]:
        X3_z = apply_zero_inflation(log_cpm_235, pct)
        X57_z = apply_zero_inflation(g57["mat"], pct)
        r_ssri_3d = spearman_rho(score_ss_rank_internal(X3_z, sig_idx, sig_w)[0].reshape(5, 47).mean(axis=0), fm)
        r_sgi_3d = spearman_rho(score_singscore_internal(X3_z, sig_idx, sig_w).reshape(5, 47).mean(axis=0), fm)
        r_ssr_3d = spearman_rho(score_single_sample_rank(X3_z, sig_idx, sig_w)[0].reshape(5, 47).mean(axis=0), fm)

        sc_ssri_57 = score_ss_rank_internal(X57_z, idx_57, sig_w)[0]
        sc_sgi_57 = score_singscore_internal(X57_z, idx_57, sig_w)
        sc_ssr_57 = score_single_sample_rank(X57_z, idx_57, sig_w)[0]
        ev_ssri = eval_threshold(g57["y"], sc_ssri_57, thr_zi["SS-Rank-Internal"])
        ev_sgi = eval_threshold(g57["y"], sc_sgi_57, thr_zi["singscore-Internal"])
        ev_ssr = eval_threshold(g57["y"], sc_ssr_57, thr_zi["SS-Rank"])
        zi_rows.append({
            "zero_inflation_pct": pct,
            "hco_3d_rho_ssri": r_ssri_3d, "hco_3d_rho_sgi": r_sgi_3d, "hco_3d_rho_ssr": r_ssr_3d,
            "array_frozen_bal_ssri": ev_ssri["bal_acc"], "array_frozen_mcc_ssri": ev_ssri["mcc"], "array_auc_ssri": float(roc_auc_score(g57["y"], sc_ssri_57)),
            "array_frozen_bal_sgi": ev_sgi["bal_acc"], "array_frozen_mcc_sgi": ev_sgi["mcc"],
            "array_frozen_bal_ssr": ev_ssr["bal_acc"], "array_frozen_mcc_ssr": ev_ssr["mcc"],
        })
        print(f"  Zero-Inflation {pct:2d}%: 3D rho SSRI={r_ssri_3d:+.3f} (SGI={r_sgi_3d:+.3f}) | Array Frozen BalAcc SSRI={ev_ssri['bal_acc']:.3f} (MCC={ev_ssri['mcc']:.3f}, AUC={roc_auc_score(g57['y'], sc_ssri_57):.3f}) vs SGI={ev_sgi['bal_acc']:.3f} vs SSR={ev_ssr['bal_acc']:.3f}")
    pd.DataFrame(zi_rows).to_csv(OUT_DIR / "v6_4_table_zero_inflation_ties_stress.csv", index=False)

if __name__ == "__main__":
    main()
