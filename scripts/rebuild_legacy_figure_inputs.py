"""Portable reconstruction draft for three legacy figure-input tables.

Original algorithms: v6_1_reviewer_fixes.py and v6_3_ultimate_upgrades.py.
Requires released helpers and local public data. Refuses to overwrite target files.
"""
from pathlib import Path
import argparse, json, os, re, sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

def main():
    p=argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    p.add_argument('--release-dir',default=repo,type=Path)
    p.add_argument('--data-dir',default=Path(os.environ.get('CARDIAC_DATA_DIR', repo / 'data')),type=Path)
    p.add_argument('--output-dir',required=True,type=Path)
    p.add_argument('--target',choices=['prevalence','v63'],required=True)
    args=p.parse_args()
    os.environ['CARDIAC_DATA_DIR']=str(args.data_dir.resolve())
    scripts=args.release_dir.resolve()/'scripts'
    sys.path[:0]=[str(scripts),str(args.release_dir.resolve())]
    import run_3d_cardiopedia_benchmark as base
    from run_cross_system_benchmark import load_gtf_1to1
    from benchmark_utils import load_gse57338, score_cohort_robust_z, youden_threshold, eval_threshold
    import ssrank_audit as sa
    global DS,OUT_DIR,load_cardiopedia_strict,log2_cpm,resolve_strict_symbols,spearman_rho,score_cohort_z,score_single_sample_rank,score_ss_rank_internal,score_singscore_internal
    DS=base.DS; OUT_DIR=args.output_dir.resolve()
    OUT_DIR.mkdir(parents=True,exist_ok=True)
    intended=(["v6_1_mc100_prevalence_shift_gse165303.csv"] if args.target=='prevalence' else ["v6_3_table_missing_gene_dropout_array.csv","v6_3_table_compact_panel_3d_vs_clinical.csv"])
    if any((OUT_DIR/n).exists() for n in intended): raise FileExistsError('Refusing to overwrite existing output')
    load_cardiopedia_strict=base.load_cardiopedia_strict;log2_cpm=base.log2_cpm
    resolve_strict_symbols=base.resolve_strict_symbols;spearman_rho=base.spearman_rho
    score_cohort_z=base.score_cohort_z;score_single_sample_rank=base.score_single_sample_rank
    score_ss_rank_internal=lambda X,idx,w:(sa.score_ss_rank_internal(X,idx,w),None)
    score_singscore_internal=sa.score_singscore_internal
    mc=pd.read_csv(DS/'ReHeaT2_zenodo_15261569/mol_consistency.csv')
    mol_valid=mc.loc[mc['mol_score'].notna() & (mc['mol_score']!=0)].copy()
    sig_syms=list(mol_valid['gene'].astype(str));sig_w=mol_valid['mol_score'].to_numpy(float)
    if args.target=='prevalence':
        rng=np.random.default_rng(20261120)
        # Original 3D Monte Carlo consumed 200 x (choice 15/10, 16/12, 16/12).
        for _ in range(200):
            rng.choice(15,size=10,replace=False);rng.choice(16,size=12,replace=False);rng.choice(16,size=12,replace=False)
        prevalence(rng,sig_syms,sig_w,score_cohort_robust_z)
    else:
        _,id2sym_1to1=load_gtf_1to1()
        v63(id2sym_1to1,mol_valid,sig_syms,sig_w,load_gse57338,youden_threshold,eval_threshold)
    print(json.dumps({'target':args.target,'written':intended}))

def prevalence(rng,sig_syms,sig_w,score_cohort_robust_z):
    df165 = pd.read_csv(DS / "GSE165303/GSE165303_RNA-Seq_master_table.txt.gz", sep="\t")
    nf_cols = sorted([c for c in df165.columns if re.fullmatch(r"NF-\d+", c)], key=lambda x: int(x.split("-")[1]))
    dcm_cols = sorted([c for c in df165.columns if re.fullmatch(r"DCM-\d+", c)], key=lambda x: int(x.split("-")[1]))
    mat_165 = df165[nf_cols + dcm_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
    fin_m = np.all(np.isfinite(mat_165), axis=0)
    mat_165 = mat_165[:, fin_m]
    syms_165 = [str(s).strip() for s, ok in zip(df165["Symbol"].fillna(""), fin_m) if ok]
    idx_165 = resolve_strict_symbols(syms_165, sig_syms)
    y_165 = np.array([0] * 51 + [1] * 50, int)

    s165_cz = score_cohort_z(mat_165, idx_165, sig_w)[0]
    s165_rob = score_cohort_robust_z(mat_165, idx_165, sig_w)[0]
    s165_ssr = score_single_sample_rank(mat_165, idx_165, sig_w)[0]
    sd165_cz = float(np.std(s165_cz, ddof=1))
    sd165_rob = float(np.std(s165_rob, ddof=1))
    thr165_cz = float(np.median(s165_cz))
    thr165_rob = float(np.median(s165_rob))
    thr165_ssr = float(np.median(s165_ssr))

    mc_prev_rows = []
    for n_dcm in (5, 10, 15, 20, 25, 30, 35, 40, 45):
        n_nf = 50 - n_dcm
        cz_drifts, rob_drifts, cz_sens, cz_spec, rob_sens, rob_spec, ssr_sens, ssr_spec = [], [], [], [], [], [], [], []
        for _ in range(100):  # 100 random probe+batch draws per prevalence point = 900 simulated cohorts!
            p_nf = rng.choice(np.arange(0, 51), size=5, replace=False)
            p_dcm = rng.choice(np.arange(51, 101), size=5, replace=False)
            rem_nf = np.setdiff1d(np.arange(0, 51), p_nf)
            rem_dcm = np.setdiff1d(np.arange(51, 101), p_dcm)
            b_nf = rng.choice(rem_nf, size=n_nf - 5, replace=False) if n_nf > 5 else np.array([], dtype=int)
            b_dcm = rng.choice(rem_dcm, size=n_dcm - 5, replace=False) if n_dcm > 5 else np.array([], dtype=int)
            batch = np.concatenate([p_nf, b_nf, p_dcm, b_dcm])
            pos_nf = np.arange(0, 5)
            pos_dcm = np.arange(n_nf, n_nf + 5)
            pos = np.concatenate([pos_nf, pos_dcm])
            base_idx = np.concatenate([p_nf, p_dcm])

            sc_cz = score_cohort_z(mat_165[batch], idx_165, sig_w)[0][pos]
            sc_rob = score_cohort_robust_z(mat_165[batch], idx_165, sig_w)[0][pos]
            sc_ssr = s165_ssr[base_idx]

            cz_drifts.append(float(np.mean(sc_cz - s165_cz[base_idx]) / sd165_cz))
            rob_drifts.append(float(np.mean(sc_rob - s165_rob[base_idx]) / sd165_rob))
            cz_spec.append(float(np.mean(sc_cz[:5] <= thr165_cz)))
            cz_sens.append(float(np.mean(sc_cz[5:] > thr165_cz)))
            rob_spec.append(float(np.mean(sc_rob[:5] <= thr165_rob)))
            rob_sens.append(float(np.mean(sc_rob[5:] > thr165_rob)))
            ssr_spec.append(float(np.mean(sc_ssr[:5] <= thr165_ssr)))
            ssr_sens.append(float(np.mean(sc_ssr[5:] > thr165_ssr)))

        pct = int(round(100 * n_dcm / 50))
        for mname, dr, sp, sn in [
            ("Cohort-Z", cz_drifts, cz_spec, cz_sens),
            ("Cohort-Robust-Z", rob_drifts, rob_spec, rob_sens),
            ("SS-Rank", [0.0] * 100, ssr_spec, ssr_sens),
        ]:
            arr = np.array(dr)
            mc_prev_rows.append({
                "dcm_prevalence_pct": pct, "method": mname,
                "drift_mean_sd": float(arr.mean()),
                "drift_ci_low": float(np.quantile(arr, 0.025)),
                "drift_ci_high": float(np.quantile(arr, 0.975)),
                "spec_mean_pct": float(100.0 * np.mean(sp)),
                "sens_mean_pct": float(100.0 * np.mean(sn)),
            })
    df_mc_prev = pd.DataFrame(mc_prev_rows)
    df_mc_prev.to_csv(OUT_DIR / "v6_1_mc100_prevalence_shift_gse165303.csv", index=False)

def v63(id2sym_1to1,mol_valid,sig_syms,sig_w,load_gse57338,youden_threshold,eval_threshold):
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph141_sub = ph141[ph141["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    raw141 = cnt141[list(ph141_sub["sample_name"].astype(str))].to_numpy(float).T
    lmat_141 = log2_cpm(raw141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    y_141 = (ph141_sub["etiology"] == "DCM").astype(int).to_numpy()

    df165 = pd.read_csv(DS / "GSE165303/GSE165303_RNA-Seq_master_table.txt.gz", sep="\t")
    nf165 = sorted([c for c in df165.columns if re.fullmatch(r"NF-\d+", c)], key=lambda x: int(x.split("-")[1]))
    dcm165 = sorted([c for c in df165.columns if re.fullmatch(r"DCM-\d+", c)], key=lambda x: int(x.split("-")[1]))
    mat_165 = df165[nf165 + dcm165].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
    fin165 = np.all(np.isfinite(mat_165), axis=0)
    mat_165 = mat_165[:, fin165]
    syms_165 = [str(s).strip() for s, ok in zip(df165["Symbol"].fillna(""), fin165) if ok]
    y_165 = np.array([0] * 51 + [1] * 50, int)

    df46 = pd.read_csv(DS / "GSE46224/GSE46224_Yang_et_al_human_heart_RNASeq.txt.gz", sep="\t")
    syms_46 = [str(s).strip() for s in df46["Gene"].fillna("")]
    cols_46_base = [f"NF{i}" for i in range(1, 9)] + [f"ICM{i}" for i in range(1, 9)] + [f"NICM{i}" for i in range(1, 9)]
    mat_46_base = np.log2(1.0 + np.maximum(df46[cols_46_base].to_numpy(float).T, 0.0))
    y_46_base = np.array([0] * 8 + [1] * 16, int)
    cols_46_all = [f"NF{i}" for i in range(1, 9)] + [f"ICM{i}" for i in range(1, 9)] + [f"ICM+LVAD{i}" for i in range(1, 9)] + [f"NICM{i}" for i in range(1, 9)] + [f"NICM+LVAD{i}" for i in range(1, 9)]
    mat_46_all = np.log2(1.0 + np.maximum(df46[cols_46_all].to_numpy(float).T, 0.0))

    g57 = load_gse57338()
    idx_141=resolve_strict_symbols(syms_141,sig_syms)
    thr_cache={("GSE141910_CPM","SS-Rank-Internal"):youden_threshold(y_141,score_ss_rank_internal(lmat_141,idx_141,sig_w)[0]),("GSE141910_CPM","singscore-Internal"):youden_threshold(y_141,score_singscore_internal(lmat_141,idx_141,sig_w))}
    rng = np.random.default_rng(20261120)
    idx_57 = resolve_strict_symbols(g57["syms"], sig_syms)
    mapped_pos_57 = [j for j, idx in enumerate(idx_57) if idx is not None]
    drop_rows = []
    for drop_pct in (0, 10, 20, 30, 50):
        n_rep = 1 if drop_pct == 0 else 20
        aucs_i, bals_i, mccs_i, bals_si = [], [], [], []
        for _ in range(n_rep):
            if drop_pct == 0:
                idx_d = idx_57
            else:
                drop_set = set(rng.choice(mapped_pos_57, size=int(len(mapped_pos_57) * drop_pct / 100), replace=False))
                idx_d = [None if j in drop_set else idx for j, idx in enumerate(idx_57)]
            s_i = score_ss_rank_internal(g57["mat"], idx_d, sig_w)[0]
            s_si = score_singscore_internal(g57["mat"], idx_d, sig_w)
            aucs_i.append(float(roc_auc_score(g57["y"], s_i)))
            ev_i = eval_threshold(g57["y"], s_i, thr_cache[("GSE141910_CPM", "SS-Rank-Internal")])
            ev_si = eval_threshold(g57["y"], s_si, thr_cache[("GSE141910_CPM", "singscore-Internal")])
            bals_i.append(ev_i["bal_acc"])
            mccs_i.append(ev_i["mcc"])
            bals_si.append(ev_si["bal_acc"])
        drop_rows.append({
            "dropout_pct": drop_pct, "n_reps": n_rep,
            "ssri_auc_mean": float(np.mean(aucs_i)),
            "ssri_frozen_bal_acc_mean": float(np.mean(bals_i)),
            "ssri_frozen_bal_acc_sd": float(np.std(bals_i)),
            "ssri_frozen_mcc_mean": float(np.mean(mccs_i)),
            "singscore_int_frozen_bal_acc_mean": float(np.mean(bals_si)),
        })
    df_drop = pd.DataFrame(drop_rows)
    df_drop.to_csv(OUT_DIR / "v6_3_table_missing_gene_dropout_array.csv", index=False)
    cp = load_cardiopedia_strict()
    log_cpm_235 = log2_cpm(cp["counts_235"])
    gene_syms_cp = cp["gene_syms"]
    fm = cp["endpoints_24h"]["Force"].mean(axis=0)
    sig_idx_cp = resolve_strict_symbols(gene_syms_cp, sig_syms)

    # Sort ALL ReHeaT2 finite nonzero genes strictly by external |mol_score|
    pos_all = mol_valid[mol_valid["mol_score"] > 0].sort_values("mol_score", ascending=False)
    neg_all = mol_valid[mol_valid["mol_score"] < 0].sort_values("mol_score", ascending=True)

    K_grid = [20, 50, 100, 200, 500, 1000, 2000, 5000, 13641]
    k_multi_rows = []
    for K in K_grid:
        if K == 13641:
            sub_k = mol_valid
        else:
            half = K // 2
            sub_k = pd.concat([pos_all.head(half), neg_all.head(half)], ignore_index=True)
        ksym = list(sub_k["gene"].astype(str))
        kw = sub_k["mol_score"].to_numpy(float)

        # 3D Cardiopedia (SS-Rank-Internal)
        s3d = score_ss_rank_internal(log_cpm_235, resolve_strict_symbols(gene_syms_cp, ksym), kw)[0].reshape(5, 47).mean(axis=0)
        rho_3d = spearman_rho(s3d, fm)

        # GSE141910 (source for frozen threshold at panel size K)
        s141_k = score_ss_rank_internal(lmat_141, resolve_strict_symbols(syms_141, ksym), kw)[0]
        thr_k = youden_threshold(y_141, s141_k)

        # GSE57338 Affymetrix Microarray (AUROC + frozen BalAcc)
        s57_k = score_ss_rank_internal(g57["mat"], resolve_strict_symbols(g57["syms"], ksym), kw)[0]
        auc_57 = float(roc_auc_score(g57["y"], s57_k))
        bal_57 = eval_threshold(g57["y"], s57_k, thr_k)["bal_acc"]

        # GSE165303 Processed (AUROC + frozen BalAcc)
        s165_k = score_ss_rank_internal(mat_165, resolve_strict_symbols(syms_165, ksym), kw)[0]
        auc_165 = float(roc_auc_score(y_165, s165_k))
        bal_165 = eval_threshold(y_165, s165_k, thr_k)["bal_acc"]

        # GSE46224 Paired LVAD (Pre vs NF AUROC + paired dz + Wilcoxon p)
        s46_k = score_ss_rank_internal(mat_46_all, resolve_strict_symbols(syms_46, ksym), kw)[0]
        d46_k = np.concatenate([s46_k[16:24], s46_k[32:40]]) - np.concatenate([s46_k[8:16], s46_k[24:32]])
        dz_46 = float(d46_k.mean() / d46_k.std(ddof=1))
        p_46 = float(stats.wilcoxon(d46_k).pvalue)

        k_multi_rows.append({
            "panel_K": K, "rho_3d_hco": rho_3d,
            "gse57338_array_auc": auc_57, "gse57338_array_frozen_bal_acc": bal_57,
            "gse165303_auc": auc_165, "gse165303_frozen_bal_acc": bal_165,
            "gse46224_lvad_dz": dz_46, "gse46224_lvad_p": p_46,
        })
    df_kmulti = pd.DataFrame(k_multi_rows)
    df_kmulti.to_csv(OUT_DIR / "v6_3_table_compact_panel_3d_vs_clinical.csv", index=False)

if __name__=="__main__": main()
