"""Rebuild the historical six-cohort calibration table.

Its GSVA row uses the original float32/column-order implementation and is not
the later tie-averaged GSVA. Figure 5A uses only Cohort-Z and SS-Rank-Internal.
Writes to a new output file and does not overwrite archived results.
"""
from pathlib import Path
import argparse,math,os,re,sys
import numpy as np
import pandas as pd
from scipy import special
from sklearn.metrics import roc_auc_score

def score_gsva_bidirectional(expr_2d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray, tau: float = 1.0) -> np.ndarray:
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
    return out

def main():
    repo=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser()
    p.add_argument('--data-dir',type=Path,default=Path(os.environ.get('CARDIAC_DATA_DIR',repo/'data')))
    p.add_argument('--seed-dir',type=Path,default=repo/'results')
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    os.environ['CARDIAC_DATA_DIR']=str(a.data_dir.resolve())
    sys.path[:0]=[str(repo/'scripts'),str(repo)]
    import run_3d_cardiopedia_benchmark as base
    from run_cross_system_benchmark import load_gtf_1to1
    from benchmark_utils import load_gse57338,load_gse116250
    import ssrank_audit as sa
    global DS,OUT_DIR,SEED_DIR,load_cardiopedia_strict,log2_cpm,resolve_strict_symbols,spearman_rho,score_cohort_z,score_single_sample_rank,score_ss_rank_internal,score_singscore_internal
    DS=base.DS;OUT_DIR=a.output_dir.resolve();SEED_DIR=a.seed_dir.resolve();OUT_DIR.mkdir(parents=True,exist_ok=True)
    dest=OUT_DIR/'v6_4_table_clinical_6cohorts_calibration.csv'
    if dest.exists():raise FileExistsError('Refusing to overwrite existing output')
    load_cardiopedia_strict=base.load_cardiopedia_strict;log2_cpm=base.log2_cpm
    resolve_strict_symbols=base.resolve_strict_symbols;spearman_rho=base.spearman_rho
    score_cohort_z=base.score_cohort_z;score_single_sample_rank=base.score_single_sample_rank
    score_ss_rank_internal=lambda X,idx,w:(sa.score_ss_rank_internal(X,idx,w),None)
    score_singscore_internal=sa.score_singscore_internal
    mc=pd.read_csv(DS/'ReHeaT2_zenodo_15261569/mol_consistency.csv')
    mv=mc.loc[mc.mol_score.notna() & (mc.mol_score!=0)]
    syms=list(mv.gene.astype(str));w=mv.mol_score.to_numpy(float)
    _,ids=load_gtf_1to1()
    calibration(ids,syms,w,load_gse116250,load_gse57338)
    print('written',dest,flush=True)

def calibration(id2sym_1to1,sig_syms,sig_w,load_gse116250,load_gse57338):
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph141_sub = ph141[ph141["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    raw141 = cnt141[list(ph141_sub["sample_name"].astype(str))].to_numpy(float).T
    lmat_141 = log2_cpm(raw141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    y_141 = (ph141_sub["etiology"] == "DCM").astype(int).to_numpy()

    m349 = pd.read_csv(SEED_DIR / "v6_gse349247_sample_scores.csv")
    df349 = pd.read_csv(DS / "GSE349247/GSE349247_heart_rnaseq_gene_counts.csv.gz")
    lmat_349 = log2_cpm(df349[list(m349["sample"])].to_numpy(float).T)
    syms_349 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in df349["Geneid"]]
    y_349 = m349["is_hf"].to_numpy(int)

    df165 = pd.read_csv(DS / "GSE165303/GSE165303_RNA-Seq_master_table.txt.gz", sep="\t")
    nf165 = sorted([c for c in df165.columns if re.fullmatch(r"NF-\d+", c)], key=lambda x: int(x.split("-")[1]))
    dcm165 = sorted([c for c in df165.columns if re.fullmatch(r"DCM-\d+", c)], key=lambda x: int(x.split("-")[1]))
    mat_165 = df165[nf165 + dcm165].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
    fin165 = np.all(np.isfinite(mat_165), axis=0)
    mat_165 = mat_165[:, fin165]
    syms_165 = [str(s).strip() for s, ok in zip(df165["Symbol"].fillna(""), fin165) if ok]
    y_165 = np.array([0] * 51 + [1] * 50, int)

    g116 = load_gse116250()

    df46 = pd.read_csv(DS / "GSE46224/GSE46224_Yang_et_al_human_heart_RNASeq.txt.gz", sep="\t")
    syms_46 = [str(s).strip() for s in df46["Gene"].fillna("")]
    cols_46_base = [f"NF{i}" for i in range(1, 9)] + [f"ICM{i}" for i in range(1, 9)] + [f"NICM{i}" for i in range(1, 9)]
    mat_46 = np.log2(df46[cols_46_base].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).T + 1.0)
    y_46 = np.array([0] * 8 + [1] * 16, int)

    g57 = load_gse57338()
    cohorts = {
        "GSE141910_CPM": {"X": lmat_141, "syms": syms_141, "y": y_141, "tech": "RNA-seq CPM", "label": "GSE141910 MAGNet\n[Upstream Ref, n=332]"},
        "GSE349247_CPM": {"X": lmat_349, "syms": syms_349, "y": y_349, "tech": "RNA-seq CPM", "label": "GSE349247 (2026)\n(13 Ctrl vs 62 HF)"},
        "GSE165303_Proc": {"X": mat_165, "syms": syms_165, "y": y_165, "tech": "RNA-seq Processed", "label": "GSE165303\n(51 NF vs 50 DCM)"},
        "GSE116250_RPKM": {"X": g116["mat"], "syms": g116["syms"], "y": g116["y"], "tech": "RNA-seq RPKM", "label": "GSE116250 (RPKM)\n(14 NF vs 50 HF)"},
        "GSE46224_FPKM": {"X": mat_46, "syms": syms_46, "y": y_46, "tech": "RNA-seq FPKM", "label": "GSE46224 (FPKM)\n(8 NF vs 16 Pre-LVAD)"},
        "GSE57338_Array": {"X": g57["mat"], "syms": g57["syms"], "y": g57["y"], "tech": "Affymetrix Array", "label": "GSE57338 (Array)\n(136 NF vs 177 HF)"},
    }

    cal_rows = []
    rng_auc = np.random.default_rng(20261120)
    for cname, cd in cohorts.items():
        idx_c = resolve_strict_symbols(cd["syms"], sig_syms)
        sc_map = {
            "Cohort-Z": score_cohort_z(cd["X"], idx_c, sig_w)[0],
            "GSVA": score_gsva_bidirectional(cd["X"], idx_c, sig_w),
            "SS-Rank": score_single_sample_rank(cd["X"], idx_c, sig_w)[0],
            "SS-Rank-Internal": score_ss_rank_internal(cd["X"], idx_c, sig_w)[0],
        }
        y = cd["y"]
        for mname, sc in sc_map.items():
            auc = float(roc_auc_score(y, sc))
            boots = []
            for _ in range(2000):
                bi = rng_auc.integers(0, len(y), len(y))
                if len(np.unique(y[bi])) < 2:
                    continue
                boots.append(roc_auc_score(y[bi], sc[bi]))
            cal_rows.append({
                "cohort": cname, "label": cd["label"], "tech": cd["tech"], "n": len(y),
                "method": mname, "auc": auc,
                "ci_low": float(np.quantile(boots, 0.025)), "ci_high": float(np.quantile(boots, 0.975)),
            })
            if mname in ("Cohort-Z", "SS-Rank-Internal", "GSVA"):
                print(f"  Cal {cname:16s} | {mname:16s}: AUROC={auc:.4f} [{np.quantile(boots,0.025):.4f},{np.quantile(boots,0.975):.4f}]")
    pd.DataFrame(cal_rows).to_csv(OUT_DIR / "v6_4_table_clinical_6cohorts_calibration.csv", index=False)

if __name__ == "__main__":
    main()
