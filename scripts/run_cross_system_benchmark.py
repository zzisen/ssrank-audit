"""
Cross-system (2D ScreenSeq, TPM1/DMD EHTs) and clinical adult myocardium benchmark.
"""

from __future__ import annotations
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Reuse clean core utilities from Part 1
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_3d_cardiopedia_benchmark import (
    DATA_DIR,
    DS,
    OUT_DIR,
    ROOT,
    V5_ROOT,
    V5_TABLES,
    bootstrap_spearman_ci,
    iter_xlsx_rows,
    load_adult_nf_reference,
    load_fixed_programs,
    log2_cpm,
    resolve_strict_symbols,
    score_cohort_z,
    score_ref_frozen_z,
    score_single_sample_rank,
    spearman_rho,
)

GTF_PATH = DATA_DIR / "reference/Homo_sapiens.GRCh38.104.gtf.gz"


def bootstrap_auc_ci(y_true, y_score, n_boot=3000, seed=20261120):
    y_true = np.asarray(y_true, int)
    y_score = np.asarray(y_score, float)
    obs = float(roc_auc_score(y_true, y_score))
    rng = np.random.default_rng(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(float(roc_auc_score(y_true[idx], y_score[idx])))
    lo, hi = np.quantile(aucs, [0.025, 0.975])
    return {"auc": obs, "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def load_gtf_1to1():
    """Exact V5 1-to-1 gene_id <-> gene_name mapping from Ensembl GRCh38.104 GTF."""
    sym_to_id = defaultdict(set)
    id_to_sym = defaultdict(set)
    with gzip.open(GTF_PATH, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 9 or parts[2] != "gene":
                continue
            attrs = dict(re.findall(r'([A-Za-z_]+) "([^"]*)"', parts[8]))
            gid = attrs.get("gene_id")
            gnm = attrs.get("gene_name")
            if gid and gnm:
                sym_to_id[gnm].add(gid)
                id_to_sym[gid].add(gnm)
    sym2id_1to1 = {}
    id2sym_1to1 = {}
    for sym, gids in sym_to_id.items():
        if len(gids) == 1:
            gid = next(iter(gids))
            if len(id_to_sym[gid]) == 1:
                sym2id_1to1[sym] = gid
                id2sym_1to1[gid] = sym
    return sym2id_1to1, id2sym_1to1


def build_adult_nf_ref_by_symbol(id2sym_1to1):
    """Build frozen Adult-NF reference mean/SD (GSE141910 NF, n=166) keyed by 1-to-1 gene symbol."""
    nf_ens, nf_lmat, n_nf = load_adult_nf_reference()
    ref_mu, ref_sd = {}, {}
    for j, eid in enumerate(nf_ens):
        sym = id2sym_1to1.get(eid)
        if sym:
            col = nf_lmat[:, j]
            sd = float(col.std(ddof=1))
            if sd > 1e-12:
                ref_mu[sym] = float(col.mean())
                ref_sd[sym] = sd
    return ref_mu, ref_sd, n_nf


def run_2d_screenseq(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd):
    print("\n[A] Running 2D ScreenSeq (GSE244740) multi-paradigm & cell-type benchmark...")
    b6_dir = DS / "GSE244740"
    # Load samples & assays using V5's exact plan via b6_all_program_condition_statuses.csv for exact eligibility parity!
    v5_b6_cond = pd.read_csv(V5_TABLES / "b6_all_program_condition_statuses.csv")
    v5_b6_main_ok = v5_b6_cond[(v5_b6_cond["program"] == "main") & (v5_b6_cond["status"] == "eligible")].copy()
    v5_b6_drugs = pd.read_csv(V5_TABLES / "b6_drug_summary.csv")

    # Load sample metadata (sm6218.txt)
    samples_df = pd.read_csv(b6_dir / "iemt_a_2273378_sm6218.txt", sep="\t")
    keep_mask = samples_df["Excluded"].astype(str).str.upper() == "FALSE"
    keep_indices = np.where(keep_mask.to_numpy())[0]
    samples_kept = samples_df.loc[keep_mask].reset_index(drop=True)
    n_kept = len(samples_kept)
    print(f"  -> ScreenSeq retained RNA samples: {n_kept} / {len(samples_df)}")

    # Read ScreenSeq counts matrix (37 MB gzipped TSV: GeneID + 3072 sample cols)
    counts_path = b6_dir / "GSE244740_processed_data_counts.txt.gz"
    df_cnt = pd.read_csv(counts_path, sep="\t", index_col=0)
    assert list(df_cnt.columns) == list(samples_df["SampleID"]), "Sample column order mismatch in ScreenSeq"
    cnt_kept = df_cnt.iloc[:, keep_indices].to_numpy(dtype=float).T  # (2899, n_genes)
    lmat = log2_cpm(cnt_kept)
    gene_ids_b6 = [str(g).split(".")[0] for g in df_cnt.index]
    gene_syms_b6 = [id2sym_1to1.get(g, "") for g in gene_ids_b6]

    # Build normalized sample -> (drug, rank) mapping matching V5's exact drug/rank condition keys
    def norm_drug(s):
        return re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip()).strip("_")

    samples_kept["drug_norm"] = samples_kept["Treatment"].map(norm_drug)
    samples_kept["rank_str"] = samples_kept["Concentration Rank"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    cond_to_sample_rows = defaultdict(list)
    for i, r in samples_kept.iterrows():
        cond_to_sample_rows[(r["drug_norm"], r["rank_str"])].append(i)

    # Check parity against V5 main Cohort-Z
    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)]
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(float)
    sig_idx = resolve_strict_symbols(gene_syms_b6, sig_syms)

    s_cz, _ = score_cohort_z(lmat, sig_idx, sig_w)
    s_rk, _ = score_single_sample_rank(lmat, sig_idx, sig_w)
    s_rf, _ = score_ref_frozen_z(lmat, gene_syms_b6, ref_mu, ref_sd, sig_syms, sig_w)

    def summarize_2d(sample_scores, label_name):
        out_rows = []
        for assay in ("Amplitude", "DNA structure"):
            sub_ok = v5_b6_main_ok[v5_b6_main_ok["assay"] == assay]
            drug_list = sorted(sub_ok["drug"].unique())
            drug_s, drug_y, within_rhos = [], [], []
            for d in drug_list:
                d_sub = sub_ok[sub_ok["drug"] == d].sort_values("concentration_rank")
                cs, cy = [], []
                for _, crow in d_sub.iterrows():
                    r_key = str(crow["concentration_rank"]).strip()
                    # Match either '1' or '1.0' or exact string
                    rows_i = cond_to_sample_rows.get((d, r_key)) or cond_to_sample_rows.get((d, f"Rank {r_key}"))
                    if not rows_i:
                        # Try matching stripped integer in rank_str
                        for (dk, rk), idxs in cond_to_sample_rows.items():
                            if dk == d and re.sub(r"[^0-9]", "", rk) == re.sub(r"[^0-9]", "", r_key):
                                rows_i = idxs
                                break
                    if rows_i:
                        cs.append(float(np.mean(sample_scores[rows_i])))
                        cy.append(float(crow["assay_value"]))
                if len(cs) >= 3:
                    drug_s.append(float(np.mean(cs)))
                    drug_y.append(float(np.mean(cy)))
                    within_rhos.append(spearman_rho(cs, cy))
            ci = bootstrap_spearman_ci(drug_s, drug_y, n_boot=3000)
            w_arr = np.array([v for v in within_rhos if np.isfinite(v)])
            out_rows.append({
                "score_name": label_name,
                "assay": assay,
                "n_drugs": len(drug_s),
                "across_drug_rho": ci["rho"],
                "across_ci_low": ci["ci_low"],
                "across_ci_high": ci["ci_high"],
                "within_drug_median_rho": float(np.median(w_arr)),
                "within_drug_iqr": float(np.quantile(w_arr, 0.75) - np.quantile(w_arr, 0.25)),
                "within_drug_pct_neg": float(100.0 * np.mean(w_arr < 0)),
            })
        return out_rows

    # Verify exact V5 parity on ScreenSeq main Cohort-Z!
    cz_res = summarize_2d(s_cz, "ReHeaT2_mol_score | Cohort-Z")
    v5_amp = float(v5_b6_drugs[v5_b6_drugs["endpoint"] == "Amplitude"].pipe(lambda d: spearman_rho(d["score_mean"], d["assay_mean"])))
    print(f"  -> ScreenSeq V5 Parity Check: Amplitude Cohort-Z rho = {cz_res[0]['across_drug_rho']:.6f} vs V5 = {v5_amp:.6f}")

    all_b6_rows = []
    all_b6_rows.extend(cz_res)
    all_b6_rows.extend(summarize_2d(s_rk, "ReHeaT2_mol_score | Single-Sample Rank"))
    all_b6_rows.extend(summarize_2d(s_rf, "ReHeaT2_mol_score | Ref-Frozen-Z Adult-NF"))

    # Official ReHeaT2 cell-type axes in 2D ScreenSeq (Cohort-Z & Single-Sample Rank)
    for cname, col in [
        ("ReHeaT2_CM", "mcell_HFscore_CM"),
        ("ReHeaT2_Fib", "mcell_HFscore_Fib"),
        ("ReHeaT2_Endo", "mcell_HFscore_Endo"),
        ("ReHeaT2_Myeloid", "mcell_HFscore_Myeloid"),
        ("ReHeaT2_PC", "mcell_HFscore_PC"),
    ]:
        sub = mc.loc[mc[col].notna() & (mc[col] != 0)]
        aidx = resolve_strict_symbols(gene_syms_b6, list(sub["gene"].astype(str)))
        aw = sub[col].to_numpy(float)
        all_b6_rows.extend(summarize_2d(score_cohort_z(lmat, aidx, aw)[0], f"{cname} | Cohort-Z"))
        all_b6_rows.extend(summarize_2d(score_single_sample_rank(lmat, aidx, aw)[0], f"{cname} | Single-Sample Rank"))

    # Fixed biological programs in 2D ScreenSeq under Single-Sample Rank (and Cohort-Z)
    for pname in ("structural_exclusive", "conduction_exclusive", "hypoxia", "upr", "oxphos", "ecm", "fibroblast_marker"):
        pidx = resolve_strict_symbols(gene_syms_b6, fixed_progs[pname])
        pw = np.ones(len(fixed_progs[pname]), float)
        all_b6_rows.extend(summarize_2d(score_cohort_z(lmat, pidx, pw)[0], f"Prog_{pname} | Cohort-Z"))
        all_b6_rows.extend(summarize_2d(score_single_sample_rank(lmat, pidx, pw)[0], f"Prog_{pname} | Single-Sample Rank"))

    df_b6_out = pd.DataFrame(all_b6_rows)
    df_b6_out.to_csv(OUT_DIR / "v6_table7_2d_screenseq_multiparadigm.csv", index=False)
    print(df_b6_out[df_b6_out["score_name"].str.contains("mol_score|ReHeaT2_CM|ReHeaT2_Fib|Prog_upr|Prog_structural")].to_string(index=False))
    return df_b6_out


def run_tpm1_and_2d3d(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd):
    print("\n[B] Running TPM1 Isogenic EHT (GSE251993) & 2D-vs-3D Factorial EHT (GSE199242)...")
    # 1. TPM1 GSE251993
    tpm_dir = DS / "GSE251993"
    expected = [f"{g}_{i}" for g in ("WT", "E54K", "E62Q") for i in (1, 2, 3)]
    df_tpm = pd.read_csv(tpm_dir / "GSE251993_raw_counts.csv.gz")
    syms_tpm = [str(s).strip() for s in df_tpm["gene_name"]]
    mat_tpm = df_tpm[expected].to_numpy(float).T  # (9, n_genes)
    lmat_tpm = log2_cpm(mat_tpm)

    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)]
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(float)

    tpm_rows = []
    # Evaluate mol_score under all 3 paradigms + CM/Fib + fixed programs
    eval_list = [
        ("ReHeaT2_mol_score", sig_syms, sig_w),
    ]
    for cname, col in [("ReHeaT2_CM", "mcell_HFscore_CM"), ("ReHeaT2_Fib", "mcell_HFscore_Fib")]:
        sub = mc.loc[mc[col].notna() & (mc[col] != 0)]
        eval_list.append((cname, list(sub["gene"].astype(str)), sub[col].to_numpy(float)))
    for pname in ("structural_exclusive", "conduction_exclusive", "hypoxia", "upr", "ecm", "oxphos", "fibroblast_marker"):
        eval_list.append((f"Prog_{pname}", fixed_progs[pname], np.ones(len(fixed_progs[pname]), float)))

    for name, gsyms, w in eval_list:
        idx = resolve_strict_symbols(syms_tpm, gsyms)
        s_cz, _ = score_cohort_z(lmat_tpm, idx, w)
        s_rk, _ = score_single_sample_rank(lmat_tpm, idx, w)
        s_rf, _ = score_ref_frozen_z(lmat_tpm, syms_tpm, ref_mu, ref_sd, gsyms, w)
        for pname, sc in [("Cohort-Z", s_cz), ("Single-Sample Rank", s_rk), ("Ref-Frozen-Z", s_rf)]:
            if name.startswith("Prog_") and pname == "Ref-Frozen-Z":
                continue
            wt, e54, e62 = float(sc[0:3].mean()), float(sc[3:6].mean()), float(sc[6:9].mean())
            tpm_rows.append({
                "signature": name,
                "paradigm": pname,
                "WT_mean": wt,
                "E54K_mean": e54,
                "E62Q_mean": e62,
                "E54K_minus_WT": e54 - wt,
                "E62Q_minus_WT": e62 - wt,
                "E62Q_minus_E54K": e62 - e54,
            })
    df_tpm_out = pd.DataFrame(tpm_rows)
    df_tpm_out.to_csv(OUT_DIR / "v6_table8_tpm1_eht_multiparadigm.csv", index=False)
    # Parity check with V5 TPM1 (WT=0.0141, E54K=-0.0307, E62Q=0.0166)
    r0 = df_tpm_out[(df_tpm_out["signature"] == "ReHeaT2_mol_score") & (df_tpm_out["paradigm"] == "Cohort-Z")].iloc[0]
    print(f"  -> TPM1 V5 Parity Check: WT={r0['WT_mean']:.4f}, E54K={r0['E54K_mean']:.4f}, E62Q={r0['E62Q_mean']:.4f}")
    print(df_tpm_out[df_tpm_out["signature"].isin(["ReHeaT2_mol_score", "ReHeaT2_CM", "ReHeaT2_Fib", "Prog_structural_exclusive", "Prog_hypoxia"])].to_string(index=False))

    # 2. GSE199242 (2D monolayer vs 3D EHT x WT vs DMD 263delG)
    df199 = pd.read_csv(DS / "GSE199242/GSE199242_countMatrix_2D+3D.csv.gz")
    cols_199 = ["WT_2D_1", "WT_2D_2", "WT_2D_3", "DMD_2D_1", "DMD_2D_2", "DMD_2D_3",
                "WT_3D_1", "WT_3D_2", "WT_3D_3", "DMD_3D_1", "DMD_3D_2", "DMD_3D_3"]
    gids_199 = [str(g).split(".")[0] for g in df199["GeneID"]]
    syms_199 = [id2sym_1to1.get(g, "") for g in gids_199]
    # Restore the finite-row filtering used in v6_part2b_fix_2d3d_and_audit.py.
    # One or more non-finite counts otherwise contaminate library sums and all scores.
    raw_199 = df199[cols_199].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid_rows = np.all(np.isfinite(raw_199), axis=1)
    syms_199 = [s for s, ok in zip(syms_199, valid_rows) if ok]
    lmat_199 = log2_cpm(raw_199[valid_rows].T)
    print(f"  -> GSE199242 finite rows: {valid_rows.sum()} / {len(valid_rows)}")
    rows_199 = []
    wanted = {"ReHeaT2_mol_score", "ReHeaT2_CM", "ReHeaT2_Fib", "Prog_structural_exclusive", "Prog_conduction_exclusive", "Prog_hypoxia", "Prog_ecm"}
    for name, gsyms, w in [entry for entry in eval_list if entry[0] in wanted]:
        idx = resolve_strict_symbols(syms_199, gsyms)
        s_cz, _ = score_cohort_z(lmat_199, idx, w)
        s_rk, _ = score_single_sample_rank(lmat_199, idx, w)
        s_rf, _ = score_ref_frozen_z(lmat_199, syms_199, ref_mu, ref_sd, gsyms, w)
        for pname, sc in [("Cohort-Z", s_cz), ("Single-Sample Rank", s_rk), ("Ref-Frozen-Z", s_rf)]:
            if name.startswith("Prog_") and pname == "Ref-Frozen-Z":
                continue
            rows_199.append({
                "signature": name,
                "paradigm": pname,
                "WT_2D": float(sc[0:3].mean()),
                "DMD_2D": float(sc[3:6].mean()),
                "WT_3D": float(sc[6:9].mean()),
                "DMD_3D": float(sc[9:12].mean()),
                "shift_3D_minus_2D_WT": float(sc[6:9].mean() - sc[0:3].mean()),
                "shift_3D_minus_2D_DMD": float(sc[9:12].mean() - sc[3:6].mean()),
                "shift_DMD_minus_WT_2D": float(sc[3:6].mean() - sc[0:3].mean()),
                "shift_DMD_minus_WT_3D": float(sc[9:12].mean() - sc[6:9].mean()),
            })
    df_199_out = pd.DataFrame(rows_199)
    df_199_out.to_csv(OUT_DIR / "v6_table9_gse199242_2d_vs_3d_shift.csv", index=False)
    print("\n=== GSE199242 2D vs 3D Baseline & DMD Shift ===")
    print(df_199_out[df_199_out["signature"].isin(["ReHeaT2_mol_score", "ReHeaT2_CM", "ReHeaT2_Fib", "Prog_structural_exclusive"])].to_string(index=False))
    return df_tpm_out, df_199_out


def run_clinical_cohorts(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd):
    print("\n[C] Running Clinical Adult Myocardium Calibration, Prevalence-Shift Stress Test & Paired LVAD...")
    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)]
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(float)

    clinical_rows = []
    # ---- 1. GSE165303 (51 NF vs 50 DCM) ----
    df165 = pd.read_csv(DS / "GSE165303/GSE165303_RNA-Seq_master_table.txt.gz", sep="\t")
    nf_cols = sorted([c for c in df165.columns if re.fullmatch(r"NF-\d+", c)], key=lambda x: int(x.split("-")[1]))
    dcm_cols = sorted([c for c in df165.columns if re.fullmatch(r"DCM-\d+", c)], key=lambda x: int(x.split("-")[1]))
    s_cols_165 = nf_cols + dcm_cols
    y_165 = np.array([0] * len(nf_cols) + [1] * len(dcm_cols), int)
    syms_165 = [str(s).strip() for s in df165["Symbol"].fillna("")]
    mat_165 = df165[s_cols_165].apply(pd.to_numeric, errors="coerce").to_numpy(float).T  # (101, n_genes)
    # Mask any gene with NaN
    fin_mask = np.all(np.isfinite(mat_165), axis=0)
    mat_165 = mat_165[:, fin_mask]
    syms_165 = [s for s, m in zip(syms_165, fin_mask) if m]

    idx_165 = resolve_strict_symbols(syms_165, sig_syms)
    s165_cz, _ = score_cohort_z(mat_165, idx_165, sig_w)
    s165_rk, _ = score_single_sample_rank(mat_165, idx_165, sig_w)
    for pname, sc in [("Cohort-Z", s165_cz), ("Single-Sample Rank", s165_rk)]:
        ci = bootstrap_auc_ci(y_165, sc)
        clinical_rows.append({
            "cohort": "GSE165303 (51 NF vs 50 DCM)",
            "provenance_note": "Post-hoc adult context; donor overlap & age/sex unresolved; processed scale",
            "signature": "ReHeaT2_mol_score",
            "paradigm": pname,
            "auc": ci["auc"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "median_ctrl": float(np.median(sc[y_165 == 0])),
            "median_case": float(np.median(sc[y_165 == 1])),
        })
    # Also CM & Fib in GSE165303
    for cname, col in [("ReHeaT2_CM", "mcell_HFscore_CM"), ("ReHeaT2_Fib", "mcell_HFscore_Fib"), ("ReHeaT2_Endo", "mcell_HFscore_Endo")]:
        sub = mc.loc[mc[col].notna() & (mc[col] != 0)]
        aidx = resolve_strict_symbols(syms_165, list(sub["gene"].astype(str)))
        aw = sub[col].to_numpy(float)
        for pname, sc in [("Cohort-Z", score_cohort_z(mat_165, aidx, aw)[0]), ("Single-Sample Rank", score_single_sample_rank(mat_165, aidx, aw)[0])]:
            ci = bootstrap_auc_ci(y_165, sc)
            clinical_rows.append({
                "cohort": "GSE165303 (51 NF vs 50 DCM)",
                "provenance_note": "Post-hoc adult context; donor overlap & age/sex unresolved; processed scale",
                "signature": cname,
                "paradigm": pname,
                "auc": ci["auc"],
                "ci_low": ci["ci_low"],
                "ci_high": ci["ci_high"],
                "median_ctrl": float(np.median(sc[y_165 == 0])),
                "median_case": float(np.median(sc[y_165 == 1])),
            })

    # KILLER CIBM EXPERIMENT: Clinical Case-Mix / Prevalence-Shift Distortion on GSE165303!
    # Fix 10 NF + 10 DCM probe patients; embed them in a 20% DCM batch (40 NF + 10 DCM) vs an 80% DCM batch (10 NF + 40 DCM)!
    probe_nf = np.arange(0, 10)
    probe_dcm = np.arange(51, 61)
    probe_idx = np.concatenate([probe_nf, probe_dcm])
    batch_20pct_dcm = np.concatenate([np.arange(0, 40), np.arange(51, 61)])   # 40 NF + 10 DCM (20% DCM)
    batch_80pct_dcm = np.concatenate([np.arange(0, 10), np.arange(51, 91)])   # 10 NF + 40 DCM (80% DCM)
    # In batch_20pct_dcm, probe_nf are indices 0..9, probe_dcm are 40..49
    # In batch_80pct_dcm, probe_nf are indices 0..9, probe_dcm are 10..19
    prev_rows = []
    for pname, fn in [
        ("Cohort-Z (transductive)", lambda M: score_cohort_z(M, idx_165, sig_w)[0]),
        ("Single-Sample Rank (inductive)", lambda M: score_single_sample_rank(M, idx_165, sig_w)[0]),
    ]:
        s_full = fn(mat_165)[probe_idx]
        s_20 = fn(mat_165[batch_20pct_dcm])
        s_20_probe = np.concatenate([s_20[0:10], s_20[40:50]])
        s_80 = fn(mat_165[batch_80pct_dcm])
        s_80_probe = np.concatenate([s_80[0:10], s_80[10:20]])
        # Evaluate zero-threshold classification accuracy under 20% vs 80% prevalence (using threshold learned at 50% balanced cohort = median(s_full))
        thr_50 = float(np.median(fn(mat_165)))
        y_probe = np.array([0] * 10 + [1] * 10)
        acc_20 = float(np.mean((s_20_probe > thr_50) == y_probe))
        acc_80 = float(np.mean((s_80_probe > thr_50) == y_probe))
        spec_20 = float(np.mean((s_20_probe[:10] <= thr_50)))
        sens_20 = float(np.mean((s_20_probe[10:] > thr_50)))
        spec_80 = float(np.mean((s_80_probe[:10] <= thr_50)))
        sens_80 = float(np.mean((s_80_probe[10:] > thr_50)))
        diff = np.abs(s_20_probe - s_80_probe)
        sd_full = float(np.std(s_full, ddof=1))
        prev_rows.append({
            "paradigm": pname,
            "mean_abs_shift_20_vs_80": float(diff.mean()),
            "max_abs_shift_20_vs_80": float(diff.max()),
            "mean_shift_in_sd": float(diff.mean() / sd_full),
            "class_flips_of_20": int(np.sum((s_20_probe > thr_50) != (s_80_probe > thr_50))),
            "acc_at_20pct_prev": acc_20,
            "sens_at_20pct_prev": sens_20,
            "spec_at_20pct_prev": spec_20,
            "acc_at_80pct_prev": acc_80,
            "sens_at_80pct_prev": sens_80,
            "spec_at_80pct_prev": spec_80,
        })
    df_prev = pd.DataFrame(prev_rows)
    df_prev.to_csv(OUT_DIR / "v6_table10_clinical_prevalence_shift_gse165303.csv", index=False)
    print("\n=== Clinical Case-Mix / Prevalence-Shift Distortion (GSE165303, 20% vs 80% DCM) ===")
    print(df_prev.to_string(index=False))

    # ---- 2. GSE349247 (Sep 2026, 75 adult LV: Control / Nonischemic / Ischemic + age/sex/race/batch) ----
    meta349 = {}
    with gzip.open(DS / "GSE349247/GSE349247_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as fh:
        ci = 0
        for line in fh:
            if line.startswith("!Sample_title"):
                meta349["sample"] = [x.strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                meta349[f"ch_{ci}"] = [x.strip('"') for x in line.strip().split("\t")[1:]]
                ci += 1
    m349 = pd.DataFrame(meta349)
    m349["status"] = m349["ch_1"].str.replace("disease status: ", "", regex=False)
    m349["sex"] = m349["ch_2"].str.replace("Sex: ", "", regex=False)
    m349["age_num"] = pd.to_numeric(m349["ch_3"].str.replace("age: ", "", regex=False), errors="coerce")
    m349["race"] = m349["ch_4"].str.replace("race: ", "", regex=False)
    m349["batch"] = m349["ch_5"].str.replace("batch: ", "", regex=False)
    m349["is_hf"] = (m349["status"] != "Control").astype(int)
    print("\n  -> GSE349247 Batch x Status cross-tab:")
    print(pd.crosstab(m349["batch"], m349["status"]))

    df349_cnt = pd.read_csv(DS / "GSE349247/GSE349247_heart_rnaseq_gene_counts.csv.gz")
    gids_349 = [str(g).split(".")[0] for g in df349_cnt["Geneid"]]
    syms_349 = [id2sym_1to1.get(g, "") for g in gids_349]
    s_order_349 = list(m349["sample"])
    mat_349 = df349_cnt[s_order_349].to_numpy(float).T
    lmat_349 = log2_cpm(mat_349)

    idx_349 = resolve_strict_symbols(syms_349, sig_syms)
    s349_cz, _ = score_cohort_z(lmat_349, idx_349, sig_w)
    s349_rk, _ = score_single_sample_rank(lmat_349, idx_349, sig_w)
    s349_rf, _ = score_ref_frozen_z(lmat_349, syms_349, ref_mu, ref_sd, sig_syms, sig_w)
    y_349 = m349["is_hf"].to_numpy(int)

    for pname, sc in [("Cohort-Z", s349_cz), ("Single-Sample Rank", s349_rk), ("Ref-Frozen-Z Adult-NF", s349_rf)]:
        ci = bootstrap_auc_ci(y_349, sc)
        # Also evaluate strictly within batch 2 (where Control=7, HF=14 — balanced batch!) and batch 1 (Control=6, HF=48)
        b1 = m349["batch"].to_numpy() == "1"
        b2 = m349["batch"].to_numpy() == "2"
        auc_b1 = float(roc_auc_score(y_349[b1], sc[b1])) if len(np.unique(y_349[b1])) == 2 else float("nan")
        auc_b2 = float(roc_auc_score(y_349[b2], sc[b2])) if len(np.unique(y_349[b2])) == 2 else float("nan")
        clinical_rows.append({
            "cohort": "GSE349247 (13 Ctrl vs 62 HF, Sep 2026)",
            "provenance_note": f"Post-ReHeaT2 release; biobank ID prefix overlap risk disclosed; batch1 AUC={auc_b1:.3f}, batch2 AUC={auc_b2:.3f}",
            "signature": "ReHeaT2_mol_score",
            "paradigm": pname,
            "auc": ci["auc"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "median_ctrl": float(np.median(sc[y_349 == 0])),
            "median_case": float(np.median(sc[y_349 == 1])),
        })

    # Covariate-adjusted model in GSE349247 (complete age/sex/race/batch subset)
    m349["score_cz"] = s349_cz
    m349["score_rk"] = s349_rk
    m349["score_rf"] = s349_rf
    m349.to_csv(OUT_DIR / "v6_gse349247_sample_scores.csv", index=False)

    # ---- 3. GSE46224 (8 NF + 16 Paired Pre-LVAD vs Post-LVAD Hearts) ----
    df46 = pd.read_csv(DS / "GSE46224/GSE46224_Yang_et_al_human_heart_RNASeq.txt.gz", sep="\t")
    syms_46 = [str(s).strip() for s in df46["Gene"].fillna("")]
    nf_46 = [f"NF{i}" for i in range(1, 9)]
    pre_icm = [f"ICM{i}" for i in range(1, 9)]
    post_icm = [f"ICM+LVAD{i}" for i in range(1, 9)]
    pre_nicm = [f"NICM{i}" for i in range(1, 9)]
    post_nicm = [f"NICM+LVAD{i}" for i in range(1, 9)]
    all_46 = nf_46 + pre_icm + post_icm + pre_nicm + post_nicm
    # GSE46224 values are FPKM/RPKM -> log2(x + 1)
    mat_46 = np.log2(1.0 + np.maximum(df46[all_46].to_numpy(float).T, 0.0))

    lvad_rows = []
    lvad_eval = [
        ("ReHeaT2_mol_score", sig_syms, sig_w),
    ]
    for cname, col in [("ReHeaT2_CM", "mcell_HFscore_CM"), ("ReHeaT2_Fib", "mcell_HFscore_Fib")]:
        sub = mc.loc[mc[col].notna() & (mc[col] != 0)]
        lvad_eval.append((cname, list(sub["gene"].astype(str)), sub[col].to_numpy(float)))
    for pname in ("structural_exclusive", "ecm", "hypoxia"):
        lvad_eval.append((f"Prog_{pname}", fixed_progs[pname], np.ones(len(fixed_progs[pname]), float)))

    for sname, gsyms, w in lvad_eval:
        idx = resolve_strict_symbols(syms_46, gsyms)
        for pname, sc in [
            ("Cohort-Z", score_cohort_z(mat_46, idx, w)[0]),
            ("Single-Sample Rank", score_single_sample_rank(mat_46, idx, w)[0]),
        ]:
            s_nf = sc[0:8]
            s_pre = np.concatenate([sc[8:16], sc[24:32]])   # 16 Pre-LVAD (8 ICM + 8 NICM)
            s_post = np.concatenate([sc[16:24], sc[32:40]]) # 16 Post-LVAD (paired!)
            delta = s_post - s_pre
            w_stat, w_p = stats.wilcoxon(delta)
            dz = float(delta.mean() / delta.std(ddof=1))
            auc_pre_nf = float(roc_auc_score(np.array([0] * 8 + [1] * 16), np.concatenate([s_nf, s_pre])))
            lvad_rows.append({
                "signature": sname,
                "paradigm": pname,
                "auc_preLVAD_vs_NF": auc_pre_nf,
                "mean_NF": float(s_nf.mean()),
                "mean_preLVAD": float(s_pre.mean()),
                "mean_postLVAD": float(s_post.mean()),
                "paired_delta_post_minus_pre": float(delta.mean()),
                "paired_cohens_dz": dz,
                "n_decreased_of_16": int(np.sum(delta < 0)),
                "wilcoxon_p_two_sided": float(w_p),
            })
    df_lvad = pd.DataFrame(lvad_rows)
    df_lvad.to_csv(OUT_DIR / "v6_table11_gse46224_paired_lvad.csv", index=False)
    print("\n=== GSE46224 Paired Mechanical Unloading (Pre-LVAD vs Post-LVAD, n=16 pairs) ===")
    print(df_lvad.to_string(index=False))

    # ---- 4. GSE141910 (MAGNet 332 NF/DCM — explicitly labeled UPSTREAM ADULT REFERENCE) ----
    ph = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph_sub = ph[ph["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    s_ids_141 = list(ph_sub["sample_name"].astype(str))
    mat_141 = cnt141[s_ids_141].to_numpy(float).T
    lmat_141 = log2_cpm(mat_141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    idx_141 = resolve_strict_symbols(syms_141, sig_syms)

    s141_cz, _ = score_cohort_z(lmat_141, idx_141, sig_w)
    s141_rk, _ = score_single_sample_rank(lmat_141, idx_141, sig_w)
    y_141 = (ph_sub["etiology"] == "DCM").astype(int).to_numpy()
    lvef = pd.to_numeric(ph_sub["LVEF"], errors="coerce").to_numpy(float)

    for pname, sc in [("Cohort-Z", s141_cz), ("Single-Sample Rank", s141_rk)]:
        ci = bootstrap_auc_ci(y_141, sc)
        lvef_mask = np.isfinite(lvef)
        lvef_rho_all = spearman_rho(sc[lvef_mask], lvef[lvef_mask])
        dcm_lvef = lvef_mask & (y_141 == 1)
        lvef_rho_dcm = spearman_rho(sc[dcm_lvef], lvef[dcm_lvef])
        clinical_rows.append({
            "cohort": "GSE141910 MAGNet (166 NF vs 166 DCM, upstream ref)",
            "provenance_note": f"Explicit upstream adult reference (not blind); LVEF rho all={lvef_rho_all:.3f} (n={int(lvef_mask.sum())}), DCM-only={lvef_rho_dcm:.3f} (n={int(dcm_lvef.sum())})",
            "signature": "ReHeaT2_mol_score",
            "paradigm": pname,
            "auc": ci["auc"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "median_ctrl": float(np.median(sc[y_141 == 0])),
            "median_case": float(np.median(sc[y_141 == 1])),
        })

    # Covariate-adjusted multivariable logistic regression on GSE349247 and GSE141910 (age + sex + race + score)
    cov_adj_rows = []
    # 4a. GSE141910 (upstream reference)
    cov141 = ph_sub[["age", "gender", "race"]].copy()
    cov141["age"] = pd.to_numeric(cov141["age"], errors="coerce")
    cov141["male"] = (cov141["gender"] == "Male").astype(float)
    cov141["aa"] = (cov141["race"] == "AA").astype(float)
    for pname, sc in [("Cohort-Z", s141_cz), ("Single-Sample Rank", s141_rk)]:
        ok = cov141["age"].notna().to_numpy()
        sc_z = (sc[ok] - sc[ok].mean()) / sc[ok].std(ddof=1)
        X_cov = np.column_stack([cov141.loc[ok, "age"].to_numpy(), cov141.loc[ok, "male"].to_numpy(), cov141.loc[ok, "aa"].to_numpy()])
        X_full = np.column_stack([sc_z, X_cov])
        m_cov = LogisticRegression(C=1e4, max_iter=1000).fit(X_cov, y_141[ok])
        m_full = LogisticRegression(C=1e4, max_iter=1000).fit(X_full, y_141[ok])
        auc_cov = float(roc_auc_score(y_141[ok], m_cov.predict_proba(X_cov)[:, 1]))
        auc_full = float(roc_auc_score(y_141[ok], m_full.predict_proba(X_full)[:, 1]))
        cov_adj_rows.append({
            "cohort": "GSE141910 (upstream ref, n=332)",
            "paradigm": pname,
            "n_complete": int(ok.sum()),
            "covariates": "age + sex + race",
            "auc_covariates_only": auc_cov,
            "auc_score_plus_covariates": auc_full,
            "delta_auc": auc_full - auc_cov,
            "std_log_odds_score": float(m_full.coef_[0, 0]),
        })

    # 4b. GSE349247 (Sep 2026)
    ok349 = m349["age_num"].notna() & (m349["race"] != "not available")
    ok349 = ok349.to_numpy()
    X_cov349 = np.column_stack([
        m349.loc[ok349, "age_num"].to_numpy(),
        (m349.loc[ok349, "sex"] == "male").astype(float).to_numpy(),
        (m349.loc[ok349, "race"] == "Black").astype(float).to_numpy(),
        (m349.loc[ok349, "batch"] == "2").astype(float).to_numpy(),
    ])
    for pname, sc in [("Cohort-Z", s349_cz), ("Single-Sample Rank", s349_rk), ("Ref-Frozen-Z Adult-NF", s349_rf)]:
        sc_z = (sc[ok349] - sc[ok349].mean()) / sc[ok349].std(ddof=1)
        X_full = np.column_stack([sc_z, X_cov349])
        m_cov = LogisticRegression(C=1e4, max_iter=1000).fit(X_cov349, y_349[ok349])
        m_full = LogisticRegression(C=1e4, max_iter=1000).fit(X_full, y_349[ok349])
        auc_cov = float(roc_auc_score(y_349[ok349], m_cov.predict_proba(X_cov349)[:, 1]))
        auc_full = float(roc_auc_score(y_349[ok349], m_full.predict_proba(X_full)[:, 1]))
        cov_adj_rows.append({
            "cohort": f"GSE349247 (Sep 2026, complete n={int(ok349.sum())})",
            "paradigm": pname,
            "n_complete": int(ok349.sum()),
            "covariates": "age + sex + race + batch",
            "auc_covariates_only": auc_cov,
            "auc_score_plus_covariates": auc_full,
            "delta_auc": auc_full - auc_cov,
            "std_log_odds_score": float(m_full.coef_[0, 0]),
        })

    df_clin = pd.DataFrame(clinical_rows)
    df_clin.to_csv(OUT_DIR / "v6_table12_clinical_cohorts_summary.csv", index=False)
    df_cov = pd.DataFrame(cov_adj_rows)
    df_cov.to_csv(OUT_DIR / "v6_table13_clinical_covariate_adjusted.csv", index=False)

    print("\n=== Clinical Cohorts Multi-Paradigm Summary ===")
    print(df_clin[["cohort", "signature", "paradigm", "auc", "ci_low", "ci_high"]].to_string(index=False))
    print("\n=== Clinical Covariate-Adjusted Logistic Models ===")
    print(df_cov.to_string(index=False))
    return df_clin, df_cov, df_prev, df_lvad


def main():
    print("Loading 1-to-1 Ensembl GRCh38.104 mapping, ReHeaT2, fixed programs, and Adult-NF frozen reference...")
    _, id2sym_1to1 = load_gtf_1to1()
    mc = pd.read_csv(DS / "ReHeaT2_zenodo_15261569/mol_consistency.csv")
    fixed_progs = load_fixed_programs()
    ref_mu, ref_sd, n_nf = build_adult_nf_ref_by_symbol(id2sym_1to1)
    print(f"  -> 1-to-1 GTF symbols: {len(id2sym_1to1)}, Adult-NF calibrated symbols: {len(ref_mu)} (n={n_nf})")

    run_2d_screenseq(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd)
    run_tpm1_and_2d3d(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd)
    run_clinical_cohorts(mc, fixed_progs, id2sym_1to1, ref_mu, ref_sd)
    print("\nPart 2 completed cleanly! All tables saved to", OUT_DIR)


if __name__ == "__main__":
    main()
