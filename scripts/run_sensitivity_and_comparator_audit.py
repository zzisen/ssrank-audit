"""
Targeted sensitivity analyses (GSE263297 missingness, GSE198945 duplicate library, external
GSE141910->GSE249925 CPM/TPM transfer) and standard comparator checks (gseapy v1.3.2 & right-continuous ECDF).
"""

from __future__ import annotations
import gzip
import json
import math
import re
import sys
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssrank_audit as sa
from run_3d_cardiopedia_benchmark import (
    DATA_DIR,
    DS,
    OUT_DIR,
    V5_TABLES,
    load_cardiopedia_strict,
    log2_cpm,
    resolve_strict_symbols,
    score_cohort_z,
)
from run_cross_system_benchmark import load_gtf_1to1
from benchmark_utils import load_gse57338 as load_gse57338_microarray
from benchmark_utils import (
    compute_welch_t_signature,
    eval_threshold,
    youden_threshold,
)
from run_external_transfer_benchmark import (
    HELDOUT,
    _col_letters_to_idx,
    load_heldout_gse198945,
    load_heldout_gse249925,
    score_gsva_from_Z,
)


def load_gse263297_raw_nan():
    """Load GSE263297 preserving NaN for blank/non-finite cells and tracking exact duplicate symbols."""
    xlsx_path = HELDOUT / "GSE263297" / "GSE263297_HUNTER_RNAseq_data_Normalised_LOG2_transformed.xlsx"
    with zipfile.ZipFile(xlsx_path) as zf:
        ss = [
            "".join(t.text or "" for t in si.iter() if t.tag.endswith("}t"))
            for si in ET.fromstring(zf.read("xl/sharedStrings.xml"))
        ]
        ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
        sample_ids, conds = [], []
        gene_syms, data_rows = [], []
        rnum = 0
        for _, el in ET.iterparse(zf.open("xl/worksheets/sheet1.xml"), events=("end",)):
            if el.tag == f"{ns}row":
                cells = {}
                for c in el:
                    ref = c.attrib.get("r", "")
                    ci = _col_letters_to_idx(ref) if ref else len(cells)
                    t = c.attrib.get("t")
                    v = c.find(f"{ns}v")
                    val = ss[int(v.text)] if (t == "s" and v is not None) else (v.text if v is not None else "")
                    cells[ci] = val
                if rnum == 1:
                    sample_ids = [cells.get(c, "").strip() for c in range(2, 23)]
                elif rnum == 2:
                    conds = [cells.get(c, "").strip() for c in range(2, 23)]
                elif rnum >= 6:
                    sym = cells.get(1, "").strip()
                    if sym and sym.upper() != "NA":
                        vals = []
                        ok_any = False
                        for c in range(2, 23):
                            raw_v = cells.get(c, "")
                            try:
                                fv = float(raw_v)
                                if math.isfinite(fv):
                                    ok_any = True
                                    vals.append(fv)
                                else:
                                    vals.append(np.nan)
                            except Exception:
                                vals.append(np.nan)
                        if ok_any:
                            gene_syms.append(sym)
                            data_rows.append(vals)
                rnum += 1
                el.clear()
    mat_nan = np.asarray(data_rows, dtype=float).T  # (21, G)
    y = np.array([0 if "donor" in c.lower() else 1 for c in conds], int)
    return {"mat_nan": mat_nan, "syms": gene_syms, "y": y, "titles": sample_ids, "conds": conds}


def build_locked_discovery_sources(id2sym_1to1, sig_syms, sig_w, k200_syms, k200_w):
    """Reconstruct exact V7.0 deduplicated discovery calibration sources (GSE141910, GSE57338, GSE165303)."""
    # 1. GSE141910
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph141_sub = ph141[ph141["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    raw141 = cnt141[list(ph141_sub["sample_name"].astype(str))].to_numpy(float).T
    lmat_141 = log2_cpm(raw141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    y_141 = (ph141_sub["etiology"] == "DCM").astype(int).to_numpy()

    # 2. GSE57338
    g57 = load_gse57338_microarray()
    codes57 = set(g57["codes"])
    codes141 = set(ph141_sub["sample_name"].astype(str).str.strip())
    mask141_ovlp = ph141_sub["sample_name"].astype(str).str.strip().isin(codes57).to_numpy()
    mask57_ovlp = np.array([c in codes141 for c in g57["codes"]], dtype=bool)
    assert int(mask141_ovlp.sum()) == 48 and int(mask57_ovlp.sum()) == 48

    # 3. GSE165303
    df165 = pd.read_csv(DS / "GSE165303/GSE165303_RNA-Seq_master_table.txt.gz", sep="\t")
    nf165 = sorted([c for c in df165.columns if re.fullmatch(r"NF-\d+", c)], key=lambda x: int(x.split("-")[1]))
    dcm165 = sorted([c for c in df165.columns if re.fullmatch(r"DCM-\d+", c)], key=lambda x: int(x.split("-")[1]))
    mat_165 = df165[nf165 + dcm165].apply(pd.to_numeric, errors="coerce").to_numpy(float).T
    fin165 = np.all(np.isfinite(mat_165), axis=0)
    mat_165 = mat_165[:, fin165]
    syms_165 = [str(s).strip() for s, ok in zip(df165["Symbol"].fillna(""), fin165) if ok]
    y_165 = np.array([0] * 51 + [1] * 50, int)

    disc = {
        "GSE141910_CPM": {"X": lmat_141[~mask141_ovlp], "y": y_141[~mask141_ovlp], "syms": syms_141},
        "GSE57338_Array": {"X": g57["mat"][~mask57_ovlp], "y": g57["y"][~mask57_ovlp], "syms": g57["syms"]},
        "GSE165303_Proc": {"X": mat_165, "y": y_165, "syms": syms_165},
    }
    dn141_s, dn141_w = compute_welch_t_signature(disc["GSE141910_CPM"]["X"], disc["GSE141910_CPM"]["y"], syms_141, 200)
    dn57_s, dn57_w = compute_welch_t_signature(disc["GSE57338_Array"]["X"], disc["GSE57338_Array"]["y"], g57["syms"], 200)
    dn165_s, dn165_w = compute_welch_t_signature(disc["GSE165303_Proc"]["X"], disc["GSE165303_Proc"]["y"], syms_165, 500)

    locked_specs = [
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE141910_CPM"),
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE57338_Array"),
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE165303_Proc"),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE141910_CPM"),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE57338_Array"),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE165303_Proc"),
        ("DeNovo_K200_MAGNet", dn141_s, dn141_w, "GSE141910_CPM"),
        ("DeNovo_K200_Array", dn57_s, dn57_w, "GSE57338_Array"),
        ("DeNovo_K500_Proc", dn165_s, dn165_w, "GSE165303_Proc"),
    ]
    return disc, locked_specs


def main():
    t0 = time.perf_counter()
    print("=== V7.1 Forensic Sensitivity & Standard Comparator Audit ===")
    _, id2sym_1to1 = load_gtf_1to1()
    mc = pd.read_csv(DS / "ReHeaT2_zenodo_15261569/mol_consistency.csv")
    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)].copy()
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(float)

    pos_r = mol_valid[mol_valid["mol_score"] > 0].sort_values("mol_score", ascending=False).head(100)
    neg_r = mol_valid[mol_valid["mol_score"] < 0].sort_values("mol_score", ascending=True).head(100)
    k200_df = pd.concat([pos_r, neg_r], ignore_index=True)
    k200_syms = list(k200_df["gene"].astype(str))
    k200_w = k200_df["mol_score"].to_numpy(float)

    disc, locked_specs = build_locked_discovery_sources(id2sym_1to1, sig_syms, sig_w, k200_syms, k200_w)
    Z_disc = {k: sa.compute_gsva_logodds(v["X"], kcdf="Gaussian") for k, v in disc.items()}

    def score_method(m, X, idx, w, Z=None, allow_nan=False):
        if m == "SS-Rank-Internal":
            return sa.score_ss_rank_internal(X, idx, w, allow_missing_per_sample=allow_nan)
        if m == "singscore-Internal":
            return sa.score_singscore_internal(X, idx, w, allow_missing_per_sample=allow_nan)
        if m == "SS-Rank":
            return sa.score_ss_rank_full(X, idx, w, allow_missing_per_sample=allow_nan)
        if m == "singscore":
            return sa.score_singscore_full(X, idx, w)
        if m == "ssGSEA":
            return sa.score_ssgsea_bidirectional(X, idx, w)
        if m == "GSVA_Gaussian":
            return score_gsva_from_Z(Z, idx, w)
        if m == "Cohort-Z":
            return score_cohort_z(X, idx, w)[0]
        raise ValueError(m)

    all7 = ["SS-Rank-Internal", "singscore-Internal", "SS-Rank", "singscore", "ssGSEA", "GSVA_Gaussian", "Cohort-Z"]
    frozen_thrs = {}
    for spec_i, (sig_name, s_g, s_w, src_name) in enumerate(locked_specs):
        idx_s = resolve_strict_symbols(disc[src_name]["syms"], s_g)
        for m in all7:
            sc_s = score_method(m, disc[src_name]["X"], idx_s, s_w, Z=Z_disc[src_name])
            frozen_thrs[(spec_i, m)] = youden_threshold(disc[src_name]["y"], sc_s)

    # =========================================================================
    # PART 1: GSE263297 SYDNEY BLANK-CELL & DUPLICATE-SYMBOL SENSITIVITY AUDIT (Audit P1-2)
    # =========================================================================
    print("\n[Part 1] GSE263297 Sydney Blank-Cell Diagnostics & 4-Regime Sensitivity Audit...")
    syd = load_gse263297_raw_nan()
    X_nan = syd["mat_nan"]
    y_syd = syd["y"]
    syms_syd = syd["syms"]
    conds_syd = np.array(syd["conds"])
    sym_counts = Counter(syms_syd)
    dup_syms = {k: v for k, v in sym_counts.items() if v > 1}

    idx_rh_syd = resolve_strict_symbols(syms_syd, sig_syms)
    rh_cols = [c for c in idx_rh_syd if c is not None]
    finite_vals = X_nan[np.isfinite(X_nan)]
    diag = {
        "n_samples": int(X_nan.shape[0]),
        "n_gene_rows": int(X_nan.shape[1]),
        "total_cells": int(X_nan.size),
        "blank_cells": int(np.isnan(X_nan).sum()),
        "blank_fraction_overall": float(np.isnan(X_nan).mean()),
        "blank_fraction_donor": float(np.isnan(X_nan[y_syd == 0]).mean()),
        "blank_fraction_icm_all": float(np.isnan(X_nan[y_syd == 1]).mean()),
        "blank_fraction_reheat2_mapped": float(np.isnan(X_nan[:, rh_cols]).mean()),
        "reheat2_mapped_genes": int(len(rh_cols)),
        "reheat2_complete_case_genes": int(np.sum(~np.any(np.isnan(X_nan[:, rh_cols]), axis=0))),
        "n_duplicate_symbols": int(len(dup_syms)),
        "duplicate_symbol_examples": dict(list(dup_syms.items())[:10]),
        "finite_min": float(finite_vals.min()),
        "finite_p01": float(np.percentile(finite_vals, 1)),
        "finite_median": float(np.median(finite_vals)),
        "finite_max": float(finite_vals.max()),
        "finite_negative_count": int(np.sum(finite_vals < 0.0)),
    }
    (OUT_DIR / "v7_1_sydney_blank_diagnostics.json").write_text(json.dumps(diag, indent=2), encoding="utf-8")
    print("  Sydney diagnostics:", json.dumps({k: v for k, v in diag.items() if k != "duplicate_symbol_examples"}))

    # Build 4 explicit regimes:
    # R1: Zero-fill primary (blank -> 0.0)
    X_r1 = np.nan_to_num(X_nan, nan=0.0)
    Z_r1 = sa.compute_gsva_logodds(X_r1, kcdf="Gaussian")
    # R2: Per-sample NaN exclusion (no zero-imputation; for SS-Rank-Internal, singscore-Internal, SS-Rank)
    # R3: Complete-case genes only (0% missing across all 21 samples; tested across ALL 7 methods!)
    cc_mask = ~np.any(np.isnan(X_nan), axis=0)
    X_r3 = X_nan[:, cc_mask]
    syms_r3 = [s for s, k in zip(syms_syd, cc_mask) if k]
    Z_r3 = sa.compute_gsva_logodds(X_r3, kcdf="Gaussian")
    # R4: Deduplicate symbols by nanmean + retain genes with <=20% missingness (zero-fill remaining vs NaN-excluded)
    df_syd = pd.DataFrame(X_nan.T, index=syms_syd)
    df_syd_dedup = df_syd.groupby(level=0, sort=False).mean()
    keep_20 = df_syd_dedup.isna().mean(axis=1) <= 0.20
    df_r4 = df_syd_dedup.loc[keep_20]
    syms_r4 = list(df_r4.index)
    X_r4_zero = df_r4.fillna(0.0).to_numpy(float).T
    X_r4_nan = df_r4.to_numpy(float).T
    Z_r4 = sa.compute_gsva_logodds(X_r4_zero, kcdf="Gaussian")

    syd_rows = []
    for regime_name, X_use, syms_use, Z_use, methods_use, allow_nan in [
        ("R1_ZeroFill_Primary", X_r1, syms_syd, Z_r1, all7, False),
        ("R2_PerSampleNaN_NoZeroFill", X_nan, syms_syd, None, ["SS-Rank-Internal", "singscore-Internal", "SS-Rank"], True),
        ("R3_CompleteCaseGenes_0pctNA", X_r3, syms_r3, Z_r3, all7, False),
        ("R4a_DedupMean_LE20pctNA_ZeroFill", X_r4_zero, syms_r4, Z_r4, all7, False),
        ("R4b_DedupMean_LE20pctNA_NaNExcl", X_r4_nan, syms_r4, None, ["SS-Rank-Internal", "singscore-Internal", "SS-Rank"], True),
    ]:
        for spec_i, (sig_name, s_g, s_w, src_name) in enumerate(locked_specs):
            idx_h = resolve_strict_symbols(syms_use, s_g)
            for m in methods_use:
                sc = score_method(m, X_use, idx_h, s_w, Z=Z_use, allow_nan=allow_nan)
                ev = eval_threshold(y_syd, sc, frozen_thrs[(spec_i, m)])
                syd_rows.append({
                    "regime": regime_name, "signature": sig_name, "calib_source": src_name,
                    "method": m, "n_mapped": sum(x is not None for x in idx_h),
                    "auc": float(roc_auc_score(y_syd, sc)), **ev,
                })
    df_syd_sens = pd.DataFrame(syd_rows)
    df_syd_sens.to_csv(OUT_DIR / "v7_1_table_sydney_missingness_sensitivity.csv", index=False)
    print("\n  Sydney 4-Regime Sensitivity Summary (mean across 9 locked models):")
    print(df_syd_sens.groupby(["regime", "method"])[["auc", "bal_acc", "mcc"]].mean().round(3).to_string())

    # =========================================================================
    # PART 2: GSE198945 ITALY F2_C14 / F2_C14_lib2 SENSITIVITY AUDIT (Audit P1-1)
    # =========================================================================
    print("\n[Part 2] GSE198945 Italy F2_C14 / F2_C14_lib2 Duplicate-Library Sensitivity Audit...")
    h198 = load_heldout_gse198945()
    titles_198 = h198["titles"]
    i_c14 = titles_198.index("F2_C14")
    i_lib2 = titles_198.index("F2_C14_lib2")
    r_c14_pair = float(np.corrcoef(h198["mat"][i_c14], h198["mat"][i_lib2])[0, 1])
    ctrl_idx = np.where(h198["y"] == 0)[0]
    other_corrs = [
        float(np.corrcoef(h198["mat"][i], h198["mat"][j])[0, 1])
        for a, i in enumerate(ctrl_idx)
        for j in ctrl_idx[a + 1 :]
        if not ({i, j} == {i_c14, i_lib2})
    ]
    print(f"  GSE198945 F2_C14 vs F2_C14_lib2 log2-CPM Pearson r = {r_c14_pair:.4f} (other Ctrl-Ctrl median r = {np.median(other_corrs):.4f}, max = {np.max(other_corrs):.4f})")

    # 3 Regimes: (1) All 40 profiles, (2) Drop F2_C14_lib2 (39), (3) Sum raw counts of F2_C14 + F2_C14_lib2 -> log2_cpm (39)
    keep_drop = [i for i in range(len(titles_198)) if i != i_lib2]
    X_it_40, y_it_40 = h198["mat"], h198["y"]
    X_it_drop, y_it_drop = h198["mat"][keep_drop], h198["y"][keep_drop]
    raw_merge = h198["raw"].copy()
    raw_merge[i_c14] = raw_merge[i_c14] + raw_merge[i_lib2]
    X_it_merge = log2_cpm(raw_merge[keep_drop])
    y_it_merge = y_it_drop

    it_rows = []
    for rname, Xi, yi in [
        ("All40_Profiles", X_it_40, y_it_40),
        ("Drop_F2_C14_lib2_39", X_it_drop, y_it_drop),
        ("MergeSum_F2_C14_39", X_it_merge, y_it_merge),
    ]:
        Zi = sa.compute_gsva_logodds(Xi, kcdf="Gaussian")
        for spec_i, (sig_name, s_g, s_w, src_name) in enumerate(locked_specs):
            idx_h = resolve_strict_symbols(h198["syms"], s_g)
            for m in all7:
                sc = score_method(m, Xi, idx_h, s_w, Z=Zi)
                ev = eval_threshold(yi, sc, frozen_thrs[(spec_i, m)])
                it_rows.append({
                    "regime": rname, "signature": sig_name, "calib_source": src_name,
                    "method": m, "auc": float(roc_auc_score(yi, sc)), **ev,
                    "c14_pair_pearson_r": r_c14_pair,
                })
    df_it_sens = pd.DataFrame(it_rows)
    df_it_sens.to_csv(OUT_DIR / "v7_1_table_italy_lib2_sensitivity.csv", index=False)
    print("\n  Italy 3-Regime Sensitivity Summary (mean across 9 locked models):")
    print(df_it_sens.groupby(["regime", "method"])[["auc", "bal_acc", "mcc"]].mean().round(4).to_string())

    # =========================================================================
    # PART 3: TRUE EXTERNAL DISCOVERY-FROZEN CPM THRESHOLD (GSE141910) -> MAYO GSE249925 CPM vs TPM (Audit P1-3)
    # =========================================================================
    print("\n[Part 3] True External Discovery-Frozen Threshold (GSE141910 CPM) -> Mayo GSE249925 CPM vs TPM...")
    h249 = load_heldout_gse249925(id2sym_1to1)
    ind5 = ["SS-Rank-Internal", "singscore-Internal", "SS-Rank", "singscore", "ssGSEA"]
    ext_tpm_rows = []
    for sname, sg, sw in [("ReHeaT2_Full", sig_syms, sig_w), ("ReHeaT2_K200", k200_syms, k200_w)]:
        idx_src = resolve_strict_symbols(disc["GSE141910_CPM"]["syms"], sg)
        idx_mayo = resolve_strict_symbols(h249["syms"], sg)
        for m in ind5:
            thr_ext = youden_threshold(disc["GSE141910_CPM"]["y"], score_method(m, disc["GSE141910_CPM"]["X"], idx_src, sw))
            sc_cpm = score_method(m, h249["mat"], idx_mayo, sw)
            sc_tpm = score_method(m, h249["mat_tpm"], idx_mayo, sw)
            ev_cpm = eval_threshold(h249["y"], sc_cpm, thr_ext)
            ev_tpm = eval_threshold(h249["y"], sc_tpm, thr_ext)
            ext_tpm_rows.append({
                "signature": sname, "calib_source": "GSE141910_CPM_dedup", "method": m,
                "mayo_cpm_ext_bal_acc": ev_cpm["bal_acc"], "mayo_cpm_ext_mcc": ev_cpm["mcc"],
                "mayo_tpm_ext_bal_acc": ev_tpm["bal_acc"], "mayo_tpm_ext_mcc": ev_tpm["mcc"],
                "mayo_tpm_auc": float(roc_auc_score(h249["y"], sc_tpm)),
                "delta_tpm_minus_cpm_bal_acc": ev_tpm["bal_acc"] - ev_cpm["bal_acc"],
            })
            print(f"  Ext GSE141910->Mayo {sname:12s} | {m:18s}: CPM BalAcc={ev_cpm['bal_acc']:.3f} -> TPM BalAcc={ev_tpm['bal_acc']:.3f} (MCC={ev_tpm['mcc']:.3f}, AUC={roc_auc_score(h249['y'], sc_tpm):.4f})")
    pd.DataFrame(ext_tpm_rows).to_csv(OUT_DIR / "v7_1_table_mayo_external_cpm_vs_tpm.csv", index=False)

    # =========================================================================
    # PART 4: OFFICIAL gseapy (v1.3.2) ssGSEA & RIGHT-CONTINUOUS ECDF COMPARATOR AUDIT (Audit P1-4)
    # =========================================================================
    print("\n[Part 4] Official gseapy (v1.3.2) ssGSEA & Right-Continuous ECDF Comparator Audit...")
    cp = load_cardiopedia_strict()
    log_cpm_235 = log2_cpm(cp["counts_235"])
    syms_cp = cp["gene_syms"]
    idx_cp = resolve_strict_symbols(syms_cp, sig_syms)
    force_24_trt = cp["endpoints_24h"]["Force"].mean(axis=0)

    comp_rows = []
    try:
        import gseapy as gp
        gp_ver = getattr(gp, "__version__", "unknown")
        print(f"  Running official gseapy {gp_ver} ssGSEA on GSE314310 (235 org.) & GSE198945 (40 prof.)...")
        up_set = [g for g, w in zip(sig_syms, sig_w) if w > 0]
        dn_set = [g for g, w in zip(sig_syms, sig_w) if w < 0]

        for dlabel, Xd, syms_d, y_or_force in [
            ("GSE314310_3D_hCO", log_cpm_235, syms_cp, force_24_trt),
            ("GSE198945_Italy_ICM", h198["mat"], h198["syms"], h198["y"]),
        ]:
            seen = {}
            for ci, s in enumerate(syms_d):
                if s and s not in seen:
                    seen[s] = ci
            u_syms = list(seen.keys())
            u_cols = [seen[s] for s in u_syms]
            expr_df = pd.DataFrame(Xd[:, u_cols].T, index=u_syms, columns=[f"S{i}" for i in range(Xd.shape[0])])
            gs_dict = {"UP": [g for g in up_set if g in seen], "DN": [g for g in dn_set if g in seen]}

            idx_u = resolve_strict_symbols(u_syms, sig_syms)
            sc_our = sa.score_ssgsea_bidirectional(Xd[:, u_cols], idx_u, sig_w, alpha=0.25)

            ss_res = gp.ssgsea(
                data=expr_df, gene_sets=gs_dict, outdir=None,
                sample_norm_method="rank", correl_norm_type="rank",
                weight=0.25, min_size=10, max_size=20000, no_plot=True, threads=4,
            )
            piv_es = ss_res.res2d.pivot(index="Name", columns="Term", values="ES").loc[expr_df.columns]
            piv_nes = ss_res.res2d.pivot(index="Name", columns="Term", values="NES").loc[expr_df.columns]
            sc_gp_es = (piv_es["UP"] - piv_es["DN"]).to_numpy(float)
            sc_gp_nes = (piv_nes["UP"] - piv_nes["DN"]).to_numpy(float)

            r_es = float(np.corrcoef(sc_our, sc_gp_es)[0, 1])
            rho_es = float(stats.spearmanr(sc_our, sc_gp_es).statistic)
            r_nes = float(np.corrcoef(sc_our, sc_gp_nes)[0, 1])
            rho_nes = float(stats.spearmanr(sc_our, sc_gp_nes).statistic)

            if dlabel == "GSE314310_3D_hCO":
                ep_our = float(stats.spearmanr(sc_our.reshape(5, 47).mean(axis=0), y_or_force).statistic)
                ep_es = float(stats.spearmanr(sc_gp_es.reshape(5, 47).mean(axis=0), y_or_force).statistic)
                ep_nes = float(stats.spearmanr(sc_gp_nes.reshape(5, 47).mean(axis=0), y_or_force).statistic)
                ep_name = "3D_24h_Force_Spearman_rho"
            else:
                ep_our = float(roc_auc_score(y_or_force, sc_our))
                ep_es = float(roc_auc_score(y_or_force, sc_gp_es))
                ep_nes = float(roc_auc_score(y_or_force, sc_gp_nes))
                ep_name = "Italy_AUROC"

            comp_rows.append({
                "dataset": dlabel, "comparator": f"gseapy_{gp_ver}_ssgsea_ES_vs_tieavg",
                "endpoint_metric": ep_name, "our_value": ep_our, "std_value": ep_es,
                "per_sample_pearson_r": r_es, "per_sample_spearman_rho": rho_es,
            })
            comp_rows.append({
                "dataset": dlabel, "comparator": f"gseapy_{gp_ver}_ssgsea_NES_vs_tieavg",
                "endpoint_metric": ep_name, "our_value": ep_our, "std_value": ep_nes,
                "per_sample_pearson_r": r_nes, "per_sample_spearman_rho": rho_nes,
            })
            print(f"  {dlabel}: our={ep_our:.4f} | gseapy ES={ep_es:.4f} (r={r_es:.6f}, rho={rho_es:.6f}) | gseapy NES={ep_nes:.4f} (r={r_nes:.6f})")
    except Exception as e:
        print(f"  [WARN] gseapy comparison skipped: {e}")

    # Compare GSVA kcdf='midrank' vs right-continuous 'ecdf' on 3D Cardiopedia AND 2D ScreenSeq (GSE244740)
    sc_gsva_g = sa.score_gsva_bidirectional(log_cpm_235, idx_cp, sig_w, kcdf="Gaussian")
    sc_gsva_m = sa.score_gsva_bidirectional(log_cpm_235, idx_cp, sig_w, kcdf="midrank")
    sc_gsva_e = sa.score_gsva_bidirectional(log_cpm_235, idx_cp, sig_w, kcdf="ecdf")
    for lbl, sv in [("GSVA_midrank_vs_Gaussian_3D", sc_gsva_m), ("GSVA_rightECDF_vs_Gaussian_3D", sc_gsva_e)]:
        comp_rows.append({
            "dataset": "GSE314310_3D_hCO", "comparator": lbl,
            "endpoint_metric": "3D_24h_Force_Spearman_rho",
            "our_value": float(stats.spearmanr(sv.reshape(5, 47).mean(axis=0), force_24_trt).statistic),
            "std_value": float(stats.spearmanr(sc_gsva_g.reshape(5, 47).mean(axis=0), force_24_trt).statistic),
            "per_sample_pearson_r": float(np.corrcoef(sv, sc_gsva_g)[0, 1]),
            "per_sample_spearman_rho": float(stats.spearmanr(sv, sc_gsva_g).statistic),
        })

    # 2D ScreenSeq: GSVA midrank vs right-continuous ECDF + export exact SS-Rank-Internal drug table for Fig 4A!
    print("  Computing 2D ScreenSeq GSVA midrank vs right-continuous ECDF + Fig 4A SS-Rank-Internal export...")
    b6_dir = DS / "GSE244740"
    v5_b6_cond = pd.read_csv(V5_TABLES / "b6_all_program_condition_statuses.csv")
    v5_b6_ok = v5_b6_cond[(v5_b6_cond["program"] == "main") & (v5_b6_cond["status"] == "eligible")].copy()
    samples_df = pd.read_csv(b6_dir / "iemt_a_2273378_sm6218.txt", sep="\t")
    keep_mask = samples_df["Excluded"].astype(str).str.upper() == "FALSE"
    samples_kept = samples_df.loc[keep_mask].reset_index(drop=True)
    df_cnt_b6 = pd.read_csv(b6_dir / "GSE244740_processed_data_counts.txt.gz", sep="\t", index_col=0)
    lmat_b6 = log2_cpm(df_cnt_b6.iloc[:, np.where(keep_mask.to_numpy())[0]].to_numpy(float).T)
    syms_b6 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in df_cnt_b6.index]
    idx_b6 = resolve_strict_symbols(syms_b6, sig_syms)

    def norm_drug(s):
        return re.sub(r"[^A-Za-z0-9]+", "_", str(s).strip()).strip("_")

    samples_kept["drug_norm"] = samples_kept["Treatment"].map(norm_drug)
    samples_kept["rank_str"] = samples_kept["Concentration Rank"].astype(str).str.replace(r"[^0-9]", "", regex=True)
    cmap = defaultdict(list)
    for i, r in samples_kept.iterrows():
        cmap[(r["drug_norm"], r["rank_str"])].append(i)

    sc_b6_cz = score_cohort_z(lmat_b6, idx_b6, sig_w)[0]
    sc_b6_ssri = sa.score_ss_rank_internal(lmat_b6, idx_b6, sig_w)
    sc_b6_mid = sa.score_gsva_bidirectional(lmat_b6, idx_b6, sig_w, kcdf="midrank")
    sc_b6_ecdf = sa.score_gsva_bidirectional(lmat_b6, idx_b6, sig_w, kcdf="ecdf")

    amp_sub = v5_b6_ok[v5_b6_ok["assay"] == "Amplitude"].copy()
    d_rows = []
    for d in sorted(amp_sub["drug"].unique()):
        grp = amp_sub[amp_sub["drug"] == d].sort_values("concentration_rank")
        v_cz, v_ri, v_mid, v_ec, v_ep = [], [], [], [], []
        for _, r in grp.iterrows():
            idxs = cmap.get((d, str(r["concentration_rank"]).strip()), [])
            if idxs and pd.notna(r["assay_value"]):
                v_cz.append(float(np.mean(sc_b6_cz[idxs])))
                v_ri.append(float(np.mean(sc_b6_ssri[idxs])))
                v_mid.append(float(np.mean(sc_b6_mid[idxs])))
                v_ec.append(float(np.mean(sc_b6_ecdf[idxs])))
                v_ep.append(float(r["assay_value"]))
        if len(v_ep) >= 3:
            d_rows.append({
                "drug": d, "assay_mean": float(np.mean(v_ep)),
                "score_cz_mean": float(np.mean(v_cz)),
                "score_ssri_mean": float(np.mean(v_ri)),
                "score_gsva_midrank_mean": float(np.mean(v_mid)),
                "score_gsva_ecdf_mean": float(np.mean(v_ec)),
            })
    df_b6_38 = pd.DataFrame(d_rows).sort_values("drug").reset_index(drop=True)
    df_b6_38.to_csv(OUT_DIR / "v7_1_screenseq_38drugs_ssri.csv", index=False)
    rho_b6_mid = float(stats.spearmanr(df_b6_38["score_gsva_midrank_mean"], df_b6_38["assay_mean"]).statistic)
    rho_b6_ecdf = float(stats.spearmanr(df_b6_38["score_gsva_ecdf_mean"], df_b6_38["assay_mean"]).statistic)
    comp_rows.append({
        "dataset": "GSE244740_2D_ScreenSeq", "comparator": "GSVA_midrank_vs_rightECDF_2D",
        "endpoint_metric": "2D_Amplitude_38drug_rho",
        "our_value": rho_b6_mid, "std_value": rho_b6_ecdf,
        "per_sample_pearson_r": float(np.corrcoef(sc_b6_mid, sc_b6_ecdf)[0, 1]),
        "per_sample_spearman_rho": float(stats.spearmanr(sc_b6_mid, sc_b6_ecdf).statistic),
    })
    print(f"  2D ScreenSeq GSVA midrank rho={rho_b6_mid:+.4f} vs right-continuous ECDF rho={rho_b6_ecdf:+.4f} (per-sample r={np.corrcoef(sc_b6_mid, sc_b6_ecdf)[0,1]:.6f})")
    pd.DataFrame(comp_rows).to_csv(OUT_DIR / "v7_1_table_comparator_gseapy_ecdf_audit.csv", index=False)

    # =========================================================================
    # PART 5: FULL-COHORT (ALL SAMPLES!) COLUMN-PERMUTATION INVARIANCE CHECK (Audit P2-1)
    # =========================================================================
    print("\n[Part 5] Full-Cohort (ALL 235 org. GSE314310 & ALL 40 prof. GSE198945) Column-Permutation Check...")
    rng_p = np.random.default_rng(20261121)
    full_perm = {}
    for dlabel, Xd, syms_d in [
        ("GSE314310_ALL_235_samples", log_cpm_235, syms_cp),
        ("GSE198945_ALL_40_samples", h198["mat"], h198["syms"]),
    ]:
        idx_d = resolve_strict_symbols(syms_d, sig_syms)
        G_d = Xd.shape[1]
        perm = rng_p.permutation(G_d)
        inv = np.argsort(perm)
        idx_p = [int(inv[c]) if c is not None else None for c in idx_d]
        Xd_p = Xd[:, perm]
        res_d = {}
        for mname, fn in [
            ("SS-Rank-Internal", lambda M, ix: sa.score_ss_rank_internal(M, ix, sig_w)),
            ("SS-Rank", lambda M, ix: sa.score_ss_rank_full(M, ix, sig_w)),
            ("singscore-Internal", lambda M, ix: sa.score_singscore_internal(M, ix, sig_w)),
            ("singscore", lambda M, ix: sa.score_singscore_full(M, ix, sig_w)),
            ("ssGSEA_tie_avg", lambda M, ix: sa.score_ssgsea_bidirectional(M, ix, sig_w)),
            ("GSVA_midrank_tie_avg", lambda M, ix: sa.score_gsva_bidirectional(M, ix, sig_w, kcdf="midrank")),
            ("GSVA_rightECDF_tie_avg", lambda M, ix: sa.score_gsva_bidirectional(M, ix, sig_w, kcdf="ecdf")),
        ]:
            diff = float(np.max(np.abs(fn(Xd, idx_d) - fn(Xd_p, idx_p))))
            res_d[mname] = diff
        # Also test Gaussian GSVA on ALL 40 samples of GSE198945 & ALL 235 of GSE314310
        diff_g = float(np.max(np.abs(
            sa.score_gsva_bidirectional(Xd, idx_d, sig_w, kcdf="Gaussian")
            - sa.score_gsva_bidirectional(Xd_p, idx_p, sig_w, kcdf="Gaussian")
        )))
        res_d["GSVA_Gaussian_tie_avg"] = diff_g
        full_perm[dlabel] = res_d
        print(f"  {dlabel} max |diff| across ALL samples:", res_d)

    # Verify ReHeaT2 official MOESM3 absence programmatically
    moesm3_path = DATA_DIR / "reference/ReHeaT_study_metadata_MOESM3.csv"
    moesm3_txt = moesm3_path.read_text(encoding="utf-8", errors="replace")
    moesm3_gses = sorted(set(re.findall(r"GSE\d+", moesm3_txt)))
    out_inv = {
        "full_cohort_column_permutation_max_abs_diff": full_perm,
        "reheat2_moesm3_included_gses": moesm3_gses,
        "heldout_absent_from_reheat2_moesm3": {
            g: (g not in moesm3_gses) for g in ["GSE198945", "GSE263297", "GSE249925"]
        },
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    (OUT_DIR / "v7_1_full_cohort_permutation_invariance.json").write_text(json.dumps(out_inv, indent=2), encoding="utf-8")
    print(f"\n=== V7.1 Audit Completed Cleanly in {out_inv['runtime_seconds']} s ===")


if __name__ == "__main__":
    main()
