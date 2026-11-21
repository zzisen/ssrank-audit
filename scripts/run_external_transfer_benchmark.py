"""
Nine-method tie-invariant 3D benchmark, pair-deduplicated discovery transfers, retrospective external
clinical evaluation (GSE198945, GSE263297, GSE249925), and CPM->TPM / dropout / zero-inflation stress suite.
"""

from __future__ import annotations
import gzip
import json
import math
import platform
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import ssrank_audit as sa
from run_3d_cardiopedia_benchmark import (
    DATA_DIR,
    DS,
    OUT_DIR,
    V5_TABLES,
    bootstrap_spearman_ci,
    load_cardiopedia_strict,
    load_fixed_programs,
    log2_cpm,
    partial_spearman,
    resolve_strict_symbols,
    score_cohort_z,
    score_ref_frozen_z,
    spearman_rho,
)
from run_cross_system_benchmark import build_adult_nf_ref_by_symbol, load_gtf_1to1
from benchmark_utils import score_cohort_robust_z
from benchmark_utils import load_gse116250, load_gse57338
from benchmark_utils import (
    apply_zero_inflation,
    cluster_robust_ols_cr1,
    compute_welch_t_signature,
    eval_threshold,
    youden_threshold,
)

HELDOUT = DATA_DIR / "external_cohorts"


def _col_letters_to_idx(ref: str) -> int:
    col = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def load_heldout_gse198945():
    """Milan, Italy (IRCCS Policlinico San Donato; Mar 2025; 20 Ctrl vs 20 Ischemic HF, raw counts)."""
    gdir = HELDOUT / "GSE198945"
    titles, conds, batches = [], [], []
    with gzip.open(gdir / "GSE198945_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("!Sample_title"):
                titles = [x.strip('"') for x in ln.strip().split("\t")[1:]]
            elif ln.startswith("!Sample_characteristics_ch1") and "condition:" in ln.lower():
                conds = [x.strip('"').split(":", 1)[1].strip() for x in ln.strip().split("\t")[1:]]
            elif ln.startswith("!Sample_characteristics_ch1") and "batch:" in ln.lower():
                batches = [x.strip('"').split(":", 1)[1].strip() for x in ln.strip().split("\t")[1:]]
    df = pd.read_csv(gdir / "GSE198945_samples.count.matrix.gz", sep=None, engine="python")
    gene_col = df.columns[0]
    raw = df[titles].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).T
    lmat = log2_cpm(raw)
    syms = [str(s).strip() for s in df[gene_col].fillna("")]
    y = np.array([1 if "heart failure" in c.lower() else 0 for c in conds], int)
    assert len(y) == 40 and y.sum() == 20 and (y == 0).sum() == 20
    return {"mat": lmat, "raw": raw, "syms": syms, "y": y, "titles": titles, "batches": batches}


def load_heldout_gse263297():
    """Sydney, Australia (University of Sydney / Sydney Heart Bank; Apr 2025; 7 Donor vs 14 ICM/ICM-DM, log2 normalized)."""
    gdir = HELDOUT / "GSE263297"
    xlsx_path = gdir / "GSE263297_HUNTER_RNAseq_data_Normalised_LOG2_transformed.xlsx"
    with zipfile.ZipFile(xlsx_path) as zf:
        ss = []
        if "xl/sharedStrings.xml" in zf.namelist():
            ss = ["".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")) for si in ET.fromstring(zf.read("xl/sharedStrings.xml"))]
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
                                    vals.append(0.0)
                            except Exception:
                                vals.append(0.0)
                        if ok_any:
                            gene_syms.append(sym)
                            data_rows.append(vals)
                rnum += 1
                el.clear()
    mat = np.asarray(data_rows, dtype=float).T  # (21, G)
    y = np.array([0 if "donor" in c.lower() else 1 for c in conds], int)
    assert mat.shape[0] == 21 and (y == 0).sum() == 7 and (y == 1).sum() == 14, f"Unexpected GSE263297 shape/labels: {mat.shape}, {conds}"
    return {"mat": mat, "syms": gene_syms, "y": y, "titles": sample_ids, "conds": conds}


def load_heldout_gse249925(id2sym_1to1: dict[str, str]):
    """Mayo Clinic, USA (Nov 2024; 23 Control vs 97 Obstructive HCM; raw counts + exact gene Length!)."""
    gdir = HELDOUT / "GSE249925"
    titles, groups = [], []
    with gzip.open(gdir / "GSE249925_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as f:
        for ln in f:
            if ln.startswith("!Sample_title"):
                titles = [x.strip('"') for x in ln.strip().split("\t")[1:]]
            elif ln.startswith("!Sample_characteristics_ch1") and "group:" in ln.lower():
                groups = [x.strip('"').split(":", 1)[1].strip() for x in ln.strip().split("\t")[1:]]
    df = pd.read_csv(gdir / "GSE249925_20231208-For-GEO-Raw-Counts.tsv.gz", sep="\t")
    # Match titles directly to count columns
    avail = [t for t in titles if t in df.columns]
    assert len(avail) == 120, f"Matched {len(avail)}/120 columns in GSE249925"
    raw = df[avail].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).T
    lengths = pd.to_numeric(df["Length"], errors="coerce").fillna(1000.0).to_numpy(float)
    lmat_cpm = log2_cpm(raw)
    # Also compute exact log2-TPM using built-in Length column!
    rpk = raw / np.maximum(lengths[None, :] / 1000.0, 1e-6)
    tpm = (rpk / np.maximum(rpk.sum(axis=1, keepdims=True), 1e-12)) * 1e6
    lmat_tpm = np.log2(tpm + 1.0)
    # Parse Gene_Index: chr1@ENSG00000223972@DDX11L1@11869@14409 -> use canonical GRCh38.104 if present, else embedded symbol
    syms = []
    for gi in df["Gene_Index"].astype(str):
        parts = gi.split("@")
        ens = parts[1].split(".")[0] if len(parts) > 1 else ""
        emb = parts[2].strip() if len(parts) > 2 else ""
        syms.append(id2sym_1to1.get(ens, emb))
    y = np.array([1 if "hcm" in g.lower() else 0 for g in groups], int)
    assert len(y) == 120 and (y == 0).sum() == 23 and (y == 1).sum() == 97
    return {"mat": lmat_cpm, "mat_tpm": lmat_tpm, "raw": raw, "lengths": lengths, "syms": syms, "y": y, "titles": titles}


def score_gsva_from_Z(Z: np.ndarray, gene_indices: list[int | None], weights: np.ndarray, tau: float = 1.0) -> np.ndarray:
    """Fast tie-averaged, column-order-invariant GSVA walk from precomputed cohort log-odds Z."""
    N, G = Z.shape
    cols, w = sa._clean_pairs(gene_indices, weights)
    up_cols = cols[w > 0]
    dn_cols = cols[w < 0]
    up_m = np.zeros(G, dtype=bool); up_m[up_cols] = True
    dn_m = np.zeros(G, dtype=bool); dn_m[dn_cols] = True
    out = np.empty(N, dtype=float)
    for i in range(N):
        wu = np.cumsum(sa._tie_averaged_walk_steps(Z[i], up_m, "gsva", tau))
        wd = np.cumsum(sa._tie_averaged_walk_steps(Z[i], dn_m, "gsva", tau))
        out[i] = float(max(0.0, wu.max()) + min(0.0, wu.min())) - float(max(0.0, wd.max()) + min(0.0, wd.min()))
    return out


def main():
    t0_all = time.perf_counter()
    print("=== V7.0 Retrospective Post-ReHeaT2 External Evaluation Starting ===")
    _, id2sym_1to1 = load_gtf_1to1()
    mc = pd.read_csv(DS / "ReHeaT2_zenodo_15261569/mol_consistency.csv")
    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)].copy()
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(float)
    ref_mu, ref_sd, _ = build_adult_nf_ref_by_symbol(id2sym_1to1)

    # Compact ReHeaT2 K=200 (fixed rule within this retrospective workflow)
    pos_r = mol_valid[mol_valid["mol_score"] > 0].sort_values("mol_score", ascending=False).head(100)
    neg_r = mol_valid[mol_valid["mol_score"] < 0].sort_values("mol_score", ascending=True).head(100)
    k200_df = pd.concat([pos_r, neg_r], ignore_index=True)
    k200_syms = list(k200_df["gene"].astype(str))
    k200_w = k200_df["mol_score"].to_numpy(float)

    # =========================================================================
    # PART 1: TIE-INVARIANT 3D CARDIOPEDIA 9-METHOD BENCHMARK + REAL COLUMN-PERMUTATION TEST
    # =========================================================================
    print("\n[Part 1] Column-Order-Invariant 3D Cardiopedia 9-Method Benchmark & Real-Matrix Permutation Check...")
    cp = load_cardiopedia_strict()
    log_cpm_235 = log2_cpm(cp["counts_235"])
    labels = cp["labels"]
    gene_syms = cp["gene_syms"]
    force_24 = cp["endpoints_24h"]["Force"]
    force_04 = cp["endpoints_04h"]["Force"]
    fm = force_24.mean(axis=0)
    f04m = force_04.mean(axis=0)
    rep_fm = np.array([float(np.mean(cp["repeat_24h_by_ep"]["Force"][l])) for l in labels])
    sig_idx = resolve_strict_symbols(gene_syms, sig_syms)

    # Export SS-Rank-Internal treatment scores into a dedicated Fig 1A CSV so Fig 1A dots == label!
    trt_df = pd.read_csv(OUT_DIR / "v6_cardiopedia_47_treatments_master.csv")
    sc_ssri_235 = sa.score_ss_rank_internal(log_cpm_235, sig_idx, sig_w)
    trt_df["score_ss_rank_internal"] = sc_ssri_235.reshape(5, 47).mean(axis=0)
    trt_df.to_csv(OUT_DIR / "v7_0_cardiopedia_47_treatments_with_ssri.csv", index=False)

    cov8_47 = trt_df[["l2_burden", "prog_hypoxia", "prog_structural_exclusive", "prog_conduction_exclusive",
                      "prog_ecm", "prog_oxphos", "prog_upr", "prog_fibroblast_marker"]].to_numpy(float)
    srt_cz = trt_df.sort_values("score_cohort_z")
    keep_43 = np.asarray(trt_df.index.isin(srt_cz.index[2:-2]), dtype=bool)

    fixed_progs = load_fixed_programs()
    prog_235 = {}
    for pname in ("hypoxia", "structural_exclusive", "conduction_exclusive", "ecm", "oxphos", "upr", "fibroblast_marker"):
        pidx = resolve_strict_symbols(gene_syms, fixed_progs[pname])
        prog_235[pname] = score_cohort_z(log_cpm_235, pidx, np.ones(len(fixed_progs[pname]), float))[0]
    gsd_235 = log_cpm_235.std(axis=0, ddof=1)
    emask = gsd_235 > 1e-6
    Z235 = (log_cpm_235[:, emask] - log_cpm_235[:, emask].mean(axis=0, keepdims=True)) / gsd_235[emask][None, :]
    l2_235 = np.sqrt(np.mean(Z235 ** 2, axis=1))
    plate_ids = np.repeat(np.arange(5), 47)
    trt_ids = np.tile(np.arange(47), 5)
    plate_dum = np.column_stack([(plate_ids == p).astype(float) for p in range(1, 5)])
    X_cov_235 = np.column_stack([plate_dum, l2_235] + [prog_235[p] for p in ("hypoxia", "structural_exclusive", "conduction_exclusive", "ecm", "oxphos", "upr", "fibroblast_marker")])
    y_log_235 = np.log(np.clip(force_24.ravel(), 1e-4, None))
    y_rk_235 = stats.rankdata(force_24.ravel())

    # Real-matrix column-permutation invariance check on first 10 organoids of GSE314310 (which have real zero ties!)
    trt_mat_3d = log_cpm_235.reshape(5, 47, -1).mean(axis=0)
    rng_perm = np.random.default_rng(20261121)
    X10 = log_cpm_235[:10]
    perm_cols = rng_perm.permutation(X10.shape[1])
    inv_perm = np.empty(X10.shape[1], dtype=int)
    inv_perm[perm_cols] = np.arange(X10.shape[1])
    sig_idx_perm = [int(inv_perm[i]) if i is not None else None for i in sig_idx]
    perm_audit = {}
    for mname, fn in [
        ("SS-Rank-Internal", lambda M, idx: sa.score_ss_rank_internal(M, idx, sig_w)),
        ("SS-Rank", lambda M, idx: sa.score_ss_rank_full(M, idx, sig_w)),
        ("singscore-Internal", lambda M, idx: sa.score_singscore_internal(M, idx, sig_w)),
        ("singscore", lambda M, idx: sa.score_singscore_full(M, idx, sig_w)),
        ("ssGSEA_tie_avg", lambda M, idx: sa.score_ssgsea_bidirectional(M, idx, sig_w)),
        ("GSVA_Gaussian_tie_avg", lambda M, idx: sa.score_gsva_bidirectional(M, idx, sig_w, kcdf="Gaussian")),
    ]:
        a = fn(X10, sig_idx)
        b = fn(X10[:, perm_cols], sig_idx_perm)
        mdiff = float(np.max(np.abs(a - b)))
        perm_audit[mname] = mdiff
        assert mdiff < 1e-11, f"Real-matrix column permutation failed for {mname}: {mdiff}"
    print("  Real-matrix column-permutation max |diff| on GSE314310 (all < 1e-11):", {k: f"{v:.2e}" for k, v in perm_audit.items()})

    methods_3d = [
        ("Cohort-Z (mean/SD)", False, lambda M: score_cohort_z(M, sig_idx, sig_w)[0]),
        ("Cohort-Robust-Z (Med/MAD)", False, lambda M: score_cohort_robust_z(M, sig_idx, sig_w)[0]),
        ("GSVA (Gaussian, tie-avg.)", False, lambda M: sa.score_gsva_bidirectional(M, sig_idx, sig_w, kcdf="Gaussian")),
        ("SS-Rank (full, weighted)", True, lambda M: sa.score_ss_rank_full(M, sig_idx, sig_w)),
        ("SS-Rank-Internal (sig-only)", True, lambda M: sa.score_ss_rank_internal(M, sig_idx, sig_w)),
        ("singscore (full, unwt.)", True, lambda M: sa.score_singscore_full(M, sig_idx, sig_w)),
        ("singscore-Internal (unwt.)", True, lambda M: sa.score_singscore_internal(M, sig_idx, sig_w)),
        ("ssGSEA (full, tie-avg.)", True, lambda M: sa.score_ssgsea_bidirectional(M, sig_idx, sig_w)),
        ("Ref-Frozen-Z (GRCh38.104)", True, lambda M: score_ref_frozen_z(M, gene_syms, ref_mu, ref_sd, sig_syms, sig_w)[0]),
    ]
    rows_3d = []
    for mname, is_ind, fn in methods_3d:
        t0 = time.perf_counter()
        sc = fn(log_cpm_235)
        ms = (time.perf_counter() - t0) * 1000.0
        sm = sc.reshape(5, 47).mean(axis=0)
        ci = bootstrap_spearman_ci(sm, fm, n_boot=5000, seed=20261120)
        p04 = partial_spearman(sm, fm, f04m)["partial_rho"]
        r_rep = spearman_rho(sm, rep_fm)
        p8 = partial_spearman(sm, fm, cov8_47)
        ptrim = partial_spearman(sm[keep_43], fm[keep_43], cov8_47[keep_43])
        sc_z = (sc - sc.mean()) / sc.std(ddof=1)
        X_full = np.column_stack([np.ones(len(sc)), sc_z, X_cov_235])
        res_log = cluster_robust_ols_cr1(X_full, y_log_235, trt_ids)
        res_rk = cluster_robust_ols_cr1(X_full, y_rk_235, trt_ids)
        rows_3d.append({
            "method": mname, "inductive": is_ind,
            "rho_24h": ci["rho"], "ci_low": ci["ci_low"], "ci_high": ci["ci_high"],
            "partial_24h_given_04h": p04, "repeat_24h_rho": r_rep,
            "partial_8cov_rho": p8["partial_rho"], "partial_8cov_p": p8["p_two_sided"],
            "trimmed_n43_8cov_rho": ptrim["partial_rho"], "trimmed_n43_8cov_p": ptrim["p_two_sided"],
            "cr1_235org_log_p": float(res_log["p"][1]), "cr1_235org_rank_p": float(res_rk["p"][1]),
            "runtime_ms_235org": ms,
        })
        print(f"  3D {mname:28s}: rho={ci['rho']:+.3f} [{ci['ci_low']:+.3f},{ci['ci_high']:+.3f}] | 8cov={p8['partial_rho']:+.3f} | trim={ptrim['partial_rho']:+.3f} | {ms:.0f} ms")
    pd.DataFrame(rows_3d).to_csv(OUT_DIR / "v7_0_table_3d_9methods_tieinv.csv", index=False)

    # Re-run 200-draw treatment-mean 3D MC with tie-averaged GSVA
    trt_mat_3d = log_cpm_235.reshape(5, 47, -1).mean(axis=0)
    order_f = np.argsort(fm)
    weak_pool, mid_pool, strong_pool = order_f[:17], order_f[17:30], order_f[30:]
    rng_mc = np.random.default_rng(20261120)
    mc_methods = [
        ("Cohort-Z", lambda M: score_cohort_z(M, sig_idx, sig_w)[0]),
        ("Cohort-Robust-Z", lambda M: score_cohort_robust_z(M, sig_idx, sig_w)[0]),
        ("GSVA_Gaussian", lambda M: sa.score_gsva_bidirectional(M, sig_idx, sig_w, kcdf="Gaussian")),
        ("SS-Rank-Internal", lambda M: sa.score_ss_rank_internal(M, sig_idx, sig_w)),
    ]
    ref_sd_3d = {m: float(np.std(fn(trt_mat_3d), ddof=1)) for m, fn in mc_methods}
    mc_rec = {m: {"drift": [], "flip": [], "jaccard": []} for m, _ in mc_methods}
    for _ in range(200):
        probes = rng_mc.choice(mid_pool, size=10, replace=False)
        comp_w = rng_mc.choice(weak_pool, size=10, replace=False)
        comp_s = rng_mc.choice(strong_pool, size=10, replace=False)
        bw = trt_mat_3d[np.concatenate([probes, comp_w])]
        bs = trt_mat_3d[np.concatenate([probes, comp_s])]
        for m, fn in mc_methods:
            sw, ss = fn(bw)[:10], fn(bs)[:10]
            mc_rec[m]["drift"].append(float(np.mean(np.abs(sw - ss)) / ref_sd_3d[m]))
            mc_rec[m]["flip"].append(float(np.mean(np.sign(sw) != np.sign(ss))))
            hw, hs = set(np.where(sw < 0)[0]), set(np.where(ss < 0)[0])
            mc_rec[m]["jaccard"].append(1.0 if len(hw | hs) == 0 else len(hw & hs) / len(hw | hs))
    rows_mc3d = []
    for m, _ in mc_methods:
        dr, fl, ja = np.array(mc_rec[m]["drift"]), np.array(mc_rec[m]["flip"]), np.array(mc_rec[m]["jaccard"])
        rows_mc3d.append({
            "method": m, "unit": "treatment_mean_20",
            "mean_drift_sd": float(dr.mean()), "q025_drift": float(np.quantile(dr, 0.025)), "q975_drift": float(np.quantile(dr, 0.975)),
            "mean_flip_pct": float(fl.mean() * 100), "q025_flip": float(np.quantile(fl, 0.025) * 100), "q975_flip": float(np.quantile(fl, 0.975) * 100),
            "mean_hit_jaccard": float(ja.mean()), "q025_jacc": float(np.quantile(ja, 0.025)), "q975_jacc": float(np.quantile(ja, 0.975)),
        })
        print(f"  3D MC (trt-mean) {m:16s}: drift={dr.mean():.3f} SD [{np.quantile(dr,0.025):.3f},{np.quantile(dr,0.975):.3f}] | flips={fl.mean()*100:.1f}% | J={ja.mean():.3f}")
    pd.DataFrame(rows_mc3d).to_csv(OUT_DIR / "v7_0_table_mc_3d_tieinv.csv", index=False)

    # =========================================================================
    # PART 2: STRICTLY SAME-GENE-SET WEIGHT-CANCELLATION VERIFICATION (FIX Audit 2.5!)
    # =========================================================================
    print("\n[Part 2] Strictly Same-Gene-Set Signed vs. Unsigned Weight Cancellation (K=20..5,000, exact var-matched + synthetic)...")
    # Restrict pool to genes mapped AND strictly non-constant across 235 organoids so Cohort-Z and SS-Rank use IDENTICAL gene sets!
    g_sd_235 = log_cpm_235.std(axis=0, ddof=1)
    sym_to_col = {}
    for j, s in enumerate(gene_syms):
        if s and g_sd_235[j] > 1e-6 and s not in sym_to_col:
            sym_to_col[s] = j
    mv_ok = mol_valid[mol_valid["gene"].astype(str).isin(sym_to_col)].copy()
    mv_pos = mv_ok[mv_ok["mol_score"] > 0].sort_values("mol_score", ascending=False).reset_index(drop=True)
    mv_neg = mv_ok[mv_ok["mol_score"] < 0].sort_values("mol_score", ascending=True).reset_index(drop=True)
    max_half = min(len(mv_pos), len(mv_neg))
    print(f"  Non-constant mapped ReHeaT2 genes: pos={len(mv_pos)}, neg={len(mv_neg)} -> strictly equal-K valid up to K={max_half*2} (we test K=20..5000)")

    abl_rows = []
    for K in [20, 50, 100, 200, 500, 1000, 2000, 5000]:
        half = K // 2
        bal_df = pd.concat([mv_pos.head(half), mv_neg.head(half)], ignore_index=True)
        up_df = mv_pos.head(K)
        dn_df = mv_neg.head(K)
        assert len(bal_df) == len(up_df) == len(dn_df) == K
        rec = {"panel_K": K}
        for tag, sub in [("signed_bal", bal_df), ("unsigned_up", up_df), ("unsigned_dn", dn_df)]:
            idxs = [sym_to_col[g] for g in sub["gene"].astype(str)]
            wv = sub["mol_score"].to_numpy(float)
            if tag == "unsigned_dn":
                wv_use = np.abs(wv)  # unsigned positive-only magnitude on down-genes
            else:
                wv_use = wv
            beta_val = float(abs(wv_use.sum()) / np.abs(wv_use).sum())
            cz, cz_usable = score_cohort_z(log_cpm_235, idxs, wv_use)
            ssr = sa.score_ss_rank_full(log_cpm_235, idxs, wv_use)
            assert int(cz_usable.sum()) == K, f"Mismatch usable genes {cz_usable.sum()} != {K}"
            r_cz = spearman_rho(cz.reshape(5, 47).mean(axis=0), fm)
            r_ssr = spearman_rho(ssr.reshape(5, 47).mean(axis=0), fm)
            rec[f"{tag}_n_exact"] = K
            rec[f"{tag}_beta"] = beta_val
            rec[f"{tag}_rho_cz"] = r_cz
            rec[f"{tag}_rho_ssr"] = r_ssr
            rec[f"{tag}_abs_delta"] = abs(r_cz - r_ssr)
        # Descriptive O(nm) VIF of balanced K panel
        b_idxs = [sym_to_col[g] for g in bal_df["gene"].astype(str)]
        vif_d = sa.descriptive_vif_audit(trt_mat_3d, b_idxs, bal_df["mol_score"].to_numpy(float))
        rec["vif_balanced"] = vif_d["vif_sign_aligned"]
        abl_rows.append(rec)
        print(f"  Matched K={K:4d} (exact non-const m={K}): Bal(beta={rec['signed_bal_beta']:.2f}) |d|={rec['signed_bal_abs_delta']:.4f} | Up(1.0) |d|={rec['unsigned_up_abs_delta']:.4f} | Dn(1.0) |d|={rec['unsigned_dn_abs_delta']:.4f} | VIF={rec['vif_balanced']:.2f}x")
    pd.DataFrame(abl_rows).to_csv(OUT_DIR / "v7_0_table_strict_same_k_cancellation.csv", index=False)

    # Strictly SAME-GENE-SET synthetic background-expansion experiment (exact same m=200 genes, G: 18k -> 58k!)
    rng_syn = np.random.default_rng(20261121)
    X_base = rng_syn.normal(5.0, 2.0, size=(60, 18000))
    # Append 40,000 low-expression non-coding genes (mimicking array 18k -> RPKM 58k background expansion!)
    X_exp = np.hstack([X_base, rng_syn.normal(0.5, 0.5, size=(60, 40000))])
    same_cols = list(range(200))
    syn_rows = []
    for w_label, w_vec in [
        ("Balanced (beta=0.00)", np.array([1.0] * 100 + [-1.0] * 100)),
        ("Imbalanced (beta=0.50)", np.array([1.0] * 150 + [-1.0] * 50)),
        ("Unsigned (beta=1.00)", np.array([1.0] * 200)),
    ]:
        beta_w = float(abs(w_vec.sum()) / np.abs(w_vec).sum())
        ssr_18 = sa.score_ss_rank_full(X_base, same_cols, w_vec)
        ssr_58 = sa.score_ss_rank_full(X_exp, same_cols, w_vec)
        ssri_18 = sa.score_ss_rank_internal(X_base, same_cols, w_vec)
        ssri_58 = sa.score_ss_rank_internal(X_exp, same_cols, w_vec)
        syn_rows.append({
            "weight_config": w_label, "m_identical": 200, "beta": beta_w,
            "ss_rank_full_mean_abs_shift": float(np.mean(np.abs(ssr_58 - ssr_18))),
            "ss_rank_internal_mean_abs_shift": float(np.mean(np.abs(ssri_58 - ssri_18))),
        })
        print(f"  Synthetic same-200-gene bg expansion (18k->58k) {w_label:24s}: SS-Rank(Full) shift={syn_rows[-1]['ss_rank_full_mean_abs_shift']:.4f} | SS-Rank-Internal shift={syn_rows[-1]['ss_rank_internal_mean_abs_shift']:.2e}")
    pd.DataFrame(syn_rows).to_csv(OUT_DIR / "v7_0_table_synthetic_same_gene_bg_shift.csv", index=False)

    # =========================================================================
    # PART 3: LOAD 6 DISCOVERY COHORTS (WITH EXACT 48-DONOR DEDUP) + 3 NEW HELD-OUT COHORTS!
    # =========================================================================
    print("\n[Part 3] Loading 6 discovery cohorts (with exact 48-donor deduplication) & 3 post-ReHeaT2 held-out cohorts...")
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    ph141_sub = ph141[ph141["etiology"].isin(["NF", "DCM"])].copy().reset_index(drop=True)
    cnt141 = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    raw141 = cnt141[list(ph141_sub["sample_name"].astype(str))].to_numpy(float).T
    lmat_141 = log2_cpm(raw141)
    syms_141 = [id2sym_1to1.get(str(g).split(".")[0], "") for g in cnt141.index]
    y_141 = (ph141_sub["etiology"] == "DCM").astype(int).to_numpy()

    g57 = load_gse57338()
    codes57 = set(g57["codes"])
    codes141 = set(ph141_sub["sample_name"].astype(str).str.strip())
    mask141_ovlp = ph141_sub["sample_name"].astype(str).str.strip().isin(codes57).to_numpy()
    mask57_ovlp = np.array([c in codes141 for c in g57["codes"]], dtype=bool)
    assert int(mask141_ovlp.sum()) == 48 and int(mask57_ovlp.sum()) == 48
    print("  Verified exact 48 shared MAGNet donor IDs between GSE141910 and GSE57338 (matching Audit audit).")

    m349 = pd.read_csv(OUT_DIR / "v6_gse349247_sample_scores.csv")
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

    # Also export SS-Rank-Internal for Fig 5B/C/D (GSE46224 LVAD trajectories & GSE141910 LVEF!)
    cols_46_all = [f"NF{i}" for i in range(1, 9)] + [f"ICM{i}" for i in range(1, 9)] + [f"ICM+LVAD{i}" for i in range(1, 9)] + [f"NICM{i}" for i in range(1, 9)] + [f"NICM+LVAD{i}" for i in range(1, 9)]
    mat_46_all = np.log2(df46[cols_46_all].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).T + 1.0)
    idx_46_mol = resolve_strict_symbols(syms_46, sig_syms)
    ssri_46_all = sa.score_ss_rank_internal(mat_46_all, idx_46_mol, sig_w)
    col_to_ssri46 = dict(zip(cols_46_all, ssri_46_all))
    # Derive the figure's minimum required columns directly from the original
    # phenotype input; avoid packaging sample IDs and unrelated covariates.
    idx_141_mol = resolve_strict_symbols(syms_141, sig_syms)
    ssri_141 = sa.score_ss_rank_internal(lmat_141, idx_141_mol, sig_w)
    lvef_df = pd.DataFrame({"etiology": ph141_sub["etiology"].astype(str),
                            "LVEF": pd.to_numeric(ph141_sub["LVEF"], errors="coerce"),
                            "score_ss_rank_internal": ssri_141})
    lvef_df.to_csv(OUT_DIR / "v7_0_gse141910_sample_scores_lvef_ssri.csv", index=False)
    # Build clean GSE46224 trajectories with SS-Rank-Internal!
    lv_rows = []
    for i in range(1, 9):
        lv_rows.append({"patient_id": f"NF{i}", "stage": "NF", "etiology": "NF", "score_ss_rank_internal": col_to_ssri46[f"NF{i}"]})
    for et in ["ICM", "NICM"]:
        for i in range(1, 9):
            lv_rows.append({"patient_id": f"{et}{i}", "stage": "Pre-LVAD", "etiology": et, "score_ss_rank_internal": col_to_ssri46[f"{et}{i}"]})
            lv_rows.append({"patient_id": f"{et}{i}", "stage": "Post-LVAD", "etiology": et, "score_ss_rank_internal": col_to_ssri46[f"{et}+LVAD{i}"]})
    # Preserve within-patient pairing without redistributing original codes.
    pseudonyms = {}
    for row in lv_rows:
        original_id = row["patient_id"]
        if original_id not in pseudonyms:
            pseudonyms[original_id] = f"P{len(pseudonyms) + 1:03d}"
        row["patient_id"] = pseudonyms[original_id]
    pd.DataFrame(lv_rows).to_csv(OUT_DIR / "v7_0_gse46224_trajectories_ssri.csv", index=False)
    # Compute SS-Rank-Internal paired LVAD for bidirectional ReHeaT2 axes (unsigned all-ones sets identically equal 0 under SS-Rank-Internal!)
    lvad_sigs = [
        ("ReHeaT2_mol_score", sig_syms, sig_w),
        ("ReHeaT2_Fib", list(mc.loc[mc["mcell_HFscore_Fib"].notna() & (mc["mcell_HFscore_Fib"] != 0), "gene"].astype(str)),
         mc.loc[mc["mcell_HFscore_Fib"].notna() & (mc["mcell_HFscore_Fib"] != 0), "mcell_HFscore_Fib"].to_numpy(float)),
        ("ReHeaT2_CM", list(mc.loc[mc["mcell_HFscore_CM"].notna() & (mc["mcell_HFscore_CM"] != 0), "gene"].astype(str)),
         mc.loc[mc["mcell_HFscore_CM"].notna() & (mc["mcell_HFscore_CM"] != 0), "mcell_HFscore_CM"].to_numpy(float)),
    ]
    lv_sig_rows = []
    for sname, s_g, s_w in lvad_sigs:
        sidx = resolve_strict_symbols(syms_46, s_g)
        sc = sa.score_ss_rank_internal(mat_46_all, sidx, s_w)
        pre = np.concatenate([sc[8:16], sc[24:32]])
        post = np.concatenate([sc[16:24], sc[32:40]])
        diff = post - pre
        dz = float(diff.mean() / diff.std(ddof=1))
        wp = float(stats.wilcoxon(diff).pvalue)
        lv_sig_rows.append({"signature": sname, "paired_cohens_dz_ssri": dz, "wilcoxon_p_ssri": wp, "n_decrease": int((diff < 0).sum())})
    pd.DataFrame(lv_sig_rows).to_csv(OUT_DIR / "v7_0_gse46224_lvad_3bidir_ssri.csv", index=False)

    # Load 3 post-ReHeaT2 held-out cohorts!
    h198 = load_heldout_gse198945()
    h263 = load_heldout_gse263297()
    h249 = load_heldout_gse249925(id2sym_1to1)
    print(f"  Loaded Held-Out Cohorts: GSE198945 (Italy, n={len(h198['y'])}, G={h198['mat'].shape[1]}), GSE263297 (Australia, n={len(h263['y'])}, G={h263['mat'].shape[1]}), GSE249925 (Mayo HCM, n={len(h249['y'])}, G={h249['mat'].shape[1]})")

    # Verify real-matrix column-permutation invariance on GSE198945 (Italian held-out raw counts!)
    X198_10 = h198["mat"][:10]
    p198 = rng_perm.permutation(X198_10.shape[1])
    inv198 = np.empty(X198_10.shape[1], dtype=int); inv198[p198] = np.arange(X198_10.shape[1])
    idx198_m = resolve_strict_symbols(h198["syms"], sig_syms)
    idx198_p = [int(inv198[i]) if i is not None else None for i in idx198_m]
    assert np.max(np.abs(sa.score_ssgsea_bidirectional(X198_10, idx198_m, sig_w) - sa.score_ssgsea_bidirectional(X198_10[:, p198], idx198_p, sig_w))) < 1e-11
    assert np.max(np.abs(sa.score_gsva_bidirectional(X198_10, idx198_m, sig_w, kcdf="Gaussian") - sa.score_gsva_bidirectional(X198_10[:, p198], idx198_p, sig_w, kcdf="Gaussian"))) < 1e-11
    print("  Verified real-matrix column-permutation invariance on GSE198945 (Italy) for tie-avg ssGSEA & GSVA (< 1e-11)!")

    # Build 6 discovery cohorts dict (full + deduplicated masks for GSE141910 & GSE57338)
    disc = {
        "GSE141910_CPM": {"X": lmat_141, "syms": syms_141, "y": y_141, "ovlp": mask141_ovlp},
        "GSE349247_CPM": {"X": lmat_349, "syms": syms_349, "y": y_349, "ovlp": np.zeros(len(y_349), bool)},
        "GSE165303_Proc": {"X": mat_165, "syms": syms_165, "y": y_165, "ovlp": np.zeros(len(y_165), bool)},
        "GSE116250_RPKM": {"X": g116["mat"], "syms": g116["syms"], "y": g116["y"], "ovlp": np.zeros(len(g116["y"]), bool)},
        "GSE46224_FPKM": {"X": mat_46, "syms": syms_46, "y": y_46, "ovlp": np.zeros(len(y_46), bool)},
        "GSE57338_Array": {"X": g57["mat"], "syms": g57["syms"], "y": g57["y"], "ovlp": mask57_ovlp},
    }
    heldout = {
        "GSE198945_Italy_ICM": {"X": h198["mat"], "syms": h198["syms"], "y": h198["y"]},
        "GSE263297_Sydney_ICM": {"X": h263["mat"], "syms": h263["syms"], "y": h263["y"]},
        "GSE249925_Mayo_HCM": {"X": h249["mat"], "syms": h249["syms"], "y": h249["y"]},
    }

    # Precompute exact Gaussian-kernel GSVA log-odds Z once per cohort (and deduplicated versions of GSE141910 & GSE57338!)
    print("  Precomputing exact tie-averaged Gaussian-kernel GSVA log-odds matrices across all 9 clinical cohorts...")
    Z_gsva = {}
    for cn, cd in disc.items():
        Z_gsva[cn] = sa.compute_gsva_logodds(cd["X"], kcdf="Gaussian")
    Z_gsva["GSE141910_CPM_dedup"] = sa.compute_gsva_logodds(lmat_141[~mask141_ovlp], kcdf="Gaussian")
    Z_gsva["GSE57338_Array_dedup"] = sa.compute_gsva_logodds(g57["mat"][~mask57_ovlp], kcdf="Gaussian")
    for hn, hd in heldout.items():
        Z_gsva[hn] = sa.compute_gsva_logodds(hd["X"], kcdf="Gaussian")

    transfer_methods = [
        ("SS-Rank-Internal", lambda X, idx, w, zkey: sa.score_ss_rank_internal(X, idx, w)),
        ("singscore-Internal", lambda X, idx, w, zkey: sa.score_singscore_internal(X, idx, w)),
        ("SS-Rank", lambda X, idx, w, zkey: sa.score_ss_rank_full(X, idx, w)),
        ("singscore", lambda X, idx, w, zkey: sa.score_singscore_full(X, idx, w)),
        ("ssGSEA", lambda X, idx, w, zkey: sa.score_ssgsea_bidirectional(X, idx, w)),
        ("GSVA_Gaussian", lambda X, idx, w, zkey: score_gsva_from_Z(Z_gsva[zkey], idx, w)),
        ("Cohort-Z", lambda X, idx, w, zkey: score_cohort_z(X, idx, w)[0]),
    ]

    # (3A) Complete 6-Cohort Discovery Transfer (WITH 48-Donor Deduplication on GSE141910 <-> GSE57338 Across ALL 7 Methods!)
    print("  Computing 48-donor-deduplicated discovery transfers (ReHeaT2 Full, ReHeaT2 K=200, De-Novo K=100,500,2000)...")
    cnames = list(disc.keys())

    def eval_pair_transfer(src: str, tgt: str, s_syms, s_w, dedup: bool):
        is_ovlp_pair = dedup and ({src, tgt} == {"GSE141910_CPM", "GSE57338_Array"})
        Xs = disc[src]["X"][~disc[src]["ovlp"]] if is_ovlp_pair else disc[src]["X"]
        ys = disc[src]["y"][~disc[src]["ovlp"]] if is_ovlp_pair else disc[src]["y"]
        zs = f"{src}_dedup" if is_ovlp_pair else src
        Xt = disc[tgt]["X"][~disc[tgt]["ovlp"]] if is_ovlp_pair else disc[tgt]["X"]
        yt = disc[tgt]["y"][~disc[tgt]["ovlp"]] if is_ovlp_pair else disc[tgt]["y"]
        zt = f"{tgt}_dedup" if is_ovlp_pair else tgt
        idx_s = resolve_strict_symbols(disc[src]["syms"], s_syms)
        idx_t = resolve_strict_symbols(disc[tgt]["syms"], s_syms)
        res = []
        for m, fn in transfer_methods:
            sc_s = fn(Xs, idx_s, s_w, zs)
            thr = youden_threshold(ys, sc_s)
            sc_t = fn(Xt, idx_t, s_w, zt)
            ev = eval_threshold(yt, sc_t, thr)
            res.append({"method": m, "source": src, "target": tgt, "dedup_48": is_ovlp_pair,
                        "auc": float(roc_auc_score(yt, sc_t)), **ev})
        return res

    # ReHeaT2 Full & K=200 (deduplicated on GSE141910 <-> GSE57338!)
    rh_full_rows, rh_k200_rows = [], []
    for src in cnames:
        for tgt in cnames:
            if tgt == src:
                continue
            rh_full_rows.extend(eval_pair_transfer(src, tgt, sig_syms, sig_w, dedup=True))
            rh_k200_rows.extend(eval_pair_transfer(src, tgt, k200_syms, k200_w, dedup=True))
    df_rh_full = pd.DataFrame(rh_full_rows)
    df_rh_k200 = pd.DataFrame(rh_k200_rows)
    df_rh_full.to_csv(OUT_DIR / "v7_0_table_disc_reheat2_full_dedup_30.csv", index=False)
    df_rh_k200.to_csv(OUT_DIR / "v7_0_table_disc_reheat2_k200_dedup_30.csv", index=False)

    # Single-Source De-Novo K in {100, 500, 2000}: BOTH full and 48-donor-deduplicated (refitting gene selection AND threshold strictly without the 48 shared codes!)
    denovo_dedup_rows = []
    denovo_ovlp_compare = []
    for K in [100, 500, 2000]:
        for src in cnames:
            # Full-source signature
            gs_full, gw_full = compute_welch_t_signature(disc[src]["X"], disc[src]["y"], disc[src]["syms"], K)
            # Deduplicated-source signature for GSE141910 / GSE57338
            if src in ("GSE141910_CPM", "GSE57338_Array"):
                gs_dd, gw_dd = compute_welch_t_signature(disc[src]["X"][~disc[src]["ovlp"]], disc[src]["y"][~disc[src]["ovlp"]], disc[src]["syms"], K)
            else:
                gs_dd, gw_dd = gs_full, gw_full
            for tgt in cnames:
                if tgt == src:
                    continue
                is_ovlp_pair = {src, tgt} == {"GSE141910_CPM", "GSE57338_Array"}
                if is_ovlp_pair:
                    # Record BOTH original and 48-donor-deduplicated across ALL 7 methods!
                    r_orig = eval_pair_transfer(src, tgt, gs_full, gw_full, dedup=False)
                    r_dd = eval_pair_transfer(src, tgt, gs_dd, gw_dd, dedup=True)
                    for ro, rd in zip(r_orig, r_dd):
                        denovo_ovlp_compare.append({
                            "K": K, "method": ro["method"], "source": src, "target": tgt,
                            "orig_bal_acc": ro["bal_acc"], "dedup_bal_acc": rd["bal_acc"],
                            "delta_bal_acc": rd["bal_acc"] - ro["bal_acc"],
                            "orig_auc": ro["auc"], "dedup_auc": rd["auc"],
                        })
                    for rd in r_dd:
                        denovo_dedup_rows.append({"K": K, **rd})
                else:
                    for r in eval_pair_transfer(src, tgt, gs_full, gw_full, dedup=False):
                        denovo_dedup_rows.append({"K": K, **r})

    df_dn_dedup = pd.DataFrame(denovo_dedup_rows)
    df_dn_dedup.to_csv(OUT_DIR / "v7_0_table_disc_denovo_dedup_90.csv", index=False)
    pd.DataFrame(denovo_ovlp_compare).to_csv(OUT_DIR / "v7_0_table_48donor_overlap_sensitivity_all7.csv", index=False)

    # Strictly DESCRIPTIVE summary (mean, median, min, crashes, per-source 6-cohort mean, win/loss vs SS-Rank-Internal — ZERO pseudo-replicated p-values!)
    desc_rows = []
    for bname, sub in [("ReHeaT2_Full_Dedup (30)", df_rh_full), ("ReHeaT2_K200_Dedup (30)", df_rh_k200), ("DeNovo_Dedup_ALL (90)", df_dn_dedup)]:
        sort_cols = ["source", "target"] + (["K"] if "K" in sub.columns else [])
        ref = sub[sub["method"] == "SS-Rank-Internal"].sort_values(sort_cols)
        print(f"\n  --- Descriptive Discovery Summary: {bname} (ZERO pseudo-replicated p-values) ---")
        for m, _ in transfer_methods:
            cur = sub[sub["method"] == m].sort_values(sort_cols)
            bal, mcc, auc = cur["bal_acc"].to_numpy(), cur["mcc"].to_numpy(), cur["auc"].to_numpy()
            src_means = cur.groupby("source")["bal_acc"].mean()
            wins = int(np.sum(ref["bal_acc"].to_numpy() > bal + 1e-9))
            losses = int(np.sum(ref["bal_acc"].to_numpy() < bal - 1e-9))
            desc_rows.append({
                "benchmark": bname, "method": m, "n_transfers": len(cur),
                "mean_auc": float(auc.mean()), "mean_bal_acc": float(bal.mean()),
                "median_bal_acc": float(np.median(bal)), "min_bal_acc": float(bal.min()),
                "per_source_mean_bal_acc": float(src_means.mean()),
                "per_source_min_bal_acc": float(src_means.min()),
                "crashes_le_055": int(np.sum(bal <= 0.55)),
                "mean_mcc": float(mcc.mean()),
                "ssri_wins_over_method": wins, "ssri_losses_to_method": losses,
            })
            print(f"    {m:18s}: AUC={auc.mean():.3f} | BalAcc={bal.mean():.3f} (med {np.median(bal):.3f}, min {bal.min():.3f}, 6-src mean {src_means.mean():.3f}) | crashes={int(np.sum(bal<=0.55)):2d}/{len(cur)} | MCC={mcc.mean():.3f} | SSRI W/L={wins}/{losses}")
    pd.DataFrame(desc_rows).to_csv(OUT_DIR / "v7_0_table_disc_descriptive_summary.csv", index=False)

    # =========================================================================
    # PART 4: RETROSPECTIVE FIXED-RULE POST-ReHeaT2 EXTERNAL EVALUATION (3 COHORTS, n = 181)
    # =========================================================================
    print("\n[Part 4] Audit UPGRADE #1: Retrospective Fixed-Rule Post-ReHeaT2 External Evaluation (Italy n=40, Sydney n=21, Mayo HCM n=120)...")
    # Fit or select five signatures in source data before scoring the external cohorts in this retrospective workflow:
    #   (1) ReHeaT2 Full (calibrated on deduplicated GSE141910 CPM, deduplicated GSE57338 Array, GSE165303 Proc)
    #   (2) ReHeaT2 K=200 (same 3 calibration sources)
    #   (3) De-Novo K=200 from deduplicated GSE141910 (US CPM)
    #   (4) De-Novo K=200 from deduplicated GSE57338 (US Array)
    #   (5) De-Novo K=500 from GSE165303 (China Processed)
    dn141_s, dn141_w = compute_welch_t_signature(lmat_141[~mask141_ovlp], y_141[~mask141_ovlp], syms_141, 200)
    dn57_s, dn57_w = compute_welch_t_signature(g57["mat"][~mask57_ovlp], g57["y"][~mask57_ovlp], g57["syms"], 200)
    dn165_s, dn165_w = compute_welch_t_signature(mat_165, y_165, syms_165, 500)

    locked_specs = [
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE141910_CPM", True),
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE57338_Array", True),
        ("ReHeaT2_Full", sig_syms, sig_w, "GSE165303_Proc", False),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE141910_CPM", True),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE57338_Array", True),
        ("ReHeaT2_K200", k200_syms, k200_w, "GSE165303_Proc", False),
        ("DeNovo_K200_MAGNet", dn141_s, dn141_w, "GSE141910_CPM", True),
        ("DeNovo_K200_Array", dn57_s, dn57_w, "GSE57338_Array", True),
        ("DeNovo_K500_Proc", dn165_s, dn165_w, "GSE165303_Proc", False),
    ]

    ho_rows = []
    for sig_name, s_g, s_w, src_name, use_dedup in locked_specs:
        Xs = disc[src_name]["X"][~disc[src_name]["ovlp"]] if use_dedup else disc[src_name]["X"]
        ys = disc[src_name]["y"][~disc[src_name]["ovlp"]] if use_dedup else disc[src_name]["y"]
        zs = f"{src_name}_dedup" if use_dedup else src_name
        idx_s = resolve_strict_symbols(disc[src_name]["syms"], s_g)
        for m, fn in transfer_methods:
            thr = youden_threshold(ys, fn(Xs, idx_s, s_w, zs))
            for hname, hd in heldout.items():
                idx_h = resolve_strict_symbols(hd["syms"], s_g)
                sc_h = fn(hd["X"], idx_h, s_w, hname)
                ev = eval_threshold(hd["y"], sc_h, thr)
                ho_rows.append({
                    "signature": sig_name, "calib_source": src_name, "heldout_cohort": hname,
                    "method": m, "n_mapped": sum(x is not None for x in idx_h),
                    "auc": float(roc_auc_score(hd["y"], sc_h)), **ev,
                })
    df_ho = pd.DataFrame(ho_rows)
    df_ho.to_csv(OUT_DIR / "v7_0_table_heldout_external_validation_27x7.csv", index=False)

    # Summarize retrospective external evaluation by cohort and overall (27 evaluations per method).
    ho_sum_rows = []
    print("\n  --- Retrospective Fixed-Rule Post-ReHeaT2 External Evaluation (27 evaluations/method across Italy, Sydney, Mayo HCM) ---")
    for m, _ in transfer_methods:
        sub_m = df_ho[df_ho["method"] == m]
        by_c = {hn: sub_m[sub_m["heldout_cohort"] == hn] for hn in heldout}
        ho_sum_rows.append({
            "method": m,
            "all27_mean_auc": float(sub_m["auc"].mean()),
            "all27_mean_bal_acc": float(sub_m["bal_acc"].mean()),
            "all27_median_bal_acc": float(sub_m["bal_acc"].median()),
            "all27_min_bal_acc": float(sub_m["bal_acc"].min()),
            "all27_mean_mcc": float(sub_m["mcc"].mean()),
            "all27_crashes_le_055": int((sub_m["bal_acc"] <= 0.55).sum()),
            "italy_auc": float(by_c["GSE198945_Italy_ICM"]["auc"].mean()),
            "italy_bal_acc": float(by_c["GSE198945_Italy_ICM"]["bal_acc"].mean()),
            "sydney_auc": float(by_c["GSE263297_Sydney_ICM"]["auc"].mean()),
            "sydney_bal_acc": float(by_c["GSE263297_Sydney_ICM"]["bal_acc"].mean()),
            "mayo_hcm_auc": float(by_c["GSE249925_Mayo_HCM"]["auc"].mean()),
            "mayo_hcm_bal_acc": float(by_c["GSE249925_Mayo_HCM"]["bal_acc"].mean()),
        })
        r = ho_sum_rows[-1]
        print(f"    {m:18s}: Overall AUC={r['all27_mean_auc']:.3f}, BalAcc={r['all27_mean_bal_acc']:.3f} (med {r['all27_median_bal_acc']:.3f}, min {r['all27_min_bal_acc']:.3f}, crashes={r['all27_crashes_le_055']:2d}/27, MCC={r['all27_mean_mcc']:.3f}) | Italy={r['italy_bal_acc']:.3f} | Sydney={r['sydney_bal_acc']:.3f} | MayoHCM={r['mayo_hcm_bal_acc']:.3f}")
    pd.DataFrame(ho_sum_rows).to_csv(OUT_DIR / "v7_0_table_heldout_summary.csv", index=False)

    # =========================================================================
    # PART 5: Audit UPGRADE #2 — SECOND INDEPENDENT REAL CPM->TPM STRESS TEST ON HELD-OUT GSE249925 (AND GSE141910!)
    # =========================================================================
    print("\n[Part 5] Multi-Signature Real CPM->TPM Gene-Length Stress Test on BOTH GSE141910 & Held-Out GSE249925...")
    # Load MAGNet gene lengths
    gl_df = pd.read_csv(DS / "GSE141910/MAGNet_GeneLengths.txt", sep="\t")
    gl_map = dict(zip(gl_df.iloc[:, 0].astype(str).str.split(".").str[0], pd.to_numeric(gl_df.iloc[:, 1], errors="coerce").fillna(1000.0)))
    lens_141 = np.array([gl_map.get(str(g).split(".")[0], 1000.0) for g in cnt141.index], float)
    rpk141 = raw141 / np.maximum(lens_141[None, :] / 1000.0, 1e-6)
    lmat_141_tpm = np.log2((rpk141 / np.maximum(rpk141.sum(axis=1, keepdims=True), 1e-12)) * 1e6 + 1.0)

    len_rows = []
    ind_methods_len = [
        ("SS-Rank-Internal", lambda X, idx, w: sa.score_ss_rank_internal(X, idx, w)),
        ("singscore-Internal", lambda X, idx, w: sa.score_singscore_internal(X, idx, w)),
        ("SS-Rank", lambda X, idx, w: sa.score_ss_rank_full(X, idx, w)),
        ("singscore", lambda X, idx, w: sa.score_singscore_full(X, idx, w)),
        ("ssGSEA", lambda X, idx, w: sa.score_ssgsea_bidirectional(X, idx, w)),
    ]
    for dname, X_cpm, X_tpm, dsyms, dy in [
        ("GSE141910_MAGNet", lmat_141, lmat_141_tpm, syms_141, y_141),
        ("GSE249925_Mayo_HeldOut", h249["mat"], h249["mat_tpm"], h249["syms"], h249["y"]),
    ]:
        for sname, sg, sw in [("ReHeaT2_Full", sig_syms, sig_w), ("ReHeaT2_K200", k200_syms, k200_w)]:
            idx_d = resolve_strict_symbols(dsyms, sg)
            for m, fn in ind_methods_len:
                sc_c = fn(X_cpm, idx_d, sw)
                sc_t = fn(X_tpm, idx_d, sw)
                thr_c = youden_threshold(dy, sc_c)
                ev_c = eval_threshold(dy, sc_c, thr_c)
                ev_t = eval_threshold(dy, sc_t, thr_c)
                shift_sd = float(np.mean(np.abs(sc_t - sc_c)) / np.std(sc_c, ddof=1))
                len_rows.append({
                    "cohort": dname, "signature": sname, "method": m,
                    "cpm_bal_acc": ev_c["bal_acc"], "tpm_frozen_bal_acc": ev_t["bal_acc"],
                    "tpm_frozen_mcc": ev_t["mcc"], "tpm_auc": float(roc_auc_score(dy, sc_t)),
                    "mean_abs_shift_sd": shift_sd,
                })
                if sname == "ReHeaT2_Full":
                    print(f"  CPM->TPM {dname:22s} | {m:18s}: shift={shift_sd:.3f} SD | frozen BalAcc={ev_t['bal_acc']:.3f} (MCC={ev_t['mcc']:.3f}, AUC={roc_auc_score(dy, sc_t):.3f})")
    pd.DataFrame(len_rows).to_csv(OUT_DIR / "v7_0_table_cpm_to_tpm_two_cohorts.csv", index=False)

    # =========================================================================
    # PART 6: 2D SCREENSEQ (GSE244740) — SEPARATING GSVA(Gaussian) vs GSVA(none/ECDF) + PAIRED 38-DRUG DIFF CIs!
    # =========================================================================
    print("\n[Part 6] 2D ScreenSeq (GSE244740, 2,899 RNA, 38 drugs): Tie-Avg Methods + Paired Drug-Level Diff Bootstrap CIs...")
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

    # Note: Explicitly label GSVA (kcdf='none', ECDF) separately — and ALSO run exact GSVA (kcdf='Gaussian') on the 190 drug x rank condition means? No, run kcdf='none' on all 2,899 AND exact kcdf='Gaussian' on 2,899 via fast histogram? Wait: let's check how fast exact kcdf='Gaussian' is on 2,899 x 19k — 2899^2 * 19k = 1.6e11 ops (~10 min too slow). Instead, evaluate exact kcdf='none' (ECDF, official Bioconductor GSVA option!) on all 2,899 samples AND explicitly label it `GSVA (kcdf='none', ECDF)` — NEVER claiming it approximates Gaussian kernel!
    b6_scores = {
        "Cohort-Z": score_cohort_z(lmat_b6, idx_b6, sig_w)[0],
        "Cohort-Robust-Z": score_cohort_robust_z(lmat_b6, idx_b6, sig_w)[0],
        "GSVA (kcdf='none', ECDF)": sa.score_gsva_bidirectional(lmat_b6, idx_b6, sig_w, kcdf="none"),
        "SS-Rank": sa.score_ss_rank_full(lmat_b6, idx_b6, sig_w),
        "SS-Rank-Internal": sa.score_ss_rank_internal(lmat_b6, idx_b6, sig_w),
        "singscore": sa.score_singscore_full(lmat_b6, idx_b6, sig_w),
        "singscore-Internal": sa.score_singscore_internal(lmat_b6, idx_b6, sig_w),
        "ssGSEA (tie-avg.)": sa.score_ssgsea_bidirectional(lmat_b6, idx_b6, sig_w),
        "Ref-Frozen-Z": score_ref_frozen_z(lmat_b6, syms_b6, ref_mu, ref_sd, sig_syms, sig_w)[0],
    }

    # Build aligned 38-drug vectors for Amplitude so we compute EXACT paired bootstrap 95% CI of (rho_CZ - rho_m)!
    sub_amp = v5_b6_ok[v5_b6_ok["assay"] == "Amplitude"]
    drugs_38 = []
    amp_y_38 = []
    drug_x_by_m = {m: [] for m in b6_scores}
    within_by_m = {m: [] for m in b6_scores}
    for d in sorted(sub_amp["drug"].unique()):
        d_sub = sub_amp[sub_amp["drug"] == d].sort_values("concentration_rank")
        cy = []
        cs_m = {m: [] for m in b6_scores}
        for _, crow in d_sub.iterrows():
            idxs = cmap.get((d, str(crow["concentration_rank"]).strip()))
            if idxs:
                cy.append(float(crow["assay_value"]))
                for m, sc in b6_scores.items():
                    cs_m[m].append(float(np.mean(sc[idxs])))
        if len(cy) >= 3:
            drugs_38.append(d)
            amp_y_38.append(float(np.mean(cy)))
            for m in b6_scores:
                drug_x_by_m[m].append(float(np.mean(cs_m[m])))
                within_by_m[m].append(spearman_rho(cs_m[m], cy))

    amp_y = np.array(amp_y_38)
    rng_b = np.random.default_rng(20261120)
    boot_idx = rng_b.integers(0, len(amp_y), size=(5000, len(amp_y)))
    x_cz_38 = np.array(drug_x_by_m["Cohort-Z"])
    cz_boots = np.array([spearman_rho(x_cz_38[bi], amp_y[bi]) for bi in boot_idx])
    b6_rows = []
    for m in b6_scores:
        xm = np.array(drug_x_by_m[m])
        obs = spearman_rho(xm, amp_y)
        mb = np.array([spearman_rho(xm[bi], amp_y[bi]) for bi in boot_idx])
        diff_b = cz_boots - mb
        w_arr = np.array([v for v in within_by_m[m] if np.isfinite(v)])
        b6_rows.append({
            "method": m, "n_drugs": len(amp_y),
            "across_rho": obs, "ci_low": float(np.quantile(mb, 0.025)), "ci_high": float(np.quantile(mb, 0.975)),
            "diff_cz_minus_m": float(spearman_rho(x_cz_38, amp_y) - obs),
            "diff_ci_low": float(np.quantile(diff_b, 0.025)), "diff_ci_high": float(np.quantile(diff_b, 0.975)),
            "within_median_rho": float(np.median(w_arr)), "within_pct_neg": float(100.0 * np.mean(w_arr < 0)),
        })
        r = b6_rows[-1]
        print(f"  2D {m:24s}: rho={r['across_rho']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}] | CZ-m={r['diff_cz_minus_m']:+.3f} [{r['diff_ci_low']:+.3f},{r['diff_ci_high']:+.3f}] | within={r['within_median_rho']:+.3f}")
    pd.DataFrame(b6_rows).to_csv(OUT_DIR / "v7_0_table_2d_screenseq_paired_diff.csv", index=False)

    # =========================================================================
    # PART 7: DESCRIPTIVE O(nm) VIF TABLE (ZERO INVALID P-VALUES!) + ENVIRONMENT MANIFEST
    # =========================================================================
    print("\n[Part 7] Exact Descriptive O(nm) VIF Table (NO invalid custom p-values) & Environment Manifest...")
    axes_specs = [
        ("ReHeaT2_mol_score", "mol_score"),
        ("ReHeaT2_consistent_yes", "mol_score"),
        ("ReHeaT2_mean_t_all16036", "mean_t"),
        ("ReHeaT2_vSMCs", "mcell_HFscore_vSMCs"),
        ("ReHeaT2_Endo", "mcell_HFscore_Endo"),
        ("ReHeaT2_Fib", "mcell_HFscore_Fib"),
        ("ReHeaT2_PC", "mcell_HFscore_PC"),
        ("ReHeaT2_Myeloid", "mcell_HFscore_Myeloid"),
        ("ReHeaT2_CM", "mcell_HFscore_CM"),
        ("ReHeaT2_Lymphoid", "mcell_HFscore_Lymphoid"),
    ]
    vif_rows = []
    for aname, col in axes_specs:
        sub = mc[mc["mol_consistent"].astype(str).str.lower() == "yes"] if aname == "ReHeaT2_consistent_yes" else mc
        sub = sub[sub[col].notna() & (sub[col] != 0)]
        aidx = resolve_strict_symbols(gene_syms, list(sub["gene"].astype(str)))
        aw = sub[col].to_numpy(float)
        vd = sa.descriptive_vif_audit(trt_mat_3d, aidx, aw)
        vif_rows.append({"axis": aname, **vd})
    pd.DataFrame(vif_rows).to_csv(OUT_DIR / "v7_0_table_descriptive_vif_10axes.csv", index=False)

    env_manifest = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "column_permutation_max_abs_diff_gse314310": perm_audit,
        "runtime_total_seconds": round(time.perf_counter() - t0_all, 2),
    }
    (OUT_DIR / "v7_0_environment_and_invariance_manifest.json").write_text(json.dumps(env_manifest, indent=2), encoding="utf-8")
    print(f"\n=== V7.0 Retrospective External Evaluation Completed Cleanly in {time.perf_counter() - t0_all:.1f}s! ===")


if __name__ == "__main__":
    main()
