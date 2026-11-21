"""
Shared benchmark helper functions (robust scaling, cohort loaders, Youden thresholding,
cluster-robust CR1 OLS, Welch t-statistic signature derivation, and zero-inflation stress).
"""

from __future__ import annotations
import gzip
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_3d_cardiopedia_benchmark import DS, V5_ROOT, resolve_strict_symbols

GPL11532_PATH = V5_ROOT / "GPL11532.annot.gz"


def score_cohort_robust_z(expr_2d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray):
    """Vectorized transductive robust Z-score using gene-wise median and scaled MAD (1.4826 * MAD)."""
    w = np.asarray(weights, float)
    valid = [(j, idx, w[j]) for j, idx in enumerate(gene_indices) if idx is not None and math.isfinite(w[j]) and w[j] != 0]
    cols = np.array([idx for _, idx, _ in valid], dtype=int)
    wv = np.array([wj for _, _, wj in valid], dtype=float)
    X = expr_2d[:, cols]  # (n, m)
    med = np.median(X, axis=0)
    mad = np.median(np.abs(X - med[None, :]), axis=0) * 1.4826
    zero_mad = ~np.isfinite(mad) | (mad <= 1e-12)
    if np.any(zero_mad):
        q75 = np.quantile(X[:, zero_mad], 0.75, axis=0)
        q25 = np.quantile(X[:, zero_mad], 0.25, axis=0)
        mad[zero_mad] = (q75 - q25) / 1.349
    keep = np.isfinite(mad) & (mad > 1e-12)
    Z = np.clip((X[:, keep] - med[None, keep]) / mad[None, keep], -6.0, 6.0)
    wk = wv[keep]
    out = (Z @ wk) / np.abs(wk).sum()
    usable = np.zeros(len(w), bool)
    for (j, _, _), ok in zip(valid, keep):
        usable[j] = bool(ok)
    return out, usable


def load_gse116250():
    """Load GSE116250 (Sweet et al. 2018, 64 adult LV RNA-seq RPKM: 14 NF, 37 DCM, 13 ICM)."""
    df = pd.read_csv(DS / "GSE116250/GSE116250_rpkm.txt.gz", sep="\t")
    sample_cols = [c for c in df.columns if re.match(r"^(NF|DCM|ICM)\d+$", str(c))]
    syms = [str(s).strip() for s in df["Common_name"].fillna("")]
    mat = np.log2(1.0 + np.maximum(df[sample_cols].apply(pd.to_numeric, errors="coerce").to_numpy(float).T, 0.0))
    fin = np.all(np.isfinite(mat), axis=0)
    mat = mat[:, fin]
    syms = [s for s, ok in zip(syms, fin) if ok]
    y = np.array([0 if c.startswith("NF") else 1 for c in sample_cols], int)
    et = ["NF" if c.startswith("NF") else ("DCM" if c.startswith("DCM") else "ICM") for c in sample_cols]
    return {"samples": sample_cols, "y": y, "etiology": et, "syms": syms, "mat": mat}


def load_gse57338():
    """Load GSE57338 (Liu et al. 2015, 313 adult LV Affymetrix GPL11532 microarray) + exact MAGNet ID overlap mask."""
    # 1. Parse GPL11532 unambiguous single-gene probe -> symbol map
    probe_to_sym = {}
    with gzip.open(GPL11532_PATH, "rt", encoding="utf-8", errors="replace") as fh:
        in_tbl = False
        hdr = None
        for line in fh:
            if line.startswith("!platform_table_begin"):
                in_tbl = True
                continue
            if line.startswith("!platform_table_end"):
                break
            if in_tbl:
                parts = line.rstrip("\n").split("\t")
                if hdr is None:
                    hdr = parts
                    id_i = hdr.index("ID")
                    sym_i = hdr.index("Gene symbol")
                    continue
                if len(parts) > sym_i:
                    pid = parts[id_i].strip()
                    sym = parts[sym_i].strip()
                    if pid and sym and "///" not in sym:
                        probe_to_sym[pid] = sym

    # 2. Parse GSE57338 series matrix metadata + expression table
    titles, hf_flags, statuses, ages, sexes = [], [], [], [], []
    expr_rows = []
    probe_ids = []
    with gzip.open(DS / "GSE57338/GSE57338_series_matrix.txt.gz", "rt", encoding="utf-8", errors="replace") as fh:
        in_tbl = False
        gsm_cols = None
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip('"') for x in line.strip().split("\t")[1:]]
            elif line.startswith("!Sample_characteristics_ch1"):
                vals = [x.strip('"') for x in line.strip().split("\t")[1:]]
                if vals and vals[0].startswith("heart failure:"):
                    hf_flags = [v.replace("heart failure: ", "").strip() for v in vals]
                elif vals and vals[0].startswith("disease status:"):
                    statuses = [v.replace("disease status: ", "").strip() for v in vals]
                elif vals and vals[0].startswith("gender:"):
                    sexes = [v.replace("gender: ", "").strip() for v in vals]
                elif vals and vals[0].startswith("age:"):
                    ages = [v.replace("age: ", "").strip() for v in vals]
            elif line.startswith("!series_matrix_table_begin"):
                in_tbl = True
                continue
            elif line.startswith("!series_matrix_table_end"):
                break
            elif in_tbl:
                parts = line.rstrip("\n").split("\t")
                if gsm_cols is None:
                    gsm_cols = [x.strip('"') for x in parts[1:]]
                    continue
                pid = parts[0].strip('"')
                if pid in probe_to_sym:
                    probe_ids.append(pid)
                    expr_rows.append([float(x) for x in parts[1:]])

    # Collapse multiple unambiguous probes per gene symbol by mean
    raw_df = pd.DataFrame(expr_rows, columns=gsm_cols)
    raw_df["sym"] = [probe_to_sym[p] for p in probe_ids]
    collapsed = raw_df.groupby("sym", sort=True).mean()
    syms = list(collapsed.index)
    mat = collapsed.to_numpy(float).T  # (313, n_genes)
    y = np.array([1 if f.lower() == "yes" else 0 for f in hf_flags], int)

    # Identify exact donor ID overlap with GSE141910 MAGNet (e.g., C02065 / P01234 tokens)
    ph141 = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    mag_ids = set(ph141["sample_name"].astype(str).str.strip())
    # Normalize GSE57338 leading code (e.g., 'C02065' vs 'C02065' or 'P01234' vs 'P01234')
    code_57 = [t.split()[0].strip() for t in titles]
    overlap_mask = np.array([c in mag_ids for c in code_57], bool)
    return {
        "samples": gsm_cols, "titles": titles, "codes": code_57,
        "y": y, "status": statuses, "age": pd.to_numeric(ages, errors="coerce"),
        "sex": sexes, "syms": syms, "mat": mat, "magnet_id_overlap": overlap_mask,
    }


def youden_threshold(y_true, scores):
    fpr, tpr, thrs = roc_curve(y_true, scores)
    return float(thrs[int(np.argmax(tpr - fpr))])


def eval_threshold(y_true, scores, thr):
    pred = (scores >= thr).astype(int)
    sens = float(np.mean(pred[y_true == 1] == 1))
    spec = float(np.mean(pred[y_true == 0] == 0))
    bal = 0.5 * (sens + spec)
    mcc = float(matthews_corrcoef(y_true, pred))
    return {"bal_acc": bal, "mcc": mcc, "sens": sens, "spec": spec}


def cluster_robust_ols_cr1(X: np.ndarray, y: np.ndarray, cluster_ids: np.ndarray):
    """OLS with Liang-Zeger CR1 cluster-robust variance estimator (small-sample corrected)."""
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    XtX_inv = np.linalg.inv(X.T @ X)
    uniq = np.unique(cluster_ids)
    G = len(uniq)
    meat = np.zeros((k, k), float)
    for g in uniq:
        m = cluster_ids == g
        Xg = X[m]
        ug = Xg.T @ resid[m]
        meat += np.outer(ug, ug)
    scale = (G / (G - 1.0)) * ((n - 1.0) / (n - k))
    V = scale * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(V))
    tval = beta / se
    pval = 2.0 * stats.t.sf(np.abs(tval), df=G - 1)
    return {"beta": beta, "se": se, "t": tval, "p": pval}


def compute_welch_t_signature(expr_2d: np.ndarray, y_bin: np.ndarray, syms: list[str], K: int):
    """Derive 100% blind de-novo balanced signed signature strictly inside Source Cohort (ZERO ReHeaT2, ZERO Target)."""
    X1 = expr_2d[y_bin == 1]
    X0 = expr_2d[y_bin == 0]
    m1, m0 = X1.mean(axis=0), X0.mean(axis=0)
    v1, v0 = X1.var(axis=0, ddof=1), X0.var(axis=0, ddof=1)
    se = np.sqrt(v1 / len(X1) + v0 / len(X0))
    valid = (se > 1e-8) & np.isfinite(m1 - m0)
    t_vals = np.zeros(expr_2d.shape[1], float)
    t_vals[valid] = (m1[valid] - m0[valid]) / se[valid]

    sym_best = {}
    for j, s in enumerate(syms):
        if not s or not valid[j]:
            continue
        if s not in sym_best or abs(t_vals[j]) > abs(sym_best[s]):
            sym_best[s] = float(t_vals[j])
    pos = sorted([(s, t) for s, t in sym_best.items() if t > 0], key=lambda x: -x[1])
    neg = sorted([(s, t) for s, t in sym_best.items() if t < 0], key=lambda x: x[1])
    half = K // 2
    chosen = pos[:half] + neg[:half]
    return [s for s, _ in chosen], np.array([t for _, t in chosen], float)


def apply_zero_inflation(expr_2d: np.ndarray, pct_zero: float) -> np.ndarray:
    """Zero the lowest pct_zero% of genes per sample using original column order.

    A nonzero tie crossing the cutoff is resolved by stable input-column order;
    this perturbation can depend on column order even when a score is tie-invariant.
    """
    if pct_zero <= 0:
        return expr_2d.copy()
    out = expr_2d.copy()
    G = out.shape[1]
    n_zero = int(round(G * (pct_zero / 100.0)))
    for i in range(out.shape[0]):
        order = np.argsort(out[i], kind="mergesort")
        out[i, order[:n_zero]] = 0.0
    return out
