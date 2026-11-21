"""
3D Cardiopedia (GSE314310) multi-paradigm scoring, 13-endpoint kinetics, and co-expression VIF benchmark.
"""

from __future__ import annotations
import csv
import gzip
import json
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("CARDIAC_DATA_DIR", REPO_ROOT / "data"))
ROOT = DATA_DIR
DS = DATA_DIR / "datasets"
V5_ROOT = DATA_DIR / "reference"
V5_TABLES = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PLATES = ("P01", "P03", "P04", "P05", "P06")
REPEAT_PLATES = ("P01", "P02", "P03", "P04", "P06")
CONTROLS = ("CTRL-WM", "CTRL-DMSO", "CTRL-Ethanol")
KINETIC_ENDPOINTS = (
    "Force",
    "BPM",
    "Ta_50",
    "Tr_50",
    "Ts_50_50",
    "Ts_85_85",
    "Ts_15_15",
    "Ta_15_30",
    "Ta_30_85",
    "Tr_85_50",
    "Tr_50_15",
    "Contract_velocity_norm",
    "Relax_velocity_norm",
)


def _col_ref_to_idx(ref: str) -> int:
    col = "".join(ch for ch in ref if ch.isalpha())
    idx = 0
    for ch in col:
        idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def iter_xlsx_rows(xlsx_path: Path, sheet_name: str | None = None):
    """Stream rows from an xlsx sheet using pure stdlib zipfile + ElementTree."""
    ns_uri = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns = {"a": ns_uri}
    with zipfile.ZipFile(xlsx_path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", ns):
                shared.append("".join(t.text or "" for t in si.findall(".//a:t", ns)))
        wb = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        rmap = {r.attrib["Id"]: r.attrib["Target"] for r in rels}
        target = None
        for sh in wb.findall(".//a:sheet", ns):
            nm = sh.attrib["name"]
            if sheet_name is None or nm == sheet_name:
                rid = sh.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
                t = rmap[rid].lstrip("/")
                target = t if t.startswith("xl/") else f"xl/{t}"
                break
        if target is None:
            raise ValueError(f"Sheet {sheet_name!r} not found in {xlsx_path}")

        row_tag = f"{{{ns_uri}}}row"
        v_tag = f"{{{ns_uri}}}v"
        is_tag = f"{{{ns_uri}}}is"
        t_tag = f"{{{ns_uri}}}t"
        for _, elem in ET.iterparse(z.open(target), events=("end",)):
            if elem.tag == row_tag:
                cells = {}
                max_i = -1
                for c in elem:
                    ref = c.attrib.get("r")
                    ci = _col_ref_to_idx(ref) if ref else (max_i + 1)
                    if ci > max_i:
                        max_i = ci
                    ctype = c.attrib.get("t")
                    v = c.find(v_tag)
                    if v is not None and v.text is not None:
                        val = shared[int(v.text)] if ctype == "s" else v.text
                    else:
                        is_el = c.find(is_tag)
                        val = "".join(x.text or "" for x in is_el.iter(t_tag)) if is_el is not None else None
                    cells[ci] = val
                row = [cells.get(i) for i in range(max_i + 1)]
                elem.clear()
                yield row


def to_float(v):
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def spearman_rho(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan")
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def bootstrap_spearman_ci(x, y, n_boot=5000, seed=20261120):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    obs = spearman_rho(x, y)
    rhos = np.empty(n_boot, float)
    for b in range(n_boot):
        rhos[b] = spearman_rho(x[idx[b]], y[idx[b]])
    rhos = rhos[np.isfinite(rhos)]
    lo, hi = np.quantile(rhos, [0.025, 0.975])
    return {"rho": obs, "ci_low": float(lo), "ci_high": float(hi), "n": int(n)}


def partial_spearman(x, y, covars: np.ndarray):
    """Partial Spearman rank correlation between x and y controlling for rank-transformed covariates."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    C = np.asarray(covars, float)
    if C.ndim == 1:
        C = C[:, None]
    mask = np.isfinite(x) & np.isfinite(y) & np.all(np.isfinite(C), axis=1)
    rx = stats.rankdata(x[mask])
    ry = stats.rankdata(y[mask])
    RC = np.column_stack([stats.rankdata(C[mask, j]) for j in range(C.shape[1])])
    X = np.column_stack([np.ones(len(rx)), RC])
    bx, *_ = np.linalg.lstsq(X, rx, rcond=None)
    by, *_ = np.linalg.lstsq(X, ry, rcond=None)
    ex = rx - X @ bx
    ey = ry - X @ by
    r = float(np.corrcoef(ex, ey)[0, 1])
    df = len(rx) - X.shape[1] - 1
    if df > 0 and abs(r) < 1.0:
        t_stat = r * math.sqrt(df / (1.0 - r * r))
        p_val = float(2.0 * stats.t.sf(abs(t_stat), df))
    else:
        t_stat, p_val = float("nan"), float("nan")
    return {"partial_rho": r, "t_stat": float(t_stat), "df": int(df), "p_two_sided": p_val, "n": int(len(rx))}


def log2_cpm(counts_2d: np.ndarray) -> np.ndarray:
    libs = counts_2d.sum(axis=1, keepdims=True)
    return np.log2(1.0 + 1e6 * counts_2d / libs)


def log2_deseq2_median_ratio(counts_2d: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factor normalization followed by log2(norm + 1)."""
    # Compute geometric mean across samples on genes with all counts > 0
    pos = np.all(counts_2d > 0, axis=0)
    if pos.sum() < 50:
        # Fallback: genes with > 50% nonzero
        pos = (counts_2d > 0).mean(axis=0) > 0.8
        sub = np.where(counts_2d[:, pos] > 0, counts_2d[:, pos], np.nan)
        log_gm = np.nanmean(np.log(sub), axis=0)
        ratios = np.exp(np.log(sub) - log_gm[None, :])
        sf = np.nanmedian(ratios, axis=1)
    else:
        sub = counts_2d[:, pos]
        log_gm = np.mean(np.log(sub), axis=0)
        sf = np.median(np.exp(np.log(sub) - log_gm[None, :]), axis=1)
    sf = sf / np.exp(np.mean(np.log(sf)))
    return np.log2(1.0 + counts_2d / sf[:, None])


def score_cohort_z(expr_2d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray):
    """Exact V5 transductive cohort-Z scoring."""
    w = np.asarray(weights, float)
    n = expr_2d.shape[0]
    out = np.zeros(n, float)
    usable = np.zeros(len(w), bool)
    for j, idx in enumerate(gene_indices):
        if idx is None or not math.isfinite(w[j]) or w[j] == 0:
            continue
        v = expr_2d[:, idx]
        sd = v.std(ddof=1)
        if not math.isfinite(sd) or sd <= 1e-12:
            continue
        usable[j] = True
        out += ((v - v.mean()) / sd) * w[j]
    denom = np.abs(w[usable]).sum()
    return out / denom, usable


def score_within_plate_z(expr_3d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray):
    """Plate-wise Z scoring (transductive within each plate separately)."""
    n_plates, n_trt, _ = expr_3d.shape
    scores = np.zeros((n_plates, n_trt), float)
    for p in range(n_plates):
        s_p, _ = score_cohort_z(expr_3d[p], gene_indices, weights)
        scores[p] = s_p
    return scores


def score_single_sample_rank(expr_2d: np.ndarray, gene_indices: list[int | None], weights: np.ndarray):
    """
    Inductive single-sample weighted fractional rank score (singscore-analog).
    For each sample i independently, rank all genes present in the expression matrix into [-0.5, +0.5],
    then take sum_g w_g * r_{i,g} / sum_g |w_g|. Zero cross-sample dependence!
    """
    w = np.asarray(weights, float)
    valid_pairs = [(j, idx, w[j]) for j, idx in enumerate(gene_indices) if idx is not None and math.isfinite(w[j]) and w[j] != 0]
    col_idx = np.array([idx for _, idx, _ in valid_pairs], dtype=int)
    w_valid = np.array([wj for _, _, wj in valid_pairs], dtype=float)
    denom = np.abs(w_valid).sum()
    n_samples, n_genes = expr_2d.shape
    out = np.zeros(n_samples, float)
    for i in range(n_samples):
        # Rank across ALL genes in the sample's own transcriptome
        r = (stats.rankdata(expr_2d[i]) - 0.5) / n_genes - 0.5
        out[i] = float(np.dot(r[col_idx], w_valid) / denom)
    return out, len(valid_pairs)


def score_ref_frozen_z(expr_2d: np.ndarray, expr_symbols: list[str], ref_mu: dict[str, float], ref_sd: dict[str, float],
                       sig_symbols: list[str], weights: np.ndarray):
    """
    Inductive externally calibrated score using frozen mean/SD from Adult Non-Failing LV (GSE141910 NF, n=166).
    Each target sample is scored individually using fixed (mu_ref, sd_ref); zero target-cohort statistics used.
    """
    sym_to_idx = {}
    counts_sym = defaultdict(int)
    for idx, s in enumerate(expr_symbols):
        if s:
            counts_sym[s] += 1
            sym_to_idx[s] = idx
    unique_map = {s: idx for s, idx in sym_to_idx.items() if counts_sym[s] == 1}
    n = expr_2d.shape[0]
    out = np.zeros(n, float)
    denom = 0.0
    used = 0
    for s, w in zip(sig_symbols, weights):
        if not math.isfinite(w) or w == 0:
            continue
        if s in unique_map and s in ref_mu and s in ref_sd and ref_sd[s] > 1e-12:
            idx = unique_map[s]
            out += ((expr_2d[:, idx] - ref_mu[s]) / ref_sd[s]) * w
            denom += abs(w)
            used += 1
    return out / denom, used


def load_adult_nf_reference():
    """Load GSE141910 NF controls (n=166) as frozen external adult myocardium calibration reference (log2-CPM)."""
    pheno = pd.read_csv(DS / "GSE141910/MAGNet_phenoData.csv")
    # Identify sample ID column and etiology column
    id_col = pheno.columns[0]
    et_col = [c for c in pheno.columns if "etiol" in c.lower()][0]
    nf_ids = set(pheno.loc[pheno[et_col].astype(str).str.upper() == "NF", id_col].astype(str))
    # Map Ensembl IDs to symbols via ensembl_to_symbol.pkl or Cardiopedia table
    counts_df = pd.read_csv(DS / "GSE141910/MAGNet_raw_Counts.csv", index_col=0)
    nf_cols = [c for c in counts_df.columns if str(c) in nf_ids]
    mat = counts_df[nf_cols].to_numpy(dtype=float).T  # (n_nf, n_genes)
    lmat = log2_cpm(mat)
    ens_ids = [str(g).split(".")[0] for g in counts_df.index]
    return ens_ids, lmat, len(nf_cols)


def load_cardiopedia_strict():
    """Load Cardiopedia 235 organoids (5 plates x 47 treatments) + all 13 endpoints at 04H and 24H + repeat experiment."""
    # Get exact 47 labels from V5 incidence file
    inc_path = V5_ROOT / "prior_art/cardiopedia_repeat_overlap_47_metadata_2026-11-16.json"
    inc_doc = json.loads(inc_path.read_text(encoding="utf-8"))
    # Wait: let's read the exact 47 labels from V5 3d_treatment_summary.csv to guarantee exact order
    v5_trt = pd.read_csv(V5_TABLES / "3d_treatment_summary.csv")
    labels = list(v5_trt["treatment"].astype(str))
    assert len(labels) == 47, f"Expected 47 labels, got {len(labels)}"
    label_set = set(labels)

    # 1. Parse RNA workbook header to locate columns
    rna_xlsx = DS / "GSE314310/GSE314310_RSEM_Counts_Cardiopedia-Ligand.xlsx"
    it = iter_xlsx_rows(rna_xlsx)
    header = next(it)
    assert header[:3] == ["gene_id", "gene_symbol", "gene_biotype"], f"Unexpected header: {header[:5]}"
    rna_map = {}
    for j, raw in enumerate(header[3:], 3):
        m = re.fullmatch(r"(.+)_(P\d+)_([A-H]\d{2})", str(raw).strip())
        if not m:
            continue
        lbl, plate, well = m.groups()
        rna_map[(plate, well)] = (j, lbl.strip())

    # 2. Parse Function Table S2 (04H and 24H) for ALL 13 kinetic endpoints
    func_xlsx = DS / "GSE314310/Cardiopedia_Table_S2_normalized_function.xlsx"
    raw_entries = defaultdict(list)
    for tm, sheet in [("04H", "Tab 1 Functional data 04H"), ("24H", "Tab 2 Functional data 24H")]:
        rows = iter_xlsx_rows(func_xlsx, sheet)
        hh = next(rows)
        ix = {str(v).strip(): i for i, v in enumerate(hh) if v is not None}
        for row in rows:
            ds_val = str(row[ix["Dataset"]]).strip() if ix["Dataset"] < len(row) and row[ix["Dataset"]] is not None else ""
            if ds_val not in ("Cardiopedia-Ligand", "Independent_Experiment"):
                continue
            lbl = str(row[ix["Treatment"]]).strip() if ix["Treatment"] < len(row) and row[ix["Treatment"]] is not None else ""
            well = str(row[ix["Wellname"]]).strip() if ix["Wellname"] < len(row) and row[ix["Wellname"]] is not None else ""
            pt = str(row[ix["Plate_timepoint"]]).strip() if ix["Plate_timepoint"] < len(row) and row[ix["Plate_timepoint"]] is not None else ""
            if not (lbl and well and pt):
                continue
            plate = pt.split("_", 1)[0]
            if pt != f"{plate}_{tm}":
                continue
            if ds_val == "Cardiopedia-Ligand" and (plate not in PLATES or lbl not in label_set):
                continue
            if ds_val == "Independent_Experiment" and (plate not in REPEAT_PLATES or lbl not in label_set):
                continue
            ep_vals = {ep: to_float(row[ix[ep]]) if ep in ix and ix[ep] < len(row) else None for ep in KINETIC_ENDPOINTS}
            raw_entries[(ds_val, tm, plate, well)].append((lbl, ep_vals))

    # Strict deduplication (drop duplicate keys just like V5)
    main_sel = {}
    rep_sel = {}
    for (ds_val, tm, plate, well), entries in raw_entries.items():
        if len(entries) != 1:
            continue
        lbl, ep_vals = entries[0]
        if ds_val == "Cardiopedia-Ligand":
            if (plate, well) not in rna_map or rna_map[(plate, well)][1] != lbl:
                continue
            main_sel[(tm, plate, lbl)] = (rna_map[(plate, well)][0], well, ep_vals)
        else:
            rep_sel[(tm, plate, lbl)] = (well, ep_vals)

    main_keys = [(p, l) for p in PLATES for l in labels]
    assert all(("24H", p, l) in main_sel for p, l in main_keys), "Missing 24H main cells"
    col_indices = [main_sel[("24H", p, l)][0] for p, l in main_keys]

    # Stream RNA counts for these 235 columns
    gene_ids, gene_syms, gene_bio = [], [], []
    mat_rows = []
    for row in it:
        gid = row[0]
        if not gid:
            continue
        gene_ids.append(str(gid).strip())
        gene_syms.append(str(row[1]).strip() if len(row) > 1 and row[1] is not None else "")
        gene_bio.append(str(row[2]).strip() if len(row) > 2 and row[2] is not None else "")
        mat_rows.append([to_float(row[c]) if c < len(row) else np.nan for c in col_indices])
    counts_235 = np.asarray(mat_rows, dtype=float).T  # (235, n_genes)

    # Assemble 5x47 endpoint arrays for 04H and 24H
    endpoints_04h = {}
    endpoints_24h = {}
    for ep in KINETIC_ENDPOINTS:
        endpoints_04h[ep] = np.array(
            [main_sel[("04H", p, l)][2][ep] if ("04H", p, l) in main_sel and main_sel[("04H", p, l)][2][ep] is not None else np.nan for p, l in main_keys],
            dtype=float,
        ).reshape(5, 47)
        endpoints_24h[ep] = np.array(
            [main_sel[("24H", p, l)][2][ep] if main_sel[("24H", p, l)][2][ep] is not None else np.nan for p, l in main_keys],
            dtype=float,
        ).reshape(5, 47)

    # Repeat complete-pair endpoints (require both 04H and 24H on same plate+well+label)
    repeat_24h_by_ep = {ep: {l: [] for l in labels} for ep in KINETIC_ENDPOINTS}
    complete_rep_count = 0
    for p in REPEAT_PLATES:
        for l in labels:
            a = rep_sel.get(("04H", p, l))
            b = rep_sel.get(("24H", p, l))
            if a is not None and b is not None and a[0] == b[0]:
                complete_rep_count += 1
                for ep in KINETIC_ENDPOINTS:
                    if b[1][ep] is not None:
                        repeat_24h_by_ep[ep][l].append(b[1][ep])

    return {
        "labels": labels,
        "gene_ids": gene_ids,
        "gene_syms": gene_syms,
        "gene_bio": gene_bio,
        "counts_235": counts_235,
        "endpoints_04h": endpoints_04h,
        "endpoints_24h": endpoints_24h,
        "repeat_24h_by_ep": repeat_24h_by_ep,
        "complete_rep_count": complete_rep_count,
    }


def resolve_strict_symbols(gene_syms: list[str], target_syms: list[str]) -> list[int | None]:
    by_sym = defaultdict(list)
    for idx, s in enumerate(gene_syms):
        if s:
            by_sym[s].append(idx)
    return [by_sym[s][0] if len(by_sym.get(s, [])) == 1 else None for s in target_syms]


def load_fixed_programs():
    gmt_path = V5_ROOT / "theory/fixed_programs_2026-11-17/fixed_programs.gmt"
    progs = {}
    for line in gmt_path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split("\t")
        if len(parts) >= 3:
            genes = [g for g in parts[2:] if g]
            progs[parts[0]] = genes
    return progs


def camera_vif_analysis(expr_trt_mean: np.ndarray, gene_indices: list[int | None], weights: np.ndarray, y_trt: np.ndarray):
    """
    CAMERA-inspired correlation-aware competitive gene-set inference (Wu & Smyth, NAR 2012):
    For each mapped gene g, compute gene-level moderated/rank statistic z_g = atanh(rho(x_g, y)) * sqrt(n - 3),
    sign-aligned by sign(w_g) (or weighted by w_g), and compute the actual mean pairwise inter-gene correlation
    rho_bar among the signature genes (using exact algebraic identity Var(sum z_g) without O(m^2) memory!)
    and Variance Inflation Factor VIF = 1 + (m - 1) * rho_bar.
    """
    n_trt = expr_trt_mean.shape[0]
    valid = [(idx, w) for idx, w in zip(gene_indices, weights) if idx is not None and math.isfinite(w) and w != 0]
    idxs = [idx for idx, _ in valid]
    ws = np.array([w for _, w in valid], float)

    # Extract treatment-level gene matrix (47 x m)
    G = expr_trt_mean[:, idxs]
    sd = G.std(axis=0, ddof=1)
    keep = np.isfinite(sd) & (sd > 1e-12)
    G = G[:, keep]
    ws = ws[keep]
    m = G.shape[1]

    # Standardize each gene across 47 treatments -> Z (47 x m)
    Z = (G - G.mean(axis=0, keepdims=True)) / G.std(axis=0, ddof=1, keepdims=True)
    # Also sign-align columns by sign(w) so positive correlation among aligned genes inflates the weighted score!
    s_w = np.sign(ws)
    Z_aligned = Z * s_w[None, :]

    # Exact average off-diagonal correlation of sign-aligned genes using O(n * m) identity:
    # sum_{i,j} corr(i,j) = ||sum_j Z_j||^2 / (n - 1)
    sum_vec = Z_aligned.sum(axis=1)
    total_corr_sum = float(np.dot(sum_vec, sum_vec) / (n_trt - 1))
    rho_bar_aligned = (total_corr_sum - m) / (m * (m - 1))
    vif_aligned = 1.0 + (m - 1) * rho_bar_aligned

    # Weighted VIF for general weights w:
    # Var(sum w_j Z_j) = sum w_j^2 + sum_{i!=j} w_i w_j corr(i,j) = ||Z w||^2 / (n - 1)
    zw = Z @ ws
    var_actual = float(np.dot(zw, zw) / (n_trt - 1))
    var_indep = float(np.dot(ws, ws))
    vif_weighted = var_actual / var_indep
    eff_n_genes = (np.sum(np.abs(ws)) ** 2) / var_actual if var_actual > 0 else float("nan")

    # Compute gene-level Spearman correlations with y_trt across all usable signature genes
    ry = stats.rankdata(y_trt)
    ry_z = (ry - ry.mean()) / ry.std(ddof=1)
    RG = np.column_stack([stats.rankdata(G[:, j]) for j in range(m)])
    RG_z = (RG - RG.mean(axis=0, keepdims=True)) / RG.std(axis=0, ddof=1, keepdims=True)
    gene_rhos = (ry_z @ RG_z) / (n_trt - 1)  # Spearman rho of each gene with 24h Force

    # Sign-aligned gene stats (negative means HF-up genes go down with higher force -> consistent with negative score-force rho)
    aligned_rhos = gene_rhos * s_w
    mean_aligned_rho = float(np.mean(aligned_rhos))
    w_mean_rho = float(np.sum(ws * gene_rhos) / np.sum(np.abs(ws)))
    # Naive SE vs CAMERA-VIF-adjusted SE across genes
    s_gene = float(np.std(aligned_rhos, ddof=1))
    se_naive = s_gene / math.sqrt(m)
    se_vif = s_gene * math.sqrt(max(vif_aligned, 1.0) / m)
    t_naive = mean_aligned_rho / se_naive
    t_vif = mean_aligned_rho / se_vif
    # Degrees of freedom bounded conservatively by n_trt - 2 (CAMERA style)
    df_cam = n_trt - 2
    p_naive = float(2.0 * stats.t.sf(abs(t_naive), m - 1))
    p_vif = float(2.0 * stats.t.sf(abs(t_vif), df_cam))

    # Also compute background mean inter-gene correlation (unsigned, raw Z)
    sum_raw = Z.sum(axis=1)
    rho_bar_raw = (float(np.dot(sum_raw, sum_raw) / (n_trt - 1)) - m) / (m * (m - 1))

    return {
        "n_genes": int(m),
        "rho_bar_raw_unsigned": float(rho_bar_raw),
        "rho_bar_sign_aligned": float(rho_bar_aligned),
        "vif_sign_aligned": float(vif_aligned),
        "vif_weighted": float(vif_weighted),
        "effective_independent_genes_weighted": float(eff_n_genes),
        "mean_sign_aligned_gene_rho": mean_aligned_rho,
        "weighted_mean_gene_rho": w_mean_rho,
        "t_naive_uncorrected": float(t_naive),
        "p_naive_uncorrected": p_naive,
        "t_camera_vif_adjusted": float(t_vif),
        "df_camera": int(df_cam),
        "p_camera_vif_adjusted": p_vif,
    }


def composition_shift_stress_test(log_expr_235: np.ndarray, force_235: np.ndarray, gene_indices, weights,
                                  gene_syms, ref_mu, ref_sd, sig_syms):
    """
    Direct computational proof of transductive distortion vs inductive invariance:
    Select a fixed probe panel of 15 intermediate-force treatments (5 plates each = 75 organoids)
    and score them under 3 batch contexts:
      Context A: Full 47-treatment atlas (235 organoids)
      Context B: Probe + 15 highest-force treatments (150 organoids — high-force skewed batch)
      Context C: Probe + 15 lowest-force treatments (150 organoids — low-force skewed batch)
    Measure for each scoring paradigm:
      1. Max absolute score shift on identical probe samples between Context B and Context C
      2. Mean absolute score shift (in SD units of Context A)
      3. Rank correlation of the 15 probe treatments between Context B and Context C
      4. Fraction of probe treatments whose sign flips between Context B and Context C!
    """
    expr_3d = log_expr_235.reshape(5, 47, -1)
    force_mean = force_235.reshape(5, 47).mean(axis=0)
    order = np.argsort(force_mean)
    low_idx = order[:15]
    mid_idx = order[16:31]  # 15 fixed probe treatments
    high_idx = order[-15:]

    contexts = {
        "full_47": np.arange(47),
        "high_skew_30": np.concatenate([mid_idx, high_idx]),
        "low_skew_30": np.concatenate([mid_idx, low_idx]),
    }

    def eval_paradigm(name, fn):
        scores_by_ctx = {}
        for cname, cidx in contexts.items():
            sub = expr_3d[:, cidx, :].reshape(-1, expr_3d.shape[2])
            s_flat = fn(sub).reshape(5, len(cidx))
            # Extract scores strictly for the 15 probe treatments (positions 0..14 in skew, or mid_idx in full)
            if cname == "full_47":
                scores_by_ctx[cname] = s_flat[:, mid_idx].mean(axis=0)
            else:
                scores_by_ctx[cname] = s_flat[:, :15].mean(axis=0)
        s_full = scores_by_ctx["full_47"]
        s_hi = scores_by_ctx["high_skew_30"]
        s_lo = scores_by_ctx["low_skew_30"]
        sd_full = float(np.std(s_full, ddof=1))
        diff = np.abs(s_hi - s_lo)
        sign_flips = int(np.sum(np.sign(s_hi) != np.sign(s_lo)))
        return {
            "paradigm": name,
            "max_abs_shift_hi_vs_lo": float(diff.max()),
            "mean_abs_shift_hi_vs_lo": float(diff.mean()),
            "mean_shift_in_probe_sd": float(diff.mean() / sd_full) if sd_full > 0 else 0.0,
            "probe_rank_rho_hi_vs_lo": spearman_rho(s_hi, s_lo),
            "probe_rank_rho_full_vs_hi": spearman_rho(s_full, s_hi),
            "probe_rank_rho_full_vs_lo": spearman_rho(s_full, s_lo),
            "sign_flip_count_of_15": sign_flips,
            "sign_flip_pct": float(100.0 * sign_flips / 15.0),
        }

    return [
        eval_paradigm("Cohort-Z (transductive)", lambda X: score_cohort_z(X, gene_indices, weights)[0]),
        eval_paradigm("Single-Sample Rank (inductive)", lambda X: score_single_sample_rank(X, gene_indices, weights)[0]),
        eval_paradigm("Ref-Frozen-Z Adult-NF (inductive)", lambda X: score_ref_frozen_z(X, gene_syms, ref_mu, ref_sd, sig_syms, weights)[0]),
    ]


def main():
    print("[1/6] Loading Cardiopedia strict 235-organoid dataset + all 13 kinetic endpoints...")
    cp = load_cardiopedia_strict()
    counts_235 = cp["counts_235"]
    log_cpm_235 = log2_cpm(counts_235)
    log_deseq_235 = log2_deseq2_median_ratio(counts_235)
    labels = cp["labels"]
    gene_syms = cp["gene_syms"]
    gene_ids_base = [g.split(".")[0] for g in cp["gene_ids"]]

    print("[2/6] Loading ReHeaT2 consensus & official cell-type weights + Adult NF calibration reference...")
    mc = pd.read_csv(DS / "ReHeaT2_zenodo_15261569/mol_consistency.csv")
    mol_valid = mc.loc[mc["mol_score"].notna() & (mc["mol_score"] != 0)].copy()
    sig_syms = list(mol_valid["gene"].astype(str))
    sig_w = mol_valid["mol_score"].to_numpy(dtype=float)
    sig_idx = resolve_strict_symbols(gene_syms, sig_syms)

    # Exact V5 parity check!
    s_cohort_z_235, usable_mask = score_cohort_z(log_cpm_235, sig_idx, sig_w)
    s_cohort_z_5x47 = s_cohort_z_235.reshape(5, 47)
    v5_pt = pd.read_csv(V5_TABLES / "3d_plate_treatment_values.csv")
    v5_pivot = v5_pt.pivot(index="plate", columns="treatment", values="score").loc[list(PLATES), labels].to_numpy(dtype=float)
    v5_force_pivot = v5_pt.pivot(index="plate", columns="treatment", values="function_24h_normalized").loc[list(PLATES), labels].to_numpy(dtype=float)
    max_score_diff = float(np.max(np.abs(s_cohort_z_5x47 - v5_pivot)))
    max_force_diff = float(np.max(np.abs(cp["endpoints_24h"]["Force"] - v5_force_pivot)))
    print(f"  -> EXACT V5 PARITY CHECK: max |score diff| = {max_score_diff:.3e}, max |force diff| = {max_force_diff:.3e}")
    assert max_score_diff < 1e-12, f"V5 parity failed: {max_score_diff}"
    assert max_force_diff < 1e-12, f"V5 force parity failed: {max_force_diff}"

    # Build Adult NF frozen reference (GSE141910 NF n=166) mapped via Cardiopedia Ensembl->symbol table
    ens_to_sym_unique = {}
    ens_sym_cnt = defaultdict(int)
    for gid_b, sym in zip(gene_ids_base, gene_syms):
        if sym:
            ens_sym_cnt[sym] += 1
            ens_to_sym_unique[gid_b] = sym
    nf_ens, nf_lmat, n_nf = load_adult_nf_reference()
    ref_mu, ref_sd = {}, {}
    for j, eid in enumerate(nf_ens):
        sym = ens_to_sym_unique.get(eid)
        if sym and ens_sym_cnt[sym] == 1:
            col = nf_lmat[:, j]
            sd = float(col.std(ddof=1))
            if sd > 1e-12:
                ref_mu[sym] = float(col.mean())
                ref_sd[sym] = sd
    print(f"  -> Adult NF reference (GSE141910 NF, n={n_nf}): {len(ref_mu)} calibrated unique genes")

    print("[3/6] Computing 4-paradigm scoring benchmark & composition-shift stress test...")
    s_plate_z_5x47 = score_within_plate_z(log_cpm_235.reshape(5, 47, -1), sig_idx, sig_w)
    s_deseq_z_5x47 = score_cohort_z(log_deseq_235, sig_idx, sig_w)[0].reshape(5, 47)
    s_ss_rank_235, n_rank_genes = score_single_sample_rank(log_cpm_235, sig_idx, sig_w)
    s_ss_rank_5x47 = s_ss_rank_235.reshape(5, 47)
    s_ss_rank_deseq_5x47 = score_single_sample_rank(log_deseq_235, sig_idx, sig_w)[0].reshape(5, 47)
    s_ref_z_235, n_ref_genes = score_ref_frozen_z(log_cpm_235, gene_syms, ref_mu, ref_sd, sig_syms, sig_w)
    s_ref_z_5x47 = s_ref_z_235.reshape(5, 47)

    force_24 = cp["endpoints_24h"]["Force"]
    force_04 = cp["endpoints_04h"]["Force"]
    rep_force_mean = np.array([float(np.mean(cp["repeat_24h_by_ep"]["Force"][l])) for l in labels])

    paradigms = {
        "Cohort-Z (log2-CPM, transductive)": s_cohort_z_5x47,
        "Cohort-Z (DESeq2-MR, transductive)": s_deseq_z_5x47,
        "Within-Plate-Z (log2-CPM, transductive)": s_plate_z_5x47,
        "Single-Sample Rank (inductive)": s_ss_rank_5x47,
        "Single-Sample Rank (DESeq2-MR, inductive)": s_ss_rank_deseq_5x47,
        "Ref-Frozen-Z Adult-NF (inductive)": s_ref_z_5x47,
    }
    paradigm_rows = []
    for pname, smat in paradigms.items():
        sm = smat.mean(axis=0)
        fm = force_24.mean(axis=0)
        f04m = force_04.mean(axis=0)
        ci = bootstrap_spearman_ci(sm, fm)
        plate_rhos = [spearman_rho(smat[i], force_24[i]) for i in range(5)]
        lopo_rhos = [spearman_rho(np.delete(smat, i, 0).mean(axis=0), np.delete(force_24, i, 0).mean(axis=0)) for i in range(5)]
        # Within-treatment centered rho
        sc = (smat - smat.mean(axis=0, keepdims=True)).ravel()
        fc = (force_24 - force_24.mean(axis=0, keepdims=True)).ravel()
        within_rho = spearman_rho(sc, fc)
        # 24h | 04h residual partial correlation
        part_04 = partial_spearman(sm, fm, f04m)
        # Concordance with Cohort-Z
        rho_vs_cz = spearman_rho(sm, s_cohort_z_5x47.mean(axis=0))
        # Repeat experiment rho
        rep_ci = bootstrap_spearman_ci(sm, rep_force_mean)
        paradigm_rows.append({
            "paradigm": pname,
            "is_inductive": "inductive" in pname,
            "between_rho_24h_force": ci["rho"],
            "ci_low": ci["ci_low"],
            "ci_high": ci["ci_high"],
            "plate_rho_min": float(min(plate_rhos)),
            "plate_rho_max": float(max(plate_rhos)),
            "lopo_min": float(min(lopo_rhos)),
            "lopo_max": float(max(lopo_rhos)),
            "within_treatment_rho": within_rho,
            "partial_rho_24h_given_04h": part_04["partial_rho"],
            "partial_p_24h_given_04h": part_04["p_two_sided"],
            "repeat_rho_24h_force": rep_ci["rho"],
            "repeat_ci_low": rep_ci["ci_low"],
            "repeat_ci_high": rep_ci["ci_high"],
            "rho_vs_cohort_z": rho_vs_cz,
        })
    df_paradigms = pd.DataFrame(paradigm_rows)
    df_paradigms.to_csv(OUT_DIR / "v6_table1_3d_scoring_paradigms.csv", index=False)
    print(df_paradigms[["paradigm", "between_rho_24h_force", "ci_low", "ci_high", "within_treatment_rho", "partial_rho_24h_given_04h", "repeat_rho_24h_force"]].to_string(index=False))

    stress = composition_shift_stress_test(log_cpm_235, force_24.ravel(), sig_idx, sig_w, gene_syms, ref_mu, ref_sd, sig_syms)
    df_stress = pd.DataFrame(stress)
    df_stress.to_csv(OUT_DIR / "v6_table2_composition_shift_stress.csv", index=False)
    print("\n=== Composition-Shift Stress Test (Transductive vs Inductive) ===")
    print(df_stress.to_string(index=False))

    print("\n[4/6] Dissecting all 13 biomechanical/kinetic endpoints at 04H and 24H...")
    kinetic_rows = []
    for ep in KINETIC_ENDPOINTS:
        y24 = cp["endpoints_24h"][ep]
        y04 = cp["endpoints_04h"][ep]
        y24m = np.nanmean(y24, axis=0)
        y04m = np.nanmean(y04, axis=0)
        rep_vals = np.array([float(np.mean(cp["repeat_24h_by_ep"][ep][l])) if cp["repeat_24h_by_ep"][ep][l] else np.nan for l in labels])
        for pname, smat in [
            ("Cohort-Z", s_cohort_z_5x47),
            ("Single-Sample Rank", s_ss_rank_5x47),
            ("Ref-Frozen-Z", s_ref_z_5x47),
        ]:
            sm = smat.mean(axis=0)
            ci24 = bootstrap_spearman_ci(sm, y24m)
            ci04 = bootstrap_spearman_ci(sm, y04m)
            part24_04 = partial_spearman(sm, y24m, y04m)
            part24_force = partial_spearman(sm, y24m, force_24.mean(axis=0)) if ep != "Force" else {"partial_rho": float("nan"), "p_two_sided": float("nan")}
            rep_rho = spearman_rho(sm, rep_vals)
            plate_rhos = [spearman_rho(smat[i], y24[i]) for i in range(5)]
            kinetic_rows.append({
                "endpoint": ep,
                "paradigm": pname,
                "rho_24h": ci24["rho"],
                "ci24_low": ci24["ci_low"],
                "ci24_high": ci24["ci_high"],
                "rho_04h": ci04["rho"],
                "partial_rho_24h_given_04h": part24_04["partial_rho"],
                "partial_p_24h_given_04h": part24_04["p_two_sided"],
                "partial_rho_24h_given_force": part24_force["partial_rho"],
                "plate_min": float(np.nanmin(plate_rhos)),
                "plate_max": float(np.nanmax(plate_rhos)),
                "repeat_rho_24h": rep_rho,
            })
    df_kinetics = pd.DataFrame(kinetic_rows)
    df_kinetics.to_csv(OUT_DIR / "v6_table3_cardiopedia_13_kinetics.csv", index=False)
    print(df_kinetics[df_kinetics["paradigm"] == "Cohort-Z"][["endpoint", "rho_24h", "ci24_low", "ci24_high", "rho_04h", "partial_rho_24h_given_04h", "repeat_rho_24h"]].to_string(index=False))

    print("\n[5/6] Evaluating official ReHeaT2 cell-type axes + fixed comparator programs + conditional partial models...")
    axis_specs = [
        ("ReHeaT2_mol_score", "mol_score", None),
        ("ReHeaT2_mean_t_all16036", "mean_t", None),
        ("ReHeaT2_consistent_yes", "mol_score", mc["mol_consistent"] == "yes"),
        ("ReHeaT2_CM", "mcell_HFscore_CM", None),
        ("ReHeaT2_Fib", "mcell_HFscore_Fib", None),
        ("ReHeaT2_Endo", "mcell_HFscore_Endo", None),
        ("ReHeaT2_Myeloid", "mcell_HFscore_Myeloid", None),
        ("ReHeaT2_Lymphoid", "mcell_HFscore_Lymphoid", None),
        ("ReHeaT2_PC", "mcell_HFscore_PC", None),
        ("ReHeaT2_vSMCs", "mcell_HFscore_vSMCs", None),
    ]
    axis_rows = []
    axis_scores_cz = {}
    axis_scores_rank = {}
    for aname, col, mask_cond in axis_specs:
        sub = mc.loc[mc[col].notna() & (mc[col] != 0)] if mask_cond is None else mc.loc[mask_cond & mc[col].notna() & (mc[col] != 0)]
        asym = list(sub["gene"].astype(str))
        aw = sub[col].to_numpy(dtype=float)
        aidx = resolve_strict_symbols(gene_syms, asym)
        scz, ausable = score_cohort_z(log_cpm_235, aidx, aw)
        scz_5 = scz.reshape(5, 47)
        srank, n_rk = score_single_sample_rank(log_cpm_235, aidx, aw)
        srank_5 = srank.reshape(5, 47)
        sref, n_rf = score_ref_frozen_z(log_cpm_235, gene_syms, ref_mu, ref_sd, asym, aw)
        sref_5 = sref.reshape(5, 47)
        axis_scores_cz[aname] = scz_5.mean(axis=0)
        axis_scores_rank[aname] = srank_5.mean(axis=0)

        ci_cz = bootstrap_spearman_ci(scz_5.mean(axis=0), force_24.mean(axis=0))
        ci_rk = bootstrap_spearman_ci(srank_5.mean(axis=0), force_24.mean(axis=0))
        ci_rf = bootstrap_spearman_ci(sref_5.mean(axis=0), force_24.mean(axis=0))
        cov_w = float(np.abs(aw[ausable]).sum() / np.abs(aw).sum())
        axis_rows.append({
            "axis": aname,
            "n_defined_genes": int(len(aw)),
            "n_usable_genes": int(ausable.sum()),
            "weight_coverage": cov_w,
            "cohort_z_rho_force": ci_cz["rho"],
            "cohort_z_ci_low": ci_cz["ci_low"],
            "cohort_z_ci_high": ci_cz["ci_high"],
            "ss_rank_rho_force": ci_rk["rho"],
            "ss_rank_ci_low": ci_rk["ci_low"],
            "ss_rank_ci_high": ci_rk["ci_high"],
            "ref_z_rho_force": ci_rf["rho"],
            "cohort_z_rho_bpm": spearman_rho(scz_5.mean(axis=0), cp["endpoints_24h"]["BPM"].mean(axis=0)),
            "cohort_z_rho_tr50": spearman_rho(scz_5.mean(axis=0), cp["endpoints_24h"]["Tr_50"].mean(axis=0)),
            "cohort_z_rho_cvel": spearman_rho(scz_5.mean(axis=0), cp["endpoints_24h"]["Contract_velocity_norm"].mean(axis=0)),
            "repeat_rho_force": spearman_rho(scz_5.mean(axis=0), rep_force_mean),
        })
    df_axes = pd.DataFrame(axis_rows)
    df_axes.to_csv(OUT_DIR / "v6_table4_reheat2_celltype_axes_3d.csv", index=False)
    print(df_axes[["axis", "n_usable_genes", "cohort_z_rho_force", "ss_rank_rho_force", "ref_z_rho_force", "cohort_z_rho_cvel", "repeat_rho_force"]].to_string(index=False))

    # Fixed comparator programs (all 9) under Cohort-Z and Single-Sample Rank + global L2 perturbation burden
    fixed_progs = load_fixed_programs()
    prog_scores_cz = {}
    prog_scores_rk = {}
    prog_rows = []
    for pname, genes in fixed_progs.items():
        pidx = resolve_strict_symbols(gene_syms, genes)
        pw = np.ones(len(genes), float)
        scz, pus = score_cohort_z(log_cpm_235, pidx, pw)
        srk, _ = score_single_sample_rank(log_cpm_235, pidx, pw)
        prog_scores_cz[pname] = scz.reshape(5, 47).mean(axis=0)
        prog_scores_rk[pname] = srk.reshape(5, 47).mean(axis=0)
        prog_rows.append({
            "program": pname,
            "n_usable": int(pus.sum()),
            "cohort_z_rho_force": spearman_rho(prog_scores_cz[pname], force_24.mean(axis=0)),
            "ss_rank_rho_force": spearman_rho(prog_scores_rk[pname], force_24.mean(axis=0)),
            "cohort_z_rho_cvel": spearman_rho(prog_scores_cz[pname], cp["endpoints_24h"]["Contract_velocity_norm"].mean(axis=0)),
            "cohort_z_rho_bpm": spearman_rho(prog_scores_cz[pname], cp["endpoints_24h"]["BPM"].mean(axis=0)),
            "cohort_z_rho_tr50": spearman_rho(prog_scores_cz[pname], cp["endpoints_24h"]["Tr_50"].mean(axis=0)),
        })
    df_progs = pd.DataFrame(prog_rows)
    df_progs.to_csv(OUT_DIR / "v6_table5_fixed_programs_3d_multikinetic.csv", index=False)

    # Global transcriptional perturbation burden (L2 norm of treatment mean z-expression across expressed genes)
    trt_expr_mean = log_cpm_235.reshape(5, 47, -1).mean(axis=0)  # (47, n_genes)
    gene_sd = trt_expr_mean.std(axis=0, ddof=1)
    expr_mask = gene_sd > 1e-6
    Z_all = (trt_expr_mean[:, expr_mask] - trt_expr_mean[:, expr_mask].mean(axis=0, keepdims=True)) / gene_sd[expr_mask][None, :]
    l2_burden = np.sqrt(np.mean(Z_all ** 2, axis=1))
    print(f"\n  -> Global transcriptional L2 perturbation burden vs 24h Force: rho = {spearman_rho(l2_burden, force_24.mean(axis=0)):+.3f}")

    # Conditional / Partial Spearman models controlling for confounders
    fm = force_24.mean(axis=0)
    cond_models = []
    for score_label, svec in [
        ("Cohort-Z (mol_score)", s_cohort_z_5x47.mean(axis=0)),
        ("Single-Sample Rank (mol_score)", s_ss_rank_5x47.mean(axis=0)),
        ("Ref-Frozen-Z (mol_score)", s_ref_z_5x47.mean(axis=0)),
        ("Cohort-Z (ReHeaT2_CM)", axis_scores_cz["ReHeaT2_CM"]),
        ("Cohort-Z (ReHeaT2_Fib)", axis_scores_cz["ReHeaT2_Fib"]),
    ]:
        cov_sets = [
            ("Unadjusted", None),
            ("+ Global L2 perturbation burden", l2_burden[:, None]),
            ("+ Hypoxia", prog_scores_cz["hypoxia"][:, None]),
            ("+ Structural_exclusive", prog_scores_cz["structural_exclusive"][:, None]),
            ("+ Fibroblast_marker + ECM", np.column_stack([prog_scores_cz["fibroblast_marker"], prog_scores_cz["ecm"]])),
            ("+ UPR + OXPHOS", np.column_stack([prog_scores_cz["upr"], prog_scores_cz["oxphos"]])),
            ("+ All 7 fixed programs + L2 burden", np.column_stack([
                l2_burden, prog_scores_cz["hypoxia"], prog_scores_cz["structural_exclusive"],
                prog_scores_cz["conduction_exclusive"], prog_scores_cz["ecm"],
                prog_scores_cz["oxphos"], prog_scores_cz["upr"], prog_scores_cz["fibroblast_marker"],
            ])),
            ("+ 04h Force baseline", force_04.mean(axis=0)[:, None]),
        ]
        for cname, C in cov_sets:
            if C is None:
                ci = bootstrap_spearman_ci(svec, fm)
                r_val, p_val = ci["rho"], float(stats.spearmanr(svec, fm).pvalue)
            else:
                res = partial_spearman(svec, fm, C)
                r_val, p_val = res["partial_rho"], res["p_two_sided"]
            cond_models.append({
                "score": score_label,
                "adjustment": cname,
                "rho_or_partial_rho": r_val,
                "p_two_sided": p_val,
            })
    df_cond = pd.DataFrame(cond_models)
    df_cond.to_csv(OUT_DIR / "v6_table6_conditional_partial_models_3d.csv", index=False)
    print("\n=== Conditional / Partial Association Models (24h Force) ===")
    print(df_cond[df_cond["score"].isin(["Cohort-Z (mol_score)", "Single-Sample Rank (mol_score)", "Cohort-Z (ReHeaT2_CM)"])].to_string(index=False))

    # Also test plate-fixed-effects linear regression on all 235 organoids (with plate dummies + cluster-robust SE by treatment)
    # Using rank-transformed force and score within-study
    print("\n[6/6] Running CAMERA-style correlation-aware VIF inference...")
    cam_main = camera_vif_analysis(trt_expr_mean, sig_idx, sig_w, fm)
    cm_sub = mc.loc[mc["mcell_HFscore_CM"].notna() & (mc["mcell_HFscore_CM"] != 0)]
    cam_cm = camera_vif_analysis(trt_expr_mean, resolve_strict_symbols(gene_syms, list(cm_sub["gene"])), cm_sub["mcell_HFscore_CM"].to_numpy(float), fm)
    fib_sub = mc.loc[mc["mcell_HFscore_Fib"].notna() & (mc["mcell_HFscore_Fib"] != 0)]
    cam_fib = camera_vif_analysis(trt_expr_mean, resolve_strict_symbols(gene_syms, list(fib_sub["gene"])), fib_sub["mcell_HFscore_Fib"].to_numpy(float), fm)

    cam_summary = {"ReHeaT2_mol_score": cam_main, "ReHeaT2_CM": cam_cm, "ReHeaT2_Fib": cam_fib}
    print(json.dumps(cam_summary, indent=2))

    # Save treatment-level master CSV for plotting & verification
    trt_df = pd.DataFrame({
        "treatment": labels,
        "force_24h": fm,
        "force_04h": force_04.mean(axis=0),
        "bpm_24h": cp["endpoints_24h"]["BPM"].mean(axis=0),
        "ta50_24h": cp["endpoints_24h"]["Ta_50"].mean(axis=0),
        "tr50_24h": cp["endpoints_24h"]["Tr_50"].mean(axis=0),
        "cvel_24h": cp["endpoints_24h"]["Contract_velocity_norm"].mean(axis=0),
        "rvel_24h": cp["endpoints_24h"]["Relax_velocity_norm"].mean(axis=0),
        "repeat_force_24h": rep_force_mean,
        "score_cohort_z": s_cohort_z_5x47.mean(axis=0),
        "score_deseq_z": s_deseq_z_5x47.mean(axis=0),
        "score_plate_z": s_plate_z_5x47.mean(axis=0),
        "score_ss_rank": s_ss_rank_5x47.mean(axis=0),
        "score_ref_frozen_z": s_ref_z_5x47.mean(axis=0),
        "score_cm_cz": axis_scores_cz["ReHeaT2_CM"],
        "score_fib_cz": axis_scores_cz["ReHeaT2_Fib"],
        "l2_burden": l2_burden,
    })
    for pname, vec in prog_scores_cz.items():
        trt_df[f"prog_{pname}"] = vec
    trt_df.to_csv(OUT_DIR / "v6_cardiopedia_47_treatments_master.csv", index=False)

    summary = {
        "v5_parity": {
            "max_abs_score_diff": max_score_diff,
            "max_abs_force_diff": max_force_diff,
            "n_usable_genes_main": int(usable_mask.sum()),
            "n_rank_genes_main": int(n_rank_genes),
            "n_ref_frozen_genes_main": int(n_ref_genes),
            "n_adult_nf_ref_samples": int(n_nf),
            "complete_repeat_pairs": int(cp["complete_rep_count"]),
        },
        "l2_burden_rho_force": spearman_rho(l2_burden, fm),
        "camera_vif": cam_summary,
    }
    (OUT_DIR / "v6_part1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nPart 1 completed cleanly. Outputs saved to", OUT_DIR)


if __name__ == "__main__":
    main()
