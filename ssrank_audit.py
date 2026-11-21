"""
ssrank_audit (V7.1): Reusable Inductive Single-Sample Transcriptional Scoring & Descriptive O(nm) VIF Audit Module.

Dependencies: Python >= 3.10, NumPy, SciPy (benchmarking scripts separately use pandas, scikit-learn, gseapy).
Public API:
  - score_ss_rank_internal(expr_2d, sig_indices, weights, allow_missing_per_sample=False)
  - score_ss_rank_full(expr_2d, sig_indices, weights, allow_missing_per_sample=False)
  - score_singscore_internal(expr_2d, sig_indices, weights)
  - score_singscore_full(expr_2d, sig_indices, weights)
  - score_ssgsea_bidirectional(expr_2d, sig_indices, weights, alpha=0.25)        # Tie-averaged bidirectional Python variant
  - score_gsva_bidirectional(expr_2d, sig_indices, weights, kcdf='Gaussian', tau=1.0)
      kcdf options: 'Gaussian' (kernel CDF), 'midrank' (alias 'none': (avg_rank-0.5)/N),
      or 'ecdf' (standard right-continuous empirical CDF rank_max/N)
  - signed_weight_imbalance(weights)
  - descriptive_vif_audit(trt_expr_2d, sig_indices, weights)                     # Exact O(nm) descriptive VIF & variance-equivalent dimension (NOT a null-hypothesis p-value)
"""

from __future__ import annotations
import math
from typing import Sequence
import numpy as np
from scipy import special, stats


def _clean_pairs(
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    n_genes: int | None = None,
):
    """
    Strictly validate and clean (gene_index, weight) pairs:
      - Rejects mismatched sequence lengths (no silent zip truncation).
      - Rejects negative or out-of-range gene indices when n_genes is given.
      - Rejects duplicate gene indices (preventing silent sign conflicts or double counting).
      - Excludes None indices and non-finite/zero weights without zero-imputation.
    """
    if len(sig_indices) != len(weights):
        raise ValueError(
            f"sig_indices length ({len(sig_indices)}) != weights length ({len(weights)})."
        )
    pairs = []
    seen: set[int] = set()
    for idx, w in zip(sig_indices, weights):
        if idx is None:
            continue
        if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
            raise ValueError(f"Gene index must be an integer or None; got {idx!r}.")
        ii = int(idx)
        if ii < 0 or (n_genes is not None and ii >= n_genes):
            raise ValueError(f"Gene index {ii} out of valid range [0, {n_genes}).")
        if ii in seen:
            raise ValueError(f"Duplicate gene index {ii} in signature specification.")
        seen.add(ii)
        wf = float(w)
        if math.isfinite(wf) and wf != 0.0:
            pairs.append((ii, wf))
    if not pairs:
        raise ValueError("No finite nonzero mapped signature genes available.")
    cols = np.array([p[0] for p in pairs], dtype=int)
    w_arr = np.array([p[1] for p in pairs], dtype=float)
    return cols, w_arr


def signed_weight_imbalance(weights: Sequence[float]) -> dict[str, float | bool | int]:
    """
    Compute algebraic sign-imbalance index beta = |sum(w)| / sum(|w|) in [0, 1].
    For full-transcriptome SS-Rank under a uniform additive rank shift c, |Delta S| = |c| * beta.
    """
    w = np.asarray(
        [float(x) for x in weights if x is not None and math.isfinite(float(x)) and float(x) != 0.0],
        float,
    )
    if w.size == 0:
        raise ValueError("No finite nonzero weights supplied.")
    beta = float(abs(w.sum()) / np.abs(w).sum())
    return {
        "n_finite_nonzero": int(w.size),
        "n_positive": int(np.sum(w > 0)),
        "n_negative": int(np.sum(w < 0)),
        "imbalance_beta": beta,
        "offset_cancellation_safe": bool(beta < 0.25),
    }


def score_ss_rank_internal(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    allow_missing_per_sample: bool = False,
) -> np.ndarray:
    """
    Inductive Signature-Internal Weighted Fractional Rank (SS-Rank-Internal).
    Ranks strictly within mapped bidirectional signature genes S per sample independently:
      (P1) Exact sample-independence: depends exclusively on x_{i, S} (zero transductive drift).
      (P2) Exact real-arithmetic invariance to strictly increasing per-sample transformations
           (subject to floating-point tie preservation in finite precision).
      (P3) Exact background-universe invariance: columns outside S are never accessed.
    Domain boundary: unsigned constant-weight sets (w_k == c > 0) evaluate identically to 0.0
    because centered internal ranks sum to 0; unsigned sets must use score_ss_rank_full().
    """
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2:
        raise ValueError("expr_2d must be a 2D array (n_samples x n_genes).")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    sub = X[:, cols]
    out = np.empty(X.shape[0], dtype=float)
    if not allow_missing_per_sample:
        if not np.all(np.isfinite(sub)):
            raise ValueError(
                "Mapped signature columns contain non-finite values; pass allow_missing_per_sample=True "
                "to drop per-sample missing probes."
            )
        m = len(cols)
        denom = float(np.abs(w).sum())
        for i in range(X.shape[0]):
            r = (stats.rankdata(sub[i], method="average") - 0.5) / m - 0.5
            out[i] = float(np.dot(r, w) / denom)
    else:
        for i in range(X.shape[0]):
            ok = np.isfinite(sub[i])
            mi = int(ok.sum())
            if mi < 2:
                out[i] = float("nan")
                continue
            wi = w[ok]
            r = (stats.rankdata(sub[i, ok], method="average") - 0.5) / mi - 0.5
            out[i] = float(np.dot(r, wi) / np.abs(wi).sum())
    return out


def score_ss_rank_full(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    allow_missing_per_sample: bool = False,
) -> np.ndarray:
    """Inductive full-transcriptome weighted fractional rank (SS-Rank)."""
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2:
        raise ValueError("expr_2d must be a 2D array.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    out = np.empty(X.shape[0], dtype=float)
    if not allow_missing_per_sample:
        if not np.all(np.isfinite(X)):
            raise ValueError("expr_2d must be a finite 2D array unless allow_missing_per_sample=True.")
        G = X.shape[1]
        denom = float(np.abs(w).sum())
        for i in range(X.shape[0]):
            r = (stats.rankdata(X[i], method="average") - 0.5) / G - 0.5
            out[i] = float(np.dot(r[cols], w) / denom)
    else:
        for i in range(X.shape[0]):
            row = X[i]
            ok_all = np.isfinite(row)
            Gi = int(ok_all.sum())
            ok_sig = ok_all[cols]
            if Gi < 2 or int(ok_sig.sum()) < 1:
                out[i] = float("nan")
                continue
            r_all = np.full(X.shape[1], np.nan, dtype=float)
            r_all[ok_all] = (stats.rankdata(row[ok_all], method="average") - 0.5) / Gi - 0.5
            wi = w[ok_sig]
            out[i] = float(np.dot(r_all[cols[ok_sig]], wi) / np.abs(wi).sum())
    return out


def score_singscore_internal(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    allow_missing_per_sample: bool = False,
) -> np.ndarray:
    """Unweighted bidirectional singscore variant restricted strictly to mapped signature genes S."""
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2:
        raise ValueError("expr_2d must be a 2D array.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    sub = X[:, cols]
    up = w > 0
    dn = w < 0
    if not (np.any(up) and np.any(dn)):
        raise ValueError("Bidirectional singscore requires both positive and negative weights.")
    out = np.empty(X.shape[0], dtype=float)
    if not allow_missing_per_sample:
        if not np.all(np.isfinite(sub)):
            raise ValueError("Mapped signature columns must be finite.")
        m = len(cols)
        m_up, m_dn = int(up.sum()), int(dn.sum())
        min_u, max_u = (m_up + 1.0) / 2.0, (2.0 * m - m_up + 1.0) / 2.0
        min_d, max_d = (m_dn + 1.0) / 2.0, (2.0 * m - m_dn + 1.0) / 2.0
        for i in range(X.shape[0]):
            r_asc = stats.rankdata(sub[i], method="average")
            r_desc = (m + 1.0) - r_asc
            out[i] = ((float(np.mean(r_asc[up])) - min_u) / (max_u - min_u) - 0.5) + (
                (float(np.mean(r_desc[dn])) - min_d) / (max_d - min_d) - 0.5
            )
    else:
        for i in range(X.shape[0]):
            ok = np.isfinite(sub[i])
            mi = int(ok.sum())
            up_i = up[ok]
            dn_i = dn[ok]
            m_up, m_dn = int(up_i.sum()), int(dn_i.sum())
            if m_up == 0 or m_dn == 0 or mi <= max(m_up, m_dn):
                out[i] = float("nan")
                continue
            min_u, max_u = (m_up + 1.0) / 2.0, (2.0 * mi - m_up + 1.0) / 2.0
            min_d, max_d = (m_dn + 1.0) / 2.0, (2.0 * mi - m_dn + 1.0) / 2.0
            r_asc = stats.rankdata(sub[i, ok], method="average")
            r_desc = (mi + 1.0) - r_asc
            out[i] = ((float(np.mean(r_asc[up_i])) - min_u) / (max_u - min_u) - 0.5) + (
                (float(np.mean(r_desc[dn_i])) - min_d) / (max_d - min_d) - 0.5
            )
    return out


def score_singscore_full(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
) -> np.ndarray:
    """Full-transcriptome unweighted bidirectional singscore Python implementation (Foroutan et al., 2018)."""
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("expr_2d must be a finite 2D array.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    up_cols = cols[w > 0]
    dn_cols = cols[w < 0]
    if len(up_cols) == 0 or len(dn_cols) == 0:
        raise ValueError("Bidirectional singscore requires both positive and negative weights.")
    G = X.shape[1]
    m_up, m_dn = len(up_cols), len(dn_cols)
    min_u, max_u = (m_up + 1.0) / 2.0, (2.0 * G - m_up + 1.0) / 2.0
    min_d, max_d = (m_dn + 1.0) / 2.0, (2.0 * G - m_dn + 1.0) / 2.0
    out = np.empty(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        r_asc = stats.rankdata(X[i], method="average")
        r_desc = (G + 1.0) - r_asc
        out[i] = ((float(np.mean(r_asc[up_cols])) - min_u) / (max_u - min_u) - 0.5) + (
            (float(np.mean(r_desc[dn_cols])) - min_d) / (max_d - min_d) - 0.5
        )
    return out


def _tie_averaged_walk_steps(values_1d: np.ndarray, in_set_mask: np.ndarray, weight_mode: str, param: float) -> np.ndarray:
    """
    Build strictly column-order-invariant random-walk steps across descending values with exact tie averaging:
      - Groups identical values into contiguous blocks (within tol = 1e-12).
      - Within every tie block of size b, assigns every block position the exact block-average hit weight
        and block-average miss step, guaranteeing identical output under ANY permutation of gene columns.
      - Degenerate zero-hit-weight guard: if total hit weight s_hit <= 0 (e.g., all genes tied at exact midpoint
        r_bar == G/2 with tau > 0 in GSVA), returns all-zero walk steps explicitly so ES = 0.0 instead of a
        spurious miss-only drift!
    """
    G = len(values_1d)
    n_set = int(in_set_mask.sum())
    n_miss = G - n_set
    if n_set == 0 or n_miss == 0:
        return np.zeros(G, dtype=float)

    order = np.argsort(-values_1d, kind="mergesort")
    v_sorted = values_1d[order]
    in_sorted = in_set_mask[order]

    tol = 1e-12
    if np.all((v_sorted[:-1] - v_sorted[1:]) > tol):
        if weight_mode == "ssgsea":
            pos_w = (np.arange(G, 0, -1, dtype=float)) ** param
        else:
            pos_w = np.abs(G / 2.0 - np.arange(1, G + 1, dtype=float)) ** param
        hit_w = np.where(in_sorted, pos_w, 0.0)
        s_hit = float(hit_w.sum())
        if s_hit <= 0.0:
            return np.zeros(G, dtype=float)
        return (hit_w / s_hit) - np.where(in_sorted, 0.0, 1.0 / n_miss)

    diff_idx = np.flatnonzero((v_sorted[:-1] - v_sorted[1:]) > tol) + 1
    starts = np.concatenate(([0], diff_idx))
    ends = np.concatenate((diff_idx, [G]))
    lengths = ends - starts
    hit_counts = np.add.reduceat(in_sorted.astype(np.int64), starts)
    hit_frac = hit_counts / lengths
    miss_frac = (lengths - hit_counts) / lengths

    if weight_mode == "ssgsea":
        avg_r = G - 0.5 * (starts + ends - 1.0)
        w_blk = (avg_r ** param) * hit_frac
    else:
        r_bar = 0.5 * (starts + ends + 1.0)
        w_blk = (np.abs(G / 2.0 - r_bar) ** param) * hit_frac

    hit_inc = np.repeat(w_blk, lengths)
    miss_inc = np.repeat(miss_frac / n_miss, lengths)
    s_hit = float(hit_inc.sum())
    if s_hit <= 0.0:
        return np.zeros(G, dtype=float)
    return (hit_inc / s_hit) - miss_inc


def score_ssgsea_bidirectional(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    alpha: float = 0.25,
) -> np.ndarray:
    """
    Tie-averaged, column-order-invariant Python variant of inductive bidirectional ssGSEA (Barbie et al., 2009):
    ES_alpha(S+) - ES_alpha(S-) normalized by G per sample (without cross-sample range rescaling).
    """
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("expr_2d must be a finite 2D array.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    up_cols = cols[w > 0]
    dn_cols = cols[w < 0]
    if len(up_cols) == 0 or len(dn_cols) == 0:
        raise ValueError("Bidirectional ssGSEA requires both positive and negative weights.")
    N, G = X.shape
    up_m = np.zeros(G, dtype=bool); up_m[up_cols] = True
    dn_m = np.zeros(G, dtype=bool); dn_m[dn_cols] = True
    out = np.empty(N, dtype=float)
    for i in range(N):
        step_u = _tie_averaged_walk_steps(X[i], up_m, "ssgsea", alpha)
        step_d = _tie_averaged_walk_steps(X[i], dn_m, "ssgsea", alpha)
        out[i] = float(np.sum(np.cumsum(step_u)) - np.sum(np.cumsum(step_d))) / G
    return out


def compute_gsva_logodds(expr_2d: np.ndarray, kcdf: str = "Gaussian") -> np.ndarray:
    """
    Compute across-sample per-gene GSVA log-odds matrix Z (N x G):
      - kcdf='Gaussian': Gaussian kernel CDF with bandwidth h_g = sd_g / 4 (Hanzelmann et al., 2013).
      - kcdf='midrank' (backwards-compatible alias 'none'): mid-distribution rank transform (average_rank - 0.5) / N.
      - kcdf='ecdf': standard right-continuous empirical CDF (rank_max / N, clipped to [eps, 1-eps]).
    All three modes are strictly TRANSDUCTIVE across samples in expr_2d.
    """
    X = np.asarray(expr_2d, dtype=np.float64)
    N, G = X.shape
    Z = np.empty((N, G), dtype=np.float64)
    eps = 1e-6
    if kcdf == "Gaussian":
        g_chunk = 512 if N <= 400 else 128
        for start in range(0, G, g_chunk):
            end = min(start + g_chunk, G)
            Xc = X[:, start:end]
            sd = Xc.std(axis=0, ddof=1)
            h = np.where(sd > 1e-8, sd * 0.25, 1.0)
            if N <= 400:
                diff = (Xc[:, None, :] - Xc[None, :, :]) / h[None, None, :]
                F = special.ndtr(diff).mean(axis=1)
            else:
                F = np.empty((N, end - start), dtype=np.float64)
                s_chunk = 128
                for i0 in range(0, N, s_chunk):
                    i1 = min(i0 + s_chunk, N)
                    diff = (Xc[i0:i1, None, :] - Xc[None, :, :]) / h[None, None, :]
                    F[i0:i1] = special.ndtr(diff).mean(axis=1)
            cm = sd <= 1e-8
            if np.any(cm):
                F[:, cm] = 0.5
            F = np.clip(F, eps, 1.0 - eps)
            Z[:, start:end] = np.log(F / (1.0 - F))
    elif kcdf in ("midrank", "none"):
        for j in range(G):
            col = X[:, j]
            if col.std(ddof=1) <= 1e-8:
                Z[:, j] = 0.0
            else:
                F = np.clip((stats.rankdata(col, method="average") - 0.5) / N, eps, 1.0 - eps)
                Z[:, j] = np.log(F / (1.0 - F))
    elif kcdf == "ecdf":
        for j in range(G):
            col = X[:, j]
            if col.std(ddof=1) <= 1e-8:
                Z[:, j] = 0.0
            else:
                F = np.clip(stats.rankdata(col, method="max") / N, eps, 1.0 - eps)
                Z[:, j] = np.log(F / (1.0 - F))
    else:
        raise ValueError(f"Unsupported kcdf={kcdf!r}; use 'Gaussian', 'midrank' ('none'), or 'ecdf'.")
    return Z


def score_gsva_bidirectional(
    expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
    kcdf: str = "Gaussian",
    tau: float = 1.0,
) -> np.ndarray:
    """
    Tie-averaged, column-order-invariant Python variant of transductive bidirectional GSVA (Hanzelmann et al., 2013).
    """
    X = np.asarray(expr_2d, dtype=float)
    if X.ndim != 2 or not np.all(np.isfinite(X)):
        raise ValueError("expr_2d must be a finite 2D array.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    up_cols = cols[w > 0]
    dn_cols = cols[w < 0]
    if len(up_cols) == 0 or len(dn_cols) == 0:
        raise ValueError("Bidirectional GSVA requires both positive and negative weights.")
    N, G = X.shape
    Z = compute_gsva_logodds(X, kcdf=kcdf)
    up_m = np.zeros(G, dtype=bool); up_m[up_cols] = True
    dn_m = np.zeros(G, dtype=bool); dn_m[dn_cols] = True
    out = np.empty(N, dtype=float)
    for i in range(N):
        wu = np.cumsum(_tie_averaged_walk_steps(Z[i], up_m, "gsva", tau))
        wd = np.cumsum(_tie_averaged_walk_steps(Z[i], dn_m, "gsva", tau))
        es_u = float(max(0.0, wu.max()) + min(0.0, wu.min()))
        es_d = float(max(0.0, wd.max()) + min(0.0, wd.min()))
        out[i] = es_u - es_d
    return out


def descriptive_vif_audit(
    trt_expr_2d: np.ndarray,
    sig_indices: Sequence[int | None],
    weights: Sequence[float],
) -> dict[str, float | int]:
    """
    Exact O(nm) algebraic computation of sign-aligned inter-gene correlation, descriptive Variance Inflation
    Factor (VIF), and variance-equivalent dimension without allocating an O(m^2) gene-gene correlation matrix.

    NOTE: Emits DESCRIPTIVE co-expression inflation diagnostics ONLY (no custom gene-axis p-value).
    Under strong negative inter-gene correlations, variance_equivalent_dimension = (sum |w_j|)^2 / Var(Z w)
    can exceed m and should be interpreted strictly as a weight-and-correlation-conditioned variance ratio,
    not as a literal count of identifiable independent genes.
    """
    X = np.asarray(trt_expr_2d, dtype=float)
    if X.ndim != 2 or X.shape[0] < 4:
        raise ValueError("Need 2D treatment matrix with n >= 4 rows.")
    cols, w = _clean_pairs(sig_indices, weights, n_genes=X.shape[1])
    G = X[:, cols]
    sd = G.std(axis=0, ddof=1)
    keep = np.isfinite(sd) & (sd > 1e-12)
    G, w = G[:, keep], w[keep]
    n, m = G.shape
    if m < 2:
        raise ValueError("Need at least 2 variable signature genes.")
    Z = (G - G.mean(axis=0, keepdims=True)) / G.std(axis=0, ddof=1, keepdims=True)
    s_w = np.sign(w)
    Z_al = Z * s_w[None, :]

    v = Z_al.sum(axis=1)
    rho_bar_al = (float(np.dot(v, v) / (n - 1)) - m) / (m * (m - 1))
    vif_al = 1.0 + (m - 1) * rho_bar_al

    u = Z @ w
    var_act = float(np.dot(u, u) / (n - 1))
    vif_w = var_act / float(np.dot(w, w))
    v_eq = float((np.sum(np.abs(w)) ** 2) / var_act) if var_act > 0 else float("nan")
    return {
        "n_samples": int(n),
        "n_genes": int(m),
        "rho_bar_sign_aligned": float(rho_bar_al),
        "vif_sign_aligned": float(vif_al),
        "vif_weighted": float(vif_w),
        "variance_equivalent_dimension": v_eq,
        "effective_independent_genes": v_eq,  # backwards-compatible alias
    }
