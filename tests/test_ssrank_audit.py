"""
Unit test suite for ssrank_audit.py (Proposition 1 invariances, P3/P4 exact counterexamples,
column-permutation tie invariance, defensive API validation, and O(nm) vs O(m^2) descriptive VIF).
"""
import math
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ssrank_audit import (
    descriptive_vif_audit,
    score_gsva_bidirectional,
    score_singscore_full,
    score_singscore_internal,
    score_ss_rank_full,
    score_ss_rank_internal,
    score_ssgsea_bidirectional,
    signed_weight_imbalance,
)


def _assert_raises_value_error(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("Expected ValueError was not raised!")


def test_defensive_clean_pairs_and_bounds():
    X = np.arange(20, dtype=float).reshape(4, 5)
    # 1. Unequal lengths rejected
    _assert_raises_value_error(lambda: score_ss_rank_internal(X, [0, 1, 2], [1.0, -1.0]))
    # 2. Negative index rejected
    _assert_raises_value_error(lambda: score_ss_rank_internal(X, [-1, 1], [1.0, -1.0]))
    # 3. Out-of-range index rejected
    _assert_raises_value_error(lambda: score_ss_rank_internal(X, [0, 5], [1.0, -1.0]))
    # 4. Duplicate / conflicting index rejected
    _assert_raises_value_error(lambda: score_ss_rank_internal(X, [1, 1], [1.0, -1.0]))


def test_proposition1_exact_invariances_and_boundaries():
    rng = np.random.default_rng(42)
    X = rng.normal(loc=5.0, scale=2.0, size=(25, 200))
    sig_idx = list(range(10, 70))
    weights = np.concatenate([rng.uniform(0.5, 3.0, 30), -rng.uniform(0.5, 3.0, 30)])

    # P1: Sample independence (zero transductive drift)
    s_full = score_ss_rank_internal(X, sig_idx, weights)
    X_skew = np.vstack([X[:5], rng.normal(50.0, 10.0, size=(100, 200))])
    assert np.allclose(s_full[:5], score_ss_rank_internal(X_skew, sig_idx, weights)[:5], atol=0.0, rtol=0.0)

    # P2: Strictly increasing per-sample transformations (exact invariance when order-preserving in float64)
    X_mono = X.copy()
    for i in range(X.shape[0]):
        X_mono[i] = np.exp(0.15 * X[i] + 0.3 * i) + 10.0 * (i + 1)
    assert np.allclose(s_full, score_ss_rank_internal(X_mono, sig_idx, weights), atol=0.0, rtol=0.0)

    # P2 float64 caveat: strictly INCREASING saturating map 1 - exp(-1000 * x) collapses distinct values in float64!
    X_pos = np.abs(X[:5]) + 1.0
    s_orig = score_ss_rank_internal(X_pos, sig_idx, weights)
    X_sat = 1.0 - np.exp(-1000.0 * X_pos)  # strictly increasing on R, saturates identically to 1.0 in float64
    assert np.all(X_sat == 1.0)
    assert np.allclose(score_ss_rank_internal(X_sat, sig_idx, weights), 0.0)
    assert np.max(np.abs(s_orig)) > 1e-3

    # P3: Exact background-universe invariance (even non-finite entries outside S)
    X_bg = np.hstack([X, np.full((25, 300), np.nan)])
    X_bg[:, :10] = np.inf
    assert np.allclose(s_full, score_ss_rank_internal(X_bg, sig_idx, weights), atol=0.0, rtol=0.0)

    # P3 Domain boundary:
    # (a) Constant-weight unsigned positive-only sets (w_k == c > 0) evaluate identically to 0.0 internally
    w_unsigned = np.ones(len(sig_idx), dtype=float)
    assert np.allclose(score_ss_rank_internal(X, sig_idx, w_unsigned), 0.0, atol=1e-15)
    # (b) Non-constant positive weights do NOT vanish identically (design scope, not universal zero theorem!)
    # Exact check: x=(1,2,3), w=(1,2,3) -> r=(-1/3, 0, +1/3) -> (1*(-1/3) + 2*0 + 3*(+1/3)) / 6 = (2/3)/6 = 1/9
    s_pos_nonconst = score_ss_rank_internal(np.array([[1.0, 2.0, 3.0]]), [0, 1, 2], [1.0, 2.0, 3.0])[0]
    assert abs(s_pos_nonconst - (1.0 / 9.0)) < 1e-12

    # P4 Exact full-background rescaling & interior-insertion counterexamples (P1-1):
    # x=(1,2,3), 0-based indices (0,2), w=(+1,-1) (beta=0):
    #   G=3: full SS-Rank = -1/3 exactly
    #   Append low background 0 (G=4): full SS-Rank = -1/4 exactly (uniform shift + 3/4 rescaling)
    #   Append interior background 1.5 (G=4, between x_0=1 and x_2=3): full SS-Rank = -3/8 exactly (gene-specific shift!)
    #   SS-Rank-Internal = -1/4 identically across all three!
    x3 = np.array([[1.0, 2.0, 3.0]])
    x4_low = np.array([[1.0, 2.0, 3.0, 0.0]])
    x4_mid = np.array([[1.0, 2.0, 3.0, 1.5]])
    r_full_3 = score_ss_rank_full(x3, [0, 2], [1.0, -1.0])[0]
    r_full_low = score_ss_rank_full(x4_low, [0, 2], [1.0, -1.0])[0]
    r_full_mid = score_ss_rank_full(x4_mid, [0, 2], [1.0, -1.0])[0]
    assert abs(r_full_3 - (-1.0 / 3.0)) < 1e-12, f"Expected -1/3, got {r_full_3}"
    assert abs(r_full_low - (-1.0 / 4.0)) < 1e-12, f"Expected -1/4, got {r_full_low}"
    assert abs(r_full_mid - (-3.0 / 8.0)) < 1e-12, f"Expected -3/8, got {r_full_mid}"
    for M in (x3, x4_low, x4_mid):
        s_int = score_ss_rank_internal(M, [0, 2], [1.0, -1.0])[0]
        assert abs(s_int - (-1.0 / 4.0)) < 1e-12, f"Expected SS-Rank-Internal -1/4, got {s_int}"

    # Per-sample missingness isolation
    X_miss = X.copy()
    X_miss[0, 10] = np.nan
    s_miss = score_ss_rank_internal(X_miss, sig_idx, weights, allow_missing_per_sample=True)
    assert math.isfinite(s_miss[0]) and np.allclose(s_miss[1:], s_full[1:], atol=0.0, rtol=0.0)

    # NA/zero weights strictly excluded
    sig_na = sig_idx + [0, 1, None]
    w_na = list(weights) + [float("nan"), 0.0, 2.5]
    assert np.allclose(s_full, score_ss_rank_internal(X, sig_na, w_na), atol=0.0, rtol=0.0)


def test_tie_column_permutation_and_gsva_kcdf_modes():
    # Adversarial tie counterexample: all-zero tied genes -> reversed columns must yield EXACTLY identical scores!
    X = np.array([[0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 2.0], [0.5, 0.5, 0.0, 0.0]])
    w = [1.0, -1.0]
    for fn in [
        lambda M, idx: score_ss_rank_internal(M, idx, w),
        lambda M, idx: score_ss_rank_full(M, idx, w),
        lambda M, idx: score_singscore_internal(M, idx, w),
        lambda M, idx: score_singscore_full(M, idx, w),
        lambda M, idx: score_ssgsea_bidirectional(M, idx, w),
        lambda M, idx: score_gsva_bidirectional(M, idx, w, kcdf="Gaussian"),
        lambda M, idx: score_gsva_bidirectional(M, idx, w, kcdf="midrank"),
        lambda M, idx: score_gsva_bidirectional(M, idx, w, kcdf="ecdf"),
    ]:
        s_fwd = fn(X, [0, 3])
        s_rev = fn(X[:, ::-1], [3, 0])
        assert np.allclose(s_fwd, s_rev, atol=1e-12, rtol=0.0), f"Column reversal broke tie invariance: {s_fwd} vs {s_rev}"

    # Zero-hit-weight guard in GSVA: all constant genes -> all Z=0 -> single midpoint tie block -> ES must be 0.0!
    X_const = np.zeros((4, 6), dtype=float)
    assert np.allclose(score_gsva_bidirectional(X_const, [0, 1, 4, 5], [1, 1, -1, -1], kcdf="Gaussian"), 0.0)

    # High zero-inflation random column permutations across all 8 rank/walk variants
    rng = np.random.default_rng(2026)
    X2 = rng.normal(size=(10, 60))
    X2[X2 < 0.5] = 0.0
    sig = list(range(20))
    w2 = [1.0] * 10 + [-1.0] * 10
    perm = rng.permutation(60)
    inv_perm = np.argsort(perm)
    sig_p = [int(inv_perm[c]) for c in sig]
    for fn in [
        lambda M, idx: score_ss_rank_internal(M, idx, w2),
        lambda M, idx: score_ss_rank_full(M, idx, w2),
        lambda M, idx: score_singscore_internal(M, idx, w2),
        lambda M, idx: score_singscore_full(M, idx, w2),
        lambda M, idx: score_ssgsea_bidirectional(M, idx, w2),
        lambda M, idx: score_gsva_bidirectional(M, idx, w2, kcdf="Gaussian"),
        lambda M, idx: score_gsva_bidirectional(M, idx, w2, kcdf="midrank"),
        lambda M, idx: score_gsva_bidirectional(M, idx, w2, kcdf="ecdf"),
    ]:
        assert np.allclose(fn(X2, sig), fn(X2[:, perm], sig_p), atol=1e-11, rtol=0.0)

    # Confirm midrank vs right-continuous ECDF differ explicitly on tied zeros [0,0,0,1]
    X3 = np.column_stack([np.array([0.0, 0.0, 0.0, 1.0]), np.array([1.0, 2.0, 3.0, 4.0]), np.array([4.0, 3.0, 2.0, 1.0]), np.array([0.0, 1.0, 0.0, 2.0])])
    s_mid = score_gsva_bidirectional(X3, [0, 1, 2, 3], [1.0, 1.0, -1.0, -1.0], kcdf="midrank")
    s_ecdf = score_gsva_bidirectional(X3, [0, 1, 2, 3], [1.0, 1.0, -1.0, -1.0], kcdf="ecdf")
    assert np.all(np.isfinite(s_mid)) and np.all(np.isfinite(s_ecdf))


def test_descriptive_vif_exact_onm_equivalence():
    rng = np.random.default_rng(7)
    n, m = 47, 80
    latent = rng.normal(size=(n, 1))
    X = 0.6 * latent + rng.normal(size=(n, m))
    w = np.concatenate([rng.uniform(0.5, 2.0, 40), -rng.uniform(0.5, 2.0, 40)])
    res = descriptive_vif_audit(X, list(range(m)), w)
    assert "p_camera_vif" not in res and "variance_equivalent_dimension" in res
    Z = (X - X.mean(0)) / X.std(0, ddof=1)
    R = np.corrcoef(Z, rowvar=False)
    s = np.sign(w)
    R_al = (s[:, None] * s[None, :]) * R
    rho_naive = (R_al.sum() - m) / (m * (m - 1))
    assert abs(res["rho_bar_sign_aligned"] - rho_naive) < 1e-12
    assert abs(res["vif_sign_aligned"] - (1.0 + (m - 1) * rho_naive)) < 1e-12
    var_naive = float(w @ R @ w)
    assert abs(res["vif_weighted"] - var_naive / float(np.dot(w, w))) < 1e-12
    assert abs(res["variance_equivalent_dimension"] - (np.abs(w).sum() ** 2) / var_naive) < 1e-11


if __name__ == "__main__":
    test_defensive_clean_pairs_and_bounds()
    test_proposition1_exact_invariances_and_boundaries()
    test_tie_column_permutation_and_gsva_kcdf_modes()
    test_descriptive_vif_exact_onm_equivalence()
    print("ALL ssrank_audit TESTS (INCL. ALL COUNTEREXAMPLES) PASSED!")
