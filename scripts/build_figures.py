"""
Publication Figure Generator (Figures 1-7) for Computer Methods and Programs in Biomedicine.
Reads derived CSV/JSON tables from ./results/ and writes 300-DPI PNG and vector PDF figures to ./figures/.
"""

from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
V5_RES = ROOT / "results"
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Colorblind-safe Okabe-Ito / Nature-style palette
C_BLUE = "#0072B2"      # Cohort-Z / Primary
C_ORANGE = "#D55E00"    # Transductive distortion / Alert / DCM
C_TEAL = "#009E73"      # Single-Sample Rank (Inductive)
C_PURPLE = "#CC79A7"    # Ref-Frozen-Z / Secondary
C_GOLD = "#E69F00"      # Comparator / Highlight
C_SKY = "#56B4E9"       # Neutral / NF
C_GRAY = "#6C757D"
C_DARK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8.8,
    "axes.titlesize": 10.0,
    "axes.titleweight": "bold",
    "axes.labelsize": 9.0,
    "xtick.labelsize": 8.2,
    "ytick.labelsize": 8.2,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def save_fig(fig, stem: str):
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved {stem}.png/.pdf")


def build_fig1():
    """Figure 1 : Complete 9-Method Tie-Invariant 3D Benchmark & Transductive Distortion Monte Carlo Tests."""
    trt = pd.read_csv(RES / "v7_0_cardiopedia_47_treatments_with_ssri.csv")
    t1_9m = pd.read_csv(RES / "v7_0_table_3d_9methods_tieinv.csv")
    mc3d_g = pd.read_csv(RES / "v7_0_table_mc_3d_tieinv.csv")
    mc_prev = pd.read_csv(RES / "v6_1_mc100_prevalence_shift_gse165303.csv")
    mc_gsva = pd.read_csv(RES / "v6_4_table_clinical_prevalence_gsva.csv")

    fig = plt.figure(figsize=(13.4, 9.2))
    gs = GridSpec(2, 2, figure=fig, wspace=0.50, hspace=0.40)

    # Panel A: 3D Cardiopedia Cohort-Z vs SS-Rank-Internal scatter overlay (dots == label!)
    axA = fig.add_subplot(gs[0, 0])
    x_cz = (trt["score_cohort_z"] - trt["score_cohort_z"].mean()) / trt["score_cohort_z"].std(ddof=1)
    x_ri = (trt["score_ss_rank_internal"] - trt["score_ss_rank_internal"].mean()) / trt["score_ss_rank_internal"].std(ddof=1)
    y = trt["force_24h"]
    axA.scatter(x_cz, y, s=34, color=C_BLUE, alpha=0.72, edgecolors="white", lw=0.4, label="Cohort-Z (transductive)")
    axA.scatter(x_ri, y, s=34, color=C_TEAL, marker="^", alpha=0.78, edgecolors="white", lw=0.4, label="SS-Rank-Internal (inductive)")
    for xv, col, ls in [(x_cz, C_BLUE, "-"), (x_ri, C_TEAL, "--")]:
        z = np.polyfit(xv, y, 1)
        xg = np.linspace(xv.min(), xv.max(), 100)
        axA.plot(xg, np.polyval(z, xg), color=col, lw=1.6, ls=ls)
    axA.text(0.03, 0.05,
             "47 ligands × 5 plates (n = 235 organoids)\nCohort-Z: ρ = −0.713 (95% CI −0.843 to −0.510)\nSS-Rank-Int.: ρ = −0.706 (95% CI −0.844 to −0.499)\nsingscore: ρ = −0.759  |  GSVA (Gauss.): ρ = −0.721",
             transform=axA.transAxes, fontsize=7.3,
             bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA", ec="#CCCCCC", lw=0.7))
    axA.set_xlabel("Standardized ReHeaT2 Score (mean of 5 plates, SD units)")
    axA.set_ylabel("24 h Normalized Contractile Force")
    axA.set_title("A. 3D Cardiac Organoids: Cohort-Z vs. SS-Rank-Internal", loc="left")
    axA.legend(frameon=False, loc="upper right", fontsize=7.8)

    # Panel B: Forest plot across ALL 9 tie-invariant scoring baselines (3 transductive + 6 inductive)
    axB = fig.add_subplot(gs[0, 1])
    short_labels = [
        "Cohort-Z [Transd.]",
        "Robust-Z [Transd.]",
        "GSVA (Gauss.) [Transd.]",
        "SS-Rank (Full) [Ind.]",
        "SS-Rank-Internal [Ind.]",
        "singscore (Full) [Ind.]",
        "singscore-Int. [Ind.]",
        "ssGSEA (tie-avg.) [Ind.]",
        "Ref-Frozen-Z [Ind.]",
    ]
    y_pos = np.arange(len(t1_9m))
    cols = [C_BLUE, C_SKY, C_ORANGE, C_TEAL, "#007A5E", "#2CA02C", "#4DAF4A", C_GOLD, C_PURPLE]
    for i, row in t1_9m.iterrows():
        axB.errorbar(row["rho_24h"], i,
                     xerr=[[row["rho_24h"] - row["ci_low"]], [row["ci_high"] - row["rho_24h"]]],
                     fmt="o", color=cols[i], capsize=3.0, lw=1.5, ms=5.6, label="24h Force (95% CI)" if i == 0 else None)
        axB.scatter(row["partial_24h_given_04h"], i, color=C_GOLD, marker="D", s=30, zorder=4,
                    label="24h | 04h Partial ρ" if i == 0 else None)
        axB.scatter(row["repeat_24h_rho"], i, color=C_DARK, marker="x", s=32, lw=1.3, zorder=4,
                    label="Repeat Exp. 24h Force" if i == 0 else None)
    axB.axvline(0, color="#888888", lw=0.9, ls=":")
    axB.axhline(2.5, color="#BBBBBB", lw=0.8, ls="--")
    axB.set_yticks(y_pos)
    axB.set_yticklabels(short_labels, fontsize=7.5)
    axB.invert_yaxis()
    axB.set_xlim(-0.98, 0.28)
    axB.set_xlabel("Spearman Correlation (ρ) with Contractile Force")
    axB.set_title("B. Complete 9-Method 3D Benchmark (3 Transd. vs. 6 Ind.)", loc="left")
    axB.legend(frameon=False, loc="upper right", fontsize=7.1)

    # Panel C: 200-Draw Treatment-Mean 3D Batch-Skew MC (Empirical 2.5–97.5% intervals — explicit unit!)
    axC = fig.add_subplot(gs[1, 0])
    m_order = ["Cohort-Z", "Cohort-Robust-Z", "GSVA_Gaussian", "SS-Rank-Internal"]
    m_lbls = ["Cohort-Z\n(Transd.)", "Robust-Z\n(Transd.)", "GSVA (Gauss.)\n(Transd.)", "SS-Rank-Int.\n(Inductive)"]
    sub_mc = mc3d_g.set_index("method").loc[m_order].reset_index()
    x_pos = np.arange(len(m_order))
    shifts = sub_mc["mean_drift_sd"].to_numpy()
    sh_lo = shifts - sub_mc["q025_drift"].to_numpy()
    sh_hi = sub_mc["q975_drift"].to_numpy() - shifts
    flips = sub_mc["mean_flip_pct"].to_numpy()
    fl_lo = flips - sub_mc["q025_flip"].to_numpy()
    fl_hi = sub_mc["q975_flip"].to_numpy() - flips

    w = 0.35
    axC.bar(x_pos - w / 2, shifts, width=w, yerr=[sh_lo, sh_hi], capsize=3.0,
            color=[C_ORANGE, C_GOLD, C_PURPLE, C_TEAL], edgecolor="white", label="Mean Drift (SD, Emp. 2.5–97.5%)")
    axC2 = axC.twinx()
    axC2.spines["top"].set_visible(False)
    axC2.bar(x_pos + w / 2, flips, width=w, yerr=[fl_lo, fl_hi], capsize=3.0,
             color=["#F4A582", "#FDE0B6", "#E5C1D6", "#B2E2CD"], edgecolor=C_DARK, lw=0.6, hatch="//",
             label="Sign-Flip Rate (%, Emp. 2.5–97.5%)")
    for idx_b, (s_v, f_v, sh_u, fl_u, jc) in enumerate(zip(shifts, flips, sub_mc["q975_drift"], sub_mc["q975_flip"], sub_mc["mean_hit_jaccard"])):
        axC.text(idx_b - w / 2, sh_u + 0.03, f"{s_v:.2f}", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
        axC2.text(idx_b + w / 2, fl_u + 2.0, f"{f_v:.0f}%", ha="center", va="bottom", fontsize=7.3, fontweight="bold")
        axC.text(idx_b, -0.17, f"J={jc:.2f}", transform=axC.get_xaxis_transform(), ha="center", va="top",
                 fontsize=7.0, color=C_TEAL if jc > 0.99 else C_ORANGE, fontweight="bold")
    axC.set_xticks(x_pos)
    axC.set_xticklabels(m_lbls, fontsize=7.4)
    axC.set_ylabel("Mean Drift on 10 Probes (SD, Trt-Mean Unit)")
    axC2.set_ylabel("Probe Sign-Flip Rate (%)")
    axC.set_ylim(0, 1.34)
    axC2.set_ylim(0, 120)
    axC.set_title("C. Force-Stratified 3D Skew MC (Trt-Mean, 200 Draws; J)", loc="left")
    h1, l1 = axC.get_legend_handles_labels()
    h2, l2 = axC2.get_legend_handles_labels()
    axC.legend(h1 + h2, l1 + l2, frameon=False, loc="upper right", fontsize=6.9)

    # Panel D: Clinical Prevalence-Shift Monte Carlo Ribbon (GSE165303, Empirical 2.5–97.5% bands!)
    axD = fig.add_subplot(gs[1, 1])
    for mname, col, mk, ls in [
        ("Cohort-Z", C_ORANGE, "o", "-"),
        ("Cohort-Robust-Z", C_PURPLE, "s", "-."),
        ("SS-Rank", C_TEAL, "^", "--"),
    ]:
        sub = mc_prev[mc_prev["method"] == mname].sort_values("dcm_prevalence_pct")
        lbl = "SS-Rank / SS-Rank-Int. (0.00 SD)" if mname == "SS-Rank" else f"{mname} (Emp. 2.5–97.5%)"
        axD.plot(sub["dcm_prevalence_pct"], sub["drift_mean_sd"], f"{ls}{mk}", color=col, lw=1.8, ms=4.8, label=lbl)
        axD.fill_between(sub["dcm_prevalence_pct"], sub["drift_ci_low"], sub["drift_ci_high"], color=col, alpha=0.14)
    gsva_range = mc_gsva["gsva_mean_drift_sd"].max() - mc_gsva["gsva_mean_drift_sd"].min()
    axD.plot(mc_gsva["prev_dcm_pct"], mc_gsva["gsva_mean_drift_sd"], ":D", color=C_BLUE, lw=1.9, ms=4.8, label=f"GSVA Gauss. ({gsva_range:.2f} SD drift)")
    axD.fill_between(mc_gsva["prev_dcm_pct"], mc_gsva["gsva_ci_lo"], mc_gsva["gsva_ci_hi"], color=C_BLUE, alpha=0.12)
    axD.axhline(0, color="#888888", lw=0.8, ls=":")
    axD.axvline(50, color="#AAAAAA", lw=0.8, ls="--")
    axD.text(51.5, 0.85, "Balanced 50%\ncalibration", fontsize=7.2, color="#555555")
    gsva_by_prev = mc_gsva.set_index("prev_dcm_pct")
    cz_by_prev = mc_prev[mc_prev["method"] == "Cohort-Z"].set_index("dcm_prevalence_pct")
    robust_by_prev = mc_prev[mc_prev["method"] == "Cohort-Robust-Z"].set_index("dcm_prevalence_pct")
    g10, g90 = gsva_by_prev.loc[10], gsva_by_prev.loc[90]
    label10 = (f"At 10% DCM:\nCZ Spec {cz_by_prev.loc[10, 'spec_mean_pct']:.0f}% | "
               f"Rob-Z {robust_by_prev.loc[10, 'spec_mean_pct']:.0f}%\nGSVA Spec = {100*g10['gsva_spec']:.0f}%")
    label90 = (f"At 90% DCM:\nCZ Sens {cz_by_prev.loc[90, 'sens_mean_pct']:.0f}% | "
               f"Rob-Z {robust_by_prev.loc[90, 'sens_mean_pct']:.0f}%\nGSVA Sens = {100*g90['gsva_sens']:.0f}%")
    axD.annotate(label10, xy=(10, g10["gsva_mean_drift_sd"]), xytext=(18, 0.66),
                 fontsize=7.0, color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9))
    axD.annotate(label90, xy=(90, g90["gsva_mean_drift_sd"]), xytext=(52, -1.04),
                 fontsize=7.0, color=C_ORANGE, arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9))
    axD.set_xlabel("Batch DCM Prevalence in GSE165303 (%)")
    axD.set_ylabel("Signed Score Drift on Probes (SD)")
    axD.set_ylim(-1.14, 1.14)
    axD.set_title("D. Clinical Prevalence-Shift MC (All 3 Transductive Drift)", loc="left")
    axD.legend(frameon=False, loc="lower left", fontsize=6.8)

    save_fig(fig, "fig1_scoring_paradigms_and_distortion")


def build_fig2():
    """Figure 2: Multi-Kinetic Biomechanical Dissection & Conditional Confounder-Adjusted Robustness."""
    kin = pd.read_csv(RES / "v6_table3_cardiopedia_13_kinetics.csv")
    cond = pd.read_csv(RES / "v6_table6_conditional_partial_models_3d.csv")

    fig = plt.figure(figsize=(12.2, 6.2))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.38)

    # Panel A: All 13 kinetic endpoints sorted by Cohort-Z 24h rho
    axA = fig.add_subplot(gs[0, 0])
    k_cz = kin[kin["paradigm"] == "Cohort-Z"].sort_values("rho_24h").reset_index(drop=True)
    k_rk = kin[kin["paradigm"] == "Single-Sample Rank"].set_index("endpoint").loc[k_cz["endpoint"]].reset_index()

    ep_pretty = {
        "Force": "Peak Force (24 h)",
        "Tr_85_50": "Relax. Time Tr 85–50",
        "Tr_50": "Relax. Time Tr 50",
        "Ts_85_85": "Systolic Dur. Ts 85–85",
        "Ts_50_50": "Systolic Dur. Ts 50–50",
        "Tr_50_15": "Relax. Time Tr 50–15",
        "Ts_15_15": "Systolic Dur. Ts 15–15",
        "Ta_50": "Activation Time Ta 50",
        "BPM": "Beating Rate (BPM)",
        "Contract_velocity_norm": "Contraction Velocity",
        "Ta_30_85": "Activation Time Ta 30–85",
        "Relax_velocity_norm": "Relaxation Velocity",
        "Ta_15_30": "Activation Time Ta 15–30",
    }
    y = np.arange(len(k_cz))
    axA.errorbar(k_cz["rho_24h"], y - 0.14,
                 xerr=[k_cz["rho_24h"] - k_cz["ci24_low"], k_cz["ci24_high"] - k_cz["rho_24h"]],
                 fmt="o", color=C_BLUE, capsize=2.8, lw=1.3, ms=5.2, label="Cohort-Z (24 h, 95% CI)")
    axA.errorbar(k_rk["rho_24h"], y + 0.14,
                 xerr=[k_rk["rho_24h"] - k_rk["ci24_low"], k_rk["ci24_high"] - k_rk["rho_24h"]],
                 fmt="^", color=C_TEAL, capsize=2.8, lw=1.3, ms=5.2, label="SS-Rank (24 h, 95% CI)")
    axA.scatter(k_cz["repeat_rho_24h"], y, color=C_ORANGE, marker="x", s=34, lw=1.3, zorder=5, label="Repeat Exp. (24 h)")
    axA.axvline(0, color="#777777", lw=0.9, ls=":")
    axA.axhspan(-0.5, 2.5, color="#E8F4F8", alpha=0.5, zorder=0)
    axA.set_yticks(y)
    axA.set_yticklabels([ep_pretty.get(e, e) for e in k_cz["endpoint"]])
    axA.invert_yaxis()
    axA.set_xlim(-0.92, 0.80)
    axA.set_xlabel("Spearman Correlation (ρ) Across 47 Treatments")
    axA.set_title("A. Multi-Kinetic Biomechanical Profile (13 Endpoints)", loc="left")
    axA.legend(frameon=False, loc="upper right", fontsize=7.6)

    # Panel B: Conditional/Partial Association Models
    axB = fig.add_subplot(gs[0, 1])
    adj_order = [
        "Unadjusted",
        "+ Global L2 perturbation burden",
        "+ Hypoxia",
        "+ Structural_exclusive",
        "+ Fibroblast_marker + ECM",
        "+ UPR + OXPHOS",
        "+ All 7 fixed programs + L2 burden",
        "+ 04h Force baseline",
    ]
    adj_labels = [
        "Unadjusted",
        "+ Global L2 burden",
        "+ Hypoxia",
        "+ Structural (excl.)",
        "+ Fibroblast + ECM",
        "+ UPR + OXPHOS",
        "+ All 7 progs + L2",
        "+ 04h Force baseline",
    ]
    y2 = np.arange(len(adj_order))
    for s_name, col, mk, off in [
        ("Cohort-Z (mol_score)", C_BLUE, "o", -0.18),
        ("Single-Sample Rank (mol_score)", C_TEAL, "^", 0.0),
        ("Cohort-Z (ReHeaT2_CM)", C_PURPLE, "s", +0.18),
    ]:
        sub = cond[cond["score"] == s_name].set_index("adjustment").loc[adj_order]
        axB.plot(sub["rho_or_partial_rho"], y2 + off, f"-{mk}", color=col, lw=1.2, ms=5.8, label=s_name)
    axB.axvline(0, color="#777777", lw=0.9, ls=":")
    axB.axhspan(5.5, 6.5, color="#FFF3CD", alpha=0.65, zorder=0)
    # Annotate full adjustment value
    r_full = float(cond[(cond["score"] == "Cohort-Z (mol_score)") & (cond["adjustment"] == "+ All 7 fixed programs + L2 burden")]["rho_or_partial_rho"].iloc[0])
    p_full = float(cond[(cond["score"] == "Cohort-Z (mol_score)") & (cond["adjustment"] == "+ All 7 fixed programs + L2 burden")]["p_two_sided"].iloc[0])
    axB.text(-0.03, 6.0, f"p = {p_full:.1e}", va="center", ha="right", fontsize=7.6, fontweight="bold", color=C_DARK)
    axB.set_yticks(y2)
    axB.set_yticklabels(adj_labels)
    axB.invert_yaxis()
    axB.set_xlim(-0.82, 0.12)
    axB.set_xlabel("Partial Spearman Correlation with 24 h Force")
    axB.set_title("B. Conditional Confounder-Adjusted Robustness", loc="left")
    axB.legend(frameon=False, loc="upper right", fontsize=7.6)

    save_fig(fig, "fig2_cardiopedia_multikinetic_and_confounders")


def build_fig3():
    """Figure 3: Published ReHeaT2 Cell-Type Axes, Signed vs Unsigned Stability & CAMERA-VIF Audit."""
    axes_df = pd.read_csv(RES / "v6_table4_reheat2_celltype_axes_3d.csv")
    progs_df = pd.read_csv(RES / "v6_table5_fixed_programs_3d_multikinetic.csv")
    w_perm = pd.read_csv(V5_RES / "weight_reassignment_2000.csv")

    fig = plt.figure(figsize=(12.6, 8.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.38)

    # Panel A: Heatmap of 10 ReHeaT2 consensus & official cell-type axes across scoring paradigms & endpoints
    axA = fig.add_subplot(gs[0, :])
    cols_hm = ["cohort_z_rho_force", "ss_rank_rho_force", "ref_z_rho_force", "repeat_rho_force",
               "cohort_z_rho_tr50", "cohort_z_rho_bpm", "cohort_z_rho_cvel"]
    col_labels = ["Force\n(Cohort-Z)", "Force\n(SS-Rank)", "Force\n(Ref-Z)", "Repeat\nForce",
                  "Relax.\nTr 50", "Beating\nBPM", "Contract.\nVel."]
    order_ax = [
        "ReHeaT2_mol_score", "ReHeaT2_consistent_yes", "ReHeaT2_mean_t_all16036",
        "ReHeaT2_vSMCs", "ReHeaT2_Endo", "ReHeaT2_Fib", "ReHeaT2_PC",
        "ReHeaT2_Myeloid", "ReHeaT2_CM", "ReHeaT2_Lymphoid",
    ]
    pretty_ax = {
        "ReHeaT2_mol_score": "Consensus mol_score (13,256)",
        "ReHeaT2_consistent_yes": "Consistent-only (8,920)",
        "ReHeaT2_mean_t_all16036": "All-gene mean_t (15,152)",
        "ReHeaT2_vSMCs": "vSMCs HF-score (3,474)",
        "ReHeaT2_Endo": "Endothelial HF-score (8,386)",
        "ReHeaT2_Fib": "Fibroblast HF-score (9,390)",
        "ReHeaT2_PC": "Pericyte HF-score (7,301)",
        "ReHeaT2_Myeloid": "Myeloid HF-score (7,978)",
        "ReHeaT2_CM": "Cardiomyocyte HF-score (10,845)",
        "ReHeaT2_Lymphoid": "Lymphoid HF-score (2,767)",
    }
    sub_hm = axes_df.set_index("axis").loc[order_ax]
    mat = sub_hm[cols_hm].to_numpy(float)
    im = axA.imshow(mat, cmap="RdBu_r", vmin=-0.85, vmax=0.85, aspect="auto")
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            txt_col = "white" if abs(v) > 0.48 else C_DARK
            axA.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=8.0, color=txt_col, fontweight="bold" if abs(v) > 0.50 else "normal")
    axA.set_xticks(np.arange(len(col_labels)))
    axA.set_xticklabels(col_labels)
    axA.set_yticks(np.arange(len(order_ax)))
    axA.set_yticklabels([pretty_ax[k] for k in order_ax])
    # Highlight negative control row
    axA.add_patch(mpatches.Rectangle((-0.5, 8.5), len(col_labels), 1.0, fill=False, edgecolor=C_ORANGE, lw=1.8, ls="--"))
    cbar = fig.colorbar(im, ax=axA, fraction=0.022, pad=0.015)
    cbar.set_label("Spearman ρ (47 treatments)")
    axA.set_title("A. Published External ReHeaT2 Cell-Type Axes Across Scoring Paradigms & Kinetics", loc="left")

    # Panel B: Signed Bidirectional Weights vs Unsigned Positive-Only Pathways (Cohort-Z vs Single-Sample Rank)
    axB = fig.add_subplot(gs[1, 0])
    axB.plot([-0.9, 0.8], [-0.9, 0.8], "--", color="#999999", lw=1.0, label="Identity (y = x)")
    axB.scatter(axes_df["cohort_z_rho_force"], axes_df["ss_rank_rho_force"], s=52, color=C_BLUE, edgecolors="white", lw=0.5,
                label="Signed Bidirectional ReHeaT2 (n = 10)", zorder=4)
    axB.scatter(progs_df["cohort_z_rho_force"], progs_df["ss_rank_rho_force"], s=56, color=C_ORANGE, marker="s", edgecolors="white", lw=0.5,
                label="Unsigned Positive-Only Sets (n = 9)", zorder=4)
    # Annotate key points
    ann_pts = [
        (float(axes_df.loc[axes_df["axis"] == "ReHeaT2_mol_score", "cohort_z_rho_force"].iloc[0]),
         float(axes_df.loc[axes_df["axis"] == "ReHeaT2_mol_score", "ss_rank_rho_force"].iloc[0]), "mol_score", (-8, -14), C_BLUE),
        (float(axes_df.loc[axes_df["axis"] == "ReHeaT2_Lymphoid", "cohort_z_rho_force"].iloc[0]),
         float(axes_df.loc[axes_df["axis"] == "ReHeaT2_Lymphoid", "ss_rank_rho_force"].iloc[0]), "Lymphoid", (6, -12), C_BLUE),
        (float(progs_df.loc[progs_df["program"] == "structural_exclusive", "cohort_z_rho_force"].iloc[0]),
         float(progs_df.loc[progs_df["program"] == "structural_exclusive", "ss_rank_rho_force"].iloc[0]), "Structural", (5, -12), C_ORANGE),
        (float(progs_df.loc[progs_df["program"] == "conduction_exclusive", "cohort_z_rho_force"].iloc[0]),
         float(progs_df.loc[progs_df["program"] == "conduction_exclusive", "ss_rank_rho_force"].iloc[0]), "Conduction\n(sign flip)", (-34, 10), C_ORANGE),
        (float(progs_df.loc[progs_df["program"] == "upr", "cohort_z_rho_force"].iloc[0]),
         float(progs_df.loc[progs_df["program"] == "upr", "ss_rank_rho_force"].iloc[0]), "UPR", (-24, 9), C_ORANGE),
    ]
    for xv, yv, txt, oxy, col in ann_pts:
        axB.annotate(txt, xy=(xv, yv), xytext=oxy, textcoords="offset points", fontsize=7.6, color=col, fontweight="bold")
    mad_signed = float(np.mean(np.abs(axes_df["cohort_z_rho_force"] - axes_df["ss_rank_rho_force"])))
    mad_unsigned = float(np.mean(np.abs(progs_df["cohort_z_rho_force"] - progs_df["ss_rank_rho_force"])))
    axB.text(0.03, 0.96, f"Mean |Δρ| (Signed):   {mad_signed:.3f}\nMean |Δρ| (Unsigned): {mad_unsigned:.3f}",
             transform=axB.transAxes, va="top", fontsize=7.8,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", lw=0.7))
    axB.set_xlim(-0.90, 0.72)
    axB.set_ylim(-0.90, 0.80)
    axB.set_xlabel("Cohort-Z Spearman ρ (24 h Force)")
    axB.set_ylabel("Single-Sample Rank Spearman ρ (24 h Force)")
    axB.set_title("B. Signed Weight Cancellation vs. Unsigned Drift", loc="left")
    axB.legend(frameon=False, loc="lower right", fontsize=7.5)

    # Panel C: Descriptive O(nm) Co-Expression VIF Audit vs Shuffled Weights (NO invalid custom feature-axis p-value!)
    axC = fig.add_subplot(gs[1, 1])
    perm_col = [c for c in w_perm.columns if "rho" in c.lower()][0]
    axC.hist(w_perm[perm_col], bins=36, color="#8DA9C4", edgecolor="white", lw=0.4, alpha=0.88,
             label="2,000 Expression-Stratified\nWeight Reassignments (0/2,000 extreme)")
    axC.axvline(-0.7127, color=C_ORANGE, lw=2.2, label="Observed mol_score (ρ = −0.713)")
    vif10 = pd.read_csv(RES / "v7_0_table_descriptive_vif_10axes.csv").set_index("axis").loc["ReHeaT2_mol_score"]
    box_txt = (
        "Descriptive Co-Expression VIF Audit (O(nm)):\n"
        f"• Usable genes m = {int(vif10['n_genes']):,}\n"
        f"• Mean aligned r̄ = +{vif10['rho_bar_sign_aligned']:.4f}\n"
        f"• Sign-aligned VIF = {vif10['vif_sign_aligned']:.1f}×\n"
        f"• Weighted VIF = {vif10['vif_weighted']:.1f}×\n"
        f"• Var.-equiv. dimension = {vif10['effective_independent_genes']:.1f}\n"
        "• Descriptive variance-inflation\n"
        "  diagnostic (no independence assumption)"
    )
    axC.text(0.03, 0.96, box_txt, transform=axC.transAxes, va="top", fontsize=7.2,
             bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8E7", ec=C_GOLD, lw=0.9))
    axC.set_xlim(-0.88, 0.80)
    axC.set_ylim(0, 178)
    axC.set_xlabel("Spearman ρ Across 47 Treatments")
    axC.set_ylabel("Reassigned Weight Count")
    axC.set_title("C. Descriptive Co-Expression VIF Audit vs. Shuffled Weights", loc="left")
    axC.legend(frameon=True, facecolor="white", framealpha=0.92, edgecolor="none", loc="upper left", bbox_to_anchor=(0.01, 0.50), fontsize=7.2)

    save_fig(fig, "fig3_celltype_axes_and_camera_vif")


def build_fig4():
    """Figure 4: Cross-System Transportability — 2D ScreenSeq, TPM1 EHT, and 2D-vs-3D Factorial EHT."""
    b6_tab = pd.read_csv(RES / "v6_table7_2d_screenseq_multiparadigm.csv")
    tpm = pd.read_csv(RES / "v6_table8_tpm1_eht_multiparadigm.csv")
    d199 = pd.read_csv(RES / "v6_table9_gse199242_2d_vs_3d_shift.csv")
    t4 = pd.read_csv(RES / "v6_table4_reheat2_celltype_axes_3d.csv")
    t5 = pd.read_csv(RES / "v6_table5_fixed_programs_3d_multikinetic.csv")

    fig = plt.figure(figsize=(12.4, 8.8))
    gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.38)

    # Panel A: 2D ScreenSeq Amplitude — Transductive Cohort-Z vs Inductive SS-Rank-Internal (exact match to Paired Diff CI!)
    b6_pd = pd.read_csv(RES / "v7_0_table_2d_screenseq_paired_diff.csv").set_index("method")
    b6_ssri = pd.read_csv(RES / "v7_1_screenseq_38drugs_ssri.csv")
    axA = fig.add_subplot(gs[0, 0])
    x1 = (b6_ssri["score_cz_mean"] - b6_ssri["score_cz_mean"].mean()) / b6_ssri["score_cz_mean"].std(ddof=1)
    x2 = (b6_ssri["score_ssri_mean"] - b6_ssri["score_ssri_mean"].mean()) / b6_ssri["score_ssri_mean"].std(ddof=1)
    y = b6_ssri["assay_mean"]
    axA.scatter(x1, y, s=36, color=C_ORANGE, alpha=0.78, edgecolors="white", lw=0.4, label="Cohort-Z (transductive)")
    axA.scatter(x2, y, s=36, color=C_TEAL, marker="^", alpha=0.78, edgecolors="white", lw=0.4, label="SS-Rank-Internal (inductive)")
    for xv, col, ls in [(x1, C_ORANGE, "-"), (x2, C_TEAL, "--")]:
        z = np.polyfit(xv, y, 1)
        xg = np.linspace(xv.min(), xv.max(), 80)
        axA.plot(xg, np.polyval(z, xg), color=col, lw=1.6, ls=ls)
    row_cz = b6_pd.loc["Cohort-Z"]
    row_ri = b6_pd.loc["SS-Rank-Internal"]
    axA.text(0.03, 0.04,
             f"2D ScreenSeq (38 drugs, 2,899 RNA samples)\n"
             f"Cohort-Z:        ρ = {row_cz['across_rho']:+.3f} (95% CI {row_cz['ci_low']:+.3f} to {row_cz['ci_high']:+.3f})\n"
             f"SS-Rank-Int.:    ρ = {row_ri['across_rho']:+.3f} (95% CI {row_ri['ci_low']:+.3f} to {row_ri['ci_high']:+.3f})\n"
             f"Paired Δρ (CZ − SS-Rank-Int.): {row_ri['diff_cz_minus_m']:+.3f} [{row_ri['diff_ci_low']:+.3f}, {row_ri['diff_ci_high']:+.3f}]",
             transform=axA.transAxes, fontsize=7.0,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", lw=0.7))
    axA.set_xlabel("Standardized Drug-Mean ReHeaT2 Score (SD units)")
    axA.set_ylabel("Mean 2D Optical Beat Amplitude")
    axA.set_title("A. 2D ScreenSeq: Transductive vs. Inductive Associations", loc="left")
    axA.legend(frameon=False, loc="upper left", fontsize=7.8)

    # Panel B: Symmetric 3D vs 2D Cross-System Heatmap across key ReHeaT2 & biological programs
    axB = fig.add_subplot(gs[0, 1])
    rows_cmp = [
        ("ReHeaT2 mol_score", "ReHeaT2_mol_score", "ReHeaT2_mol_score | Cohort-Z", "ReHeaT2_mol_score | Single-Sample Rank"),
        ("ReHeaT2 CM", "ReHeaT2_CM", "ReHeaT2_CM | Cohort-Z", "ReHeaT2_CM | Single-Sample Rank"),
        ("ReHeaT2 Fib", "ReHeaT2_Fib", "ReHeaT2_Fib | Cohort-Z", "ReHeaT2_Fib | Single-Sample Rank"),
        ("Structural (excl.)", "structural_exclusive", "Prog_structural_exclusive | Cohort-Z", "Prog_structural_exclusive | Single-Sample Rank"),
        ("UPR Stress", "upr", "Prog_upr | Cohort-Z", "Prog_upr | Single-Sample Rank"),
        ("Hypoxia", "hypoxia", "Prog_hypoxia | Cohort-Z", "Prog_hypoxia | Single-Sample Rank"),
        ("ECM", "ecm", "Prog_ecm | Cohort-Z", "Prog_ecm | Single-Sample Rank"),
    ]
    mat_cmp = np.zeros((len(rows_cmp), 5), float)
    for i, (lbl, k3d, k2cz, k2rk) in enumerate(rows_cmp):
        if k3d.startswith("ReHeaT2"):
            mat_cmp[i, 0] = float(t4.loc[t4["axis"] == k3d, "cohort_z_rho_force"].iloc[0])
            mat_cmp[i, 1] = float(t4.loc[t4["axis"] == k3d, "ss_rank_rho_force"].iloc[0])
        else:
            mat_cmp[i, 0] = float(t5.loc[t5["program"] == k3d, "cohort_z_rho_force"].iloc[0])
            mat_cmp[i, 1] = float(t5.loc[t5["program"] == k3d, "ss_rank_rho_force"].iloc[0])
        mat_cmp[i, 2] = float(b6_tab[(b6_tab["score_name"] == k2cz) & (b6_tab["assay"] == "Amplitude")]["across_drug_rho"].iloc[0])
        mat_cmp[i, 3] = float(b6_tab[(b6_tab["score_name"] == k2rk) & (b6_tab["assay"] == "Amplitude")]["across_drug_rho"].iloc[0])
        mat_cmp[i, 4] = float(b6_tab[(b6_tab["score_name"] == k2cz) & (b6_tab["assay"] == "DNA structure")]["within_drug_median_rho"].iloc[0])
    imB = axB.imshow(mat_cmp, cmap="RdBu_r", vmin=-0.80, vmax=0.80, aspect="auto")
    for r in range(mat_cmp.shape[0]):
        for c in range(mat_cmp.shape[1]):
            v = mat_cmp[r, c]
            tc = "white" if abs(v) > 0.45 else C_DARK
            axB.text(c, r, f"{v:+.2f}", ha="center", va="center", fontsize=8.0, color=tc, fontweight="bold" if abs(v) > 0.48 else "normal")
    axB.set_xticks(range(5))
    axB.set_xticklabels(["3D Force\n(CZ)", "3D Force\n(SS-Rk)", "2D Amp.\n(CZ)", "2D Amp.\n(SS-Rk)", "2D DNA\n(Within)"], fontsize=7.8)
    axB.set_yticks(range(len(rows_cmp)))
    axB.set_yticklabels([r[0] for r in rows_cmp])
    cbarB = fig.colorbar(imB, ax=axB, fraction=0.04, pad=0.02)
    cbarB.set_label("Spearman ρ")
    axB.set_title("B. 3D Organoid vs. 2D Monolayer Program Symmetry", loc="left")

    # Panel C: TPM1 Isogenic Genetic EHT (WT / E54K / E62Q) — Cohort-Z vs Inductive Ref-Frozen-Z CM vs Peak Force
    axC = fig.add_subplot(gs[1, 0])
    genos = ["E54K\n(53 µN)", "WT\n(161 µN)", "E62Q\n(522 µN)"]
    xg = np.arange(3)
    # Extract normalized trajectories for key signatures
    def get_t(sig, par):
        r = tpm[(tpm["signature"] == sig) & (tpm["paradigm"] == par)].iloc[0]
        return np.array([r["E54K_mean"], r["WT_mean"], r["E62Q_mean"]], float)

    # Plot E54K -> WT -> E62Q (increasing force!) for Cohort-Z mol_score, Ref-Frozen-Z ReHeaT2_CM, Cohort-Z Hypoxia, Cohort-Z OXPHOS
    v_cz = get_t("ReHeaT2_mol_score", "Cohort-Z")
    v_cm_rf = get_t("ReHeaT2_CM", "Ref-Frozen-Z")
    v_hyp = get_t("Prog_hypoxia", "Cohort-Z")
    v_ox = get_t("Prog_oxphos", "Cohort-Z")

    axC.plot(xg, v_cm_rf, "-^", color=C_TEAL, lw=2.2, ms=7.0, label="ReHeaT2_CM (Ref-Frozen-Z, inv. to Force)")
    axC.plot(xg, v_cz, "-o", color=C_BLUE, lw=2.0, ms=6.5, label="mol_score (Cohort-Z, n=9 transductive)")
    axC.plot(xg, v_hyp, "--s", color=C_ORANGE, lw=1.6, ms=6.0, label="Hypoxia (Cohort-Z, metabolic demand)")
    axC.plot(xg, v_ox, ":D", color=C_PURPLE, lw=1.6, ms=6.0, label="OXPHOS (Cohort-Z)")
    axC.axhline(0, color="#999999", lw=0.8, ls=":")
    axC.set_xticks(xg)
    axC.set_xticklabels(genos)
    axC.set_ylim(-0.62, 0.56)
    axC.set_ylabel("Mean Transcriptional Score (n = 3 RNA / genotype)")
    axC.set_xlabel("Isogenic TPM1 Genotype (Ordered by Peak Force)")
    axC.set_title("C. Isogenic TPM1 EHTs (GSE251993): Genetic Context Limit", loc="left")
    axC.legend(frameon=False, loc="lower left", fontsize=7.2)

    # Panel D: GSE199242 2D Monolayer vs 3D EHT (WT & DMD 263delG) under Inductive Single-Sample Rank
    axD = fig.add_subplot(gs[1, 1])
    sigs_199 = ["ReHeaT2_mol_score", "ReHeaT2_CM", "ReHeaT2_Fib", "Prog_structural_exclusive"]
    lbls_199 = ["mol_score", "ReHeaT2_CM", "ReHeaT2_Fib", "Structural"]
    sub199 = d199[(d199["paradigm"] == "Single-Sample Rank") & (d199["signature"].isin(sigs_199))].set_index("signature").loc[sigs_199]
    x4 = np.arange(len(sigs_199))
    bw = 0.26
    axD.bar(x4 - bw, sub199["shift_3D_minus_2D_WT"], width=bw, color=C_BLUE, label="3D − 2D (WT)")
    axD.bar(x4, sub199["shift_3D_minus_2D_DMD"], width=bw, color=C_SKY, label="3D − 2D (DMD)")
    axD.bar(x4 + bw, sub199["shift_DMD_minus_WT_3D"], width=bw, color=C_ORANGE, label="DMD − WT (in 3D EHT)")
    axD.axhline(0, color=C_DARK, lw=0.8)
    axD.set_xticks(x4)
    axD.set_xticklabels(lbls_199)
    axD.set_ylabel("Δ Single-Sample Rank Score")
    axD.set_title("D. 2D vs. 3D Architecture & DMD Shift (GSE199242)", loc="left")
    axD.legend(frameon=False, loc="best", fontsize=7.6)

    save_fig(fig, "fig4_cross_system_2d_eht_transportability")


def build_fig5():
    """Figure 5: Clinical Adult Myocardium Calibration, Paired LVAD Recovery & LVEF Hierarchy."""
    lvad_tab = pd.read_csv(RES / "v6_table11_gse46224_paired_lvad.csv")

    fig = plt.figure(figsize=(12.6, 8.6))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.38)

    # Panel A: All 6 Clinical Cohorts Calibration Forest Plot (Cohort-Z vs SS-Rank-Internal)
    cal6 = pd.read_csv(RES / "v6_4_table_clinical_6cohorts_calibration.csv")
    axA = fig.add_subplot(gs[0, 0])
    cohorts_ord = [
        "GSE165303_Proc",
        "GSE349247_CPM",
        "GSE116250_RPKM",
        "GSE46224_FPKM",
        "GSE57338_Array",
        "GSE141910_CPM",
    ]
    c_labels = [
        "GSE165303 (Proc.)\n(51 NF vs 50 DCM)",
        "GSE349247 (2026)\n(13 Ctrl vs 62 HF)",
        "GSE116250 (RPKM)\n(14 NF vs 50 HF)",
        "GSE46224 (FPKM)\n(8 NF vs 16 Pre-LVAD)",
        "GSE57338 (Array)\n(136 NF vs 177 HF)",
        "GSE141910 MAGNet\n[Upstream Ref, n=332]",
    ]
    y_c = np.arange(len(cohorts_ord))
    for i, coh in enumerate(cohorts_ord):
        for par, col, mk, off in [("Cohort-Z", C_BLUE, "o", -0.15), ("SS-Rank-Internal", C_TEAL, "^", +0.15)]:
            sub = cal6[(cal6["cohort"] == coh) & (cal6["method"] == par)]
            if len(sub):
                r = sub.iloc[0]
                axA.errorbar(r["auc"], i + off,
                             xerr=[[max(0, r["auc"] - r["ci_low"])], [max(0, r["ci_high"] - r["auc"])]],
                             fmt=mk, color=col, capsize=2.8, lw=1.4, ms=5.8,
                             label=par if i == 0 else None)
                axA.text(r["auc"] - 0.008, i + off - 0.05, f"{r['auc']:.3f}", ha="right", va="bottom", fontsize=7.0, color=col, fontweight="bold")
    axA.set_yticks(y_c)
    axA.set_yticklabels(c_labels, fontsize=7.6)
    axA.invert_yaxis()
    axA.set_xlim(0.84, 1.02)
    axA.set_xlabel("Clinical Discrimination AUROC (95% Bootstrap CI)")
    axA.set_title("A. All-6-Cohort Clinical Calibration (4 Technologies)", loc="left")
    axA.legend(frameon=False, loc="lower left", fontsize=7.6)

    # Panel B: Paired LVAD Mechanical Unloading Trajectories (GSE46224, SS-Rank-Internal!)
    lvad_ssri = pd.read_csv(RES / "v7_0_gse46224_trajectories_ssri.csv")
    lvad_3s = pd.read_csv(RES / "v7_0_gse46224_lvad_3bidir_ssri.csv").set_index("signature")
    axB = fig.add_subplot(gs[0, 1])
    nf_s = lvad_ssri[lvad_ssri["stage"] == "NF"]["score_ss_rank_internal"].to_numpy()
    pre_df = lvad_ssri[lvad_ssri["stage"] == "Pre-LVAD"].set_index("patient_id")
    post_df = lvad_ssri[lvad_ssri["stage"] == "Post-LVAD"].set_index("patient_id").loc[pre_df.index]

    rng = np.random.default_rng(20261120)
    axB.scatter(0 + rng.uniform(-0.08, 0.08, len(nf_s)), nf_s, s=36, color=C_SKY, edgecolors=C_DARK, lw=0.4, zorder=4)
    axB.plot([-0.16, 0.16], [np.median(nf_s)] * 2, color=C_DARK, lw=2.2, zorder=5)

    for pid in pre_df.index:
        y0 = float(pre_df.loc[pid, "score_ss_rank_internal"])
        y1 = float(post_df.loc[pid, "score_ss_rank_internal"])
        et = pre_df.loc[pid, "etiology"]
        col_l = C_TEAL if y1 < y0 else C_ORANGE
        axB.plot([1, 2], [y0, y1], "-", color=col_l, alpha=0.68, lw=1.35)
        axB.scatter([1, 2], [y0, y1], s=28, color=col_l, marker="o" if et == "ICM" else "s", edgecolors="white", lw=0.4, zorder=4)
    axB.plot([0.84, 1.16], [np.median(pre_df["score_ss_rank_internal"])] * 2, color=C_DARK, lw=2.2, zorder=5)
    axB.plot([1.84, 2.16], [np.median(post_df["score_ss_rank_internal"])] * 2, color=C_DARK, lw=2.2, zorder=5)

    row_lv_ri = lvad_3s.loc["ReHeaT2_mol_score"]
    axB.text(0.03, 0.96,
             f"Paired LVAD Unloading (n = 16 pairs)\n{int(row_lv_ri['n_decrease'])} / 16 decrease toward NF\nPaired Cohen's dz = {row_lv_ri['paired_cohens_dz_ssri']:.2f}\nWilcoxon p = {row_lv_ri['wilcoxon_p_ssri']:.4f}\nPre-LVAD vs NF AUROC = 1.000",
             transform=axB.transAxes, va="top", fontsize=7.5,
             bbox=dict(boxstyle="round,pad=0.3", fc="#F8F9FA", ec="#CCCCCC", lw=0.7))
    axB.set_xticks([0, 1, 2])
    axB.set_xticklabels(["Non-Failing\n(n = 8)", "Pre-LVAD\n(n = 16)", "Post-LVAD\n(n = 16)"])
    axB.set_ylabel("SS-Rank-Internal ReHeaT2 Score")
    axB.set_title("B. Paired Clinical Mechanical Unloading (GSE46224)", loc="left")

    # Panel C: Paired LVAD Effect Size Comparison (SS-Rank-Internal for bidirectional; *Full SS-Rank for unsigned sets)
    axC = fig.add_subplot(gs[1, 0])
    lv_rk = lvad_tab[lvad_tab["paradigm"] == "Single-Sample Rank"].copy().set_index("signature")
    sig_ord_lv = ["ReHeaT2_mol_score", "ReHeaT2_Fib", "ReHeaT2_CM", "Prog_structural_exclusive", "Prog_ecm", "Prog_hypoxia"]
    lbl_lv = ["ReHeaT2 mol_score", "ReHeaT2 Fib", "ReHeaT2 CM", "Structural (excl.)*", "ECM Organization*", "Hypoxia*"]
    dz_vals, p_vals = [], []
    for sig in sig_ord_lv:
        if sig in lvad_3s.index:
            dz_vals.append(float(lvad_3s.loc[sig, "paired_cohens_dz_ssri"]))
            p_vals.append(float(lvad_3s.loc[sig, "wilcoxon_p_ssri"]))
        else:
            dz_vals.append(float(lv_rk.loc[sig, "paired_cohens_dz"]))
            p_vals.append(float(lv_rk.loc[sig, "wilcoxon_p_two_sided"]))
    y_lv = np.arange(len(sig_ord_lv))
    bar_cols = [C_TEAL if v < 0 and p < 0.05 else C_ORANGE if v > 0 and p < 0.05 else C_GRAY
                for v, p in zip(dz_vals, p_vals)]
    axC.barh(y_lv, dz_vals, color=bar_cols, edgecolor="white", height=0.62)
    axC.axvline(0, color=C_DARK, lw=0.9)
    for i, (dz, p) in enumerate(zip(dz_vals, p_vals)):
        ha = "right" if dz < 0 else "left"
        dx = -0.04 if dz < 0 else 0.04
        axC.text(dz + dx, i, f"dz={dz:+.2f} (p={p:.3f})", va="center", ha=ha, fontsize=7.5)
    axC.set_yticks(y_lv)
    axC.set_yticklabels(lbl_lv)
    axC.invert_yaxis()
    axC.set_xlim(-1.85, 1.60)
    axC.set_xlabel("Paired Post − Pre LVAD Cohen's dz (SS-Rank-Int.; *Full for Unsigned)")
    axC.set_title("C. Unloading Recovery Specificity (*Full SS-Rank for Unsigned Sets)", loc="left")

    # Panel D: Between-State vs Within-State Functional Hierarchy in Clinical Myocardium (MAGNet LVEF, SS-Rank-Internal!)
    lvef_ri = pd.read_csv(RES / "v7_0_gse141910_sample_scores_lvef_ssri.csv")
    axD = fig.add_subplot(gs[1, 1])
    lv_ok = lvef_ri[lvef_ri["LVEF"].notna() & lvef_ri["score_ss_rank_internal"].notna()].copy()
    nf_lv = lv_ok[lv_ok["etiology"] == "NF"]
    dcm_lv = lv_ok[lv_ok["etiology"] == "DCM"]
    from scipy.stats import spearmanr
    rho_all_ri = float(spearmanr(lv_ok["score_ss_rank_internal"], lv_ok["LVEF"]).statistic)
    rho_dcm_ri = float(spearmanr(dcm_lv["score_ss_rank_internal"], dcm_lv["LVEF"]).statistic)
    axD.scatter(nf_lv["score_ss_rank_internal"], nf_lv["LVEF"] * 100, s=26, color=C_SKY, alpha=0.80, edgecolors="white", lw=0.3, label=f"NF (n = {len(nf_lv)})")
    axD.scatter(dcm_lv["score_ss_rank_internal"], dcm_lv["LVEF"] * 100, s=26, color=C_ORANGE, alpha=0.72, edgecolors="white", lw=0.3, label=f"DCM (n = {len(dcm_lv)})")
    z_all = np.polyfit(lv_ok["score_ss_rank_internal"], lv_ok["LVEF"] * 100, 1)
    xg_all = np.linspace(lv_ok["score_ss_rank_internal"].min(), lv_ok["score_ss_rank_internal"].max(), 80)
    axD.plot(xg_all, np.polyval(z_all, xg_all), "-", color=C_DARK, lw=1.6, label=f"Overall (NF+DCM): ρ = {rho_all_ri:.3f}")
    z_dcm = np.polyfit(dcm_lv["score_ss_rank_internal"], dcm_lv["LVEF"] * 100, 1)
    xg_dcm = np.linspace(dcm_lv["score_ss_rank_internal"].min(), dcm_lv["score_ss_rank_internal"].max(), 80)
    axD.plot(xg_dcm, np.polyval(z_dcm, xg_dcm), "--", color=C_ORANGE, lw=1.6, label=f"Within-DCM only: ρ = {rho_dcm_ri:.3f}")
    axD.set_xlabel("SS-Rank-Internal ReHeaT2 Score")
    axD.set_ylabel("Left Ventricular Ejection Fraction (%)")
    axD.set_title("D. Between- vs. Within-State Clinical Hierarchy (GSE141910 Ref.)", loc="left")
    axD.legend(frameon=False, loc="upper right", fontsize=7.4)

    save_fig(fig, "fig5_clinical_calibration_and_lvad_recovery")


def build_fig6():
    """Figure 6: Operational Computational Transportability & Scoring Audit Framework (Graphical Summary)."""
    fig, ax = plt.subplots(figsize=(12.2, 6.0))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)

    # Header banner
    ax.add_patch(mpatches.FancyBboxPatch((1.5, 48.5), 97, 6.0, boxstyle="round,pad=0.4", fc="#1B3A4B", ec="none"))
    ax.text(50, 51.5, "Operational Framework for Transportable Transcriptional Scoring Across Cardiac Models & Clinical Cohorts",
            ha="center", va="center", color="white", fontsize=11.0, fontweight="bold")

    # 4 vertical stage boxes
    stages = [
        (2.0, 26.5, 22.0, 19.5, "#E8F4F8", C_BLUE,
         "1. Scoring Formulation",
         "• Use SS-Rank-Internal for\n  bidirectional signatures\n  (Prop. 1: P1–P3 exact)\n• Zero transductive drift\n  vs Cohort-Z/Robust-Z/GSVA\n• Unsigned sets (w = +1):\n  use full-background SS-Rank\n  here (internal = 0)"),
        (26.5, 26.5, 22.0, 19.5, "#E6F5F0", C_TEAL,
         "2. Algebraic & VIF Audit",
         "• Exclude NA weights (2,395\n  never zero-imputed)\n• Same-gene 18k→58k shift:\n  full SS-Rank 0.013 vs 0.334\n  (SS-Rank-Int. = 0.000)\n• Report O(nm) descriptive\n  VIF (144.5×; V_eq = 68)\n  with sample-axis inference"),
        (51.0, 26.5, 22.0, 19.5, "#FFF4E6", C_GOLD,
         "3. Hierarchical Scope",
         "• Separate between-state\n  from within-state levels:\n  – 3D hCO: −0.71 vs +0.09\n  – LVEF: −0.69 vs −0.07\n• Dissect 13 kinetics\n  (Force & Tr50 vs null BPM)\n• Rank strata help interpret\n  the 3D organoid trough"),
        (75.5, 26.5, 22.5, 19.5, "#FDECE8", C_ORANGE,
         "4. External Eval. & Limits",
         "• 3 Retrospective ext. cohorts\n  (n=181; 3×9 evals): AUC\n  0.955, 0/27 crash (SSRI)\n• 2D ScreenSeq: 5/6 ind.\n  below Cohort-Z (paired CI)\n• Pair-level dedup, missingness,\n  & biobank boundary checks"),
    ]
    for x, y, w, h, fc, ec, title, body in stages:
        ax.add_patch(mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35", fc=fc, ec=ec, lw=1.4))
        ax.text(x + w / 2, y + h - 1.8, title, ha="center", va="top", fontsize=9.2, fontweight="bold", color=ec)
        ax.text(x + 1.0, y + h - 4.4, body, ha="left", va="top", fontsize=7.5, color=C_DARK, linespacing=1.32)

    # Arrows between stages
    for ax_x in (24.3, 48.8, 73.3):
        ax.annotate("", xy=(ax_x + 2.0, 36.2), xytext=(ax_x - 0.1, 36.2),
                    arrowprops=dict(arrowstyle="->", color=C_DARK, lw=1.6))

    # Bottom empirical evidence scorecard table
    ax.add_patch(mpatches.FancyBboxPatch((2.0, 2.0), 96.0, 21.5, boxstyle="round,pad=0.35", fc="#FAFAFA", ec="#BBBBBB", lw=1.0))
    ax.text(3.5, 21.8, "Empirical Cross-Domain Benchmark Summary Across 13 Public Human Cardiac Cohorts (n = 4,261 Profiles)",
            fontsize=8.8, fontweight="bold", color=C_DARK, va="top")

    headers = ["Domain / Cohort", "N & Endpoint", "Cohort-Z", "SS-Rank-Int.", "Key Methodological Takeaway"]
    col_x = [3.5, 25.0, 41.5, 52.0, 64.5]
    for cx, htxt in zip(col_x, headers):
        ax.text(cx, 18.8, htxt, fontsize=7.8, fontweight="bold", color="#1B3A4B", va="center")
    ax.plot([3.2, 96.8], [17.4, 17.4], color="#CCCCCC", lw=0.8)

    prevalence = pd.read_csv(RES / "v6_1_mc100_prevalence_shift_gse165303.csv")
    gsva_prevalence = pd.read_csv(RES / "v6_4_table_clinical_prevalence_gsva.csv")
    ranges = [float(g["drift_mean_sd"].max() - g["drift_mean_sd"].min())
              for method, g in prevalence.groupby("method") if method in {"Cohort-Z", "Cohort-Robust-Z"}]
    ranges.append(float(gsva_prevalence["gsva_mean_drift_sd"].max() - gsva_prevalence["gsva_mean_drift_sd"].min()))
    clinical_drift_text = f"Prevalence drift: CZ/RobZ/GSVA {min(ranges):.2f}–{max(ranges):.2f} SD vs 0"

    rows_sc = [
        ("3D hCO (GSE314310)", "235 org. / 24h Force", "ρ = −0.713", "ρ = −0.706*", "Treatment-mean Jaccard 1.00 vs 0.31–0.56"),
        ("2D ScreenSeq (GSE244740)", "2,899 RNA / Amplitude", "ρ = +0.531*", "ρ = +0.284 NS", "Paired Δρ (CZ−SSRI) = +0.247 [+0.094, +0.423]"),
        ("TPM1 & DMD (2 EHT cohorts)", "21 RNA / Force & 2D/3D", "Distorted", "—", "Ref-Z CM ordering; 3D structural shift"),
        ("6 Discovery Adult LV", "925 LV (pair-48 dedup.)", "AUC .89–1.0", "AUC .88–1.00", clinical_drift_text),
        ("3 External Clinical LV", "181 prof. (IT, AU, US)", "AUC 0.930", "AUC 0.955", "3×9 evals: 0/27 crash (vs 10–12/27 full); BalAcc 0.789"),
        ("Stress Tests (2 cohorts)", "Within-coh. TPM, Drop/0s", "Requires re-c.", "0.75–0.92", "Leads within-cohort CPM→TPM, 50% drop, 70% zeros"),
    ]
    for ri, row in enumerate(rows_sc):
        ry = 15.6 - ri * 2.35
        if ri % 2 == 1:
            ax.add_patch(mpatches.Rectangle((3.2, ry - 1.1), 93.6, 2.2, fc="#F0F4F8", ec="none"))
        for cx, val in zip(col_x, row):
            fw = "bold" if cx == 52.0 else "normal"
            col_t = C_TEAL if cx == 52.0 else C_DARK
            ax.text(cx, ry, val, fontsize=7.3, color=col_t, fontweight=fw, va="center")

    save_fig(fig, "fig6_framework_decision_matrix")


def build_fig7():
    """Figure 7 : Strictly Equal-K Rank Drift, 3D vs Clinical Panel Scaling, Dedup Discovery & Retrospective External Clinical Evaluation, and Two-Cohort Stress Suite."""
    abl = pd.read_csv(RES / "v7_0_table_strict_same_k_cancellation.csv")
    kmulti = pd.read_csv(RES / "v6_3_table_compact_panel_3d_vs_clinical.csv")
    dsum = pd.read_csv(RES / "v7_0_table_disc_descriptive_summary.csv")
    hsum = pd.read_csv(RES / "v7_0_table_heldout_summary.csv").set_index("method")
    df_len = pd.read_csv(RES / "v7_0_table_cpm_to_tpm_two_cohorts.csv")
    df_drop = pd.read_csv(RES / "v6_3_table_missing_gene_dropout_array.csv")
    df_zi = pd.read_csv(RES / "v6_4_table_zero_inflation_ties_stress.csv")

    fig = plt.figure(figsize=(13.6, 9.4))
    gs = GridSpec(2, 2, figure=fig, wspace=0.34, hspace=0.42)

    # Panel A: Strictly Equal Non-Constant K (20 .. 5,000 ONLY — no mismatched 13.3k point!)
    axA = fig.add_subplot(gs[0, 0])
    xk = np.arange(len(abl))
    klbls = [str(int(k)) for k in abl["panel_K"]]
    axA.plot(xk, abl["signed_bal_abs_delta"], "-o", color=C_TEAL, lw=2.2, ms=6.0,
             label="Balanced Signed (β ≤ 0.04, exact K)")
    axA.plot(xk, abl["unsigned_up_abs_delta"], "--s", color=C_GOLD, lw=1.8, ms=5.5,
             label="Unsigned Up-Only (β = 1.0, exact K)")
    axA.plot(xk, abl["unsigned_dn_abs_delta"], "-.^", color=C_ORANGE, lw=2.0, ms=6.0,
             label="Unsigned Down-Only (β = 1.0, exact K)")
    axA.set_xticks(xk)
    axA.set_xticklabels(klbls, fontsize=7.8)
    axA.set_xlabel("Matched Panel Size K (ReHeaT2)")
    axA.set_ylabel("|Δρ| Between Cohort-Z and Full SS-Rank (24 h Force)")
    axA.set_ylim(-0.02, 0.62)
    axA.set_title("A. Same-Size (K = 20–5,000) Signed vs. Unsigned Rank Drift", loc="left")
    axA.legend(frameon=False, loc="upper left", fontsize=7.4)

    # Panel B: 3D Organoids vs Clinical Microarray & Paired LVAD Across Panel Size K
    axB = fig.add_subplot(gs[0, 1])
    xkm = np.arange(len(kmulti))
    klm = [str(int(k)) if k < 10000 else "13.6k" for k in kmulti["panel_K"]]
    axB.plot(xkm, -kmulti["rho_3d_hco"], "-o", color=C_ORANGE, lw=2.1, ms=5.8, label="3D hCO Force |ρ| (SS-Rank-Int.)")
    axB.plot(xkm, kmulti["gse57338_array_frozen_bal_acc"], "--^", color=C_BLUE, lw=2.0, ms=5.6, label="GSE57338 Array Frozen BalAcc")
    axB.plot(xkm, -kmulti["gse46224_lvad_dz"], "-.s", color=C_TEAL, lw=2.0, ms=5.4, label="GSE46224 Paired LVAD |dz|")
    axB.axvspan(3.6, 5.4, color="#FFF3CD", alpha=0.55, zorder=0)
    axB.annotate("Ranks 101–1,000:\nAdult immune/sparse\n(hCO ρ = −0.15 to −0.28;\nClinical varies with panel size)",
                 xy=(4.0, 0.41), xytext=(1.55, 0.34), fontsize=7.0, color=C_DARK,
                 bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8E7", ec=C_GOLD, lw=0.7),
                 arrowprops=dict(arrowstyle="->", color=C_ORANGE, lw=0.9))
    axB.set_xticks(xkm)
    axB.set_xticklabels(klm, fontsize=7.8)
    axB.set_xlabel("Balanced ReHeaT2 Panel Size K (SS-Rank-Internal)")
    axB.set_ylabel("3D |ρ|, Clinical Frozen BalAcc, or Paired LVAD |dz|")
    axB.set_ylim(0.32, 1.04)
    axB.set_title("B. 3D vs. Clinical Panel Scaling & Immune-Stratum Trough", loc="left")
    axB.legend(frameon=False, loc="lower right", fontsize=7.3)

    # Panel C: Dedup Discovery (ReHeaT2 30 & De-Novo 90) + Retrospective External Clinical Evaluation (27 across IT/AU/US!)
    axC = fig.add_subplot(gs[1, 0])
    m_ord = ["singscore", "ssGSEA", "SS-Rank", "singscore-Internal", "SS-Rank-Internal"]
    m_lbl = ["singscore\n(Full)", "ssGSEA\n(tie-avg.)", "SS-Rank\n(Full)", "singscore-\nInternal", "SS-Rank-\nInternal"]
    s_rh = dsum[dsum["benchmark"] == "ReHeaT2_Full_Dedup (30)"].set_index("method")
    s_dn = dsum[dsum["benchmark"] == "DeNovo_Dedup_ALL (90)"].set_index("method")

    x = np.arange(len(m_ord))
    bw = 0.26
    v_rh = [float(s_rh.loc[m, "mean_bal_acc"]) for m in m_ord]
    v_dn = [float(s_dn.loc[m, "mean_bal_acc"]) for m in m_ord]
    v_ho = [float(hsum.loc[m, "all27_mean_bal_acc"]) for m in m_ord]
    base_y = 0.48
    axC.bar(x - bw, np.array(v_rh) - base_y, bottom=base_y, width=bw, color=C_BLUE, edgecolor="white", label="ReHeaT2 Dedup (30 discovery)")
    axC.bar(x, np.array(v_dn) - base_y, bottom=base_y, width=bw, color=C_SKY, edgecolor="white", label="De-Novo Dedup (90 discovery)")
    axC.bar(x + bw, np.array(v_ho) - base_y, bottom=base_y, width=bw, color=C_TEAL, edgecolor="white", label="External Clinical (3 cohorts × 9)")
    for i in range(len(m_ord)):
        axC.text(i - bw, v_rh[i] + 0.012, f"{v_rh[i]:.2f}", ha="center", va="bottom", fontsize=6.6, color=C_BLUE, fontweight="bold")
        axC.text(i, v_dn[i] + 0.012, f"{v_dn[i]:.2f}", ha="center", va="bottom", fontsize=6.6, color="#005F99")
        axC.text(i + bw, v_ho[i] + 0.012, f"{v_ho[i]:.2f}", ha="center", va="bottom", fontsize=6.6, color=C_TEAL, fontweight="bold")
    axC.axhline(0.50, color="#999999", lw=0.8, ls=":")
    axC.set_xticks(x)
    axC.set_xticklabels(m_lbl, fontsize=7.4)
    axC.set_ylim(0.36, 1.07)
    axC.set_ylabel("Mean Frozen-Threshold Balanced Accuracy")
    axC.text(0.50, 0.025,
             "Retrospective External Clinical (3 cohorts × 9 models): SS-Rank-Int. 0/27\ncrashes, AUC 0.955, BalAcc 0.789 vs. 10–12/27 crashes for full-matrix rankers",
             transform=axC.transAxes, ha="center", va="bottom", fontsize=6.5,
             bbox=dict(boxstyle="round,pad=0.24", fc="#E6F5F0", ec=C_TEAL, lw=0.7))
    axC.set_title("C. Pair-Dedup Discovery & External Clinical Transfer", loc="left")
    axC.legend(frameon=False, loc="upper left", fontsize=6.8)

    # Panel D: Missing-Gene Dropout (0-50%), Zero-Inflation Ties (0-70%), & Two-Cohort Real CPM->TPM Length Stress
    axD = fig.add_subplot(gs[1, 1])
    x_dr = df_drop["dropout_pct"].to_numpy()
    axD.errorbar(x_dr, df_drop["ssri_frozen_bal_acc_mean"], yerr=df_drop["ssri_frozen_bal_acc_sd"],
                 fmt="-o", color=C_TEAL, lw=2.1, ms=5.8, capsize=3.2, label="SS-Rank-Int.: Random Gene Dropout (0–50%)")
    axD.plot(df_zi["zero_inflation_pct"], df_zi["array_frozen_bal_ssri"], "-.^", color="#005F99", lw=2.0, ms=5.8,
             label="SS-Rank-Int.: Per-Sample Zero-Inflation (0–70%)")
    axD.plot(df_zi["zero_inflation_pct"], df_zi["array_frozen_bal_sgi"], "--s", color=C_GOLD, lw=1.7, ms=5.2,
             label="singscore-Int.: Per-Sample Zero-Inflation (0–70%)")
    lf_mag = df_len[(df_len["cohort"] == "GSE141910_MAGNet") & (df_len["signature"] == "ReHeaT2_Full")].set_index("method")
    lf_mayo = df_len[(df_len["cohort"] == "GSE249925_Mayo_HeldOut") & (df_len["signature"] == "ReHeaT2_Full")].set_index("method")
    tpm_mag_ri = float(lf_mag.loc["SS-Rank-Internal", "tpm_frozen_bal_acc"])
    tpm_mayo_ri = float(lf_mayo.loc["SS-Rank-Internal", "tpm_frozen_bal_acc"])
    tpm_mayo_ssg = float(lf_mayo.loc["ssGSEA", "tpm_frozen_bal_acc"])
    axD.axhline(tpm_mag_ri, color=C_TEAL, lw=1.3, ls=":")
    axD.axhline(tpm_mayo_ri, color=C_SKY, lw=1.3, ls=":")
    axD.axhline(tpm_mayo_ssg, color=C_ORANGE, lw=1.3, ls=":")
    axD.text(71.5, tpm_mag_ri - 0.008, f"Within-Cohort CPM→TPM MAGNet SSRI: {tpm_mag_ri:.2f}", ha="right", va="top", fontsize=6.7, color=C_TEAL, fontweight="bold")
    axD.text(71.5, tpm_mayo_ri - 0.008, f"Within-Cohort CPM→TPM Mayo SSRI: {tpm_mayo_ri:.2f}", ha="right", va="top", fontsize=6.7, color="#005F99")
    axD.text(71.5, tpm_mayo_ssg + 0.012, f"Within-Cohort CPM→TPM Mayo singscore (0.51) / ssGSEA ({tpm_mayo_ssg:.2f})", ha="right", va="bottom", fontsize=6.6, color=C_ORANGE, fontweight="bold")
    axD.set_xlabel("Perturbation Severity (% Missing Genes or % Zero-Inflated Ties)")
    axD.set_ylabel("Frozen-Threshold Balanced Accuracy")
    axD.set_xlim(-3, 73)
    axD.set_ylim(0.46, 0.99)
    axD.set_title("D. Dropout, Zero Ties, & Within-Cohort CPM→TPM Stress", loc="left")
    axD.legend(frameon=False, loc=(0.01, 0.22), fontsize=6.8)

    save_fig(fig, "fig7_sparsity_cancellation_and_crossplatform")


def main():
    build_fig1()
    build_fig2()
    build_fig3()
    build_fig4()
    build_fig5()
    build_fig6()
    build_fig7()
    print("All 7 publication figures generated cleanly in", FIG_DIR)


if __name__ == "__main__":
    main()
