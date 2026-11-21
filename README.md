# ssrank-audit: Signature-Internal Inductive Rank Scoring for Transcriptional Signatures

Companion software and reproducibility repository for:

> **Zisen Zhou (2026).** *Signature-Internal Inductive Rank Scoring for Human Heart-Failure Transcriptional Signatures: Algebraic Invariances, Cross-Platform Transfer Boundaries, and Retrospective External Evaluation.* Manuscript prepared for *Computer Methods and Programs in Biomedicine* (Elsevier).

- **Zenodo Archival Release (derived source data + figures + code snapshot):** Pending Zenodo publication.
- **Core Module Dependencies:** Python >= 3.10, `numpy`, and `scipy` only (`ssrank_audit.py`).

## Quickstart

```python
import numpy as np
import ssrank_audit as sa

X = np.array([[1., 3., 2.], [2., 4., 1.], [3., 2., 5.], [4., 1., 3.]])
X_treatments = X
gene_idx = [0, 1, 2]
weights = np.array([1., -1., 0.5])
scores = sa.score_ss_rank_internal(X, gene_idx, weights)

# Linear-time O(nm) descriptive co-expression VIF diagnostic
vif = sa.descriptive_vif_audit(X_treatments[:, gene_idx], list(range(len(gene_idx))), weights)
print(vif["vif_sign_aligned"], vif["variance_equivalent_dimension"])
```

## Running the Unit Test Suite

```bash
pip install -r requirements.txt
python tests/test_ssrank_audit.py
python tests/test_release_helpers.py
```

The test suite verifies Proposition 1 (P1–P4) including exact constant vs. non-constant positive-weight behavior (`0.0` vs. `1/9`), exact three-gene background-expansion counterexamples (`-1/3 -> -1/4 -> -3/8` vs. invariant `-1/4`), tie-averaged column-permutation invariance, input validation, `GSVA` `kcdf="midrank"` vs. `"ecdf"`, and $O(nm)$ vs. $O(m^2)$ VIF identity.

## Repository Structure

```text
.
├── ssrank_audit.py                              # Standalone NumPy/SciPy scoring & O(nm) VIF module
├── tests/
│   └── test_ssrank_audit.py                     # Complete algebraic, tie-invariance, and API unit tests
├── scripts/
│   ├── benchmark_utils.py                       # Shared cohort loaders, thresholding, CR1 OLS, stress helpers
│   ├── run_3d_cardiopedia_benchmark.py          # 3D Cardiopedia (GSE314310) 9-method & 13-endpoint benchmark
│   ├── run_cross_system_benchmark.py            # 2D ScreenSeq, TPM1/DMD EHTs, and clinical calibration
│   ├── run_external_transfer_benchmark.py       # Pair-dedup discovery, 3 external cohorts, CPM->TPM/dropout/zeros
│   ├── run_sensitivity_and_comparator_audit.py  # Sydney NA, Italy lib2, Mayo external TPM, gseapy/ECDF checks
│   └── build_figures.py                         # Reproduces Figures 1–7 from derived results/ tables
├── data/
│   ├── dataset_manifest.json                    # Public study accessions and context
│   └── input_files.json                         # Input paths, URLs, byte sizes and SHA-256 checksums
├── requirements.txt
└── LICENSE
```

## Public data and benchmark reproduction

External-cohort evaluations are retrospective. Although fitted parameters are held fixed during a given evaluation, no independent timestamped preregistration demonstrates that the entire plan was frozen before first examining target-cohort outcomes; this is not a claim of prospective or blinded validation. `data/input_files.json` records the 26 raw inputs required by the benchmark scripts, including expected relative paths, original URLs, sizes and SHA-256 digests of the local copies. File-level byte checks cannot establish suitability, rights or donor independence. Original patient-level input datasets are not redistributed here.

For an exact tested primary Python stack (Windows, CPython 3.10.18), run:

```bash
python -m venv .venv
# Activate .venv using the command for your operating system.
python -m pip install -r requirements-tested.txt
python tests/test_ssrank_audit.py
python tests/test_release_helpers.py
python scripts/prepare_inputs.py --download
python scripts/prepare_gene_sets.py
python scripts/prepare_inputs.py
```

The input preparer checks every file's SHA-256; it will not overwrite mismatched files and returns an error if any file is missing. The ReHeaT2 entry is a member of the 581,769,280-byte upstream Zenodo ZIP. Access/download may fail or require manual retrieval; consult `data/input_files.json` for the exact source/member and copy it to the named destination. The original pre-existing Enrichr libraries are fetched locally by `prepare_gene_sets.py`; their content and the reconstructed GMT are checked against frozen checksums. The fixed GMT itself is not redistributed in this package pending confirmation of component-specific terms. See `THIRD_PARTY.md`.

By default, downloads and regenerated reference files are placed in `data/`. To use another location, set the `CARDIAC_DATA_DIR` environment variable before running the scripts, copy the supplied `data/reference/` tree there, and run both prepare scripts with `--data-dir` pointing to that same location. The benchmark scripts read `datasets/`, `external_cohorts/`, and `reference/` below that location and write results to the repository's `results/` directory. These scripts can overwrite existing results: use a disposable copy if you need to preserve the archived files.

The four project-derived seed tables are included in `results/`. After preparing the public inputs and fixed GMT above, they can also be rebuilt from original sources into a separate directory with `python scripts/rebuild_v5_seed_tables.py --output-dir scratch/v5_seed_tables`. This runs the hash-verified historical 3D, B6 and functional-repeat implementations in a disposable temporary workspace, and refuses to overwrite existing seed tables. It preserves the historical methods and data ordering; its local operator marker is neither an independent preregistration nor evidence of a blinded analysis. Copy regenerated tables into a disposable analysis copy only after checking them against the published tables. The independent replay on CPython 3.10.18 generated all four tables with byte-identical SHA-256 hashes. Openpyxl is required for this replay.

The project metadata in `data/reference/` are also required. From a copy of the Zenodo package, the intended execution order is:

```bash
python scripts/run_3d_cardiopedia_benchmark.py
python scripts/run_cross_system_benchmark.py
python scripts/run_external_transfer_benchmark.py
python scripts/run_sensitivity_and_comparator_audit.py
python scripts/build_figures.py
```

The post-hoc 2,000-reassignment reference used in Figure 3 can be rebuilt independently with `python scripts/rebuild_weight_reassignment.py --release-dir . --data-dir data --output scratch/weight_reassignment_2000.csv --compare results/weight_reassignment_2000.csv` (the comparison target is intended for the Zenodo archive). The script refuses to overwrite an existing output. This is a descriptive reference rather than a P value.

Three historical figure-input tables can be regenerated with `python scripts/rebuild_legacy_figure_inputs.py --target prevalence --output-dir scratch/legacy` and `python scripts/rebuild_legacy_figure_inputs.py --target v63 --output-dir scratch/legacy`. `python scripts/rebuild_zero_stress.py --output-dir scratch/legacy` reconstructs the historical fixed-column-order zero-inflation table. These commands refuse to overwrite their output files. In the historical zero perturbation, nonzero boundary ties can be resolved differently if input columns are reordered; see `REPRODUCIBILITY_STATUS.md`.

For the corrected Gaussian GSVA clinical-prevalence curve, `scripts/recompute_gsva_prevalence.py` generates matched historical/current batch files (seed `20261120`, 40 draws per prevalence). A full run is `python scripts/recompute_gsva_prevalence.py --output-dir scratch/gsva_batches`; it supports `--first-pct`, `--last-pct`, and `--max-batches` for partitioning. All nine levels are needed for the complete curve. Assemble completed batches with `python scripts/assemble_gsva_prevalence.py --batch-dirs scratch/gsva_batches --reference data/provenance/v6_4_table_clinical_prevalence_gsva_historical_float32.csv --output-dir scratch/gsva_summary`. The final figure input is the current tie-averaged/float64 curve; the old float32 stable-order result is preserved only as a historical reference. Baseline/seed consistency and unique batch coverage are checked; overlapping duplicates must agree. Historical six-cohort calibration can be reconstructed by `python scripts/rebuild_historical_calibration.py --output-dir scratch/legacy` after the cross-system benchmark creates `results/v6_gse349247_sample_scores.csv`; its GSVA row uses the old method, while Figure 5A selects only Cohort-Z and SS-Rank-Internal.

The 10-permutation zero-boundary sensitivity table is `results/zero_tie_permutation_results.csv`. Rebuild it using `python scripts/rebuild_zero_tie_permutation.py --output scratch/zero_tie_permutation_results.csv`. It uses seeds 2026112101–2026112110 and provides a descriptive sensitivity range, not a confidence interval.

The GitHub code package carries the four seed tables but does not contain all the figure inputs; the planned separate archive contains selected derived results. See `REPRODUCIBILITY_STATUS.md` for the exact scope and limitations of tests, including incomplete source-to-manuscript provenance. Regeneration of all the manuscript's supplementary numbers has not been demonstrated.

## License

MIT License for the original software code (see `LICENSE`). Third-party data and annotations retain their own terms; see `REPRODUCIBILITY_STATUS.md`.
