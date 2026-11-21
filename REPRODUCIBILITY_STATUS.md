# Reproducibility status

The v1.0.0 Zenodo archive is https://doi.org/10.5281/zenodo.24414234.

## Tests performed

On Windows, a new CPython 3.10.18 virtual environment was installed with the primary package versions listed in `requirements-tested.txt`. Both test scripts passed, as did `pip check`. Transitive dependencies are not fully locked and full benchmark reanalysis has so far used the pre-existing environment with the same primary versions.

A 26-file input inventory with source URLs, byte counts, and SHA-256 was created; a clean-directory download test retrieved and validated all 26 files, including extraction of ReHeaT2 `mol_consistency.csv` from its 581,769,280-byte ZIP. These checks are time- and network-specific; a later download can fail or upstream files can change. The fixed GMT was reconstructed locally from three Enrichr libraries; source and output SHA-256 matched the frozen project files. The reconstructed GMT is not included in the distribution draft.

Using staged derived results, the final `scripts/build_figures.py` completed in an isolated copy with the original Windows/conda rendering stack (Matplotlib 3.10.0, FreeType 2.13.3, Pillow 11.3.0); all seven PNG SHA-256 hashes matched the figures in the local archive draft. A separate fresh pip environment also regenerated all seven figures, but PNG hashes and a small percentage of pixels differed; it used FreeType 2.6.1 and Pillow 12.3.0. Thus cross-environment byte identity is not established. These checks verify figure generation from staged tables, not derivation of all figures from raw data.

## Important limitations

- The four derived seed tables in `results/` are inputs to the benchmark scripts. `rebuild_v5_seed_tables.py` now replays the original hash-bound 3D and B6 analyses and functional-repeat computation from the verified source files in a disposable temporary workspace; an independent run in the tested virtual environment reproduced all four tables byte-for-byte. The fixed design metadata are packaged with the historical replay; the reconstructed third-party GMT is not redistributed.
- After removing four unused CSV reads and one unused JSON read, the figure script reads 28 CSV tables. Seven do not have a literal output path in the four packaged benchmark scripts. Separate scripts now reproduce the weight-reassignment table, three v6.1/v6.3 tables, and the v6.4 zero-stress table. The v6.4 six-cohort calibration is also reproducible; its GSVA row is historical and is not used by Figure 5A. The clinical GSVA-prevalence figure now uses the current float64/tie-averaged method, with the old float32/stable-order result kept separately under `data/provenance/`; matched recomputation and assembly scripts are included. The assembly was tested on all 360 existing draws, and one draw was independently repeated through the packaged producer. A full 360-draw rerun through the packaged producer remains untested. The entire graph of generators has not been exercised as one fresh raw-to-all-figure run, and complete reproduction of every supplementary analysis is not established.
- Fixes in this draft restore missing `math` and GPL11532-path definitions and the GSE199242 finite-row filtering and output column found in the existing development version. These are release-packaging repairs, not new scientific analyses.
- Third-party and sample-level derived-data redistribution rights remain to be resolved; see `THIRD_PARTY.md`.
- The four benchmark scripts completed in an isolated staging copy. Subsequent release-only changes to retrospective wording and to the generation of minimized clinical tables have been tested locally but the entire long-running external benchmark was not repeated after those changes. Per-script comparisons are recorded in the local project audit, not implied by a passing core unit test.

## Historical zero-inflation construction

`rebuild_zero_stress.py` reproduces the historical table for the original gene-column ordering, including the original stable-sort cutoff. If nonzero genes tie across the cutoff, a different input column order can produce a different perturbed matrix; this concerns construction of the stress test, not the separate tie behavior of a scoring rule. At 20%, 40%, 60%, and 70% zeroing, respectively 12, 216, 210, and 202 of 235 Cardiopedia samples have a nonzero tie crossing the cutoff. No such boundary ties were observed among 313 GSE57338 samples at those four rates. These counts do not by themselves quantify the impact on results.

## Updated clinical GSVA curve

The historical float32 stable-order curve was recomputed on the same 360 batches as the current float64 tie-averaged method (seed 20261120; 40 draws at each of nine prevalences). Historical results reproduce the archived curve within 3.4e-16. The current total drift is 1.249083 SD, versus 1.248912 SD historically; specificity at 10% DCM is 0.536111 and sensitivity at 90% DCM is 0.664583 for both. Changing both numerical precision and tie handling means their effects are not separately identified by the old/new difference. Fig. 1D now reads its curve and annotations directly from the corrected table; Fig. 6's range is also derived from inputs.

## Zero-boundary sensitivity rerun

The included `results/zero_tie_permutation_results.csv` records 10 fixed gene-column permutations per zeroing level and two internal rank scores, seed 2026112101–2026112110. The packaged `rebuild_zero_tie_permutation.py` was rerun against local public inputs, reproducing all 80 rows and 10 columns exactly. These are exploratory ranges for the specified perturbations, not confidence intervals.
