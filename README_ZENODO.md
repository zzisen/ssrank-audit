# Archival Data, Figures, and Software Deposit (Zenodo Release v1.0.0)

**Manuscript:** *Signature-Internal Inductive Rank Scoring for Human Heart-Failure Transcriptional Signatures: Algebraic Invariances, Cross-Platform Transfer Boundaries, and Retrospective External Evaluation*  
**Author:** Zisen Zhou (Auckland Bioengineering Institute, University of Auckland, New Zealand; zisen.zhou@auckland.ac.nz)  
**Target Journal:** *Computer Methods and Programs in Biomedicine* (Elsevier)  
**GitHub Repository:** https://github.com/zzisen/ssrank-audit

## Contents

- `ssrank_audit.py` & `tests/test_ssrank_audit.py` — Standalone NumPy/SciPy scoring and $O(nm)$ descriptive VIF module with full unit test suite.
- `scripts/` — Portable relative-path benchmark scripts and `build_figures.py`.
- `results/` — Selected derived source-data tables (CSV/JSON), including inputs used to regenerate Figures 1–7. Completeness for every manuscript and supplementary-table calculation has not yet been verified.
- `figures/` — All seven publication figures in both 300-DPI PNG and vector PDF formats.
- `data/input_files.json` — 26 raw input paths, source URLs, sizes and SHA-256 hashes; the original accession context is in `data/dataset_manifest.json`.

## Offline Verification & Figure Reproduction

After downloading this archive:

```bash
pip install -r requirements.txt
python tests/test_ssrank_audit.py
python tests/test_release_helpers.py
python scripts/build_figures.py
```

Raw third-party GEO and Zenodo `15261569` expression datasets remain hosted at their original repositories and are not redistributed in this archive.

For reanalysis rather than figure-only regeneration, consult `README.md`. Some figure input tables are staged derived results whose full upstream producing scripts are not included in the package. Third-party component and sample-level redistribution status is described in `THIRD_PARTY.md`; do not assume the MIT software license applies to those materials.
