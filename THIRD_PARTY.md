# Third-party material and redistribution status

The MIT license in this directory covers original software only. It does not relicense datasets, annotations, images, clinical metadata or third-party content. Public access alone does not establish redistribution rights.

## Fixed program sources

`data/reference/theory/fixed_programs_2026-11-17/manifest.json` identifies the exact original libraries, entries and frozen hashes. `scripts/prepare_gene_sets.py` reconstructs the identical analysis GMT from the Enrichr library endpoint on the reader's machine. The reconstructed GMT combines:

- Reactome 2022 entries for muscle contraction, cardiac conduction and extracellular matrix; see [Reactome's license](https://reactome.org/license/), which distinguishes data from other content.
- MSigDB Hallmark 2020 entries for hypoxia, oxidative phosphorylation and unfolded protein response; see [MSigDB license terms](https://gsea-msigdb.org/gsea/msigdb_license_terms.jsp). The underlying MSigDB version and any component-specific terms for this Enrichr snapshot must be verified before redistribution.
- PanglaoDB Augmented 2021 fibroblast entry from Enrichr. The applicable redistribution terms for this particular snapshot remain unverified. See [PanglaoDB](https://panglaodb.se/) and [Enrichr](https://maayanlab.cloud/Enrichr/).

`fixed_programs.gmt` is deliberately excluded from this draft's intended distribution; obtain the libraries and reconstruct it for personal analysis, subject to the providers' terms. A successful checksum check verifies identity, not permission.

## Other data

`data/input_files.json` lists exact source URLs, original accessions, byte counts and hashes. [NCBI GEO states](https://www.ncbi.nlm.nih.gov/geo/info/disclaimer.html) that NCBI itself does not restrict GEO data use or distribution; submitters or third parties may nevertheless assert rights, so its disclaimer does not establish a blanket license. `results/` contains research-derived results including sample-level records. The retained GSE141910 figure table contains etiology, LVEF, and a per-sample score; the retained GSE46224 figure table contains internally pseudonymized patient codes, etiology, stage, and paired scores. Two unused tables with original study sample codes and additional age/gender/race fields were removed from this draft. Pseudonymization and data minimization do not establish permission or guarantee anonymity. The source terms and applicable institutional privacy requirements must be checked before public deposit; de-identification and GEO access do not automatically grant redistribution permission. Aggregation, local regeneration or removal may be needed if permission cannot be established. The ReHeaT study-metadata CSV and Cardiopedia overlap metadata retain information from the cited third-party publications; source-specific conditions must also be respected.

No uploaded/published release or blanket third-party authorization is represented by this draft.

## Historical V5 inputs

The historical seed replay requires ScreenSeq's `iemt_a_2273378_sm6211.txt` author supplement and the ReHeaT2 publication's `41467_2025_62219_MOESM10_ESM.csv`. These are obtained from the providers through `data/input_files.json` and are not included as source files in this draft. The project-authored fixed incidence metadata and hash-bound historical implementation are included for reproducibility. The fixed GMT must be reconstructed locally from provider data as described above. Access and checksum matching are not a license to redistribute those inputs.
