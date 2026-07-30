# Data Directory

`manifests/datasets.csv` records every external dataset used by the artifact.
Full third-party datasets are intentionally not redistributed.

Run the download commands in `REPRODUCE.md`; downloaded files are placed under
`data/raw/` and ignored by Git.

`synthetic/dna_classification.csv` is generated DNA-like text used only to
exercise parsing and schema checks. It is not a subset of any benchmark and
must not be used to reproduce reported metrics.

`representative/etgp_metadata_subset.tsv` is a six-row, CC0-licensed subset of
the real DNALongBench ETGP metadata. It documents the true tabular schema but
does not include the 450-kb sequence inputs.
