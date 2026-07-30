# Data Availability and Redistribution Boundary

## Decision

Full third-party datasets are **not** stored in the GitHub staging branch and
are **not** copied into the AAAI ZIP. They are either too large for the
supplementary archive or distributed under terms that should not be
relicensed by this project. The package instead contains:

1. exact dataset identifiers, versions, task names, and licenses;
2. download/preparation code that retrieves data from the official source;
3. a six-row CC0 ETGP metadata subset showing the real input schema;
4. a synthetic sequence sample for offline smoke tests;
5. derived, machine-readable metric files needed to audit reported numbers.

## Dataset Manifest

| Dataset | Identifier/version | Role | License/terms | Retrieval |
|---|---|---|---|---|
| hg38 MLM assets | `hg38.ml.fa`, `sequences_human.bed` | pretraining | source terms | `scripts/download_hg38_pretrain_data.sh` |
| GenomicBenchmarks | package `0.0.9`, dataset version 0 | GB | per-source terms; package code Apache-2.0 | `scripts/download_genomicbenchmarks_data.py` |
| Genomics Long-Range Benchmark | `InstaDeepAI/genomics-long-range-benchmark`, `variant_effect_causal_eqtl` | VEP | CC BY-NC-SA 4.0 | loaded by `vep_embeddings.py` |
| DNALongBench ETGP | DOI `10.7910/DVN/CTEQXX`, version 1 | ETGP | CC0 1.0 | `scripts/download_dnalongbench_450k_data.py` |

The machine-readable copy is `data/manifests/datasets.csv`.

## Expected Local Layout

```text
data/raw/
  hg38/
    hg38.ml.fa
    human-sequences.bed
  genomic_benchmarks/
    <official task directories>
  dnalongbench/
    <Dataverse ETGP directory tree>
```

The paths are ignored by Git. Users may place equivalent verified copies
elsewhere and override the root arguments.

## Derived Data

VEP embeddings and trained checkpoints are large derived artifacts and are
not included. Their configuration and SHA-256 identifiers are included, and
the package can regenerate them from the public inputs. Tabular evaluation
grids contain no genomic sequences or participant-level data and are included
under `results/`.

## Included Representative Subset

`data/representative/etgp_metadata_subset.tsv` contains one positive and one
negative ETGP metadata row from each official split. The rows come from
DNALongBench data-file ID `10443715` and remain under CC0 1.0. The associated
450-kb reference sequence is omitted; the subset documents schema and labels,
while the downloader retrieves the complete reproducible input.
