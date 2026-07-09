# Reproduction Guide

This guide maps each paper table or figure to the code and files needed to
recreate it.

## Setup

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
make smoke
```

If CUDA extension wheels for `mamba-ssm`, `causal-conv1d`, or `flash-attn` are
not available from pip for the review machine, install wheels matching the local
CUDA/PyTorch environment before launching full experiments.

## Data

Prepare datasets as described in `data/README.md`.

The primary data groups are:

- GenomicBenchmarks legacy downstream tasks.
- Caduceus long-range variant-effect prediction task.
- DNALongBench eQTL and ETGP 450 kb classification tasks.

Run data preparation helpers where applicable:

```bash
bash scripts/download_hg38_pretrain_data.sh
python scripts/download_dnalongbench_450k_data.py --help
```

## Table 2: Main Normal/Flipped Results

The main result table compares MLBN-1k with public Caduceus-Ph-131k and
Caduceus-PS-131k under normal and flipped evaluation.

Relevant files:

- `configs/model/mlbn_dna.yaml`
- `configs/model/caduceus_hf.yaml`
- `configs/dataset/genomic_benchmark.yaml`
- `configs/dataset/dnalongbench_450k.yaml`
- `scripts/evaluate_genomic_benchmark_split.py`
- `scripts/evaluate_genomic_benchmark_flip_val.py`
- `scripts/evaluate_genomic_benchmark_global_metrics.py`
- `slurm_scripts/submit_genomic_benchmark_small_sweep_l40.sh`
- `slurm_scripts/run_genomic_benchmark_reversal_test_small_l40.sh`
- `slurm_scripts/run_dnalongbench450k.sh`

Aggregate raw CSV files into a paper-style table:

```bash
python scripts/aggregate_results.py \
  --input results/raw/table2_runs.csv \
  --output results/tables/table2_main_results.csv
```

Expected input schema is documented in `results/README.md`.

## Table 3: Controlled Same-Budget Comparison

This table is the fair mechanism comparison:

- Mamba2-1k
- Mamba2-RevPh-1k
- MLBN-1k

Relevant configs:

- `configs/model/mamba_dna.yaml`
- `configs/model/mlbn_dna.yaml`
- `configs/model/mlbn_lm.yaml`
- `configs/pipeline/hg38.yaml`
- `configs/experiment/hg38/hg38.yaml`

Recommended aggregation:

```bash
python scripts/aggregate_results.py \
  --input results/raw/table3_same_budget_runs.csv \
  --output results/tables/table3_same_budget_controls.csv
```

## Table 4: Ablation

The main ablations should isolate whether the gain comes from layer-wise
directional interaction rather than merely seeing both orientations.

Minimum ablations:

- Full MLBN.
- Without cross-attention.
- Without discrepancy-aware fusion.

Optional appendix ablations:

- Without gradient equilibrium.
- Without orientation-aware local processing.
- Simple average fusion instead of discrepancy-aware fusion.

Recommended aggregation:

```bash
python scripts/aggregate_results.py \
  --input results/raw/table4_ablation_runs.csv \
  --output results/tables/table4_ablation.csv
```

## Figure Inputs

Paper figures should be generated from the same table CSVs, not from manually
copied numbers.

- Fig. 2 uses `results/tables/table2_main_results.csv`.
- Fig. 3 uses a long-format file with sequence length or dependency distance,
  model, orientation, metric, and value.
- Appendix variance plots use seed-level raw files from `results/raw/`.

## Notes For Reviewers

Large-scale pretraining and 450 kb sequence finetuning may require cluster GPU
resources. The package therefore provides:

- Full code and configs.
- Slurm scripts used for large experiments.
- A smoke test for environment sanity.
- A documented raw-result schema for reconstructing paper tables.

