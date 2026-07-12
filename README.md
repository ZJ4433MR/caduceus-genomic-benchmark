# Anonymous Review Package

This repository contains the anonymous code package for an AAAI-27 submission on
reducing scan-direction bias in long-range DNA sequence models with a
mutual-learning bidirectional Mamba encoder.

The package is organized to support review-time reproducibility. It includes the
model code, training and evaluation configs, data-preparation scripts, Slurm
launchers used for large runs, and lightweight aggregation utilities for
reconstructing the paper tables from raw result files.

## Main Claim Supported By This Package

The paper studies pure sequence order reversal as a controllable diagnostic of
the order component underlying reverse-complement processing. The method does
not claim full reverse-complement equivariance. Instead, it tests whether
layer-wise directional communication can reduce scan-direction bias under
normal and flipped evaluation.

The central empirical comparisons are:

- Published strong-baseline comparison:
  MLBN-1k versus public Caduceus-Ph-131k and Caduceus-PS-131k.
- Controlled same-budget comparison:
  Mamba2-1k versus Mamba2-RevPh-1k versus MLBN-1k.
- Long-range external validation:
  DNALongBench eQTL and ETGP, both using 450 kb sequence inputs.

## Repository Map

```text
configs/          Hydra configs for pretraining, finetuning, and evaluation.
caduceus/         Caduceus model/tokenizer components reused as baselines.
src/              MLBN, dataloaders, metrics, tasks, and training utilities.
scripts/          Data preparation, evaluation, and table aggregation scripts.
slurm_scripts/    Cluster launchers used for full-scale experiments.
data/             Dataset manifest and expected local data layout.
results/          Expected raw/processed/table/figure result layout.
checkpoints/      Placeholder for local checkpoints or reviewer-supplied paths.
docs/             Baselines, hardware, hyperparameters, and anonymization notes.
train.py          Main Hydra training entry point.
```

## Quick Start

```bash
conda env create -f caduceus_env.yml
conda activate caduceus_env
python scripts/test_imports.sh
```

For a minimal smoke test that checks imports and config availability:

```bash
make smoke
```

Full pretraining and 450 kb long-range finetuning require GPU resources and are
normally launched through the scripts in `slurm_scripts/`. The commands and
expected result formats are described in `REPRODUCE.md`.

## Ablation Handoff

For the teammate implementing and running the MLBN ablations, start with
[`ABLATION_HANDOFF.md`](ABLATION_HANDOFF.md). It documents:

- the exact baseline code and configuration files;
- the required structural ablations and their code locations;
- checkpoint rules for structural and inference-only ablations;
- a low-cost three-seed screening stage and five-seed final stage;
- Slurm command templates for pretraining, downstream training, and paired
  normal/flipped evaluation;
- the required result schema and handoff checklist.

This branch intentionally contains code and lightweight configuration only.
Datasets, checkpoints, caches, generated outputs, and local cluster paths must
be shared separately and must not be committed to Git.

## What Is Not Included

The archive does not include large raw genomic datasets, pretrained checkpoint
files, or private cluster logs. Dataset acquisition and expected directory
layouts are documented in `data/README.md`, and checkpoint path conventions are
documented in `checkpoints/README.md`.

No author names, affiliations, private usernames, private GitHub URLs, or local
absolute paths are intentionally included in this branch.
