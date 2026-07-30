# Reproducing the Experiments

Commands below assume the environment from `INSTALLATION.md` and a Linux host.
Large profiles require multiple GPUs. Replace only path variables; keep the
listed scientific overrides fixed when reproducing reported settings.

## 1. Prepare Public Data

```bash
bash scripts/download_hg38_pretrain_data.sh
python scripts/download_genomicbenchmarks_data.py --root data/raw/genomic_benchmarks
python scripts/download_dnalongbench_450k_data.py --root data/raw/dnalongbench
```

The VEP embedding script loads
`InstaDeepAI/genomics-long-range-benchmark`, task
`variant_effect_causal_eqtl`, through Hugging Face Datasets and builds the
reference sequence from hg38. Dataset identifiers and licenses are in
`data/manifests/datasets.csv`.

## 2. Pretrain Flipped-GARI

The pretraining objective is masked-nucleotide prediction with masking
probability 0.15. No sequence-reversal or reverse-complement augmentation is
used.

```bash
bash scripts/launch/pretrain_flipped_gari.sh gb
bash scripts/launch/pretrain_flipped_gari.sh vep
bash scripts/launch/pretrain_flipped_gari.sh etgp
```

The 2-kb and 5-kb ETGP profiles continue from the 1-kb ETGP checkpoint:

```bash
PRETRAINED_CHECKPOINT=/path/to/flipped_gari_etgp_1kb.ckpt \
  bash scripts/launch/pretrain_flipped_gari.sh etgp_2kb

PRETRAINED_CHECKPOINT=/path/to/flipped_gari_etgp_1kb.ckpt \
  bash scripts/launch/pretrain_flipped_gari.sh etgp_5kb
```

## 3. GenomicBenchmarks

Run one task and split seed:

```bash
PRETRAINED_CHECKPOINT=/path/to/flipped_gari_gb_1kb.ckpt \
GB_DATA_ROOT=data/raw/genomic_benchmarks \
TASK=demo_coding_vs_intergenomic_seqs \
SEED=1 LR=1e-3 BATCH_SIZE=256 \
  bash scripts/launch/run_genomicbenchmarks.sh
```

The script trains once, selects the checkpoint by original-order validation
accuracy, and evaluates that checkpoint on original-order and
sequence-reversed test inputs. The eight task names, selected learning rates,
batch sizes, and readout exception for OCR are in
`docs/HYPERPARAMETERS.md`.

`results/raw/genomicbenchmarks/fixed_profile_seed_metrics.csv` contains all
five seeds for a single fixed profile. The submitted Flipped-GARI row is a
task-wise summary over formal configurations; its exact aggregate and
selection labels are in `results/processed/genomicbenchmarks/`.

## 4. Variant-Effect Prediction

Dump original-order and sequence-reversed embeddings from a frozen 1-kb
checkpoint at 131-kb input:

```bash
FLIPPED_GARI_CHECKPOINT=/path/to/flipped_gari_vep_1kb.ckpt \
  bash scripts/launch/run_vep.sh embeddings_primary
```

Run the retained ten-subset hard-label SVC protocol:

```bash
EMBED_ROOT=outputs/vep/flipped_gari_131kb \
  bash scripts/launch/run_vep.sh probe_primary
```

For the input-context study, run `embeddings_context_1kb`,
`embeddings_context_16kb`, and `embeddings_context_131kb`. These profiles set
`--window_size_bp=1024`, so the 1-kb condition never pads, tiles, or repeats
boundary tokens. Primary and pretraining-comparison profiles use a 1,536-bp
readout on 131-kb inputs.

The retained benchmark protocol scans
`C={0.1,0.3,1,3,5,10,30,100}` for each stated condition. The component
diagnostic additionally transfers the original-order-selected value unchanged
to sequence-reversed evaluation. Raw grids and summaries are under
`results/raw/vep/` and `results/processed/vep/`.

## 5. Direct 450-kb ETGP

```bash
PRETRAINED_CHECKPOINT=/path/to/flipped_gari_etgp_1kb.ckpt \
CONTEXT_KB=1 SEED=2222 \
  bash scripts/launch/run_etgp.sh
```

For the context variants, use `CONTEXT_KB=2` or `5` and the corresponding
checkpoint. The launcher applies the task-head and backbone learning rates
documented in `docs/HYPERPARAMETERS.md`, consumes the complete 450-kb
sequence, and computes metrics from continuous scores over the full split.

## 6. Verify Packaged Numbers

```bash
python scripts/verify_reported_results.py
```

This recomputes table aggregates from the included CSV files and prints a
separate ETGP provenance reconciliation. It never silently substitutes a
displayed value for a raw-run aggregate.
