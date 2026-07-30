#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python}"
TASK="${TASK:?Set TASK to one of the eight GenomicBenchmarks task names}"
SEED="${SEED:?Set SEED to one of 1,2,3,4,5}"
LR="${LR:?Set the task-specific learning rate from docs/HYPERPARAMETERS.md}"
BATCH_SIZE="${BATCH_SIZE:?Set the task-specific batch size}"
GB_DATA_ROOT="${GB_DATA_ROOT:-data/raw/genomic_benchmarks}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:?Set the 1-kb GB checkpoint path}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/genomicbenchmarks/${TASK}/seed-${SEED}}"

EXTRA=()
if [[ "${TASK}" == "human_ocr_ensembl" ]]; then
  EXTRA+=(model.dropout_rate1=0.05 model.dropout_rate2=0.05 model.embed_dropout=0.05)
fi

"${PYTHON}" -m train \
  experiment=hg38/genomic_benchmark \
  model=flipped_gari_gb \
  dataset.dataset_name="${TASK}" \
  dataset.dest_path="${GB_DATA_ROOT}" \
  dataset.train_val_split_seed="${SEED}" \
  dataset.batch_size="${BATCH_SIZE}" \
  dataset.rc_aug=false \
  +dataset.conjoin_train=false \
  +dataset.conjoin_test=false \
  optimizer.lr="${LR}" \
  optimizer.weight_decay=0.01 \
  trainer.devices=1 \
  trainer.max_epochs=10 \
  trainer.precision=32 \
  train.seed="${SEED}" \
  train.global_batch_size="${BATCH_SIZE}" \
  train.pretrained_model_path="${PRETRAINED_CHECKPOINT}" \
  train.pretrained_model_state_hook._name_=load_backbone \
  train.pretrained_model_state_hook.freeze_backbone=false \
  train.test=true \
  wandb=null \
  hydra.run.dir="${OUTPUT_DIR}" \
  "${EXTRA[@]}"

SELECTED_CHECKPOINT="${SELECTED_CHECKPOINT:-${OUTPUT_DIR}/checkpoints/val/accuracy.ckpt}"

for view in original_order sequence_reversed; do
  reverse=false
  if [[ "${view}" == sequence_reversed ]]; then
    reverse=true
  fi
  "${PYTHON}" scripts/evaluate_genomic_benchmark_split.py \
    experiment=hg38/genomic_benchmark \
    model=flipped_gari_gb \
    dataset.dataset_name="${TASK}" \
    dataset.dest_path="${GB_DATA_ROOT}" \
    dataset.train_val_split_seed="${SEED}" \
    dataset.reverse_test="${reverse}" \
    ++eval.split=test \
    ++eval.ckpt_path="${SELECTED_CHECKPOINT}" \
    trainer.devices=1 \
    wandb=null
done
