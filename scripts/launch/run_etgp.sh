#!/usr/bin/env bash
set -Eeuo pipefail

PYTHON="${PYTHON:-python}"
CONTEXT_KB="${CONTEXT_KB:-1}"
SEED="${SEED:?Set SEED to 2222 or 3333}"
ETGP_DATA_ROOT="${ETGP_DATA_ROOT:-data/raw/dnalongbench}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:?Set the Flipped-GARI checkpoint path}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/etgp/context-${CONTEXT_KB}kb/seed-${SEED}}"

if [[ "${SEED}" != 2222 && "${SEED}" != 3333 ]]; then
  echo "SEED must be 2222 or 3333" >&2
  exit 2
fi

EPOCHS=5
DEVICES="${DEVICES:-8}"
if (( 8 % DEVICES != 0 )); then
  echo "DEVICES must divide the fixed global batch size of 8" >&2
  exit 2
fi
ACCUMULATE=$((8 / DEVICES))

case "${CONTEXT_KB}" in
  1)
    BACKBONE_LR=6e-4
    ;;
  2)
    BACKBONE_LR=5e-5
    ;;
  5)
    BACKBONE_LR=2e-5
    ;;
  *)
    echo "CONTEXT_KB must be 1, 2, or 5" >&2
    exit 2
    ;;
esac

"${PYTHON}" -m train \
  experiment=hg38/dnalongbench_etgp_flipped_gari_short_transfer \
  model=flipped_gari_etgp \
  dataset.data_root="${ETGP_DATA_ROOT}" \
  dataset.max_length=450000 \
  dataset.batch_size=1 \
  dataset.batch_size_eval=1 \
  dataset.rc_aug=false \
  dataset.reverse_aug=false \
  trainer.devices="${DEVICES}" \
  trainer.num_nodes=1 \
  trainer.accumulate_grad_batches="${ACCUMULATE}" \
  trainer.max_epochs="${EPOCHS}" \
  trainer.precision=bf16 \
  trainer.gradient_clip_val=1.0 \
  train.seed="${SEED}" \
  train.global_batch_size=8 \
  train.backbone_lr="${BACKBONE_LR}" \
  train.pretrained_model_path="${PRETRAINED_CHECKPOINT}" \
  train.pretrained_model_state_hook._name_=load_backbone \
  train.pretrained_model_state_hook.freeze_backbone=false \
  optimizer.lr=6e-4 \
  optimizer.weight_decay=0.1 \
  train.test=false \
  wandb=null \
  hydra.run.dir="${OUTPUT_DIR}"

echo "Evaluate each validation checkpoint with scripts/evaluate_dnalongbench450k_global_metrics.py."
echo "Select by full-split validation AUROC, then evaluate the selected checkpoint once on test."
