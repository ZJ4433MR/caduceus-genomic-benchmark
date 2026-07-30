#!/usr/bin/env bash
set -Eeuo pipefail

PROFILE="${1:-gb}"
PYTHON="${PYTHON:-python}"
HG38_DATA_ROOT="${HG38_DATA_ROOT:-data/raw/hg38}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/pretraining}"
PRETRAINED_CHECKPOINT="${PRETRAINED_CHECKPOINT:-}"

case "${PROFILE}" in
  gb)
    MODEL=flipped_gari_lm_gb
    CONTEXT=1024
    STEPS=16000
    LR=1.5e-3
    WARMUP=1600
    GLOBAL_BATCH=1024
    MICRO_BATCH=32
    DEVICES=4
    PRECISION=16
    CLIP=1.0
    ;;
  vep)
    MODEL=flipped_gari_lm_vep
    CONTEXT=1024
    STEPS=16000
    LR=1e-4
    WARMUP=3200
    GLOBAL_BATCH=1024
    MICRO_BATCH=8
    DEVICES=8
    PRECISION=32
    CLIP=0.5
    ;;
  etgp)
    MODEL=flipped_gari_lm_etgp
    CONTEXT=1024
    STEPS=16000
    LR=1e-4
    WARMUP=3200
    GLOBAL_BATCH=1024
    MICRO_BATCH=8
    DEVICES=8
    PRECISION=32
    CLIP=0.5
    ;;
  etgp_2kb)
    MODEL=flipped_gari_lm_etgp
    CONTEXT=2048
    STEPS=2000
    LR=1e-4
    WARMUP=100
    GLOBAL_BATCH=512
    MICRO_BATCH=4
    DEVICES=4
    PRECISION=32
    CLIP=0.5
    ;;
  etgp_5kb)
    MODEL=flipped_gari_lm_etgp
    CONTEXT=5120
    STEPS=5000
    LR=1e-4
    WARMUP=100
    GLOBAL_BATCH=200
    MICRO_BATCH=1
    DEVICES=4
    PRECISION=32
    CLIP=0.5
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    exit 2
    ;;
esac

ARGS=(
  -m train
  experiment=hg38/hg38
  model="${MODEL}"
  dataset.bed_file="${HG38_DATA_ROOT}/human-sequences.bed"
  dataset.fasta_file="${HG38_DATA_ROOT}/hg38.ml.fa"
  dataset.max_length="${CONTEXT}"
  dataset.max_length_val="${CONTEXT}"
  dataset.max_length_test="${CONTEXT}"
  dataset.batch_size="${MICRO_BATCH}"
  dataset.batch_size_eval="${MICRO_BATCH}"
  dataset.mlm=true
  dataset.mlm_probability=0.15
  dataset.rc_aug=false
  dataset.plain_flip_aug=false
  optimizer.lr="${LR}"
  optimizer.weight_decay=0.1
  train.global_batch_size="${GLOBAL_BATCH}"
  train.seed=2222
  trainer.devices="${DEVICES}"
  trainer.num_nodes=1
  trainer.max_steps="${STEPS}"
  trainer.precision="${PRECISION}"
  trainer.gradient_clip_val="${CLIP}"
  scheduler.warmup_t="${WARMUP}"
  wandb=null
  hydra.run.dir="${OUTPUT_ROOT}/${PROFILE}"
)

if [[ "${PROFILE}" == etgp_2kb || "${PROFILE}" == etgp_5kb ]]; then
  if [[ -z "${PRETRAINED_CHECKPOINT}" ]]; then
    echo "PRETRAINED_CHECKPOINT is required for ${PROFILE}" >&2
    exit 2
  fi
  ARGS+=(
    train.ckpt=null
    train.pretrained_model_path="${PRETRAINED_CHECKPOINT}"
    train.pretrained_model_strict_load=true
    scheduler.t_initial=8000
    scheduler.lr_min=1e-5
  )
fi

"${PYTHON}" "${ARGS[@]}"
