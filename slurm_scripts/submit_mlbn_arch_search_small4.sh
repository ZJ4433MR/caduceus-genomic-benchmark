#!/bin/bash

# Submit MLBN architecture-search pretraining jobs, then evaluate each
# successful checkpoint on a compact four-task GenomicBenchmarks subset.

set -Eeo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${rc}" >&2' ERR

SUBMIT_DIR="$(pwd)"
if [ -f "${SUBMIT_DIR}/setup_env.sh" ]; then
  REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../setup_env.sh" ]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Run this script from the repository root or slurm_scripts/." >&2
  exit 1
fi

cd "${REPO_ROOT}" || exit

ARCHES="${ARCHES:-72:4 96:2 56:7 64:5}"
TASKS="${TASKS:-dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn}"
SEARCH_TAG="${SEARCH_TAG:-}"
PRETRAIN_LR="${PRETRAIN_LR:-1.2e-3}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.0}"
MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
PRETRAIN_BATCH_SIZE="${PRETRAIN_BATCH_SIZE:-32}"
PRETRAIN_NUM_DEVICES="${PRETRAIN_NUM_DEVICES:-8}"
PRETRAIN_PARTITION="${PRETRAIN_PARTITION:-}"
PRETRAIN_GRES="${PRETRAIN_GRES:-}"
PRETRAIN_NTASKS_PER_NODE="${PRETRAIN_NTASKS_PER_NODE:-}"
PRETRAIN_MEM="${PRETRAIN_MEM:-}"
PRETRAIN_TIME="${PRETRAIN_TIME:-}"
PRETRAIN_EXCLUDE="${PRETRAIN_EXCLUDE:-}"
DOWNSTREAM_MAX_EPOCHS="${DOWNSTREAM_MAX_EPOCHS:-10}"
GB_DATA_DIR="${GB_DATA_DIR:-${HOME}/datahome/caduceus}"
HG38_DATA_DIR="${HG38_DATA_DIR:-${HOME}/datahome/caduceus/hg38}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${GB_DATA_DIR}/.cache/huggingface}"
NUM_HEADS="${NUM_HEADS:-4}"

short_task_name() {
  case "$1" in
    demo_coding_vs_intergenomic_seqs) echo "coding_intergenic" ;;
    demo_human_or_worm) echo "human_worm" ;;
    human_enhancers_cohn) echo "enh_cohn" ;;
    human_enhancers_ensembl) echo "enh_ensembl" ;;
    human_ensembl_regulatory) echo "regulatory" ;;
    human_nontata_promoters) echo "nontata" ;;
    human_ocr_ensembl) echo "ocr" ;;
    dummy_mouse_enhancers_ensembl) echo "mouse" ;;
    *) echo "$1" | tr '_' '-' | cut -c1-24 ;;
  esac
}

lr_tag="${PRETRAIN_LR//./p}"
lr_tag="${lr_tag//-/m}"
dp_tag="${DROP_PATH_RATE//./p}"

echo "Submitting MLBN architecture search"
echo "ARCHES=${ARCHES}"
echo "TASKS=${TASKS}"
echo "SEARCH_TAG=${SEARCH_TAG:-none}"
echo "PRETRAIN_LR=${PRETRAIN_LR}, DROP_PATH_RATE=${DROP_PATH_RATE}, MAX_STEPS=${MAX_STEPS}"
echo "PRETRAIN_NUM_DEVICES=${PRETRAIN_NUM_DEVICES}, PRETRAIN_BATCH_SIZE=${PRETRAIN_BATCH_SIZE}"
echo "GB_DATA_DIR=${GB_DATA_DIR}"
echo "HG38_DATA_DIR=${HG38_DATA_DIR}"

PRETRAIN_SBATCH_ARGS=()
if [ -n "${PRETRAIN_PARTITION}" ]; then
  PRETRAIN_SBATCH_ARGS+=(--partition="${PRETRAIN_PARTITION}")
fi
if [ -n "${PRETRAIN_GRES}" ]; then
  PRETRAIN_SBATCH_ARGS+=(--gres="${PRETRAIN_GRES}")
fi
if [ -n "${PRETRAIN_NTASKS_PER_NODE}" ]; then
  PRETRAIN_SBATCH_ARGS+=(--ntasks-per-node="${PRETRAIN_NTASKS_PER_NODE}")
fi
if [ -n "${PRETRAIN_MEM}" ]; then
  PRETRAIN_SBATCH_ARGS+=(--mem="${PRETRAIN_MEM}")
fi
if [ -n "${PRETRAIN_TIME}" ]; then
  PRETRAIN_SBATCH_ARGS+=(-t "${PRETRAIN_TIME}")
fi
if [ -n "${PRETRAIN_EXCLUDE}" ]; then
  PRETRAIN_SBATCH_ARGS+=(--exclude="${PRETRAIN_EXCLUDE}")
fi

for arch in ${ARCHES}; do
  d_model="${arch%%:*}"
  n_layer="${arch##*:}"
  if [ "${d_model}" = "${n_layer}" ]; then
    echo "Invalid arch '${arch}', expected d_model:n_layer" >&2
    exit 1
  fi

  if [ -n "${SEARCH_TAG}" ]; then
    run_tag="archsearch_${SEARCH_TAG}_d${d_model}_l${n_layer}_lr${lr_tag}_dp${dp_tag}"
    display_name="mlbn_pretrained_arch_${SEARCH_TAG}_d${d_model}_l${n_layer}"
  else
    run_tag="archsearch_d${d_model}_l${n_layer}_lr${lr_tag}_dp${dp_tag}"
    display_name="mlbn_pretrained_arch_d${d_model}_l${n_layer}"
  fi
  pretrain_wandb_name="mlbn_lm_seqlen-1k_d_model-${d_model}_n_layer-${n_layer}_lr-${PRETRAIN_LR}_dp-${DROP_PATH_RATE}_${run_tag}"
  pretrain_ckpt="/host/outputs/pretrain/hg38/${pretrain_wandb_name}/checkpoints/last.ckpt"

  pretrain_job_id="$(
    sbatch --parsable \
      "${PRETRAIN_SBATCH_ARGS[@]}" \
      --job-name="pre_mlbn_d${d_model}_l${n_layer}" \
      --export="ALL,LR=${PRETRAIN_LR},DROP_PATH_RATE=${DROP_PATH_RATE},D_MODEL=${d_model},N_LAYER=${n_layer},NUM_HEADS=${NUM_HEADS},MAX_STEPS=${MAX_STEPS},GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE},BATCH_SIZE=${PRETRAIN_BATCH_SIZE},NUM_DEVICES=${PRETRAIN_NUM_DEVICES},HG38_DATA_DIR=${HG38_DATA_DIR},RUN_TAG=${run_tag}" \
      slurm_scripts/run_pretrain_mlbn_a800.sh
  )"
  echo "arch=${arch} pretrain_job=${pretrain_job_id} ckpt=${pretrain_ckpt}"

  for task in ${TASKS}; do
    if [ ! -d "${GB_DATA_DIR}/${task}" ]; then
      echo "Missing dataset directory: ${GB_DATA_DIR}/${task}" >&2
      exit 1
    fi

    task_short="$(short_task_name "${task}")"
    train_job_id="$(
      sbatch --parsable \
        --dependency="afterok:${pretrain_job_id}" \
        --job-name="gb_${task_short}_mlbn_d${d_model}l${n_layer}" \
        --export="ALL,TASK=${task},MODEL_VARIANT=mlbn_pretrained,DISPLAY_NAME=${display_name},D_MODEL=${d_model},N_LAYER=${n_layer},NUM_HEADS=${NUM_HEADS},MAX_EPOCHS=${DOWNSTREAM_MAX_EPOCHS},GB_DATA_DIR=${GB_DATA_DIR},MLBN_PRETRAINED_PATH=${pretrain_ckpt},HF_CACHE_DIR=${HF_CACHE_DIR}" \
        slurm_scripts/run_genomic_benchmark_small_l40.sh
    )"

    eval_job_id="$(
      sbatch --parsable \
        --dependency="afterok:${train_job_id}" \
        --job-name="gb_eval_${task_short}_d${d_model}l${n_layer}" \
        --export="ALL,TASK=${task},MODEL_VARIANTS=mlbn_pretrained,DISPLAY_NAME=${display_name},D_MODEL=${d_model},N_LAYER=${n_layer},NUM_HEADS=${NUM_HEADS},SOURCE_JOB_ID=${train_job_id},GB_DATA_DIR=${GB_DATA_DIR},HF_CACHE_DIR=${HF_CACHE_DIR}" \
        slurm_scripts/run_genomic_benchmark_reversal_test_small_l40.sh
    )"

    echo "arch=${arch} task=${task} train_job=${train_job_id} eval_job=${eval_job_id}"
  done
done
