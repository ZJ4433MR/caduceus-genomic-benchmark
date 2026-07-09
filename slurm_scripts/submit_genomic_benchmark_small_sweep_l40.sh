#!/bin/bash

# Submit the GenomicBenchmarks comparison sweep to L40.
# Each training job runs the 5 original split seeds; each evaluation job depends
# on its matching training job and runs normal_test plus flipped_test.

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

TASKS="${TASKS:-dummy_mouse_enhancers_ensembl demo_coding_vs_intergenomic_seqs demo_human_or_worm human_enhancers_cohn human_enhancers_ensembl human_ensembl_regulatory human_nontata_promoters human_ocr_ensembl}"
MODEL_VARIANTS="${MODEL_VARIANTS:-mlbn_pretrained ps_hf ph_hf}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
SEEDS="${SEEDS:-1 2 3 4 5}"
GB_DATA_DIR="${GB_DATA_DIR:-${HOME}/datahome/caduceus}"
MLBN_PRETRAINED_PATH="${MLBN_PRETRAINED_PATH:-/host/outputs/pretrain/hg38/mlbn_lm_seqlen-1k_d_model-80_n_layer-3_lr-1.2e-3_dp-0.0_paramfair_lr1p2em3_dp0p0/checkpoints/last.ckpt}"
HF_CACHE_DIR="${HF_CACHE_DIR:-${GB_DATA_DIR}/.cache/huggingface}"

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

echo "Submitting GenomicBenchmarks small sweep"
echo "Tasks: ${TASKS}"
echo "Model variants: ${MODEL_VARIANTS}"
echo "Seeds: ${SEEDS}"
echo "Max epochs: ${MAX_EPOCHS}"
echo "Data dir: ${GB_DATA_DIR}"
echo "MLBN pretrained path: ${MLBN_PRETRAINED_PATH}"
echo "HF cache dir: ${HF_CACHE_DIR}"

for task in ${TASKS}; do
  if [ ! -d "${GB_DATA_DIR}/${task}" ]; then
    echo "Missing dataset directory: ${GB_DATA_DIR}/${task}" >&2
    exit 1
  fi

  task_short="$(short_task_name "${task}")"
  for variant in ${MODEL_VARIANTS}; do
    train_name="gb_${task_short}_${variant}"
    eval_name="gb_eval_${task_short}_${variant}"

    train_job_id="$(
      sbatch --parsable \
        --job-name="${train_name}" \
        --export="ALL,TASK=${task},MODEL_VARIANT=${variant},MAX_EPOCHS=${MAX_EPOCHS},GB_DATA_DIR=${GB_DATA_DIR},MLBN_PRETRAINED_PATH=${MLBN_PRETRAINED_PATH},HF_CACHE_DIR=${HF_CACHE_DIR}" \
        slurm_scripts/run_genomic_benchmark_small_l40.sh
    )"

    eval_job_id="$(
      sbatch --parsable \
        --dependency="afterok:${train_job_id}" \
        --job-name="${eval_name}" \
        --export="ALL,TASK=${task},MODEL_VARIANTS=${variant},SOURCE_JOB_ID=${train_job_id},GB_DATA_DIR=${GB_DATA_DIR},HF_CACHE_DIR=${HF_CACHE_DIR}" \
        slurm_scripts/run_genomic_benchmark_reversal_test_small_l40.sh
    )"

    echo "task=${task} variant=${variant} train_job=${train_job_id} eval_job=${eval_job_id}"
  done
done
