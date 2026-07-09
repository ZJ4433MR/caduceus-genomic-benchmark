#!/bin/bash

# Submit a compact MLBN pre-training LR sweep for the hg38 1k setup used
# before GenomicBenchmarks fine-tuning.

set -Eeo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit

LRS="${LRS:-8e-3 3e-3 1e-3}"
DROP_PATH_RATES="${DROP_PATH_RATES:-0.0}"
D_MODEL="${D_MODEL:-80}"
N_LAYER="${N_LAYER:-3}"
NUM_HEADS="${NUM_HEADS:-4}"
SEQLEN="${SEQLEN:-1024}"
MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
HG38_DATA_DIR="${HG38_DATA_DIR:-${HOME}/datahome/caduceus/hg38}"
SBATCH_PARTITION="${SBATCH_PARTITION:-}"
SBATCH_GRES="${SBATCH_GRES:-}"

echo "Submitting MLBN hg38 pre-training sweep"
echo "LRS=${LRS}"
echo "DROP_PATH_RATES=${DROP_PATH_RATES}"
echo "D_MODEL=${D_MODEL}, N_LAYER=${N_LAYER}, NUM_HEADS=${NUM_HEADS}"
echo "SEQLEN=${SEQLEN}, MAX_STEPS=${MAX_STEPS}, GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "HG38_DATA_DIR=${HG38_DATA_DIR}"
echo "SBATCH_PARTITION=${SBATCH_PARTITION:-script default}"
echo "SBATCH_GRES=${SBATCH_GRES:-script default}"

SBATCH_ARGS=()
if [ -n "${SBATCH_PARTITION}" ]; then
  SBATCH_ARGS+=(--partition="${SBATCH_PARTITION}")
fi
if [ -n "${SBATCH_GRES}" ]; then
  SBATCH_ARGS+=(--gres="${SBATCH_GRES}")
fi

for lr in ${LRS}; do
  for drop_path in ${DROP_PATH_RATES}; do
    lr_tag="${lr//./p}"
    lr_tag="${lr_tag//-}"
    dp_tag="${drop_path//./p}"
    run_tag="sweep_lr${lr_tag}_dp${dp_tag}"
    job_name="pre_mlbn_${run_tag}"
    sbatch \
      "${SBATCH_ARGS[@]}" \
      --job-name="${job_name}" \
      --export="ALL,LR=${lr},DROP_PATH_RATE=${drop_path},D_MODEL=${D_MODEL},N_LAYER=${N_LAYER},NUM_HEADS=${NUM_HEADS},SEQLEN=${SEQLEN},MAX_STEPS=${MAX_STEPS},GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE},BATCH_SIZE=${BATCH_SIZE},HG38_DATA_DIR=${HG38_DATA_DIR},RUN_TAG=${run_tag}" \
      slurm_scripts/run_pretrain_mlbn_a800.sh
  done
done
