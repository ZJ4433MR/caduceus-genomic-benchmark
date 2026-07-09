#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=L40
#SBATCH -t 96:00:00
#SBATCH --mem=100G
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=vep_mlbn_embed
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

set -Eeo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${rc}" >&2' ERR

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [ -f "${SUBMIT_DIR}/setup_env.sh" ]; then
  REPO_ROOT="${SUBMIT_DIR}"
elif [ -f "${SUBMIT_DIR}/../setup_env.sh" ]; then
  REPO_ROOT="$(cd "${SUBMIT_DIR}/.." && pwd)"
else
  echo "Could not find setup_env.sh from submit directory: ${SUBMIT_DIR}" >&2
  exit 1
fi

cd "${REPO_ROOT}" || exit
set -u
export HYDRA_FULL_ERROR=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

CONTAINER_DIR="${CONTAINER_DIR:-/path/to/shared_containers}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-caduceus-pytorch-2.2.0-cuda12.1-cudnn8-devel.sif}"
SIF_IMAGE="${SIF_IMAGE:-${CONTAINER_DIR}/${PYTORCH_IMAGE}}"
VENV_PYTHON="${VENV_PYTHON:-/host/.venv/bin/python}"

if ! command -v module >/dev/null 2>&1; then
  if [ -f /etc/profile.d/modules.sh ]; then
    source /etc/profile.d/modules.sh || true
  elif [ -f /usr/share/Modules/init/bash ]; then
    source /usr/share/Modules/init/bash || true
  fi
fi
if command -v module >/dev/null 2>&1; then
  module load Singularity/4.2.1 || true
fi
if ! command -v singularity >/dev/null 2>&1; then
  echo "singularity command not found. Try loading the Singularity module on this cluster." >&2
  exit 1
fi
if [ ! -f "${SIF_IMAGE}" ]; then
  echo "Singularity image not found: ${SIF_IMAGE}" >&2
  exit 1
fi

SEQ_LEN="${SEQ_LEN:-131072}"
BP_PER_TOKEN="${BP_PER_TOKEN:-1}"
NUM_DEVICES="${NUM_DEVICES:-1}"
NUM_WORKERS="${NUM_WORKERS:-0}"
EMBED_DUMP_BATCH_SIZE="${EMBED_DUMP_BATCH_SIZE:-1}"
DOWNSTREAM_SAVE_DIR="${DOWNSTREAM_SAVE_DIR:-./outputs/downstream/vep_embeddings}"
MLBN_CONFIG="${MLBN_CONFIG:-configs/model/mlbn_dna.yaml}"
MLBN_CHECKPOINT="${MLBN_CHECKPOINT:-outputs/pretrain/hg38/mlbn_lm_seqlen-1k_d_model-80_n_layer-3_lr-1.2e-3_dp-0.0_pt_lr1p2em3_wd0p1_ms16k_wu1600_dp0/checkpoints/last.ckpt}"
VEP_TASK_NAME="${VEP_TASK_NAME:-variant_effect_causal_eqtl}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-4}"
DOWNLOAD_TIMEOUT="${DOWNLOAD_TIMEOUT:-300}"
TOKENIZE_BATCH_SIZE="${TOKENIZE_BATCH_SIZE:-16}"
STREAM_TOKENIZE="${STREAM_TOKENIZE:-true}"
RUN_NAME="${RUN_NAME:-mlbn_d80l3_pt1p2wd0p1_1k_downstream-seqlen=131k_probe}"
JOB_CACHE_ROOT="${JOB_CACHE_ROOT:-${SLURM_TMPDIR:-/tmp/${USER}/caduceus-vep-${SLURM_JOB_ID:-manual}}}"
JOB_HF_HOME="${JOB_HF_HOME:-${JOB_CACHE_ROOT}/huggingface}"
VEP_REFERENCE_CACHE="${VEP_REFERENCE_CACHE:-${REPO_ROOT}/.cache/huggingface/datasets}"

if [ ! -f "${MLBN_CONFIG}" ]; then
  echo "MLBN config not found: ${MLBN_CONFIG}" >&2
  exit 1
fi
if [ ! -f "${MLBN_CHECKPOINT}" ]; then
  echo "MLBN checkpoint not found: ${MLBN_CHECKPOINT}" >&2
  exit 1
fi
mkdir -p .cache/huggingface/hub .cache/huggingface/datasets .cache/huggingface/transformers .cache/huggingface/modules
mkdir -p "${JOB_HF_HOME}/hub" "${JOB_HF_HOME}/datasets" "${JOB_HF_HOME}/transformers" "${JOB_HF_HOME}/modules"

echo "Resolved REPO_ROOT=${REPO_ROOT}"
echo "SEQ_LEN=${SEQ_LEN}, BP_PER_TOKEN=${BP_PER_TOKEN}, NUM_DEVICES=${NUM_DEVICES}"
echo "RUN_NAME=${RUN_NAME}"
echo "VEP_TASK_NAME=${VEP_TASK_NAME}"
echo "DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}, DOWNLOAD_TIMEOUT=${DOWNLOAD_TIMEOUT}"
echo "TOKENIZE_BATCH_SIZE=${TOKENIZE_BATCH_SIZE}"
echo "STREAM_TOKENIZE=${STREAM_TOKENIZE}"
echo "JOB_CACHE_ROOT=${JOB_CACHE_ROOT}"
echo "JOB_HF_HOME=${JOB_HF_HOME}"
echo "VEP_REFERENCE_CACHE=${VEP_REFERENCE_CACHE}"
echo "CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
echo "MLBN_CONFIG=${MLBN_CONFIG}"
echo "MLBN_CHECKPOINT=${MLBN_CHECKPOINT}"
echo "NOTE: This is an MLBN small-model VEP probe. It is not parameter-matched to the original 131k Caduceus PS/PH VEP models."

singularity exec --nv --pwd /host \
  --bind "${REPO_ROOT}:/host:rw" \
  --bind "${JOB_CACHE_ROOT}:${JOB_CACHE_ROOT}:rw" \
  "${SIF_IMAGE}" \
  env PYTHONPATH=/host \
  HF_HOME="${JOB_HF_HOME}" \
  HUGGINGFACE_HUB_CACHE="${JOB_HF_HOME}/hub" \
  HF_DATASETS_CACHE="${JOB_HF_HOME}/datasets" \
  HF_MODULES_CACHE="${JOB_HF_HOME}/modules" \
  TRANSFORMERS_CACHE="${JOB_HF_HOME}/transformers" \
  XDG_CACHE_HOME="${JOB_CACHE_ROOT}/xdg" \
  VEP_REFERENCE_CACHE="/host/.cache/huggingface/datasets" \
  CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}" \
  "${VENV_PYTHON}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc-per-node="${NUM_DEVICES}" \
    vep_embeddings.py \
      --num_workers="${NUM_WORKERS}" \
      --seq_len="${SEQ_LEN}" \
      --bp_per_token="${BP_PER_TOKEN}" \
      --embed_dump_batch_size="${EMBED_DUMP_BATCH_SIZE}" \
      --downstream_save_dir="${DOWNSTREAM_SAVE_DIR}" \
      --name="${RUN_NAME}" \
      --model_name_or_path="mlbn-local" \
      --mlbn_config="${MLBN_CONFIG}" \
      --mlbn_checkpoint="${MLBN_CHECKPOINT}" \
      --vep_task_name="${VEP_TASK_NAME}" \
      --download_retries="${DOWNLOAD_RETRIES}" \
      --download_timeout="${DOWNLOAD_TIMEOUT}" \
      --tokenize_batch_size="${TOKENIZE_BATCH_SIZE}" \
      $(if [ "${STREAM_TOKENIZE}" = "true" ]; then echo "--stream_tokenize"; fi) \
      --no-rcps
