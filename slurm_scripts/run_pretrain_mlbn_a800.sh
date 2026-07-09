#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=A800
#SBATCH -t 96:00:00
#SBATCH --mem=896G
#SBATCH --gres=gpu:a800:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=pretrain_mlbn
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Pre-train the MLBN DNA language model on the hg38 data used by the
# first Caduceus downstream experiment (GenomicBenchmarks).

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
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton/${SLURM_JOB_ID:-manual_$$}}"
export HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "${TRITON_CACHE_DIR}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}"

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
if [ ! -f "${REPO_ROOT}/.venv/pyvenv.cfg" ]; then
  echo "Project venv not found under ${REPO_ROOT}/.venv" >&2
  echo "Run 'make install' from ${REPO_ROOT} before submitting this job." >&2
  exit 1
fi

HG38_DATA_DIR="${HG38_DATA_DIR:-${HOME}/datahome/caduceus/hg38}"
HG38_DATA_DIR="$(realpath "${HG38_DATA_DIR}")"
FASTA_FILE="${HG38_DATA_DIR}/hg38.ml.fa"
BED_FILE="${HG38_DATA_DIR}/human-sequences.bed"
if [ ! -f "${FASTA_FILE}" ] || [ ! -f "${BED_FILE}" ]; then
  echo "hg38 pre-training data is missing under ${HG38_DATA_DIR}" >&2
  echo "Expected ${FASTA_FILE} and ${BED_FILE}" >&2
  exit 1
fi

NUM_DEVICES="${NUM_DEVICES:-8}"
SEQLEN="${SEQLEN:-1024}"
MAX_STEPS="${MAX_STEPS:-10000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MICRO_BATCHES_PER_OPT_STEP="$(( (GLOBAL_BATCH_SIZE + (BATCH_SIZE * NUM_DEVICES) - 1) / (BATCH_SIZE * NUM_DEVICES) ))"
DEFAULT_VAL_CHECK_INTERVAL="$(( MAX_STEPS > 1 ? (MAX_STEPS * MICRO_BATCHES_PER_OPT_STEP) / 2 : 1 ))"
if [ "${DEFAULT_VAL_CHECK_INTERVAL}" -lt 1 ]; then
  DEFAULT_VAL_CHECK_INTERVAL=1
fi
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-${DEFAULT_VAL_CHECK_INTERVAL}}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.125}"
MLM_PROBABILITY="${MLM_PROBABILITY:-0.15}"
RC_AUG="${RC_AUG:-false}"
LR="${LR:-8e-3}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
PRECISION="${PRECISION:-16}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-true}"
D_MODEL="${D_MODEL:-80}"
N_LAYER="${N_LAYER:-3}"
KERNEL_SIZE="${KERNEL_SIZE:-4}"
D_STATE="${D_STATE:-16}"
D_CONV="${D_CONV:-4}"
EXPAND="${EXPAND:-2}"
HEAD_DIM="${HEAD_DIM:-16}"
NUM_HEADS="${NUM_HEADS:-4}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.0}"
DROPOUT_RATE1="${DROPOUT_RATE1:-0.0}"
DROPOUT_RATE2="${DROPOUT_RATE2:-0.0}"
EMBED_DROPOUT="${EMBED_DROPOUT:-0.0}"
TRAIN_CKPT="${TRAIN_CKPT:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"
RUN_TAG="${RUN_TAG:-}"

SEQLEN_DIS="${SEQLEN}"
if [ "${SEQLEN}" = "1024" ]; then
  SEQLEN_DIS="1k"
fi
WANDB_NAME="mlbn_lm_seqlen-${SEQLEN_DIS}_d_model-${D_MODEL}_n_layer-${N_LAYER}_lr-${LR}_dp-${DROP_PATH_RATE}"
if [ -n "${RUN_TAG}" ]; then
  WANDB_NAME="${WANDB_NAME}_${RUN_TAG}"
fi
HOST_OUTPUT_DIR="${REPO_ROOT}/outputs/pretrain/hg38/${WANDB_NAME}"
HYDRA_RUN_DIR="/host/outputs/pretrain/hg38/${WANDB_NAME}"
CHECKPOINT_DIR="${HYDRA_RUN_DIR}/checkpoints"
mkdir -p "${HOST_OUTPUT_DIR}/checkpoints" watch_folder

SINGULARITY_BINDS=(--bind "${REPO_ROOT}:/host:rw")
if [[ "${HG38_DATA_DIR}" == "${REPO_ROOT}"/* ]]; then
  CONTAINER_HG38_DATA_DIR="/host/${HG38_DATA_DIR#${REPO_ROOT}/}"
else
  SINGULARITY_BINDS+=(--bind "${HG38_DATA_DIR}:${HG38_DATA_DIR}:rw")
  CONTAINER_HG38_DATA_DIR="${HG38_DATA_DIR}"
fi

SINGULARITY_BASE=(
  singularity exec
  --nv
  --pwd /host
  --env HF_HOME=/host/.cache/huggingface
  --env TRANSFORMERS_CACHE=/host/.cache/huggingface/transformers
  "${SINGULARITY_BINDS[@]}"
  "${SIF_IMAGE}"
)

echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "Resolved REPO_ROOT=${REPO_ROOT}"
echo "Singularity image: ${SIF_IMAGE}"
echo "Container Python executable: ${VENV_PYTHON}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "hg38 host data dir=${HG38_DATA_DIR}"
echo "hg38 container data dir=${CONTAINER_HG38_DATA_DIR}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
"${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
try:
    import mamba_ssm
    print("mamba-ssm:", getattr(mamba_ssm, "__version__", "unknown"))
    print("Mamba2 available:", hasattr(mamba_ssm, "Mamba2"))
except Exception as exc:
    print("mamba-ssm import check failed:", repr(exc))
    raise
PY

if [ ! -f "${FASTA_FILE}.fai" ]; then
  echo "Building pyfaidx index for ${FASTA_FILE}"
  "${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" - <<PY
from pyfaidx import Fasta
Fasta("${CONTAINER_HG38_DATA_DIR}/hg38.ml.fa")
PY
fi

TRAIN_ARGS=(
  -m train
  experiment=hg38/hg38
  callbacks.model_checkpoint.dirpath="${CHECKPOINT_DIR}"
  callbacks.model_checkpoint_every_n_steps.dirpath="${CHECKPOINT_DIR}"
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps=500
  dataset.bed_file="${CONTAINER_HG38_DATA_DIR}/human-sequences.bed"
  dataset.fasta_file="${CONTAINER_HG38_DATA_DIR}/hg38.ml.fa"
  dataset.max_length="${SEQLEN}"
  dataset.max_length_val="${SEQLEN}"
  dataset.max_length_test="${SEQLEN}"
  dataset.batch_size="${BATCH_SIZE}"
  dataset.batch_size_eval="${BATCH_SIZE}"
  dataset.mlm=true
  dataset.mlm_probability="${MLM_PROBABILITY}"
  dataset.rc_aug="${RC_AUG}"
  dataset.num_workers=12
  model=mlbn_lm
  model.vocab_size=12
  model.d_model="${D_MODEL}"
  model.n_layer="${N_LAYER}"
  model.kernel_size="${KERNEL_SIZE}"
  model.d_state="${D_STATE}"
  model.d_conv="${D_CONV}"
  model.expand="${EXPAND}"
  model.head_dim="${HEAD_DIM}"
  model.num_heads="${NUM_HEADS}"
  model.drop_path_rate="${DROP_PATH_RATE}"
  model.dropout_rate1="${DROPOUT_RATE1}"
  model.dropout_rate2="${DROPOUT_RATE2}"
  model.embed_dropout="${EMBED_DROPOUT}"
  optimizer.lr="${LR}"
  optimizer.weight_decay="${WEIGHT_DECAY}"
  train.global_batch_size="${GLOBAL_BATCH_SIZE}"
  trainer.devices="${NUM_DEVICES}"
  trainer.num_nodes=1
  trainer.max_steps="${MAX_STEPS}"
  trainer.precision="${PRECISION}"
  trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}"
  trainer.num_sanity_val_steps=0
  trainer.limit_val_batches="${LIMIT_VAL_BATCHES}"
  +trainer.val_check_interval="${VAL_CHECK_INTERVAL}"
  wandb=null
  hydra.run.dir="${HYDRA_RUN_DIR}"
)

if [ -n "${TRAIN_CKPT}" ]; then
  TRAIN_ARGS+=(train.ckpt="${TRAIN_CKPT}")
fi

if [ -n "${EXTRA_OVERRIDES}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_OVERRIDES})
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

if [ "${NUM_DEVICES}" -gt 1 ]; then
  TRAIN_ARGS+=(
    '+trainer.strategy={_target_:pytorch_lightning.strategies.DDPStrategy,find_unused_parameters:'"${DDP_FIND_UNUSED_PARAMETERS}"',gradient_as_bucket_view:true}'
  )
fi

echo "Starting MLBN hg38 pre-training"
echo "LR=${LR}, global_batch_size=${GLOBAL_BATCH_SIZE}, per_gpu_batch_size=${BATCH_SIZE}, max_steps=${MAX_STEPS}"
echo "micro_batches_per_optimizer_step=${MICRO_BATCHES_PER_OPT_STEP}"
echo "val_check_interval=${VAL_CHECK_INTERVAL}, limit_val_batches=${LIMIT_VAL_BATCHES}"
echo "DDP find_unused_parameters=${DDP_FIND_UNUSED_PARAMETERS}"
echo "MLBN d_model=${D_MODEL}, n_layer=${N_LAYER}, heads=${NUM_HEADS}, drop_path=${DROP_PATH_RATE}"
srun "${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"
