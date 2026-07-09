#!/bin/bash
# Pre-train the non-Caduceus comparison backbones for the DNALONGBENCH 450k
# ETGP/eQTL experiments. This script intentionally uses an isolated venv.
#
# MODEL_KIND:
#   mamba1       plain Mamba1 MLM pre-training
#   mamba1_flip  Mamba1 MLM pre-training with plain flipped augmentation
#   mlbn         MLBN MLM pre-training

#SBATCH --get-user-env
#SBATCH --partition=A800
#SBATCH -t 96:00:00
#SBATCH --mem=896G
#SBATCH --gres=gpu:a800:8
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=pretrain_131k
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
cd "${REPO_ROOT}" || exit 1

set -u
export HYDRA_FULL_ERROR=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

MODEL_KIND="${MODEL_KIND:-mamba1}"
NUM_DEVICES="${NUM_DEVICES:-8}"
SEQLEN="${SEQLEN:-131072}"
MAX_STEPS="${MAX_STEPS:-50000}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
BATCH_SIZE="${BATCH_SIZE:-1}"
MLM_PROBABILITY="${MLM_PROBABILITY:-0.15}"
LR="${LR:-}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
PRECISION="${PRECISION:-}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-}"
LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-0.125}"
VAL_CHECK_INTERVAL="${VAL_CHECK_INTERVAL:-5000}"
CHECKPOINT_EVERY_N_STEPS="${CHECKPOINT_EVERY_N_STEPS:-5000}"
DDP_FIND_UNUSED_PARAMETERS="${DDP_FIND_UNUSED_PARAMETERS:-true}"
TRAIN_CKPT="${TRAIN_CKPT:-}"
RUN_TAG="${RUN_TAG:-}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

CONTAINER_DIR="${CONTAINER_DIR:-/path/to/shared_containers}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-caduceus-pytorch-2.2.0-cuda12.1-cudnn8-devel.sif}"
SIF_IMAGE="${SIF_IMAGE:-${CONTAINER_DIR}/${PYTORCH_IMAGE}}"
VENV_PYTHON="${VENV_PYTHON:-/host/.venvs/dnalongbench450k-mamba1/bin/python}"

if [[ "${VENV_PYTHON}" == /host/* ]]; then
  HOST_VENV_PYTHON="${REPO_ROOT}/${VENV_PYTHON#/host/}"
  if [ ! -e "${HOST_VENV_PYTHON}" ] && [ ! -L "${HOST_VENV_PYTHON}" ]; then
    echo "Isolated venv python not found: ${HOST_VENV_PYTHON}" >&2
    exit 2
  fi
fi

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

HG38_DATA_DIR="${HG38_DATA_DIR:-${HOME}/datahome/caduceus/hg38}"
HG38_DATA_DIR="$(realpath "${HG38_DATA_DIR}")"
FASTA_FILE="${HG38_DATA_DIR}/hg38.ml.fa"
BED_FILE="${HG38_DATA_DIR}/human-sequences.bed"
if [ ! -f "${FASTA_FILE}" ] || [ ! -f "${BED_FILE}" ]; then
  echo "hg38 pre-training data is missing under ${HG38_DATA_DIR}" >&2
  echo "Expected ${FASTA_FILE} and ${BED_FILE}" >&2
  exit 2
fi

PLAIN_FLIP_AUG=false
RC_AUG=false
DISPLAY_NAME=""
MODEL_OVERRIDES=()
case "${MODEL_KIND}" in
  mamba1)
    DISPLAY_NAME="mamba1"
    LR="${LR:-8e-3}"
    PRECISION="${PRECISION:-16}"
    GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
    MODEL_OVERRIDES=(
      model=mamba
      model.config.d_model="${D_MODEL:-256}"
      model.config.n_layer="${N_LAYER:-16}"
    )
    ;;
  mamba1_flip)
    DISPLAY_NAME="mamba1_plain_flip"
    PLAIN_FLIP_AUG=true
    LR="${LR:-8e-3}"
    PRECISION="${PRECISION:-16}"
    GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
    MODEL_OVERRIDES=(
      model=mamba
      model.config.d_model="${D_MODEL:-256}"
      model.config.n_layer="${N_LAYER:-16}"
    )
    ;;
  mlbn)
    DISPLAY_NAME="mlbn"
    LR="${LR:-1e-4}"
    PRECISION="${PRECISION:-bf16}"
    GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.5}"
    MODEL_OVERRIDES=(
      model=mlbn_lm
      model.vocab_size=12
      model.d_model="${D_MODEL:-192}"
      model.n_layer="${N_LAYER:-13}"
      model.kernel_size="${KERNEL_SIZE:-4}"
      model.d_state="${D_STATE:-16}"
      model.d_conv="${D_CONV:-4}"
      model.expand="${EXPAND:-2}"
      model.head_dim="${HEAD_DIM:-16}"
      model.num_heads="${NUM_HEADS:-3}"
      model.dropout_rate1="${DROPOUT_RATE1:-0.08}"
      model.dropout_rate2="${DROPOUT_RATE2:-0.08}"
      model.drop_path_rate="${DROP_PATH_RATE:-0.18}"
      model.embed_dropout="${EMBED_DROPOUT:-0.0}"
    )
    ;;
  *)
    echo "MODEL_KIND must be one of: mamba1, mamba1_flip, mlbn. Got: ${MODEL_KIND}" >&2
    exit 2
    ;;
esac

WANDB_NAME="${DISPLAY_NAME}_mlm_seqlen-131k_lr-${LR}_gb-${GLOBAL_BATCH_SIZE}_steps-${MAX_STEPS}"
if [ -n "${RUN_TAG}" ]; then
  WANDB_NAME="${WANDB_NAME}_${RUN_TAG}"
fi
HOST_OUTPUT_DIR="${REPO_ROOT}/outputs/pretrain/dnalongbench450k_131k/${WANDB_NAME}"
HYDRA_RUN_DIR="/host/outputs/pretrain/dnalongbench450k_131k/${WANDB_NAME}"
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

MICRO_BATCHES_PER_OPT_STEP="$(( (GLOBAL_BATCH_SIZE + (BATCH_SIZE * NUM_DEVICES) - 1) / (BATCH_SIZE * NUM_DEVICES) ))"

echo "REPO_ROOT=${REPO_ROOT}"
echo "Singularity image=${SIF_IMAGE}"
echo "VENV_PYTHON=${VENV_PYTHON}"
echo "MODEL_KIND=${MODEL_KIND}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "HG38_DATA_DIR=${HG38_DATA_DIR}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "global_batch_size=${GLOBAL_BATCH_SIZE}, per_gpu_batch_size=${BATCH_SIZE}, grad_accum=${MICRO_BATCHES_PER_OPT_STEP}"
nvidia-smi || true

"${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" - <<'PY'
import sys
import torch
import mamba_ssm
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA devices:", torch.cuda.device_count())
print("mamba-ssm:", getattr(mamba_ssm, "__version__", "unknown"))
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
  callbacks.model_checkpoint_every_n_steps.every_n_train_steps="${CHECKPOINT_EVERY_N_STEPS}"
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
  dataset.plain_flip_aug="${PLAIN_FLIP_AUG}"
  dataset.num_workers=12
  "${MODEL_OVERRIDES[@]}"
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

if [ "${NUM_DEVICES}" -gt 1 ]; then
  TRAIN_ARGS+=(
    '+trainer.strategy={_target_:pytorch_lightning.strategies.DDPStrategy,find_unused_parameters:'"${DDP_FIND_UNUSED_PARAMETERS}"',gradient_as_bucket_view:true}'
  )
fi

if [ -n "${EXTRA_OVERRIDES}" ]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${EXTRA_OVERRIDES})
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

echo "Starting 131k hg38 MLM pre-training: ${MODEL_KIND}"
srun "${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"
