#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=L40
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=gb_small_l40
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Train one GenomicBenchmarks model for the 5 original train/val split seeds.
# MODEL_VARIANT can be mlbn_small, mlbn_pretrained, ps_scratch, ph_scratch,
# ps_hf, or ph_hf.

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
echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-unset}"
echo "Resolved REPO_ROOT=${REPO_ROOT}"
set -u
export HYDRA_FULL_ERROR=1
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${REPO_ROOT}/.cache/triton/${SLURM_JOB_ID:-manual_$$}}"
export TRITON_DUMP_DIR="${TRITON_DUMP_DIR:-${TRITON_CACHE_DIR}/dump}"
export TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR:-${TRITON_CACHE_DIR}/override}"
mkdir -p "${TRITON_CACHE_DIR}" "${TRITON_DUMP_DIR}" "${TRITON_OVERRIDE_DIR}"
echo "TRITON_CACHE_DIR=${TRITON_CACHE_DIR}"
echo "TRITON_DUMP_DIR=${TRITON_DUMP_DIR}"
echo "TRITON_OVERRIDE_DIR=${TRITON_OVERRIDE_DIR}"

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

TASK="${TASK:-dummy_mouse_enhancers_ensembl}"
MODEL_VARIANT="${MODEL_VARIANT:-mlbn_small}"
GB_DATA_DIR="${GB_DATA_DIR:-${HOME}/datahome/caduceus}"
GB_DATA_DIR="$(realpath "${GB_DATA_DIR}")"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
PRETRAINED_MODEL_PATH="null"
PRETRAINED_HOOK_NAME="null"
USE_HF_CACHE=false
WANDB_SUFFIX=""
HF_MODEL_NAME_OR_PATH_OVERRIDE="${HF_MODEL_NAME_OR_PATH:-}"

case "${MODEL_VARIANT}" in
  mlbn_small|mlbn_pretrained)
    MODEL_CONFIG="mlbn_dna"
    MODEL_NAME="dna_embedding_mlbn"
    if [ "${MODEL_VARIANT}" = "mlbn_pretrained" ]; then
      DISPLAY_NAME="${DISPLAY_NAME:-mlbn_pretrained_452k}"
      if [ -z "${MLBN_PRETRAINED_PATH:-}" ]; then
        echo "MLBN_PRETRAINED_PATH is required for MODEL_VARIANT=mlbn_pretrained." >&2
        exit 1
      fi
      PRETRAINED_MODEL_PATH="${MLBN_PRETRAINED_PATH}"
      PRETRAINED_HOOK_NAME="load_backbone"
    else
      DISPLAY_NAME="${DISPLAY_NAME:-mlbn_small_452k}"
    fi
    LR="${LR:-1e-3}"
    WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    PRECISION="${PRECISION:-32}"
    GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.5}"
    RC_AUG="${RC_AUG:-false}"
    CONJOIN_TRAIN_DECODER=false
    CONJOIN_TEST=false
    EXTRA_ARGS=(
      model.d_model="${D_MODEL:-80}"
      model.n_layer="${N_LAYER:-3}"
      model.kernel_size="${KERNEL_SIZE:-4}"
      model.d_state="${D_STATE:-16}"
      model.d_conv="${D_CONV:-4}"
      model.expand="${EXPAND:-2}"
      model.head_dim="${HEAD_DIM:-16}"
      model.num_heads="${NUM_HEADS:-4}"
      model.drop_path_rate="${DROP_PATH_RATE:-0}"
      model.dropout_rate1="${DROPOUT_RATE1:-0}"
      model.dropout_rate2="${DROPOUT_RATE2:-0}"
      model.embed_dropout="${EMBED_DROPOUT:-0.0}"
    )
    OPTIMIZER_ARGS=(optimizer.lr="${LR}" optimizer.weight_decay="${WEIGHT_DECAY}")
    TRAINER_ARGS=(trainer.precision="${PRECISION}" trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}")
    WANDB_SUFFIX="_rc_aug-${RC_AUG}"
    ;;
  ps_scratch)
    MODEL_CONFIG="caduceus_ps_scratch"
    MODEL_NAME="dna_embedding_caduceus"
    DISPLAY_NAME="${DISPLAY_NAME:-caduceus_ps_scratch}"
    LR="${LR:-2e-3}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    RC_AUG="${RC_AUG:-false}"
    CONJOIN_TRAIN_DECODER=true
    CONJOIN_TEST=false
    EXTRA_ARGS=()
    OPTIMIZER_ARGS=(optimizer.lr="${LR}")
    TRAINER_ARGS=()
    WANDB_SUFFIX="_scratch"
    ;;
  ph_scratch)
    MODEL_CONFIG="caduceus_ph_scratch"
    MODEL_NAME="dna_embedding_caduceus"
    DISPLAY_NAME="${DISPLAY_NAME:-caduceus_ph_scratch}"
    LR="${LR:-2e-3}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    RC_AUG="${RC_AUG:-false}"
    CONJOIN_TRAIN_DECODER=false
    CONJOIN_TEST=true
    EXTRA_ARGS=()
    OPTIMIZER_ARGS=(optimizer.lr="${LR}")
    TRAINER_ARGS=()
    WANDB_SUFFIX="_scratch"
    ;;
  ps_hf)
    MODEL_CONFIG="caduceus_hf"
    MODEL_NAME="dna_embedding_caduceus"
    DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
    LOCAL_HF_MODEL_PATH="${GB_DATA_DIR}/models/${DEFAULT_HF_MODEL_NAME_OR_PATH##*/}"
    HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH_OVERRIDE}"
    if [ -z "${HF_MODEL_NAME_OR_PATH}" ]; then
      if [ -d "${LOCAL_HF_MODEL_PATH}" ]; then
        HF_MODEL_NAME_OR_PATH="${LOCAL_HF_MODEL_PATH}"
      else
        HF_MODEL_NAME_OR_PATH="${DEFAULT_HF_MODEL_NAME_OR_PATH}"
      fi
    fi
    DISPLAY_NAME="${DISPLAY_NAME:-caduceus_ps_paper_hf}"
    LR="${LR:-2e-3}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    RC_AUG="${RC_AUG:-false}"
    CONJOIN_TRAIN_DECODER=true
    CONJOIN_TEST=false
    EXTRA_ARGS=(
      model.hf_model_name_or_path="${HF_MODEL_NAME_OR_PATH}"
      model.local_files_only="${HF_LOCAL_FILES_ONLY:-true}"
    )
    OPTIMIZER_ARGS=(optimizer.lr="${LR}")
    TRAINER_ARGS=()
    USE_HF_CACHE=true
    WANDB_SUFFIX="_rc_aug-${RC_AUG}"
    ;;
  ph_hf)
    MODEL_CONFIG="caduceus_hf"
    MODEL_NAME="dna_embedding_caduceus"
    DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
    LOCAL_HF_MODEL_PATH="${GB_DATA_DIR}/models/${DEFAULT_HF_MODEL_NAME_OR_PATH##*/}"
    HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH_OVERRIDE}"
    if [ -z "${HF_MODEL_NAME_OR_PATH}" ]; then
      if [ -d "${LOCAL_HF_MODEL_PATH}" ]; then
        HF_MODEL_NAME_OR_PATH="${LOCAL_HF_MODEL_PATH}"
      else
        HF_MODEL_NAME_OR_PATH="${DEFAULT_HF_MODEL_NAME_OR_PATH}"
      fi
    fi
    DISPLAY_NAME="${DISPLAY_NAME:-caduceus_ph_paper_hf}"
    LR="${LR:-2e-3}"
    BATCH_SIZE="${BATCH_SIZE:-256}"
    RC_AUG="${RC_AUG:-false}"
    CONJOIN_TRAIN_DECODER=false
    CONJOIN_TEST=true
    EXTRA_ARGS=(
      model.hf_model_name_or_path="${HF_MODEL_NAME_OR_PATH}"
      model.local_files_only="${HF_LOCAL_FILES_ONLY:-true}"
    )
    OPTIMIZER_ARGS=(optimizer.lr="${LR}")
    TRAINER_ARGS=()
    USE_HF_CACHE=true
    WANDB_SUFFIX="_rc_aug-${RC_AUG}"
    ;;
  *)
    echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}. Expected mlbn_small, mlbn_pretrained, ps_scratch, ph_scratch, ps_hf, or ph_hf." >&2
    exit 1
    ;;
esac

if [ ! -d "${GB_DATA_DIR}/${TASK}" ]; then
  echo "GenomicBenchmarks dataset not found: ${GB_DATA_DIR}/${TASK}" >&2
  exit 1
fi

mkdir -p watch_folder
SINGULARITY_BINDS=(--bind "${REPO_ROOT}:/host:rw")
if [[ "${GB_DATA_DIR}" == "${REPO_ROOT}"/* ]]; then
  CONTAINER_GB_DATA_DIR="/host/${GB_DATA_DIR#${REPO_ROOT}/}"
else
  SINGULARITY_BINDS+=(--bind "${GB_DATA_DIR}:${GB_DATA_DIR}:rw")
  CONTAINER_GB_DATA_DIR="${GB_DATA_DIR}"
fi
if [ "${USE_HF_CACHE}" = true ]; then
  if [ -z "${HF_CACHE_DIR:-}" ]; then
    if [ -d "${GB_DATA_DIR}/.cache/huggingface" ]; then
      HF_CACHE_DIR="${GB_DATA_DIR}/.cache/huggingface"
    elif [ -d "${GB_DATA_DIR}/.cache/hub" ]; then
      HF_CACHE_DIR="${GB_DATA_DIR}/.cache"
    else
      HF_CACHE_DIR="${REPO_ROOT}/.cache/huggingface"
    fi
  fi
  mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/transformers"
  if [[ "${HF_CACHE_DIR}" == "${REPO_ROOT}"/* ]]; then
    CONTAINER_HF_CACHE_DIR="/host/${HF_CACHE_DIR#${REPO_ROOT}/}"
  elif [[ "${HF_CACHE_DIR}" == "${GB_DATA_DIR}"/* ]]; then
    CONTAINER_HF_CACHE_DIR="${HF_CACHE_DIR}"
  else
    SINGULARITY_BINDS+=(--bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}:rw")
    CONTAINER_HF_CACHE_DIR="${HF_CACHE_DIR}"
  fi
fi

SINGULARITY_BASE=(
  singularity exec
  --nv
  --pwd /host
  "${SINGULARITY_BINDS[@]}"
  "${SIF_IMAGE}"
)

echo "Singularity image: ${SIF_IMAGE}"
echo "Container Python executable: ${VENV_PYTHON}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi || true
SINGULARITY_ENV=(
  env
  PYTHONPATH=/host
  TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
  TRITON_DUMP_DIR="${TRITON_DUMP_DIR}"
  TRITON_OVERRIDE_DIR="${TRITON_OVERRIDE_DIR}"
)
if [ "${USE_HF_CACHE}" = true ]; then
  SINGULARITY_ENV+=(
    HF_HOME="${CONTAINER_HF_CACHE_DIR}"
    HF_HUB_CACHE="${CONTAINER_HF_CACHE_DIR}/hub"
    HF_DATASETS_CACHE="${CONTAINER_HF_CACHE_DIR}/datasets"
    TRANSFORMERS_CACHE="${CONTAINER_HF_CACHE_DIR}/transformers"
  )
  echo "Hugging Face model: ${HF_MODEL_NAME_OR_PATH}"
  echo "Hugging Face cache dir: ${HF_CACHE_DIR}"
fi
"${SINGULARITY_BASE[@]}" "${SINGULARITY_ENV[@]}" "${VENV_PYTHON}" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
try:
    import mamba_ssm
    print("mamba-ssm:", getattr(mamba_ssm, "__version__", "unknown"))
    print("Mamba2 available:", hasattr(mamba_ssm, "Mamba2"))
except Exception as exc:
    print("mamba-ssm import check skipped:", repr(exc))
PY

WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}${WANDB_SUFFIX}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

for seed in ${SEEDS}; do
  HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/job-${RUN_ID}/seed-${seed}"
  CHECKPOINT_DIR="${HYDRA_RUN_DIR}/checkpoints"
  mkdir -p "${HYDRA_RUN_DIR}"

  echo "*****************************************************"
  echo "Running GenomicBenchmarks task=${TASK}, variant=${MODEL_VARIANT}, model=${MODEL_CONFIG}, seed=${seed}"
  echo "GenomicBenchmarks host data dir=${GB_DATA_DIR}"
  echo "GenomicBenchmarks container data dir=${CONTAINER_GB_DATA_DIR}"
  echo "LR=${LR}, batch_size=${BATCH_SIZE}, max_epochs=${MAX_EPOCHS}, rc_aug=${RC_AUG}"
  echo "pretrained_model_path=${PRETRAINED_MODEL_PATH}, pretrained_hook=${PRETRAINED_HOOK_NAME}"

  TRAIN_ARGS=(
    -m train
    experiment=hg38/genomic_benchmark
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000
    callbacks.model_checkpoint.dirpath="${CHECKPOINT_DIR}"
    callbacks.model_checkpoint_every_n_steps.dirpath="${CHECKPOINT_DIR}"
    dataset.dataset_name="${TASK}"
    dataset.dest_path="${CONTAINER_GB_DATA_DIR}"
    dataset.train_val_split_seed="${seed}"
    dataset.batch_size="${BATCH_SIZE}"
    dataset.rc_aug="${RC_AUG}"
    +dataset.conjoin_train=false
    +dataset.conjoin_test="${CONJOIN_TEST}"
    loader.num_workers="${NUM_WORKERS}"
    model="${MODEL_CONFIG}"
    model._name_="${MODEL_NAME}"
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}"
    +decoder.conjoin_test="${CONJOIN_TEST}"
    trainer.devices=1
    trainer.max_epochs="${MAX_EPOCHS}"
    train.ckpt=null
    train.pretrained_model_path="${PRETRAINED_MODEL_PATH}"
    train.pretrained_model_state_hook._name_="${PRETRAINED_HOOK_NAME}"
    wandb=null
    hydra.run.dir="${HYDRA_RUN_DIR}"
    "${OPTIMIZER_ARGS[@]}"
    "${TRAINER_ARGS[@]}"
    "${EXTRA_ARGS[@]}"
  )

  if [ "${MODEL_NAME}" = "dna_embedding_caduceus" ]; then
    TRAIN_ARGS+=(model.conjoin_test="${CONJOIN_TEST}")
  fi

  "${SINGULARITY_BASE[@]}" "${SINGULARITY_ENV[@]}" "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"
  echo "Completed seed=${seed}"
  echo "*****************************************************"
done
