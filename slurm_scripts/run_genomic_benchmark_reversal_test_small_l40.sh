#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=L40
#SBATCH -t 12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --job-name=gb_rev_test_small
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Re-evaluate small GenomicBenchmarks checkpoints on normal test and pure
# sequence-flipped test. The flip is ACGT -> TGCA, not reverse-complement.

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
GB_DATA_DIR="${GB_DATA_DIR:-${HOME}/datahome/caduceus}"
GB_DATA_DIR="$(realpath "${GB_DATA_DIR}")"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
MODEL_VARIANTS="${MODEL_VARIANTS:-mlbn_small ps_scratch ph_scratch}"
EVAL_MODES="${EVAL_MODES:-normal_test flipped_test}"
BASE_SOURCE_JOB_ID="${SOURCE_JOB_ID:-}"
BASE_DISPLAY_NAME="${DISPLAY_NAME:-}"
BASE_LR="${LR:-}"
BASE_BATCH_SIZE="${BATCH_SIZE:-}"
BASE_HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH:-}"

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
  HF_HOME="${CONTAINER_HF_CACHE_DIR}"
  HF_HUB_CACHE="${CONTAINER_HF_CACHE_DIR}/hub"
  HF_DATASETS_CACHE="${CONTAINER_HF_CACHE_DIR}/datasets"
  TRANSFORMERS_CACHE="${CONTAINER_HF_CACHE_DIR}/transformers"
)
"${SINGULARITY_BASE[@]}" "${SINGULARITY_ENV[@]}" "${VENV_PYTHON}" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

for variant in ${MODEL_VARIANTS}; do
  PRETRAINED_MODEL_PATH="null"
  PRETRAINED_HOOK_NAME="null"
  RC_AUG="${RC_AUG:-false}"
  case "${variant}" in
    mlbn_small)
      MODEL_CONFIG="mlbn_dna"
      MODEL_NAME="dna_embedding_mlbn"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-mlbn_small_452k}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${MLBN_SMALL_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-1e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}"
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
        optimizer.weight_decay="${WEIGHT_DECAY:-0.01}"
        trainer.precision="${PRECISION:-32}"
        trainer.gradient_clip_val="${GRADIENT_CLIP_VAL:-0.5}"
      )
      ;;
    mlbn_pretrained)
      MODEL_CONFIG="mlbn_dna"
      MODEL_NAME="dna_embedding_mlbn"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-mlbn_pretrained_452k}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${MLBN_PRETRAINED_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-1e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-64}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}"
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
        optimizer.weight_decay="${WEIGHT_DECAY:-0.01}"
        trainer.precision="${PRECISION:-32}"
        trainer.gradient_clip_val="${GRADIENT_CLIP_VAL:-0.5}"
      )
      ;;
    ps_scratch)
      MODEL_CONFIG="caduceus_ps_scratch"
      MODEL_NAME="dna_embedding_caduceus"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-caduceus_ps_scratch}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${PS_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-2e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_scratch"
      CONJOIN_TRAIN_DECODER=true
      CONJOIN_TEST=false
      EXTRA_ARGS=()
      ;;
    ph_scratch)
      MODEL_CONFIG="caduceus_ph_scratch"
      MODEL_NAME="dna_embedding_caduceus"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-caduceus_ph_scratch}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${PH_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-2e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_scratch"
      CONJOIN_TRAIN_DECODER=false
      CONJOIN_TEST=true
      EXTRA_ARGS=()
      ;;
    ps_hf)
      MODEL_CONFIG="caduceus_hf"
      MODEL_NAME="dna_embedding_caduceus"
      DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
      LOCAL_HF_MODEL_PATH="${GB_DATA_DIR}/models/${DEFAULT_HF_MODEL_NAME_OR_PATH##*/}"
      HF_MODEL_NAME_OR_PATH="${BASE_HF_MODEL_NAME_OR_PATH}"
      if [ -z "${HF_MODEL_NAME_OR_PATH}" ]; then
        if [ -d "${LOCAL_HF_MODEL_PATH}" ]; then
          HF_MODEL_NAME_OR_PATH="${LOCAL_HF_MODEL_PATH}"
        else
          HF_MODEL_NAME_OR_PATH="${DEFAULT_HF_MODEL_NAME_OR_PATH}"
        fi
      fi
      MODEL_HF_PATH="${HF_MODEL_NAME_OR_PATH}"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-caduceus_ps_paper_hf}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${PS_HF_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-2e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-false"
      CONJOIN_TRAIN_DECODER=true
      CONJOIN_TEST=false
      EXTRA_ARGS=(
        model.hf_model_name_or_path="${MODEL_HF_PATH}"
        model.local_files_only="${HF_LOCAL_FILES_ONLY:-true}"
      )
      ;;
    ph_hf)
      MODEL_CONFIG="caduceus_hf"
      MODEL_NAME="dna_embedding_caduceus"
      DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
      LOCAL_HF_MODEL_PATH="${GB_DATA_DIR}/models/${DEFAULT_HF_MODEL_NAME_OR_PATH##*/}"
      HF_MODEL_NAME_OR_PATH="${BASE_HF_MODEL_NAME_OR_PATH}"
      if [ -z "${HF_MODEL_NAME_OR_PATH}" ]; then
        if [ -d "${LOCAL_HF_MODEL_PATH}" ]; then
          HF_MODEL_NAME_OR_PATH="${LOCAL_HF_MODEL_PATH}"
        else
          HF_MODEL_NAME_OR_PATH="${DEFAULT_HF_MODEL_NAME_OR_PATH}"
        fi
      fi
      MODEL_HF_PATH="${HF_MODEL_NAME_OR_PATH}"
      DISPLAY_NAME="${BASE_DISPLAY_NAME:-caduceus_ph_paper_hf}"
      VARIANT_SOURCE_JOB_ID="${BASE_SOURCE_JOB_ID:-${PH_HF_SOURCE_JOB_ID:-}}"
      LR="${BASE_LR:-2e-3}"
      BATCH_SIZE="${BASE_BATCH_SIZE:-256}"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-false"
      CONJOIN_TRAIN_DECODER=false
      CONJOIN_TEST=true
      EXTRA_ARGS=(
        model.hf_model_name_or_path="${MODEL_HF_PATH}"
        model.local_files_only="${HF_LOCAL_FILES_ONLY:-true}"
      )
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac

  if [ -z "${VARIANT_SOURCE_JOB_ID}" ]; then
    echo "SOURCE_JOB_ID is required for variant=${variant}." >&2
    exit 1
  fi

  for seed in ${SEEDS}; do
    CKPT_ROOT="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/job-${VARIANT_SOURCE_JOB_ID}/seed-${seed}"
    if [ ! -d "${CKPT_ROOT}" ]; then
      echo "Checkpoint root not found: ${CKPT_ROOT}" >&2
      exit 1
    fi
    CKPT_PATH="$(find "${CKPT_ROOT}" -path "*/checkpoints/val/accuracy.ckpt" -type f | sort | tail -n 1)"
    if [ -z "${CKPT_PATH}" ] || [ ! -f "${CKPT_PATH}" ]; then
      echo "Checkpoint not found under: ${CKPT_ROOT}" >&2
      exit 1
    fi
    CONTAINER_CKPT_PATH="/host/${CKPT_PATH#./}"

    for mode in ${EVAL_MODES}; do
      case "${mode}" in
        normal_test)
          REVERSE_TEST=false
          ;;
        flipped_test)
          REVERSE_TEST=true
          ;;
        *)
          echo "Unsupported eval mode: ${mode}" >&2
          exit 1
          ;;
      esac

      HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/${mode}_eval/job-${SLURM_JOB_ID:-manual}/seed-${seed}"
      echo "*****************************************************"
      echo "Evaluating test split: variant=${variant}, seed=${seed}, mode=${mode}, reverse_test=${REVERSE_TEST}, checkpoint=${CKPT_PATH}"
      EVAL_ARGS=(
        scripts/evaluate_genomic_benchmark_split.py
        experiment=hg38/genomic_benchmark
        dataset.dataset_name="${TASK}"
        dataset.dest_path="${CONTAINER_GB_DATA_DIR}"
        dataset.train_val_split_seed="${seed}"
        dataset.batch_size="${BATCH_SIZE}"
        dataset.rc_aug=false
        dataset.reverse_val=false
        dataset.reverse_test="${REVERSE_TEST}"
        +dataset.conjoin_train=false
        +dataset.conjoin_test="${CONJOIN_TEST}"
        loader.num_workers="${NUM_WORKERS}"
        model="${MODEL_CONFIG}"
        model._name_="${MODEL_NAME}"
        +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}"
        +decoder.conjoin_test="${CONJOIN_TEST}"
        trainer.devices=1
        train.ckpt=null
        train.pretrained_model_path="${PRETRAINED_MODEL_PATH}"
        train.pretrained_model_state_hook._name_="${PRETRAINED_HOOK_NAME}"
        wandb=null
        +eval.split=test
        +eval.ckpt_path="${CONTAINER_CKPT_PATH}"
        hydra.run.dir="${HYDRA_RUN_DIR}"
        optimizer.lr="${LR}"
        "${EXTRA_ARGS[@]}"
      )

      if [ "${MODEL_NAME}" = "dna_embedding_caduceus" ]; then
        EVAL_ARGS+=(model.conjoin_test="${CONJOIN_TEST}")
      fi

      "${SINGULARITY_BASE[@]}" "${SINGULARITY_ENV[@]}" "${VENV_PYTHON}" "${EVAL_ARGS[@]}"
      echo "Completed test eval: variant=${variant}, seed=${seed}, mode=${mode}"
      echo "*****************************************************"
    done
  done
done
