#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=L40
#SBATCH -t 04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --job-name=gb_mouse_flip_val_scratch
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Evaluate randomly initialized mouse-enhancer checkpoints on validation
# sequences reversed along the sequence direction. This is a pure sequence flip:
# ACGT -> TGCA, not reverse-complement.

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
MODEL_VARIANTS="${MODEL_VARIANTS:-ps_scratch ph_scratch mlbn}"

PS_SOURCE_JOB_ID="${PS_SOURCE_JOB_ID:-1481855}"
PH_SOURCE_JOB_ID="${PH_SOURCE_JOB_ID:-1481856}"
MLBN_SOURCE_JOB_ID="${MLBN_SOURCE_JOB_ID:-1481818}"

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
"${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" - <<'PY'
import sys
import torch
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
PY

for variant in ${MODEL_VARIANTS}; do
  case "${variant}" in
    ps_scratch)
      MODEL_CONFIG="caduceus_ps_scratch"
      MODEL_NAME="dna_embedding_caduceus"
      DISPLAY_NAME="caduceus_ps_scratch"
      SOURCE_JOB_ID="${PS_SOURCE_JOB_ID}"
      LR="2e-3"
      BATCH_SIZE="256"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_scratch"
      CONJOIN_TRAIN_DECODER=true
      CONJOIN_TEST=false
      EXTRA_ARGS=()
      ;;
    ph_scratch)
      MODEL_CONFIG="caduceus_ph_scratch"
      MODEL_NAME="dna_embedding_caduceus"
      DISPLAY_NAME="caduceus_ph_scratch"
      SOURCE_JOB_ID="${PH_SOURCE_JOB_ID}"
      LR="2e-3"
      BATCH_SIZE="256"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_scratch"
      CONJOIN_TRAIN_DECODER=false
      CONJOIN_TEST=true
      EXTRA_ARGS=()
      ;;
    mlbn)
      MODEL_CONFIG="mlbn_dna"
      MODEL_NAME="dna_embedding_mlbn"
      DISPLAY_NAME="mlbn_dna"
      SOURCE_JOB_ID="${MLBN_SOURCE_JOB_ID}"
      LR="1e-4"
      BATCH_SIZE="64"
      WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-false"
      CONJOIN_TRAIN_DECODER=false
      CONJOIN_TEST=false
      EXTRA_ARGS=(model.drop_path_rate=0.1 trainer.precision=32)
      ;;
    *)
      echo "Unsupported variant: ${variant}" >&2
      exit 1
      ;;
  esac

  for seed in ${SEEDS}; do
    CKPT_ROOT="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/job-${SOURCE_JOB_ID}/seed-${seed}"
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
    HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/flip_val_eval/job-${SLURM_JOB_ID:-manual}/seed-${seed}"

    echo "*****************************************************"
    echo "Evaluating flipped validation split: variant=${variant}, seed=${seed}, checkpoint=${CKPT_PATH}"
    "${SINGULARITY_BASE[@]}" env \
      PYTHONPATH=/host \
      "${VENV_PYTHON}" scripts/evaluate_genomic_benchmark_flip_val.py \
        experiment=hg38/genomic_benchmark \
        dataset.dataset_name="${TASK}" \
        dataset.dest_path="${CONTAINER_GB_DATA_DIR}" \
        dataset.train_val_split_seed="${seed}" \
        dataset.batch_size="${BATCH_SIZE}" \
        dataset.rc_aug=false \
        dataset.reverse_val=true \
        dataset.reverse_test=false \
        +dataset.conjoin_train=false \
        +dataset.conjoin_test="${CONJOIN_TEST}" \
        loader.num_workers="${NUM_WORKERS}" \
        model="${MODEL_CONFIG}" \
        model._name_="${MODEL_NAME}" \
        model.conjoin_test="${CONJOIN_TEST}" \
        +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}" \
        +decoder.conjoin_test="${CONJOIN_TEST}" \
        trainer.devices=1 \
        train.ckpt=null \
        train.pretrained_model_path=null \
        train.pretrained_model_state_hook._name_=null \
        wandb=null \
        +eval.ckpt_path="${CONTAINER_CKPT_PATH}" \
        hydra.run.dir="${HYDRA_RUN_DIR}" \
        "${EXTRA_ARGS[@]}"
    echo "Completed flip-val eval: variant=${variant}, seed=${seed}"
    echo "*****************************************************"
  done
done
