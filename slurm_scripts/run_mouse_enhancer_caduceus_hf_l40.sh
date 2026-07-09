#!/bin/bash
#SBATCH --get-user-env
#SBATCH --partition=L40
#SBATCH -t 24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:l40:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=gb_mouse_caduceus_hf_l40
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Fine-tune the published Caduceus-PS Hugging Face checkpoint on the first
# GenomicBenchmarks dataset: dummy_mouse_enhancers_ensembl.
#
# This branch uses the Singularity + repo-local .venv workflow documented in
# INSTALLATION.md, so batch jobs should not rely on conda/python being present
# in the login shell PATH.

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
HOST_VENV_PYTHON="${REPO_ROOT}/.venv/bin/python"

mkdir -p .cache/huggingface/hub .cache/huggingface/datasets .cache/huggingface/transformers watch_folder

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
GB_DATA_DIR="${GB_DATA_DIR:-${REPO_ROOT}/data/genomic_benchmark}"
MODEL_VARIANT="${MODEL_VARIANT:-ps}"
case "${MODEL_VARIANT}" in
  ps)
    DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
    DEFAULT_DISPLAY_NAME="caduceus_ps_paper_hf"
    DEFAULT_CONJOIN_TRAIN_DECODER=true
    DEFAULT_CONJOIN_TEST=false
    DEFAULT_RC_AUG=false
    ;;
  ph)
    DEFAULT_HF_MODEL_NAME_OR_PATH="kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
    DEFAULT_DISPLAY_NAME="caduceus_ph_paper_hf"
    DEFAULT_CONJOIN_TRAIN_DECODER=false
    DEFAULT_CONJOIN_TEST=true
    DEFAULT_RC_AUG=false
    ;;
  *)
    echo "Unsupported MODEL_VARIANT=${MODEL_VARIANT}. Expected 'ps' or 'ph'." >&2
    exit 1
    ;;
esac

if [ -z "${HF_MODEL_NAME_OR_PATH:-}" ]; then
  LOCAL_HF_MODEL_PATH="${GB_DATA_DIR}/models/${DEFAULT_HF_MODEL_NAME_OR_PATH##*/}"
  if [ -d "${LOCAL_HF_MODEL_PATH}" ]; then
    HF_MODEL_NAME_OR_PATH="${LOCAL_HF_MODEL_PATH}"
  else
    HF_MODEL_NAME_OR_PATH="${DEFAULT_HF_MODEL_NAME_OR_PATH}"
  fi
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
DISPLAY_NAME="${DISPLAY_NAME:-${DEFAULT_DISPLAY_NAME}}"
LR="${LR:-2e-3}"
BATCH_SIZE="${BATCH_SIZE:-256}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
CONJOIN_TRAIN_DECODER="${CONJOIN_TRAIN_DECODER:-${DEFAULT_CONJOIN_TRAIN_DECODER}}"
CONJOIN_TEST="${CONJOIN_TEST:-${DEFAULT_CONJOIN_TEST}}"
RC_AUG="${RC_AUG:-${DEFAULT_RC_AUG}}"

SINGULARITY_BINDS=(--bind "${REPO_ROOT}:/host:rw")
if [[ "${GB_DATA_DIR}" == "${REPO_ROOT}"/* ]]; then
  CONTAINER_GB_DATA_DIR="/host/${GB_DATA_DIR#${REPO_ROOT}/}"
else
  mkdir -p "${GB_DATA_DIR}"
  SINGULARITY_BINDS+=(--bind "${GB_DATA_DIR}:${GB_DATA_DIR}:rw")
  CONTAINER_GB_DATA_DIR="${GB_DATA_DIR}"
fi

mkdir -p "${HF_CACHE_DIR}/hub" "${HF_CACHE_DIR}/datasets" "${HF_CACHE_DIR}/transformers"
if [[ "${HF_CACHE_DIR}" == "${REPO_ROOT}"/* ]]; then
  CONTAINER_HF_CACHE_DIR="/host/${HF_CACHE_DIR#${REPO_ROOT}/}"
else
  SINGULARITY_BINDS+=(--bind "${HF_CACHE_DIR}:${HF_CACHE_DIR}:rw")
  CONTAINER_HF_CACHE_DIR="${HF_CACHE_DIR}"
fi

if [ ! -d "${GB_DATA_DIR}/${TASK}" ]; then
  echo "GenomicBenchmarks dataset not found: ${GB_DATA_DIR}/${TASK}" >&2
  echo "Run scripts/prefetch_mouse_enhancer_assets.sh from ${REPO_ROOT} before submitting this job." >&2
  exit 1
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
"${SINGULARITY_BASE[@]}" env \
  PYTHONPATH=/host \
  HF_HOME="${CONTAINER_HF_CACHE_DIR}" \
  HF_HUB_CACHE="${CONTAINER_HF_CACHE_DIR}/hub" \
  HF_DATASETS_CACHE="${CONTAINER_HF_CACHE_DIR}/datasets" \
  TRANSFORMERS_CACHE="${CONTAINER_HF_CACHE_DIR}/transformers" \
  "${VENV_PYTHON}" -c "import sys, torch; print('Python:', sys.executable); print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

for seed in ${SEEDS}; do
  HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/job-${RUN_ID}/seed-${seed}"
  CHECKPOINT_DIR="${HYDRA_RUN_DIR}/checkpoints"
  mkdir -p "${HYDRA_RUN_DIR}"

  echo "*****************************************************"
  echo "Running GenomicBenchmarks task=${TASK}, variant=${MODEL_VARIANT}, model=${HF_MODEL_NAME_OR_PATH}, seed=${seed}"
  echo "GenomicBenchmarks host data dir=${GB_DATA_DIR}"
  echo "GenomicBenchmarks container data dir=${CONTAINER_GB_DATA_DIR}"
  echo "Hugging Face host cache dir=${HF_CACHE_DIR}"
  echo "Hugging Face container cache dir=${CONTAINER_HF_CACHE_DIR}"
  echo "LR=${LR}, batch_size=${BATCH_SIZE}, max_epochs=${MAX_EPOCHS}, rc_aug=${RC_AUG}"

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
    model=caduceus_hf
    model._name_=dna_embedding_caduceus
    model.hf_model_name_or_path="${HF_MODEL_NAME_OR_PATH}"
    model.local_files_only=true
    model.conjoin_test="${CONJOIN_TEST}"
    +decoder.conjoin_train="${CONJOIN_TRAIN_DECODER}"
    +decoder.conjoin_test="${CONJOIN_TEST}"
    optimizer.lr="${LR}"
    trainer.devices=1
    trainer.max_epochs="${MAX_EPOCHS}"
    train.ckpt=null
    train.pretrained_model_path=null
    train.pretrained_model_state_hook._name_=null
    wandb=null
    hydra.run.dir="${HYDRA_RUN_DIR}"
  )

  "${SINGULARITY_BASE[@]}" env \
    PYTHONPATH=/host \
    HF_HOME="${CONTAINER_HF_CACHE_DIR}" \
    HF_HUB_CACHE="${CONTAINER_HF_CACHE_DIR}/hub" \
    HF_DATASETS_CACHE="${CONTAINER_HF_CACHE_DIR}/datasets" \
    TRANSFORMERS_CACHE="${CONTAINER_HF_CACHE_DIR}/transformers" \
    "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"

  echo "*****************************************************"
done
