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
#SBATCH --job-name=gb_mouse_mlbn_l40
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --open-mode=append

# Train the DNA-token MLBN bidirectional Mamba backbone on the first
# GenomicBenchmarks dataset: dummy_mouse_enhancers_ensembl.

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
DEFAULT_GB_DATA_DIR="${HOME}/datahome/caduceus"
GB_DATA_DIR="${GB_DATA_DIR:-${DEFAULT_GB_DATA_DIR}}"
GB_DATA_DIR="$(realpath "${GB_DATA_DIR}")"
DISPLAY_NAME="${DISPLAY_NAME:-mlbn_dna}"
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
PRECISION="${PRECISION:-32}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-0.5}"
DROP_PATH_RATE="${DROP_PATH_RATE:-0.1}"
DROPOUT_RATE1="${DROPOUT_RATE1:-0.08}"
DROPOUT_RATE2="${DROPOUT_RATE2:-0.08}"
EMBED_DROPOUT="${EMBED_DROPOUT:-0.0}"
D_MODEL="${D_MODEL:-192}"
N_LAYER="${N_LAYER:-13}"
KERNEL_SIZE="${KERNEL_SIZE:-4}"
D_STATE="${D_STATE:-16}"
D_CONV="${D_CONV:-4}"
EXPAND="${EXPAND:-2}"
HEAD_DIM="${HEAD_DIM:-16}"
NUM_HEADS="${NUM_HEADS:-3}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_EPOCHS="${MAX_EPOCHS:-10}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEEDS="${SEEDS:-1 2 3 4 5}"
RC_AUG="${RC_AUG:-false}"

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
import mamba_ssm
print("Python:", sys.executable)
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("mamba-ssm:", getattr(mamba_ssm, "__version__", "unknown"))
print("Mamba2 available:", hasattr(mamba_ssm, "Mamba2"))
PY

WANDB_NAME="${DISPLAY_NAME}_${TASK}_lr-${LR}_batch_size-${BATCH_SIZE}_rc_aug-${RC_AUG}"
RUN_ID="${RUN_ID:-${SLURM_JOB_ID:-manual_$(date +%Y%m%d_%H%M%S)}}"

for seed in ${SEEDS}; do
  HYDRA_RUN_DIR="./outputs/downstream/gb_cv5/${TASK}/${WANDB_NAME}/job-${RUN_ID}/seed-${seed}"
  CHECKPOINT_DIR="${HYDRA_RUN_DIR}/checkpoints"
  mkdir -p "${HYDRA_RUN_DIR}"

  echo "*****************************************************"
  echo "Running GenomicBenchmarks task=${TASK}, model=mlbn_dna, seed=${seed}"
  echo "GenomicBenchmarks host data dir=${GB_DATA_DIR}"
  echo "GenomicBenchmarks container data dir=${CONTAINER_GB_DATA_DIR}"
  echo "LR=${LR}, weight_decay=${WEIGHT_DECAY}, batch_size=${BATCH_SIZE}, max_epochs=${MAX_EPOCHS}, rc_aug=${RC_AUG}"
  echo "precision=${PRECISION}, gradient_clip_val=${GRADIENT_CLIP_VAL}"
  echo "MLBN architecture: d_model=${D_MODEL}, n_layer=${N_LAYER}, kernel_size=${KERNEL_SIZE}, d_state=${D_STATE}, d_conv=${D_CONV}, expand=${EXPAND}, head_dim=${HEAD_DIM}, num_heads=${NUM_HEADS}"
  echo "MLBN regularization: drop_path_rate=${DROP_PATH_RATE}, dropout_rate1=${DROPOUT_RATE1}, dropout_rate2=${DROPOUT_RATE2}, embed_dropout=${EMBED_DROPOUT}"

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
    +dataset.conjoin_test=false
    loader.num_workers="${NUM_WORKERS}"
    model=mlbn_dna
    model._name_=dna_embedding_mlbn
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
    +decoder.conjoin_train=false
    +decoder.conjoin_test=false
    optimizer.lr="${LR}"
    optimizer.weight_decay="${WEIGHT_DECAY}"
    trainer.devices=1
    trainer.max_epochs="${MAX_EPOCHS}"
    trainer.precision="${PRECISION}"
    trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}"
    train.ckpt=null
    train.pretrained_model_path=null
    train.pretrained_model_state_hook._name_=null
    wandb=null
    hydra.run.dir="${HYDRA_RUN_DIR}"
  )

  "${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"
  echo "Completed seed=${seed}"
  echo "*****************************************************"
done
