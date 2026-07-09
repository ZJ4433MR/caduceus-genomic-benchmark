#!/bin/bash
# Fine-tune one DNALONGBENCH 450k ETGP/eQTL run.
#
# MODEL_KIND:
#   caduceus_ps  official Caduceus-PS 131k checkpoint
#   caduceus_ph  official Caduceus-Ph 131k checkpoint with post-hoc RC conjoining
#   mamba1       plain Mamba1 checkpoint from 131k hg38 MLM pre-training
#   mamba1_flip  Mamba1 checkpoint pre-trained with plain flipped augmentation;
#                downstream uses plain flipped augmentation and eval averaging
#   mamba1_tta   optional ablation: plain Mamba1 checkpoint with eval averaging only
#   mlbn         MLBN checkpoint from 131k hg38 MLM pre-training
#
# TASK_NAME:
#   etgp or eqtl

#SBATCH --get-user-env
#SBATCH --partition=A800
#SBATCH -t 48:00:00
#SBATCH --mem=160G
#SBATCH --gres=gpu:a800:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -N 1
#SBATCH --requeue
#SBATCH --job-name=dnalb450k
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
TASK_NAME="${TASK_NAME:-etgp}"
EQTL_SUBSET="${EQTL_SUBSET:-Cells_Cultured_fibroblasts}"
SEEDS="${SEEDS:-1}"
LR="${LR:-6e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
BATCH_SIZE_EVAL="${BATCH_SIZE_EVAL:-1}"
EPOCHS="${EPOCHS:-30}"
DEVICES="${DEVICES:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$((BATCH_SIZE * DEVICES))}"
USE_SRUN="${USE_SRUN:-false}"
SRUN_NTASKS="${SRUN_NTASKS:-${DEVICES}}"
SRUN_NTASKS_PER_NODE="${SRUN_NTASKS_PER_NODE:-${SRUN_NTASKS}}"
SRUN_GPUS_PER_TASK="${SRUN_GPUS_PER_TASK:-}"
SRUN_CPUS_PER_TASK="${SRUN_CPUS_PER_TASK:-${SLURM_CPUS_PER_TASK:-1}}"
if [ "${USE_SRUN}" = "true" ]; then
  TRAINER_DEVICES="${TRAINER_DEVICES:-${DEVICES}}"
else
  TRAINER_DEVICES="${TRAINER_DEVICES:-${DEVICES}}"
fi
PRECISION="${PRECISION:-32}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_LENGTH="${MAX_LENGTH:-450000}"
GRADIENT_CLIP_VAL="${GRADIENT_CLIP_VAL:-1.0}"
SMOKE="${SMOKE:-false}"
REQUIRE_PRETRAINED="${REQUIRE_PRETRAINED:-true}"
EQTL_CHECKPOINT_MODEL_FORWARD="${EQTL_CHECKPOINT_MODEL_FORWARD:-true}"
EQTL_FREEZE_MODEL_FORWARD="${EQTL_FREEZE_MODEL_FORWARD:-false}"
EQTL_FROZEN_MODEL_EVAL="${EQTL_FROZEN_MODEL_EVAL:-true}"
TASK_CHECKPOINT_MODEL_FORWARD="${TASK_CHECKPOINT_MODEL_FORWARD:-false}"
TASK_FREEZE_MODEL_FORWARD="${TASK_FREEZE_MODEL_FORWARD:-false}"
TASK_FROZEN_MODEL_EVAL="${TASK_FROZEN_MODEL_EVAL:-true}"
EXTRA_OVERRIDES="${EXTRA_OVERRIDES:-}"

DATA_ROOT="${DATA_ROOT:-data/dnalongbench/data_long_range_dna}"
REF_DATA_ROOT="${REF_DATA_ROOT:-data/reference}"
CADUCEUS_MODEL_ROOT="${CADUCEUS_MODEL_ROOT:-checkpoints/caduceus}"
CADUCEUS_LOCAL_FILES_ONLY="${CADUCEUS_LOCAL_FILES_ONLY:-true}"
CONTAINER_DIR="${CONTAINER_DIR:-/path/to/shared_containers}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-caduceus-pytorch-2.2.0-cuda12.1-cudnn8-devel.sif}"
SIF_IMAGE="${SIF_IMAGE:-${CONTAINER_DIR}/${PYTORCH_IMAGE}}"
VENV_PYTHON="${VENV_PYTHON:-/host/.venvs/dnalongbench450k-mamba1/bin/python}"

if [ "${SMOKE}" = "true" ]; then
  EPOCHS="${SMOKE_EPOCHS:-1}"
  MAX_STEPS="${SMOKE_MAX_STEPS:-1}"
  LIMIT_TRAIN_BATCHES="${SMOKE_LIMIT_TRAIN_BATCHES:-1}"
  LIMIT_VAL_BATCHES="${SMOKE_LIMIT_VAL_BATCHES:-1}"
  TRAIN_TEST="${SMOKE_TRAIN_TEST:-false}"
  REQUIRE_PRETRAINED="${SMOKE_REQUIRE_PRETRAINED:-false}"
else
  MAX_STEPS="${MAX_STEPS:-null}"
  LIMIT_TRAIN_BATCHES="${LIMIT_TRAIN_BATCHES:-1.0}"
  LIMIT_VAL_BATCHES="${LIMIT_VAL_BATCHES:-1.0}"
  TRAIN_TEST="${TRAIN_TEST:-true}"
fi

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

case "${TASK_NAME}" in
  etgp)
    EXPERIMENT="hg38/dnalongbench_etgp"
    TASK_SUFFIX="etgp"
    EXPECTED_CONFIGS=(
      "${DATA_ROOT}/enhancer_target_gene/config/CRISPRi_EPI_K562_hg19.config"
      "${DATA_ROOT}/enhancer_promoter_interaction/CRISPRi_EPI/config/CRISPRi_EPI_K562_hg19.config"
      "${DATA_ROOT}/config/CRISPRi_EPI_K562_hg19.config"
    )
    ;;
  eqtl)
    EXPERIMENT="hg38/dnalongbench_eqtl"
    TASK_SUFFIX="eqtl_${EQTL_SUBSET}"
    EXPECTED_CONFIGS=(
      "${DATA_ROOT}/eQTL/config/gtex_hg38.${EQTL_SUBSET}.config"
      "${DATA_ROOT}/eqtl/config/gtex_hg38.${EQTL_SUBSET}.config"
      "${DATA_ROOT}/config/gtex_hg38.${EQTL_SUBSET}.config"
    )
    ;;
  *)
    echo "TASK_NAME must be etgp or eqtl, got ${TASK_NAME}" >&2
    exit 2
    ;;
esac

CONFIG_FOUND=false
for config_file in "${EXPECTED_CONFIGS[@]}"; do
  if [ -f "${config_file}" ]; then
    CONFIG_FOUND=true
    break
  fi
done
if [ "${CONFIG_FOUND}" != "true" ]; then
  echo "DNALONGBENCH data/config not found under DATA_ROOT=${DATA_ROOT}" >&2
  printf 'Expected one of:\n' >&2
  printf '  %s\n' "${EXPECTED_CONFIGS[@]}" >&2
  exit 2
fi

host_to_container_path() {
  local path="$1"
  if [[ "${path}" == "${REPO_ROOT}"/* ]]; then
    echo "/host/${path#${REPO_ROOT}/}"
  else
    echo "${path}"
  fi
}

SINGULARITY_BINDS=(--bind "${REPO_ROOT}:/host:rw")
add_bind_dir() {
  local dir="$1"
  if [ -n "${dir}" ] && [ -e "${dir}" ] && [[ "${dir}" != "${REPO_ROOT}"* ]]; then
    SINGULARITY_BINDS+=(--bind "${dir}:${dir}:rw")
  fi
}
add_bind_dir "${DATA_ROOT}"
add_bind_dir "${REF_DATA_ROOT}"

DEFAULT_MAMBA_CKPT="${REPO_ROOT}/outputs/pretrain/dnalongbench450k_131k/mamba1_mlm_seqlen-131k_lr-8e-3_gb-8_steps-50000/checkpoints/last.ckpt"
DEFAULT_MAMBA_FLIP_CKPT="${REPO_ROOT}/outputs/pretrain/dnalongbench450k_131k/mamba1_plain_flip_mlm_seqlen-131k_lr-8e-3_gb-8_steps-50000/checkpoints/last.ckpt"
DEFAULT_MLBN_CKPT="${REPO_ROOT}/outputs/pretrain/dnalongbench450k_131k/mlbn_mlm_seqlen-131k_lr-1e-4_gb-8_steps-50000_l40_4gpu_bf16_lr1e4/checkpoints/last.ckpt"

DISPLAY_NAME=""
MODEL_ARGS=()
CKPT_PATH=""
TOKENIZER_NAME="char"
DATASET_RC_AUG=false
DATASET_REVERSE_AUG=false
DATASET_CONJOIN_TRAIN=false
DATASET_CONJOIN_TEST=false
MODEL_CONJOIN_TEST=false
DECODER_CONJOIN_TRAIN=false
DECODER_CONJOIN_TEST=false
DECODER_PLAIN_BIDIR_EVAL=false

case "${MODEL_KIND}" in
  caduceus_ps)
    DISPLAY_NAME="caduceus_ps_131k"
    TOKENIZER_NAME="caduceus"
    DECODER_CONJOIN_TRAIN=true
    DECODER_CONJOIN_TEST=true
    MODEL_PATH="${MODEL_PATH:-${CADUCEUS_MODEL_ROOT}/caduceus-ps_seqlen-131k_d_model-256_n_layer-16}"
    if [ ! -d "${MODEL_PATH}" ]; then
      MODEL_PATH="kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"
    else
      add_bind_dir "${CADUCEUS_MODEL_ROOT}"
    fi
    MODEL_ARGS=(
      model=caduceus_hf
      model._name_=dna_embedding_caduceus
      model.hf_model_name_or_path="$(host_to_container_path "${MODEL_PATH}")"
      model.local_files_only="${CADUCEUS_LOCAL_FILES_ONLY}"
    )
    ;;
  caduceus_ph)
    DISPLAY_NAME="caduceus_ph_131k"
    TOKENIZER_NAME="caduceus"
    MODEL_PATH="${MODEL_PATH:-${CADUCEUS_MODEL_ROOT}/caduceus-ph_seqlen-131k_d_model-256_n_layer-16}"
    if [ ! -d "${MODEL_PATH}" ]; then
      MODEL_PATH="kuleshov-group/caduceus-ph_seqlen-131k_d_model-256_n_layer-16"
    else
      add_bind_dir "${CADUCEUS_MODEL_ROOT}"
    fi
    MODEL_ARGS=(
      model=caduceus_hf
      model._name_=dna_embedding_caduceus
      model.hf_model_name_or_path="$(host_to_container_path "${MODEL_PATH}")"
      model.local_files_only="${CADUCEUS_LOCAL_FILES_ONLY}"
      model.conjoin_test=true
    )
    DATASET_CONJOIN_TEST=true
    MODEL_CONJOIN_TEST=true
    DECODER_CONJOIN_TEST=true
    ;;
  mamba1)
    DISPLAY_NAME="mamba1_131k"
    CKPT_PATH="${MAMBA_CKPT:-${DEFAULT_MAMBA_CKPT}}"
    MODEL_ARGS=(
      model=mamba_dna
      model.config.d_model="${D_MODEL:-256}"
      model.config.n_layer="${N_LAYER:-16}"
    )
    ;;
  mamba1_flip)
    DISPLAY_NAME="mamba1_plain_flip_131k"
    CKPT_PATH="${MAMBA_FLIP_CKPT:-${DEFAULT_MAMBA_FLIP_CKPT}}"
    DATASET_REVERSE_AUG=true
    DECODER_PLAIN_BIDIR_EVAL=true
    MODEL_ARGS=(
      model=mamba_dna
      model.config.d_model="${D_MODEL:-256}"
      model.config.n_layer="${N_LAYER:-16}"
    )
    ;;
  mamba1_tta)
    DISPLAY_NAME="mamba1_tta_131k"
    CKPT_PATH="${MAMBA_CKPT:-${DEFAULT_MAMBA_CKPT}}"
    DECODER_PLAIN_BIDIR_EVAL=true
    MODEL_ARGS=(
      model=mamba_dna
      model.config.d_model="${D_MODEL:-256}"
      model.config.n_layer="${N_LAYER:-16}"
    )
    ;;
  mlbn)
    DISPLAY_NAME="mlbn_131k"
    CKPT_PATH="${MLBN_CKPT:-${DEFAULT_MLBN_CKPT}}"
    MODEL_ARGS=(model=mlbn_dna_d192_l13)
    ;;
  *)
    echo "MODEL_KIND must be caduceus_ps, caduceus_ph, mamba1, mamba1_flip, mamba1_tta, or mlbn. Got ${MODEL_KIND}" >&2
    exit 2
    ;;
esac

if [ -n "${DISPLAY_NAME_OVERRIDE:-}" ]; then
  DISPLAY_NAME="${DISPLAY_NAME_OVERRIDE}"
fi

if [ -n "${CKPT_PATH}" ]; then
  if [ ! -f "${CKPT_PATH}" ]; then
    if [ "${REQUIRE_PRETRAINED}" = "true" ]; then
      echo "Pre-trained checkpoint not found for ${MODEL_KIND}: ${CKPT_PATH}" >&2
      echo "Set the corresponding *_CKPT env var, or run with SMOKE=true / REQUIRE_PRETRAINED=false for a scratch smoke test." >&2
      exit 2
    fi
    CKPT_PATH=""
  else
    add_bind_dir "$(dirname "${CKPT_PATH}")"
  fi
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

echo "REPO_ROOT=${REPO_ROOT}"
echo "SIF_IMAGE=${SIF_IMAGE}"
echo "VENV_PYTHON=${VENV_PYTHON}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "TASK_NAME=${TASK_NAME}"
echo "MODEL_KIND=${MODEL_KIND}"
echo "DISPLAY_NAME=${DISPLAY_NAME}"
echo "CKPT_PATH=${CKPT_PATH:-scratch_or_hf}"
echo "SMOKE=${SMOKE}"
echo "USE_SRUN=${USE_SRUN}"
echo "DEVICES=${DEVICES}"
echo "TRAINER_DEVICES=${TRAINER_DEVICES}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
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
import kipoiseq, pyfaidx, tabix, pandas
print("DNALONGBENCH deps: ok")
PY

for seed in ${SEEDS}; do
  RUN_NAME="${DISPLAY_NAME}_${TASK_SUFFIX}_seed-${seed}_lr-${LR}"
  if [ "${SMOKE}" = "true" ]; then
    RUN_NAME="smoke_${RUN_NAME}"
  fi
  HOST_OUTPUT_DIR="${REPO_ROOT}/outputs/downstream/dnalongbench450k/${TASK_SUFFIX}/${RUN_NAME}"
  HYDRA_RUN_DIR="/host/outputs/downstream/dnalongbench450k/${TASK_SUFFIX}/${RUN_NAME}"
  CHECKPOINT_DIR="${HYDRA_RUN_DIR}/checkpoints"
  mkdir -p "${HOST_OUTPUT_DIR}/checkpoints" watch_folder

  TRAIN_ARGS=(
    -m train
    experiment="${EXPERIMENT}"
    callbacks.model_checkpoint.dirpath="${CHECKPOINT_DIR}"
    callbacks.model_checkpoint_every_n_steps.dirpath="${CHECKPOINT_DIR}"
    callbacks.model_checkpoint_every_n_steps.every_n_train_steps=5000
    dataset.data_root="$(host_to_container_path "${DATA_ROOT}")"
    dataset.task_name="${TASK_NAME}"
    dataset.subset="${EQTL_SUBSET}"
    dataset.tokenizer_name="${TOKENIZER_NAME}"
    dataset.max_length="${MAX_LENGTH}"
    dataset.batch_size="${BATCH_SIZE}"
    dataset.batch_size_eval="${BATCH_SIZE_EVAL}"
    dataset.rc_aug="${DATASET_RC_AUG}"
    dataset.reverse_aug="${DATASET_REVERSE_AUG}"
    dataset.conjoin_train="${DATASET_CONJOIN_TRAIN}"
    dataset.conjoin_test="${DATASET_CONJOIN_TEST}"
    loader.num_workers="${NUM_WORKERS}"
    "${MODEL_ARGS[@]}"
    model.conjoin_test="${MODEL_CONJOIN_TEST}"
    ++decoder.conjoin_train="${DECODER_CONJOIN_TRAIN}"
    ++decoder.conjoin_test="${DECODER_CONJOIN_TEST}"
    ++decoder.plain_bidir_eval="${DECODER_PLAIN_BIDIR_EVAL}"
    optimizer.lr="${LR}"
    optimizer.weight_decay="${WEIGHT_DECAY}"
    train.seed="${seed}"
    ++train.global_batch_size="${GLOBAL_BATCH_SIZE}"
    train.ckpt=null
    train.test="${TRAIN_TEST}"
    trainer.devices="${TRAINER_DEVICES}"
    trainer.max_epochs="${EPOCHS}"
    trainer.precision="${PRECISION}"
    trainer.gradient_clip_val="${GRADIENT_CLIP_VAL}"
    trainer.num_sanity_val_steps=0
    trainer.limit_train_batches="${LIMIT_TRAIN_BATCHES}"
    trainer.limit_val_batches="${LIMIT_VAL_BATCHES}"
    wandb=null
    hydra.run.dir="${HYDRA_RUN_DIR}"
  )

  if [ "${TASK_NAME}" = "eqtl" ]; then
    TRAIN_ARGS+=(
      ++task.checkpoint_model_forward="${EQTL_CHECKPOINT_MODEL_FORWARD}"
      ++task.checkpoint_use_reentrant=false
      ++task.freeze_model_forward="${EQTL_FREEZE_MODEL_FORWARD}"
      ++task.frozen_model_eval="${EQTL_FROZEN_MODEL_EVAL}"
    )
  elif [ "${TASK_NAME}" = "etgp" ]; then
    TRAIN_ARGS+=(
      ++task.checkpoint_model_forward="${TASK_CHECKPOINT_MODEL_FORWARD}"
      ++task.checkpoint_use_reentrant=false
      ++task.freeze_model_forward="${TASK_FREEZE_MODEL_FORWARD}"
      ++task.frozen_model_eval="${TASK_FROZEN_MODEL_EVAL}"
    )
  fi

  if [ "${MAX_STEPS}" != "null" ]; then
    TRAIN_ARGS+=(++trainer.max_steps="${MAX_STEPS}")
  fi

  if [ "${USE_SRUN}" = "true" ]; then
    TRAIN_ARGS+=(
      ++trainer.strategy="${TRAINER_STRATEGY:-ddp}"
      ++trainer.num_nodes="${TRAINER_NUM_NODES:-1}"
    )
  fi

  if [ -n "${CKPT_PATH}" ]; then
    TRAIN_ARGS+=(
      train.pretrained_model_path="$(host_to_container_path "${CKPT_PATH}")"
      train.pretrained_model_strict_load=false
      train.pretrained_model_state_hook._name_=load_backbone
    )
  else
    TRAIN_ARGS+=(
      train.pretrained_model_path=null
      train.pretrained_model_state_hook._name_=null
    )
  fi

  if [ -n "${EXTRA_OVERRIDES}" ]; then
    # shellcheck disable=SC2206
    EXTRA_ARGS=(${EXTRA_OVERRIDES})
    TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
  fi

  echo "*****************************************************"
  echo "Running ${RUN_NAME}"
  echo "Hydra dir=${HYDRA_RUN_DIR}"
  RUN_PREFIX=()
  if [ "${USE_SRUN}" = "true" ]; then
    RUN_PREFIX=(
      srun
      --ntasks="${SRUN_NTASKS}"
      --ntasks-per-node="${SRUN_NTASKS_PER_NODE}"
      --cpus-per-task="${SRUN_CPUS_PER_TASK}"
      --gpu-bind=none
    )
    if [ -n "${SRUN_GPUS_PER_TASK}" ]; then
      RUN_PREFIX+=(--gpus-per-task="${SRUN_GPUS_PER_TASK}")
    fi
  fi
  "${RUN_PREFIX[@]}" "${SINGULARITY_BASE[@]}" env PYTHONPATH=/host "${VENV_PYTHON}" "${TRAIN_ARGS[@]}"
  echo "*****************************************************"
done
