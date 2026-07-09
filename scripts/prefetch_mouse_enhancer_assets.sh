#!/bin/bash

# Download the small GenomicBenchmarks mouse enhancer dataset and the published
# Caduceus-PS Hugging Face checkpoint before submitting GPU jobs. Run this on a
# login node that has network/proxy access; compute nodes may be offline.

set -Eeo pipefail
trap 'rc=$?; echo "ERROR: ${BASH_SOURCE[0]} failed at line ${LINENO} with exit code ${rc}" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}" || exit

CONTAINER_DIR="${CONTAINER_DIR:-/path/to/shared_containers}"
PYTORCH_IMAGE="${PYTORCH_IMAGE:-caduceus-pytorch-2.2.0-cuda12.1-cudnn8-devel.sif}"
SIF_IMAGE="${SIF_IMAGE:-${CONTAINER_DIR}/${PYTORCH_IMAGE}}"
VENV_PYTHON="${VENV_PYTHON:-/host/.venv/bin/python}"
TASK="${TASK:-dummy_mouse_enhancers_ensembl}"
GB_DATA_DIR="${GB_DATA_DIR:-${REPO_ROOT}/data/genomic_benchmark}"
HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH:-kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16}"

mkdir -p "${GB_DATA_DIR}" .cache/huggingface/hub .cache/huggingface/datasets .cache/huggingface/transformers

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
  echo "Run 'make install' from ${REPO_ROOT} before prefetching assets." >&2
  exit 1
fi

echo "Prefetching GenomicBenchmarks task: ${TASK}"
echo "Dataset destination: ${GB_DATA_DIR}"
echo "Prefetching Hugging Face model: ${HF_MODEL_NAME_OR_PATH}"

SINGULARITY_BINDS=(--bind "${REPO_ROOT}:/host:rw")
if [[ "${GB_DATA_DIR}" == "${REPO_ROOT}"/* ]]; then
  CONTAINER_GB_DATA_DIR="/host/${GB_DATA_DIR#${REPO_ROOT}/}"
else
  SINGULARITY_BINDS+=(--bind "${GB_DATA_DIR}:${GB_DATA_DIR}:rw")
  CONTAINER_GB_DATA_DIR="${GB_DATA_DIR}"
fi

echo "Container dataset destination: ${CONTAINER_GB_DATA_DIR}"

singularity exec \
  --pwd /host \
  "${SINGULARITY_BINDS[@]}" \
  "${SIF_IMAGE}" env \
  PYTHONPATH=/host \
  HF_HOME=/host/.cache/huggingface \
  HF_HUB_CACHE=/host/.cache/huggingface/hub \
  HF_DATASETS_CACHE=/host/.cache/huggingface/datasets \
  TRANSFORMERS_CACHE=/host/.cache/huggingface/transformers \
  TASK="${TASK}" \
  GB_DATA_DIR="${CONTAINER_GB_DATA_DIR}" \
  HF_MODEL_NAME_OR_PATH="${HF_MODEL_NAME_OR_PATH}" \
  "${VENV_PYTHON}" - <<'PY'
import os
from pathlib import Path

from genomic_benchmarks.data_check import is_downloaded
from genomic_benchmarks.loc2seq import download_dataset
from huggingface_hub import snapshot_download

task = os.environ["TASK"]
gb_data_dir = Path(os.environ["GB_DATA_DIR"])
model_name = os.environ["HF_MODEL_NAME_OR_PATH"]

gb_data_dir.mkdir(parents=True, exist_ok=True)
if is_downloaded(task, cache_path=gb_data_dir):
    print(f"{task} already downloaded under {gb_data_dir}")
else:
    download_dataset(task, version=0, dest_path=gb_data_dir)
    print(f"Downloaded {task} under {gb_data_dir}")

snapshot_download(repo_id=model_name)
print(f"Cached Hugging Face model: {model_name}")
PY
