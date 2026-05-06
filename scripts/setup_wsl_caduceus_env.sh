#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-caduceus_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.8}"
PROXY_PORT="${PROXY_PORT:-7897}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

SUDO=()
if [[ "${EUID}" -ne 0 ]]; then
  SUDO=(sudo)
fi

detect_proxy() {
  local candidates=()
  candidates+=("http://127.0.0.1:${PROXY_PORT}")
  if command -v ip >/dev/null 2>&1; then
    local gateway_ip
    gateway_ip="$(ip route | awk '/^default / {print $3; exit}' || true)"
    if [[ -n "${gateway_ip}" ]]; then
      candidates+=("http://${gateway_ip}:${PROXY_PORT}")
    fi
  fi
  if [[ -f /etc/resolv.conf ]]; then
    local host_ip
    host_ip="$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf || true)"
    if [[ -n "${host_ip}" ]]; then
      candidates+=("http://${host_ip}:${PROXY_PORT}")
    fi
  fi

  if [[ -n "${HTTP_PROXY:-}" ]]; then
    echo "${HTTP_PROXY}"
    return 0
  fi

  if ! command -v curl >/dev/null 2>&1; then
    echo "${candidates[1]:-${candidates[0]}}"
    return 0
  fi

  for proxy in "${candidates[@]}"; do
    if curl -I --connect-timeout 3 --max-time 8 --proxy "${proxy}" https://github.com >/dev/null 2>&1; then
      echo "${proxy}"
      return 0
    fi
  done
}

proxy_url="$(detect_proxy || true)"
if [[ -n "${proxy_url}" ]]; then
  export HTTP_PROXY="${proxy_url}"
  export HTTPS_PROXY="${proxy_url}"
  export http_proxy="${proxy_url}"
  export https_proxy="${proxy_url}"
  if command -v git >/dev/null 2>&1; then
    git config --global http.proxy "${proxy_url}"
    git config --global https.proxy "${proxy_url}"
  fi
  cat >/etc/apt/apt.conf.d/95proxy <<EOF
Acquire::http::Proxy "${proxy_url}";
Acquire::https::Proxy "${proxy_url}";
EOF
  echo "Using proxy: ${proxy_url}"
else
  echo "No proxy detected. Continuing with direct network."
fi

"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y build-essential git git-lfs wget curl unzip ca-certificates ninja-build iproute2
if [[ -n "${proxy_url}" ]]; then
  git config --global http.proxy "${proxy_url}"
  git config --global https.proxy "${proxy_url}"
fi
git lfs install

if ! command -v conda >/dev/null 2>&1; then
  if [[ ! -x "${HOME}/miniconda3/bin/conda" ]]; then
    wget -O /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
  fi
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
  conda init bash >/dev/null || true
else
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true
conda config --set solver libmamba || true

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" \
    python="${PYTHON_VERSION}" pip=23.3.1 \
    pytorch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0 pytorch-cuda=12.1 \
    -c pytorch -c nvidia -c defaults
fi

conda activate "${ENV_NAME}"
if ! command -v nvcc >/dev/null 2>&1; then
  conda install -y -c nvidia cuda-nvcc=12.1.105 cuda-cudart-dev=12.1.105
fi
export PIP_DEFAULT_TIMEOUT="${PIP_DEFAULT_TIMEOUT:-120}"
export PIP_RETRIES="${PIP_RETRIES:-10}"
python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_DEFAULT_TIMEOUT}" --upgrade pip setuptools wheel packaging ninja
python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_DEFAULT_TIMEOUT}" \
  transformers==4.38.1 \
  huggingface-hub==0.24.7 \
  safetensors==0.4.5 \
  genomic-benchmarks==0.0.9 \
  scikit-learn==1.3.2 \
  pandas==2.0.3 \
  tqdm==4.66.1 \
  einops==0.7.0

MAX_JOBS="${MAX_JOBS:-2}" python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_DEFAULT_TIMEOUT}" --no-build-isolation causal-conv1d==1.2.0.post2
MAX_JOBS="${MAX_JOBS:-2}" python -m pip install --retries "${PIP_RETRIES}" --timeout "${PIP_DEFAULT_TIMEOUT}" --no-build-isolation mamba-ssm==1.2.0.post1

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import mamba_ssm, causal_conv1d
print("mamba_ssm and causal_conv1d import ok")
PY

echo "Environment ready. Activate it with: conda activate ${ENV_NAME}"
