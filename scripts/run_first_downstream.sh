#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-caduceus_env}"
PROXY_PORT="${PROXY_PORT:-7897}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/downstream/first_genomic_benchmark}"
MODEL_ID="${MODEL_ID:-kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3}"
SEED="${SEED:-1}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
elif [[ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda was not found. Run scripts/setup_wsl_caduceus_env.sh first." >&2
  exit 1
fi

conda activate "${ENV_NAME}"
export PYTHONPATH="${repo_root}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

proxy_args=()
if [[ -n "${HTTP_PROXY:-}" ]]; then
  proxy_args+=(--proxy "${HTTP_PROXY}")
else
  for host in 127.0.0.1 "$(ip route | awk '/^default / {print $3; exit}' 2>/dev/null || true)" "$(awk '/^nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null || true)"; do
    [[ -z "${host}" ]] && continue
    proxy="http://${host}:${PROXY_PORT}"
    if curl -I --connect-timeout 3 --max-time 8 --proxy "${proxy}" https://huggingface.co >/dev/null 2>&1; then
      export HTTP_PROXY="${proxy}"
      export HTTPS_PROXY="${proxy}"
      export http_proxy="${proxy}"
      export https_proxy="${proxy}"
      proxy_args+=(--proxy "${proxy}")
      echo "Using proxy: ${proxy}"
      break
    fi
  done
fi

python scripts/finetune_first_genomic_benchmark.py \
  --dataset dummy_mouse_enhancers_ensembl \
  --model-id "${MODEL_ID}" \
  --max-length "${MAX_LENGTH}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --eval-batch-size "${EVAL_BATCH_SIZE}" \
  --gradient-accumulation-steps "${GRAD_ACCUM}" \
  --lr 1e-3 \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_DIR}" \
  "${proxy_args[@]}"
