#!/usr/bin/env bash
set -euo pipefail

SEEDS="${SEEDS:-2 3}"
PROXY_PORT="${PROXY_PORT:-7897}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/downstream}"
RUN_PREFIX="${RUN_PREFIX:-first_genomic_benchmark}"

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

proxy="http://172.30.112.1:${PROXY_PORT}"
if ! curl -I --connect-timeout 3 --max-time 8 --proxy "${proxy}" https://huggingface.co >/dev/null 2>&1; then
  gateway="$(ip route | awk '/^default / {print $3; exit}' 2>/dev/null || true)"
  proxy="http://${gateway}:${PROXY_PORT}"
fi

export HTTP_PROXY="${proxy}"
export HTTPS_PROXY="${proxy}"
export http_proxy="${proxy}"
export https_proxy="${proxy}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

ps_model="kuleshov-group/caduceus-ps_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"
ph_model="kuleshov-group/caduceus-ph_seqlen-1k_d_model-118_n_layer-4_lr-8e-3"

for seed in ${SEEDS}; do
  echo "=== PS seed ${seed} ==="
  MODEL_ID="${ps_model}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_PREFIX}_ps_seed${seed}" \
    SEED="${seed}" \
    bash scripts/run_first_downstream.sh

  echo "=== PH seed ${seed} ==="
  MODEL_ID="${ph_model}" \
    OUTPUT_DIR="${OUTPUT_ROOT}/${RUN_PREFIX}_ph_seed${seed}" \
    SEED="${seed}" \
    bash scripts/run_first_downstream.sh
done
