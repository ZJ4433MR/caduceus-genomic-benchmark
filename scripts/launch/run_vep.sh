#!/usr/bin/env bash
set -Eeuo pipefail

MODE="${1:-embeddings_primary}"
PYTHON="${PYTHON:-python}"
FLIPPED_GARI_CHECKPOINT="${FLIPPED_GARI_CHECKPOINT:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/vep}"
EMBED_ROOT="${EMBED_ROOT:-${OUTPUT_ROOT}/flipped_gari_131kb}"
DEVICES="${DEVICES:-8}"

run_embeddings() {
  local seq_len="$1"
  local window_size="$2"
  local name="$3"
  if [[ -z "${FLIPPED_GARI_CHECKPOINT}" ]]; then
    echo "FLIPPED_GARI_CHECKPOINT is required" >&2
    exit 2
  fi
  "${PYTHON}" -m torch.distributed.run \
    --standalone --nnodes=1 --nproc-per-node="${DEVICES}" \
    vep_embeddings.py \
    --model_name_or_path=flipped-gari-local \
    --flipped_gari_config=configs/model/flipped_gari_vep.yaml \
    --flipped_gari_checkpoint="${FLIPPED_GARI_CHECKPOINT}" \
    --vep_task_name=variant_effect_causal_eqtl \
    --seq_len="${seq_len}" \
    --bp_per_token=1 \
    --window_size_bp="${window_size}" \
    --eval_orientation=flip \
    --autocast_dtype=bf16 \
    --train_samples_per_tss_bucket=5000 \
    --train_subset_seed=2222 \
    --downstream_save_dir="${OUTPUT_ROOT}" \
    --name="${name}"
}

case "${MODE}" in
  embeddings_primary)
    run_embeddings 131072 1536 flipped_gari_131kb
    ;;
  embeddings_context_1kb)
    run_embeddings 1024 1024 flipped_gari_context_1kb
    ;;
  embeddings_context_16kb)
    run_embeddings 16384 1024 flipped_gari_context_16kb
    ;;
  embeddings_context_131kb)
    run_embeddings 131072 1024 flipped_gari_context_131kb
    ;;
  probe_primary)
    "${PYTHON}" vep_svm_eval.py \
      --path_to_outputs="${OUTPUT_ROOT}" \
      --embed_path="${EMBED_ROOT}" \
      --model_name=Flipped-GARI \
      --key=concat_avg_ws \
      --cs 0.1 0.3 1 3 5 10 30 100 \
      --seeds 1 2 3 4 5 6 7 8 9 10 \
      --sample_size=5000 \
      --score_mode=hard_label \
      --use_tissue \
      --output_prefix="${OUTPUT_ROOT}/flipped_gari_primary"
    ;;
  *)
    echo "Unknown mode: ${MODE}" >&2
    exit 2
    ;;
esac
