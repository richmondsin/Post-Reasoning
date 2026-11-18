#!/bin/bash
# Supervised Post-Reason tuning, step 1: the base model writes its own post-answer justifications (self-distillation)
# for the 3,500 NuminaMath problems in $PR_WAREHOUSE/post_reflective_data/numina_3500_integer_master.jsonl.
# Usage: scripts/self_distill.sh <key>      (run sft/prepare_self_reflection_numina_data_creation.py once first)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
key="${1:?usage: scripts/self_distill.sh <key>}"
row="$(model_row "$key")" || { echo "unknown key $key"; exit 1; }
IFS='|' read -r repo local served dport flags <<< "$row"
W="${PR_WAREHOUSE:-warehouse}"
PORT="${PORT:-$dport}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
mkdir -p "$W/logs"
LOG="$W/logs/vllm_${key}_${PORT}.log"
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$W/models/$local" --port "$PORT" --served-model-name "$served" \
  --enable-prefix-caching --trust-remote-code $flags > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM stopped; see $LOG"; exit 1; }
  sleep 5
done
python -u "sft/self_distill/prepare_self_reflection_data_numina_${key}.py" --port "$PORT" --model_name "$served"
