#!/bin/bash
# Serve one model with vLLM and run the few-shot evaluations on it.
# Usage: scripts/eval_model.sh <key> [model path or HF repo id]
#   key        one of the keys in scripts/models.sh (it selects the eval scripts, the served name and the vLLM flags)
#   model      default: $PR_WAREHOUSE/models/<local folder> (the base model). Give a Post-Reason SFT model here,
#              e.g. richmondsin/Qwen3.5-4B-Post-Reasoning.
# Environment:
#   BENCHES     default "easy2hard gsm8k gpqa_main mmlu_pro bbh_full" (also: math)
#   STRATEGIES  default "Direct_Answer,Post_Reason" (also: Thinking, Thinking_Post for models with a thinking mode,
#               Basic_CoT)
#   GPUS        default: $CUDA_VISIBLE_DEVICES or 0
#   PORT        default: the port in scripts/models.sh
#   PR_WAREHOUSE  data, models and results folder (default: ./warehouse)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
key="${1:?usage: scripts/eval_model.sh <key> [model]}"
row="$(model_row "$key")" || { echo "unknown key $key"; exit 1; }
IFS='|' read -r repo local served dport flags <<< "$row"
W="${PR_WAREHOUSE:-warehouse}"
MODEL="${2:-$W/models/$local}"
PORT="${PORT:-$dport}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
BENCHES="${BENCHES:-easy2hard gsm8k gpqa_main mmlu_pro bbh_full}"
STRATEGIES="${STRATEGIES:-Direct_Answer,Post_Reason}"
mkdir -p "$W/logs"
LOG="$W/logs/vllm_${key}_${PORT}.log"
echo "serve $MODEL as $served on port $PORT (GPUs $GPUS); log $LOG"
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" --port "$PORT" --served-model-name "$served" \
  --enable-prefix-caching --trust-remote-code $flags > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
until curl -sf "http://localhost:$PORT/health" > /dev/null; do
  kill -0 $SERVER 2>/dev/null || { echo "vLLM stopped; see $LOG"; exit 1; }
  sleep 5
done
for b in $BENCHES; do
  python -u "eval/open/eval_base_${b}_fewshot_${key}.py" --port "$PORT" --model_name "$served" --strategies "$STRATEGIES"
done
