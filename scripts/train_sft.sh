#!/bin/bash
# Supervised Post-Reason tuning, steps 2 and 3: LoRA training with the loss on the post-answer justification only,
# then the merge of the adapter into the base model (LLaMA-Factory export).
# Usage: GPUS=0,1 scripts/train_sft.sh <key>
# The merge file sft/merge/merge_<key>_lora.yaml names the last checkpoint of our runs (e.g. checkpoint-87). If your run
# ends at another step, change adapter_name_or_path.
set -euo pipefail
cd "$(dirname "$0")/.."
key="${1:?usage: scripts/train_sft.sh <key>}"
GPUS="${GPUS:-${CUDA_VISIBLE_DEVICES:-0}}"
python -u "sft/train/train_sft_reflective_${key}.py" --gpu "$GPUS" --target_mode all
if [ "${PR_WAREHOUSE:-warehouse}" != "warehouse" ]; then
  echo "note: sft/merge/merge_${key}_lora.yaml uses ./warehouse paths; edit them for PR_WAREHOUSE=$PR_WAREHOUSE"
fi
llamafactory-cli export "sft/merge/merge_${key}_lora.yaml"
