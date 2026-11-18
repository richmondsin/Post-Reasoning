#!/bin/bash
# Download base models into $PR_WAREHOUSE/models/<local folder> (the folder names that the SFT and merge files use).
# Usage: scripts/download_models.sh qwen35_4b llama31_8b ...   (no argument: all 14 models)
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/models.sh
W="${PR_WAREHOUSE:-warehouse}"
for key in ${@:-$ALL_KEYS}; do
  row="$(model_row "$key")" || { echo "unknown key $key"; exit 1; }
  IFS='|' read -r repo local served port flags <<< "$row"
  hf download "$repo" --local-dir "$W/models/$local"
done
