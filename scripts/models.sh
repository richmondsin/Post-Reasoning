#!/bin/bash
# One row per model key: the Hugging Face base model, the local folder under $PR_WAREHOUSE/models,
# the served model name (the eval scripts send it as --model_name), the vLLM port and the extra vLLM flags.
# The paper's settings: Qwen3.5 serves with --language-model-only and the qwen3 reasoning parser; Ministral-3 with the
# Mistral tokenizer; Llama-3.3-70B with tensor parallel 2.
model_row() {
  case "$1" in
    qwen35_4b)        echo "Qwen/Qwen3.5-4B|Qwen35-4B|Qwen35-4B|8019|--language-model-only --reasoning-parser qwen3 --gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    qwen35_9b)        echo "Qwen/Qwen3.5-9B|Qwen35-9B|Qwen35-9B|8020|--language-model-only --reasoning-parser qwen3 --gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    qwen35_27b)       echo "Qwen/Qwen3.5-27B|Qwen35-27B|Qwen35-27B|8021|--language-model-only --reasoning-parser qwen3 --gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    llama31_8b)       echo "meta-llama/Llama-3.1-8B-Instruct|Llama-31-8B-Instruct|Llama-31-8B-Instruct|8018|--gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    llama33_70b)      echo "meta-llama/Llama-3.3-70B-Instruct|Llama-33-70B-Instruct|Llama-33-70B-Instruct|8009|--tensor-parallel-size 2 --gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    gemma3_12b)       echo "google/gemma-3-12b-it|gemma-3-12b-it|gemma-3-12b-it|8010|--gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    gemma3_27b)       echo "google/gemma-3-27b-it|gemma-3-27b-it|gemma-3-27b-it|8011|--gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    ministral3_8b)    echo "mistralai/Ministral-3-8B-Instruct-2512|Ministral-3-8B-Instruct-2512|Ministral-3-8B-Instruct-2512|8015|--tokenizer_mode mistral --config_format mistral --load_format auto --gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    ministral3_14b)   echo "mistralai/Ministral-3-14B-Instruct-2512|Ministral-3-14B-Instruct-2512|Ministral-3-14B-Instruct-2512|8016|--tokenizer_mode mistral --config_format mistral --load_format auto --gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    mistralsmall_24b) echo "mistralai/Mistral-Small-24B-Instruct-2501|Mistral-Small-24B-Instruct-2501|Mistral-Small-24B-Instruct-2501|8017|--gpu-memory-utilization 0.70 --max-model-len 16384" ;;
    qwen3_8b)         echo "Qwen/Qwen3-8B|Qwen3-8B|Qwen3-8B|8012|--gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    qwen3_14b)        echo "Qwen/Qwen3-14B|Qwen3-14B|Qwen3-14B|8013|--gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    qwen3_32b)        echo "Qwen/Qwen3-32B|Qwen3-32B|Qwen3-32B|8014|--gpu-memory-utilization 0.90 --max-model-len 24576" ;;
    gptoss_20b)       echo "openai/gpt-oss-20b|gpt-oss-20b|gpt-oss-20b|8008|--gpu-memory-utilization 0.80 --max-model-len 32768" ;;
    *) return 1 ;;
  esac
}
ALL_KEYS="qwen35_4b qwen35_9b qwen35_27b llama31_8b llama33_70b gemma3_12b gemma3_27b ministral3_8b ministral3_14b mistralsmall_24b qwen3_8b qwen3_14b qwen3_32b gptoss_20b"
