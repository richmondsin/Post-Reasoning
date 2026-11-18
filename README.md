<div align="center">
<h1>Post-Reasoning</h1>
</div>

<div align="center">
<h2>Post-Reasoning: Improving the Performance of Non-Thinking Models at Negligible Cost</h2>
</div>
<div align="center">

[Richmond Sin Jing Xuan](https://richmondsin.vercel.app/)<sup>1</sup>,
[Rishabh Bhardwaj](https://www.alphaxiv.org/@rishabh-bhardwaj)<sup>1</sup>,
[Soujanya Poria](https://soujanyaporia.github.io/)<sup>1</sup>

<sup>1</sup>[DeCLaRe Lab](https://declare-lab.github.io/), Nanyang Technological University

[![arXiv](https://img.shields.io/badge/arXiv-2605.06165-b31b1b.svg)](https://arxiv.org/abs/2605.06165)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/collections/richmondsin/post-reasoning-6afdf19ca204e7937ef56620)

</div>

This repository contains the official implementation and evaluation code of **Post-Reasoning: Improving the Performance
of Non-Thinking Models at Negligible Cost** ([arXiv 2605.06165](https://arxiv.org/abs/2605.06165)).

Reasoning traces before the answer increase the token cost and the latency of each query. Many tasks need little or no
reasoning. Post-Reasoning improves instruction-tuned (non-thinking) models with a different instruction: the model
states the final answer first, then it justifies the answer. No reasoning token comes before the answer. A stop marker
after the answer (for example `Explanation:`) gives the answer at the cost of a direct answer. Supervised Post-Reason
tuning puts this behaviour into the weights: the loss is only on the justification that comes after the answer.

## ✨ Highlights

- ⚡ **No tokens before the answer**: the model writes the answer first. You can stop the generation at the end of the
  answer, so the answer costs the same as a direct answer.
- 🧩 **Only an instruction**: "State the final answer immediately and justify your answer." No extra model, no tool and
  no change to the decoder.
- 🎓 **Supervised Post-Reason tuning**: LoRA training on self-distilled justifications for 3,500 NuminaMath problems. The
  loss is masked on the prompt and on the answer.
- 📈 **Results of the paper**: 117 model-benchmark settings (13 open and proprietary models, 7 model families, 9
  benchmarks). Post-Reason prompting improves on direct answering in 84.62 % of the settings, with a mean relative
  improvement of 15.57 %. The tuned models are better than direct answering in 82 of 90 settings (91.11 %) and better
  than Post-Reason prompting in 66 of 90 settings (73.33 %). The full tables are in the
  [paper](https://arxiv.org/abs/2605.06165).

## 🧩 Benchmarks

| Benchmark | Source | Test items | Answer |
|---|---|---|---|
| AMC 8 / AMC 10 / AMC 12 | [Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench) (E2H-AMC) | 268 / 445 / 271 | integer |
| HMMT Feb / HMMT Nov | [Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench) (E2H-AMC, HMMT rows) | 416 / 367 | integer |
| GSM8K | [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (test) | 1,319 | number |
| GPQA Main | [idavidrein/gpqa](https://huggingface.co/datasets/idavidrein/gpqa) (gpqa_main) | 445 | letter A-D |
| MMLU-Pro | [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) (test, random 3,000, seed 42) | 3,000 | letter A-J |
| BIG-Bench Hard | [lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh) (all 27 tasks) | 6,511 | task answer |

All prompts are 3-shot. Every prompt asks for the answer in the form `Answer: X`. The Post-Reason shots add an
`Explanation:` after the answer. `data/prepare_math_fewshot_data.py` (MATH) is used only in the training ablation of
the paper.

## 🧠 Method

- **Direct**: the instruction asks for the answer only ("Answer immediately without any explanation or reasoning").
- **Post-Reason prompting**: the instruction asks for the answer first, then the justification ("State the final answer
  immediately as 'Answer: [Answer].' THEN, explain your reasoning in 'Explanation: '").
- **Supervised Post-Reason tuning**, three steps:
  1. `sft/prepare_self_reflection_numina_data_creation.py` selects 3,500 integer-answer problems from
     [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT). It removes the sources that overlap with
     the test benchmarks (`aops_forum`, `math`, `gsm8k`, `amc_aime`) and the proofs.
  2. Self-distillation: the base model writes a formal derivation that ends at the known answer
     (`sft/self_distill/`).
  3. LoRA training (r 16, alpha 32, dropout 0.05; learning rate 2e-5, cosine schedule, 3 epochs, effective batch 32).
     The loss is only on the justification tokens: `-sum_i log p(c_i | x, y, c_<i)`, where x is the prompt, y the answer
     and c the justification (`sft/train/`). Then LLaMA-Factory merges the adapter into the base model (`sft/merge/`).

## 🛠️ Installation

### Requirements

- Linux with NVIDIA GPUs. vLLM serves the open models. Llama-3.3-70B uses 2 GPUs (tensor parallel 2).
- Python 3.11, PyTorch, vLLM, transformers, peft, datasets. The merge step uses
  [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (`llamafactory-cli`).
- For the API models: an API key in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`), or in a
  `.env` file in the folder where you run the script (see `.env.example`).
- The training scripts log to Weights & Biases (`report_to="wandb"`). Run `wandb login`, or set `WANDB_MODE=offline`.

### Install

```bash
git clone https://github.com/richmondsin/Post-Reasoning && cd Post-Reasoning
# install PyTorch and vLLM for your CUDA version first
pip install -r requirements.txt
# data, models and results go to ./warehouse; set PR_WAREHOUSE to use another folder
export PR_WAREHOUSE=$PWD/warehouse
```

Run all commands from the root of the repository.

## 🧑‍💻 Usage

The model keys are `qwen35_4b`, `qwen35_9b`, `qwen35_27b`, `llama31_8b`, `llama33_70b`, `gemma3_12b`, `gemma3_27b`,
`ministral3_8b`, `ministral3_14b` and `mistralsmall_24b` (the 10 models of the main tables), and `qwen3_8b`,
`qwen3_14b`, `qwen3_32b` and `gptoss_20b` (native thinking, Appendix D). `scripts/models.sh` gives the base model, the
served name and the vLLM settings of each key.

### 1. Prepare the test prompts

```bash
python data/prepare_easy2hard_data.py          # AMC 8/10/12, HMMT Feb/Nov
python data/prepare_gsm8k_fewshot_data.py
python data/prepare_gpqa_main_fewshot_data.py  # GPQA is gated on Hugging Face: accept its terms first
python data/prepare_mmlu_pro_fewshot_data.py
python data/prepare_bbh_full_fewshot_data.py
```

### 2. Test a model: Direct and Post-Reason prompting

```bash
# the base model: download it to $PR_WAREHOUSE/models, then serve it with vLLM and run the 5 benchmark scripts
scripts/download_models.sh qwen35_4b
GPUS=0 scripts/eval_model.sh qwen35_4b
# a released Post-Reason tuned model (from the Hugging Face Hub)
GPUS=0 scripts/eval_model.sh qwen35_4b richmondsin/Qwen3.5-4B-Post-Reasoning
# native thinking with and without the Post-Reason instruction (Appendix D)
GPUS=0 STRATEGIES=Thinking,Thinking_Post scripts/eval_model.sh qwen3_8b
# Llama-3.3-70B on 2 GPUs; only GPQA and MMLU-Pro
GPUS=0,1 BENCHES="gpqa_main mmlu_pro" scripts/eval_model.sh llama33_70b
```

Each script writes one file per arm and benchmark:
`$PR_WAREHOUSE/results/base_<served name>_<arm>_<benchmark>_fewshot_<k>.jsonl`, with one row per item (`truth`, `pred`,
`is_correct`, `response`, `raw_response`). The accuracy is the mean of `is_correct`. The arms are `Direct_Answer`,
`Post_Reason`, `Thinking` (thinking on, answer only) and `Thinking_Post` (thinking on, Post-Reason instruction).

You can also run one script by hand against a running OpenAI-compatible server:

```bash
python eval/open/eval_base_gpqa_main_fewshot_qwen35_4b.py --port 8019 --model_name Qwen35-4B --strategies Direct_Answer,Post_Reason
```

### 3. Test the API models

```bash
python eval/api/run_eval_all_base_gpt_54_mini_batch.py      # GPT-5.4-Mini (OpenAI Batch API)
python eval/api/run_eval_all_base_claude_haiku_45_batch.py  # Claude-Haiku-4.5 (Anthropic Batch API)
python eval/api/run_eval_all_base_gemini_25_flash.py        # Gemini-2.5-Flash
python eval/api/run_eval_lowthink.py --help                 # the low-thinking runs (Appendix D)
```

The `*_patch.py` scripts run again only the items that have no result (for example after an API error).

### 4. Supervised Post-Reason tuning

```bash
# the 3,500 NuminaMath problems (once)
python sft/prepare_self_reflection_numina_data_creation.py
# self-distillation: the base model writes the justifications
GPUS=0 scripts/self_distill.sh qwen35_4b
# LoRA training on the justifications only, then the merge into a full model
GPUS=0 scripts/train_sft.sh qwen35_4b
# test the merged model
GPUS=0 scripts/eval_model.sh qwen35_4b $PR_WAREHOUSE/models/Qwen35-4B-numina-Reflection-Merged-v4
```

## 📦 Released models

The Post-Reason tuned models (the LoRA adapter merged into the base model, bf16) of the 10 models in the paper, in the
Hugging Face collection [`richmondsin/post-reasoning`](https://huggingface.co/collections/richmondsin/post-reasoning-6afdf19ca204e7937ef56620):

| Family | Base model | Post-Reason tuned model |
|---|---|---|
| Qwen3.5 | [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) | [richmondsin/Qwen3.5-4B-Post-Reasoning](https://huggingface.co/richmondsin/Qwen3.5-4B-Post-Reasoning) |
| Qwen3.5 | [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | [richmondsin/Qwen3.5-9B-Post-Reasoning](https://huggingface.co/richmondsin/Qwen3.5-9B-Post-Reasoning) |
| Qwen3.5 | [Qwen/Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) | [richmondsin/Qwen3.5-27B-Post-Reasoning](https://huggingface.co/richmondsin/Qwen3.5-27B-Post-Reasoning) |
| Llama | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | [richmondsin/Llama-3.1-8B-Instruct-Post-Reasoning](https://huggingface.co/richmondsin/Llama-3.1-8B-Instruct-Post-Reasoning) |
| Llama | [meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) | [richmondsin/Llama-3.3-70B-Instruct-Post-Reasoning](https://huggingface.co/richmondsin/Llama-3.3-70B-Instruct-Post-Reasoning) |
| Gemma-3 | [google/gemma-3-12b-it](https://huggingface.co/google/gemma-3-12b-it) | [richmondsin/gemma-3-12b-it-Post-Reasoning](https://huggingface.co/richmondsin/gemma-3-12b-it-Post-Reasoning) |
| Gemma-3 | [google/gemma-3-27b-it](https://huggingface.co/google/gemma-3-27b-it) | [richmondsin/gemma-3-27b-it-Post-Reasoning](https://huggingface.co/richmondsin/gemma-3-27b-it-Post-Reasoning) |
| Ministral-3 | [mistralai/Ministral-3-8B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512) | [richmondsin/Ministral-3-8B-Instruct-2512-Post-Reasoning](https://huggingface.co/richmondsin/Ministral-3-8B-Instruct-2512-Post-Reasoning) |
| Ministral-3 | [mistralai/Ministral-3-14B-Instruct-2512](https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512) | [richmondsin/Ministral-3-14B-Instruct-2512-Post-Reasoning](https://huggingface.co/richmondsin/Ministral-3-14B-Instruct-2512-Post-Reasoning) |
| Mistral | [mistralai/Mistral-Small-24B-Instruct-2501](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) | [richmondsin/Mistral-Small-24B-Instruct-2501-Post-Reasoning](https://huggingface.co/richmondsin/Mistral-Small-24B-Instruct-2501-Post-Reasoning) |

Each model uses the license of its base model.

A quick test with vLLM:

```python
from vllm import LLM, SamplingParams
llm = LLM(model="richmondsin/Qwen3.5-4B-Post-Reasoning")
msg = [{"role": "user", "content": "What is 10 + 2? State the final answer immediately and justify your answer."}]
print(llm.chat(msg, SamplingParams(temperature=0, max_tokens=512))[0].outputs[0].text)
# to read only the answer, stop at the end of the answer: SamplingParams(..., stop=["Explanation"])
```

## 📁 Repository layout

```text
data/              build the 3-shot test prompts (Direct and Post-Reason) of each benchmark
eval/open/         one evaluation script for each benchmark and open model (OpenAI-compatible client to a vLLM server)
eval/api/          the API models: GPT-5.4-Mini, Claude-Haiku-4.5, Gemini-2.5-Flash, and the low-thinking runs
sft/               the NuminaMath training problems
sft/self_distill/  self-distillation of the post-answer justifications, one script for each model
sft/train/         LoRA training with the loss on the justification only, one script for each model
sft/merge/         LLaMA-Factory export files: merge the adapter into the base model
scripts/           model table, download, serve + evaluate, self-distill, train + merge
```

## 📝 Citation

```bibtex
@article{sin2026postreasoning,
  title   = {Post-Reasoning: Improving the Performance of Non-Thinking Models at Negligible Cost},
  author  = {Sin, Richmond Jing Xuan and Bhardwaj, Rishabh and Poria, Soujanya},
  journal = {arXiv preprint arXiv:2605.06165},
  year    = {2026}
}
```

## 📜 License

MIT ([LICENSE](LICENSE)). The released models follow the licenses of their base models. The datasets follow their own
licenses.

## 🙏 Acknowledgements

We build on [vLLM](https://github.com/vllm-project/vllm) (serving), [PEFT](https://github.com/huggingface/peft) (LoRA)
and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) (merge). The training problems come from
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT). The test sets come from
[Easy2Hard-Bench](https://huggingface.co/datasets/furonghuang-lab/Easy2Hard-Bench),
[GSM8K](https://huggingface.co/datasets/openai/gsm8k), [GPQA](https://huggingface.co/datasets/idavidrein/gpqa),
[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) and [BIG-Bench Hard](https://github.com/suzgunmirac/BIG-Bench-Hard).
