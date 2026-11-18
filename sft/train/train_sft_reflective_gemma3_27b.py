import os
import argparse

# 🌟 PREVENT FRAGMENTATION
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# --- 0. CLI ARGUMENTS ---
parser = argparse.ArgumentParser(description="Multi-GPU Targeted SFT with Manual Masking")
parser.add_argument("--gpu", type=str, required=True, help="GPU device ID(s) e.g. '0' or '0,1'")
parser.add_argument("--target_mode", type=str, choices=["all", "targeted"], default="targeted")
parser.add_argument("--layer_start", type=int, default=16)
parser.add_argument("--layer_end", type=int, default=32)
args = parser.parse_args()

# 🌟 SET ENVIRONMENT FOR MULTI-GPU BEFORE LOADING TORCH/MODEL
# This makes only the specified GPUs visible to the process.
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print(f"🚀 Initializing SFT on Visible GPUs: {args.gpu}")

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# --- 1. GLOBAL CONFIG ---
VERSION = "v2" 
NUM_ITER = 2
MODEL_PATH = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "models/gemma-3-27b-it")
# DATA_FILE = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"post_reflective_data/mistral_self_taught_sft_data_v{NUM_ITER}.json")
# DATA_FILE = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"post_reflective_data/gemma3_27b_mmlu_v{NUM_ITER}.json")
DATA_FILE = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"post_reflective_data/gemma3_27b_numina_v{NUM_ITER}.json")

SYSTEM_PROMPT = "You are a world-class expert in mathematics. State the final answer first, then explain how you got the answer."

if args.target_mode == "all":
    # OUTPUT_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"checkpoints/gemma3-27b-reflection-mmlu-all-{VERSION}")
    # wandb_name = f"gemma3-27b-reflection-mmlu-all-{VERSION}"
    OUTPUT_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"checkpoints/gemma3-27b-reflection-numina-all-{VERSION}")
    wandb_name = f"gemma3-27b-reflection-numina-all-{VERSION}"
else:
    # OUTPUT_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"checkpoints/gemma3-27b-reflection-mmlu-targeted-{args.layer_start}-{args.layer_end}-{VERSION}")
    # wandb_name = f"gemma3-27b-reflection-mmlu-targeted-{args.layer_start}-{args.layer_end}-{VERSION}"
    OUTPUT_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"checkpoints/gemma3-27b-reflection-numina-targeted-{args.layer_start}-{args.layer_end}-{VERSION}")
    wandb_name = f"gemma3-27b-reflection-numina-targeted-{args.layer_start}-{args.layer_end}-{VERSION}"

# --- 2. CUSTOM DATASET CLASS ---
class MaskedReasoningDataset(Dataset):
    def __init__(self, json_data, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.features = []
        
        print(f"Preprocessing {len(json_data)} examples with Expert Formatting and Strict Masking...")
        for item in json_data:
            messages = item["messages"]
            raw_user_content = messages[0]["content"]
            assistant_content = messages[1]["content"]

            user_content = f"Question: {raw_user_content}" if not raw_user_content.startswith("Question:") else raw_user_content

            chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
            prompt_text = tokenizer.apply_chat_template(chat_messages, tokenize=False, add_generation_prompt=True)
            full_text = prompt_text + assistant_content + tokenizer.eos_token

            tokenized = tokenizer(full_text, truncation=True, max_length=max_length, add_special_tokens=False, return_offsets_mapping=True)
            input_ids, attention_mask, offsets = tokenized["input_ids"], tokenized["attention_mask"], tokenized["offset_mapping"]
            labels = list(input_ids)

            char_start_idx = full_text.find("Explanation:", len(prompt_text))
            match_token_idx = None
            if char_start_idx != -1:
                for idx, (start, end) in enumerate(offsets):
                    if end > char_start_idx:
                        match_token_idx = idx
                        break
            
            if match_token_idx is not None:
                for i in range(match_token_idx): labels[i] = -100
            else:
                labels = [-100] * len(labels)
                
            # 🌟 ADD THIS: Create an array of 0s for pure text
            token_type_ids = [0] * len(input_ids)

            self.features.append({
                "input_ids": input_ids, 
                "attention_mask": attention_mask, 
                "labels": labels,
                "token_type_ids": token_type_ids
            })

    def __len__(self): return len(self.features)
    def __getitem__(self, i): return self.features[i]

# --- 3. INITIALIZATION ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token, tokenizer.padding_side = tokenizer.eos_token, "right"

raw_data = load_dataset("json", data_files=DATA_FILE, split="train")
split_data = raw_data.train_test_split(test_size=0.05)
train_dataset = MaskedReasoningDataset(split_data["train"], tokenizer)
eval_dataset = MaskedReasoningDataset(split_data["test"], tokenizer)

# --- 4. LORA & MODEL LOADING ---
if args.target_mode == "all":
    targeted_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
else:
    targeted_modules = [f"model.layers.{i}.{m}" for i in range(args.layer_start, args.layer_end + 1) 
                        for m in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]]

lora_config = LoraConfig(
    r=16, 
    lora_alpha=32, 
    target_modules=targeted_modules, 
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
    )

print(f"Loading Model with device_map='auto' (Distributing across GPUs: {args.gpu})...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, 
    dtype=torch.bfloat16, 
    device_map="auto",
    attn_implementation="flash_attention_2"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# --- 5. TRAINING ARGUMENTS ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=16,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=5,
    num_train_epochs=3,
    eval_strategy="steps",
    eval_steps=20,
    save_strategy="epoch",
    bf16=True,
    optim="adamw_torch",
    report_to="wandb",
    run_name=wandb_name,
    remove_unused_columns=False,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

trainer = Trainer(
    model=model, 
    args=training_args, 
    train_dataset=train_dataset, 
    eval_dataset=eval_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100, pad_to_multiple_of=8)
)

print("\nStarting Training...")
trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")