import json
import os
import argparse
import asyncio
import random
import re
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
CONCURRENT_REQUESTS = 100 
NUM_ITER = 2
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8010)
parser.add_argument("--model_name", type=str, default="gemma-3-12b-it")
parser.add_argument("--output_file", type=str, default=os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), f"post_reflective_data/gemma3_12b_numina_v{NUM_ITER}.json"))
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- 2. LOCAL DATASET LOADER ---
def get_local_dataset():
    file_path = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "post_reflective_data/numina_3500_integer_master.jsonl")
    print(f"Loading local master dataset from {file_path}...")
    
    dataset = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                dataset.append(json.loads(line.strip()))
        print(f"Targeting clean subset: {len(dataset)} questions")
        return dataset
    except Exception as e:
        print(f"Failed to load local dataset: {e}")
        return []

# --- 3. HIGH-QUALITY PROMPT ENGINEERING (BLIND DERIVATION) ---
def create_pure_justification_prompt(question, truth_value):
    sys_msg = "You are a world-class mathematician writing a formal proof."
    
    user_msg = f"""Question:
{question}

(Secret Target: The formal derivation MUST organically conclude that {truth_value} is correct.)

Task: Write a flawless, rigorous, step-by-step mathematical derivation solving this problem.

CRITICAL CONSTRAINTS:
1. Tone: Formal, academic, and highly rigorous.
2. NO ORACLE LEAKS: Never mention "The answer is given as" or "We know the answer is". Act as if you are proving it from scratch.
3. NO CONVERSATIONAL FILLER: Do not use phrases like "Let's think step by step", "I initially thought", or "Let's break this down."
4. NO PREFIXES: Start your very first word with the mathematical facts, equations, or direct logical deductions. Do not write "Explanation:" or "Derivation:" or "Step 1:" at the very beginning."""

    return sys_msg, user_msg

# --- 4. THE ASYNC WORKER ---
def quality_check(explanation):
    lower_exp = explanation.lower()
    
    bad_phrases = [
        "is given as", "provided answer", "correct answer is given", 
        "secret target", "let's think", "i initially thought",
        "let's break this down", "as stated in the prompt"
    ]
    if any(phrase in lower_exp for phrase in bad_phrases):
        return False, "Contains forbidden phrases or oracle leaks."
        
    if len(explanation.split()) < 20:
        return False, "Explanation too short (lacks rigor)."
        
    return True, "Pass"

async def generate_item(item, index, semaphore):
    async with semaphore:
        # Directly extract the pre-cleaned data from our JSONL file
        q_text = item.get("question", "").strip()
        truth_value = str(item.get("truth_value", "")).strip()
        
        # We NO LONGER extract or pass the messy_solution. The model is blind!
        if not q_text or not truth_value:
            return None
            
        sys_msg, user_msg = create_pure_justification_prompt(q_text, truth_value)
        
        for attempt in range(5):
            try:
                response = await client.chat.completions.create(
                    model=args.model_name,
                    messages=[
                        {"role": "system", "content": sys_msg},
                        {"role": "user", "content": user_msg}
                    ],
                    temperature=0.3,
                    max_tokens=4096
                )

                # 🌟 THE LENGTH LIMIT CHECK 🌟
                if response.choices[0].finish_reason == "length":
                    # Print it immediately so you can monitor it in real-time
                    print(f"\n[Item {index} - Attempt {attempt+1}] Model hit max_tokens (4096) and truncated!")
                    raise ValueError("Generation truncated! Model hit max_tokens mid-sentence.")

                explanation = response.choices[0].message.content.strip()
                
                if explanation:                  
                    # Clean out any prefix the model might try to add
                    explanation = re.sub(r"^(explanation|derivation|answer|solution)[:\-\s]*", "", explanation, flags=re.IGNORECASE).strip()
                    
                    if not explanation:
                        raise ValueError("Output was empty after cleaning prefixes.")

                    passed, reason = quality_check(explanation)
                    if not passed:
                        raise ValueError(f"Quality Check Failed: {reason}")

                    # Format the final output to match your SFT requirements exactly
                    sft_user_prompt = (
                        f"Question: {q_text}\n\n"
                        "Think hard without outputting any explanation. State the final answer first, then explain how you got it. "
                        "Output format: 'Answer: [Value]. Explanation: [reasoning]'."
                    )
                    sft_assistant_response = f"Answer: {truth_value}. Explanation: {explanation}"
                    
                    return {
                        "question": sft_user_prompt,
                        "true_answer": truth_value,
                        "messages": [
                            {"role": "user", "content": sft_user_prompt},
                            {"role": "assistant", "content": sft_assistant_response}
                        ]
                    }
                    
            except Exception as e:
                if attempt == 4:
                    print(f"\n[Fail after 5 tries] {e}")
                await asyncio.sleep(1)
        return None

# --- 5. MAIN ASYNC LOOP ---
async def run_generation():
    ds = get_local_dataset()
    if not ds: return
    
    print(f"Generating Blind Derivations with {args.model_name}...")
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)
    
    tasks = [generate_item(item, i, semaphore) for i, item in enumerate(ds)]
    
    success_count = 0
    final_dataset = []
    
    for future in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Generating"):
        result = await future
        if result is not None:
            success_count += 1
            final_dataset.append(result)
            
    sft_export = [{"messages": item["messages"]} for item in final_dataset]
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(sft_export, f, indent=2, ensure_ascii=False)
        
    jsonl_output_file = args.output_file.replace(".json", ".jsonl")
    with open(jsonl_output_file, 'w', encoding='utf-8') as f:
        for entry in final_dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print("\n" + "="*60)
    print("=== BLIND DERIVATION GENERATION COMPLETE ===")
    print(f"Successfully generated {success_count} / {len(tasks)} pure derivations.")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(run_generation())
