import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "math_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 2
CONCURRENT_REQUESTS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8012)
parser.add_argument("--model_name", type=str, default="Qwen3-8B")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- 2. STRATEGY SELECTOR & TEMPLATES ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    "Thinking",
    # "Basic_CoT",
    # "Post_Reason",
    "Thinking_Post"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a direct math expert. You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Basic_CoT": {
        "sys_msg": "You are a chain-of-thought math expert. Think step-by-step to solve the problem, clearly laying out your mathematical reasoning. Finally, clearly state the final answer.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Thinking": {
        "sys_msg": "You are a direct math expert. You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
        "enable_thinking": True,
        "gen_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 16384
        }
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final result first, then justify.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Thinking_Post": {
        "sys_msg": "You are a post-reasoning math expert. State the final result first, then justify.",
        "enable_thinking": True,
        "gen_params": {
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 16384
        }
    }
}

# --- 3. MATH CLEANING & EXTRACTION LOGIC ---
def clean_math_string(s):
    """Strips $, \boxed{}, \text{}, prefixes, and periods to get the raw string."""
    if not isinstance(s, str): return "FAIL"
    
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', s)
    if box_match: s = box_match.group(1)
        
    prefixes = [r"The final answer is", r"The answer is", r"My answer is", r"Answer:?", r"Answer"]
    pattern = r'^(?:' + '|'.join(prefixes) + r')\s*:?\s*'
    s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    
    s = s.replace('$', '').replace('**', '').replace(',', '').strip()
    s = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', s)
    s = s.rstrip('.').rstrip(';').strip()
    return s

def extract_math_answer(text, strategy):
    if not isinstance(text, str): return "FAIL"
    
    # --- SCRUB <THINK> TAGS FOR EXTRACTION ---
    if strategy in ["Thinking", "Thinking_Post"]:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
        
    clean = clean.replace("</think>", "").replace("<think>", "").strip()
    
    triggers = ["The final answer is:", "The final answer is", "The answer is:", "The answer is", "My answer is:", "My answer is", "Answer:", "Answer"]
    candidate = clean

    first_answer_strats = ["Post_Reason", "Direct_Answer", "Thinking", "Thinking_Post"]

    if strategy in first_answer_strats:
        for trigger in triggers:
            idx = clean.lower().find(trigger.lower())
            if idx != -1 and idx < 150: 
                parts = re.split(re.escape(trigger), clean, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    break

    stop_phrases = ["Explanation:", "Explanation", "Here's", "Here is", "Let's", "Note:", "The ", "To ", "Question:"]
    for phrase in stop_phrases:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1:
            candidate = candidate[:idx]
            
    return clean_math_string(candidate)

# --- 4. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, strat, config, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        subset = item.get('subset', 'unknown')

        # --- NEW: Prepend "/think" to the user prompt for Thinking strategies ---
        if strat in ["Thinking", "Thinking_Post"]:
            full_user_prompt = f"/think\n{full_user_prompt}"
        
        # Build extra_body with Qwen3 specific vLLM kwargs
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0),
            "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}
        }

        try:
            full_api_response = await client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": config["sys_msg"]},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=config["gen_params"].get("temperature", 0.1),
                top_p=config["gen_params"].get("top_p", 0.8),
                max_tokens=config["gen_params"].get("max_new_tokens", 2048),
                extra_body=extra_body_params
            )

            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Predict & Score
            pred = extract_math_answer(response_text, strat)
            
            # Check accuracy (Exact match or Float equivalence)
            is_correct = 0
            if str(pred).strip().lower() == truth.lower():
                is_correct = 1
            else:
                try:
                    if float(pred) == float(truth):
                        is_correct = 1
                except:
                    pass

            # --- SCRUB <THINK> FOR LOGGING SAFELY ---
            if strat in ["Thinking", "Thinking_Post"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                
            clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()

            return {
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": clean_log_text.replace("\n", " "), 
                "raw_response": raw_response_json,
                "subset": subset 
            }
        except Exception as e:
            print(f"Error on item: {e}")
            return None

# --- 5. THE MAIN ASYNC LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
            
        # --- DATA ROUTER FOR THINKING STRATEGIES ---
        data_strat = strat
        if strat == "Thinking":
            data_strat = "Direct_Answer"
        elif strat == "Thinking_Post":
            data_strat = "Post_Reason"
            
        input_file = f"{DATA_DIR}/prepared_math_fewshot_{data_strat}.jsonl"
        
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
        
        config = TEMPLATES[strat]
        output_path = f"{RESULTS_DIR}/base_{model_label}_{strat}_math_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (Thinking Block: {config['enable_thinking']}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        tasks = [process_single_item(item, strat, config, semaphore) for item in prepared_data]
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        for res in completed_responses:
            if res is not None:
                results.append(res)
        
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_path, orient='records', lines=True)
            acc = df['is_correct'].mean() * 100
            print(f"🔥 MATH [{strat}] Accuracy: {acc:.2f}% | Saved {len(results)} results to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_eval())