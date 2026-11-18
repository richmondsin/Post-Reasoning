import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "math_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100 

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8017) # Adjust port as needed
parser.add_argument("--model_name", type=str, default="Mistral-Small-24B-Instruct-2501")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- 2. STRATEGY SELECTOR & TEMPLATES ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    # "Basic_CoT",
    # "Post_Reason",
    "Post_Summary",
    "Post_Confidence"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a direct math expert. You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
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
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final result first, then justify.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Post_Summary": {
        "sys_msg": "You are a post-reasoning math expert. State the final result first, then briefly summarize the problem and your answer in a single sentence.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Post_Confidence": {
        "sys_msg": "You are a post-reasoning math expert. State the final result first, then state your confidence level (0-100%) in this answer and briefly explain why.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    }
}

# --- 3. MATH CLEANING & EXTRACTION LOGIC ---
def clean_math_string(s):
    """Strips $, \boxed{}, \text{}, prefixes, and periods to get the raw string."""
    if not isinstance(s, str): return "FAIL"
    
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', s)
    if box_match:
        s = box_match.group(1)
        
    prefixes = [r"The final answer is", r"The answer is", r"My answer is", r"Answer:?", r"Answer"]
    pattern = r'^(?:' + '|'.join(prefixes) + r')\s*:?\s*'
    s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    
    s = s.replace('$', '').replace('**', '').replace(',', '').strip()
    s = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', s)
    s = s.rstrip('.').rstrip(';').strip()
    
    return s

def extract_math_answer(text, strategy):
    if not isinstance(text, str): return "FAIL"
    
    clean = text.strip()
    triggers = ["The final answer is:", "The final answer is", "The answer is:", "The answer is", "My answer is:", "My answer is", "Answer:", "Answer"]
    candidate = clean

    if strategy in ["Post_Reason", "Direct_Answer", "Post_Summary", "Post_Confidence"]:
        for trigger in triggers:
            idx = clean.lower().find(trigger.lower())
            if idx != -1 and idx < 150: 
                parts = re.split(re.escape(trigger), clean, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    break

    stop_phrases = ["Explanation:", "Explanation", "Here's", "Here is", "Let's", "Note:", "The ", "To ", "Question:", "Summary:", "Confidence:"]
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
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        # Prevent runaway hallucination on few-shot prompts
        if strat == "Direct_Answer":
            extra_body_params["stop"] = ["\n", "Explanation:", "Q:"]
        else:
            extra_body_params["stop"] = ["Q:"]

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
            
            # Float-tolerant integer accuracy (e.g., 16.0 == 16)
            is_correct = 0
            if str(pred).strip().lower() == truth.lower():
                is_correct = 1
            else:
                try:
                    if float(pred) == float(truth):
                        is_correct = 1
                except:
                    pass

            return {
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": response_text.replace("\n", " "), 
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
            
        input_file = f"{DATA_DIR}/prepared_math_fewshot_{strat}.jsonl"
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
        
        config = TEMPLATES[strat]
        output_path = f"{RESULTS_DIR}/base_{model_label}_{strat}_math_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
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
            print(f"Saved {len(results)} results to {output_path}")

if __name__ == "__main__":
    asyncio.run(run_eval())