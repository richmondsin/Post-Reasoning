import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gsm8k_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100 

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8017)
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
        "sys_msg": "You are a direct math expert. Output ONLY the final numeric answer. Do not provide any reasoning or explanation.",
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
        "sys_msg": "You are a chain-of-thought math expert. Think step-by-step to logically solve the problem. After detailing your reasoning, explicitly state the final numeric answer.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final numeric answer first, then explain your reasoning.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    },
    "Post_Summary": {
        "sys_msg": "You are a post-reasoning math expert. State the final numeric answer first, then briefly summarize the problem and your answer in a single sentence.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    },
    "Post_Confidence": {
        "sys_msg": "You are a post-reasoning math expert. State the final numeric answer first, then state your confidence level (0-100%) in this answer and briefly explain why.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    }
}

# --- 3. NUMERIC EXTRACTOR ---
def extract_gsm8k_number(text, strat):
    """Aggressively extracts numeric answers and cleans formatting."""
    if not isinstance(text, str): return "FAIL"
    
    candidate = text.strip()

    # Look for "Answer:" triggers
    triggers = ["The final answer is:", "The answer is:", "Answer:"]
    if strat in ["Post_Reason", "Direct_Answer", "Post_Summary", "Post_Confidence"]:
        for trigger in triggers:
            idx = candidate.lower().find(trigger.lower())
            if idx != -1 and idx < 150: 
                parts = re.split(re.escape(trigger), candidate, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    break

    # Truncate at explanation keywords
    stop_phrases = ["Explanation:", "Here's", "Let's", "Note:", "Summary:", "Confidence:"]
    for phrase in stop_phrases:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1:
            candidate = candidate[:idx]
            
    # Try to find a LaTeX boxed answer first
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', candidate)
    if box_match:
        candidate = box_match.group(1)

    # Clean punctuation and currency
    candidate = candidate.replace(',', '').replace('$', '').strip()
    
    # Extract the first/last number depending on strategy
    nums = re.findall(r'-?\d+\.?\d*', candidate)
    if nums:
        # For Post Reason, the answer is stated FIRST. 
        return nums[0] if strat in ["Post_Reason", "Direct_Answer", "Post_Summary", "Post_Confidence"] else nums[-1]

    return "FAIL"

# --- 4. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        try:
            full_api_response = await client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": config["sys_msg"]},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=config["gen_params"]["temperature"],
                top_p=config["gen_params"]["top_p"],
                max_tokens=config["gen_params"]["max_new_tokens"],
                extra_body=extra_body_params
            )

            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Predict & Score
            pred = extract_gsm8k_number(response_text, strat)
            
            is_correct = 0
            if pred != "FAIL":
                try:
                    if float(pred) == float(truth):
                        is_correct = 1
                except ValueError:
                    pass
            
            return {
                "subset": item.get('subset', 'gsm8k_main'), 
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": response_text.replace("\n", " "), 
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item: {e}")
            return None

# --- 5. MAIN ASYNC LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
            
        input_file = f"{DATA_DIR}/prepared_gsm8k_fewshot_{strat}.jsonl"
        
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Did you run the prep script?")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
        
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_gsm8k_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (Model: {args.model_name} | Port: {args.port}) ---")
        print(f"Loaded {len(prepared_data)} GSM8K few-shot prompts.")
        
        # Queue background tasks
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        
        # Execute concurrently
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        # Filter None values
        for res in completed_responses:
            if res is not None:
                results.append(res)
        
        # Save to JSONL using pandas
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())