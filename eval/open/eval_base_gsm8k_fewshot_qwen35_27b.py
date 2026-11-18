import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gsm8k_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 2
CONCURRENT_REQUESTS = 100 

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8021)
parser.add_argument("--model_name", type=str, default="Qwen35-27B")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- 2. STRATEGY SELECTOR & TEMPLATES ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer", 
    # "Basic_CoT",
    # "Post_Reason",
    "Thinking",
    "Thinking_Post"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

# Strict Qwen3.5 Templates with custom penalties
TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a direct math expert. Output ONLY the final numeric answer. Do not provide any reasoning or explanation.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Basic_CoT": {
        "sys_msg": "You are a chain-of-thought math expert. Think step-by-step to logically solve the problem. After detailing your reasoning, explicitly state the final numeric answer.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 2048
        }
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final numeric answer first, then explain your reasoning.",
        "enable_thinking": False,
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    },
    "Thinking": {
        "sys_msg": "You are a direct math expert. Output ONLY the final numeric answer. Do not provide any reasoning or explanation.",
        "enable_thinking": True,
        "gen_params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 16384
        }
    },
    "Thinking_Post": {
        "sys_msg": "You are a post-reasoning math expert. State the final numeric answer first, then explain your reasoning.",
        "enable_thinking": True,
        "gen_params": {
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0.0,
            "presence_penalty": 1.5,
            "repetition_penalty": 1.0,
            "do_sample": True,
            "max_new_tokens": 16384
        }
    }
}

# --- 3. NUMERIC EXTRACTOR ---
def extract_gsm8k_number(text, strat, enable_thinking):
    """Aggressively extracts numeric answers, cleans formatting, and safely scrubs think tags."""
    if not isinstance(text, str): return "FAIL"
    
    clean = text
    
    # NEW: Safely strip <think> tags completely before extraction
    if enable_thinking:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
        clean = clean.replace("</think>", "").replace("<think>", "").strip()
    
    candidate = clean

    # Look for "Answer:" triggers
    triggers = ["The final answer is:", "The answer is:", "Answer:"]
    first_answer_strats = ["Post_Reason", "Direct_Answer", "Thinking", "Thinking_Post"]
    
    if strat in first_answer_strats:
        for trigger in triggers:
            idx = candidate.lower().find(trigger.lower())
            if idx != -1 and idx < 150: 
                parts = re.split(re.escape(trigger), candidate, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    break

    # Truncate at explanation keywords
    stop_phrases = ["Explanation:", "Here's", "Let's", "Note:"]
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
        # For Post Reason, Direct, and Thinking, the answer is stated FIRST. 
        return nums[0] if strat in first_answer_strats else nums[-1]

    return "FAIL"

# --- 4. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        
        # Build extra_body with strict Qwen3.5 params
        extra_body_params = {
            "top_k": config["gen_params"]["top_k"],
            "min_p": config["gen_params"]["min_p"],
            "repetition_penalty": config["gen_params"]["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": config["enable_thinking"]}
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
                presence_penalty=config["gen_params"]["presence_penalty"],
                max_tokens=config["gen_params"]["max_new_tokens"],
                extra_body=extra_body_params
            )

            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Predict & Score
            pred = extract_gsm8k_number(response_text, strat, config["enable_thinking"])
            
            is_correct = 0
            if pred != "FAIL":
                try:
                    if float(pred) == float(truth):
                        is_correct = 1
                except ValueError:
                    pass
            
            # Safely scrub <think> tags for clean logging
            if config["enable_thinking"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()

            return {
                "subset": item.get('subset', 'gsm8k_main'), 
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": clean_log_text.replace("\n", " "), 
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
            
        # NEW: Data routing mapping
        data_strat = strat
        if strat == "Thinking":
            data_strat = "Direct_Answer"
        elif strat == "Thinking_Post":
            data_strat = "Post_Reason"
            
        input_file = f"{DATA_DIR}/prepared_gsm8k_fewshot_{data_strat}.jsonl"
        
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