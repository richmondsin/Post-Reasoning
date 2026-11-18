import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "bbh_full_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100  # Process 100 inputs simultaneously

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8017)
parser.add_argument("--model_name", type=str, default="Mistral-Small-24B-Instruct-2501")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

# Initialize ASYNC vLLM Client
client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# Strategy Selector
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    "Basic_CoT",
    # "Post_Reason"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a Direct-Answer engine. Output ONLY the final answer. Do not provide any explanation or reasoning.",
        "gen_params": {
            "temperature": 0.7,
            "top_p": 0.8,
            "top_k": 20,
            "min_p": 0.0,
            "do_sample": True,
            "max_new_tokens": 4096
        }
    },
    "Basic_CoT": {
        "sys_msg": "You are a Chain-of-Thought engine. Think step-by-step to logically solve the problem. After explaining your full reasoning process, state the final integer answer.",
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
        "sys_msg": "You are a Post-Reasoning engine. State the final answer first, then explain your logic.",
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

# --- 2. UNIVERSAL BBH EXTRACTOR ---
def extract_answer(text, strat):
    if not isinstance(text, str): return "FAIL"
    
    clean = text.replace("**", "").strip()
    candidate = clean
    
    # Strategy 1: LaTeX Boxed
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(.*?)\s*\}', clean, re.IGNORECASE)
    if boxed_matches: 
        candidate = boxed_matches[0].strip() if strat in ["Post_Reason", "Direct_Answer"] else boxed_matches[-1].strip()
    else:
        # Strategy 2: "Answer:" variants
        pattern = r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$)"
        ans_matches = re.findall(pattern, clean, re.IGNORECASE | re.DOTALL)
        if ans_matches: 
            ans = ans_matches[0] if strat in ["Post_Reason", "Direct_Answer"] else ans_matches[-1]
            ans = re.sub(r"^(Answer|:)\s*", "", ans, flags=re.IGNORECASE).strip()
            candidate = ans.strip()
        else:
            # Strategy 3: Fallback line extraction
            if strat in ["Post_Reason", "Direct_Answer"]:
                candidate = clean.split('\n')[0].strip()
            else:
                candidate = clean.split('\n')[-1].strip()

    # --- THE FIXED MULTIPLE CHOICE SNATCHER ---
    mc_match = re.match(r'^\s*[\(\[]?([A-Z])[\]\)\.]*(?:\s|$)', candidate, re.IGNORECASE)
    if mc_match:
        return f"({mc_match.group(1).upper()})"

    return candidate

def normalize_string(s: str) -> str:
    """Universal normalizer that standardizes lists, options, and strings for exact matching."""
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    
    s = re.sub(r'^(answer|output):\s*', '', s).strip()
    s = s.lstrip(':').strip()
    
    # --- THE CRITICAL FIX: Strip trailing periods BEFORE checking for parentheses ---
    s = s.rstrip('.').strip()
    
    if s.startswith('(') and s.endswith(')'): s = s[1:-1]
    if s.startswith('[') and s.endswith(']'): s = s[1:-1]
    
    # Convert commas to spaces (e.g., "a, b, c" -> "a  b  c")
    s = s.replace(',', ' ')
    s = s.replace("'", "").replace('"', '')
    
    # Collapse multiple spaces into a single space
    s = re.sub(r'\s+', ' ', s)
    
    return s.strip()

def evaluate_correctness(pred, truth):
    pred_norm = normalize_string(pred)
    truth_norm = normalize_string(truth)
    
    if pred_norm == truth_norm: return 1
    try:
        if float(pred_norm) == float(truth_norm): return 1
    except ValueError:
        pass
    return 0

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    """Handles a single API request for BBH."""
    async with semaphore:
        # Load directly from the prepared JSONL attributes
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        task_name = item['task_name']
        
        extra_body = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        # Added "Q:" to stop tokens to prevent few-shot hallucination runaway
        if strat == "Direct_Answer":
            extra_body["stop"] = ["Explanation:", "Reasoning:", "Here's why", "Q:"]
        
        try:
            # Standard API Call (No prefill hacks)
            full_api_response = await client.chat.completions.create(
                model=args.model_name,
                messages=[
                    {"role": "system", "content": config["sys_msg"]},
                    {"role": "user", "content": full_user_prompt}
                ],
                temperature=config["gen_params"].get("temperature", 0.7),
                top_p=config["gen_params"].get("top_p", 0.8),
                max_tokens=config["gen_params"].get("max_new_tokens", 2048),
                extra_body=extra_body
            )

            # Prevent NoneType crash
            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Predict & Evaluate
            pred = extract_answer(response_text, strat)
            is_correct = evaluate_correctness(pred, truth)

            return {
                "task": task_name,
                "truth": truth,
                "pred": pred,
                "is_correct": is_correct,
                "response": response_text.replace("\n", " "),
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item in {task_name}: {e}")
            return None

# --- 4. MAIN ASYNC LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
        
        input_file = f"{DATA_DIR}/prepared_bbh_{strat}.jsonl"
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Run prep script first.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
                
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_bbh_all_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        # Pass strat down to worker
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        
        # Execute and gather responses
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        # Filter None values
        for res in completed_responses:
            if res is not None:
                results.append(res)

        # Save iterative JSONL using Pandas
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())