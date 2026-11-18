import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from datasets import load_dataset
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "bbh_full_data")
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

# --- STRATEGY SELECTOR ---
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
        "sys_msg": "You are a Direct-Answer engine. Output ONLY the final answer. Do not provide any explanation or reasoning.",
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
        "sys_msg": "You are a Chain-of-Thought engine. Think step-by-step to logically solve the problem. After explaining your full reasoning process, state the final integer answer.",
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
        "sys_msg": "You are a Direct-Answer engine. Output ONLY the final answer. Do not provide any explanation or reasoning.",
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
    "Post_Reason": {
        "sys_msg": "You are a Post-Reasoning engine. State the final answer first, then explain your logic.",
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
    "Thinking_Post": {
        "sys_msg": "You are a Post-Reasoning engine. State the final answer first, then explain your logic.",
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

# --- 2. UNIVERSAL BBH EXTRACTOR ---
def extract_answer(text, strat):
    if not isinstance(text, str): return "FAIL"
    
    clean = text
    
    # NEW: Strip <think> tags completely before extraction for Thinking strategies
    if strat in ["Thinking", "Thinking_Post"]:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', clean, flags=re.IGNORECASE | re.DOTALL)
        
    clean = clean.replace("**", "").strip()
    candidate = clean
    
    # Helper to group strategies that put the answer at the beginning
    first_answer_strats = ["Post_Reason", "Direct_Answer", "Thinking", "Thinking_Post"]
    
    # Strategy 1: LaTeX Boxed
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(.*?)\s*\}', clean, re.IGNORECASE)
    if boxed_matches: 
        candidate = boxed_matches[0].strip() if strat in first_answer_strats else boxed_matches[-1].strip()
    else:
        # Strategy 2: "Answer:" variants
        pattern = r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$)"
        ans_matches = re.findall(pattern, clean, re.IGNORECASE | re.DOTALL)
        if ans_matches: 
            ans = ans_matches[0] if strat in first_answer_strats else ans_matches[-1]
            ans = re.sub(r"^(Answer|:)\s*", "", ans, flags=re.IGNORECASE).strip()
            candidate = ans.strip()
        else:
            # Strategy 3: Fallback line extraction
            if strat in first_answer_strats:
                candidate = clean.split('\n')[0].strip()
            else:
                candidate = clean.split('\n')[-1].strip()

    # --- BUG FIX: Multiple Choice Option Snatcher ---
    # Accurately snatches "(E), "E), "E.", "(E)." or standalone "E"
    mc_match = re.match(r'^\s*[\(\[]?([A-Z])[\]\)\.]*(?:\s|$)', candidate, re.IGNORECASE)
    if mc_match:
        return f"({mc_match.group(1).upper()})"

    return candidate

def normalize_string(s: str) -> str:
    """Universal normalizer that standardizes lists, options, and strings for exact matching."""
    if not isinstance(s, str): return ""
    s = s.lower().strip()
    
    # Strip common prefixes
    s = re.sub(r'^(answer|output):\s*', '', s).strip()
    
    # --- NEW: Strip stray leading colons left by double "Answer: Answer:" ---
    s = s.lstrip(':').strip()
    
    # --- BUG FIX: Strip trailing periods BEFORE checking parentheses ---
    s = s.replace("'", "").replace('"', '').rstrip('.').strip()
    
    if s.startswith('(') and s.endswith(')'): s = s[1:-1]
    if s.startswith('[') and s.endswith(']'): s = s[1:-1]
    
    # List Flattener for word_sorting
    s = s.replace(',', ' ')
    s = s.replace("'", "").replace('"', '').rstrip('.')
    
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
    async with semaphore:
        # Load directly from the prepared JSONL attributes
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        task_name = item['task_name']
        
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
                temperature=config["gen_params"].get("temperature", 0.7),
                top_p=config["gen_params"].get("top_p", 0.8),
                max_tokens=config["gen_params"].get("max_new_tokens", 2048),
                extra_body=extra_body_params
            )

            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            pred = extract_answer(response_text, strat)
            is_correct = evaluate_correctness(pred, truth)
            
            # NEW: Clean log text safely handles unclosed tags for Thinking strats
            if strat in ["Thinking", "Thinking_Post"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()
            
            return {
                "task": task_name,
                "truth": truth,
                "pred": pred,
                "is_correct": is_correct,
                "response": clean_log_text.replace("\n", " "),
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
        
        # NEW: Data routing mapping
        data_strat = strat
        if strat == "Thinking":
            data_strat = "Direct_Answer"
        elif strat == "Thinking_Post":
            data_strat = "Post_Reason"
            
        input_file = f"{DATA_DIR}/prepared_bbh_{data_strat}.jsonl"
        
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
                
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_bbh_all_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (Thinking: {config['enable_thinking']}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        for res in completed_responses:
            if res is not None: results.append(res)

        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())