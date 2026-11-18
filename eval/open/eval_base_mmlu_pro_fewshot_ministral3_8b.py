import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "mmlu_pro_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100  # Process 100 questions simultaneously

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8015)
parser.add_argument("--model_name", type=str, default="Ministral-3-8B-Instruct-2512")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

# Initialize ASYNC vLLM Client
client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- STRATEGY SELECTOR ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    "Basic_CoT",
    # "Post_Reason"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are an expert academic AI. You must answer complex, graduate-level multiple-choice questions across diverse domains. Output ONLY the letter of the correct option (A through J). Do not provide any explanation, reasoning, or caveats.",
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
        "sys_msg": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. Think step-by-step to arrive at the correct answer. After your reasoning, you must state the final option letter (A through J).",
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
        "sys_msg": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. You must state the final option letter (A through J) first, and then provide a rigorous scientific or logical justification for your choice.",
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

# --- 2. MMLU-PRO EXTRACTOR ---
def extract_answer(text, strat):
    if not isinstance(text, str): return "FAIL"
    
    # Scrub native <think> tags for future-proofing
    clean = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    clean = clean.replace("**", "").strip()
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

    # --- MMLU-Pro Option Snatcher (A-J) ---
    mc_match = re.match(r'^\s*\(?([A-J])(?:[)\.]\s|[)\.]$|$)', candidate, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).upper()

    # Absolute fallback: Find the first/last A-J letter standing alone
    fallback_matches = re.findall(r'\b([A-J])\b', candidate, re.IGNORECASE)
    if fallback_matches:
        return fallback_matches[0].upper() if strat in ["Post_Reason", "Direct_Answer"] else fallback_matches[-1].upper()

    return candidate.strip()

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    """Handles a single API request for MMLU-Pro."""
    async with semaphore:
        # Pull the fully baked prompt from the JSONL
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip().upper()
        category = item.get('category', 'unknown')
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        # Apply strict stop tokens to prevent few-shot hallucination
        if strat == "Direct_Answer":
            extra_body_params["stop"] = ["\n", "Explanation:", "Reasoning:", "Here's why", "Q:"]
        else:
            extra_body_params["stop"] = ["Q:", "Q :", "\nQ", "\nQ:"]

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
            pred = extract_answer(response_text, strat)
            is_correct = 1 if pred == truth else 0
            
            # Scrub <think> tags for clean logging
            clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
            clean_log_text = clean_log_text.replace("</think>", "").strip()

            return {
                "category": category, # Save category (math, health, etc.) for analysis
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": clean_log_text.replace("\n", " "), 
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item in {category}: {e}")
            return None

# --- 4. MAIN ASYNC LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
            
        input_file = f"{DATA_DIR}/prepared_mmlu_pro_fewshot_{strat}.jsonl"
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Please run the preparation script.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
        
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_mmlu_pro_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
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