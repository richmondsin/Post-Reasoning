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
    "Direct_Answer", 
    # "Thinking", 
    # "Basic_CoT",
    # "Post_Reason",
    # "Thinking_Post"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are an expert academic AI. You must answer complex, graduate-level multiple-choice questions across diverse domains. Output ONLY the letter of the correct option (A through J). Do not provide any explanation, reasoning, or caveats.",
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
        "sys_msg": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. Think step-by-step to arrive at the correct answer. After your reasoning, you must state the final option letter (A through J).",
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
    "Thinking": {
        "sys_msg": "You are an expert academic AI. You must answer complex, graduate-level multiple-choice questions across diverse domains. Output ONLY the letter of the correct option (A through J). Do not provide any explanation, reasoning, or caveats.",
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
        "sys_msg": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. You must state the final option letter (A through J) first, and then provide a rigorous scientific or logical justification for your choice.",
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
        "sys_msg": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. You must state the final option letter (A through J) first, and then provide a rigorous scientific or logical justification for your choice.",
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

# --- 2. MMLU-PRO EXTRACTOR ---
def extract_answer(text, strat, enable_thinking):
    if not isinstance(text, str): return "FAIL"
    
    clean = text
    
    # NEW: Safely strip <think> tags completely before extraction
    if enable_thinking:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', text, flags=re.IGNORECASE | re.DOTALL).strip()
        clean = clean.replace("</think>", "").replace("<think>", "").strip()
    
    # Clean out bolding
    clean = clean.replace("**", "")
    candidate = clean
    
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

    # --- MMLU-Pro Option Snatcher (A-J) ---
    mc_match = re.match(r'^\s*\(?([A-J])(?:[)\.]\s|[)\.]$|$)', candidate, re.IGNORECASE)
    if mc_match:
        return mc_match.group(1).upper()

    # Absolute fallback: Find the first/last A-J letter standing alone
    fallback_matches = re.findall(r'\b([A-J])\b', candidate, re.IGNORECASE)
    if fallback_matches:
        return fallback_matches[0].upper() if strat in first_answer_strats else fallback_matches[-1].upper()

    return candidate.strip()

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        # Pull the fully baked prompt from the JSONL
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip().upper()
        category = item.get('category', 'unknown')
        
        # Build extra_body with Qwen 3.5 specific vLLM kwargs
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "repetition_penalty": config["gen_params"].get("repetition_penalty", 1.0),
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
                presence_penalty=config["gen_params"].get("presence_penalty", 1.5),
                max_tokens=config["gen_params"].get("max_new_tokens", 4096), 
                extra_body=extra_body_params
            )

            # Prevent NoneType crash
            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Predict & Score using strategy-aware extractor
            pred = extract_answer(response_text, strat, config["enable_thinking"])
            is_correct = 1 if pred == truth else 0
            
            # Safely scrub <think> tags for clean logging
            if config["enable_thinking"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()
            
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
    # Replace slash in Qwen/Qwen3.5-4B so it doesn't break directory paths
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
            
        input_file = f"{DATA_DIR}/prepared_mmlu_pro_fewshot_{data_strat}.jsonl"
        
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Please run the preparation script.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
        
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_mmlu_pro_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (Thinking: {config['enable_thinking']}) via Port {args.port} ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        # Schedule the items for this strategy
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        for res in completed_responses:
            if res is not None: results.append(res)

        # Save to JSONL using Pandas
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())