import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gpqa_main_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8016)
parser.add_argument("--model_name", type=str, default="Ministral-3-14B-Instruct-2512")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- STRATEGY SELECTOR ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    "Basic_CoT",
    # "Post_Reason"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

# --- EXPERIMENT TEMPLATES ---
TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
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
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). Think step-by-step through the scientific principles required to solve the problem, then state your final answer clearly.",
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
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then explain your scientific reasoning.",
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

# --- 2. UNIVERSAL GPQA EXTRACTOR ---
def extract_answer_letter(text):
    if not isinstance(text, str): return "FAIL"
    clean = text.strip()
    
    def normalize(val):
        # 1. Strip common LaTeX/Text debris from inside the match
        val = re.sub(r'\\text\s*\{|\}|\$|\)|[:\.]|\*', '', val, flags=re.IGNORECASE)
        val = val.strip().upper()
        # 2. Map numbers to letters (GPQA has 4 options)
        d_map = {"1":"A", "2":"B", "3":"C", "4":"D"}
        return d_map.get(val, val)

    # --- STRATEGY 0: <answer> TAGS (Highest Priority for your SFT) ---
    tag_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', clean, re.IGNORECASE | re.DOTALL)
    if tag_matches:
        extracted = normalize(tag_matches[-1])
        if re.match(r'^[A-D]$', extracted): return extracted

    # --- STRATEGY 1: COMPLEX LaTeX BOXED ---
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(?:\\text\s*\{)?\s*([A-D]|[1-4])\s*\}?', clean, re.IGNORECASE)
    if boxed_matches: 
        return normalize(boxed_matches[-1])

    # --- STRATEGY 2: TAG FALLBACK (Unclosed) ---
    unclosed_matches = re.findall(r'<answer>\s*([A-D]|[1-4])', clean, re.IGNORECASE)
    if unclosed_matches:
        return normalize(unclosed_matches[-1])

    # --- STRATEGY 3: Standard Fallback (Handles Answer: **A**) ---
    ans_matches = re.findall(r'Answer:\s*.*?([A-D]|[1-4])(?![a-z])', clean, re.IGNORECASE | re.DOTALL)
    if ans_matches: 
        return normalize(ans_matches[-1])
    
    # --- STRATEGY 4: Start of line "A." or "A)" ---
    match = re.match(r'^([A-D]|[1-4])\s*[:\).]', clean, re.IGNORECASE)
    if match: return normalize(match.group(1))
    
    # --- STRATEGY 5: Standalone Token ---
    norm_text = re.sub(r'<.*?>|[\(\)\.\$\s\*]', '', clean).upper()
    if re.match(r'^[A-D]$', norm_text):
        return norm_text
    
    return "FAIL"

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        # Load directly from the prepared JSONL attributes
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        # Physical stop tokens to prevent hallucinating extra few-shot questions
        if strat == "Direct_Answer":
            extra_body_params["stop"] = ["Explanation:", "Reasoning:", "Here's why", "Q:"]
        else:
            extra_body_params["stop"] = ["Q:"]

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
            
            pred = extract_answer_letter(response_text)
            
            return {
                "truth": truth, 
                "pred": pred, 
                "is_correct": 1 if pred == truth else 0,
                "response": response_text.replace("\n", " "),
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item: {e}")
            return None
        
# --- 4. THE MAIN ASYNC LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
        
        input_file = f"{DATA_DIR}/prepared_gpqa_fewshot_{strat}.jsonl"
        
        # Check if the prepared data for this strategy actually exists
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Please run the GPQA prep script for this strategy.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
                
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_gpqa_main_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        results = []
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        for res in completed_responses:
            if res is not None: results.append(res)
        
        # Save results to JSONL via Pandas
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

if __name__ == "__main__": 
    asyncio.run(run_eval())