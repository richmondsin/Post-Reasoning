import os
import json, random, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gpqa_main_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100

# Argument parser to receive the port from your .sh script
parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8018)
parser.add_argument("--model_name", type=str, default="Llama-31-8B-Instruct")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
args = parser.parse_args()

# Initialize ASYNC vLLM Client
client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- NEW: Strategy Selector ---
ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    # "Basic_CoT",
    # "Post_Reason",
    "Post_Summary",
    "Post_Confidence"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

# --- 1. NEW EXPERIMENT TEMPLATES ---
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
            "max_new_tokens": 4096
        }
    },
    "Post_Summary": {
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then briefly summarize the problem and your answer in a single sentence.",
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
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then state your confidence level (0-100%) in this answer and briefly explain why.",
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

def extract_answer_letter(text):
    if not isinstance(text, str): return "FAIL"
    clean = text.strip()
    
    def normalize(val):
        # 1. Strip common LaTeX/Text debris from inside the match
        val = re.sub(r'\\text\s*\{|\}|\$|\)|[:\.]', '', val, flags=re.IGNORECASE)
        val = val.strip().upper()
        
        # 2. Map numbers to letters
        d_map = {"1":"A", "2":"B", "3":"C", "4":"D", "5":"E", "6":"F", "7":"G", "8":"H", "9":"I", "10":"J"}
        return d_map.get(val, val)

    # --- STRATEGY 0: <answer> TAGS (Highest Priority for your new SFT) ---
    tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>', clean, re.IGNORECASE | re.DOTALL)
    if tag_match:
        extracted = normalize(tag_match.group(1))
        # Ensure the result is a single valid letter A-J
        if re.match(r'^[A-J]$', extracted):
            return extracted

    # --- STRATEGY 1: COMPLEX LaTeX BOXED (Fixes your examples) ---
    # Captures \boxed{\text{C}}, \boxed{D)}, \boxed{A} etc.
    # The inner group (.*?) is lazy to avoid over-capturing
    match = re.search(r'\\boxed\s*\{\s*(?:\\text\s*\{)?\s*([A-J]|[1-9]|10(?:\))?)\s*\}?', clean, re.IGNORECASE)
    if match: 
        return normalize(match.group(1))

    # --- STRATEGY 2: TAG FALLBACK (Unclosed) ---
    match = re.search(r'<answer>\s*([A-J]|[1-9]|10)', clean, re.IGNORECASE)
    if match:
        return normalize(match.group(1))

    # --- STRATEGY 3: Standard Fallback (Answer: A) ---
    match = re.search(r'Answer:\s*\(?(?:Option\s+)?([A-J]|[1-9]|10)\)?', clean, re.IGNORECASE)
    if match: 
        return normalize(match.group(1))
    
    # --- STRATEGY 4: Start of line "A." or "A)" ---
    match = re.match(r'^([A-J]|[1-9]|10)\s*[:\).]', clean, re.IGNORECASE)
    if match: 
        return normalize(match.group(1))
    
    # --- STRATEGY 5: Standalone Token ---
    # Strip tags and punctuation for zero-shot "Direct" responses
    norm_text = re.sub(r'<.*?>|[\(\)\.\$\s]', '', clean).upper()
    if re.match(r'^[A-J]$', norm_text):
        return norm_text
    
    return "FAIL"

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    """Handles a single API request, restricted by the semaphore to prevent overwhelming the server."""
    async with semaphore:
        # Load directly from the prepared JSONL attributes
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        # Prevent runaway hallucination on few-shot prompts
        if strat == "Direct_Answer":
            extra_body_params["stop"] = ["Explanation:", "Reasoning:", "Here's why", "Q:"]
        # else:
        #     extra_body_params["stop"] = ["Q:"]
        
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
    
    # The semaphore acts as a throttle. It only allows 100 requests to be "in flight" at once.
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for strat in ACTIVE_STRATEGIES:
        if strat not in TEMPLATES: continue
        
        input_file = f"{DATA_DIR}/prepared_gpqa_fewshot_{strat}.jsonl"
        if not os.path.exists(input_file):
            print(f"Skipping {strat}: Could not find {input_file}. Run prep script first.")
            continue
            
        prepared_data = []
        with open(input_file, 'r') as f:
            for line in f:
                prepared_data.append(json.loads(line))
                
        config = TEMPLATES[strat]
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_gpqa_main_fewshot_{NUM_ITER}.jsonl"
        print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
        print(f"Loaded {len(prepared_data)} pre-assembled prompts.")
        
        # Create a list of background tasks (one for every question in the dataset)
        tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
        
        results = []
        # tqdm.gather runs them all and updates the progress bar as they finish
        completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
        # Filter out any None values (from errors) and append to results
        for res in completed_responses:
            if res is not None:
                results.append(res)
        
        # Save results to JSONL using pandas
        if results:
            df = pd.DataFrame(results)
            df.to_json(output_file, orient='records', lines=True)
            print(f"Saved {len(results)} results to {output_file}")

# # --- 4. THE MAIN ASYNC LOOP (WITH AUTO-RETRY LOGIC) ---
# async def run_eval():
#     model_label = args.model_name.replace("/", "_")
    
#     # The semaphore acts as a throttle. It only allows 100 requests to be "in flight" at once.
#     semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

#     for strat in ACTIVE_STRATEGIES:
#         if strat not in TEMPLATES: continue
        
#         input_file = f"{DATA_DIR}/prepared_gpqa_fewshot_{strat}.jsonl"
#         if not os.path.exists(input_file):
#             print(f"Skipping {strat}: Could not find {input_file}. Run prep script first.")
#             continue
            
#         prepared_data = []
#         with open(input_file, 'r') as f:
#             for line in f:
#                 prepared_data.append(json.loads(line))
                
#         config = TEMPLATES[strat]
#         output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_gpqa_main_fewshot_{NUM_ITER}.jsonl"
#         print(f"\n--- Strategy: {strat} (via Async vLLM Port {args.port}) ---")
        
#         # --- NEW: RETRY & PATCH LOGIC ---
#         indices_to_run = list(range(len(prepared_data)))
#         existing_results = []
        
#         if os.path.exists(output_file):
#             print(f"Found existing results at {output_file}. Scanning for extraction failures...")
#             existing_df = pd.read_json(output_file, orient='records', lines=True)
#             existing_results = existing_df.to_dict('records')
            
#             # Identify the exact row numbers where the regex returned "FAIL"
#             failed_indices = existing_df.index[existing_df['pred'] == 'FAIL'].tolist()
            
#             if not failed_indices:
#                 print("🎉 No failures found in this file! Moving to next strategy.")
#                 continue
                
#             print(f"⚠️ Found {len(failed_indices)} failures. Retrying ONLY these prompts...")
#             indices_to_run = failed_indices
#         else:
#             print(f"Loaded {len(prepared_data)} pre-assembled prompts for a fresh run.")
        
#         # Build tasks ONLY for the indices we need (either all of them, or just the fails)
#         tasks = [process_single_item(prepared_data[i], config, strat, semaphore) for i in indices_to_run]
        
#         # tqdm.gather runs them all and updates the progress bar as they finish
#         completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{strat}]")
        
#         # --- MERGE & SAVE LOGIC ---
#         if os.path.exists(output_file):
#             # We are patching an existing file
#             for idx, new_res in zip(indices_to_run, completed_responses):
#                 if new_res is not None:
#                     existing_results[idx] = new_res # Overwrite the failed row with the new attempt
#             final_results = existing_results
#         else:
#             # We are saving a brand new file
#             final_results = [res for res in completed_responses if res is not None]
        
#         # Save results to JSONL using pandas
#         if final_results:
#             df = pd.DataFrame(final_results)
#             df.to_json(output_file, orient='records', lines=True)
#             print(f"Saved {len(final_results)} patched results to {output_file}")

if __name__ == "__main__": 
    asyncio.run(run_eval())