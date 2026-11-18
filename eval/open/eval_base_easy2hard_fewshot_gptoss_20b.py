import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "easy2hard_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100 

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8008)
parser.add_argument("--model_name", type=str, default="gpt-oss-20b")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
parser.add_argument("--subsets", type=str, default="",
                    help="comma-separated subsets: amc8,amc10,amc12,hmmt_feb,hmmt_nov (default: E2H_SUBSETS below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

# --- 2. STRATEGIES AND SUBSETS ---
ACTIVE_STRATEGIES = [
    "Thinking",
    "Thinking_Post"
]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]

E2H_SUBSETS = [
    "amc8", 
    "amc10", 
    "amc12", 
    "hmmt_feb", 
    "hmmt_nov"
]
if args.subsets:
    E2H_SUBSETS = [s.strip() for s in args.subsets.split(",") if s.strip()]

# --- 3. TEMPLATES ---
TEMPLATES = {
    "Thinking": {
        "sys_msg": "You are a direct math expert. Output ONLY the final integer answer. Do not provide any reasoning or explanation.",
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "do_sample": True, "max_new_tokens": 32768}
    },
    "Thinking_Post": {
        "sys_msg": "You are a post-reasoning math expert. State the final integer answer first, then explain your reasoning.",
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "do_sample": True, "max_new_tokens": 32768}
    }
}

# --- 4. ANSWER EXTRACTION (GPQA-Style for Integers) ---
def extract_integer_answer(text, strat):
    if not isinstance(text, str): return "FAIL"
    clean = text.strip()
    
    def get_int(val):
        """Helper to strip formatting and grab the first integer it sees."""
        val = val.replace(',', '').replace('$', '').strip()
        nums = re.findall(r'-?\d+', val)
        return nums[0] if nums else "FAIL"

    # --- STRATEGY 0: <answer> TAGS ---
    tag_match = re.search(r'<answer>\s*(.*?)\s*</answer>', clean, re.IGNORECASE | re.DOTALL)
    if tag_match:
        # Check if they put \boxed{} inside the <answer> tag
        box_match = re.search(r'\\boxed\s*\{\s*(.*?)\s*\}', tag_match.group(1), re.IGNORECASE)
        extracted = box_match.group(1) if box_match else tag_match.group(1)
        res = get_int(extracted)
        if res != "FAIL": return res

    # --- STRATEGY 1: COMPLEX LaTeX BOXED ---
    match = re.search(r'\\boxed\s*\{\s*(.*?)\s*\}', clean, re.IGNORECASE)
    if match: 
        res = get_int(match.group(1))
        if res != "FAIL": return res

    # --- STRATEGY 2: TAG FALLBACK (Unclosed) ---
    match = re.search(r'<answer>\s*(.*)', clean, re.IGNORECASE | re.DOTALL)
    if match: 
        box_match = re.search(r'\\boxed\s*\{\s*(.*?)\s*\}', match.group(1), re.IGNORECASE)
        extracted = box_match.group(1) if box_match else match.group(1)
        res = get_int(extracted)
        if res != "FAIL": return res

    # --- STRATEGY 3: Standard Fallback (Handles Answer: **42**) ---
    match = re.search(r'Answer:\s*(.*?)(?:\n|$)', clean, re.IGNORECASE)
    if match: 
        box_match = re.search(r'\\boxed\s*\{\s*(.*?)\s*\}', match.group(1), re.IGNORECASE)
        extracted = box_match.group(1) if box_match else match.group(1)
        res = get_int(extracted)
        if res != "FAIL": return res
    
    # --- STRATEGY 4: Brute Force General Fallback ---
    candidate = clean
    stop_phrases = ["Explanation:", "Here's", "Let's", "Note:", "Proof:"]
    for phrase in stop_phrases:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1:
            candidate = candidate[:idx]
            
    nums = re.findall(r'-?\d+', candidate.replace(',', '').replace('$', ''))
    if nums:
        # For Direct/Post/Thinking strategies, grab the first number
        if strat in ["Post_Reason", "Direct_Answer", "Thinking", "Thinking_Post"]:
            return nums[0]
        else:
            return nums[-1]
    
    return "FAIL"

# --- 5. WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        subset = item['subset']
        
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
            
            pred = extract_integer_answer(response_text, strat)
            is_correct = 1 if str(pred) == truth else 0

            return {
                "subset": subset, 
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": response_text.replace("\n", " "), 
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item in {subset}: {e}")
            return None

# --- 6. MAIN LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for subset in E2H_SUBSETS:
        for strat in ACTIVE_STRATEGIES:
            if strat not in TEMPLATES: continue
            
            # Route to the correct underlying data files
            data_strat = "Direct_Answer" if strat == "Thinking" else "Post_Reason"
            input_file = f"{DATA_DIR}/prepared_{subset}_fewshot_{data_strat}.jsonl"
            
            if not os.path.exists(input_file):
                print(f"Skipping {subset} [{strat}]: File not found ({input_file}).")
                continue
                
            prepared_data = []
            with open(input_file, 'r') as f:
                for line in f:
                    prepared_data.append(json.loads(line))
            
            config = TEMPLATES[strat]
            output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_{subset}_fewshot_{NUM_ITER}.jsonl"
            print(f"\n--- Strategy: {strat} | Subset: {subset} (Model: {args.model_name}) ---")
            
            tasks = [process_single_item(item, config, strat, semaphore) for item in prepared_data]
            results = []
            completed_responses = await tqdm.gather(*tasks, desc=f"Eval [{subset} - {strat}]")
            
            for res in completed_responses:
                if res is not None:
                    results.append(res)
            
            if results:
                df = pd.DataFrame(results)
                df.to_json(output_file, orient='records', lines=True)
                acc = df['is_correct'].mean() * 100
                print(f"🔥 {subset} [{strat}] Accuracy: {acc:.2f}% | Saved {len(results)} results.")

if __name__ == "__main__":
    asyncio.run(run_eval())