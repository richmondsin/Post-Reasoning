import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "easy2hard_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 100 

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8010)
parser.add_argument("--model_name", type=str, default="gemma-3-12b-it")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
parser.add_argument("--subsets", type=str, default="",
                    help="comma-separated subsets: amc8,amc10,amc12,hmmt_feb,hmmt_nov (default: E2H_SUBSETS below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

ACTIVE_STRATEGIES = [
    # "Direct_Answer", 
    "Basic_CoT",
    # "Post_Reason"
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

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a direct math expert. Output ONLY the final integer answer. Do not provide any reasoning or explanation.",
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "do_sample": True, "max_new_tokens": 4096}
    },
    "Basic_CoT": {
        "sys_msg": "You are a chain-of-thought math expert. Think step-by-step to logically solve the problem. After explaining your full reasoning process, state the final integer answer.",
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "do_sample": True, "max_new_tokens": 4096}
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final integer answer first, then explain your reasoning.",
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "do_sample": True, "max_new_tokens": 4096}
    }
}

def extract_integer_answer(text, strat):
    if not isinstance(text, str): return "FAIL"
    
    clean = text.strip()
    
    candidate = clean

    triggers = ["The final answer is:", "The answer is:", "Answer:"]
    if strat in ["Post_Reason", "Direct_Answer"]:
        for trigger in triggers:
            idx = candidate.lower().find(trigger.lower())
            if idx != -1 and idx < 150: 
                parts = re.split(re.escape(trigger), candidate, flags=re.IGNORECASE, maxsplit=1)
                if len(parts) > 1:
                    candidate = parts[1].strip()
                    break

    stop_phrases = ["Explanation:", "Here's", "Let's", "Note:", "Proof:"]
    for phrase in stop_phrases:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1:
            candidate = candidate[:idx]
            
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', candidate)
    if box_match:
        candidate = box_match.group(1)

    candidate = candidate.replace(',', '').replace('$', '').strip()
    
    nums = re.findall(r'-?\d+', candidate)
    if nums:
        return nums[0] if strat in ["Post_Reason", "Direct_Answer"] else nums[-1]

    return "FAIL"

async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        subset = item['subset']
        
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0)
        }

        if strat == "Direct_Answer":
            extra_body_params["stop"] = ["\n", "Explanation:", "Reasoning:", "Q:", "</think>"]
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
            
            pred = extract_integer_answer(response_text, strat)
            is_correct = 1 if str(pred) == truth else 0
            
            clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
            clean_log_text = clean_log_text.replace("</think>", "").strip()

            return {
                "subset": subset, 
                "truth": truth, 
                "pred": pred, 
                "is_correct": is_correct,
                "response": clean_log_text.replace("\n", " "), 
                "raw_response": raw_response_json
            }
        except Exception as e:
            print(f"Error on item in {subset}: {e}")
            return None

async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for subset in E2H_SUBSETS:
        for strat in ACTIVE_STRATEGIES:
            if strat not in TEMPLATES: continue
                
            input_file = f"{DATA_DIR}/prepared_{subset}_fewshot_{strat}.jsonl"
            
            if not os.path.exists(input_file):
                print(f"Skipping {subset} [{strat}]: File not found.")
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
