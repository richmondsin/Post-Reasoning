import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "easy2hard_data")
RESULTS_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1
CONCURRENT_REQUESTS = 20

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=8020)
parser.add_argument("--model_name", type=str, default="Qwen35-9B")
parser.add_argument("--strategies", type=str, default="",
                    help="comma-separated arms to run, e.g. Direct_Answer,Post_Reason (default: ACTIVE_STRATEGIES below)")
parser.add_argument("--subsets", type=str, default="",
                    help="comma-separated subsets: amc8,amc10,amc12,hmmt_feb,hmmt_nov (default: E2H_SUBSETS below)")
args = parser.parse_args()

client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="vllm-speed")

ACTIVE_STRATEGIES = [
    # "Direct_Answer", 
    # "Basic_CoT",
    # "Post_Reason",
    "Thinking",
    "Thinking_Post"
    ]
if args.strategies:
    ACTIVE_STRATEGIES = [s.strip() for s in args.strategies.split(",") if s.strip()]
E2H_SUBSETS = [
    # "amc8", 
    # "amc10", 
    # "amc12", 
    "hmmt_feb", 
    "hmmt_nov"
    ]
if args.subsets:
    E2H_SUBSETS = [s.strip() for s in args.subsets.split(",") if s.strip()]

TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are a direct math expert. Output ONLY the final integer answer. Do not provide any reasoning or explanation.",
        "enable_thinking": False,
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "do_sample": True, "max_new_tokens": 4096}
    },
    "Basic_CoT": {
        "sys_msg": "You are a chain-of-thought math expert. Think step-by-step to logically solve the problem. After explaining your full reasoning process, state the final integer answer.",
        "enable_thinking": False,
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "do_sample": True, "max_new_tokens": 4096}
    },
    "Post_Reason": {
        "sys_msg": "You are a post-reasoning math expert. State the final integer answer first, then explain your reasoning.",
        "enable_thinking": False,
        "gen_params": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "do_sample": True, "max_new_tokens": 4096}
    },
    "Thinking": {
        "sys_msg": "You are a direct math expert. Output ONLY the final integer answer. Do not provide any reasoning or explanation.",
        "enable_thinking": True,
        "gen_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "do_sample": True, "max_new_tokens": 16384}
    },
    "Thinking_Post": {
        "sys_msg": "You are a post-reasoning math expert. State the final integer answer first, then explain your reasoning.",
        "enable_thinking": True,
        "gen_params": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0, "do_sample": True, "max_new_tokens": 16384}
    }
}

def extract_integer_answer(text, strat, enable_thinking):
    if not isinstance(text, str): return "FAIL"
    
    clean = text
    
    # NEW: Strip <think> tags completely before extraction for Thinking strategies
    if enable_thinking:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
        clean = clean.replace("</think>", "").replace("<think>", "").strip()
    
    candidate = clean

    # Helper for strats that output answer at the start
    first_answer_strats = ["Post_Reason", "Direct_Answer", "Thinking", "Thinking_Post"]

    triggers = ["The final answer is:", "The answer is:", "Answer:"]
    if strat in first_answer_strats:
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
        return nums[0] if strat in first_answer_strats else nums[-1]

    return "FAIL"

async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        subset = item['subset']
        
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
            
            # Predict
            pred = extract_integer_answer(response_text, strat, config["enable_thinking"])
            is_correct = 1 if str(pred) == truth else 0
            
            # Clean log
            if config["enable_thinking"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()

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
                
            # NEW: Data routing mapping
            data_strat = strat
            if strat == "Thinking":
                data_strat = "Direct_Answer"
            elif strat == "Thinking_Post":
                data_strat = "Post_Reason"
                
            input_file = f"{DATA_DIR}/prepared_{subset}_fewshot_{data_strat}.jsonl"
            
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