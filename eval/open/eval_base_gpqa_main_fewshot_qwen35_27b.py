import os
import json, re, os, argparse, asyncio
import pandas as pd
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gpqa_main_data")
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

# --- EXPERIMENT TEMPLATES (MERGED PARAMS + GPQA DOMAIN) ---
TEMPLATES = {
    "Direct_Answer": {
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
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
    "Basic_CoT": {
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). Think step-by-step through the scientific principles required to solve the problem, then state your final answer clearly.",
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
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
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
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then explain your scientific reasoning.",
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
    "Thinking_Post": {
        "sys_msg": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then explain your scientific reasoning.",
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

# --- 2. CLEANED SFT-TAG EXTRACTOR ---
def extract_answer_letter(text, strat, enable_thinking):
    if not isinstance(text, str): return "FAIL"
    
    clean = text
    
    # NEW: Strip <think> tags completely before extraction based on strategy
    if enable_thinking:
        clean = re.sub(r'<think>.*?(?:</think>|$)', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
    else:
        clean = re.sub(r'<think>.*?</think>', '', clean, flags=re.IGNORECASE | re.DOTALL).strip()
        clean = clean.replace("</think>", "").replace("<think>", "").strip()
    
    def normalize(val):
        # 1. Strip common LaTeX/Text debris from inside the match
        val = re.sub(r'\\text\s*\{|\}|\$|\)|[:\.]', '', val, flags=re.IGNORECASE)
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

    # --- STRATEGY 3: Standard Fallback (Answer: A) ---
    ans_matches = re.findall(r'Answer:\s*\(?(?:Option\s+)?([A-D]|[1-4])\)?', clean, re.IGNORECASE)
    if ans_matches: 
        return normalize(ans_matches[-1])
    
    # --- STRATEGY 4: Start of line "A." or "A)" ---
    match = re.match(r'^([A-D]|[1-4])\s*[:\).]', clean, re.IGNORECASE)
    if match: return normalize(match.group(1))
    
    # --- STRATEGY 5: Standalone Token ---
    norm_text = re.sub(r'<.*?>|[\(\)\.\$\s]', '', clean).upper()
    if re.match(r'^[A-D]$', norm_text):
        return norm_text
    
    return "FAIL"

# --- 3. THE ASYNC WORKER FUNCTION ---
async def process_single_item(item, config, strat, semaphore):
    async with semaphore:
        # Load directly from the prepared JSONL attributes
        full_user_prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()
        
        # VLLM specific kwargs for Qwen thinking + repetition penalty
        extra_body_params = {
            "top_k": config["gen_params"].get("top_k", 20),
            "min_p": config["gen_params"].get("min_p", 0.0),
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
                presence_penalty=config["gen_params"].get("presence_penalty", 0.0), # Passed directly to OpenAI spec
                max_tokens=config["gen_params"].get("max_new_tokens", 2048),
                extra_body=extra_body_params
            )

            response_text = full_api_response.choices[0].message.content or ""
            raw_response_json = full_api_response.model_dump_json()
            
            # Pass the enable_thinking flag to the extractor
            pred = extract_answer_letter(response_text, strat, config["enable_thinking"])
            
            # Clean log text safely handles unclosed tags for Thinking strats
            if config["enable_thinking"]:
                clean_log_text = re.sub(r'<think>.*?(?:</think>|$)', '', response_text, flags=re.IGNORECASE | re.DOTALL).strip()
            else:
                clean_log_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.IGNORECASE | re.DOTALL)
                clean_log_text = clean_log_text.replace("</think>", "").replace("<think>", "").strip()
            
            return {
                "truth": truth, 
                "pred": pred, 
                "is_correct": 1 if pred == truth else 0,
                "response": clean_log_text.replace("\n", " "), # Logs ONLY the external response
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
        
        # NEW: Data routing mapping
        data_strat = strat
        if strat == "Thinking":
            data_strat = "Direct_Answer"
        elif strat == "Thinking_Post":
            data_strat = "Post_Reason"
            
        input_file = f"{DATA_DIR}/prepared_gpqa_fewshot_{data_strat}.jsonl"
        
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
        print(f"\n--- Strategy: {strat} (Thinking Block: {config['enable_thinking']}) ---")
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