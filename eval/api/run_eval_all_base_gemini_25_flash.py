import json, re, os, argparse, asyncio, random
import pandas as pd
from google import genai
from google.genai import types
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# --- 1. CONFIGURATION ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

client = genai.Client(api_key=API_KEY)

BASE_DIR = os.environ.get("PR_WAREHOUSE", "warehouse")
RESULTS_DIR = f"{BASE_DIR}/results"
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1

# Concurrency & Retry Settings
CONCURRENT_REQUESTS = 20  
MAX_RETRIES = 8

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gemini-2.5-flash")
args = parser.parse_args()

ACTIVE_STRATEGIES = [
    "Direct_Answer", 
    "Post_Reason"
    ]

# Global Generation Parameters for Gemini
GEN_PARAMS = {
    "temperature": 0.7, 
    "top_p": 0.8, 
    "top_k": 20, 
    "max_output_tokens": 8192
}

# --- 2. BENCHMARK-SPECIFIC SYSTEM PROMPTS ---
BENCHMARK_PROMPTS = {
    "bbh": {
        "Direct_Answer": "You are a Direct-Answer engine. Output ONLY the final answer. Do not provide any explanation or reasoning.",
        "Post_Reason": "You are a Post-Reasoning engine. State the final answer first, then explain your logic."
    },
    "easy2hard": {
        "Direct_Answer": "You are a direct math expert. Output ONLY the final integer answer. Do not provide any reasoning or explanation.",
        "Post_Reason": "You are a post-reasoning math expert. State the final integer answer first, then explain your reasoning."
    },
    "gpqa": {
        "Direct_Answer": "You are an expert in graduate-level science (biology, physics, and chemistry). You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
        "Post_Reason": "You are an expert in graduate-level science (biology, physics, and chemistry). State the answer letter first, then explain your scientific reasoning."
    },
    "gsm8k": {
        "Direct_Answer": "You are a direct math expert. Output ONLY the final numeric answer. Do not provide any reasoning or explanation.",
        "Post_Reason": "You are a post-reasoning math expert. State the final numeric answer first, then explain your reasoning."
    },
    "math": {
        "Direct_Answer": "You are a direct math expert. You must output ONLY the final answer. Do not provide any reasoning, or explanation.",
        "Post_Reason": "You are a post-reasoning math expert. State the final result first, then justify."
    },
    "mmlu_pro": {
        "Direct_Answer": "You are an expert academic AI. You must answer complex, graduate-level multiple-choice questions across diverse domains. Output ONLY the letter of the correct option (A through J). Do not provide any explanation, reasoning, or caveats.",
        "Post_Reason": "You are an expert academic AI answering complex, graduate-level multiple-choice questions across diverse domains. You must state the final option letter (A through J) first, and then provide a rigorous scientific or logical justification for your choice."
    }
}

# --- 3. EXTRACTOR FUNCTIONS ---
first_answer_strats = ["Direct_Answer", "Post_Reason"]

def extract_bbh(text, strat):
    clean = text.replace("**", "")
    candidate = clean
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(.*?)\s*\}', clean, re.IGNORECASE)
    if boxed_matches: 
        candidate = boxed_matches[0].strip() if strat in first_answer_strats else boxed_matches[-1].strip()
    else:
        ans_matches = re.findall(r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$), clean, re.IGNORECASE | re.DOTALL)")
        if ans_matches: 
            candidate = re.sub(r"^(Answer|:)\s*", "", ans_matches[0] if strat in first_answer_strats else ans_matches[-1], flags=re.IGNORECASE).strip()
        else:
            candidate = clean.split('\n')[0].strip() if strat in first_answer_strats else clean.split('\n')[-1].strip()
            
    mc_match = re.match(r'^\s*[\(\[]?([A-Z])[\]\)\.]*(?:\s|$)', candidate, re.IGNORECASE)
    if mc_match: return f"({mc_match.group(1).upper()})"
    return candidate.rstrip('.').strip()

def extract_e2h_gsm8k(text, strat):
    candidate = text
    for trigger in ["The final answer is:", "The answer is:", "Answer:"]:
        idx = candidate.lower().find(trigger.lower())
        if idx != -1 and idx < 150: 
            parts = re.split(re.escape(trigger), candidate, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) > 1:
                candidate = parts[1].strip()
                break
    for phrase in ["Explanation:", "Here's", "Let's", "Note:", "Proof:"]:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1: candidate = candidate[:idx]
            
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', candidate)
    if box_match: candidate = box_match.group(1)
    
    nums = re.findall(r'-?\d+\.?\d*', candidate.replace(',', '').replace('$', '').strip())
    if nums: return nums[0] if strat in first_answer_strats else nums[-1]
    return "FAIL"

def extract_gpqa(text, strat):
    clean = text
    def normalize(val):
        val = re.sub(r'\\text\s*\{|\}|\$|\)|[:\.]', '', val, flags=re.IGNORECASE).strip().upper()
        return {"1":"A", "2":"B", "3":"C", "4":"D"}.get(val, val)

    tag_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', clean, re.IGNORECASE | re.DOTALL)
    if tag_matches and re.match(r'^[A-D]$', normalize(tag_matches[-1])): return normalize(tag_matches[-1])
    
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(?:\\text\s*\{)?\s*([A-D]|[1-4])\s*\}?', clean, re.IGNORECASE)
    if boxed_matches: return normalize(boxed_matches[-1])

    ans_matches = re.findall(r'Answer:\s*\(?(?:Option\s+)?([A-D]|[1-4])\)?', clean, re.IGNORECASE)
    if ans_matches: return normalize(ans_matches[-1])
    
    match = re.match(r'^([A-D]|[1-4])\s*[:\).]', clean, re.IGNORECASE)
    if match: return normalize(match.group(1))
    
    norm_text = re.sub(r'<.*?>|[\(\)\.\$\s]', '', clean).upper()
    if re.match(r'^[A-D]$', norm_text): return norm_text
    return "FAIL"

def extract_math(text, strat):
    candidate = text
    for trigger in ["The final answer is:", "The answer is:", "My answer is:", "Answer:"]:
        idx = text.lower().find(trigger.lower())
        if idx != -1 and idx < 150: 
            parts = re.split(re.escape(trigger), text, flags=re.IGNORECASE, maxsplit=1)
            if len(parts) > 1:
                candidate = parts[1].strip()
                break
    for phrase in ["Explanation", "Here's", "Let's", "Note:", "The ", "To ", "Question:"]:
        idx = candidate.lower().find(phrase.lower())
        if idx != -1: candidate = candidate[:idx]
            
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', candidate)
    if box_match: candidate = box_match.group(1)
    
    candidate = re.sub(r'^(?:The final answer is|The answer is|Answer:?)\s*:?\s*', '', candidate, flags=re.IGNORECASE)
    return re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', candidate.replace('$', '').replace('**', '').replace(',', '')).rstrip('.').rstrip(';').strip()

def extract_mmlu_pro(text, strat):
    candidate = text.replace("**", "")
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(.*?)\s*\}', candidate, re.IGNORECASE)
    if boxed_matches: 
        candidate = boxed_matches[0].strip() if strat in first_answer_strats else boxed_matches[-1].strip()
    else:
        ans_matches = re.findall(r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$), candidate, re.IGNORECASE | re.DOTALL)")
        if ans_matches: 
            candidate = re.sub(r"^(Answer|:)\s*", "", ans_matches[0] if strat in first_answer_strats else ans_matches[-1], flags=re.IGNORECASE).strip()
        else:
            candidate = candidate.split('\n')[0].strip() if strat in first_answer_strats else candidate.split('\n')[-1].strip()

    mc_match = re.match(r'^\s*\(?([A-J])(?:[)\.]\s|[)\.]$|$)', candidate, re.IGNORECASE)
    if mc_match: return mc_match.group(1).upper()
    fallback_matches = re.findall(r'\b([A-J])\b', candidate, re.IGNORECASE)
    if fallback_matches: return fallback_matches[0].upper() if strat in first_answer_strats else fallback_matches[-1].upper()
    return candidate.strip()

def evaluate_correctness(pred, truth, benchmark):
    if benchmark in ["bbh", "gpqa", "mmlu_pro"]:
        return 1 if str(pred).strip().lower() == str(truth).strip().lower() else 0
    else:
        if str(pred).strip().lower() == str(truth).strip().lower(): return 1
        try:
            return 1 if float(pred) == float(truth) else 0
        except:
            return 0

# --- 4. REGISTRY ---
BENCHMARKS = {
    "bbh": {"dir": "bbh_full_data", "prefix": "prepared_bbh_{data_strat}.jsonl", "extractor": extract_bbh, "subsets": ["main"]},
    "easy2hard": {"dir": "easy2hard_data", "prefix": "prepared_{subset}_fewshot_{data_strat}.jsonl", "extractor": extract_e2h_gsm8k, "subsets": [
        "amc8", 
        "amc10", 
        "amc12", 
        "hmmt_feb", 
        "hmmt_nov"
        ]},
    "gpqa": {"dir": "gpqa_main_data", "prefix": "prepared_gpqa_fewshot_{data_strat}.jsonl", "extractor": extract_gpqa, "subsets": ["main"]},
    "gsm8k": {"dir": "gsm8k_data", "prefix": "prepared_gsm8k_fewshot_{data_strat}.jsonl", "extractor": extract_e2h_gsm8k, "subsets": ["main"]},
    "math": {"dir": "math_data", "prefix": "prepared_math_fewshot_{data_strat}.jsonl", "extractor": extract_math, "subsets": ["main"]},
    "mmlu_pro": {"dir": "mmlu_pro_data", "prefix": "prepared_mmlu_pro_fewshot_{data_strat}.jsonl", "extractor": extract_mmlu_pro, "subsets": ["main"]}
}

# --- 5. ASYNC WORKER WITH SMART JITTER BACKOFF ---
async def process_single_item(item, benchmark_name, subset, sys_msg, strat, extractor, semaphore, pbar):
    async with semaphore:
        prompt = item['full_user_prompt']
        truth = str(item['truth']).strip()

        gemini_config = types.GenerateContentConfig(
            system_instruction=sys_msg,
            temperature=GEN_PARAMS["temperature"],
            top_p=GEN_PARAMS["top_p"],
            top_k=GEN_PARAMS["top_k"],
            max_output_tokens=GEN_PARAMS["max_output_tokens"],
            thinking_config=types.ThinkingConfig(thinking_budget=0)
        )

        for attempt in range(MAX_RETRIES):
            try:
                response = await client.aio.models.generate_content(
                    model=args.model_name,
                    contents=prompt,
                    config=gemini_config
                )
                
                response_text = response.text or ""
                pred = extractor(response_text, strat)
                is_correct = evaluate_correctness(pred, truth, benchmark_name)
                
                clean_log_text = response_text.replace("\n", " ")
                
                pbar.update(1) 
                return {
                    "benchmark": benchmark_name,
                    "subset": subset,
                    "truth": truth, 
                    "pred": pred, 
                    "is_correct": is_correct,
                    "response": clean_log_text, 
                    "raw_response": response.model_dump_json() if hasattr(response, 'model_dump_json') else str(response)
                }
                
            except Exception as e:
                error_str = str(e).lower()
                if "429" in error_str or "503" in error_str or "quota" in error_str:
                    if attempt < MAX_RETRIES - 1:
                        delay = (2 ** attempt) + random.uniform(0.1, 3.0)
                        await asyncio.sleep(delay)
                    else:
                        pbar.update(1)
                        print(f"\n[FAILED - RATE LIMIT] {benchmark_name} - {subset}")
                        return None
                else:
                    pbar.update(1)
                    print(f"\n[FAILED - API ERROR] {benchmark_name} - {subset}: {e}")
                    return None

# --- 6. MAIN LOOP ---
async def run_eval():
    model_label = args.model_name.replace("/", "_")
    semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS)

    for b_name, b_info in BENCHMARKS.items():
        for strat in ACTIVE_STRATEGIES:
            # Dynamically pull the correct prompt for this benchmark and strategy
            sys_msg = BENCHMARK_PROMPTS[b_name][strat]
            
            # Map the strategy directly to the file naming convention
            data_strat = strat
            
            for subset in b_info["subsets"]:
                filename = b_info["prefix"].format(subset=subset, data_strat=data_strat)
                input_file = f"{BASE_DIR}/{b_info['dir']}/{filename}"
                
                if not os.path.exists(input_file):
                    print(f"Skipping {b_name} [{subset}] - {strat}: File not found")
                    continue
                    
                prepared_data = []
                with open(input_file, 'r') as f:
                    for line in f:
                        prepared_data.append(json.loads(line))
                
                # --- EXACT NAMING LOGIC ---
                if b_name == "bbh":
                    file_mid = "bbh_all"
                elif b_name == "gpqa":
                    file_mid = "gpqa_main"
                elif b_name == "easy2hard":
                    file_mid = subset # e.g., "amc8" without "easy2hard_"
                else:
                    file_mid = b_name # gsm8k, math, mmlu_pro
                    
                output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_{file_mid}_fewshot_{NUM_ITER}.jsonl"
                # --------------------------
                
                print(f"\n--- Benchmark: {b_name.upper()} | Subset: {subset} | Strat: {strat} ---")
                
                with tqdm(total=len(prepared_data), desc=f"Eval {b_name}") as pbar:
                    tasks = [process_single_item(item, b_name, subset, sys_msg, strat, b_info["extractor"], semaphore, pbar) for item in prepared_data]
                    completed = await asyncio.gather(*tasks)
                
                results = [res for res in completed if res is not None]

                if results:
                    df = pd.DataFrame(results)
                    df.to_json(output_file, orient='records', lines=True)
                    acc = df['is_correct'].mean() * 100
                    print(f"🔥 {b_name.upper()} [{subset}] Accuracy: {acc:.2f}% | Saved to {output_file}")

if __name__ == "__main__":
    asyncio.run(run_eval())