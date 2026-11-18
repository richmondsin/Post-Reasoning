import json, re, os, time, argparse
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

# ==========================================
# --- 1. CONFIGURATION & TARGET SELECTION ---
# ==========================================
load_dotenv(override=True)
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Use the synchronous client for Batch API polling
client = OpenAI(api_key=API_KEY)

BASE_DIR = os.environ.get("PR_WAREHOUSE", "warehouse")
RESULTS_DIR = f"{BASE_DIR}/results"
BATCH_TEMP_DIR = f"{BASE_DIR}/batch_temp"  # Directory to store intermediate batch files
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(BATCH_TEMP_DIR, exist_ok=True)
NUM_ITER = 1

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="gpt-5.4-mini")
args = parser.parse_args()
MODEL_NAME = args.model_name

# ------------------------------------------
# CHOOSE STRATEGIES (Uncomment the one you want)
# ------------------------------------------
ACTIVE_STRATEGIES = [
    # "Direct_Answer", 
    "Post_Reason"
]  

# ------------------------------------------
# CHOOSE BENCHMARKS (Uncomment the one you want)
# ------------------------------------------
TARGET_BENCHMARKS = [
    # "bbh", 
    # "easy2hard", 
    # "gpqa", 
    # "gsm8k", 
    # "math", 
    "mmlu_pro"
] 

# Global Generation Parameters
GEN_PARAMS = {
    "max_output_tokens": 8192  # Using max_output_tokens for Responses API
}

# --- 2. EXACT REQUESTED SYSTEM PROMPTS ---
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

active_benchmarks = {k: v for k, v in BENCHMARKS.items() if k in TARGET_BENCHMARKS}
if not active_benchmarks:
    print(f"❌ Error: No valid benchmarks selected.")
    exit(1)


# --- 5. BATCH PROCESSING LOOP ---
def run_batch_eval():
    model_label = MODEL_NAME.replace("/", "_")
    
    active_batches = []
    finished_batches = []

    print("========================================")
    print("PHASE 1: BATCH CREATION & SUBMISSION")
    print("========================================")

    for b_name, b_info in active_benchmarks.items():
        for strat in ACTIVE_STRATEGIES:
            sys_msg = BENCHMARK_PROMPTS[b_name][strat]
            data_strat = strat
            
            for subset in b_info["subsets"]:
                filename = b_info["prefix"].format(subset=subset, data_strat=data_strat)
                input_file = f"{BASE_DIR}/{b_info['dir']}/{filename}"
                
                if not os.path.exists(input_file):
                    print(f"Skipping {b_name} [{subset}] - {strat}: File not found")
                    continue
                
                # --- Read original data and build the Batch JSONL ---
                batch_file_path = f"{BATCH_TEMP_DIR}/req_{b_name}_{subset}_{strat}.jsonl"
                item_lookup = {}
                
                with open(input_file, 'r') as f_in, open(batch_file_path, 'w') as f_out:
                    for idx, line in enumerate(f_in):
                        item = json.loads(line)
                        custom_id = f"req_{idx}"
                        item_lookup[custom_id] = item
                        
                        # --- Format required by OpenAI Batch API targeting /v1/responses ---
                        request_obj = {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/responses", # <--- TARGET RESPONSES ENDPOINT
                            "body": {
                                "model": MODEL_NAME,
                                "input": [ # <--- 'input' INSTEAD OF 'messages'
                                    {"role": "system", "content": sys_msg},
                                    {"role": "user", "content": item['full_user_prompt']}
                                ],
                                "max_output_tokens": GEN_PARAMS["max_output_tokens"],
                                "reasoning": {"effort": "none"} # <--- SUPPRESS THINKING
                            }
                        }
                        
                        f_out.write(json.dumps(request_obj) + "\n")
                
                print(f"Submitting: {b_name.upper()} [{subset}] - {strat} ({len(item_lookup)} items)...")
                
                try:
                    # 1. Upload the file to OpenAI
                    with open(batch_file_path, "rb") as file_obj:
                        batch_input_file = client.files.create(
                          file=file_obj,
                          purpose="batch"
                        )
                    
                    # 2. Trigger the Batch job (Notice the endpoint points to /v1/responses)
                    batch = client.batches.create(
                        input_file_id=batch_input_file.id,
                        endpoint="/v1/responses", # <--- ENDPOINT MATCHES REQUEST URL
                        completion_window="24h",
                        metadata={
                          "description": f"{b_name}_{subset}_{strat}"
                        }
                    )
                    
                    active_batches.append({
                        "batch_id": batch.id,
                        "b_name": b_name,
                        "subset": subset,
                        "strat": strat,
                        "extractor": b_info["extractor"],
                        "item_lookup": item_lookup,
                        "total_items": len(item_lookup)
                    })
                    print(f" -> Batch created! ID: {batch.id}")
                    
                except Exception as e:
                    print(f" -> ERROR submitting batch: {e}")

    print("\n========================================")
    print("PHASE 2: CONCURRENT POLLING")
    print("========================================")

    while active_batches:
        print(f"\n[Polling {len(active_batches)} Active Batches...]")
        still_active = []
        
        for batch_info in active_batches:
            try:
                batch_status = client.batches.retrieve(batch_info["batch_id"])
                status = batch_status.status
                
                completed_count = batch_status.request_counts.completed if batch_status.request_counts else 0
                failed_count = batch_status.request_counts.failed if batch_status.request_counts else 0
                total = batch_info["total_items"]
                
                print(f" -> {batch_info['b_name'].upper()} [{batch_info['subset']}] ({batch_info['strat']}): {status.upper()} (Done: {completed_count}/{total}, Failed: {failed_count})")
                
                if status == "completed":
                    # Store output file id for phase 3
                    batch_info["output_file_id"] = batch_status.output_file_id
                    finished_batches.append(batch_info)
                elif status in ["failed", "expired", "cancelled"]:
                    print(f"    WARNING: Batch {status}. Removing from queue.")
                    if batch_status.error_file_id:
                        print(f"    Error file ID available: {batch_status.error_file_id}")
                else:
                    still_active.append(batch_info)
            except Exception as e:
                print(f" -> Error retrieving status for {batch_info['batch_id']}: {e}")
                still_active.append(batch_info) 
                
        active_batches = still_active
        if active_batches:
            time.sleep(30) 

    print("\n========================================")
    print("PHASE 3: RETRIEVAL & EVALUATION")
    print("========================================")

    for batch_info in finished_batches:
        b_name = batch_info["b_name"]
        subset = batch_info["subset"]
        strat = batch_info["strat"]
        item_lookup = batch_info["item_lookup"]
        extractor = batch_info["extractor"]
        output_file_id = batch_info["output_file_id"]
        
        print(f"\nEvaluating: {b_name.upper()} [{subset}] - {strat}...")
        results_list = []
        
        try:
            # Download the results file
            file_response = client.files.content(output_file_id)
            results_text = file_response.text
            
            # Parse JSONL output
            for line in results_text.strip().split('\n'):
                if not line: continue
                result_obj = json.loads(line)
                custom_id = result_obj["custom_id"]
                
                original_item = item_lookup[custom_id]
                truth = str(original_item['truth']).strip()
                
                # Check for errors in the individual request
                if result_obj["response"]["status_code"] != 200:
                    print(f" -> Request {custom_id} failed: {result_obj['response']['body']}")
                    continue
                    
                # --- Extract from the /v1/responses format ---
                response_text = ""
                body = result_obj["response"]["body"]

                if "output" in body and len(body["output"]) > 0:
                    for block in body["output"][0].get("content", []):
                        if block.get("type") == "output_text":
                            response_text += block.get("text", "")
                
                pred = extractor(response_text, strat)
                is_correct = evaluate_correctness(pred, truth, b_name)
                
                results_list.append({
                    "benchmark": b_name,
                    "subset": subset,
                    "truth": truth, 
                    "pred": pred, 
                    "is_correct": is_correct,
                    "response": response_text.replace("\n", " "), 
                    "raw_response": json.dumps(result_obj)
                })
                    
        except Exception as e:
            print(f"Error retrieving/processing results for {batch_info['batch_id']}: {e}")
            continue
            
        if b_name == "bbh": file_mid = "bbh_all"
        elif b_name == "gpqa": file_mid = "gpqa_main"
        elif b_name == "easy2hard": file_mid = subset
        else: file_mid = b_name
            
        output_file = f"{RESULTS_DIR}/base_{model_label}_{strat}_{file_mid}_fewshot_{NUM_ITER}.jsonl"

        if results_list:
            df = pd.DataFrame(results_list)
            df.to_json(output_file, orient='records', lines=True)
            acc = df['is_correct'].mean() * 100
            print(f"🔥 {b_name.upper()} [{subset}] Batch Accuracy: {acc:.2f}% | Saved to {output_file}")

if __name__ == "__main__":
    run_batch_eval()