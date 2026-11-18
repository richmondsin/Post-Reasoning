import json
import os
import re
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "easy2hard_data")
os.makedirs(DATA_DIR, exist_ok=True)

NUM_SHOTS = 3

ACTIVE_STRATEGIES = [
    "Direct_Answer", 
    "Basic_CoT",
    "Post_Reason"
]

SUFFIXES = {
    "Direct_Answer": "\nAnswer immediately without any explanation or reasoning. Output ONLY the final integer answer.",
    "Basic_CoT": "\nThink step-by-step to logically solve the problem. After explaining your full reasoning process, state the final integer answer as 'Answer: [Number]'.",
    "Post_Reason": "\nState the final integer answer immediately as 'Answer: [Number].' THEN, explain your reasoning in 'Explanation: '."
}

# --- 2. FILTERING & BUILDER LOGIC ---
def is_strict_integer(ans_str):
    """Checks if the answer is a pure integer (allows negative signs)."""
    if not isinstance(ans_str, str): ans_str = str(ans_str)
    ans_str = ans_str.strip().replace(',', '')
    return bool(re.fullmatch(r'-?\d+', ans_str))

def build_few_shot_prefix(few_shots, strat):
    """Constructs the few-shot examples block depending on strategy."""
    examples = []
    for item in few_shots:
        prob = item['problem']
        truth = item['truth']
        sol = item['solution']
        
        if strat == "Direct_Answer":
            examples.append(f"Q: {prob}\nAnswer: {truth}")
        elif strat == "Basic_CoT":
            # Reasoning first, explicitly formatted answer second
            examples.append(f"Q: {prob}\n{sol}\nAnswer: {truth}")
        elif strat == "Post_Reason":
            # Answer first, reasoning second
            examples.append(f"Q: {prob}\nAnswer: {truth}\nExplanation: {sol}")
            
    return "\n\n".join(examples) + "\n\n"

# --- 3. MAIN PREP LOOP ---
def prepare_easy2hard():
    print(f"\n--- PREPARING EASY2HARD BENCHMARK ({NUM_SHOTS}-SHOT) ---")
    print("Loading 'furonghuang-lab/Easy2Hard-Bench' [E2H-AMC, split='eval']...")
    
    # Using 'eval' to perfectly match your Hugging Face configuration
    ds = load_dataset("furonghuang-lab/Easy2Hard-Bench", "E2H-AMC", split="eval")
    
    categorized_data = {
        "AMC8": [], "AMC10": [], "AMC12": [], 
        "HMMT-Feb": [], "HMMT-Nov": []
    }
    
    skipped_non_integer = 0
    skipped_unmapped_tag = 0

    print("Filtering and routing problems...")
    for item in ds:
        raw_ans = str(item.get("answer", ""))
        difficulty = item.get("item_difficulty", None)
        
        # 1. BULLETPROOF TAG MATCHING
        row_str = str(item).replace(" ", "").upper()
        
        subset_name = None
        if "AMC8" in row_str:
            subset_name = "AMC8"
        elif "AMC10" in row_str:
            subset_name = "AMC10"
        elif "AMC12" in row_str:
            subset_name = "AMC12"
        elif "HMMT" in row_str and "FEB" in row_str:
            subset_name = "HMMT-Feb"
        elif "HMMT" in row_str and "NOV" in row_str:
            subset_name = "HMMT-Nov"
            
        if not subset_name:
            skipped_unmapped_tag += 1
            continue
            
        # 2. INTEGER CHECK
        if not is_strict_integer(raw_ans):
            skipped_non_integer += 1
            continue
            
        # 3. ROUTE TO SUBSET
        categorized_data[subset_name].append({
            "problem": item["problem"],
            "truth": raw_ans.strip(),
            "difficulty": difficulty,
            "solution": item.get("solution", "No solution provided.")
        })

    print(f"\n[FILTER SUMMARY]")
    print(f"Skipped (Unmapped/AIME): {skipped_unmapped_tag}")
    print(f"Skipped (Non-Integer Answer): {skipped_non_integer}")
    
    # --- 4. BUILD JSONL PROMPTS ---
    print("\nWriting JSONL files for evaluation...")
    for strat in ACTIVE_STRATEGIES:
        for subset, items in categorized_data.items():
            if len(items) <= NUM_SHOTS:
                print(f"⚠️ Skipping {subset} [{strat}]: Found {len(items)} integer problems (Need > {NUM_SHOTS}).")
                continue
            
            # Split into few-shot prefix and test set
            few_shots = items[:NUM_SHOTS]
            test_items = items[NUM_SHOTS:]
            
            few_shot_prefix = build_few_shot_prefix(few_shots, strat)
            output_file = f"{DATA_DIR}/prepared_{subset.lower().replace('-', '_')}_fewshot_{strat}.jsonl"
            
            with open(output_file, "w") as f:
                for task in test_items:
                    full_prompt = f"{few_shot_prefix}Q: {task['problem']}{SUFFIXES[strat]}"
                    
                    row = {
                        "dataset": "easy2hard",
                        "subset": subset,
                        "truth": task['truth'],
                        "difficulty": task['difficulty'],
                        "full_user_prompt": full_prompt
                    }
                    f.write(json.dumps(row) + "\n")
                    
            print(f"✅ {subset} [{strat}]: Used {NUM_SHOTS} shots, saved {len(test_items)} test prompts to {output_file}")

if __name__ == "__main__":
    prepare_easy2hard()