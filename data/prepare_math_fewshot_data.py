import os
import json, os, re
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "math_data")
os.makedirs(DATA_DIR, exist_ok=True)

SUBSETS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
QUESTIONS_PER_SUBSET = 1000

ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    # "Basic_CoT",
    # "Post_Reason",
    "Post_Summary",
    "Post_Confidence"
]

SUFFIXES = {
    "Direct_Answer": "\nAnswer immediately without any explanation or reasoning. Output format: 'Answer: [Answer].'",
    "Basic_CoT": "\nThink step-by-step to solve the problem, clearly laying out your mathematical reasoning. After your reasoning, clearly state the final answer in the format: 'Answer: [Answer].'",
    "Post_Reason": "\nState the final answer immediately as 'Answer: [Answer].' THEN, explain your reasoning in 'Explanation: '.",
    "Post_Summary": "\nState the final answer immediately, then briefly summarize the problem and your answer in a single sentence. Output format: 'Answer: [Answer]. Summary: [summary]'",
    "Post_Confidence": "\nState the final answer immediately, then state your confidence level (0-100%) in this answer and briefly explain why. Output format: 'Answer: [Answer]. Confidence: [X%]. Explanation: [reasoning]'"
}

# --- 2. MATH EXTRACTION LOGIC ---
def clean_math_string(s):
    if not isinstance(s, str): return "FAIL"
    box_match = re.search(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', s)
    if box_match: s = box_match.group(1)
    
    prefixes = [r"The final answer is", r"The answer is", r"My answer is", r"Answer:?", r"Answer"]
    pattern = r'^(?:' + '|'.join(prefixes) + r')\s*:?\s*'
    s = re.sub(pattern, '', s, flags=re.IGNORECASE)
    
    s = s.replace('$', '').replace('**', '').replace(',', '').strip()
    s = re.sub(r'\\text\s*\{\s*(.*?)\s*\}', r'\1', s)
    s = s.rstrip('.').rstrip(';').strip()
    return s

def get_ground_truth(raw_solution):
    boxed_matches = re.findall(r'\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}', raw_solution)
    ans = boxed_matches[-1] if boxed_matches else raw_solution
    return clean_math_string(ans)

def is_integer(s):
    return str(s).strip().replace('-', '', 1).isdigit()

# --- 3. FEW-SHOT BUILDER ---
def build_few_shot_prompt(few_shot_items, strat):
    examples = []
    for item in few_shot_items:
        prob = item['problem']
        truth = get_ground_truth(item['solution'])
        sol = item['solution'] # Use the official solution as the explanation block
        
        if strat == "Direct_Answer":
            examples.append(f"Q: {prob}\nAnswer: {truth}")
        elif strat == "Thinking":
            examples.append(f"Q: {prob}\nA: Let's think step by step.\n{sol}\nAnswer: {truth}")
        elif strat == "Basic_CoT":
            examples.append(f"Q: {prob}\n{sol}\nAnswer: {truth}")
        elif strat in ["Post_Reason", "Thinking_Post"]:
            examples.append(f"Q: {prob}\nAnswer: {truth}\nExplanation: {sol}")
        elif strat == "Post_Summary":
            # Extract just the first sentence of the explanation to simulate a brief summary
            summary_sentence = sol.split('.')[0] + "." if '.' in sol else sol
            examples.append(f"Q: {prob}\nAnswer: {truth}\nSummary: {summary_sentence}")
        elif strat == "Post_Confidence":
            # Inject a mock high confidence level before the reasoning to train the format
            examples.append(f"Q: {prob}\nAnswer: {truth}\nConfidence: 95%.\nExplanation: {sol}")
            
    return "\n\n".join(examples) + "\n\n"

# --- 4. MAIN PREP LOOP ---
def prepare_data():
    print("Loading 'algebra' train split to extract 3 Few-Shot examples...")
    train_ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="train")
    
    few_shot_items = []
    for item in train_ds:
        if is_integer(get_ground_truth(item['solution'])):
            few_shot_items.append(item)
            if len(few_shot_items) == 3: break
            
    print("Few-shot examples extracted successfully.")

    print("\nProcessing TEST splits for all subsets...")
    test_tasks = []
    
    for subset in SUBSETS:
        ds_full = load_dataset("EleutherAI/hendrycks_math", subset, split="test")
        subset_count = 0
        
        for item in ds_full:
            if subset_count >= QUESTIONS_PER_SUBSET: break
            
            truth_val = get_ground_truth(item['solution'])
            if is_integer(truth_val):
                test_tasks.append({
                    "problem": item['problem'],
                    "truth": truth_val,
                    "subset": subset
                })
                subset_count += 1
        print(f"  - {subset}: Found {subset_count} integer test questions.")

    print(f"\nTotal integer test questions: {len(test_tasks)}")

    print("\nWriting JSONL files for evaluation...")
    for strat in ACTIVE_STRATEGIES:
        output_file = f"{DATA_DIR}/prepared_math_fewshot_{strat}.jsonl"
        few_shot_prefix = build_few_shot_prompt(few_shot_items, strat)
        
        prepared_data = []
        for task in test_tasks:
            full_user_prompt = f"{few_shot_prefix}Q: {task['problem']}{SUFFIXES[strat]}"
            prepared_data.append({
                "truth": task['truth'],
                "subset": task['subset'],
                "full_user_prompt": full_user_prompt
            })
            
        with open(output_file, "w") as f:
            for r in prepared_data:
                f.write(json.dumps(r) + "\n")
        print(f"✅ Saved {len(prepared_data)} prompts to {output_file}")

if __name__ == "__main__":
    prepare_data()