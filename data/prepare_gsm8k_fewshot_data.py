import os
import json, os, re
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gsm8k_data")
os.makedirs(DATA_DIR, exist_ok=True)

ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    # "Basic_CoT", 
    # "Post_Reason",
    "Post_Summary",
    "Post_Confidence"
]

SUFFIXES = {
    "Direct_Answer": "\nAnswer immediately without any explanation or reasoning. Output format: 'Answer: [Answer].'",
    "Basic_CoT": "\nThink step-by-step to logically solve the problem. After detailing your reasoning, explicitly state the final numeric answer in the format: 'Answer: [Answer].'",
    "Post_Reason": "\nState the final answer immediately as 'Answer: [Answer].' THEN, explain your reasoning in 'Explanation: '.",
    "Post_Summary": "\nState the final answer immediately, then briefly summarize the problem and your answer in a single sentence. Output format: 'Answer: [Answer]. Summary: [summary]'",
    "Post_Confidence": "\nState the final answer immediately, then state your confidence level (0-100%) in this answer and briefly explain why. Output format: 'Answer: [Answer]. Confidence: [X%]. Explanation: [reasoning]'"
}

# --- 2. GSM8K EXTRACTION & CLEANING LOGIC ---
def clean_gsm8k_solution(raw_answer):
    """Removes the <<calculator>> annotations and the #### truth block."""
    # Split off the truth block
    reasoning = raw_answer.split("####")[0].strip()
    
    # Remove the calculator annotations (e.g., <<4*5=20>>) for cleaner few-shots
    reasoning = re.sub(r'<<.*?>>', '', reasoning)
    
    return reasoning.strip()

def extract_gsm8k_truth(raw_answer):
    """Extracts the final answer after the #### symbol."""
    if "####" in raw_answer:
        # GSM8K often has commas in large numbers (e.g., 130,000), so we strip them
        return raw_answer.split("####")[-1].replace(',', '').strip()
    return "FAIL"

# --- 3. FEW-SHOT BUILDER ---
def build_few_shot_prompt(few_shot_items, strat):
    examples = []
    for item in few_shot_items:
        prob = item['question']
        truth = extract_gsm8k_truth(item['answer'])
        sol = clean_gsm8k_solution(item['answer'])
        
        if strat == "Direct_Answer":
            examples.append(f"Q: {prob}\nAnswer: {truth}")
        elif strat == "Basic_CoT":
            # Provide the reasoning steps first, then the final answer
            examples.append(f"Q: {prob}\nExplanation: {sol}\nAnswer: {truth}")
        elif strat == "Post_Reason":
            # Provide the answer first, then the reasoning steps
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
def prepare_gsm8k():
    print("\n--- PREPARING GSM8K (FEW-SHOT) ---")
    print("Loading 'openai/gsm8k' dataset...")
    
    # Load the "main" config of GSM8K
    ds_train = load_dataset("openai/gsm8k", "main", split="train")
    ds_test = load_dataset("openai/gsm8k", "main", split="test")
    
    # Grab exactly 4 examples from the train split for the few-shot prefix
    # (4-shot or 5-shot is the industry standard for GSM8K)
    few_shot_items = [ds_train[i] for i in range(4)]
    print(f"Extracted {len(few_shot_items)} few-shot examples from train split.")

    test_tasks = []
    for item in ds_test:
        truth_val = extract_gsm8k_truth(item['answer'])
        
        if truth_val != "FAIL":
            test_tasks.append({
                "problem": item['question'],
                "truth": truth_val,
                "subset": "gsm8k_main"
            })
            
    print(f"Found {len(test_tasks)} valid test questions.")

    print("\nWriting JSONL files for evaluation...")
    for strat in ACTIVE_STRATEGIES:
        output_file = f"{DATA_DIR}/prepared_gsm8k_fewshot_{strat}.jsonl"
        few_shot_prefix = build_few_shot_prompt(few_shot_items, strat)
        
        prepared_data = []
        for task in test_tasks:
            full_user_prompt = f"{few_shot_prefix}Q: {task['problem']}{SUFFIXES[strat]}"
            prepared_data.append({
                "truth": task['truth'],
                "dataset": "gsm8k",
                "subset": task['subset'],
                "full_user_prompt": full_user_prompt
            })
            
        with open(output_file, "w") as f:
            for r in prepared_data: f.write(json.dumps(r) + "\n")
        print(f"✅ Saved {len(prepared_data)} {strat} prompts to {output_file}")

if __name__ == "__main__":
    prepare_gsm8k()