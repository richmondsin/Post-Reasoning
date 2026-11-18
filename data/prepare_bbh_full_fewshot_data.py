import os
import json, re, os
import urllib.request
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "bbh_full_data")
os.makedirs(DATA_DIR, exist_ok=True)

ACTIVE_STRATEGIES = [
    "Direct_Answer", 
    "Basic_CoT",     # Added Basic_CoT
    "Post_Reason"
]

BBH_TASKS = [
    "boolean_expressions", "causal_judgement", "date_understanding", "disambiguation_qa",
    "dyck_languages", "formal_fallacies", "geometric_shapes", "hyperbaton",
    "logical_deduction_five_objects", "logical_deduction_seven_objects",
    "logical_deduction_three_objects", "movie_recommendation", "multistep_arithmetic_two",
    "navigate", "object_counting", "penguins_in_a_table", "reasoning_about_colored_objects",
    "ruin_names", "salient_translation_error_detection", "snarks", "sports_understanding",
    "temporal_sequences", "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects", "tracking_shuffled_objects_three_objects",
    "web_of_lies", "word_sorting"
]

# We only need the suffixes here to build the final prompt string
SUFFIXES = {
    "Direct_Answer": "\nAnswer immediately. Output format: 'Answer: [Answer].'",
    "Basic_CoT": "\nThink step-by-step to logically solve the problem. After detailing your reasoning, explicitly state the final answer. Output format: 'Answer: [Answer].'",
    "Post_Reason": "\nState the final answer immediately as 'Answer: [Answer].' THEN, provide your reasoning in 'Explanation: '."
}

# --- 2. DYNAMIC FEW-SHOT PARSER ---
FEW_SHOT_CACHE = {}

def build_few_shot_prompt(task_name, strat):
    """Downloads and repackages the official BBH CoT prompts into target strategies."""
    if task_name not in FEW_SHOT_CACHE:
        url = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/main/cot-prompts/{task_name}.txt"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            text = urllib.request.urlopen(req).read().decode('utf-8')
            FEW_SHOT_CACHE[task_name] = text
        except Exception as e:
            print(f"Warning: Failed to fetch prompt for {task_name} ({e}). Falling back to zero-shot.")
            return "" 

    raw_text = FEW_SHOT_CACHE[task_name]
    
    # 1. Grab the official task instruction
    instruction_match = re.search(r"^(.*?)(?=Q:), raw_text, re.DOTALL)")
    instruction = instruction_match.group(1).strip() if instruction_match else ""
    
    # Strip out the official BIG-Bench canary warning block
    if "-----\n" in instruction:
        instruction = instruction.split("-----\n")[-1].strip()

    # 2. Rip open the Q&A blocks
    qa_pattern = r"Q:\s*(.*?)\s*A:\s*Let's think step by step\.\s*(.*?)(?:So the answer is|The answer is|the answer is)\s*(.*?)\.\n"
    matches = re.findall(qa_pattern, raw_text + "\n", re.DOTALL)
    
    formatted_examples = [f"Task Description: {instruction}"] if instruction else []
    
    # 3. Repackage into strategy format
    for q, reasoning, ans in matches[:3]:
        q, reasoning, ans = q.strip(), reasoning.strip(), ans.strip()
        
        if strat == "Direct_Answer":
            formatted_examples.append(f"Q: {q}\nAnswer: {ans}.")
        elif strat == "Basic_CoT":
            # Provide the reasoning steps first, then explicitly state the answer format
            formatted_examples.append(f"Q: {q}\nExplanation: {reasoning}\nAnswer: {ans}.")
        elif strat == "Post_Reason":
            # Provide the answer first, then the reasoning steps
            formatted_examples.append(f"Q: {q}\nAnswer: {ans}. Explanation: {reasoning}")
        else:
            # Fallback safety
            formatted_examples.append(f"Q: {q}\nA: Let's think step by step. {reasoning} So the answer is {ans}.")
            
    if len(formatted_examples) <= 1: 
        return "" 
        
    return "\n\n".join(formatted_examples) + "\n\n"

# --- 3. MAIN PREPARATION LOOP ---
def prepare_data():
    print("\n--- PREPARING BBH (FEW-SHOT) ---")
    print("Loading all BBH Tasks from HuggingFace...")
    all_bbh_tasks = []
    
    for task_name in BBH_TASKS:
        try:
            ds = load_dataset("lukaemon/bbh", task_name, split="test")
            for item in ds:
                item['task_name'] = task_name 
                all_bbh_tasks.append(item)
        except Exception as e:
            print(f"Skipping {task_name}: {e}")

    print(f"Total BBH samples loaded: {len(all_bbh_tasks)}\n")

    print("Writing JSONL files for evaluation...")
    for strat in ACTIVE_STRATEGIES:
        output_file = f"{DATA_DIR}/prepared_bbh_{strat}.jsonl"
        prepared_data = []

        for item in all_bbh_tasks:
            q_text = item['input']
            
            # Task-Specific Prompt Injections
            if item['task_name'] in ["sports_understanding", "web_of_lies"]:
                q_text += " Answer yes or no."
            elif item['task_name'] == "date_understanding":
                q_text += "\nGive your answer based on the letter."
            elif item['task_name'] == "boolean_expressions":
                q_text += "? Answer True or False."
                
            truth = str(item['target']).strip()
            
            # Build the fully assembled prompt
            few_shot_prefix = build_few_shot_prompt(item['task_name'], strat)
            full_user_prompt = f"{few_shot_prefix}Q: {q_text}{SUFFIXES[strat]}"
            
            prepared_data.append({
                "task_name": item['task_name'],
                "dataset": "bbh",
                "subset": item['task_name'],
                "truth": truth,
                "full_user_prompt": full_user_prompt
            })

        # Save to disk
        with open(output_file, "w") as f:
            for r in prepared_data: 
                f.write(json.dumps(r) + "\n")
                
        print(f"✅ Saved {len(prepared_data)} fully assembled {strat} prompts to {output_file}")

if __name__ == "__main__":
    prepare_data()