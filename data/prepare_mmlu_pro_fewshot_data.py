import os
import json, os
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "mmlu_pro_data")
os.makedirs(DATA_DIR, exist_ok=True)

TARGET_TOTAL = 3000  # <--- Cap applied here during preparation

ACTIVE_STRATEGIES = [
    # "Direct_Answer",
    # "Post_Reason",
    # "Basic_CoT",
    "Post_Summary",
    "Post_Confidence"
]

SUFFIXES = {
    "Direct_Answer": "\nQuestion: Which of the given choices A through J is the correct answer?\nOutput ONLY the correct letter in this exact format: 'Answer: [Letter]'.",
    "Post_Reason": "\nQuestion: Which of the given choices A through J is the correct answer?\nState the final answer immediately in this exact format: 'Answer: [Letter]'. THEN, provide your rigorous explanation starting with 'Explanation: '.",
    "Basic_CoT": "\nQuestion: Which of the given choices A through J is the correct answer?\nThink step-by-step and provide your reasoning starting with 'Explanation: '. After your reasoning, you MUST state the final answer exactly as 'Answer: [Letter]'.",
    "Post_Summary": "\nQuestion: Which of the given choices A through J is the correct answer?\nState the final answer immediately, then briefly summarize the question and your selected answer in a single sentence. Output format: 'Answer: [Letter]. Summary: [summary]'",
    "Post_Confidence": "\nQuestion: Which of the given choices A through J is the correct answer?\nState the final answer immediately, then state your confidence level (0-100%) in this answer and briefly explain why. Output format: 'Answer: [Letter]. Confidence: [X%]. Explanation: [reasoning]'"
}

def format_question(item):
    """Formats the MMLU-Pro question and its up-to-10 options."""
    prompt = f"Q: {item['question']}\nOptions:\n"
    for i, opt in enumerate(item['options']):
        prompt += f"{chr(65+i)}. {opt}\n"
    return prompt

def build_few_shot_prompt(few_shot_items, strat):
    """Builds the 3-shot prefix using the official MMLU-Pro CoT explanations."""
    examples = []
    for item in few_shot_items:
        q_text = format_question(item)
        truth = item['answer']
        explanation = item['cot_content'] # The official reasoning trace
        
        if strat == "Direct_Answer":
            examples.append(f"{q_text}Answer: {truth}")
        elif strat == "Post_Reason":
            examples.append(f"{q_text}Answer: {truth}\nExplanation: {explanation}")
        elif strat == "Basic_CoT":
            # For standard CoT, the explanation precedes the final answer
            examples.append(f"{q_text}Explanation: {explanation}\nAnswer: {truth}")
        elif strat == "Post_Summary":
            # Extract just the first sentence of the explanation to simulate a brief summary
            summary_sentence = explanation.split('.')[0] + "." if '.' in explanation else explanation
            examples.append(f"{q_text}Answer: {truth}\nSummary: {summary_sentence}")
        elif strat == "Post_Confidence":
            # Inject a mock high confidence level before the reasoning to train the format
            examples.append(f"{q_text}Answer: {truth}\nConfidence: 95%.\nExplanation: {explanation}")
            
    return "\n\n".join(examples) + "\n\n"

def prepare_data():
    print("Loading MMLU-Pro dataset...")
    val_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="validation")
    test_ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    
    # SHUFFLE AND CAP: Ensure a mix of all subjects, limited to TARGET_TOTAL
    print(f"Shuffling and selecting a random subset of {TARGET_TOTAL} questions...")
    test_ds = test_ds.shuffle(seed=42).select(range(min(TARGET_TOTAL, len(test_ds))))
    
    # 1. Group validation items by category to make category-specific 3-shot prompts
    print("Building category-specific 3-shot pools...")
    val_by_category = {}
    for item in val_ds:
        cat = item['category']
        if cat not in val_by_category: 
            val_by_category[cat] = []
        if len(val_by_category[cat]) < 3: # Limit to 3 shots
            val_by_category[cat].append(item)

    # 2. Build and save files for each strategy
    print("\nWriting JSONL files for evaluation...")
    for strat in ACTIVE_STRATEGIES:
        output_file = f"{DATA_DIR}/prepared_mmlu_pro_fewshot_{strat}.jsonl"
        prepared_data = []
        
        for task in test_ds:
            cat = task['category']
            # Get the 3 shots for this specific category
            shots = val_by_category.get(cat, val_by_category[list(val_by_category.keys())[0]])
            
            few_shot_prefix = build_few_shot_prompt(shots, strat)
            q_text = format_question(task)
            full_user_prompt = f"{few_shot_prefix}{q_text}{SUFFIXES[strat]}"
            
            prepared_data.append({
                "truth": task['answer'],
                "category": cat,
                "full_user_prompt": full_user_prompt
            })
            
        with open(output_file, "w") as f:
            for r in prepared_data:
                f.write(json.dumps(r) + "\n")
        print(f"✅ Saved {len(prepared_data)} prompts to {output_file}")

if __name__ == "__main__":
    prepare_data()