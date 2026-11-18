import os
import json, os, random
from datasets import load_dataset

# --- 1. CONFIGURATION ---
DATA_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "gpqa_main_data")
os.makedirs(DATA_DIR, exist_ok=True)

ACTIVE_STRATEGIES = [
    # "Direct_Answer", 
    # "Basic_CoT", 
    # "Post_Reason",
    "Post_Summary",
    "Post_Confidence"
]

SUFFIXES = {
    "Direct_Answer": "\nWhich option is correct? Provide only the letter as your response without any explanation. Output format 'Answer: [Letter].'",
    "Basic_CoT": "\nLet's think step by step concisely. Which option is correct? Output format 'Explanation: [reasoning]. Answer: [Letter].'",
    "Post_Reason": "\nWhich option is correct? Think hard without outputting any explanation. State the final answer immediately and justify your answer. Output format: 'Answer: [Letter]. Explanation: [reasoning]'",
    "Post_Summary": "\nWhich option is correct? State the final answer immediately, then briefly summarize the question and your answer in a single sentence. Output format: 'Answer: [Letter]. Summary: [summary]'",
    "Post_Confidence": "\nWhich option is correct? State the final answer immediately, then state your confidence level (0-100%) in this answer and briefly explain why. Output format: 'Answer: [Letter]. Confidence: [X%]. Explanation: [reasoning]'"
}

# --- 2. GPQA FORMATTER ---
def format_gpqa_item(item, seed_val):
    """Deterministically shuffles options and returns the prompt, truth, and expert explanation."""
    random.seed(seed_val)
    
    correct_text = item['Correct Answer']
    distractors = [item['Incorrect Answer 1'], item['Incorrect Answer 2'], item['Incorrect Answer 3']]
    
    all_opts = [{"text": correct_text, "is_correct": True}]
    for d in distractors:
        all_opts.append({"text": d, "is_correct": False})
    
    random.shuffle(all_opts)
    
    labels = ["A", "B", "C", "D"]
    options_str = ""
    truth_letter = ""
    
    for label, opt in zip(labels, all_opts):
        options_str += f"{label}) {opt['text']}\n"
        if opt['is_correct']:
            truth_letter = label
            
    q_text = f"{item['Question']}\n\nOptions:\n{options_str}"
    
    # Clean up the official explanation (sometimes it contains messy HTML/newlines)
    explanation = item.get('Explanation', "Based on standard scientific principles.").strip()
    
    return q_text.strip(), truth_letter, explanation

# --- 3. DYNAMIC FEW-SHOT BUILDER ---
def build_few_shot_prompt(few_shot_items, strat):
    formatted_examples = []
    
    for i, item in enumerate(few_shot_items):
        q_text, ans, exp = format_gpqa_item(item, seed_val=42+i)
        
        if strat == "Direct_Answer":
            formatted_examples.append(f"Q: {q_text}\nAnswer: {ans}.")
        elif strat == "Basic_CoT":
            formatted_examples.append(f"Q: {q_text}\nA: Let's think step by step. {exp}. Answer: {ans}.")
        elif strat == "Post_Reason":
            formatted_examples.append(f"Q: {q_text}\nAnswer: {ans}. Explanation: {exp}")
        elif strat == "Post_Summary":
            # Extract just the first sentence of the explanation to simulate a brief summary
            summary_sentence = exp.split('.')[0] + "." if '.' in exp else exp
            formatted_examples.append(f"Q: {q_text}\nAnswer: {ans}. Summary: {summary_sentence}")
        elif strat == "Post_Confidence":
            # Inject a mock high confidence level before the reasoning to train the format
            formatted_examples.append(f"Q: {q_text}\nAnswer: {ans}. Confidence: 95%. Explanation: {exp}")
            
    return "\n\n".join(formatted_examples) + "\n\n"

# --- 4. MAIN LOOP ---
def prepare_data():
    print("Loading GPQA Main Dataset...")
    ds = load_dataset("idavidrein/gpqa", "gpqa_main", split="train")
    
    # 1. Isolate the first 3 items for Few-Shot mapping
    few_shot_items = [ds[i] for i in range(3)]
    
    # 2. Isolate the rest of the dataset for Testing
    test_items = [ds[i] for i in range(3, len(ds))]
    print(f"Isolated 3 Few-Shot examples. Remaining test items: {len(test_items)}")

    for strat in ACTIVE_STRATEGIES:
        print(f"Preparing data for strategy: {strat}...")
        output_file = f"{DATA_DIR}/prepared_gpqa_fewshot_{strat}.jsonl"
        prepared_data = []

        few_shot_prefix = build_few_shot_prompt(few_shot_items, strat)

        for i, item in enumerate(test_items):
            # Use the item index + 100 as the seed to guarantee deterministic shuffling for the test set
            q_text, truth, _ = format_gpqa_item(item, seed_val=100+i)
            
            full_user_prompt = f"{few_shot_prefix}Q: {q_text}{SUFFIXES[strat]}"
            
            prepared_data.append({
                "truth": truth,
                "full_user_prompt": full_user_prompt
            })

        with open(output_file, "w") as f:
            for r in prepared_data: 
                f.write(json.dumps(r) + "\n")
                
        print(f"Saved {len(prepared_data)} fully assembled prompts to {output_file}")

if __name__ == "__main__":
    prepare_data()