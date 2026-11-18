import json
import os
import re
from datasets import load_dataset

# --- CONFIGURATION ---
TARGET_COUNT = 3500
OUTPUT_DIR = os.path.join(os.environ.get("PR_WAREHOUSE", "warehouse"), "post_reflective_data")
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"numina_{TARGET_COUNT}_integer_master.jsonl")

def build_master_dataset():
    print("🚀 Downloading AI-MO/NuminaMath-CoT (859k rows)...")
    try:
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
    except Exception as e:
        print(f"❌ Failed to load dataset: {e}")
        return

    # --- 1. THE STRICT FILTER ---
    def is_computational(item):
        source = item.get('source', '').lower()
        q_text = item.get('problem', '').lower()
        solution = item.get('solution', '')
        
        # RULE 1: Reject Contaminated Sources
        banned_sources = ["aops_forum", "math", "gsm8k", "amc_aime"]
        if source in banned_sources:
            return False
            
        # RULE 2: Reject pure proofs
        bad_prompts = ["prove that", "show that", "demonstrate that"]
        if any(phrase in q_text for phrase in bad_prompts):
            return False
            
        # RULE 3: Strict Integer Filter
        # Must contain a boxed positive or negative integer.
        matches = re.findall(r'\\boxed\{\s*(-?\d+)\s*\}', solution)
        if not matches:
            return False
            
        return True

    print("🛡️ Applying strict integer filter (purging proofs, fractions, and text)...")
    filtered_ds = ds.filter(is_computational)
    
    # --- 2. SHUFFLE AND SELECT ---
    print(f"🔀 Shuffling and selecting top {TARGET_COUNT} questions...")
    # Seed 42 ensures you get the exact same dataset if you ever need to run this again
    final_ds = filtered_ds.shuffle(seed=42).select(range(TARGET_COUNT))

    # --- 3. FORMAT AND EXTRACT TO LOCAL DATA DICT ---
    print("📦 Extracting target values and formatting data...")
    clean_data = []
    
    for item in final_ds:
        q_text = item.get('problem', '').strip()
        messy_solution = item.get('solution', '').strip()
        source = item.get('source', 'unknown')

        # Grab all integer boxed matches in the solution
        matches = re.findall(r'\\boxed\{\s*(-?\d+)\s*\}', messy_solution)
        
        # Always take the very last one (avoids grabbing intermediate boxed steps)
        truth_value = matches[-1].strip() if matches else ""
        
        clean_data.append({
            "source": source,
            "question": q_text,
            "truth_value": truth_value,
            "reference_solution": messy_solution
        })

    # --- 4. SAVE TO DISK ---
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in clean_data:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print("\n" + "="*60)
    print("🎉 MASTER DATASET CREATION COMPLETE!")
    print(f"Saved {len(clean_data)} purely integer-based questions to disk.")
    print("="*60 + "\n")

if __name__ == "__main__":
    build_master_dataset()