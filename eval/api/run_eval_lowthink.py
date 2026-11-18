"""Low-thinking-level re-run of Direct_Answer vs Post_Reason for the API models.

Identical prompts, identical prepared data files, identical extractors/grading as
run_eval_all_base_{claude_haiku_45,claude_sonnet_46,gpt_54_mini}.py.
The ONLY difference is the reasoning/thinking setting:

  claude-haiku-4-5   thinking={"type":"enabled","budget_tokens":1024}   (min budget;
                                                Haiku 4.5 has no `effort` parameter)
  claude-sonnet-4-6  thinking={"type":"adaptive"} + output_config={"effort":"low"}
  gpt-5.4-mini       reasoning={"effort":"low"}

(baseline runs used thinking disabled / reasoning effort "none")

Outputs go to a *separate* model label so nothing overwrites the baseline runs:
  base_<model>-lowthink_<strat>_<bench>_fewshot_1.jsonl

Resumable: completed benchmark/strategy pairs are skipped; a partially finished
pair resumes from its .partial file.

Usage:
  python run_eval_lowthink.py --model claude-haiku-4-5 [--concurrency 30] [--limit N]
"""
import json, re, os, asyncio, random, argparse, time
import pandas as pd
from tqdm.asyncio import tqdm
from dotenv import load_dotenv

# ==========================================
# --- 1. CONFIGURATION ---
# ==========================================
load_dotenv(override=True)

BASE_DIR = os.environ.get("PR_WAREHOUSE", "warehouse")
RESULTS_DIR = os.getenv("PR_RESULTS_DIR", f"{BASE_DIR}/post_reasoning_results")
os.makedirs(RESULTS_DIR, exist_ok=True)
NUM_ITER = 1

ACTIVE_STRATEGIES = ["Direct_Answer", "Post_Reason"]

# (benchmark, subset) in ascending size order so small ones land first
TASK_ORDER = [
    ("easy2hard", "amc8"),      # 268
    ("easy2hard", "amc12"),     # 271
    ("easy2hard", "hmmt_nov"),  # 367
    ("easy2hard", "hmmt_feb"),  # 416
    ("easy2hard", "amc10"),     # 445
    ("gpqa",      "main"),      # 445
    ("gsm8k",     "main"),      # 1319
    ("mmlu_pro",  "main"),      # 3000
    ("math",      "main"),      # 3124
    ("bbh",       "main"),      # 6511
]

# --- 2. BENCHMARK-SPECIFIC SYSTEM PROMPTS (verbatim from the baseline scripts) ---
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

# --- 3. EXTRACTOR FUNCTIONS (baseline logic; the stray-quote typo in the
#     `ans_matches` regex of extract_bbh/extract_mmlu_pro is repaired so the
#     call does not raise TypeError) ---
first_answer_strats = ["Direct_Answer", "Post_Reason"]

def extract_bbh(text, strat):
    clean = text.replace("**", "")
    candidate = clean
    boxed_matches = re.findall(r'\\boxed\s*\{\s*(.*?)\s*\}', clean, re.IGNORECASE)
    if boxed_matches:
        candidate = boxed_matches[0].strip() if strat in first_answer_strats else boxed_matches[-1].strip()
    else:
        ans_matches = re.findall(r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$)", clean, re.IGNORECASE | re.DOTALL)
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
        ans_matches = re.findall(r"(?:Answer|Output|The final answer is|The answer is)\s*:?\s*(.*?)(?:\.\s|\n|Explanation:|Reasoning:|$)", candidate, re.IGNORECASE | re.DOTALL)
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
    "bbh": {"dir": "bbh_full_data", "prefix": "prepared_bbh_{data_strat}.jsonl", "extractor": extract_bbh, "file_mid": "bbh_all"},
    "easy2hard": {"dir": "easy2hard_data", "prefix": "prepared_{subset}_fewshot_{data_strat}.jsonl", "extractor": extract_e2h_gsm8k, "file_mid": None},
    "gpqa": {"dir": "gpqa_main_data", "prefix": "prepared_gpqa_fewshot_{data_strat}.jsonl", "extractor": extract_gpqa, "file_mid": "gpqa_main"},
    "gsm8k": {"dir": "gsm8k_data", "prefix": "prepared_gsm8k_fewshot_{data_strat}.jsonl", "extractor": extract_e2h_gsm8k, "file_mid": "gsm8k"},
    "math": {"dir": "math_data", "prefix": "prepared_math_fewshot_{data_strat}.jsonl", "extractor": extract_math, "file_mid": "math"},
    "mmlu_pro": {"dir": "mmlu_pro_data", "prefix": "prepared_mmlu_pro_fewshot_{data_strat}.jsonl", "extractor": extract_mmlu_pro, "file_mid": "mmlu_pro"},
}

# ==========================================
# --- 5. PROVIDER CALLS (the only real difference vs baseline) ---
# ==========================================
MODEL_CFG = {
    "claude-haiku-4-5":  {"provider": "anthropic", "max_tokens": 8192,
                          "think": {"type": "enabled", "budget_tokens": 1024}},
    "gpt-5.4-mini":      {"provider": "openai", "max_tokens": 8192, "effort": "low"},
    # gemini-2.5-flash does NOT accept thinking_level -- the live API returns
    # 400 "Thinking level is not supported for this model". thinking_level (low|medium|
    # high) exists only on the newer Gemini 3.x flash models (e.g. gemini-flash-latest),
    # which in turn reject the baseline's thinking_budget=0. So on 2.5-flash the low
    # setting has to be a small thinking_budget; 1024 matches Haiku 4.5's budget.
    # Sampling params kept verbatim from run_eval_all_base_gemini_25_flash.py.
    "gemini-2.5-flash":  {"provider": "gemini", "max_tokens": 8192, "budget": 1024,
                          "temperature": 0.7, "top_p": 0.8, "top_k": 20},
}

def make_client(model):
    cfg = MODEL_CFG[model]
    if cfg["provider"] == "anthropic":
        from anthropic import AsyncAnthropic
        return AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    if cfg["provider"] == "gemini":
        from google import genai
        return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def call_model(client, model, sys_msg, prompt):
    """Returns (visible_text, raw_json)."""
    cfg = MODEL_CFG[model]
    if cfg["provider"] == "anthropic":
        kwargs = dict(model=model, system=sys_msg,
                      messages=[{"role": "user", "content": prompt}],
                      max_tokens=cfg["max_tokens"], thinking=cfg["think"])
        if "effort" in cfg:
            kwargs["output_config"] = {"effort": cfg["effort"]}
        r = await client.messages.create(**kwargs)
        # thinking blocks are skipped: only the visible answer text is graded
        text = next((b.text for b in r.content if b.type == "text"), "")
        return text, r.model_dump_json()
    if cfg["provider"] == "gemini":
        from google.genai import types
        r = await client.aio.models.generate_content(
            model=model, contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=sys_msg,
                temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"],
                max_output_tokens=cfg["max_tokens"],
                thinking_config=types.ThinkingConfig(thinking_budget=cfg["budget"]),
            ))
        return (r.text or ""), r.model_dump_json()
    r = await client.responses.create(
        model=model,
        input=[{"role": "system", "content": sys_msg},
               {"role": "user", "content": prompt}],
        reasoning={"effort": cfg["effort"]},
        max_output_tokens=cfg["max_tokens"],
    )
    return (r.output_text or ""), r.model_dump_json()

MAX_RETRIES = 8

async def process_single_item(client, model, idx, item, bench, subset, sys_msg, strat,
                              extractor, semaphore, pbar, sink, lock):
    async with semaphore:
        prompt, truth = item['full_user_prompt'], str(item['truth']).strip()
        for attempt in range(MAX_RETRIES):
            try:
                text, raw = await call_model(client, model, sys_msg, prompt)
                pred = extractor(text, strat)
                rec = {"idx": idx, "benchmark": bench, "subset": subset, "truth": truth,
                       "pred": pred, "is_correct": evaluate_correctness(pred, truth, bench),
                       "response": text.replace("\n", " "), "raw_response": raw}
                async with lock:
                    sink.write(json.dumps(rec) + "\n"); sink.flush()
                pbar.update(1)
                return rec
            except Exception as e:
                es = str(e).lower()
                retryable = any(k in es for k in ["429", "500", "502", "503", "529", "rate limit",
                                                  "overloaded", "timeout", "connection", "quota"])
                if retryable and attempt < MAX_RETRIES - 1:
                    await asyncio.sleep((2 ** attempt) + random.uniform(0.1, 3.0))
                else:
                    pbar.update(1)
                    print(f"\n[FAILED] {bench}/{subset}/{strat} idx={idx}: {type(e).__name__}: {str(e)[:200]}")
                    return None

# ==========================================
# --- 6. MAIN ---
# ==========================================
async def run(model, concurrency, limit, only):
    label = f"{model}-lowthink"
    client = make_client(model)
    sem = asyncio.Semaphore(concurrency)
    lock = asyncio.Lock()
    summary_path = f"{RESULTS_DIR}/SUMMARY_{label}.txt"

    print("=" * 70)
    print(f"MODEL {model}   label={label}")
    print(f"thinking cfg: {MODEL_CFG[model]}")
    print(f"concurrency={concurrency}  limit={limit}")
    print("=" * 70, flush=True)

    for bench, subset in TASK_ORDER:
        if only and bench not in only and subset not in only:
            continue
        info = BENCHMARKS[bench]
        file_mid = info["file_mid"] or subset
        for strat in ACTIVE_STRATEGIES:
            out = f"{RESULTS_DIR}/base_{label}_{strat}_{file_mid}_fewshot_{NUM_ITER}.jsonl"
            part = out + ".partial"
            if os.path.exists(out):
                print(f"[skip done] {file_mid} {strat}", flush=True); continue

            infile = f"{BASE_DIR}/{info['dir']}/" + info["prefix"].format(subset=subset, data_strat=strat)
            if not os.path.exists(infile):
                print(f"[skip missing input] {infile}", flush=True); continue
            data = [json.loads(l) for l in open(infile) if l.strip()]
            if limit: data = data[:limit]

            done = {}
            if os.path.exists(part):
                for l in open(part):
                    try:
                        r = json.loads(l); done[r["idx"]] = r
                    except Exception: pass
            todo = [(i, d) for i, d in enumerate(data) if i not in done]
            print(f"\n--- {file_mid} | {strat} | {len(data)} items ({len(done)} cached, {len(todo)} to run) ---", flush=True)
            if not todo and not done: continue

            sys_msg = BENCHMARK_PROMPTS[bench][strat]
            t0 = time.time()
            with open(part, "a") as sink, tqdm(total=len(todo), desc=f"{file_mid[:12]}/{strat[:6]}") as pbar:
                tasks = [process_single_item(client, model, i, d, bench, subset, sys_msg, strat,
                                             info["extractor"], sem, pbar, sink, lock)
                         for i, d in todo]
                await asyncio.gather(*tasks)

            rows = list(done.values())
            if os.path.exists(part):
                seen = set()
                rows = []
                for l in open(part):
                    try:
                        r = json.loads(l)
                        if r["idx"] not in seen:
                            seen.add(r["idx"]); rows.append(r)
                    except Exception: pass
            if not rows: continue
            if len(rows) < len(data):
                # Do NOT finalize an incomplete pair -- a finalized file is the
                # "already done" marker and would be silently skipped on resume.
                # Keep the .partial so the next run backfills the missing items.
                print(f"⚠️  INCOMPLETE {file_mid} {strat}: {len(rows)}/{len(data)} rows. "
                      f"Keeping .partial, NOT finalizing. Re-run to backfill.", flush=True)
                continue
            rows.sort(key=lambda r: r["idx"])
            df = pd.DataFrame([{k: v for k, v in r.items() if k != "idx"} for r in rows])
            df.to_json(out, orient="records", lines=True)
            acc = df["is_correct"].mean() * 100
            msg = (f"{file_mid:10s} {strat:14s} acc={acc:6.2f}%  n={len(df)}/{len(data)}  "
                   f"{time.time()-t0:6.0f}s")
            print("🔥 " + msg, flush=True)
            with open(summary_path, "a") as f:
                f.write(msg + "\n")
            os.remove(part)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODEL_CFG))
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these benchmark/subset names")
    a = ap.parse_args()
    asyncio.run(run(a.model, a.concurrency, a.limit, a.only))
