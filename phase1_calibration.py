#!/usr/bin/env python3
"""
Phase 1: Calibration Pipeline for Epistemic Self-Awareness Research

For each question in MMLU/GSM8K:
1. Generate answer with logprobs
2. Extract confidence score g(x) from logprobs
3. Record answer, confidence, correctness

Outputs: JSONL file with per-question calibration data.
"""

import json, os, time, math, argparse, random
from pathlib import Path
from openai import OpenAI

# ── Config ──────────────────────────────────────────────────────────────────

FIREWORKS_KEY = "fw_5zi81nUgymrjk112Er1ay5"
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"
OUTPUT_DIR = Path(__file__).parent / "results" / "phase1"

MODELS = {
    "mixtral-8x22b": "accounts/fireworks/models/mixtral-8x22b-instruct",
    "cogito-671b": "accounts/cogito/models/cogito-671b-v2-p1",
    "deepseek-v3p2": "accounts/fireworks/models/deepseek-v3p2",
    "glm-5": "accounts/fireworks/models/glm-5",
}

# Only use models that give clean short answers for MMLU
CLEAN_ANSWER_MODELS = ["mixtral-8x22b", "cogito-671b"]

# ── Dataset Loading ──────────────────────────────────────────────────────────

def load_mmlu(subjects=None, n_per_subject=50, seed=42):
    """Load MMLU questions. Returns list of dicts with question, choices, answer, subject."""
    from datasets import load_dataset
    
    dataset = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)
    
    random.seed(seed)
    records = []
    
    # Group by subject
    by_subject = {}
    for item in dataset:
        subj = item.get("subject", "unknown")
        if subjects and subj not in subjects:
            continue
        by_subject.setdefault(subj, []).append(item)
    
    for subj, items in by_subject.items():
        sampled = random.sample(items, min(n_per_subject, len(items)))
        for item in sampled:
            choices = [item["choices"][i] for i in range(4)]
            records.append({
                "question_id": f"mmlu_{subj}_{hash(item['question']) % 10000}",
                "dataset": "mmlu",
                "subject": subj,
                "question": item["question"],
                "choices": choices,
                "answer_idx": item["answer"],
                "answer_letter": "ABCD"[item["answer"]],
                "answer_text": choices[item["answer"]],
                "uncertainty_type": "factual",
            })
    
    return records


def load_gsm8k(n=200, seed=42):
    """Load GSM8K math reasoning questions."""
    from datasets import load_dataset
    
    dataset = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
    random.seed(seed)
    sampled = random.sample(list(dataset), min(n, len(dataset)))
    
    records = []
    for i, item in enumerate(sampled):
        # Extract the numeric answer from the #### format
        answer_str = item["answer"].split("####")[-1].strip().replace(",", "")
        try:
            answer_num = float(answer_str)
        except ValueError:
            answer_num = answer_str
        
        records.append({
            "question_id": f"gsm8k_{i}",
            "dataset": "gsm8k",
            "subject": "math",
            "question": item["question"],
            "choices": None,
            "answer_idx": None,
            "answer_letter": None,
            "answer_text": str(answer_num),
            "uncertainty_type": "reasoning",
        })
    
    return records


# ── Prompt Templates ────────────────────────────────────────────────────────

MMLU_PROMPT = """Answer the following multiple choice question. Respond with ONLY the letter (A, B, C, or D). Do not explain your reasoning.

Question: {question}
A) {choice_a}
B) {choice_b}
C) {choice_c}
D) {choice_d}

Answer:"""

GSM8K_PROMPT = """Solve this math problem. Respond with ONLY the final number. Do not show your work or explain.

Problem: {question}

Answer:"""


# ── Confidence Extraction ───────────────────────────────────────────────────

def extract_confidence_from_logprobs(logprobs_content):
    """
    Extract confidence metrics from logprob data.
    
    Returns dict with:
    - mean_prob: average token probability across the answer
    - min_prob: minimum token probability (weakest link)
    - first_token_prob: probability of the first generated token
    - answer_token_probs: list of (token, prob) for answer tokens
    - entropy: total entropy across answer tokens
    - top_alt_probs: for each token, the probability mass on the top alternative
    """
    if not logprobs_content:
        return None
    
    token_probs = []
    entropies = []
    alt_mass = []
    
    for tok in logprobs_content:
        prob = math.exp(tok.logprob)
        token_probs.append(prob)
        
        # Calculate entropy for this token position
        if tok.top_logprobs:
            pos_entropy = -sum(
                math.exp(alt.logprob) * tok.logprob  # not quite right, fix below
                for alt in tok.top_logprobs
                if math.exp(alt.logprob) > 0
            )
            # Correct entropy calculation
            pos_entropy = -sum(
                math.exp(alt.logprob) * alt.logprob
                for alt in tok.top_logprobs
                if math.exp(alt.logprob) > 0
            )
            entropies.append(pos_entropy)
            
            # Probability mass on alternatives (1 - top prob)
            alt_mass.append(1.0 - prob)
        else:
            entropies.append(0)
            alt_mass.append(0)
    
    # Skip the first token if it's a BOS/special token with prob=1.0
    answer_tokens = []
    for tok in logprobs_content:
        answer_tokens.append((tok.token, math.exp(tok.logprob)))
    
    return {
        "mean_prob": sum(token_probs) / len(token_probs) if token_probs else 0,
        "min_prob": min(token_probs) if token_probs else 0,
        "first_token_prob": token_probs[0] if token_probs else 0,
        "answer_token_probs": answer_tokens[:20],  # cap at 20 for storage
        "total_entropy": sum(entropies),
        "mean_entropy": sum(entropies) / len(entropies) if entropies else 0,
        "max_entropy": max(entropies) if entropies else 0,
        "mean_alt_mass": sum(alt_mass) / len(alt_mass) if alt_mass else 0,
        "max_alt_mass": max(alt_mass) if alt_mass else 0,
        "n_tokens": len(token_probs),
    }


# ── Scoring ─────────────────────────────────────────────────────────────────

def score_mmlu(answer_text, correct_letter):
    """Score MMLU answer. Answer should be a single letter A-D."""
    answer = answer_text.strip().upper()
    # Take first character if longer
    if len(answer) >= 1:
        first_char = answer[0]
        return first_char == correct_letter.upper()
    return False

def score_gsm8k(answer_text, correct_answer):
    """Score GSM8K answer. Extract number and compare."""
    # Try to extract a number from the answer
    import re
    numbers = re.findall(r'-?\d+\.?\d*', answer_text.replace(',', ''))
    if not numbers:
        return False
    try:
        predicted = float(numbers[-1])  # Take last number (usually the final answer)
        correct = float(correct_answer.replace(',', ''))
        # Allow 1% tolerance for floating point
        return abs(predicted - correct) / max(abs(correct), 1) < 0.01
    except (ValueError, ZeroDivisionError):
        return False


# ── Main Pipeline ───────────────────────────────────────────────────────────

def run_calibration(model_name, model_id, questions, output_file, 
                    max_tokens=10, temperature=0.0, delay=0.5, resume=True):
    """
    Run calibration for one model on a set of questions.
    Supports resume — skips already-completed question IDs.
    """
    client = OpenAI(api_key=FIREWORKS_KEY, base_url=FIREWORKS_BASE)
    
    # Load existing results for resume
    completed_ids = set()
    if resume and output_file.exists():
        with open(output_file) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    completed_ids.add(rec["question_id"])
                except json.JSONDecodeError:
                    pass
    
    remaining = [q for q in questions if q["question_id"] not in completed_ids]
    print(f"[{model_name}] {len(completed_ids)} completed, {len(remaining)} remaining")
    
    errors = 0
    for i, q in enumerate(remaining):
        # Build prompt
        if q["dataset"] == "mmlu":
            prompt = MMLU_PROMPT.format(
                question=q["question"],
                choice_a=q["choices"][0],
                choice_b=q["choices"][1],
                choice_c=q["choices"][2],
                choice_d=q["choices"][3],
            )
        elif q["dataset"] == "gsm8k":
            prompt = GSM8K_PROMPT.format(question=q["question"])
        else:
            prompt = q["question"]
        
        try:
            resp = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                logprobs=True,
                top_logprobs=5,
                temperature=temperature,
            )
            
            answer_text = resp.choices[0].message.content or ""
            has_logprobs = resp.choices[0].logprobs is not None
            
            # Extract confidence
            confidence = None
            if has_logprobs and resp.choices[0].logprobs.content:
                confidence = extract_confidence_from_logprobs(
                    resp.choices[0].logprobs.content
                )
            
            # Score
            if q["dataset"] == "mmlu":
                correct = score_mmlu(answer_text, q["answer_letter"])
            elif q["dataset"] == "gsm8k":
                correct = score_gsm8k(answer_text, q["answer_text"])
            else:
                correct = None
            
            # Compute composite confidence score g(x)
            # g = mean_prob captures overall certainty
            # We also compute a "danger zone" indicator
            g_score = confidence["mean_prob"] if confidence else None
            in_danger_zone = (0.3 <= g_score <= 0.7) if g_score is not None else None
            
            record = {
                "question_id": q["question_id"],
                "dataset": q["dataset"],
                "subject": q.get("subject"),
                "uncertainty_type": q.get("uncertainty_type"),
                "model": model_name,
                "answer": answer_text[:200],
                "correct": correct,
                "g_score": g_score,
                "in_danger_zone": in_danger_zone,
                "confidence_detail": confidence,
                "prompt": prompt[:500],
            }
            
            # Append immediately (crash-safe)
            with open(output_file, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
            
            if (i + 1) % 20 == 0:
                print(f"  [{model_name}] {i+1}/{len(remaining)} done "
                      f"(g={g_score:.3f}, correct={correct})" if g_score 
                      else f"  [{model_name}] {i+1}/{len(remaining)} done")
            
        except Exception as e:
            errors += 1
            print(f"  [{model_name}] Error on {q['question_id']}: {type(e).__name__}: {str(e)[:100]}")
            if errors > 10:
                print(f"  [{model_name}] Too many errors, stopping.")
                break
            time.sleep(2)  # Back off on error
        
        time.sleep(delay)  # Rate limit
    
    total = len(completed_ids) + len(remaining)
    print(f"[{model_name}] Complete: {total} questions, {errors} errors")
    return output_file


# ── Entry Point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Calibration Pipeline")
    parser.add_argument("--models", nargs="+", default=["mixtral-8x22b", "cogito-671b"],
                       help="Models to test")
    parser.add_argument("--n-mmlu", type=int, default=50, 
                       help="Questions per MMLU subject")
    parser.add_argument("--n-gsm8k", type=int, default=200,
                       help="Number of GSM8K questions")
    parser.add_argument("--subjects", nargs="+", default=None,
                       help="MMLU subjects to include (default: all)")
    parser.add_argument("--max-tokens", type=int, default=10,
                       help="Max tokens for model response")
    parser.add_argument("--delay", type=float, default=0.5,
                       help="Delay between API calls (seconds)")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh instead of resuming")
    parser.add_argument("--pilot", action="store_true",
                       help="Small pilot: 10 per subject, 50 GSM8K")
    args = parser.parse_args()
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    print("Loading datasets...")
    n_per_subject = 10 if args.pilot else args.n_mmlu
    n_gsm8k = 50 if args.pilot else args.n_gsm8k
    
    mmlu_questions = load_mmlu(subjects=args.subjects, n_per_subject=n_per_subject)
    gsm8k_questions = load_gsm8k(n=n_gsm8k)
    all_questions = mmlu_questions + gsm8k_questions
    
    print(f"Loaded {len(mmlu_questions)} MMLU + {len(gsm8k_questions)} GSM8K = {len(all_questions)} questions")
    
    # Run calibration for each model
    for model_name in args.models:
        if model_name not in MODELS:
            print(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
            continue
        
        model_id = MODELS[model_name]
        output_file = OUTPUT_DIR / f"calibration_{model_name}.jsonl"
        
        print(f"\n{'='*60}")
        print(f"Running calibration: {model_name} ({model_id})")
        print(f"Output: {output_file}")
        print(f"{'='*60}")
        
        run_calibration(
            model_name=model_name,
            model_id=model_id,
            questions=all_questions,
            output_file=output_file,
            max_tokens=args.max_tokens,
            delay=args.delay,
            resume=not args.no_resume,
        )
    
    print("\n✓ Phase 1 calibration complete!")


if __name__ == "__main__":
    main()
