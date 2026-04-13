#!/usr/bin/env python3
"""
Phase 1 (GPU Track): Activation Probing — Hidden State Extraction
================================================================

Loads Qwen3.5-4B on Modal T4 GPU, runs factual QA (MMLU) and
reasoning (GSM8K) questions through the model, extracts intermediate
layer activations, and saves them for probe training.

This is the GO/NO-GO experiment:
  - If linear probes on intermediate activations can predict correctness
    above chance (AUROC > 0.65), the KSS framework has legs.
  - If not, we need to rethink the approach.

Architecture:
  - Qwen3.5-4B: 40 layers (0-39), we extract from layers [5,10,15,20,25,30,35,39]
  - Activations: last-token hidden state at each target layer
  - Storage: numpy arrays + metadata JSONL
  - Datasets: MMLU (factual) + GSM8K (reasoning)

Run on Modal:
  modal run probe_extract.py --n-mmlu 100 --n-gsm8k 50
  modal run probe_extract.py --n-mmlu 500 --n-gsm8k 200   # full run
"""

import modal
import json
import os
import time
import argparse
import numpy as np
from pathlib import Path

# ── Modal Setup ────────────────────────────────────────────────────────────

app = modal.App("epistemic-probe-extract")

# Image with all dependencies — cached after first build
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.52.0",  # Qwen3.5 needs 4.52+
        "accelerate",
        "datasets",
        "safetensors",
        "scikit-learn",
        "numpy",
    )
)

# Persistent volume for model cache — avoids re-downloading 8GB every run
volume = modal.Volume.from_name("epistemic-model-cache", create_if_missing=True)
VOLUME_MOUNT = "/data"
MODEL_DIR = "/data/models"
RESULTS_DIR = "/data/results"

# Target layers to extract from Qwen3.5-4B (40 layers total, indices 0-39)
TARGET_LAYERS = [5, 10, 15, 20, 25, 30, 35, 39]

# Model choices — Qwen3.5-4B primary, Gemma-3-4b-it fallback
MODEL_ID = "Qwen/Qwen3.5-4B"
# MODEL_ID = "google/gemma-3-4b-it"  # fallback option


@app.cls(
    gpu="T4",
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=3600,  # 1 hour max
    memory=16384,  # 16GB RAM
)
class ProbeExtractor:
    """Extract intermediate activations from Qwen3.5-4B on factual + reasoning questions."""

    @modal.enter()
    def load_model(self):
        """Load model + tokenizer once at container start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_path = os.path.join(MODEL_DIR, MODEL_ID.replace("/", "_"))

        if os.path.exists(model_path):
            print(f"Loading cached model from {model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        else:
            print(f"Downloading {MODEL_ID} from HuggingFace...")
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            self.model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            # Cache for future runs
            self.tokenizer.save_pretrained(model_path)
            self.model.save_pretrained(model_path)
            volume.commit()
            print(f"Model cached to {model_path}")

        self.model.eval()
        self.device = next(self.model.parameters()).device

        # Determine model architecture
        # Different model families have different layer attribute names
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            n_layers = len(self.model.model.layers)
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'encoder'):
            n_layers = len(self.model.model.encoder.layers)
        else:
            # Fallback: count from config
            n_layers = getattr(self.model.config, 'num_hidden_layers', 40)
        print(f"Model loaded: {MODEL_ID}, {n_layers} layers, device={self.device}")

        # Adjust target layers if model has different depth
        self.target_layers = [l for l in TARGET_LAYERS if l < n_layers]
        if n_layers - 1 not in self.target_layers:
            self.target_layers.append(n_layers - 1)  # always include final layer

        print(f"Extracting from layers: {self.target_layers}")

    def _extract_activations_inner(self, questions: list[dict]) -> list[dict]:
        """
        Run questions through model, extract intermediate activations.

        Strategy: Two passes per question
          1. Forward pass (prefill only) → extract hidden states at each target layer
          2. Generate pass → get the model's answer for scoring

        We extract the LAST TOKEN hidden state from each intermediate layer
        during the PREFILL phase. This is the pre-generation signal — what the
        model "knows" before it commits to any output tokens.

        Returns list of dicts with metadata (activations saved as .npy files).
        """
        import torch

        results = []
        total = len(questions)
        act_dir = os.path.join(RESULTS_DIR, "activations")
        os.makedirs(act_dir, exist_ok=True)

        for i, q in enumerate(questions):
            if i % 10 == 0:
                print(f"  Processing {i}/{total}: {q['question_id']}")

            prompt = q["prompt"]
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            input_len = inputs["input_ids"].shape[1]

            # ── Pass 1: Prefill forward → extract hidden states ──
            with torch.no_grad():
                forward_outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                )

            # hidden_states: tuple of (1, seq_len, hidden_dim), one per layer
            # Take the LAST token position — it has attended to the full prompt
            last_idx = input_len - 1

            activations = {}
            for layer_idx in self.target_layers:
                hs = forward_outputs.hidden_states[layer_idx]
                act = hs[0, last_idx, :].cpu().float().numpy()
                activations[f"layer_{layer_idx}"] = act

            # Also grab final-layer logits for top-k answer probability
            logits = forward_outputs.logits[0, last_idx, :].cpu().float().numpy()
            top_token_id = int(logits.argmax())
            top_token_prob = float(torch.softmax(torch.tensor(logits), dim=0)[top_token_id])

            # ── Pass 2: Generate → get answer text for scoring ──
            with torch.no_grad():
                gen_outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=64,
                    do_sample=False,  # greedy decoding
                )

            generated_ids = gen_outputs[0][input_len:]
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Save activations as numpy arrays (separate from JSON metadata)
            for layer_name, act in activations.items():
                np.save(
                    os.path.join(act_dir, f"{q['question_id']}__{layer_name}.npy"),
                    act,
                )

            # Save logits (for logit-lens style analysis later)
            np.save(
                os.path.join(act_dir, f"{q['question_id']}__logits.npy"),
                logits,
            )

            result = {
                "question_id": q["question_id"],
                "dataset": q["dataset"],
                "subject": q.get("subject", ""),
                "uncertainty_type": q.get("uncertainty_type", ""),
                "prompt": prompt,
                "generated_text": generated_text,
                "correct_answer": q.get("correct_answer", ""),
                "top_token_id": top_token_id,
                "top_token_prob": top_token_prob,
                "input_len": input_len,
            }
            results.append(result)

            # Periodic volume commit (every 50 questions)
            if (i + 1) % 50 == 0:
                volume.commit()
                print(f"  [Checkpoint] Saved {i+1}/{total}, volume committed")

        volume.commit()
        return results

    @modal.method()
    def load_and_extract(self, n_mmlu: int = 100, n_gsm8k: int = 50) -> dict:
        """Load datasets and run full extraction pipeline. Supports resume."""
        from datasets import load_dataset
        import random

        random.seed(42)

        # ── Resume: check for already-processed question IDs ──
        existing_ids = set()
        meta_path = os.path.join(RESULTS_DIR, "probe_extract_results.jsonl")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                for line in f:
                    if line.strip():
                        rec = json.loads(line)
                        existing_ids.add(rec["question_id"])
            print(f"Resuming: {len(existing_ids)} questions already processed")

        questions = []

        # ── MMLU ──
        print(f"Loading MMLU (n={n_mmlu})...")
        mmlu = load_dataset("cais/mmlu", "all", split="test", trust_remote_code=True)

        # Sample evenly across subjects
        by_subject = {}
        for item in mmlu:
            subj = item.get("subject", "unknown")
            by_subject.setdefault(subj, []).append(item)

        subjects = sorted(by_subject.keys())
        n_per_subj = max(1, n_mmlu // len(subjects))
        mmlu_count = 0

        for subj in subjects:
            items = by_subject[subj]
            sampled = random.sample(items, min(n_per_subj, len(items)))
            for item in sampled:
                choices_str = "\n".join(
                    [f"{chr(65+i)}) {item['choices'][i]}" for i in range(4)]
                )
                prompt = (
                    f"Answer the following multiple choice question. "
                    f"Respond with ONLY the letter (A, B, C, or D). "
                    f"Do not explain your reasoning.\n\n"
                    f"Question: {item['question']}\n{choices_str}\n\nAnswer:"
                )
                questions.append({
                    "question_id": f"mmlu_{subj}_{hash(item['question']) % 10000}",
                    "dataset": "mmlu",
                    "subject": subj,
                    "uncertainty_type": "factual",
                    "prompt": prompt,
                    "correct_answer": chr(65 + item["answer"]),
                })
                mmlu_count += 1
                if mmlu_count >= n_mmlu:
                    break
            if mmlu_count >= n_mmlu:
                break

        print(f"  Loaded {mmlu_count} MMLU questions across {len(set(q['subject'] for q in questions if q['dataset']=='mmlu'))} subjects")

        # ── GSM8K ──
        print(f"Loading GSM8K (n={n_gsm8k})...")
        gsm8k = load_dataset("openai/gsm8k", "main", split="test", trust_remote_code=True)
        sampled_gsm = random.sample(list(gsm8k), min(n_gsm8k, len(gsm8k)))

        for item in sampled_gsm:
            prompt = (
                f"Solve this math problem. Show your work step by step, "
                f"then give the final answer as a number.\n\n"
                f"Question: {item['question']}\n\nAnswer:"
            )
            # Extract numeric answer from GSM8K format "#### 123"
            answer_text = item["answer"]
            if "####" in answer_text:
                answer_num = answer_text.split("####")[-1].strip()
            else:
                answer_num = answer_text.strip()

            questions.append({
                "question_id": f"gsm8k_{hash(item['question']) % 10000}",
                "dataset": "gsm8k",
                "subject": "math",
                "uncertainty_type": "reasoning",
                "prompt": prompt,
                "correct_answer": answer_num,
            })

        print(f"  Loaded {len(sampled_gsm)} GSM8K questions")
        print(f"Total: {len(questions)} questions")

        # Filter out already-processed questions (resume support)
        if existing_ids:
            before = len(questions)
            questions = [q for q in questions if q["question_id"] not in existing_ids]
            skipped = before - len(questions)
            print(f"  Skipped {skipped} already-processed questions, {len(questions)} remaining")

        if not questions:
            print("All questions already processed! Nothing to do.")
            # Still load summary for return value
            summary_path = os.path.join(RESULTS_DIR, "extraction_summary.json")
            if os.path.exists(summary_path):
                with open(summary_path) as f:
                    return json.load(f)
            return {"model": MODEL_ID, "n_questions": 0}

        # Run extraction (direct call, not remote — we're already on the GPU)
        results = self._extract_activations_inner(questions)

        # Score answers
        for r in results:
            if r["dataset"] == "mmlu":
                # First uppercase letter in generated text
                gen = r["generated_text"].strip()
                r["model_answer"] = gen[0].upper() if gen and gen[0].upper() in "ABCD" else "?"
                r["correct"] = r["model_answer"] == r["correct_answer"]
            elif r["dataset"] == "gsm8k":
                # Extract last number from generated text
                import re
                numbers = re.findall(r"[-+]?\d*\.?\d+", r["generated_text"])
                r["model_answer"] = numbers[-1] if numbers else "?"
                try:
                    r["correct"] = abs(float(r["model_answer"]) - float(r["correct_answer"])) / max(float(r["correct_answer"]), 1) < 0.01
                except (ValueError, ZeroDivisionError):
                    r["correct"] = False

        # Save metadata JSONL (no activations — those are .npy files)
        results_path = os.path.join(RESULTS_DIR, "probe_extract_results.jsonl")
        with open(results_path, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")

        # Summary stats
        n_correct = sum(1 for r in results if r.get("correct"))
        n_mmlu_correct = sum(1 for r in results if r.get("correct") and r["dataset"] == "mmlu")
        n_gsm8k_correct = sum(1 for r in results if r.get("correct") and r["dataset"] == "gsm8k")

        summary = {
            "model": MODEL_ID,
            "target_layers": self.target_layers,
            "n_questions": len(results),
            "n_mmlu": sum(1 for r in results if r["dataset"] == "mmlu"),
            "n_gsm8k": sum(1 for r in results if r["dataset"] == "gsm8k"),
            "n_correct": n_correct,
            "n_mmlu_correct": n_mmlu_correct,
            "n_gsm8k_correct": n_gsm8k_correct,
        }

        # Save summary
        with open(os.path.join(RESULTS_DIR, "extraction_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        volume.commit()
        print(f"\nExtraction complete: {summary}")
        return summary


@app.local_entrypoint()
def main(
    n_mmlu: int = 100,
    n_gsm8k: int = 50,
):
    """Run activation extraction on Modal GPU."""
    extractor = ProbeExtractor()
    summary = extractor.load_and_extract.remote(n_mmlu=n_mmlu, n_gsm8k=n_gsm8k)
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {summary['model']}")
    print(f"Questions: {summary['n_questions']} ({summary['n_mmlu']} MMLU + {summary['n_gsm8k']} GSM8K)")
    print(f"Accuracy: {summary['n_correct']}/{summary['n_questions']} ({100*summary['n_correct']/max(summary['n_questions'],1):.1f}%)")
    print(f"  MMLU: {summary['n_mmlu_correct']}/{summary['n_mmlu']} ({100*summary['n_mmlu_correct']/max(summary['n_mmlu'],1):.1f}%)")
    print(f"  GSM8K: {summary['n_gsm8k_correct']}/{summary['n_gsm8k']} ({100*summary['n_gsm8k_correct']/max(summary['n_gsm8k'],1):.1f}%)")
    print(f"Target layers: {summary['target_layers']}")
    print(f"Activations saved to volume 'epistemic-model-cache' at /results/activations/")
    print(f"\nNext step: Run probe_train.py to train linear probes on these activations")
