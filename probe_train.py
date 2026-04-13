#!/usr/bin/env python3
"""
Phase 1 (GPU Track): Linear Probe Training & Evaluation
========================================================

Trains linear probes on intermediate layer activations to predict
whether the model's answer is correct. This is the GO/NO-GO experiment:

  AUROC > 0.65 per layer → KSS framework has legs, proceed to Phase 2
  AUROC ≈ 0.5           → No signal in activations, rethink approach

Probe design:
  - Logistic regression (L2 regularized) — simplest possible probe
  - Training-free probe (Duan-style): mean activation difference as direction
  - Evaluates each layer separately to find the "correctness direction"
  - Compares factual vs reasoning datasets

Run on Modal:
  modal run probe_train.py
"""

import modal
import json
import os
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

# ── Modal Setup ────────────────────────────────────────────────────────────

app = modal.App("epistemic-probe-train")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "scikit-learn",
        "numpy",
    )
)

volume = modal.Volume.from_name("epistemic-model-cache", create_if_missing=True)
VOLUME_MOUNT = "/data"
RESULTS_DIR = "/data/results"


@app.cls(
    image=image,
    volumes={VOLUME_MOUNT: volume},
    timeout=1800,  # 30 min — loading big numpy arrays can be slow
    memory=16384,  # 16GB RAM for large numpy arrays
)
class ProbeTrainer:
    """Train and evaluate linear probes on extracted activations."""

    @modal.method()
    def train_probes(self) -> dict:
        """Load activations + metadata, train probes, evaluate."""

        # Load metadata
        meta_path = os.path.join(RESULTS_DIR, "probe_extract_results.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No extraction results found at {meta_path}. "
                f"Run probe_extract.py first."
            )

        with open(meta_path) as f:
            records = [json.loads(line) for line in f]

        print(f"Loaded {len(records)} records")

        # Find which layers we have (skip logits — 248K dims, too large for LR)
        act_dir = os.path.join(RESULTS_DIR, "activations")
        sample_files = [f for f in os.listdir(act_dir) if f.endswith(".npy")]
        layer_names = sorted(set(
            f.split("__")[1].replace(".npy", "") 
            for f in sample_files 
            if not f.endswith("__logits.npy")
        ))
        print(f"Found layers: {layer_names}")

        # Build arrays per layer
        results_by_layer = {}

        for layer_name in layer_names:
            print(f"\n{'='*50}")
            print(f"Processing {layer_name}")
            print(f"{'='*50}")

            # Load activations for this layer
            X_list = []
            y_list = []
            q_ids = []
            datasets = []

            for rec in records:
                qid = rec["question_id"]
                act_path = os.path.join(act_dir, f"{qid}__{layer_name}.npy")

                if not os.path.exists(act_path):
                    continue

                act = np.load(act_path)
                X_list.append(act)
                y_list.append(int(rec.get("correct", False)))
                q_ids.append(qid)
                datasets.append(rec["dataset"])

            if len(X_list) < 10:
                print(f"  Too few samples ({len(X_list)}), skipping")
                continue

            X = np.array(X_list)
            y = np.array(y_list)

            print(f"  Samples: {len(y)}, Correct: {y.sum()}, Wrong: {len(y)-y.sum()}")
            print(f"  Activation dim: {X.shape[1]}")

            # Skip if all same class
            if y.sum() < 3 or (len(y) - y.sum()) < 3:
                print(f"  Too few samples in one class, skipping")
                continue

            # ── Probe 1: Training-Free (Duan-style) ──
            # Compute mean activation for correct vs wrong
            # Use the difference as a probe direction
            mean_correct = X[y == 1].mean(axis=0)
            mean_wrong = X[y == 0].mean(axis=0)
            direction = mean_correct - mean_wrong
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Project all activations onto this direction
            projections = X @ direction

            # AUROC of this single direction
            try:
                auroc_training_free = roc_auc_score(y, projections)
                auprc_training_free = average_precision_score(y, projections)
            except ValueError:
                auroc_training_free = 0.5
                auprc_training_free = y.mean()

            print(f"  Training-free probe AUROC: {auroc_training_free:.3f}")
            print(f"  Training-free probe AUPRC: {auprc_training_free:.3f}")

            # ── Probe 2: Logistic Regression with Cross-Validation ──
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # 5-fold stratified CV
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

            # Logistic regression with L2 (default)
            lr = LogisticRegression(
                C=1.0,
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
            )

            cv_auroc = cross_val_score(lr, X_scaled, y, cv=cv, scoring="roc_auc")
            cv_auprc = cross_val_score(lr, X_scaled, y, cv=cv, scoring="average_precision")
            cv_acc = cross_val_score(lr, X_scaled, y, cv=cv, scoring="accuracy")

            print(f"  Logistic regression CV AUROC: {cv_auroc.mean():.3f} ± {cv_auroc.std():.3f}")
            print(f"  Logistic regression CV AUPRC:  {cv_auprc.mean():.3f} ± {cv_auprc.std():.3f}")
            print(f"  Logistic regression CV Acc:    {cv_acc.mean():.3f} ± {cv_acc.std():.3f}")

            # Train on full data for coefficient analysis
            lr_full = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
            lr_full.fit(X_scaled, y)

            # ── Probe 3: Separate factual vs reasoning ──
            factual_mask = np.array([d == "mmlu" for d in datasets])
            reasoning_mask = np.array([d == "gsm8k" for d in datasets])

            factual_results = {}
            reasoning_results = {}

            if factual_mask.sum() >= 10:
                X_fact = X_scaled[factual_mask]
                y_fact = y[factual_mask]
                if y_fact.sum() >= 3 and (len(y_fact) - y_fact.sum()) >= 3:
                    cv_fact = cross_val_score(lr, X_fact, y_fact, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc")
                    factual_results = {
                        "auroc_mean": float(cv_fact.mean()),
                        "auroc_std": float(cv_fact.std()),
                        "n_samples": int(factual_mask.sum()),
                        "n_correct": int(y_fact.sum()),
                    }
                    print(f"  Factual (MMLU) AUROC: {cv_fact.mean():.3f} ± {cv_fact.std():.3f}")

            if reasoning_mask.sum() >= 10:
                X_reas = X_scaled[reasoning_mask]
                y_reas = y[reasoning_mask]
                if y_reas.sum() >= 3 and (len(y_reas) - y_reas.sum()) >= 3:
                    cv_reas = cross_val_score(lr, X_reas, y_reas, cv=StratifiedKFold(3, shuffle=True, random_state=42), scoring="roc_auc")
                    reasoning_results = {
                        "auroc_mean": float(cv_reas.mean()),
                        "auroc_std": float(cv_reas.std()),
                        "n_samples": int(reasoning_mask.sum()),
                        "n_correct": int(y_reas.sum()),
                    }
                    print(f"  Reasoning (GSM8K) AUROC: {cv_reas.mean():.3f} ± {cv_reas.std():.3f}")

            results_by_layer[layer_name] = {
                "n_samples": int(len(y)),
                "n_correct": int(y.sum()),
                "n_wrong": int(len(y) - y.sum()),
                "activation_dim": int(X.shape[1]),
                "training_free": {
                    "auroc": float(auroc_training_free),
                    "auprc": float(auprc_training_free),
                },
                "logistic_regression": {
                    "auroc_mean": float(cv_auroc.mean()),
                    "auroc_std": float(cv_auroc.std()),
                    "auprc_mean": float(cv_auprc.mean()),
                    "auprc_std": float(cv_auprc.std()),
                    "acc_mean": float(cv_acc.mean()),
                    "acc_std": float(cv_acc.std()),
                },
                "factual_only": factual_results,
                "reasoning_only": reasoning_results,
            }

        # ── Summary: Best Layer & GO/NO-GO Decision ──
        print(f"\n{'='*60}")
        print(f"LAYER COMPARISON — AUROC by Layer")
        print(f"{'='*60}")

        best_layer = None
        best_auroc = 0.0

        for layer_name, lr in sorted(results_by_layer.items()):
            auroc = lr["logistic_regression"]["auroc_mean"]
            tf_auroc = lr["training_free"]["auroc"]
            marker = ""
            if auroc > best_auroc:
                best_auroc = auroc
                best_layer = layer_name
            if auroc > 0.65:
                marker = " ✓ SIGNAL DETECTED"
            elif auroc < 0.55:
                marker = " ✗ near random"
            print(f"  {layer_name}: LR={auroc:.3f}, TF={tf_auroc:.3f}{marker}")

        decision = "GO" if best_auroc > 0.65 else "NO-GO"
        print(f"\n{'='*60}")
        print(f"GO/NO-GO DECISION: {decision}")
        print(f"Best layer: {best_layer} (AUROC={best_auroc:.3f})")
        if decision == "GO":
            print(f"→ Intermediate activations contain correctness signal")
            print(f"→ Proceed to Phase 2: Steering from probe signals")
        else:
            print(f"→ No reliable correctness signal in activations")
            print(f"→ Consider: different model, different layers, nonlinear probes")
        print(f"{'='*60}")

        # Save full results
        output = {
            "model": records[0].get("model", "unknown") if records else "unknown",
            "decision": decision,
            "best_layer": best_layer,
            "best_auroc": float(best_auroc),
            "layers": results_by_layer,
        }

        with open(os.path.join(RESULTS_DIR, "probe_results.json"), "w") as f:
            json.dump(output, f, indent=2)

        volume.commit()
        return output


@app.local_entrypoint()
def main():
    """Train probes on extracted activations."""
    trainer = ProbeTrainer()
    results = trainer.train_probes.remote()

    print(f"\n{'='*60}")
    print(f"PROBE TRAINING RESULTS")
    print(f"{'='*60}")
    print(f"Decision: {results['decision']}")
    print(f"Best layer: {results['best_layer']} (AUROC={results['best_auroc']:.3f})")
    print(f"\nPer-layer breakdown:")
    for layer, data in sorted(results["layers"].items()):
        lr_auroc = data["logistic_regression"]["auroc_mean"]
        tf_auroc = data["training_free"]["auroc"]
        fact = data.get("factual_only", {})
        reas = data.get("reasoning_only", {})
        fact_str = f", Factual={fact['auroc_mean']:.3f}" if fact else ""
        reas_str = f", Reasoning={reas['auroc_mean']:.3f}" if reas else ""
        print(f"  {layer}: LR={lr_auroc:.3f}, TF={tf_auroc:.3f}{fact_str}{reas_str}")
