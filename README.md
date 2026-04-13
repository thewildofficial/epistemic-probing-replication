# Epistemic Probing Replication: Correctness Signals in Qwen3.5-4B Intermediate Activations

> We replicate and extend the *No Answer Needed* (2025) finding that linear probes on intermediate LLM activations can predict answer correctness **before generation**. Confirmed on Qwen3.5-4B — but the same reasoning gap persists: the signal works for factual recall, not for mathematical reasoning.

## One-Sentence Summary

Linear probes on Qwen3.5-4B's intermediate activations predict factual answer correctness (AUROC 0.87) before the model generates anything — but fail on reasoning tasks (AUROC 0.64), replicating the *No Answer Needed* reasoning gap.

## Key Findings

| Finding | Result |
|---------|--------|
| Correctness signal exists in intermediate layers | ✅ Confirmed — AUROC up to 0.895 |
| Training-free probes nearly match logistic regression | ✅ Training-free peak: 0.877 vs LR peak: 0.895 |
| Signal strengthens with depth | ✅ Layer 5: 0.74 → Layer 30: 0.90 |
| Factual QA (MMLU) is predictable | ✅ Peak AUROC 0.87 at layers 30-31 |
| Reasoning (GSM8K) is predictable | ❌ Peak AUROC 0.64 — barely above chance |
| The reasoning gap is real | ✅ Confirmed — same wall as No Answer Needed (2025) |

## Results

### Full Results Table (5-fold Cross-Validated)

| Layer | LR AUROC | TF AUROC | Factual AUROC | Reasoning AUROC |
|-------|----------|----------|---------------|-----------------|
| 5     | 0.742 ± 0.023 | 0.779 | 0.655 ± 0.012 | 0.326 ± 0.068 |
| 10    | 0.761 ± 0.022 | 0.802 | 0.670 ± 0.013 | 0.362 ± 0.049 |
| 15    | 0.798 ± 0.034 | 0.791 | 0.721 ± 0.029 | 0.550 ± 0.095 |
| 20    | 0.876 ± 0.016 | 0.826 | 0.872 ± 0.014 | 0.539 ± 0.084 |
| 25    | 0.875 ± 0.015 | 0.877 | 0.856 ± 0.018 | 0.574 ± 0.105 |
| 30    | **0.895 ± 0.015** | 0.875 | 0.869 ± 0.018 | 0.621 ± 0.118 |
| 31    | 0.893 ± 0.010 | 0.875 | **0.870 ± 0.020** | **0.642 ± 0.089** |

- **LR** = Logistic Regression (L2, balanced, 5-fold CV)
- **TF** = Training-Free (mean activation difference direction, zero parameters)
- **Factual** = MMLU only (456 questions, 215 correct)
- **Reasoning** = GSM8K only (200 questions, 7 correct)

### Model Accuracy

| Dataset | Questions | Correct | Accuracy |
|---------|-----------|---------|----------|
| MMLU (factual) | 456 | 215 | 47.1% |
| GSM8K (reasoning) | 200 | 7 | 3.5% |
| **Total** | **656** | **222** | **33.8%** |

## Method

### Model
- **Qwen3.5-4B** — 32 transformer layers, 2560 hidden dimension, ~4B parameters
- Runs on a single NVIDIA T4 GPU (16GB VRAM) via [Modal](https://modal.com)
- Base model (not instruction-tuned) — minimal alignment interference with internal signals

### Datasets
- **MMLU** (cais/mmlu, "all" config, test split) — 57 subjects, sampled ~8 questions per subject = 456 questions
- **GSM8K** (openai/gsm8k, "main" config, test split) — 200 grade-school math problems

### Probe Design

**Two probe types:**

1. **Training-Free (Duan-style)** — Compute mean activation for correct answers, mean for wrong answers, take the difference as a "correctness direction." Project all activations onto this direction. Zero trainable parameters.
2. **Logistic Regression** — L2-regularized logistic regression with balanced class weights, 5-fold stratified cross-validation.

**Activation extraction:**
- Forward pass (prefill only) with `output_hidden_states=True`
- Extract last-token hidden state at layers [5, 10, 15, 20, 25, 30, 31]
- Separate generate pass for answer text (greedy decoding)

### Why Last Token at Prefill?

The last token position at the end of the prompt has attended to the full question via causal attention. Its hidden state encodes the model's "understanding" of the question *before* it commits to any output tokens.

## Comparison with No Answer Needed (2025)

| Aspect | No Answer Needed (2025) | This Work |
|--------|------------------------|-----------|
| Model | Llama 2 7B/13B/70B | Qwen3.5-4B |
| Factual probe AUROC | >0.7 (TriviaQA) | 0.65–0.87 (MMLU) |
| Reasoning probe AUROC | ~0.5 (GSM8K, fails) | 0.33–0.64 (GSM8K, weak) |
| Training-free probe | Not tested | 0.78–0.88 (competitive with LR) |
| Minimum training samples | 160 | Not tested (full 656 used) |
| Probe layer range | Layers 10–20 of 28 | Layers 5–31 of 32 |
| Causal steering | Not tested | Not tested (next step) |
| Model scale | 7B–70B | 4B (smallest tested) |

**Key replication:** The reasoning gap is confirmed. The signal that works for factual recall does not reliably transfer to mathematical reasoning on either model family.

## The Reasoning Gap: What We Know and Don't Know

**What we know:**
- Factual correctness is linearly separable in activation space (AUROC 0.87)
- Reasoning correctness is barely separable (AUROC 0.64)
- The gap persists across all layers and both probe types
- This is not a model-specific artifact — it appears in both Llama and Qwen architectures

**What we don't know:**
- Is the reasoning gap because uncertainty is distributed across token positions (not captured at last token)?
- Is it because the model hasn't *started* reasoning at prefill — the computation happens during generation?
- Is it because reasoning correctness is inherently nonlinear (you're either right or wrong, no smooth gradient)?
- Is the GSM8K result skewed by the 3.5% accuracy (only 7 correct samples — extreme class imbalance)?

**The open question:** Does the reasoning gap reflect a fundamental limitation of prefill probing, or is it an artifact of our extraction method?

## What's Next

This is a **live research project**. Next steps:

1. **Multi-position probing** — Extract activations from multiple token positions, not just the last. Reasoning uncertainty may be encoded across the question tokens.
2. **Base vs aligned comparison** — Run the same pipeline on Qwen3.5-4B-Instruct to test the hypothesis that RLHF degrades epistemic signal.
3. **Larger model** — Test on Qwen3.5-9B or 27B to see if the reasoning gap shrinks with scale.
4. **Causal steering** — Use the correctness direction to *steer* the model toward verification strategies when uncertain (the real goal of the KSS project).

## Repo Structure

```
├── README.md                  ← You are here
├── RESULTS.md                 ← Detailed per-layer numbers
├── probe_extract.py           ← Modal script: extract activations from Qwen3.5-4B
├── probe_train.py             ← Modal script: train linear probes on extracted activations
├── phase1_calibration.py      ← API-based calibration (logprobs, Fireworks)
├── phase1_analysis.py         ← Analysis of logprob calibration results
├── generate_plots.py          ← Plot generation script
├── plots/                     ← Publication-quality figures
│   ├── 01_auroc_by_layer.png
│   ├── 02_factual_vs_reasoning.png
│   ├── 03_tf_vs_lr_scatter.png
│   ├── 04_activation_space_schematic.png
│   └── 05_summary_dashboard.png
├── whiteboard_findings.excalidraw  ← Visual whiteboard of findings
├── results/                   ← Experimental results
│   ├── probe_results_full.json
│   ├── phase1/                ← Logprob calibration results
│   └── analysis/              ← Deep analysis outputs
└── experiment_design.md       ← Original experiment design document
```

## Requirements

- **Modal** account (for GPU access) — `pip install modal && modal setup`
- **HuggingFace** access (Qwen3.5-4B is public)
- Python 3.11+
- See `probe_extract.py` and `probe_train.py` for exact dependency lists

## Running

```bash
# Extract activations (needs Modal, ~20min on T4)
modal run probe_extract.py --n-mmlu 500 --n-gsm8k 200

# Train probes (CPU only, <1min)
modal run probe_train.py

# Generate plots
python generate_plots.py
```

## Citation

```bibtex
@misc{aban2026epistemic,
  title={Epistemic Probing Replication: Correctness Signals in Qwen3.5-4B Intermediate Activations},
  author={Aban H},
  year={2026},
  howpublished={\url{https://github.com/thewildofficial/epistemic-probing-replication}},
  note={Replication study of No Answer Needed (Duan et al., 2025)}
}
```

## Acknowledgments

- **No Answer Needed** (Duan et al., 2025) — the original finding this replicates
- **Kumaran et al. (2026)** — causal evidence for confidence direction
- **Lugoloobi et al. (2026)** — extending probing to math/coding tasks
- **Modal** — serverless GPU platform that made this experiment possible for under $5

## License

MIT
