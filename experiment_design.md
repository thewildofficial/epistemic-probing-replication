# Epistemic Self-Awareness in LLMs — Experiment Design

## One-Sentence Contribution

Pre-generation uncertainty signals (Knowledge Sufficiency Scores) can steer LLM behavior toward better outcomes than naive generation, and this steering effect is strongest in the partial-knowledge zone (g ≈ 0.5) where the model knows enough to be dangerous.

## Core Hypotheses

### H1: The Partial-Knowledge Danger Zone
**Claim**: LLMs produce their highest rate of confidently wrong answers when their internal confidence is moderate (0.3-0.7), not when confidence is very low (<0.3) or very high (>0.7).
- **Null**: Error rate is independent of confidence bucket
- **Test**: Bin model outputs by self-reported confidence; measure factual accuracy per bucket; test for non-monotonic relationship

### H2: Steering Works Better Than Abstention
**Claim**: Using a pre-generation uncertainty signal to *steer* the model (ask clarifying questions, switch to chain-of-thought, generate alternatives) produces better outcomes than simply abstaining or answering naively.
- **Null**: Steering strategies don't outperform naive generation or abstention
- **Test**: Compare three conditions on the same question set: (A) naive generation, (B) abstention when uncertain, (C) dialectical steering when uncertain. Measure: accuracy on answered questions, coverage (fraction not abstained), overall utility = accuracy × coverage.

### H3: Dialectical Steering Outperforms Simple Steering
**Claim**: A dialectical steering strategy (generate thesis + antithesis + synthesis when uncertain) outperforms simple chain-of-thought or self-consistency checks.
- **Null**: Dialectical steering = simple CoT/self-consistency
- **Test**: Compare three steering strategies on questions where g < 0.7: (A) chain-of-thought, (B) self-consistency (N=5), (C) dialectical (thesis→antithesis→synthesis). Blind judge evaluation.

### H4: The Three-Type Uncertainty Typology
**Claim**: Uncertainty types (factual, reasoning, ambiguity) respond differently to steering strategies — factual uncertainty responds to retrieval, reasoning uncertainty to CoT, ambiguity to clarification questions.
- **Null**: All uncertainty types respond equally to all strategies
- **Test**: Classify uncertain questions by type. Apply each strategy to each type. Measure which strategy works best per type. Test for interaction effect.

## Experiment Design

### Dataset
We need questions where we can measure:
1. Model confidence (via logprobs or self-assessment)
2. Ground truth accuracy
3. Uncertainty type (factual/reasoning/ambiguity)

**Candidate datasets**:
- MMLU (factual, multiple choice — has ground truth)
- GSM8K (reasoning, math — has ground truth)
- TriviaQA (factual, open-ended)
- Natural Questions (factual + ambiguous)
- StrategyQA (requires reasoning about strategy — yes/no with reasoning)
- Our own curated set of deliberately ambiguous questions

**Minimum viable**: MMLU + GSM8K + a hand-curated ambiguity set (30-50 questions with multiple valid answers)

### Models
- Primary: Accessible via Fireworks API
  - `accounts/fireworks/models/mixtral-8x22b-instruct` — clean answers, logprobs work
  - `accounts/cogito/models/cogito-671b-v2-p1` — clean answers, logprobs work
  - `accounts/fireworks/models/glm-5p1` — verbose CoT, logprobs work (needs answer extraction)
  - `accounts/fireworks/models/deepseek-v3p2` — verbose CoT, logprobs work (needs answer extraction)
- NOTE: Models listed previously (Qwen2.5-72B, Llama-3.1-70B) DO NOT EXIST on Fireworks as of April 2026
- Logprobs confirmed working on all Fireworks text models (tested April 2026)
- If budget allows: Claude Haiku via API for judge panel

### Pipeline

```
Phase 1: Calibration
  For each question:
  1. Generate answer with logprobs
  2. Extract confidence score g(h_l(x)) approximation
  3. Record answer, confidence, correctness
  
Phase 2: Steering
  For questions where g < 0.7:
  1. Classify uncertainty type
  2. Apply steering strategy (type-matched vs generic vs none)
  3. Record steered answer, correctness
  
Phase 3: Evaluation
  1. Compare accuracy × coverage across conditions
  2. Blind judge panel for open-ended outputs
  3. Statistical significance tests (McNemar, bootstrap CI, Cohen's h)
```

### Compute Budget Estimate
- Phase 1 (calibration): ~500 questions × 3 models × ~500 tokens = ~750K tokens ≈ $1-3
- Phase 2 (steering): ~200 uncertain questions × 3 strategies × ~1500 tokens = ~900K tokens ≈ $2-5
- Phase 3 (judge panel): ~200 outputs × 3 judges × ~500 tokens = ~300K tokens ≈ $1-2
- **Total: ~$5-15** depending on model choice

### Timeline
- Day 1 (tonight): Set up experiment infrastructure, write scripts
- Day 1-2: Run Phase 1 (calibration) — overnight via background process
- Day 2: Analyze calibration results, set up Phase 2
- Day 2-3: Run Phase 2 (steering experiments) — overnight
- Day 3: Analyze results, run statistical tests
- Day 3-4: Draft experiment log and paper skeleton
- Day 5+: Iterate on paper, add ablations

## Key Open Questions
1. ~~Can we get logprobs from Fireworks API?~~ **RESOLVED: Yes, logprobs work on 7/8 models (April 2026)**
2. ~~Which models are available and affordable on Fireworks?~~ **RESOLVED: See Models section above**
3. Should we start with a smaller pilot (100 questions, 1 model) before committing to full runs? — **Yes, recommended. Start with Mixtral on 100 MMLU questions.**

## Venues
- ICLR 2027 (deadline ~Sept 2026) — best fit for understanding-focused contribution
- NeurIPS 2026 (deadline ~May 2026) — tighter, needs full results by then
- TMLR (rolling) — no deadline pressure, good for rigorous negative/partial results
