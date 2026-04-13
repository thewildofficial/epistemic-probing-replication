#!/usr/bin/env python3
"""Test Fireworks API — models, logprobs, and confidence estimation."""

from openai import OpenAI
import json, os

FIREWORKS_KEY = "fw_5zi81nUgymrjk112Er1ay5"
FIREWORKS_BASE = "https://api.fireworks.ai/inference/v1"

client = OpenAI(api_key=FIREWORKS_KEY, base_url=FIREWORKS_BASE)

# List available models
print("=== Available Models (sampling) ===")
# We know glm-5p1 works — let's test logprobs on it and other models

models_to_test = [
    "accounts/fireworks/models/glm-5p1",
    "accounts/fireworks/models/llama-v3p1-70b-instruct",
    "accounts/fireworks/models/qwen2p5-72b-instruct",
    "accounts/fireworks/models/mixtral-8x7b-instruct",
]

test_msg = [{"role": "user", "content": "What is the capital of France? Answer with just the city name."}]

for model in models_to_test:
    print(f"\n--- Testing {model} ---")
    try:
        # Test with logprobs
        resp = client.chat.completions.create(
            model=model,
            messages=test_msg,
            max_tokens=10,
            logprobs=True,
            top_logprobs=5,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        print(f"  Response: {content}")
        print(f"  Logprobs: {resp.choices[0].logprobs is not None}")
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            for tok in resp.choices[0].logprobs.content[:5]:
                prob = 2.718 ** tok.logprob  # Convert logprob to probability
                print(f"  Token: '{tok.token}' prob={prob:.4f} logprob={tok.logprob:.4f}")
        else:
            print("  No logprob content returned")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {str(e)[:200]}")

# Test a knowledge question where the model might be uncertain
print("\n\n=== Testing Uncertainty Detection ===")
uncertain_questions = [
    "What year was the city of Timbuktu founded?",  # Factual uncertainty
    "Is it better to specialize or generalize in your career?",  # Ambiguity
    "If a train travels at 60 mph and another at 40 mph towards each other from 200 miles apart, when do they meet?",  # Reasoning
]

model = "accounts/fireworks/models/glm-5p1"
for q in uncertain_questions:
    print(f"\nQ: {q}")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": q}],
            max_tokens=100,
            logprobs=True,
            top_logprobs=5,
            temperature=0.0,
        )
        content = resp.choices[0].message.content
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            # Calculate average token probability
            probs = [2.718 ** t.logprob for t in resp.choices[0].logprobs.content]
            avg_prob = sum(probs) / len(probs) if probs else 0
            min_prob = min(probs) if probs else 0
            print(f"  A: {content[:100]}")
            print(f"  Avg token prob: {avg_prob:.4f}, Min: {min_prob:.4f}, Tokens: {len(probs)}")
        else:
            print(f"  A: {content[:100]} (no logprobs)")
    except Exception as e:
        print(f"  Error: {e}")

print("\n=== Done ===")
