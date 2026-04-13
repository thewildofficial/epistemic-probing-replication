#!/usr/bin/env python3
"""Test API access and logprobs availability for experiment models."""

import os, json, sys

# Source .env manually
env_path = os.path.expanduser("~/.hermes/.env")
if os.path.exists(env_path):
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, _, val = line.partition('=')
                os.environ.setdefault(key.strip(), val.strip())

from openai import OpenAI

# Test GLM API
print("=== Testing GLM API ===")
glm_key = os.getenv("GLM_API_KEY", "")
if not glm_key:
    print("GLM_API_KEY not found")
else:
    client = OpenAI(api_key=glm_key, base_url="https://open.bigmodel.cn/api/paas/v4")
    try:
        resp = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=5
        )
        print(f"Response: {resp.choices[0].message.content}")
        print(f"Logprobs available: {resp.choices[0].logprobs is not None}")
        if resp.choices[0].logprobs and resp.choices[0].logprobs.content:
            for tok in resp.choices[0].logprobs.content[:3]:
                print(f"  Token: {tok.token}, LogProb: {tok.logprob:.4f}")
                if tok.top_logprobs:
                    for alt in tok.top_logprobs[:3]:
                        print(f"    Alt: {alt.token} ({alt.logprob:.4f})")
    except Exception as e:
        print(f"Error: {e}")

# Test KIMI API
print("\n=== Testing KIMI API ===")
kimi_key = os.getenv("KIMI_API_KEY", "")
if not kimi_key:
    print("KIMI_API_KEY not found")
else:
    client = OpenAI(api_key=kimi_key, base_url="https://api.moonshot.cn/v1")
    try:
        resp = client.chat.completions.create(
            model="moonshot-v1-8k",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
            logprobs=True,
            top_logprobs=5
        )
        print(f"Response: {resp.choices[0].message.content}")
        print(f"Logprobs available: {resp.choices[0].logprobs is not None}")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

# Test OpenRouter
print("\n=== Testing OpenRouter API ===")
or_key = os.getenv("OPENROUTER_API_KEY", "")
if not or_key:
    print("OPENROUTER_API_KEY not found")
else:
    client = OpenAI(api_key=or_key, base_url="https://openrouter.ai/api/v1")
    try:
        # Try a cheap model first
        resp = client.chat.completions.create(
            model="meta-llama/llama-3.1-8b-instruct:free",
            messages=[{"role": "user", "content": "What is 2+2? Answer with just the number."}],
            max_tokens=10,
        )
        print(f"Response: {resp.choices[0].message.content}")
        print(f"Model: meta-llama/llama-3.1-8b-instruct:free")
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

print("\n=== API Check Complete ===")
