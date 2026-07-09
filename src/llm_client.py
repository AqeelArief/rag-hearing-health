"""
llm_client.py

Unified client for calling LLMs via Groq's free-tier API.

NOTE ON THIS VERSION: originally used Groq + Google Gemini for a
cross-provider comparison. Switched to two Groq-hosted models instead,
for two reasons:
  1. llama-3.1-8b-instant (the original Groq model used here) was
     deprecated by Groq on June 17, 2026.
  2. Gemini's free-tier daily quota turned out to be unreliable for this
     project's account (as low as 20 requests/day in practice, vs the
     ~1,000-1,500/day Google generally advertises) -- nowhere near enough
     for hundreds of experiment calls.
Running two differently-sized models on the same provider (Groq) still
gives a meaningful "does model choice affect accuracy" comparison, and
avoids depending on a second, less reliable free tier.

Environment variable required:
    GROQ_API_KEY

Get a free key at:
    https://console.groq.com/keys
"""

import os
import time
from groq import Groq

# --- Client setup ---------------------------------------------------------

_groq_client = None


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Run: export GROQ_API_KEY='your-key-here'"
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# --- Model name mapping ----------------------------------------------------
# These are the two "LLM" factor levels used in config.py's factorial grid.
# Both are current (non-deprecated) Groq-hosted models as of July 2026.
# Swap the underlying model strings here if Groq deprecates these later
# too -- check https://console.groq.com/docs/deprecations for the latest.

MODEL_MAP = {
    "gpt_oss_20b": "openai/gpt-oss-20b",     # smaller/faster model
    "gpt_oss_120b": "openai/gpt-oss-120b",   # larger/more capable model
}


def call_llm(model_key: str, prompt: str, max_retries: int = 3, temperature: float = 0.0) -> str:
    """
    Call an LLM by its config.py key ("gpt_oss_20b" or "gpt_oss_120b") and
    return the text response. Retries on transient errors / rate limits
    with backoff, since the free tier is more prone to 429s.
    """
    if model_key not in MODEL_MAP:
        raise ValueError(f"Unknown model_key '{model_key}'. Expected one of {list(MODEL_MAP)}")

    model_name = MODEL_MAP[model_key]

    for attempt in range(max_retries):
        try:
            return _call_groq(model_name, prompt, temperature)
        except Exception as e:
            wait = (attempt + 1) * 5
            print(f"[llm_client] Error calling {model_key} (attempt {attempt+1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                print(f"[llm_client] Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise

    raise RuntimeError("call_llm exhausted retries without returning or raising")


def _call_groq(model_name: str, prompt: str, temperature: float) -> str:
    client = _get_groq_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=512,
    )
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Quick manual test: python llm_client.py
    test_prompt = "In one sentence, what causes noise-induced hearing loss?"
    print("Testing gpt_oss_20b...")
    print(call_llm("gpt_oss_20b", test_prompt))
    print()
    print("Testing gpt_oss_120b...")
    print(call_llm("gpt_oss_120b", test_prompt))