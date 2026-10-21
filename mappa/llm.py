"""The one place that talks to an LLM provider. Everything else calls call_llm(system, user).

Switch provider with the MAPPA_PROVIDER env var ("groq" or "anthropic"), or pass a different
function into read_policy(). Each backend returns the same LLMResult, so nothing downstream
knows or cares which model answered.
"""

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

PROVIDER = os.environ.get("MAPPA_PROVIDER", "groq")

GROQ_MODEL = os.environ.get("MAPPA_GROQ_MODEL", "openai/gpt-oss-120b")
ANTHROPIC_MODEL = "claude-opus-5-5"

# USD per 1M tokens (input, output). Groq figures from console.groq.com/docs, October 2026.
# Reasoning tokens are billed as output tokens on both providers.
PRICES = {
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-20b": (0.075, 0.30),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),
}


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    model: str
    ok: bool = True  # False if the response was cut off or refused


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1_000_000


def call_groq(system: str, user: str) -> LLMResult:
    """Groq's chat endpoint (same request shape as OpenAI's), via plain HTTP: no SDK to explain.

    temperature=0 is accepted here, so it's set: the model picks its most likely token each step.
    That reduces run-to-run variation but doesn't remove it (batching on the server still changes
    results slightly), which is why every run's raw output is also saved to disk.
    """
    import requests

    body = {
        "model": GROQ_MODEL,
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": 0,
        # Groq checks the reply parses as a JSON object. Our own validation still runs after.
        "response_format": {"type": "json_object"},
        "max_completion_tokens": 8000,
    }
    if GROQ_MODEL.startswith("openai/gpt-oss"):
        # Extraction is reading, not puzzle-solving: keep (billed) reasoning short, and don't send it back.
        body |= {"reasoning_effort": "low", "include_reasoning": False}

    headers = {"Authorization": f"Bearer {os.environ['GROQ_API_KEY']}"}
    for attempt in range(6):
        resp = requests.post("https://api.groq.com/openai/v1/chat/completions",
                             json=body, headers=headers, timeout=120)
        # 429 = over a rate limit; 5xx = Groq-side hiccup. Short waits (the per-minute limit) are
        # worth sitting out. A long wait means the DAILY token cap is used up: stop loudly. Returning
        # a failed chunk instead would look like "the policy says nothing" and quietly sink recall.
        if resp.status_code == 429 or resp.status_code >= 500:
            wait = float(resp.headers.get("retry-after", 2 ** attempt))
            if wait > 120:
                raise RuntimeError(f"Groq limit hit, retry in {wait:.0f}s: {resp.text[:300]}")
            time.sleep(wait)
            continue
        break
    else:
        raise RuntimeError(f"Groq still failing after 6 attempts: {resp.status_code} {resp.text[:300]}")

    if resp.status_code != 200:
        # e.g. 400 "json_validate_failed": the model produced broken JSON. Count it as a failed attempt.
        return LLMResult("", 0, 0, GROQ_MODEL, ok=False)

    data = resp.json()
    choice = data["choices"][0]
    return LLMResult(
        text=choice["message"]["content"] or "",
        input_tokens=data["usage"]["prompt_tokens"],
        output_tokens=data["usage"]["completion_tokens"],
        model=GROQ_MODEL,
        ok=choice["finish_reason"] == "stop",
    )


def call_anthropic(system: str, user: str) -> LLMResult:
    """Claude via the Anthropic SDK. No temperature: current Claude models reject it (HTTP 400)."""
    import anthropic

    resp = anthropic.Anthropic().messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=16000,
        system=system,
        messages=[{"role": "user", "content": user}],
        output_config={"effort": "low"},
    )
    return LLMResult(
        text="".join(b.text for b in resp.content if b.type == "text"),
        input_tokens=resp.usage.input_tokens,
        output_tokens=resp.usage.output_tokens,
        model=ANTHROPIC_MODEL,
        ok=resp.stop_reason == "end_turn",
    )


def call_llm(system: str, user: str) -> LLMResult:
    return call_groq(system, user) if PROVIDER == "groq" else call_anthropic(system, user)


# ---------------------------------------------------------------- raw output log + replay
# The spec's "temperature 0" can't make runs identical, so reproducibility comes from keeping every
# raw reply. Each call is stored under a hash of exactly what was sent (prompt + excerpt): replaying
# a run gives the same rows with no API call, and re-scoring old runs after a code change is free.

LLM = Callable[[str, str], LLMResult]


def call_key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}\x00{user}".encode()).hexdigest()[:16]


def recording(llm: LLM, path: Path) -> LLM:
    """Wrap an LLM so every reply is appended to a JSONL file (one line per call)."""
    def wrapped(system: str, user: str) -> LLMResult:
        res = llm(system, user)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": call_key(system, user), **asdict(res)}) + "\n")
        return res
    return wrapped


def replay(path: Path) -> LLM:
    """An 'LLM' that answers from a recording. Fails if asked something that was never recorded,
    e.g. because the prompt changed since the recording was made."""
    saved: dict[str, list[LLMResult]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        saved.setdefault(rec.pop("key"), []).append(LLMResult(**rec))

    def llm(system: str, user: str) -> LLMResult:
        # A call that was retried has several replies; hand them out in the order they came.
        replies = saved.get(call_key(system, user))
        if not replies:
            raise KeyError("no recorded reply for this call: the prompt or policy changed, re-record it")
        return replies.pop(0) if len(replies) > 1 else replies[0]
    return llm
