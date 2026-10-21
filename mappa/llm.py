"""Talking to a language model that runs on your own machine.

Ollama, LM Studio, vLLM and llama.cpp all serve the same "OpenAI-compatible" HTTP API, so one small
function covers them all (a hosted service such as Groq works too: set MAPPA_LLM_API_KEY).

Asking for JSON: the best servers accept a JSON schema and force the reply to fit it ("json_schema"; llama.cpp's
server spells it json_object + schema). Others only promise valid JSON ("json_object"), and some only follow
instructions ("prompt"). We try them in that order, remember which one works, and check every reply ourselves anyway.

Every reply is saved under data/cache/llm, keyed by a hash of the exact request. Re-running an evaluation
replays saved answers instead of recomputing them, and the saved files are the record of what the model said.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

DEFAULT_URL = os.environ.get("MAPPA_LLM_URL", "http://localhost:11434/v1")  # Ollama's default address
DEFAULT_MODEL = os.environ.get("MAPPA_LLM_MODEL", "qwen2.5:7b-instruct")
API_KEY = os.environ.get("MAPPA_LLM_API_KEY")  # only for hosted endpoints
MODES = ("json_schema", "json_object_schema", "json_object", "prompt")
_working_mode: dict[tuple[str, str], int] = {}


class LLMError(RuntimeError):
    pass


def chat_json(messages: list[dict], schema: dict, *, model: str, base_url: str, seed: int = 0,
              max_tokens: int = 1500, rep: int = 0, cache_dir: Path | None = None, timeout: int = 300) -> dict:
    """Ask for a JSON reply. Returns {data, usage, seconds, mode, finish_reason, cached}; data is None if unparseable.

    rep is part of the cache key, so repeated runs (to measure stability) really are re-run.
    """
    base = {"model": model, "messages": messages, "temperature": 0, "seed": seed, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps({"request": base, "schema": schema, "rep": rep}, sort_keys=True).encode()).hexdigest()
    path = Path(cache_dir) / key[:2] / f"{key}.json" if cache_dir else None
    if path and path.exists():
        return {**json.loads(path.read_text()), "cached": True}

    first = _working_mode.get((base_url, model), 0)
    for mode in MODES[first:]:
        body = dict(base)
        if mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "claims", "schema": schema, "strict": True}}
        elif mode == "json_object_schema":
            body["response_format"] = {"type": "json_object", "schema": schema}
        else:
            body["messages"] = _with_schema(messages, schema)
            if mode == "json_object":
                body["response_format"] = {"type": "json_object"}
        t0 = time.monotonic()
        r = _post(f"{base_url.rstrip('/')}/chat/completions", body, timeout)
        seconds = time.monotonic() - t0
        if r.status_code in (400, 422) or (r.status_code >= 500 and "response_format" in r.text):
            continue  # this server does not support this way of asking for JSON (llama.cpp answers 500)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        reply = r.json()
        choice = reply["choices"][0]
        data = parse_json(choice["message"].get("content") or "")
        finish = choice.get("finish_reason")
        if data is None and mode != "prompt" and finish != "length":
            continue  # the server ignored the JSON request: try a looser mode
        _working_mode[(base_url, model)] = MODES.index(mode)
        usage = reply.get("usage") or {}
        result = {"data": data, "usage": {"prompt_tokens": usage.get("prompt_tokens", 0),
                                          "completion_tokens": usage.get("completion_tokens", 0)},
                  "seconds": seconds, "mode": mode, "finish_reason": finish, "cached": False}
        if path and data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return result
    raise LLMError("the server rejected every way of asking for JSON")


def _post(url: str, body: dict, timeout: int) -> requests.Response:
    """POST with retries: network errors and server errors twice, rate limits (hosted endpoints) up to 5 times."""
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    for attempt in range(6):
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            if attempt >= 2:
                raise LLMError(f"cannot reach {url}: {e}") from e
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 and attempt < 5:
            time.sleep(min(float(r.headers.get("retry-after") or 2 ** attempt), 120))
            continue
        if r.status_code >= 500 and attempt < 2 and "response_format" not in r.text:
            time.sleep(2 ** attempt)
            continue
        return r
    raise LLMError(f"gave up on {url}")


def _with_schema(messages: list[dict], schema: dict) -> list[dict]:
    """For servers that cannot enforce a schema: spell it out in the instructions instead."""
    note = "\n\nReply with JSON only, matching this JSON schema:\n" + json.dumps(schema)
    return [{**m, "content": m["content"] + note} if m["role"] == "system" else m for m in messages]


def parse_json(content: str) -> dict | None:
    """The JSON object in a reply, ignoring any <think>...</think> reasoning or text around it."""
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    try:
        data = json.loads(content)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(content[start:end + 1])
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        return None
