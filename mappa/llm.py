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
              max_tokens: int = 1500, rep: int = 0, cache_dir: Path | None = None, timeout: int = 600) -> dict:
    """Ask for a JSON reply. Returns {data, usage, seconds, mode, finish_reason, served_by, cached};
    data is None if the reply could not be parsed. Any failure to get a reply raises LLMError.

    rep is part of the cache key, so repeated runs (to measure stability) really are re-run.
    """
    base = {"model": model, "messages": messages, "temperature": 0, "seed": seed, "max_tokens": max_tokens}
    key = hashlib.sha256(json.dumps({"request": base, "schema": schema, "rep": rep, "server": base_url},
                                    sort_keys=True).encode()).hexdigest()
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
        r, seconds = _post(f"{base_url.rstrip('/')}/chat/completions", body, timeout)
        if r.status_code in (400, 422) or (r.status_code >= 500 and "response_format" in r.text):
            continue  # this server does not support this way of asking for JSON (llama.cpp answers 500)
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
        try:
            reply = r.json()
            choice = reply["choices"][0]
            content = choice["message"].get("content") or ""
            finish = choice.get("finish_reason")
            usage = reply.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"unexpected reply: {r.text[:300]}") from e
        data = parse_json(content)
        if data is None and mode != "prompt" and finish != "length":
            continue  # the server ignored the JSON request: try a looser mode
        _working_mode[(base_url, model)] = MODES.index(mode)
        result = {"data": data, "usage": {"prompt_tokens": usage.get("prompt_tokens") or 0,
                                          "completion_tokens": usage.get("completion_tokens") or 0},
                  "seconds": seconds, "mode": mode, "finish_reason": finish,
                  "served_by": reply.get("model") or model, "cached": False}
        if path and data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return result
    raise LLMError("the server rejected every way of asking for JSON")


def _post(url: str, body: dict, timeout: int) -> tuple[requests.Response, float]:
    """POST with retries: connection errors and server errors twice, rate limits (hosted endpoints) up to 5 times.
    A timeout is not retried: a single-stream local server would just redo the same long generation.
    Returns the response and the seconds the successful attempt took."""
    headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}
    for attempt in range(6):
        t0 = time.monotonic()
        try:
            r = requests.post(url, json=body, headers=headers, timeout=timeout)
        except requests.Timeout as e:
            raise LLMError(f"no reply from {url} within {timeout}s") from e
        except requests.RequestException as e:
            if attempt >= 2:
                raise LLMError(f"cannot reach {url}: {e}") from e
            time.sleep(2 ** attempt)
            continue
        seconds = time.monotonic() - t0
        if r.status_code == 429 and attempt < 5:
            try:
                wait = float(r.headers.get("retry-after") or 2 ** attempt)
            except ValueError:  # an HTTP date instead of seconds
                wait = 2 ** attempt
            time.sleep(min(wait, 120))
            continue
        if r.status_code >= 500 and attempt < 2 and "response_format" not in r.text:
            time.sleep(2 ** attempt)
            continue
        return r, seconds
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
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for n, m in enumerate(re.finditer(r"\{", content)):  # the first complete object, wherever it starts
        try:
            data, _ = decoder.raw_decode(content, m.start())
        except json.JSONDecodeError:
            if n == 0 and (salvaged := _salvage_claims(content)):
                return salvaged  # the reply itself was cut off: keep what was completed
            continue
        if isinstance(data, dict):
            return data
    return None


def _salvage_claims(content: str) -> dict | None:
    """A reply cut off mid-way (the model hit its length limit): keep every claim that was completed."""
    m = re.search(r'"claims"\s*:\s*\[', content)
    if not m:
        return None
    decoder, pos, claims = json.JSONDecoder(), m.end(), []
    while True:
        while pos < len(content) and content[pos] in " \n\r\t,":
            pos += 1
        try:
            item, pos = decoder.raw_decode(content, pos)
        except json.JSONDecodeError:
            break
        claims.append(item)
    return {"claims": claims} if claims else None
