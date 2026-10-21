"""Talking to Google's Gemini models.

Gemini serves the "OpenAI-compatible" HTTP API that most model services speak, so one small function does all
the talking (--base-url can still point it at any other OpenAI-compatible server).
The key comes from the GEMINI_API_KEY environment variable (create one at https://aistudio.google.com/apikey).

Asking for JSON: first we ask Gemini to force its reply to fit our JSON schema ("json_schema"). If a model refuses
that, we ask only for valid JSON ("json_object"), then fall back to plain instructions ("prompt"). Every reply is
checked by our own code either way.

Every reply is saved under data/cache/llm, keyed by a hash of the exact request. Re-running an evaluation
replays saved answers instead of paying for them again, and the saved files are the record of what the model said.
"""

import hashlib
import json
import os
import re
import time
from pathlib import Path

import requests

DEFAULT_URL = os.environ.get("MAPPA_LLM_URL", "https://generativelanguage.googleapis.com/v1beta/openai")
DEFAULT_MODEL = os.environ.get("MAPPA_LLM_MODEL", "gemini-3.8-flash")
# Gemini models think before answering, and that thinking counts against max_tokens. "low" keeps it short;
# set MAPPA_LLM_REASONING="" for models that do not think.
REASONING_EFFORT = os.environ.get("MAPPA_LLM_REASONING", "low")
MODES = ("json_schema", "json_object", "prompt")
# How a server says "I cannot do this way of asking for JSON" (as opposed to any other bad request).
UNSUPPORTED_FORMAT = re.compile(r"response_format|schema", re.I)
_working_mode: dict[tuple[str, str], int] = {}


class LLMError(RuntimeError):
    """This request failed; the next one may work (the window is counted as failed)."""


class LLMStop(RuntimeError):
    """Every request will fail the same way (bad key, unknown model, daily quota used up): stop the run."""


def api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("MAPPA_LLM_API_KEY")


def chat_json(messages: list[dict], schema: dict, *, model: str, base_url: str, max_tokens: int = 8192,
              rep: int = 0, cache_dir: Path | None = None, timeout: int = 300) -> dict:
    """Ask for a JSON reply. Returns {data, usage, seconds, mode, finish_reason, served_by, cached};
    data is None if the reply could not be parsed.

    rep is part of the cache key, so repeated runs (to measure stability) really are re-run.
    """
    base = {"model": model, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if REASONING_EFFORT:
        base["reasoning_effort"] = REASONING_EFFORT
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
        else:
            body["messages"] = _with_schema(messages, schema)
            if mode == "json_object":
                body["response_format"] = {"type": "json_object"}
        r, seconds = _post(f"{base_url.rstrip('/')}/chat/completions", body, timeout)
        if r.status_code in (400, 422) and (UNSUPPORTED_FORMAT.search(r.text)
                                            or (mode == "json_schema" and "API key" not in r.text)):
            # This model does not support this way of asking for JSON: try a looser one. Gemini rejects schema
            # features it lacks with a bare "invalid argument", so in schema mode any 400 but a key problem counts.
            continue
        if r.status_code in (400, 401, 403, 404):  # bad key, no access, unknown model: every request would fail
            raise LLMStop(f"{base_url} refused the request (HTTP {r.status_code}): {_error_message(r)}")
        if r.status_code != 200:
            raise LLMError(f"HTTP {r.status_code}: {_error_message(r)}")
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
            continue  # the model ignored the JSON request: try a looser mode
        _working_mode[(base_url, model)] = MODES.index(mode)
        result = {"data": data, "usage": {"prompt_tokens": usage.get("prompt_tokens") or 0,
                                          "completion_tokens": usage.get("completion_tokens") or 0},
                  "seconds": seconds, "mode": mode, "finish_reason": finish,
                  "served_by": reply.get("model") or model, "cached": False}
        if path and data is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result))
        return result
    raise LLMStop(f"{model} rejected every way of asking for JSON")


def _post(url: str, body: dict, timeout: int) -> tuple[requests.Response, float]:
    """POST with retries: connection errors twice, server errors 4 times, rate limits 6 times, waiting as long as
    the service asks. A daily quota that is used up stops the run (cached answers are kept; re-run tomorrow).
    Returns the response and the seconds the successful attempt took."""
    key = api_key()
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    for attempt in range(7):
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
        if r.status_code == 429:
            if re.search(r"per ?day|PerDay|daily", r.text, re.I):
                raise LLMStop(f"daily quota used up: {_error_message(r)}. Cached answers are kept; re-run later.")
            if attempt < 6:
                time.sleep(_retry_wait(r, attempt))
                continue
        if r.status_code >= 500 and attempt < 4:  # includes Gemini's "model under high demand" (503)
            time.sleep(2 ** (attempt + 1))
            continue
        return r, seconds
    raise LLMError(f"gave up on {url}")


def _retry_wait(r: requests.Response, attempt: int) -> float:
    """Seconds to wait after a rate limit: Gemini puts it in the body ("retryDelay": "37s")."""
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', r.text)
    try:
        wait = float(m.group(1)) if m else float(r.headers.get("retry-after") or 2 ** (attempt + 1))
    except ValueError:  # retry-after given as an HTTP date
        wait = 2 ** (attempt + 1)
    return min(wait + 1, 90)


def _error_message(r: requests.Response) -> str:
    try:
        err = r.json()
        err = err[0] if isinstance(err, list) else err
        return str(err.get("error", {}).get("message") or err)[:300]
    except (ValueError, AttributeError, IndexError):
        return r.text[:300]


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
