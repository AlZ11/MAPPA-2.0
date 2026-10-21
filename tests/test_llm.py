"""The local-LLM path, tested against a fake server: no model is needed to check what the code does with replies."""

import json
import re

import pytest

from mappa import llm
from mappa.policy_reader import LLM

POLICY = ("Information we collect\nWe collect your heart rate and sleep data to personalise your plan.\n"
          "We share your email address with Mailchimp, our newsletter provider.\nWe never sell your personal data.")


class FakeResponse:
    def __init__(self, status, payload=None, content=None, finish="stop"):
        self.status_code, self.headers = status, {}
        self.text = json.dumps(payload) if payload else ""
        self._payload = payload or {"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                    "usage": {"prompt_tokens": 900, "completion_tokens": 50}}

    def json(self):
        return self._payload


@pytest.fixture
def server(monkeypatch):
    """A fake OpenAI-compatible server. Set server.replies to a function of (task, request body)."""
    calls = []

    class Server:
        replies = None

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        task = "flows" if "turn what it says about personal data" in json["messages"][0]["content"] else "practices"
        return Server.replies(task, json)

    monkeypatch.setattr(llm.requests, "post", post)
    monkeypatch.setattr(llm, "_working_mode", {})
    Server.calls = calls
    return Server


def reply(claims):
    return FakeResponse(200, content=json.dumps({"claims": claims}))


FLOW = {"quote": "We share your email address with Mailchimp, our newsletter provider.", "category": "Personal info",
        "subtype": "Email address", "action": "share", "negated": False, "recipient": "service_provider",
        "recipient_name": "Mailchimp", "purposes": ["Developer communications"]}


def test_valid_claims_kept_invented_ones_dropped(server):
    made_up = {**FLOW, "quote": "We sell your email address to advertisers."}
    bad_enum = {**FLOW, "category": "Emails"}
    wrong_name = {**FLOW, "quote": "We collect your heart rate and sleep data to personalise your plan", "action": "collect",
                  "category": "Health and fitness", "subtype": "Precise location", "recipient": "none",
                  "recipient_name": "Fitbit", "purposes": ["Personalization"]}
    denial = {"quote": "we never sell your personal data", "category": "Generic personal data", "subtype": "unspecified",
              "action": "sell", "negated": True, "recipient": "unspecified", "recipient_name": "", "purposes": []}
    server.replies = lambda task, body: reply([FLOW, made_up, bad_enum, wrong_name, denial] if task == "flows" else [])
    spans = [m.span() for m in re.finditer(r"[^\n]+", POLICY)]  # one segment per line
    claims, stats = LLM(model="m", base_url="http://x").extract(POLICY, spans)

    assert stats["dropped"] == {"not_found": 1, "schema": 1, "subtype_reset": 1, "detail_not_in_quote": 1}
    share = next(c for c in claims if c.action == "share")
    assert share.seg == 2 and share.detail == "Mailchimp" and share.recipient == "service_provider"
    assert POLICY[share.start:share.end] == share.quote
    health = next(c for c in claims if c.category == "Health and fitness")
    assert health.subtype is None and health.detail is None and health.recipient is None  # fixed, not dropped
    sell = next(c for c in claims if c.action == "sell")
    assert sell.polarity == "deny" and sell.quote == "We never sell your personal data"  # the policy's own casing
    assert stats["kept"] == 3 and stats["windows"] == 2 and stats["prompt_tokens"] == 1800


def test_unparseable_reply_is_a_failed_window(server):
    server.replies = lambda task, body: FakeResponse(200, content="Sorry, I cannot help with that.")
    claims, stats = LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert claims == [] and stats["failed_windows"] == 1 and stats["dropped"] == {"unparseable": 1}


def test_falls_back_when_server_rejects_json_schema(server):
    def replies(task, body):
        if body.get("response_format", {}).get("type") == "json_schema":
            return FakeResponse(400, {"error": "response_format json_schema not supported"})
        return reply([])
    server.replies = replies
    LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert [c.get("response_format", {}).get("type") for c in server.calls] == ["json_schema", "json_object"]
    assert "schema" in server.calls[1]["response_format"]  # llama.cpp's spelling of schema-constrained JSON


def test_falls_back_to_plain_json_when_schemas_are_unsupported(server):
    def replies(task, body):
        fmt = body.get("response_format", {})
        if fmt.get("type") == "json_schema" or "schema" in fmt:
            return FakeResponse(500, {"error": "response_format: Input should be 'text' or 'json_object'"})
        return reply([])
    server.replies = replies
    LLM(model="m2", base_url="http://y", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert server.calls[-1]["response_format"] == {"type": "json_object"}
    assert "matching this JSON schema" in server.calls[-1]["messages"][0]["content"]


def test_cache_replays_answers_but_not_across_repetitions(server, tmp_path):
    server.replies = lambda task, body: reply([FLOW])
    args = dict(model="m", base_url="http://x", tasks=("flows",), cache_dir=tmp_path)
    first, _ = LLM(**args).extract(POLICY, [(0, len(POLICY))])
    second, stats = LLM(**args).extract(POLICY, [(0, len(POLICY))])
    assert first == second and len(server.calls) == 1 and stats["seconds"] >= 0
    LLM(**args, rep=1).extract(POLICY, [(0, len(POLICY))])
    assert len(server.calls) == 2  # a stability repetition really asks again


def test_unreachable_server_counts_failed_windows(monkeypatch):
    def down(*a, **k):
        raise llm.requests.ConnectionError("refused")
    monkeypatch.setattr(llm.requests, "post", down)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    claims, stats = LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert claims == [] and stats["failed_windows"] == 1 and stats["dropped"] == {"llm_error": 1}
