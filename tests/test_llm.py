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
        headers = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append(json)
        Server.headers.append(headers)
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


def test_falls_back_when_the_model_rejects_json_schema(server):
    def replies(task, body):
        if body.get("response_format", {}).get("type") == "json_schema":
            return FakeResponse(400, {"error": {"message": "response_format json_schema is not supported"}})
        return reply([])
    server.replies = replies
    LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert [c.get("response_format", {}).get("type") for c in server.calls] == ["json_schema", "json_object"]
    assert "matching this JSON schema" in server.calls[1]["messages"][0]["content"]


def test_bad_key_stops_the_run_instead_of_cycling_modes(server):
    server.replies = lambda task, body: FakeResponse(400, [{"error": {"message": "API key not valid. Please pass a valid API key."}}])
    with pytest.raises(llm.LLMStop, match="API key not valid"):
        LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert len(server.calls) == 1


def test_gemini_needs_a_key_and_sends_it(server, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("MAPPA_LLM_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        LLM()  # the default endpoint is Gemini's
    monkeypatch.setenv("GEMINI_API_KEY", "k-123")
    server.replies = lambda task, body: reply([])
    LLM(tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert server.headers[-1]["Authorization"] == "Bearer k-123"
    assert server.calls[-1]["reasoning_effort"] == "low" and server.calls[-1]["model"] == llm.DEFAULT_MODEL


def test_rate_limits_wait_as_asked_and_daily_quota_stops(server, monkeypatch):
    waits = []
    monkeypatch.setattr(llm.time, "sleep", waits.append)
    replies = iter([FakeResponse(429, {"error": {"message": "Resource exhausted", "details": [{"retryDelay": "7s"}]}}),
                    reply([FLOW])])
    server.replies = lambda task, body: next(replies)
    claims, _ = LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert waits == [8.0] and len(claims) == 1
    server.replies = lambda task, body: FakeResponse(429, {"error": {"message": "Quota exceeded for GenerateRequestsPerDayPerProjectPerModel"}})
    with pytest.raises(llm.LLMStop, match="daily quota"):
        LLM(model="m2", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])


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


def test_truncated_reply_keeps_completed_claims():
    cut = '{"claims": [{"quote": "a", "practice": "retention", "negated": false, "detail": ""}, {"quote": "b", "prac'
    assert llm.parse_json(cut) == {"claims": [{"quote": "a", "practice": "retention", "negated": False, "detail": ""}]}
    assert llm.parse_json('<think>x</think>{"claims": []}') == {"claims": []}
    assert llm.parse_json("no json here") is None


def test_loose_json_with_wrong_types_or_shape_is_counted_not_trusted(server):
    no_negated = {k: v for k, v in FLOW.items() if k != "negated"}
    numeric_name = {**FLOW, "recipient_name": 5}
    server.replies = lambda task, body: reply([no_negated, numeric_name])
    claims, stats = LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert claims == [] and stats["dropped"] == {"schema": 2}  # a missing "negated" never becomes "affirm"
    server.replies = lambda task, body: FakeResponse(200, content=json.dumps({"flows": [FLOW]}))
    claims, stats = LLM(model="m", base_url="http://x", tasks=("flows",)).extract(POLICY, [(0, len(POLICY))])
    assert claims == [] and stats["failed_windows"] == 1  # wrong shape is a failed window, not "nothing found"
