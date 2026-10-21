"""One-command demo of the policy reader on a short fictional health-app policy.

    python demo.py         # replays a recorded Groq run: no API key, no network, same output every time
    python demo.py live    # calls Groq for real (needs GROQ_API_KEY) and re-records the replies

The demo policy (demo_policy.txt) is written for this repo, so unlike APP-350 it can be committed.
One made-up quote is planted on purpose, to show the quote check catching it.
"""

import json
import sys
from pathlib import Path

from llm import LLM, LLMResult, call_llm, recording, replay
from policy_reader import SYSTEM_PROMPT, read_policy

HERE = Path(__file__).parent
POLICY = HERE / "demo_policy.txt"
RECORDING = HERE / "demo_recording.jsonl"

PLANTED = {
    "data_type": "Health and fitness", "action": "sell", "recipient": "unnamed_third_party",
    "purpose": "advertising_or_marketing", "condition": "unspecified", "negated": False,
    "quote": "We sell your heart rate history to insurance companies.",
}


def with_planted_row(llm: LLM) -> LLM:
    """Add one fabricated row to the first data-practice reply, as a hallucinating model would."""
    planted = False

    def wrapped(system: str, user: str) -> LLMResult:
        nonlocal planted
        res = llm(system, user)
        if system == SYSTEM_PROMPT and not planted:
            data = json.loads(res.text)
            data["rows"].append(PLANTED)
            res = LLMResult(json.dumps(data), res.input_tokens, res.output_tokens, res.model, res.ok)
            planted = True
        return res
    return wrapped


def main(live: bool) -> None:
    if live:
        RECORDING.unlink(missing_ok=True)
        llm = recording(call_llm, RECORDING)  # the recording holds only the model's real replies
    else:
        llm = replay(RECORDING)
    result = read_policy(POLICY.read_text(encoding="utf-8"), llm=with_planted_row(llm))

    print(f"Policy: {POLICY.name}   ({'live Groq call' if live else 'replayed from ' + RECORDING.name})\n")

    print("WHAT THE POLICY SAYS HAPPENS")
    for r in (r for r in result.rows if not r.negated):
        print(f"  {r.action:<8} {r.data_type:<26} -> {r.recipient:<20} {r.purpose:<36} {r.condition}")
        print(f"           \"{r.quote}\"")

    print("\nWHAT THE POLICY SAYS DOES NOT HAPPEN  (checked later against the Play label, APK and traffic)")
    for r in (r for r in result.rows if r.negated):
        print(f"  NOT {r.action:<8} {r.data_type:<26} -> {r.recipient}")
        print(f"           \"{r.quote}\"")

    print("\nOTHER CLAIMS  (security, retention, rights, breach, children, automated decisions, AI)")
    for c in result.claims:
        detail = f" [{c.detail}]" if c.detail else ""
        print(f"  {c.topic:<20} {c.value}{detail}")
        print(f"           \"{c.quote}\"")

    print("\nDROPPED BY THE CHECKS")
    for d in result.dropped:
        tag = "  <- planted by this demo" if d["row"].get("quote") == PLANTED["quote"] else ""
        print(f"  {d['reason']:<24} \"{d['row'].get('quote', '')}\"{tag}")

    s = result.stats
    print(f"\nchunks={s.chunks}  rows kept={len(result.rows)}  claims kept={len(result.claims)}  "
          f"invalid={s.invalid_rows + s.invalid_claims}  made-up quotes={s.fabricated_quotes + s.fabricated_claim_quotes}")
    print(f"tokens in/out={s.input_tokens:,}/{s.output_tokens:,}   cost=${s.cost_usd:.5f} "
          f"(what the recorded run cost on Groq; a replay itself costs nothing)")


if __name__ == "__main__":
    main(live=sys.argv[1:] == ["live"])
