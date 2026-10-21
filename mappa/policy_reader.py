"""Policy reader: privacy policy text -> validated, quote-checked rows (see schema.Row).

Pipeline per policy:
    chunk -> LLM extract per chunk -> schema validation -> quote check -> recipient check
          -> merge + dedupe -> token/cost totals

Two passes over the same chunks, each with its own prompt:
    rows   = data practices (collect / share / sell X, or deny it)
    claims = everything else the spec asks for (security, retention, rights, children, AI...)
Separate passes cost about twice as much, but each prompt stays short and focused, and the
data-practice numbers stay comparable with runs made before claims existed.
"""

import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Callable

from pydantic import ValidationError

from llm import LLMResult, call_llm, cost_usd
from schema import (ACTIONS, CLAIM_VALUES, CONDITIONS, DATA_TYPES, FIRST_PARTY, PURPOSES,
                    UNNAMED_THIRD_PARTY, Claim, Row)

# ---------------------------------------------------------------- prompt

# One line per Play category, from Play Console's own definitions. Without these the model
# guesses from the label alone (e.g. it filed "device model" under Device or other IDs).
CATEGORY_HINTS = """\
  Location: approximate or precise location (GPS, cell tower, Wi-Fi, location from IP address)
  Personal info: name, email, phone number, postal address, user ID, age, gender, other profile details
  Financial info: payment card or bank details, purchase history, credit info
  Health and fitness: health conditions, symptoms, medication, fitness or activity data
  Messages: emails, SMS, in-app chat messages
  Photos and videos / Audio files / Files and docs / Calendar: the user's own media, files, events
  Contacts: the user's address book / contact list
  App activity: in-app actions, search history, installed apps, user-generated content
  Web browsing: the user's web browsing history
  App info and performance: crash logs, diagnostics, device model, OS version, performance data
  Device or other IDs: identifiers for a device, browser or app install (advertising ID, IMEI,
    MAC address, Android ID, cookies and similar, IP address)"""

QUOTE_RULE = """quote: ONE sentence copied character-for-character from the excerpt that supports it.
    Copy the WHOLE sentence even if it is long. Never shorten it, never use "...", never join
    two sentences, never fix typos. A quote that is not an exact copy is thrown away."""

SYSTEM_PROMPT = f"""You extract data practices from one excerpt of a mobile app's privacy policy.

Return JSON only, in this exact shape: {{"rows": [ {{...}}, ... ]}}
Each row has these fields, using ONLY the allowed values:

- data_type: one of {list(DATA_TYPES)}
{CATEGORY_HINTS}
- action: one of {list(ACTIONS)}
    collect = the app/developer obtains the data; share = it is passed to another company;
    sell = it is exchanged for money or other value.
- recipient: "{FIRST_PARTY}" for the app developer itself, "{UNNAMED_THIRD_PARTY}" for
    unnamed partners/advertisers/providers, or the company's name exactly as the policy writes it.
- purpose: one of {list(PURPOSES)}
    Use "unspecified" if the policy gives no purpose for this practice.
- condition: one of {list(CONDITIONS)}
- negated: false if the policy says the practice happens; true if it says it does NOT happen
    ("we do not collect your location", "we never sell your personal information").
- {QUOTE_RULE}

Rules:
- Include denials as rows with negated=true. "We may collect X" counts as a practice (negated=false).
- Skip generic statements that name no data type.
- One row per (data_type, action, recipient, purpose, negated). Several rows may share a quote.
- If the excerpt contains no practices, return {{"rows": []}}."""

CLAIMS_PROMPT = f"""You extract non-data statements from one excerpt of a mobile app's privacy policy:
security, retention, user rights, breach handling and privacy contact, children, automated
decisions, and AI.

Return JSON only, in this exact shape: {{"claims": [ {{...}}, ... ]}}
Each claim has: topic, value, detail, quote.

- topic and value: the value must be one listed under its topic:
{chr(10).join(f"    {t}: {list(v)}" for t, v in CLAIM_VALUES.items())}
- detail: "" unless the value needs a specific: the retention period (fixed_period), the age
    (minimum_age), the contact email/address/role (privacy_contact), the AI provider's name
    (data_sent_to_ai_provider). Copy it exactly as written in the quote.
- {QUOTE_RULE}

Meanings that are easy to confuse:
- automated_decisions is about decisions made about the user by a computer program (credit,
  eligibility, pricing, risk scores, content moderation, health triage), not about analytics.
- profiling = building a profile of the user to predict interests or behaviour.
- no_security_guarantee = "no method of transmission is 100% secure" style disclaimers.

Rules:
- Only what the excerpt actually states. No inferences.
- If the excerpt has none of these statements, return {{"claims": []}}."""


# ---------------------------------------------------------------- chunking


def chunk_policy(text: str, max_chars: int = 8000) -> list[str]:
    """Split on paragraph breaks, then pack whole paragraphs into chunks of <= max_chars.

    Bigger chunks mean fewer calls (the system prompt is re-sent with every chunk) but give
    the model more text to skim per call, which tends to lower recall.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]

    pieces: list[str] = []
    for p in paragraphs:
        if len(p) <= max_chars:
            pieces.append(p)
        else:
            # A single huge paragraph: fall back to splitting on sentence ends.
            pieces.extend(_pack(re.split(r"(?<=[.!?])\s+", p), max_chars, sep=" "))
    return _pack(pieces, max_chars, sep="\n\n")


def _pack(pieces: list[str], max_chars: int, sep: str) -> list[str]:
    chunks, current = [], ""
    for piece in pieces:
        if current and len(current) + len(sep) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}{sep}{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------- quote check

_CHAR_MAP = str.maketrans({
    "‘": "'", "’": "'", "‛": "'", "′": "'", "`": "'",
    "“": '"', "”": '"', "‟": '"', "″": '"',
    "–": "-", "—": "-", "−": "-",
    " ": " ",
})


def normalise(s: str) -> str:
    """Make text comparable without changing its words: unify quote/dash characters and whitespace."""
    s = unicodedata.normalize("NFKC", s).translate(_CHAR_MAP)
    return re.sub(r"\s+", " ", s).strip()


def quote_found(quote: str, policy_lower: str) -> bool:
    """Every word of the quote must appear in the policy, in order, with nothing in between.

    Two differences are forgiven, because they change no words: capital letters, and punctuation
    at the very end. Models often quote half of a sentence at a comma and then add a full stop
    ("We do not sell your personal information to anyone.") or capitalise its first word. Without
    this, exactly the denials the consistency engine needs were being thrown away. A quote with
    "..." or any changed word still fails.
    """
    return normalise(quote).rstrip(" .,;:").casefold() in policy_lower


# ---------------------------------------------------------------- extraction


@dataclass
class Stats:
    chunks: int = 0
    failed_chunks: int = 0       # no parseable JSON even after one retry
    raw_rows: int = 0            # rows the model returned
    invalid_rows: int = 0        # dropped: a field outside the allowed values
    fabricated_quotes: int = 0   # dropped: quote not found in the policy
    unknown_recipients: int = 0  # dropped: named company that never appears in the policy
    duplicates: int = 0          # removed in merge
    raw_claims: int = 0
    invalid_claims: int = 0
    fabricated_claim_quotes: int = 0
    unsupported_details: int = 0  # dropped: claim detail not found inside its own quote
    failed_claim_chunks: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def fabricated_rate(self) -> float:
        """Share of schema-valid rows whose quote isn't in the policy."""
        checked = self.raw_rows - self.invalid_rows
        return self.fabricated_quotes / checked if checked else 0.0


@dataclass
class PolicyResult:
    rows: list[Row]
    stats: Stats
    claims: list[Claim] = field(default_factory=list)
    dropped: list[dict] = field(default_factory=list)  # every dropped row + why, for inspection


def parse_list(text: str, key: str) -> list | None:
    """Pull the list under `key` out of the model's reply. None means unusable output."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        items = json.loads(text)[key]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return items if isinstance(items, list) else None


def extract_chunk(chunk: str, llm: Callable[[str, str], LLMResult], stats: Stats,
                  prompt: str = SYSTEM_PROMPT, key: str = "rows") -> list | None:
    """Ask the LLM about one chunk. Retries once if the reply isn't usable JSON."""
    for _ in range(2):
        res = llm(prompt, f"Policy excerpt:\n\n{chunk}")
        stats.input_tokens += res.input_tokens
        stats.output_tokens += res.output_tokens
        stats.cost_usd += cost_usd(res.model, res.input_tokens, res.output_tokens)
        items = parse_list(res.text, key) if res.ok else None
        if items is not None:
            return items
    return None


def read_policy(text: str, llm: Callable[[str, str], LLMResult] = call_llm,
                claims: bool = True) -> PolicyResult:
    stats = Stats()
    policy_norm = normalise(text)
    policy_lower = policy_norm.casefold()
    kept: dict[tuple, Row] = {}
    kept_claims: dict[tuple, Claim] = {}
    dropped: list[dict] = []

    for chunk in chunk_policy(text):
        stats.chunks += 1
        raw = extract_chunk(chunk, llm, stats)
        if raw is None:
            stats.failed_chunks += 1
            raw = []

        for item in raw:
            stats.raw_rows += 1

            # 1. Allowed values only. Anything off-list is dropped, never coerced.
            try:
                row = Row.model_validate(item)
            except ValidationError as e:
                stats.invalid_rows += 1
                dropped.append({"reason": "invalid", "row": item, "error": str(e.errors()[0]["msg"])})
                continue

            # 2. The quote must exist in the policy, or the model made it up.
            if not quote_found(row.quote, policy_lower):
                stats.fabricated_quotes += 1
                dropped.append({"reason": "fabricated_quote", "row": item})
                continue

            # 3. A named recipient must be named somewhere in the policy.
            if row.recipient not in (FIRST_PARTY, UNNAMED_THIRD_PARTY) and row.recipient.casefold() not in policy_lower:
                stats.unknown_recipients += 1
                dropped.append({"reason": "unknown_recipient", "row": item})
                continue

            # 4. Merge: the same practice stated in several places counts once (first quote kept).
            #    "We collect X" and "we don't collect X" are different keys: both are kept, and
            #    the pair is itself worth flagging later (the policy contradicts itself).
            key = (row.data_type, row.action, row.recipient.casefold(), row.purpose, row.condition, row.negated)
            if key in kept:
                stats.duplicates += 1
            else:
                kept[key] = row

        if not claims:
            continue
        raw = extract_chunk(chunk, llm, stats, prompt=CLAIMS_PROMPT, key="claims")
        if raw is None:
            stats.failed_claim_chunks += 1
            raw = []

        for item in raw:
            stats.raw_claims += 1
            try:
                claim = Claim.model_validate(item)
            except ValidationError as e:
                stats.invalid_claims += 1
                dropped.append({"reason": "invalid_claim", "row": item, "error": str(e.errors()[0]["msg"])})
                continue
            if not quote_found(claim.quote, policy_lower):
                stats.fabricated_claim_quotes += 1
                dropped.append({"reason": "fabricated_claim_quote", "row": item})
                continue
            # The detail (a period, an age, a company) must be copied from its own quote. This is the
            # recipient check made stricter: it ties "30 days" to the sentence that actually says it.
            if claim.detail and normalise(claim.detail).casefold() not in normalise(claim.quote).casefold():
                stats.unsupported_details += 1
                dropped.append({"reason": "unsupported_detail", "row": item})
                continue
            key = (claim.topic, claim.value, claim.detail.casefold())
            if key not in kept_claims:
                kept_claims[key] = claim

    return PolicyResult(rows=list(kept.values()), stats=stats, dropped=dropped,
                        claims=list(kept_claims.values()))


if __name__ == "__main__":
    # Usage: python policy_reader.py <APP-350 policy id | path to a .txt policy>
    arg = sys.argv[1]
    if arg.isdigit():
        from app350 import load_policy
        policy_text = load_policy(int(arg)).text
    else:
        policy_text = open(arg, encoding="utf-8").read()

    result = read_policy(policy_text)
    for r in result.rows:
        verb = f"NOT {r.action}" if r.negated else r.action
        print(f"{r.data_type:<26} {verb:<11} {r.recipient:<22} {r.purpose:<28} {r.condition:<13} | {r.quote[:90]}")
    print()
    for c in result.claims:
        detail = f" ({c.detail})" if c.detail else ""
        print(f"{c.topic:<20} {c.value + detail:<45} | {c.quote[:90]}")
    s = result.stats
    print(json.dumps(asdict(s) | {"fabricated_rate": round(s.fabricated_rate, 3)}, indent=2))
