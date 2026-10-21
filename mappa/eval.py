"""Score the policy reader against APP-350 expert labels.

Scoring is per policy: for each Play category, did the reader say the policy collects/shares
it, and did the experts? That matches what MAPPA needs downstream (compare a policy's claims
against its Play label), and it's how the APP-350 authors scored their own classifiers.

Usage:
    python eval.py [n_policies]          # real LLM run on the first n TEST policies (default 10)
    python eval.py oracle [n_policies]   # no API calls: checks the pipeline + scoring with a fake LLM
"""

import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from app350 import Policy, load_policy, policy_ids
from mappings import EVALUABLE
from llm import GROQ_MODEL, LLMResult, call_llm, recording, replay
from policy_reader import FIRST_PARTY, PolicyResult, read_policy

# Run outputs quote APP-350 text, so they stay local (licence: no redistribution).
RESULTS_DIR = Path(__file__).parent / "results"

# Groq's free tier caps tokens per day (about 40 policies' worth), so a 100-policy eval spans days.
# Every raw reply is saved, per model and policy. On a rerun, any call already answered (same prompt,
# same excerpt) is replayed from disk instead of re-sent: the run picks up where it stopped, a prompt
# edit automatically means fresh calls, and the checks and scoring always re-run on current code.
RAW_DIR = RESULTS_DIR / "raw" / GROQ_MODEL.replace("/", "_")


def read_with_cache(policy: Policy) -> PolicyResult:
    path = RAW_DIR / f"policy_{policy.policy_id}.jsonl"
    saved = replay(path) if path.exists() else None
    live = recording(call_llm, path)

    def llm(system: str, user: str) -> LLMResult:
        if saved is not None:
            try:
                return saved(system, user)
            except KeyError:
                pass
        return live(system, user)

    # APP-350 has no labels for the claims pass, so skip it here: it would double the API calls
    # for output nothing can score.
    return read_policy(policy.text, llm=llm, claims=False)


def predicted(result: PolicyResult) -> tuple[set[str], set[tuple[str, str]], set[str]]:
    """Reader rows -> the same shape as the gold labels: (types, (type, party) pairs, denied types)."""
    types, pairs, denied = set(), set(), set()
    for row in result.rows:
        if row.data_type not in EVALUABLE:
            continue
        if row.negated:
            denied.add(row.data_type)
            continue
        types.add(row.data_type)
        # APP-350's "1stParty" = the developer gets it; "3rdParty" = another company gets it.
        party = "1stParty" if row.recipient == FIRST_PARTY and row.action == "collect" else "3rdParty"
        pairs.add((row.data_type, party))
    return types, pairs, denied


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def score(gold: list[set], pred: list[set], labels: list) -> dict:
    """Per-label and micro-averaged P/R/F1 over policies."""
    out, total = {}, [0, 0, 0]
    for label in labels:
        tp = sum(label in g and label in p for g, p in zip(gold, pred))
        fp = sum(label not in g and label in p for g, p in zip(gold, pred))
        fn = sum(label in g and label not in p for g, p in zip(gold, pred))
        out[label] = (tp, fp, fn, *prf(tp, fp, fn))
        total = [total[0] + tp, total[1] + fp, total[2] + fn]
    out["MICRO"] = (*total, *prf(*total))
    return out


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    print(f"  {'label':<38} {'TP':>4} {'FP':>4} {'FN':>4}   {'P':>5} {'R':>5} {'F1':>5}")
    for label, (tp, fp, fn, p, r, f) in table.items():
        name = label if isinstance(label, str) else " / ".join(label)
        print(f"  {name:<38} {tp:>4} {fp:>4} {fn:>4}   {p:5.2f} {r:5.2f} {f:5.2f}")


def oracle_llm(policy: Policy):
    """A fake LLM that answers from the expert labels, plus one made-up quote per chunk.

    If the pipeline and scoring are correct, this must give P = R = 1.0 and exactly one
    fabricated quote per chunk. It measures nothing about any real model.
    """
    def llm(system: str, user: str) -> LLMResult:
        rows = [
            {"data_type": play, "action": "collect" if party == "1stParty" else "share",
             "recipient": FIRST_PARTY if party == "1stParty" else "unnamed_third_party",
             "purpose": "unspecified", "condition": "unspecified", "quote": sentence}
            for sentence, play, party in policy.gold_sentences
            if sentence in user
        ] + [
            {"data_type": play, "action": "collect", "recipient": FIRST_PARTY, "purpose": "unspecified",
             "condition": "unspecified", "quote": sentence, "negated": True}
            for sentence, play in policy.gold_neg_sentences
            if sentence in user
        ]
        rows.append({"data_type": "Location", "action": "sell", "recipient": "unnamed_third_party",
                     "purpose": "other", "condition": "unspecified",
                     "quote": "We sell your precise location to anyone who asks."})
        return LLMResult(json.dumps({"rows": rows, "claims": []}), 0, 0, "oracle")
    return llm


def run(n: int, oracle: bool = False) -> None:
    ids = policy_ids("TEST")[:n]
    gold_t, gold_p, gold_n, pred_t, pred_p, pred_n, per_policy = [], [], [], [], [], [], []
    totals = {"raw_rows": 0, "invalid_rows": 0, "fabricated_quotes": 0, "unknown_recipients": 0,
              "failed_chunks": 0, "chunks": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}

    for pid in ids:
        policy = load_policy(pid)
        t0 = time.time()
        result = read_policy(policy.text, llm=oracle_llm(policy), claims=False) if oracle else read_with_cache(policy)
        s = result.stats
        types, pairs, denied = predicted(result)

        gold_t.append(policy.gold_types)
        gold_p.append(policy.gold_pairs)
        pred_t.append(types)
        pred_p.append(pairs)
        gold_n.append(policy.gold_neg_types)
        pred_n.append(denied)
        for k in totals:
            totals[k] += getattr(s, k)
        print(f"policy {pid:>3} {policy.name[:28]:<28} chunks={s.chunks:<3} rows={len(result.rows):<3} "
              f"fabricated={s.fabricated_quotes:<3} invalid={s.invalid_rows:<3} "
              f"${s.cost_usd:.4f}  {time.time() - t0:.1f}s")
        per_policy.append({"policy_id": pid, "stats": asdict(s), "rows": [r.model_dump() for r in result.rows],
                           "dropped": result.dropped, "gold_types": sorted(policy.gold_types),
                           "gold_pairs": sorted(policy.gold_pairs)})

    print_table("Data category (did the policy disclose it at all?)", score(gold_t, pred_t, EVALUABLE))
    pair_labels = [(c, p) for c in EVALUABLE for p in ("1stParty", "3rdParty")]
    print_table("Data category + who gets it", score(gold_p, pred_p, pair_labels))
    print_table("Denials (\"we do NOT collect/share X\")", score(gold_n, pred_n, EVALUABLE))

    checked = totals["raw_rows"] - totals["invalid_rows"]
    print(f"\nPolicies: {len(ids)}   chunks: {totals['chunks']}   failed chunks: {totals['failed_chunks']}")
    print(f"Rows from model: {totals['raw_rows']}   invalid (off-schema): {totals['invalid_rows']}   "
          f"unknown recipient: {totals['unknown_recipients']}")
    print(f"Fabricated-quote rate: {totals['fabricated_quotes']}/{checked} = "
          f"{totals['fabricated_quotes'] / checked if checked else 0:.1%}")
    print(f"Tokens in/out: {totals['input_tokens']:,} / {totals['output_tokens']:,}   "
          f"cost: ${totals['cost_usd']:.4f} total, ${totals['cost_usd'] / len(ids):.4f} per policy")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"{'oracle' if oracle else 'run'}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps({"totals": totals, "policies": per_policy}, indent=1))
    print(f"Saved: {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    is_oracle = bool(args) and args[0] == "oracle"
    if is_oracle:
        args = args[1:]
    try:
        run(int(args[0]) if args else 10, oracle=is_oracle)
    except RuntimeError as e:  # Groq's daily cap: replies so far are saved, rerun later to continue
        sys.exit(f"Stopped: {e}\nEvery reply so far is saved; run the same command later to carry on.")
