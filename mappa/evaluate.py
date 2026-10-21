"""Measuring the policy reader against gold labels, and humans against each other.

Plain-English definitions used throughout:
  precision  of the things the extractor claimed, the share that were right.
  recall     of the things really in the policy, the share the extractor found.
  F1         one number balancing the two (their harmonic mean).
  L1         document level: "does this policy say it collects Location?" (what Build 2 compares
             against the label, the headline).
  L2         segment level: the same question per paragraph (a diagnostic: was it found in the right place?).
  reference  the F1 of a lazy extractor that says "yes" to every label. A real extractor must beat it.
  95% CI     re-sampling the documents 1,000 times shows how much the score would move on a different sample.
  kappa      agreement between two annotators after removing the agreement expected by pure chance.
"""

import csv
import json
import platform
import random
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .datasets import SCOPE, GoldDoc, load, parse_codes
from .policy_reader import PROMPT_VERSION, segment_of
from .store import Store, git_commit
from .text import find_quote, segment
from .vocab import CATEGORIES, GENERIC, PRACTICES, Claim


def _act(action: str | None) -> str | None:
    return "share" if action == "sell" else action  # Play has no "sell": selling is a kind of sharing


PROJECTIONS = {
    # headline: affirmed data flows, as (category, action)
    "flow": lambda c: {(c.category, _act(c.action))} if c.kind == "flow" and c.polarity == "affirm" and c.action else set(),
    "deny": lambda c: {(c.category, _act(c.action))} if c.kind == "flow" and c.polarity == "deny" and c.action else set(),
    "mention": lambda c: {c.category} if c.kind == "flow" and c.category not in (None, "*") else set(),
    "practice": lambda c: {(c.practice, c.polarity)} if c.kind == "practice" else set(),
    "subtype": lambda c: ({(c.category, c.subtype, _act(c.action))}
                          if c.kind == "flow" and c.polarity == "affirm" and c.subtype and c.action else set()),
    "purpose": lambda c: set(c.purposes) if c.kind == "flow" and c.polarity == "affirm" else set(),
}
CORPUS_PROJECTIONS = {
    "app350": ("flow", "deny", "subtype"),
    "opp115": ("flow", "deny", "practice", "purpose"),
    "unibo": ("mention",),
    "indomain": ("flow", "deny", "practice"),
}
MIN_SUPPORT = 5


# ---------------------------------------------------------------- counting
def labels_of(claims, projection: str, scope) -> set:
    out = set()
    for c in claims:
        c = scope(c)
        if c is not None:
            out |= PROJECTIONS[projection](c)
    return out


def counts(pairs: list[tuple[set, set]]) -> tuple[dict, list]:
    """pairs: (gold labels, predicted labels) per unit. -> per-label [tp, fp, fn], per-unit (tp, fp, fn)."""
    per_label, per_unit = defaultdict(lambda: [0, 0, 0]), []
    for gold, pred in pairs:
        for lab in gold & pred:
            per_label[lab][0] += 1
        for lab in pred - gold:
            per_label[lab][1] += 1
        for lab in gold - pred:
            per_label[lab][2] += 1
        per_unit.append((len(gold & pred), len(pred - gold), len(gold - pred)))
    return dict(per_label), per_unit


def prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else None
    return p, r, f


def bootstrap(per_doc: list[tuple[int, int, int]], n: int = 1000, seed: int = 0) -> tuple[float, float] | None:
    """95% interval of micro-F1 when the documents are re-sampled with replacement."""
    if not per_doc:
        return None
    rng = random.Random(seed)
    scores = []
    for _ in range(n):
        tp = fp = fn = 0
        for _ in per_doc:
            a, b, c = per_doc[rng.randrange(len(per_doc))]
            tp, fp, fn = tp + a, fp + b, fn + c
        scores.append(prf(tp, fp, fn)[2] or 0.0)
    scores.sort()
    return scores[int(0.025 * n)], scores[int(0.975 * n) - 1]


def summarise(per_label: dict, n_units: int, per_doc: list | None = None) -> dict:
    """Per-label and averaged scores. n_units: how many documents (L1) or segments (L2) were judged."""
    rows = []
    for lab, (tp, fp, fn) in sorted(per_label.items(), key=lambda kv: str(kv[0])):
        support = tp + fn
        p, r, f = prf(tp, fp, fn)
        prevalence = support / n_units if n_units else 0
        rows.append({"label": lab, "tp": tp, "fp": fp, "fn": fn, "support": support, "p": p, "r": r,
                     "f1": f if support >= MIN_SUPPORT else None,
                     "reference_f1": 2 * prevalence / (1 + prevalence) if prevalence <= 1 else None})
    tp, fp, fn = (sum(v[i] for v in per_label.values()) for i in range(3))
    p, r, f = prf(tp, fp, fn)
    scored = [row for row in rows if row["f1"] is not None]
    return {
        "micro": {"p": p, "r": r, "f1": f, "ci95": bootstrap(per_doc) if per_doc else None},
        "macro_f1": statistics.mean(row["f1"] for row in scored) if scored else None,
        "macro_reference_f1": statistics.mean(row["reference_f1"] for row in scored
                                              if row["reference_f1"] is not None) if scored else None,
        "labels": rows,
    }


def score(docs: list[GoldDoc], predictions: list[list[Claim]], corpus: str) -> dict:
    scope = SCOPE[corpus]
    out = {}
    n_segments = sum(len(d.gold) for d in docs)
    for proj in CORPUS_PROJECTIONS[corpus]:
        l1 = [(labels_of(set().union(*d.gold) if d.gold else set(), proj, scope), labels_of(pred, proj, scope))
              for d, pred in zip(docs, predictions)]
        per_label, per_doc = counts(l1)
        l2_pairs, l2_doc = [], []
        for d, pred in zip(docs, predictions):
            by_seg = defaultdict(list)
            for c in pred:
                by_seg[c.seg].append(c)
            pairs = [(labels_of(g, proj, scope), labels_of(by_seg.get(i, []), proj, scope)) for i, g in enumerate(d.gold)]
            l2_pairs += pairs
            _, unit = counts(pairs)
            l2_doc.append(tuple(map(sum, zip(*unit))) if unit else (0, 0, 0))
        l2_label, _ = counts(l2_pairs)
        out[proj] = {"L1": summarise(per_label, len(docs), per_doc), "L2": summarise(l2_label, n_segments, l2_doc)}
    return out


# ---------------------------------------------------------------- running an evaluation
def stability(reps: list[list[set]]) -> float | None:
    """Mean over documents of the mean pairwise Jaccard similarity of the label sets of repeated runs."""
    if len(reps) < 2:
        return None
    per_doc = []
    for sets in zip(*reps):
        sims = [len(a & b) / len(a | b) if a | b else 1.0 for i, a in enumerate(sets) for b in sets[i + 1:]]
        per_doc.append(statistics.mean(sims))
    return statistics.mean(per_doc)


def quote_matcher_check(docs: list[GoldDoc]) -> dict:
    """Share of APP-350's annotated sentences the quote matcher finds in their own policy (should be >= 99%)."""
    results = Counter(find_quote(q, d.text, min_words=1)[1] for d in docs for q in d.quotes)
    total = sum(results.values())
    return {"found": results["ok"] / total if total else None, "n": total, "reasons": dict(results)}


def run_eval(extractor, corpora: list[str], root: Path, *, split: str = "test", reps: int = 1,
             subset: int | None = None, gold_path: Path | None = None) -> dict:
    report = {
        "extractor": extractor.name, "model": extractor.model, "split": split, "reps": reps, "subset": subset,
        "prompt_version": PROMPT_VERSION if extractor.name == "llm" else None,
        "model_sha256": getattr(extractor, "sha256", None),  # the exact trained baseline file
        "git_commit": git_commit(), "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "machine": f"{platform.platform()} | {platform.processor() or platform.machine()} | python {platform.python_version()}",
        "corpora": {},
    }
    for corpus in corpora:
        if corpus in ("unibo", "indomain") and split != "test":
            print(f"skipping {corpus}: it is test data only, so it is never used for {split} (prompt) work")
            continue
        docs = load(corpus, root, None if corpus in ("unibo", "indomain") else split, gold_path)[:subset]
        runs, stats = [], []
        for rep in range(reps if extractor.name != "baseline" else 1):  # the baseline is deterministic
            if hasattr(extractor, "rep"):
                extractor.rep = rep
            preds = []
            for i, d in enumerate(docs, 1):
                claims, st = extractor.extract(d.text, d.spans)
                preds.append(segment_of(claims, d.spans))
                stats.append(st)
                if sys.stdout.isatty():
                    print(f"  {corpus} rep {rep} [{i}/{len(docs)}] {d.doc_id}: {len(claims)} claims   ", end="\r", flush=True)
            runs.append(preds)
        result = {"docs": len(docs), "scores": score(docs, runs[0], corpus), "cost": cost(stats)}
        if len(runs) > 1:
            proj = CORPUS_PROJECTIONS[corpus][0]
            sets = [[labels_of(p, proj, SCOPE[corpus]) for p in preds] for preds in runs]
            f1s = [score(docs, preds, corpus)[proj]["L1"]["micro"]["f1"] or 0.0 for preds in runs]
            result["stability"] = {"projection": proj, "jaccard": stability(sets),
                                   "f1_mean": statistics.mean(f1s), "f1_sd": statistics.pstdev(f1s)}
        if corpus == "app350":
            result["quote_matcher"] = quote_matcher_check(docs)
        report["corpora"][corpus] = result
    return report


def cost(stats: list[dict]) -> dict:
    dropped = Counter()
    for s in stats:
        dropped.update(s.get("dropped", {}))
    kept = sum(s.get("kept", 0) for s in stats)
    quote_fail = sum(dropped[k] for k in ("not_found", "too_short", "elided"))
    secs = [s.get("seconds", 0.0) for s in stats]
    tokens = [s.get("prompt_tokens", 0) + s.get("completion_tokens", 0) for s in stats]
    return {
        "seconds_per_doc_median": statistics.median(secs) if secs else None,
        "seconds_per_doc_mean": statistics.mean(secs) if secs else None,
        "tokens_per_doc_mean": statistics.mean(tokens) if tokens else None,
        "quote_failure_rate": quote_fail / (kept + quote_fail) if kept + quote_fail else 0.0,
        "dropped": dict(dropped),
        "windows": sum(s.get("windows", 0) for s in stats),
        "failed_windows": sum(s.get("failed_windows", 0) for s in stats),
        "cached_windows": sum(s.get("cached_windows", 0) for s in stats),
    }


def _fmt(x, digits=2):
    return "  -  " if x is None else f"{x:.{digits}f}"


def print_report(report: dict) -> None:
    version = f", prompt {report['prompt_version']}" if report["prompt_version"] else ""
    print(f"\n== {report['extractor']} ({report['model']}{version}), {report['split']} split, commit {report['git_commit']} ==")
    for corpus, res in report["corpora"].items():
        print(f"\n## {corpus}: {res['docs']} documents")
        for proj, levels in res["scores"].items():
            for level, s in levels.items():
                if not any(row["tp"] + row["fp"] + row["fn"] for row in s["labels"]):
                    print(f"{proj:8} {level}  (nothing to score: no gold and no predictions)")
                    continue
                if not any(row["tp"] + row["fp"] for row in s["labels"]):
                    print(f"{proj:8} {level}  NO PREDICTIONS: recall 0 on {sum(r['fn'] for r in s['labels'])} gold labels")
                    continue
                m = s["micro"]
                ci = f"[{_fmt(m['ci95'][0])}, {_fmt(m['ci95'][1])}]" if m["ci95"] else ""
                print(f"{proj:8} {level}  micro P {_fmt(m['p'])} R {_fmt(m['r'])} F1 {_fmt(m['f1'])} {ci}"
                      f"   macro F1 {_fmt(s['macro_f1'])} (say-yes-to-everything reference {_fmt(s['macro_reference_f1'])})")
                if level == "L1" and proj in ("flow", "practice", "mention"):
                    for row in s["labels"]:
                        lab = " / ".join(map(str, row["label"])) if isinstance(row["label"], tuple) else row["label"]
                        f1 = _fmt(row["f1"]) if row["f1"] is not None else f"n={row['support']}"
                        print(f"    {lab:45} F1 {f1:>6}  P {_fmt(row['p'])} R {_fmt(row['r'])}"
                              f"  (tp {row['tp']}, fp {row['fp']}, fn {row['fn']}; ref {_fmt(row['reference_f1'])})")
        c = res["cost"]
        print(f"cost: {_fmt(c['seconds_per_doc_median'], 1)} s/doc (median), {_fmt(c['tokens_per_doc_mean'], 0)} tokens/doc;"
              f" quote failures {_fmt(c['quote_failure_rate'])}; failed windows {c['failed_windows']}/{c['windows']}"
              f" ({c['cached_windows']} replayed from cache, timed at their original run);"
              f" dropped {c['dropped'] or 'none'}")
        if "stability" in res:
            st = res["stability"]
            print(f"stability over {report['reps']} runs ({st['projection']}): Jaccard {_fmt(st['jaccard'])},"
                  f" F1 {_fmt(st['f1_mean'])} +/- {_fmt(st['f1_sd'])}")
        if "quote_matcher" in res:
            q = res["quote_matcher"]
            print(f"quote matcher finds {_fmt(q['found'], 4)} of {q['n']} annotated APP-350 sentences")


def save_report(report: dict, root: Path) -> Path:
    out = Path(root) / "results"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{report['extractor']}_{report['created'].replace(':', '')}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    return path


# ---------------------------------------------------------------- one policy, explained
MARKS = {"found": "✓", "false_alarm": "+", "missed": "−", "not_judged": "·"}
LEGEND = """How to read this:
  ✓ found        the experts labelled it and the reader found it
  + false alarm  the reader claimed it, but the experts did not label it
  − missed       the experts labelled it, the reader did not find it
  · not judged   outside what this corpus labels (e.g. APP-350 never labels health data), so neither right nor wrong
Each claim shows the policy text the reader quoted as its evidence."""


def _demo_label(c: Claim) -> tuple | None:
    if c.kind == "practice":
        return ("practice", c.practice, c.polarity)
    if c.kind == "flow" and c.action:
        return ("flow", c.category, _act(c.action), c.polarity)
    if c.kind == "flow" and c.category not in (None, "*"):
        return ("mention", c.category)
    return None


def _label_text(label: tuple) -> str:
    kind, *rest = label
    if kind == "flow":
        category, action, polarity = rest
        return f"{'NOT ' if polarity == 'deny' else ''}{action} {category}"
    if kind == "practice":
        practice, polarity = rest
        return f"{'NOT ' if polarity == 'deny' else ''}{practice}"
    return f"mentions {rest[0]}"


def compare(doc: GoldDoc, claims: list[Claim], corpus: str) -> list[dict]:
    """Per segment: what the experts labelled next to what the reader claimed, each marked as in MARKS."""
    scope = SCOPE[corpus]
    by_seg = defaultdict(list)
    for c in claims:
        by_seg[c.seg].append(c)
    rows = []
    for i, ((s, e), gold) in enumerate(zip(doc.spans, doc.gold)):
        expert = {lab for g in gold if (j := scope(g)) is not None and (lab := _demo_label(j))}
        items, judged = [], set()
        for c in by_seg.get(i, []):
            shown = _demo_label(c)
            if shown is None:
                continue
            j = scope(c)
            if j is None:
                items.append(("not_judged", shown, c.quote))
            else:
                lab = _demo_label(j)
                judged.add(lab)
                items.append(("found" if lab in expert else "false_alarm", shown, c.quote))
        items += [("missed", lab, None) for lab in sorted(expert - judged, key=str)]
        if items:
            rows.append({"seg": i, "text": doc.text[s:e], "items": items})
    return rows


def pick_demo_doc(docs: list[GoldDoc], corpus: str, max_words: int = 2500) -> GoldDoc:
    """A readable policy with plenty going on: under max_words, with the most distinct expert labels."""
    scope = SCOPE[corpus]

    def variety(d: GoldDoc) -> int:
        return len({_demo_label(j) for g in d.gold for c in g if (j := scope(c)) is not None})

    short = [d for d in docs if len(d.text.split()) <= max_words] or docs
    return max(short, key=variety)  # max() keeps the first (lowest id) on ties, so the pick is repeatable


def demo(extractor, corpus: str, root: Path, doc_id: str | None = None, split: str = "test") -> None:
    docs = load(corpus, root, split)
    if doc_id:
        doc = next((d for d in docs if d.doc_id == doc_id), None)
        if doc is None:
            raise SystemExit(f"no {corpus} {split} policy with id {doc_id}")
    else:
        doc = pick_demo_doc(docs, corpus)
    claims, stats = extractor.extract(doc.text, doc.spans)
    claims = segment_of(claims, doc.spans)

    print(LEGEND)
    print(f"\n== {corpus} {split} policy {doc.doc_id} ({doc.name}): {len(doc.text.split())} words, "
          f"{len(doc.spans)} paragraphs | reader: {extractor.name} ({extractor.model}) ==")
    rows = compare(doc, claims, corpus)
    for row in rows:
        preview = " ".join(row["text"].split())
        print(f"\n[paragraph {row['seg']}] {preview[:220]}{'...' if len(preview) > 220 else ''}")
        for mark, label, quote in row["items"]:
            evidence = f'  "{" ".join(quote.split())[:110]}"' if quote else ""
            print(f"   {MARKS[mark]} {_label_text(label):42}{evidence}")
    print(f"\n{len(rows)} of {len(doc.spans)} paragraphs had something to show; the rest said nothing the "
          "experts or the reader labelled.")

    print("\n== Whole policy: does it say X anywhere? ==")
    for proj, levels in score([doc], [claims], corpus).items():
        if proj in ("subtype", "purpose"):  # finer diagnostics; see `evaluate` for those
            continue
        rows_ = [r for r in levels["L1"]["labels"] if r["tp"] + r["fp"] + r["fn"]]
        if not rows_:
            continue
        m = levels["L1"]["micro"]
        print(f"{proj}: precision {_fmt(m['p'])}, recall {_fmt(m['r'])}, F1 {_fmt(m['f1'])}")
        for r in rows_:
            lab = " / ".join(map(str, r["label"])) if isinstance(r["label"], tuple) else r["label"]
            verdict = "found" if r["tp"] else ("false alarm" if r["fp"] else "missed")
            print(f"   {lab:40} {verdict}")
    if extractor.name == "llm":
        quote_fail = sum(stats["dropped"].get(k, 0) for k in ("not_found", "too_short", "elided"))
        print(f"\n== The copied-quote check ==\n{quote_fail} claims were thrown away because their quote is not in the "
              f"policy ({stats['kept']} kept). Examples:")
        for why, quote in stats.get("rejected_examples", []):
            print(f'   [{why}] "{quote[:140]}"')
        replayed = f" ({stats['cached_windows']} answers replayed from cache)" if stats["cached_windows"] else ""
        print(f"{stats['windows']} model calls, {stats['failed_windows']} failed; "
              f"{stats['prompt_tokens'] + stats['completion_tokens']} tokens; {stats['seconds']:.0f}s{replayed}")


# ---------------------------------------------------------------- human annotation
CODEBOOK_HEADER = """MAPPA annotation codebook. One row per policy segment; write codes in the `codes` column, separated by ';'.
Annotate independently and without looking at any model output.

Data flows:   <category>:<collect|share|sell>[:not][:sp]
  :not  the policy says it does NOT do this ("we never sell your health data" -> Health and fitness:sell:not)
  :sp   the recipient is a service provider acting for the developer (hosting, payments, support)
Practices:    <practice>[:not]
"""


def annotate(store: Store, run: str, app_ids: list[str], out: Path) -> None:
    """Write a blank annotation sheet (one row per segment) plus its codebook."""
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["app_id", "text_sha", "seg", "text", "codes", "note"])
        for app_id in app_ids:
            app = store.db.execute("SELECT text_sha FROM apps WHERE run = ? AND app_id = ?", (run, app_id)).fetchone()
            if not app or not app["text_sha"]:
                raise SystemExit(f"{app_id}: no policy text in run {run}")
            text = store.get_blob(app["text_sha"]).decode()
            for i, (s, e) in enumerate(segment(text)):
                w.writerow([app_id, app["text_sha"], i, text[s:e], "", ""])
    lines = [CODEBOOK_HEADER, "Categories (and the data types they cover):"]
    lines += [f"  {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()]
    lines += [f"  {GENERIC}: personal data mentioned without saying which kind", "", "Practices:"]
    lines += [f"  {name}: {definition}" for name, definition in PRACTICES.items()]
    Path(str(out) + ".codebook.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote {out} and {out}.codebook.txt")


def cohen_kappa(a: list[bool], b: list[bool]) -> float | None:
    n = len(a)
    if n == 0:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return None if pe == 1 else (po - pe) / (1 - pe)


def _sheet(path: Path) -> dict[tuple[str, int], set]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        claims = parse_codes(r["codes"])
        out[(r["app_id"], int(r["seg"]))] = {(c.kind, c.category or c.practice, c.action, c.polarity) for c in claims}
    return out


def kappa_report(path_a: Path, path_b: Path) -> dict:
    a, b = _sheet(path_a), _sheet(path_b)
    if a.keys() != b.keys():
        raise SystemExit("the two sheets do not cover the same segments")
    units = sorted(a)
    labels = sorted(set().union(*a.values(), *b.values()), key=str)
    per_label, all_a, all_b = {}, [], []
    for lab in labels:
        va = [lab in a[u] for u in units]
        vb = [lab in b[u] for u in units]
        all_a += va
        all_b += vb
        per_label[lab] = {"kappa": cohen_kappa(va, vb), "agreement": sum(x == y for x, y in zip(va, vb)) / len(units),
                          "prevalence": (sum(va) + sum(vb)) / (2 * len(units))}
    pooled = cohen_kappa(all_a, all_b)
    print(f"{len(units)} segments, {len(labels)} labels; pooled kappa {_fmt(pooled)}")
    for lab, s in per_label.items():
        name = ":".join(str(x) for x in lab[1:] if x)
        print(f"  {name:45} kappa {_fmt(s['kappa'])}  agreement {_fmt(s['agreement'])}  prevalence {_fmt(s['prevalence'])}")
    return {"pooled_kappa": pooled, "labels": {str(k): v for k, v in per_label.items()}}
