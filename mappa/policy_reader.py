"""Part 2, the policy reader: privacy policy text -> the same evidence rows as the label.

Two extractors share one interface, extract(text, spans) -> (claims, stats):
  Baseline  classical text classifiers (TF-IDF + logistic regression), trained on APP-350 and OPP-115.
            Runs anywhere in seconds; the yardstick the LLM has to beat.
  LLM       a local open model (see llm.py), reading windows of the policy into fixed forms.
Every claim carries a quote copied from the policy and the character offsets where it was found.
The evaluation (evaluate.py) calls exactly the same extract() as read_policies() below.
"""

import json
import pickle
import time
from bisect import bisect_right
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from . import llm
from .collect import GENRES
from .store import Store
from .text import find_quote, normalise, segment, sentences
from .vocab import (ACTIONS, CATEGORIES, DATA_VERBS, GENERIC, HEALTH_TERMS, PRACTICES, PURPOSES, RECIPIENTS, RULES,
                    SHARE_WORDS, Claim)

PROMPT_VERSION = "p1"  # bump on any prompt change; freeze before running on a test split


class Extractor(Protocol):
    name: str
    model: str

    def extract(self, text: str, spans: list[tuple[int, int]]) -> tuple[list[Claim], dict]:
        ...


# ---------------------------------------------------------------- baseline
# Which corpus teaches which label. Each label learns only from a corpus that annotates it: training
# "Health and fitness" on APP-350, which never labels health, would teach it that health text is a negative.
FLOW_TEACHERS = {
    "Contacts": "app350", "Location": "app350", "Device or other IDs": "app350",
    "Personal info": "opp115", "Health and fitness": "opp115", "Financial info": "opp115",
    "App activity": "opp115", "Generic personal data": "opp115",
}
PRACTICE_TEACHER = "opp115"
MIN_POSITIVES = 10  # fewer training examples than this and a label is not learned at all
THRESHOLD = 0.5


def _labels(c: Claim) -> list[str]:
    """Training labels a gold claim stands for: 'flow|Location|collect' or 'practice|retention'."""
    if c.kind == "flow" and c.polarity == "affirm" and c.action:
        return [f"flow|{c.category}|{'share' if c.action == 'sell' else c.action}"]
    if c.kind == "practice" and c.polarity == "affirm":
        return [f"practice|{c.practice}"]
    return []


def _vectorizer() -> TfidfVectorizer:
    # stop_words stays None: English stop-word lists delete "not" and "never", which carry the denials.
    return TfidfVectorizer(ngram_range=(1, 2), binary=True, min_df=2, sublinear_tf=True)


def _classifier() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", max_iter=2000, C=4.0)


class Baseline:
    name = "baseline"
    model = "tfidf-logreg"

    def __init__(self, vectorizers: dict, classifiers: dict, polarity: tuple, info: dict):
        self.vectorizers = vectorizers  # corpus -> fitted TfidfVectorizer
        self.classifiers = classifiers  # label -> (corpus, fitted LogisticRegression)
        self.polarity = polarity  # (vectorizer, classifier): is this sentence a denial?
        self.info = info

    @classmethod
    def train(cls, corpora: dict[str, list]) -> "Baseline":
        """corpora: {"app350": [GoldDoc...], "opp115": [GoldDoc...]}, training splits only."""
        vectorizers, classifiers, info = {}, {}, {"skipped": {}}
        for corpus, docs in corpora.items():
            texts = [d.text[s:e] for d in docs for s, e in d.spans]
            gold = [set(lab for c in g for lab in _labels(c)) for d in docs for g in d.gold]
            vec = _vectorizer().fit(texts)
            X = vec.transform(texts)
            vectorizers[corpus] = vec
            wanted = [f"flow|{cat}|{act}" for cat, t in FLOW_TEACHERS.items() if t == corpus for act in ("collect", "share")]
            if corpus == PRACTICE_TEACHER:
                wanted += sorted({lab for g in gold for lab in g if lab.startswith("practice|")})
            for label in wanted:
                y = [label in g for g in gold]
                if sum(y) < MIN_POSITIVES:
                    info["skipped"][label] = sum(y)
                    continue
                classifiers[label] = (corpus, _classifier().fit(X, y))
        # Denials: sentences APP-350 marks NOT_PERFORMED vs sentences it marks PERFORMED only.
        sents, neg = [], []
        for d in corpora["app350"]:
            for (s, e), g in zip(d.spans, d.gold):
                if g:
                    sents.append(d.text[s:e])
                    neg.append(any(c.polarity == "deny" for c in g))
        pol_vec = _vectorizer().fit(sents)
        polarity = (pol_vec, _classifier().fit(pol_vec.transform(sents), neg))
        info["labels"] = sorted(classifiers)
        return cls(vectorizers, classifiers, polarity, info)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(pickle.dumps(self))

    @staticmethod
    def load(path: Path) -> "Baseline":
        if not path.exists():
            raise SystemExit(f"no trained baseline at {path}: run `python -m mappa train` first")
        return pickle.loads(path.read_bytes())

    def extract(self, text: str, spans: list[tuple[int, int]]) -> tuple[list[Claim], dict]:
        t0 = time.process_time()
        claims = []
        segs = [text[s:e] for s, e in spans]
        X = {corpus: vec.transform(segs) for corpus, vec in self.vectorizers.items()}
        scores = {label: clf.predict_proba(X[corpus])[:, 1] for label, (corpus, clf) in self.classifiers.items()}
        pol_vec, pol_clf = self.polarity
        for i, (s, e) in enumerate(spans):
            hits = [label for label in scores if scores[label][i] >= THRESHOLD]
            sents = sentences(text, s, e)
            rule_hits = [(name, span) for name, rx in RULES.items() for span in sents if rx.search(text[span[0]:span[1]])]
            health = HEALTH_TERMS.search(text, s, e)
            if not hits and not rule_hits and not health:
                continue
            sent_texts = [text[a:b] for a, b in sents]
            denial = pol_clf.predict_proba(pol_vec.transform(sent_texts))[:, 1] if sent_texts else []
            for label in hits:
                # The quote is the sentence the same classifier is most confident about.
                corpus, clf = self.classifiers[label]
                best = int(clf.predict_proba(self.vectorizers[corpus].transform(sent_texts))[:, 1].argmax())
                a, b = sents[best]
                kind, *rest = label.split("|")
                polarity = "deny" if kind == "flow" and denial[best] >= THRESHOLD else "affirm"
                if kind == "flow":
                    claims.append(Claim(kind="flow", category=rest[0], action=rest[1], polarity=polarity,
                                        quote=text[a:b], seg=i, start=a, end=b))
                else:
                    claims.append(Claim(kind="practice", practice=rest[0], quote=text[a:b], seg=i, start=a, end=b))
            if not any(label.startswith("flow|Health and fitness|") for label in self.classifiers):
                for j, (a, b) in enumerate(sents):  # keyword rule, see vocab.HEALTH_TERMS
                    sentence = text[a:b]
                    if HEALTH_TERMS.search(sentence) and DATA_VERBS.search(sentence):
                        action = "share" if SHARE_WORDS.search(sentence) else "collect"
                        claims.append(Claim(kind="flow", category="Health and fitness", action=action,
                                            polarity="deny" if denial[j] >= THRESHOLD else "affirm",
                                            quote=sentence, seg=i, start=a, end=b))
                        break
            first = {}
            for name, span in rule_hits:
                first.setdefault(name, span)  # first matching sentence per rule and segment
            for name, (a, b) in first.items():
                j = sents.index((a, b))
                claims.append(Claim(kind="practice", practice=name, polarity="deny" if denial[j] >= THRESHOLD else "affirm",
                                    quote=text[a:b], seg=i, start=a, end=b))
        return claims, {"seconds": time.process_time() - t0, "dropped": {}}


# ---------------------------------------------------------------- local LLM
SUBTYPES = sorted({sub for subs in CATEGORIES.values() for sub in subs})
FLOW_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "maxItems": 25, "items": {
        "type": "object", "additionalProperties": False,
        "required": ["quote", "category", "subtype", "action", "negated", "recipient", "recipient_name", "purposes"],
        "properties": {
            "quote": {"type": "string"},  # first, so the model copies the text before it labels it
            "category": {"enum": [*CATEGORIES, GENERIC]},
            "subtype": {"enum": [*SUBTYPES, "unspecified"]},
            "action": {"enum": list(ACTIONS)},
            "negated": {"type": "boolean"},
            "recipient": {"enum": ["none", *RECIPIENTS]},
            "recipient_name": {"type": "string"},
            "purposes": {"type": "array", "maxItems": 7, "items": {"enum": list(PURPOSES)}},
        }}}},
}
PRACTICE_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["claims"],
    "properties": {"claims": {"type": "array", "maxItems": 15, "items": {
        "type": "object", "additionalProperties": False, "required": ["quote", "practice", "negated", "detail"],
        "properties": {
            "quote": {"type": "string"},
            "practice": {"enum": list(PRACTICES)},
            "negated": {"type": "boolean"},
            "detail": {"type": "string"},
        }}}},
}

QUOTE_RULES = """quote: copy ONE contiguous passage from the excerpt exactly as written (same words, same order),
at most two sentences. Never paraphrase, never shorten with "...", never correct typos. For a list item, quote the whole item."""

FLOW_PROMPT = f"""You read one excerpt of a mobile health app's privacy policy and turn what it says about personal data into fixed-form claims.

Output one claim for each (data type, action) the excerpt states, AND for each one it explicitly denies.
- {QUOTE_RULES}
- category / subtype: from the list below; subtype "unspecified" when the text does not say which.
- action: collect (the developer obtains or receives the data), share (it is passed to another company or person),
  sell (it is exchanged for money or other value).
- negated: true only when the text says this does NOT happen ("we never sell your data").
- recipient (share/sell only; "none" for collect): service_provider (acts for the developer: hosting, payments, support,
  processing on its behalf), affiliate (same corporate group), authorities (police, courts, regulators),
  third_party (any other company: advertisers, partners, data brokers), unspecified.
- recipient_name: the company the quote names as the recipient, else "".
- purposes: only purposes the quote itself states, from the list below; else [].
If the excerpt says nothing about personal data, return {{"claims": []}}.

Categories and data types:
""" + "\n".join(f"- {cat}: {', '.join(subs)}" for cat, subs in CATEGORIES.items()) + f"""
- {GENERIC}: personal information mentioned without saying which kind ("we collect personal information").
An IP address or advertising ID used to identify a device is Device or other IDs; used to estimate where someone is,
it is Location / Approximate location. Medical, symptom, medication, mood or menstrual data is Health info;
exercise, steps, workouts and sleep are Fitness info.

Purposes: {"; ".join(PURPOSES)}."""

PRACTICE_PROMPT = f"""You read one excerpt of a mobile health app's privacy policy and list the practices it describes.

Output one claim per practice the excerpt states, AND per practice it explicitly denies.
- {QUOTE_RULES}
- practice: one of the names below.
- negated: true only when the text says the practice does NOT happen ("we do not use automated decision-making").
- detail: a short value copied from the quote if it gives one (a retention period, a minimum age, the AI provider's
  name, a contact email or address), else "".
If the excerpt describes none of these, return {{"claims": []}}.

Practices:
""" + "\n".join(f"- {name}: {definition}" for name, definition in PRACTICES.items())

# Two invented examples (never from any test data), one per task, showing the expected shape.
FLOW_EXAMPLE = (
    "We record your workouts and step count to tailor your plan. We share your email address with Mailchimp, "
    "which sends our newsletters for us. We do not sell your personal information.",
    {"claims": [
        {"quote": "We record your workouts and step count to tailor your plan", "category": "Health and fitness",
         "subtype": "Fitness info", "action": "collect", "negated": False, "recipient": "none", "recipient_name": "",
         "purposes": ["Personalization"]},
        {"quote": "We share your email address with Mailchimp, which sends our newsletters for us",
         "category": "Personal info", "subtype": "Email address", "action": "share", "negated": False,
         "recipient": "service_provider", "recipient_name": "Mailchimp", "purposes": ["Developer communications"]},
        {"quote": "We do not sell your personal information", "category": GENERIC, "subtype": "unspecified",
         "action": "sell", "negated": True, "recipient": "unspecified", "recipient_name": "", "purposes": []},
    ]},
)
PRACTICE_EXAMPLE = (
    "We keep your journal entries for 24 months after you close your account. Our AI coach is powered by OpenAI. "
    "The app is not intended for anyone under 16.",
    {"claims": [
        {"quote": "We keep your journal entries for 24 months after you close your account", "practice": "retention",
         "negated": False, "detail": "24 months"},
        {"quote": "Our AI coach is powered by OpenAI", "practice": "ai_processing", "negated": False, "detail": "OpenAI"},
        {"quote": "The app is not intended for anyone under 16", "practice": "children", "negated": False, "detail": "16"},
    ]},
)
TASKS = {"flows": (FLOW_PROMPT, FLOW_SCHEMA, FLOW_EXAMPLE, 1500), "practices": (PRACTICE_PROMPT, PRACTICE_SCHEMA, PRACTICE_EXAMPLE, 600)}


def messages(task: str, excerpt: str) -> list[dict]:
    prompt, _, (example_text, example_answer), _ = TASKS[task]
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Excerpt:\n<<<\n{example_text}\n>>>"},
        {"role": "assistant", "content": json.dumps(example_answer)},
        {"role": "user", "content": f"Excerpt:\n<<<\n{excerpt}\n>>>"},
    ]


def windows(text: str, spans: list[tuple[int, int]], max_words: int) -> list[tuple[int, int]]:
    """Pack consecutive whole segments into windows of about max_words: (first, last) segment index."""
    out, first, words = [], 0, 0
    for i, (s, e) in enumerate(spans):
        n = len(text[s:e].split())
        if i > first and words + n > max_words:
            out.append((first, i - 1))
            first, words = i, 0
        words += n
    if spans:
        out.append((first, len(spans) - 1))
    return out


def verify(items: list, task: str, text: str, win_start: int, win_end: int,
           spans: list[tuple[int, int]]) -> tuple[list[Claim], Counter]:
    """Keep only well-formed items whose quote really is in the window. Everything dropped is counted by reason."""
    kept, dropped = [], Counter()
    window = text[win_start:win_end]
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("quote"), str):
            dropped["schema"] += 1
            continue
        if task == "flows":
            cat, act = item.get("category"), item.get("action")
            purposes = item.get("purposes") or []
            if (cat not in CATEGORIES and cat != GENERIC) or act not in ACTIONS or not isinstance(purposes, list) \
                    or any(p not in PURPOSES for p in purposes) or item.get("recipient") not in ("none", *RECIPIENTS):
                dropped["schema"] += 1
                continue
        elif item.get("practice") not in PRACTICES:
            dropped["schema"] += 1
            continue
        span, why = find_quote(item["quote"], window)
        if why != "ok":
            dropped[why] += 1
            continue
        start, end = win_start + span[0], win_start + span[1]
        quote = text[start:end]  # the policy's own words, not the model's copy of them
        seg = segment_index(spans, start)
        detail = (item.get("recipient_name") if task == "flows" else item.get("detail")) or None
        if detail and normalise(detail)[0] not in normalise(quote)[0]:
            dropped["detail_not_in_quote"] += 1  # the claim stays; only the unsupported detail goes
            detail = None
        polarity = "deny" if item.get("negated") is True else "affirm"
        if task == "flows":
            sub = item.get("subtype")
            if sub not in CATEGORIES.get(cat, ()):
                if sub not in (None, "", "unspecified"):
                    dropped["subtype_reset"] += 1  # e.g. "Precise location" filed under Personal info
                sub = None
            recipient = None if act == "collect" else ("unspecified" if item["recipient"] == "none" else item["recipient"])
            kept.append(Claim(kind="flow", category=cat, subtype=sub, action=act, polarity=polarity, recipient=recipient,
                              detail=detail, purposes=tuple(p for p in PURPOSES if p in purposes),
                              quote=quote, seg=seg, start=start, end=end))
        else:
            kept.append(Claim(kind="practice", practice=item["practice"], polarity=polarity, detail=detail,
                              quote=quote, seg=seg, start=start, end=end))
    return kept, dropped


class LLM:
    name = "llm"

    def __init__(self, model: str | None = None, base_url: str | None = None, *, tasks=("flows", "practices"),
                 rep: int = 0, window_words: int = 1200, cache_dir: Path | None = None):
        self.model = model or llm.DEFAULT_MODEL
        self.base_url = base_url or llm.DEFAULT_URL
        self.tasks, self.rep, self.window_words, self.cache_dir = tasks, rep, window_words, cache_dir

    def extract(self, text: str, spans: list[tuple[int, int]]) -> tuple[list[Claim], dict]:
        stats = {"seconds": 0.0, "prompt_tokens": 0, "completion_tokens": 0, "windows": 0, "failed_windows": 0,
                 "kept": 0, "modes": Counter()}
        dropped, claims = Counter(), []
        for first, last in windows(text, spans, self.window_words):
            start, end = spans[first][0], spans[last][1]
            for task in self.tasks:
                _, schema, _, max_tokens = TASKS[task]
                stats["windows"] += 1
                try:
                    res = llm.chat_json(messages(task, text[start:end]), schema, model=self.model, base_url=self.base_url,
                                        max_tokens=max_tokens, rep=self.rep, cache_dir=self.cache_dir)
                except llm.LLMError as e:
                    stats["failed_windows"] += 1
                    dropped["llm_error"] += 1
                    stats.setdefault("errors", []).append(str(e)[:200])
                    continue
                stats["seconds"] += res["seconds"]
                stats["prompt_tokens"] += res["usage"]["prompt_tokens"]
                stats["completion_tokens"] += res["usage"]["completion_tokens"]
                stats["modes"][res["mode"]] += 1
                if res["data"] is None:
                    stats["failed_windows"] += 1
                    dropped["truncated_output" if res["finish_reason"] == "length" else "unparseable"] += 1
                    continue
                if res["finish_reason"] == "length":
                    dropped["truncated_output"] += 1  # completed claims kept; the model may have had more to say
                # A prompt far smaller than what we sent means the server cut it (Ollama's default context is small).
                if res["usage"]["prompt_tokens"] and res["usage"]["prompt_tokens"] < 0.7 * (end - start) / 4:
                    dropped["input_maybe_truncated"] += 1
                kept, why = verify(res["data"].get("claims"), task, text, start, end, spans)
                claims += kept
                dropped.update(why)
        unique = list(dict.fromkeys(claims))  # identical claims from the same place count once
        stats["kept"] = len(unique)
        stats["dropped"] = dict(dropped)
        stats["modes"] = dict(stats["modes"])
        return unique, stats


# ---------------------------------------------------------------- reading a run's policies
def segment_index(spans: list[tuple[int, int]], pos: int) -> int:
    """The segment a character position belongs to (the last one starting at or before it)."""
    return max(0, bisect_right([s for s, _ in spans], pos) - 1)


def segment_of(claims: list[Claim], spans: list[tuple[int, int]]) -> list[Claim]:
    """Fill in each claim's segment from where its quote was found."""
    return [replace(c, seg=segment_index(spans, c.start)) if c.seg is None and c.start is not None else c
            for c in claims]


def read_policies(store: Store, run: str, extractor: Extractor, rep: int = 0) -> Counter:
    """Run an extractor over every frozen policy text of a run and store the rows."""
    counts = Counter()
    apps = [a for a in store.apps(run, GENRES) if a["text_sha"]]
    for i, app in enumerate(apps, 1):
        text = store.get_blob(app["text_sha"]).decode()
        spans = segment(text)
        claims, stats = extractor.extract(text, spans)
        status = "partial" if stats.get("failed_windows") else "ok"
        store.save_read(run, app["app_id"], "policy", extractor.name, segment_of(claims, spans),
                        input_sha=app["text_sha"], stats=stats, model=extractor.model,
                        prompt_version=PROMPT_VERSION if extractor.name == "llm" else "", rep=rep, status=status)
        counts[status] += 1
        dropped = sum(stats.get("dropped", {}).values())
        print(f"[{i}/{len(apps)}] {app['app_id']}: {len(claims)} claims"
              f"{f', {dropped} dropped' if dropped else ''} ({stats.get('seconds', 0):.1f}s)", flush=True)
    return counts
