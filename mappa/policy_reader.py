"""Part 2, the policy reader: privacy policy text -> the same evidence rows as the label.

Two extractors share one interface, extract(text, spans) -> (claims, stats):
  Baseline  classical text classifiers (TF-IDF + logistic regression), trained on APP-350 and OPP-115.
            Runs anywhere in seconds; the yardstick the LLM has to beat.
  LLM       a local open model (see llm.py), reading windows of the policy into fixed forms.
Every claim carries a quote copied from the policy and the character offsets where it was found.
The evaluation (evaluate.py) calls exactly the same extract() as read_policies() below.
"""

import pickle
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Protocol

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .collect import GENRES
from .store import Store
from .text import segment, sentences
from .vocab import DATA_VERBS, HEALTH_TERMS, RULES, SHARE_WORDS, Claim

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


# ---------------------------------------------------------------- reading a run's policies
def segment_of(claims: list[Claim], spans: list[tuple[int, int]]) -> list[Claim]:
    """Fill in each claim's segment from where its quote was found."""
    out = []
    for c in claims:
        if c.seg is None and c.start is not None:
            seg = next((i for i, (s, e) in enumerate(spans) if s <= c.start < e), None)
            c = replace(c, seg=seg)
        out.append(c)
    return out


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
