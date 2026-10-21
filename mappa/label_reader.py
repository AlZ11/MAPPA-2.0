"""Part 3, the label reader: a frozen Data Safety page -> evidence rows.

Each data type in the label becomes one flow row, e.g.
  share   Personal info > Email address   purposes: App functionality, Account management
  collect Financial info > User payment info (optional)
The label is the developer's own claim, not ground truth: these rows say what the label *says*.

Status of every read: ok | not_found (app removed) | no_sections (system app) | error (page
or parser problem). Rows are written only for ok. A broken page must never turn into
"No data collected", because Build 2 would then report contradictions that do not exist.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import astuple

from bs4 import BeautifulSoup

from . import play
from .collect import GENRES
from .store import Store
from .vocab import CATEGORIES, EMPTY_TITLES, LABEL_PRACTICES, SECTION_TITLES, Claim, label_key, split_purposes

LABEL_FIELD = 137  # position of the Data Safety label in the page's JSON (Play's field 138)


def parse_label(html: str) -> tuple[str, list[Claim], list[str]]:
    """(status, claims, notes). Notes list unknown strings and error reasons. Never raises."""
    if play.is_soft_404(html):
        return "not_found", [], []
    blob = play.find_blob(html, lambda b: isinstance(play.jget(b, 1, 2, LABEL_FIELD), list))
    if blob is None:
        return "error", [], ["no label data in the page"]
    node = play.jget(blob, 1, 2, LABEL_FIELD)
    sections = play.jget(node, 4)
    if not sections:
        return "no_sections", [], []  # e.g. Google Play services: exempt system apps have no label

    claims, notes = [], []
    try:
        for section in sections:
            title, categories = section[1] or "", section[0]
            key = label_key(title)
            if key in SECTION_TITLES and categories:
                claims += _section_claims(title, SECTION_TITLES[key], categories, notes)
            elif key in EMPTY_TITLES and not categories:
                # The developer's explicit "none". For sharing, Play's own definition excludes service
                # providers, legal requests and anonymised data, hence recipient third_party.
                action = EMPTY_TITLES[key]
                claims.append(Claim(kind="flow", category="*", action=action, polarity="deny",
                                    recipient="third_party" if action == "share" else None, quote=title))
            else:
                return "error", [], [f"unexpected section {title!r} ({'with' if categories else 'without'} data)"]
        for practice in play.jget(node, 9, 2) or []:
            text = play.jget(practice, 1) or ""
            name, polarity = LABEL_PRACTICES.get(label_key(text), (text, "affirm"))
            if label_key(text) not in LABEL_PRACTICES:
                notes.append(f"unknown practice {text!r}")
            claims.append(Claim(kind="practice", practice=name, polarity=polarity, quote=f"Security practices › {text}"))
    except (TypeError, IndexError, AttributeError) as e:  # the JSON layout moved: fail loudly, write nothing
        return "error", [], [f"unexpected label layout: {type(e).__name__}: {e}"]

    problems = cross_check(html, claims)
    if problems:
        return "error", [], problems
    return "ok", claims, notes


def _section_claims(title: str, action: str, categories: list, notes: list[str]) -> list[Claim]:
    claims = []
    for cat in categories:
        name = play.jget(cat, 0, 1)
        if name not in CATEGORIES:
            notes.append(f"unknown category {name!r}")
        for entry in play.jget(cat, 4) or []:
            subtype, optional, purposes_text = (list(entry) + [None, None, None])[:3]
            if subtype not in CATEGORIES.get(name, ()):
                notes.append(f"unknown data type {name!r} > {subtype!r}")
            purposes, leftover = split_purposes(purposes_text or "")
            if leftover:
                notes.append(f"unknown purpose {leftover!r}")
            claims.append(Claim(kind="flow", category=name, subtype=subtype, action=action, polarity="affirm",
                                recipient="third_party" if action == "share" else None, purposes=purposes,
                                optional=bool(optional), quote=f"{title} › {name} › {subtype}"))
    return claims


def cross_check(html: str, claims: list[Claim]) -> list[str]:
    """Check the JSON against what the page visibly shows. A mismatch means the parser is wrong."""
    problems = []
    for action, prefix in (("share", "sLabel"), ("collect", "cLabel")):
        regions = len(set(re.findall(rf'id="{prefix}(\d+)"', html)))
        categories = len({c.category for c in claims if c.action == action and c.polarity == "affirm"})
        if regions != categories:
            problems.append(f"{action}: page shows {regions} categories, JSON has {categories}")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    visible = label_key(soup.get_text(" "))
    for c in claims:
        for text in (c.quote or "").split(" › "):
            if text != "Security practices" and label_key(text) not in visible:
                problems.append(f"{text!r} is in the JSON but not on the page")
    return problems


def canonical_sha(claims: list[Claim]) -> str:
    """Fingerprint of what the label says. Play shuffles category order on every request, so the raw
    page hash changes even when the label did not; this one only changes when the content does."""
    rows = sorted(json.dumps(astuple(c)) for c in claims)
    return hashlib.sha256("\n".join(rows).encode()).hexdigest()


def read_labels(store: Store, run: str) -> Counter:
    """Parse every frozen label of a run into evidence rows. Re-running replaces earlier reads."""
    statuses, notes = Counter(), Counter()
    for app in store.apps(run, GENRES):
        fetch = store.fetch(run, app["app_id"], "label")
        body = store.body(run, app["app_id"], "label")
        if body is None:
            statuses["not_fetched"] += 1
            store.update_app(run, app["app_id"], label_status="not_fetched")
            continue
        status, claims, app_notes = parse_label(body.decode("utf-8", "replace"))
        notes.update(app_notes)
        statuses[status] += 1
        store.save_read(run, app["app_id"], "label", "play_label", claims, input_sha=fetch["sha256"],
                        stats={"dropped": dict(Counter(app_notes))}, status=status)
        store.update_app(run, app["app_id"], label_status=status,
                         label_sha=canonical_sha(claims) if status == "ok" else None)
    print(f"labels in run {run}: {dict(statuses)}")
    for note, n in notes.most_common():
        print(f"  {n:4} x {note}")
    return statuses
