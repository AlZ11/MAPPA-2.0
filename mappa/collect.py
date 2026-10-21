"""Part 1, the collector: find Australian health apps on Google Play and freeze their claims.

A run is one wave of data collection:
  discover   searches Play and lists candidate apps (run kind "discover");
  snapshot   fetches, for each app, the details page, then (Health & Fitness / Medical only) the
             Data Safety page and the privacy policy (run kind "snapshot").
A snapshot taken --from an earlier snapshot re-fetches exactly the same apps (a panel). So the
December and January waves compare the same apps, and apps that disappeared are a finding.
"""

import csv
import hashlib
import io
import os
import re
from urllib.parse import urlsplit

from . import play
from .store import Store
from .text import html_to_text, policy_date

GENRES = {"HEALTH_AND_FITNESS", "MEDICAL"}
CATEGORY_PAGES = ("HEALTH_AND_FITNESS",)  # MEDICAL's category page lists its apps via JavaScript only

# Seed searches. Play's web search returns ~30 apps per query; the union is the sample frame.
QUERIES = (
    # conditions
    "diabetes", "blood sugar tracker", "blood pressure", "heart rate monitor", "asthma", "migraine",
    "epilepsy seizure diary", "arthritis", "chronic pain", "cancer support", "kidney disease", "IBS",
    "endometriosis", "PCOS", "thyroid", "dementia", "ADHD", "autism", "allergy", "skin check",
    # mental health
    "mental health", "depression", "anxiety", "mood tracker", "therapy", "CBT", "mindfulness",
    "meditation", "sleep", "insomnia", "sleep tracker", "stress relief", "eating disorder recovery",
    "quit smoking", "alcohol tracker", "addiction recovery", "suicide prevention", "journal mental health",
    # women's and family health
    "period tracker", "fertility tracker", "ovulation", "pregnancy", "baby tracker", "breastfeeding",
    "menopause", "pill reminder", "contraception", "parenting newborn",
    # fitness and nutrition
    "fitness", "workout", "running", "cycling", "step counter", "pedometer", "gym", "yoga", "pilates",
    "home workout", "weight loss", "calorie counter", "diet", "nutrition", "intermittent fasting",
    "water reminder", "meal planner", "macro tracker",
    # services and care
    "telehealth", "telehealth australia", "online doctor australia", "GP appointment", "book doctor",
    "pharmacy", "prescription", "medication reminder", "Medicare", "my health record", "NDIS",
    "Bupa", "Medibank", "nib health", "HCF", "private health insurance", "pathology results",
    "dental", "physiotherapy", "hearing test", "eye test", "first aid", "symptom checker",
    "healthdirect", "hospital", "nurse", "vaccination", "sexual health", "men's health",
    "women's health", "aged care", "carer",
    # devices and AI
    "AI health assistant", "AI doctor", "health coach", "ECG", "smartwatch health", "glucose monitor",
    "blood test results", "health tracker",
)

MIN_POLICY_WORDS = 300
POLICY_WORDS = re.compile(r"personal (?:information|data)|we (?:may )?collect|third[- ]part", re.I)


def discover(store: Store, run: str, queries=QUERIES) -> int:
    """Search Play and register every app found. Returns the number of distinct apps."""
    store.start_run(run, "discover", {"queries": list(queries), "categories": list(CATEGORY_PAGES)})
    pages = [(f"search:{q}", play.search_url(q)) for q in queries]
    pages += [(f"category:{c}", play.category_url(c)) for c in CATEGORY_PAGES]
    for i, (kind, url) in enumerate(pages, 1):
        _fetch(store, run, "*", kind, url)
        html = (store.body(run, "*", kind) or b"").decode("utf-8", "replace")
        for rank, app_id in enumerate(play.app_ids(html), 1):
            store.add_app(run, app_id, kind, rank)
        print(f"[{i}/{len(pages)}] {kind}: {len(store.apps(run))} apps so far", flush=True)
    return len(store.apps(run))


def snapshot(store: Store, run: str, source_run: str, *, limit: int | None = None, render: bool = True,
             only: list[str] | None = None) -> None:
    """Freeze details, label and policy for every app of source_run. Safe to re-run: it resumes."""
    source_kind = store.run_kind(source_run)
    if source_kind is None:
        raise SystemExit(f"unknown run {source_run!r}")
    # From a discovery run: all candidates. From a snapshot run: its in-genre apps (the panel), plus any
    # whose details failed there (genre unknown), so no app silently leaves the panel.
    panel = source_kind == "snapshot"
    rows = [r for r in store.apps(source_run) if not panel or r["genre"] in GENRES or r["genre"] is None]
    store.start_run(run, "snapshot", {"from": source_run, "limit": limit, "render": render, "apps": only})
    if only:  # named apps first, even if discovery never found them
        known = {r["app_id"]: r for r in rows}
        for app_id in only:
            r = known.get(app_id)
            store.add_app(run, app_id, r["found_via"] if r else "named", r["rank"] if r else 0)
    else:
        for r in rows[:limit]:
            store.add_app(run, r["app_id"], r["found_via"], r["rank"])

    apps = [a for a in store.apps(run) if not only or a["app_id"] in only][:limit]
    for i, app in enumerate(apps, 1):
        app_id = app["app_id"]
        line = snapshot_app(store, run, app_id, render, panel)
        print(f"[{i}/{len(apps)}] {app_id}: {line}", flush=True)


def _has_play_data(body: bytes) -> bool:
    """Play sometimes answers 200 with an empty shell page. Those must be retried, not frozen as done."""
    return b"AF_initDataCallback" in body or play.is_soft_404(body.decode("utf-8", "replace"))


def _fetch(store: Store, run: str, app_id: str, kind: str, url: str, tries: int = 3) -> None:
    if not store.pending(run, app_id, kind):
        return
    for _ in range(tries):
        f = play.get(url)
        if f.status != 200 or kind in ("policy", "policy_rendered") or _has_play_data(f.body):
            break
        f = f._replace(status=0, note="HTTP 200 but no app data in the page")
    store.record_fetch(run, app_id, kind, url=url, final_url=f.final_url, status=f.status,
                       content_type=f.content_type, body=f.body, note=f.note)


def snapshot_app(store: Store, run: str, app_id: str, render: bool, panel: bool = False) -> str:
    _fetch(store, run, app_id, "details", play.details_url(app_id))
    det = store.fetch(run, app_id, "details")
    if det["status"] == 404:
        store.update_app(run, app_id, label_status="not_found")
        return "removed from Play"
    html = store.body(run, app_id, "details")
    if html is None:
        return f"details failed ({det['status']}, {det['note']})"
    meta = play.parse_details(html.decode("utf-8", "replace"))
    store.update_app(run, app_id, **meta)
    note = ""
    if meta["genre"] not in GENRES:
        if not panel:
            return f"skipped (genre {meta['genre']})"
        note = f"; category is now {meta['genre']} (kept: panel)"

    _fetch(store, run, app_id, "label", play.datasafety_url(app_id))
    if not meta["policy_url"]:
        store.update_app(run, app_id, policy_flags="no_policy_url")
        return f"label fetched; NO privacy policy link{note}"
    fetch_policy(store, run, app_id, meta["policy_url"], render)
    app = store.db.execute("SELECT policy_flags FROM apps WHERE run = ? AND app_id = ?", (run, app_id)).fetchone()
    return f"label + policy fetched ({app['policy_flags'] or 'ok'}){note}"


def fetch_policy(store: Store, run: str, app_id: str, url: str, render: bool) -> None:
    """Plain download first. If that yields almost no text (a page built by JavaScript), render it."""
    _fetch(store, run, app_id, "policy", url)
    text = policy_text(store, run, app_id)
    if render and len(text.split()) < MIN_POLICY_WORDS and _readable(store.fetch(run, app_id, "policy")) == "html":
        if store.pending(run, app_id, "policy_rendered"):
            try:
                final_url, html = render_page(url)
                store.record_fetch(run, app_id, "policy_rendered", url=url, final_url=final_url, status=200,
                                   content_type="text/html", body=html)
            except Exception as e:  # browser failures must not stop a 1,000-app crawl
                store.record_fetch(run, app_id, "policy_rendered", url=url, final_url=url, status=0,
                                   content_type="", body=b"", note=f"{type(e).__name__}: {e}"[:300])
        text = policy_text(store, run, app_id)
    flags = policy_flags(store.fetch(run, app_id, "policy"), text)
    sha = store.put_blob(text.encode()) if text else None
    store.update_app(run, app_id, text_sha=sha, policy_date=policy_date(text), policy_flags=",".join(flags) or None)


def policy_text(store: Store, run: str, app_id: str) -> str:
    """The policy's visible text: from the plain download, or the rendered page if that has more."""
    best = ""
    for kind in ("policy", "policy_rendered"):
        body = store.body(run, app_id, kind)
        how = _readable(store.fetch(run, app_id, kind))
        if body and how:
            if how == "html":
                text = html_to_text(body)
            else:
                text = "\n".join(" ".join(ln.split()) for ln in body.decode("utf-8", "replace").splitlines() if ln.strip())
            if len(text) > len(best):
                best = text
    return best


def render_page(url: str) -> tuple[str, bytes]:
    """Load the page in headless Chromium and return (final URL, HTML after scripts ran)."""
    from playwright.sync_api import sync_playwright  # optional dependency, imported only when needed

    with sync_playwright() as p:
        # MAPPA_CHROMIUM points at an existing Chromium when Playwright's own download is unavailable.
        browser = p.chromium.launch(headless=True, executable_path=os.environ.get("MAPPA_CHROMIUM") or None)
        try:
            page = browser.new_page(user_agent=play.USER_AGENT)
            page.goto(url, wait_until="networkidle", timeout=45_000)
            return page.url, page.content().encode()
        finally:
            browser.close()


def _readable(fetch) -> str | None:
    """'html', 'text' (plain text, e.g. a PRIVACY.md), or None when we cannot read this content type."""
    if fetch is None or fetch["status"] != 200:
        return None
    content_type = (fetch["content_type"] or "").lower()
    if "html" in content_type or not content_type:
        return "html"
    return "text" if content_type.startswith("text/") else None


def policy_flags(fetch, text: str) -> list[str]:
    """Why a policy may not be usable. These are findings to count, not errors."""
    if fetch is None or fetch["status"] != 200:
        return [f"http_{fetch['status'] if fetch else 'none'}"]
    flags = []
    url = (fetch["final_url"] or "").lower()
    if "pdf" in (fetch["content_type"] or "") or url.endswith(".pdf"):
        flags.append("pdf")
    elif _readable(fetch) is None:
        flags.append(f"unsupported_type:{fetch['content_type']}")
    if urlsplit(url).hostname in ("docs.google.com", "drive.google.com", "sites.google.com"):
        flags.append("google_doc")
    if text and "privacy" not in text.lower() and len(POLICY_WORDS.findall(text)) < 3:
        flags.append("not_a_policy")  # e.g. a store page or a language picker
    lines = text.split("\n")
    if text and len(text.split()) < 1500 and sum("cookie" in ln.lower() for ln in lines) >= 0.3 * len(lines):
        flags.append("cookie_panel")  # the real policy loads behind a click, or in another frame
    if len(text.split()) < MIN_POLICY_WORDS:
        flags.append("short")
    return flags


def status(store: Store, run: str, manifest: str | None = None) -> None:
    """Print what a run holds; optionally write a CSV manifest of every fetch and print its hash."""
    rows = store.db.execute("SELECT kind, status, COUNT(*) n FROM fetches WHERE run = ? "
                            "GROUP BY CASE WHEN kind LIKE 'search:%' THEN 'search' ELSE kind END, status",
                            (run,)).fetchall()
    apps = store.apps(run)
    in_genre = [a for a in apps if a["genre"] in GENRES]
    print(f"run {run} ({store.run_kind(run)}): {len(apps)} apps, {len(in_genre)} in Health & Fitness / Medical")
    for r in rows:
        kind = "search" if r["kind"].startswith("search:") else r["kind"]
        print(f"  fetch {kind:16} status {r['status']}: {r['n']}")
    flags: dict[str, int] = {}
    for a in in_genre:
        for f in (a["policy_flags"] or "").split(","):
            if f:
                flags[f] = flags.get(f, 0) + 1
    labels: dict[str, int] = {}
    for a in [a for a in apps if a["genre"] in GENRES or a["label_status"]]:
        labels[a["label_status"] or "not read"] = labels.get(a["label_status"] or "not read", 0) + 1
    print(f"  policy flags: {flags or 'none'}")
    print(f"  label status: {labels}")
    pending = store.db.execute("SELECT COUNT(*) n FROM fetches WHERE run = ? AND status NOT IN (200, 404)",
                               (run,)).fetchone()["n"]
    print(f"  fetches still failing (will retry, max 3 attempts): {pending}")
    if manifest:
        buf = io.StringIO()
        cur = store.db.execute("SELECT run, app_id, kind, url, final_url, status, sha256, bytes, fetched_at "
                               "FROM fetches WHERE run = ? ORDER BY app_id, kind", (run,))
        w = csv.writer(buf)
        w.writerow([d[0] for d in cur.description])
        w.writerows(cur.fetchall())
        data = buf.getvalue().encode()
        with open(manifest, "wb") as f:
            f.write(data)
        print(f"  manifest {manifest}: sha256 {hashlib.sha256(data).hexdigest()}")
