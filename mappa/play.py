"""A minimal Google Play client: polite fetching plus the parsing that both page types share.

Play pages carry their data twice: as visible HTML and as JSON inside
<script>AF_initDataCallback({key: 'ds:N', data: [...]})</script> blocks. MAPPA reads the JSON.
It is the same data the page draws from, and it does not depend on CSS class names, which change
whenever Google rebuilds the site.
"""

import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Any, Callable, NamedTuple
from urllib.parse import quote, urlsplit

import requests

BASE = "https://play.google.com/store"
LOCALE = "hl=en&gl=AU"
USER_AGENT = os.environ.get(
    "MAPPA_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0 Safari/537.36",
)
# Seconds between requests to the same host. Play's robots.txt disallows the Data Safety and
# search pages, so the crawl stays slow, fetches each page once per wave, and never logs in.
MIN_INTERVAL = {"play.google.com": 2.5}
DEFAULT_INTERVAL = 1.0
MAX_429_IN_A_ROW = 5


class Fetched(NamedTuple):
    status: int  # HTTP status, or 0 when the request never got an answer
    final_url: str  # after redirects: where the content actually came from
    content_type: str
    body: bytes
    note: str | None = None


class RateLimited(RuntimeError):
    pass


_last_request: dict[str, float] = {}
_429_streak = 0


def _wait_turn(host: str) -> None:
    interval = MIN_INTERVAL.get(host, DEFAULT_INTERVAL) * (1 + 0.3 * random.random())
    wait = _last_request.get(host, 0.0) + interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    _last_request[host] = time.monotonic()


def get(url: str, *, timeout: int = 30, max_bytes: int = 20_000_000, tries: int = 3) -> Fetched:
    """GET with a per-host delay, backoff on 429/5xx and a size cap. Never raises on HTTP errors."""
    global _429_streak
    host = urlsplit(url).hostname or ""
    headers = {"User-Agent": USER_AGENT, "Accept-Language": "en-AU,en;q=0.9"}
    result = Fetched(0, url, "", b"", "not attempted")
    for attempt in range(tries):
        _wait_turn(host)
        try:
            with requests.get(url, headers=headers, timeout=timeout, stream=True) as r:
                body = b"".join(r.iter_content(65536, decode_unicode=False))[:max_bytes]
                result = Fetched(r.status_code, r.url, r.headers.get("content-type", ""), body)
        except requests.RequestException as e:
            result = Fetched(0, url, "", b"", f"{type(e).__name__}: {e}"[:300])
            time.sleep(2 ** attempt)
            continue
        if result.status == 429:
            _429_streak += 1
            if _429_streak >= MAX_429_IN_A_ROW:
                raise RateLimited(f"{MAX_429_IN_A_ROW} rate-limit replies in a row from {host}; stopping. "
                                  "Re-run later: the crawl resumes where it stopped.")
        else:
            _429_streak = 0
        if result.status == 429 or result.status >= 500:
            time.sleep(10 * 2 ** attempt)
            continue
        return result
    return result


def details_url(app_id: str) -> str:
    return f"{BASE}/apps/details?id={app_id}&{LOCALE}"


def datasafety_url(app_id: str) -> str:
    return f"{BASE}/apps/datasafety?id={app_id}&{LOCALE}"


def search_url(query: str) -> str:
    return f"{BASE}/search?q={quote(query)}&c=apps&{LOCALE}"


def category_url(category: str) -> str:
    return f"{BASE}/apps/category/{category}?{LOCALE}"


def app_ids(html: str) -> list[str]:
    """Every app id linked from a page, in page order, without repeats."""
    return list(dict.fromkeys(re.findall(r"/store/apps/details\?id=([A-Za-z0-9_.]+)", html)))


def blobs(html: str) -> list[Any]:
    """The JSON payload of every AF_initDataCallback block on the page."""
    out = []
    for m in re.finditer(r"AF_initDataCallback\((\{.*?\})\);</script>", html, re.S):
        data = re.search(r"data:(.*), sideChannel: \{\}\}$", m.group(1), re.S)
        if data:
            try:
                out.append(json.loads(data.group(1)))
            except json.JSONDecodeError:
                pass
    return out


def jget(obj: Any, *path: int | str) -> Any:
    """Follow a path into Play's JSON, returning None as soon as anything is missing.

    Google's encoding is sparse: when the high-numbered fields of a list are mostly empty, they are
    moved into a dict at the end of the list, keyed by str(index + 1). So position 137 can be either
    obj[137] or obj[-1]["138"]. This getter handles both, so field paths work on every app.
    """
    for key in path:
        if isinstance(obj, dict):
            obj = obj.get(str(key))
        elif isinstance(obj, list) and isinstance(key, int):
            tail = obj[-1] if obj and isinstance(obj[-1], dict) else None
            if tail is not None and key >= len(obj) - 1:
                obj = tail.get(str(key + 1))
            elif 0 <= key < len(obj):
                obj = obj[key]
            else:
                return None
        else:
            return None
        if obj is None:
            return None
    return obj


def find_blob(html: str, pred: Callable[[Any], Any]) -> Any | None:
    """The first JSON blob that has what we need. Found by content, never by its 'ds:N' key."""
    return next((b for b in blobs(html) if pred(b)), None)


def is_soft_404(html: str) -> bool:
    """A removed app's Data Safety page answers HTTP 200 with a 'Not Found' body."""
    return "requested URL was not found on this server" in html or (
        "<title>Not Found</title>" in html and "AF_initDataCallback" not in html)


def parse_details(html: str) -> dict:
    """App metadata from a details page. Missing fields come back as None, never as errors."""
    d = find_blob(html, lambda b: jget(b, 1, 2, 79) is not None and isinstance(jget(b, 1, 2, 0, 0), str))
    out = {
        "title": jget(d, 1, 2, 0, 0),
        "developer": jget(d, 1, 2, 68, 0),
        "genre": jget(d, 1, 2, 79, 0, 0, 2),
        "installs": jget(d, 1, 2, 13, 1),
        "version": jget(d, 1, 2, 140, 0, 0, 0),
        "policy_url": jget(d, 1, 2, 99, 0, 5, 2),
        "updated": None,
    }
    epoch = jget(d, 1, 2, 145, 0, 1, 0)
    if isinstance(epoch, int):
        out["updated"] = datetime.fromtimestamp(epoch, timezone.utc).date().isoformat()
    # Fallbacks from the schema.org block and the visible link, in case the JSON layout moves.
    ld = re.search(r'<script type="application/ld\+json"[^>]*>(.*?)</script>', html, re.S)
    if ld:
        try:
            meta = json.loads(ld.group(1))
            out["title"] = out["title"] or meta.get("name")
            out["genre"] = out["genre"] or meta.get("applicationCategory")
        except json.JSONDecodeError:
            pass
    if not out["policy_url"]:
        m = re.search(r'aria-label="Privacy Policy (\S+) will open', html)
        out["policy_url"] = m.group(1) if m else None
    return out
