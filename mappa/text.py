"""Turning policy web pages into clean text, and finding things in that text."""

import re
import unicodedata
from functools import lru_cache

from bs4 import BeautifulSoup, NavigableString

# Code and widgets: never text.
ALWAYS_DROP = ["script", "style", "noscript", "template", "svg", "button", "iframe", "select", "input"]
# Page furniture: dropped unless it holds most of the page. Some sites wrap the whole policy in a <form>,
# and broken HTML can leave the policy inside an unclosed <header> or <nav>.
FURNITURE = ["nav", "header", "footer", "form", "aside"]
# Cookie-consent widgets (OneTrust, Cookiebot, CookieYes...), matched narrowly: a policy section with
# id="cookies" or "disclosure-with-your-consent" is content, not a banner.
COOKIE_BANNER = re.compile(r"onetrust|cookiebot|\bcky-|cookie-?(?:banner|notice|bar|consent|popup|modal)"
                           r"|consent-?(?:banner|manager|popup|modal)|gdpr-?(?:banner|popup)", re.I)
HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
# Tags that start a new line. Everything else (links, bold, spans) stays inside its sentence.
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section",
              "article", "main", "blockquote", "pre", "dd", "dt", "dl", "address", "figcaption", "td", "th"]


def html_to_text(html: str | bytes) -> str:
    """Visible policy text, one line per block (paragraph, heading, list item, table row).

    The line structure matters: it is what the policy reader later splits into segments.
    Python's own HTML parser is used on purpose: lxml "repairs" broken pages by moving everything after
    an unclosed tag inside it, which can put the whole policy inside a <header> that then gets dropped.
    """
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(ALWAYS_DROP):
        tag.decompose()
    body = soup.body or soup
    total = len(body.get_text(" ", strip=True))

    def minor(tag) -> bool:  # never drop something that holds most of the page's text
        return not tag.decomposed and len(tag.get_text(" ", strip=True)) < 0.5 * total

    for tag in soup(FURNITURE):
        if minor(tag):
            tag.decompose()
    for tag in soup.find_all(attrs={"id": COOKIE_BANNER}) + soup.find_all(attrs={"class": COOKIE_BANNER}):
        if tag.name not in HEADINGS and minor(tag):
            tag.decompose()
    # If <main> or <article> holds most of the text, the rest is navigation and adverts.
    total = len(body.get_text(" ", strip=True))
    for name in ("main", "article"):
        part = body.find(name)
        if part is not None and total and len(part.get_text(" ", strip=True)) >= 0.5 * total:
            body = part
            break
    for br in body.find_all("br"):
        br.replace_with(NavigableString("\n"))
    for tag in body.find_all(BLOCK_TAGS):
        tag.insert_before(NavigableString("\n"))
        tag.insert_after(NavigableString("\n"))
    lines, seen = [], set()
    for line in body.get_text().split("\n"):
        line = " ".join(line.split())
        if not line:
            continue
        # Responsive sites render the same text twice (desktop and mobile layouts). Drop long repeats.
        if len(line.split()) >= 20:
            if line in seen:
                continue
            seen.add(line)
        lines.append(line)
    return "\n".join(lines)


MONTH = r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?"
DATE = (r"(\d{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+,?\s+\d{4}|[A-Za-z]+\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
        r"|\d{4}-\d{2}-\d{2}|\d{1,2}[/.]\d{1,2}[/.]\d{2,4}|" + MONTH + r"\s+\d{4})")
POLICY_DATE = re.compile(r"(?:last\s+(?:updated|modified|revised)|effective(?:\s+date)?|updated(?:\s+on)?)"
                         r"\s*(?:on|as of|date)?\s*[:\-]?\s*" + DATE, re.I)


def policy_date(text: str) -> str | None:
    """The 'Last updated' / 'Effective' date as written in the policy, if it states one."""
    m = POLICY_DATE.search(text[:5000]) or POLICY_DATE.search(text)
    return m.group(1) if m else None


# ---------------------------------------------------------------- segments and sentences
def segment(text: str, max_words: int = 200) -> list[tuple[int, int]]:
    """Split text into contiguous (start, end) spans of whole lines, each at most ~max_words.

    A short line without a full stop (a heading, or "We collect:") is joined to what follows,
    so a heading travels with its paragraph and a lead-in with its list. A block that is too long
    is split at sentence ends.
    """
    spans, current, words = [], [], 0
    for m in re.finditer(r"[^\n]+", text):
        line = m.group()
        n = len(line.split())
        if current and words + n > max_words and not all(_joins_next(text[s:e]) for s, e in current):
            spans.append((current[0][0], current[-1][1]))
            current, words = [], 0
        current.append(m.span())
        words += n
        if not _joins_next(line):
            spans.append((current[0][0], current[-1][1]))
            current, words = [], 0
    if current:
        spans.append((current[0][0], current[-1][1]))
    out = []
    for s, e in spans:
        if len(text[s:e].split()) <= max_words:
            out.append((s, e))
            continue
        chunk_start, chunk_words = s, 0
        for ss, se in sentences(text, s, e):
            n = len(text[ss:se].split())
            if chunk_words and chunk_words + n > max_words:
                out.append((chunk_start, _rstrip_end(text, chunk_start, ss)))
                chunk_start, chunk_words = ss, 0
            chunk_words += n
        out.append((chunk_start, e))
    return out


def _joins_next(line: str) -> bool:
    line = line.strip()
    return len(line.split()) <= 12 and not line.endswith((".", "!", "?", '."', ".)"))


def _rstrip_end(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1].isspace():
        end -= 1
    return end


SENTENCE_END = re.compile(r"(?<=[.!?])[\"'”’)\]]*\s+(?=[A-Z0-9\"“‘(\[])|\n+")
ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "inc.", "ltd.", "co.", "no.", "u.s.", "pty.", "vs.", "mr.", "ms.", "dr.", "st.")
# An abbreviation only counts as a whole word: "request." ends a sentence even though it ends in "st.".
ABBREVIATION_END = re.compile(r"(?:^|[\s(\"'/-])(?:" + "|".join(map(re.escape, ABBREVIATIONS)) + r")$")


def sentences(text: str, start: int = 0, end: int | None = None) -> list[tuple[int, int]]:
    """Sentence spans inside text[start:end]. Simple rules: a sentence ends at . ! ? or a line break."""
    end = len(text) if end is None else end
    spans, s = [], start
    for m in SENTENCE_END.finditer(text, start, end):
        piece = text[s:m.start()].strip()
        if piece and not ABBREVIATION_END.search(piece.lower()):
            spans.append(_trim(text, s, m.start()))
            s = m.end()
    if text[s:end].strip():
        spans.append(_trim(text, s, end))
    return spans


def _trim(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    return s, _rstrip_end(text, s, e)


# ---------------------------------------------------------------- the copied-sentence check
# Characters that differ between how a model (or a person) copies text and how a web page stores it.
_SAME_AS = {
    **dict.fromkeys("‘’‚‛′", "'"),
    **dict.fromkeys("“”„″", '"'),
    **dict.fromkeys("‐‑‒–—−", "-"),
    **dict.fromkeys("​‌‍﻿­", ""),
}


@lru_cache(maxsize=16)
def normalise(s: str) -> tuple[str, tuple[int, ...]]:
    """A comparison form of s, plus where each of its characters came from in s.

    Unicode forms unified, curly quotes and dashes made plain, invisible characters removed,
    every run of whitespace made one space, case removed. Nothing else: no fuzzy matching.
    """
    out, where, space = [], [], True
    for i, ch in enumerate(s):
        for c in unicodedata.normalize("NFKC", ch):
            c = _SAME_AS.get(c, c)
            if not c:
                continue
            if c.isspace():
                if space:
                    continue
                c, space = " ", True
            else:
                space = False
            for cc in c.casefold():
                out.append(cc)
                where.append(i)
    if out and out[-1] == " ":
        out.pop()
        where.pop()
    return "".join(out), tuple(where)


def find_quote(quote: str, text: str, min_words: int = 4) -> tuple[tuple[int, int] | None, str]:
    """Where a claimed quote sits in the text: ((start, end), "ok"), or (None, why not).

    Exact match after normalise(), ignoring only surrounding quote marks and trailing punctuation
    (a model quoting the first clause of a longer sentence drops the comma that followed it).
    Reasons: not_found | elided ("..." in the quote) | too_short (under min_words, unless it is a whole line).
    """
    q = normalise(quote)[0].strip(" '\"").rstrip(".,;:!? ")
    if not q:
        return None, "not_found"
    t, where = normalise(text)
    pos = t.find(q)
    if pos < 0:
        return None, "elided" if "..." in q else "not_found"
    while pos >= 0:  # a short quote is fine where it is a whole line (a list item), even if it also occurs earlier
        start, end = where[pos], where[pos + len(q) - 1] + 1
        if len(q.split()) >= min_words or _is_whole_line(text, start, end, q):
            return (start, end), "ok"
        pos = t.find(q, pos + 1)
    return None, "too_short"


def _is_whole_line(text: str, start: int, end: int, q: str) -> bool:
    line_start = text.rfind("\n", 0, start) + 1
    line_end = text.find("\n", end)
    line = text[line_start:len(text) if line_end < 0 else line_end]
    return normalise(line)[0].rstrip(".,;:!? ") == q
