"""Turning policy web pages into clean text, and finding things in that text."""

import re

from bs4 import BeautifulSoup, NavigableString

# Page furniture that is never part of the policy itself.
DROP_TAGS = ["script", "style", "noscript", "template", "svg", "nav", "header", "footer", "form",
             "button", "iframe", "select", "input", "aside"]
COOKIE_BANNER = re.compile(r"cookie|consent|onetrust|cookiebot|gdpr-banner", re.I)
# Tags that start a new line. Everything else (links, bold, spans) stays inside its sentence.
BLOCK_TAGS = ["p", "div", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "table", "section",
              "article", "main", "blockquote", "pre", "dd", "dt", "dl", "address", "figcaption", "td", "th"]


def html_to_text(html: str | bytes) -> str:
    """Visible policy text, one line per block (paragraph, heading, list item, table row).

    The line structure matters: it is what the policy reader later splits into segments.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(DROP_TAGS):
        tag.decompose()
    body = soup.body or soup
    total = len(body.get_text(" ", strip=True))
    for tag in soup.find_all(attrs={"id": COOKIE_BANNER}) + soup.find_all(attrs={"class": COOKIE_BANNER}):
        # A cookie banner is a small part of the page; never drop something holding most of the text.
        if not tag.decomposed and len(tag.get_text(" ", strip=True)) < 0.5 * total:
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


DATE = r"(\d{1,2}(?:st|nd|rd|th)?\s+\w+,?\s+\d{4}|\w+\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})"
POLICY_DATE = re.compile(r"(?:last\s+(?:updated|modified|revised)|effective(?:\s+date)?|updated(?:\s+on)?)\s*(?:on|as of|date)?\s*[:\-]?\s*"
                         + DATE, re.I)


def policy_date(text: str) -> str | None:
    """The 'Last updated' / 'Effective' date as written in the policy, if it states one."""
    m = POLICY_DATE.search(text[:5000]) or POLICY_DATE.search(text)
    return m.group(1) if m else None
