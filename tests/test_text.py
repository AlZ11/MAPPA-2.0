from mappa.text import html_to_text, policy_date

PAGE = """<html><head><script>var x = "tracking code";</script></head><body>
<nav>Home | Products | Login</nav>
<div id="onetrust-consent-sdk">We use cookies. Accept all?</div>
<main>
  <h1>Privacy Policy</h1><p>Last updated: 12 March 2026</p>
  <p>We collect your <b>email address</b> and <a href="#">location</a> when you sign up.</p>
  <ul><li>Name</li><li>Date&nbsp;of   birth</li></ul>
  <p>This sentence is repeated by the responsive layout of the website and it is long enough to count as a duplicated block.</p>
  <p>This sentence is repeated by the responsive layout of the website and it is long enough to count as a duplicated block.</p>
</main>
<footer>Copyright 2026</footer>
<script id="__NEXT_DATA__" type="application/json">{"props": "We collect your email address"}</script>
</body></html>"""


def test_html_to_text_keeps_policy_drops_furniture():
    text = html_to_text(PAGE)
    lines = text.split("\n")
    assert lines[0] == "Privacy Policy"
    assert "We collect your email address and location when you sign up." in lines  # inline tags stay inline
    assert "Name" in lines and "Date of birth" in lines  # list items are lines; NBSP and runs collapsed
    assert sum("responsive layout" in ln for ln in lines) == 1  # long repeated block kept once
    for junk in ("tracking code", "Login", "cookies", "Copyright", "props"):
        assert junk not in text


def test_policy_date():
    assert policy_date("Privacy Policy\nLast updated: 12 March 2026\n...") == "12 March 2026"
    assert policy_date("Effective date: October 13, 2026") == "October 13, 2026"
    assert policy_date("No date here at all.") is None


from mappa.text import find_quote, segment, sentences  # noqa: E402

POLICY = ("Privacy Policy\nInformation we collect\n"
          "We collect your email address. We do not sell it, e.g. to Acme Inc. or anyone else.\n"
          "We may collect:\nName\nEmail\nLocation (GPS)\nThat is everything we collect from you here.")


def test_segment_joins_headings_and_lead_ins_and_covers_the_text():
    spans = segment(POLICY)
    texts = [POLICY[s:e] for s, e in spans]
    assert texts[0].startswith("Privacy Policy\nInformation we collect\nWe collect your email")
    assert texts[1].startswith("We may collect:\nName") and texts[1].endswith("from you here.")
    assert all(POLICY[e:s2].strip() == "" for (_, e), (s2, _) in zip(spans, spans[1:]))  # contiguous


def test_segment_splits_long_blocks_at_sentences():
    long = " ".join(f"Sentence number {i} is here." for i in range(100))  # 500 words, one line
    spans = segment(long, max_words=200)
    assert len(spans) == 3 and all(len(long[s:e].split()) <= 200 for s, e in spans)
    assert all(long[s:e].endswith(".") for s, e in spans)


def test_sentences_respect_abbreviations():
    got = [POLICY[s:e] for s, e in sentences(POLICY)]
    assert "We do not sell it, e.g. to Acme Inc. or anyone else." in got


def test_find_quote_normalises_but_never_fuzzes():
    text = "We do not sell your “personal information” to anyone, ever.\nName"
    span, why = find_quote('WE DO NOT SELL YOUR "PERSONAL INFORMATION" TO ANYONE.', text)
    assert why == "ok" and text[span[0]:span[1]].startswith("We do") and text[span[0]:span[1]].endswith("anyone")
    assert find_quote("We do not sell your data to anyone", text) == (None, "not_found")
    assert find_quote("We do not sell ... to anyone", text) == (None, "elided")
    assert find_quote("sell your", text) == (None, "too_short")
    assert find_quote("Name", text)[1] == "ok"  # short, but a whole line (a list item)


def test_broken_html_never_swallows_the_policy():
    policy = "<p>" + "We collect your email address to run your account. " * 30 + "</p>"
    unclosed_header = f"<html><body><header><div>Menu</header><main>{policy}</main></body></html>"
    form_wrapper = f"<html><body><form id='aspnetForm'><main>{policy}</main></form></body></html>"
    for page in (unclosed_header, form_wrapper):
        assert len(html_to_text(page).split()) == 270  # every word of the policy survives
    cookie_section = f"<main><h2 id='cookies'>Cookies</h2><section id='cookies-we-use'>We use Meta Pixel.</section>{policy}</main>"
    assert "Meta Pixel" in html_to_text(cookie_section) and "Cookies" in html_to_text(cookie_section)


def test_dates_sentences_and_repeated_short_quotes():
    assert policy_date("Last Updated: September 2026") == "September 2026"
    assert policy_date("Last updated: 21.05.2026") == "21.05.2026"
    text = "You can delete your account on request. We never share your health data."
    assert len(sentences(text)) == 2  # "request." is not the abbreviation "st."
    listed = "We collect your email address and name.\nWe collect:\nEmail address\nName"
    span, why = find_quote("Email address", listed)
    assert why == "ok" and listed[span[0]:span[1]] == "Email address" and span[0] > 40  # the list item, not the sentence
