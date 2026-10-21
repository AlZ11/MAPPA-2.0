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
