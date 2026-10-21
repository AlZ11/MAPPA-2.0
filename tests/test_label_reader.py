"""Label reader on real (trimmed) Data Safety pages from 21 Oct 2026."""

from conftest import fixture_html

from mappa.label_reader import canonical_sha, parse_label
from mappa.vocab import LABEL_PRACTICES, label_key


def flows(claims, polarity="affirm"):
    return {(c.action, c.category, c.subtype) for c in claims if c.kind == "flow" and c.polarity == polarity}


def test_calm_label():
    status, claims, notes = parse_label(fixture_html("label_calm.html.gz"))
    assert status == "ok" and notes == []
    rows = flows(claims)
    assert ("share", "Personal info", "Email address") in rows
    assert ("collect", "Health and fitness", "Fitness info") in rows
    assert ("collect", "Location", "Approximate location") in rows
    assert {c for (a, c, _) in rows if a == "share"} == {"Financial info", "Device or other IDs", "Personal info", "App activity"}
    assert flows(claims, "deny") == set()
    payment = next(c for c in claims if c.subtype == "User payment info")
    assert payment.optional is True and payment.purposes == ("App functionality",)
    purchase = next(c for c in claims if c.subtype == "Purchase history" and c.action == "share")
    assert "Fraud prevention, security, and compliance" in purchase.purposes  # one purpose, despite its commas
    assert purchase.recipient == "third_party" and purchase.quote == "Data shared › Financial info › Purchase history"
    practices = {(c.practice, c.polarity) for c in claims if c.kind == "practice"}
    assert practices == {("encrypted_in_transit", "affirm"), ("deletion", "affirm")}


def test_no_data_label_gives_exactly_two_denials():
    status, claims, _ = parse_label(fixture_html("label_no_data.html.gz"))
    assert status == "ok"
    assert [(c.action, c.category, c.polarity, c.recipient) for c in claims] == [
        ("share", "*", "deny", "third_party"), ("collect", "*", "deny", None)]


def test_unencrypted_practice_is_a_denial():
    status, claims, _ = parse_label(fixture_html("label_unencrypted.html.gz"))
    assert status == "ok"
    assert ("encrypted_in_transit", "deny") in {(c.practice, c.polarity) for c in claims if c.kind == "practice"}


def test_pages_without_a_label_give_no_rows():
    assert parse_label(fixture_html("label_system_app.html.gz")) == ("no_sections", [], [])
    assert parse_label(fixture_html("label_removed_app.html.gz")) == ("not_found", [], [])
    assert parse_label("<html><body>Something went wrong</body></html>")[:2] == ("error", [])


def test_broken_or_mismatched_pages_never_become_denials():
    html = fixture_html("label_calm.html.gz")
    # a section title the parser does not know
    status, claims, notes = parse_label(html.replace('"Data shared"', '"Data maybe shared"', 1))
    assert (status, claims) == ("error", []) and "unexpected section" in notes[0]
    # the visible page disagrees with the JSON (one category region missing)
    status, claims, notes = parse_label(html.replace('id="sLabel0"', 'id="gone"', 1))
    assert (status, claims) == ("error", []) and "page shows" in notes[0]


def test_canonical_sha_ignores_category_order():
    _, claims, _ = parse_label(fixture_html("label_calm.html.gz"))
    assert canonical_sha(claims) == canonical_sha(list(reversed(claims)))
    assert canonical_sha(claims) != canonical_sha(claims[1:])


def test_curly_apostrophes_are_normalised():
    assert LABEL_PRACTICES[label_key("Data can’t be deleted")] == ("deletion", "deny")


def _page(node_json: str, visible: str) -> str:
    blob = ("<script>AF_initDataCallback({key: 'ds:3', hash: '1', data:[null,[null,null,[null,{\"138\":"
            + node_json + "}]]], sideChannel: {}});</script>")
    return f"<html><body>{blob}<div>{visible}</div></body></html>"


def test_label_layout_shifts_are_errors_not_exemptions():
    # The JSON no longer has the sections where we look, but the page visibly shows them.
    status, claims, notes = parse_label(_page("[null,null,null,null,null]", "<h2>Data shared</h2>"))
    assert (status, claims) == ("error", []) and "label sections" in notes[0]
    # A genuine exempt app shows nothing of the sort.
    assert parse_label(_page("[null,null,null,null,null]", "Google Play services"))[0] == "no_sections"
