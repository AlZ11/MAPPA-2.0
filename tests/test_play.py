from conftest import fixture_html

from mappa import play


def test_jget_follows_sparse_trailing_dict():
    dense = [0, 1, 2, ["x", ["y"]]]
    assert play.jget(dense, 3, 1, 0) == "y"
    sparse = ["title", {"5": ["moved"], "139": "far"}]  # fields 4 and 138 moved into the tail
    assert play.jget(sparse, 0) == "title"
    assert play.jget(sparse, 4, 0) == "moved"
    assert play.jget(sparse, 138) == "far"
    assert play.jget(sparse, 2) is None
    assert play.jget(None, 1) is None
    assert play.jget(dense, 9, 9) is None


def test_app_ids_in_page_order_without_repeats():
    html = ('<a href="/store/apps/details?id=com.b">b</a><a href="/store/apps/details?id=com.a.x">a</a>'
            '<a href="/store/apps/details?id=com.b&hl=en">b again</a>')
    assert play.app_ids(html) == ["com.b", "com.a.x"]


def test_parse_details_on_real_calm_page():
    meta = play.parse_details(fixture_html("details_calm.html.gz"))
    assert meta["title"] == "Calm - Sleep, Meditate, Relax"
    assert meta["genre"] == "HEALTH_AND_FITNESS"
    assert meta["developer"] == "Calm.com, Inc."
    assert meta["policy_url"] == "https://www.calm.com/privacy"
    assert meta["installs"] >= 50_000_000
    assert meta["updated"] and meta["updated"][:2] == "20"


def test_parse_details_on_empty_page_returns_nones():
    assert set(play.parse_details("<html></html>").values()) == {None}
