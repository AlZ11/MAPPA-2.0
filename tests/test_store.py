import hashlib

from mappa.vocab import Claim


def test_blob_is_content_addressed_and_written_once(store):
    data = b"<html>policy v1</html>"
    sha = store.put_blob(data)
    assert sha == hashlib.sha256(data).hexdigest()
    path = store._blob_path(sha)
    mtime = path.stat().st_mtime_ns
    assert store.put_blob(data) == sha
    assert path.stat().st_mtime_ns == mtime  # not rewritten
    assert store.get_blob(sha) == data


def test_pending_makes_the_crawl_resumable(store):
    args = dict(url="u", final_url="u", content_type="text/html")
    assert store.pending("r", "a", "details")
    store.record_fetch("r", "a", "details", status=200, body=b"x", **args)
    assert not store.pending("r", "a", "details")
    store.record_fetch("r", "b", "details", status=404, body=b"", **args)
    assert not store.pending("r", "b", "details")  # gone is a final answer
    for attempt in range(3):
        assert store.pending("r", "c", "details")
        store.record_fetch("r", "c", "details", status=503, body=b"", **args)
    assert not store.pending("r", "c", "details")  # gave up after 3 attempts
    assert store.fetch("r", "c", "details")["attempts"] == 3


def test_save_read_replaces_same_key(store):
    claims = [Claim(kind="flow", category="Location", action="collect", purposes=("Analytics",), quote="q")]
    stats = {"seconds": 1.0, "dropped": {"not_found": 2}}
    store.save_read("r", "a", "policy", "baseline", claims, input_sha="s", stats=stats)
    store.save_read("r", "a", "policy", "baseline", claims * 2, input_sha="s", stats=stats)
    rows = store.claims("r", "a")
    assert len(rows) == 2 and rows[0]["category"] == "Location" and rows[0]["purposes"] == '["Analytics"]'
    store.save_read("r", "a", "policy", "baseline", claims, input_sha="s", stats=stats, rep=1)
    assert len(store.claims("r", "a")) == 3  # a different repetition is a different read
