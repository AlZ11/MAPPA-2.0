import pytest

from mappa.datasets import GoldDoc
from mappa.evaluate import bootstrap, cohen_kappa, counts, prf, score, stability, summarise
from mappa.vocab import Claim

LOC = Claim(kind="flow", category="Location", action="collect")
EMAIL = Claim(kind="flow", category="Personal info", subtype="Email address", action="share")


def test_prf_and_counts():
    assert prf(2, 1, 1) == pytest.approx((2 / 3, 2 / 3, 2 / 3))
    assert prf(0, 0, 3) == (None, 0.0, 0.0)  # never predicted: F1 is 0, not missing
    assert prf(0, 0, 0) == (None, None, None)
    per_label, per_unit = counts([({"a", "b"}, {"a", "c"}), ({"b"}, set())])
    assert per_label == {"a": [1, 0, 0], "b": [0, 0, 2], "c": [0, 1, 0]}
    assert per_unit == [(1, 1, 1), (0, 0, 1)]


def test_summarise_reference_and_support_cutoff():
    s = summarise({"x": [5, 0, 5], "y": [1, 0, 0]}, n_units=10)
    x, y = s["labels"]
    assert x["f1"] == pytest.approx(2 / 3) and x["reference_f1"] == pytest.approx(2 * 1.0 / 2.0)
    assert y["f1"] is None  # support 1 < 5: counts only
    assert s["macro_f1"] == pytest.approx(2 / 3)


def test_score_respects_scope_and_levels():
    doc = GoldDoc("app350", "1", "test", "a\n\nb", [(0, 1), (3, 4)], [{LOC}, set()])
    out_of_scope = Claim(kind="flow", category="Health and fitness", action="collect", seg=0)
    pred = [Claim(kind="flow", category="Location", action="collect", polarity="affirm", seg=1), out_of_scope]
    s = score([doc], [pred], "app350")["flow"]
    assert s["L1"]["micro"]["f1"] == 1.0  # right document, health ignored (APP-350 does not label it)
    assert s["L2"]["micro"]["f1"] == 0.0  # wrong segment


def test_kappa_textbook_and_degenerate():
    a = [True] * 20 + [True] * 5 + [False] * 10 + [False] * 15
    b = [True] * 20 + [False] * 5 + [True] * 10 + [False] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.4)
    assert cohen_kappa([True, True], [True, True]) is None  # chance agreement is 1: undefined


def test_stability_and_bootstrap():
    assert stability([[{"a"}, {"b"}], [{"a"}, {"b", "c"}]]) == pytest.approx((1 + 0.5) / 2)
    assert stability([[{"a"}]]) is None
    per_doc = [(3, 1, 0), (2, 0, 2), (5, 1, 1)]
    lo, hi = bootstrap(per_doc)
    assert 0 <= lo <= hi <= 1 and bootstrap(per_doc) == (lo, hi)  # deterministic


def test_opp115_cannot_judge_practice_polarity():
    doc = GoldDoc("opp115", "1", "test", "a", [(0, 1)], [{Claim(kind="practice", practice="retention")}])
    denied = Claim(kind="practice", practice="retention", polarity="deny", seg=0)
    assert score([doc], [[denied]], "opp115")["practice"]["L1"]["micro"]["f1"] == 1.0


def test_demo_marks_found_missed_false_alarm_and_not_judged():
    from mappa.evaluate import compare
    share_email = Claim(kind="flow", category="Personal info", subtype="Email address", action="share")
    doc = GoldDoc("app350", "1", "test", "a\n\nb", [(0, 1), (3, 4)], [{LOC, share_email}, set()])
    claims = [Claim(kind="flow", category="Location", action="collect", quote="q1", seg=0),          # right
              Claim(kind="flow", category="Contacts", action="collect", quote="q2", seg=0),          # not labelled
              Claim(kind="flow", category="Health and fitness", action="collect", quote="q3", seg=1)]  # APP-350 can't judge
    rows = compare(doc, claims, "app350")
    marks = {(r["seg"], mark, label[1]) for r in rows for mark, label, _, _ in r["items"]}
    assert marks == {(0, "found", "Location"), (0, "false_alarm", "Contacts"), (0, "missed", "Personal info"),
                     (1, "not_judged", "Health and fitness")}
