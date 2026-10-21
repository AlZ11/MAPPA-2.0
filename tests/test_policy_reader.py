from mappa.datasets import GoldDoc
from mappa.policy_reader import Baseline, segment_of
from mappa.vocab import Claim

GPS = "We collect your precise GPS location to show nearby clinics."
NOT_GPS = "We do not collect your GPS location."
OTHER = "This policy explains our terms of use and governing law."


def toy(n=12):
    docs = []
    for i in range(n):
        texts = [GPS, OTHER, NOT_GPS] if i % 2 else [OTHER, GPS]
        gold = [{Claim(kind="flow", category="Location", action="collect")} if t == GPS else
                {Claim(kind="flow", category="Location", action="collect", polarity="deny")} if t == NOT_GPS else set()
                for t in texts]
        text = "\n\n".join(texts)
        spans, pos = [], 0
        for t in texts:
            spans.append((pos, pos + len(t)))
            pos += len(t) + 2
        docs.append(GoldDoc("app350", str(i), "train", text, spans, gold))
    return docs


def test_baseline_learns_and_quotes_from_the_text():
    model = Baseline.train({"app350": toy(), "opp115": toy()})
    assert "flow|Location|collect" in model.classifiers
    text = f"{OTHER}\n\n{GPS}"
    claims, stats = model.extract(text, [(0, len(OTHER)), (len(OTHER) + 2, len(text))])
    loc = [c for c in claims if c.category == "Location"]
    assert loc and loc[0].seg == 1 and loc[0].quote == GPS and text[loc[0].start:loc[0].end] == GPS
    assert stats["dropped"] == {}


def test_segment_of_uses_the_quote_position():
    claims = segment_of([Claim(kind="flow", start=15)], [(0, 10), (12, 30)])
    assert claims[0].seg == 1
