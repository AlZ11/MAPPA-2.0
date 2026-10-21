import json

from mappa.datasets import _app350_claim, _load_unibo, _opp115_claims, parse_codes
from mappa.vocab import Claim


def test_app350_labels_translate_to_play_terms():
    assert _app350_claim("Location_GPS_3rdParty", "PERFORMED") == Claim(
        kind="flow", category="Location", subtype="Precise location", action="share", polarity="affirm")
    assert _app350_claim("Contact_E_Mail_Address_1stParty", "NOT_PERFORMED") == Claim(
        kind="flow", category="Personal info", subtype="Email address", action="collect", polarity="deny")
    assert _app350_claim("Identifier_IMEI_1stParty", "PERFORMED").category == "Device or other IDs"
    assert _app350_claim("Contact_Password_1stParty", "PERFORMED") is None  # no Play equivalent
    assert _app350_claim("SSO", "PERFORMED") is None


def test_opp115_rows_translate_to_flows_and_practices():
    attrs = {"Personal Information Type": {"value": "Health"}, "Purpose": {"value": "Advertising"},
             "Does/Does Not": {"value": "Does Not"}}
    assert _opp115_claims("Third Party Sharing/Collection", attrs) == [Claim(
        kind="flow", category="Health and fitness", action="share", polarity="deny", purposes=("Advertising or marketing",))]
    assert _opp115_claims("First Party Collection/Use", {"Personal Information Type": {"value": "Unspecified"}}) == []
    assert _opp115_claims("Data Security", {"Security Measure": {"value": "Secure data transfer"}})[0].practice == "encrypted_in_transit"
    assert _opp115_claims("User Access, Edit and Deletion", {"Access Type": {"value": "Delete account (full)"}})[0].practice == "deletion"
    assert _opp115_claims("International and Specific Audiences", {"Audience Type": {"value": "Californians"}}) == []


def test_unibo_offsets_survive_windows_line_ends_and_duplicates(tmp_path):
    plain = "Intro line.\r\nWe collect your weight and height.\r\nNothing else."
    start = plain.index("weight")
    span = {"start": start, "end": start + 6, "type": "SubCategory", "attributes": {"Kind": "HealthFitness"}}
    doc = {"document": {"plainText": plain}, "annotations": [span, dict(span)]}
    for name in __import__("mappa.datasets", fromlist=["UNIBO_NAMES"]).UNIBO_NAMES:
        (tmp_path / f"{name}.json").write_text(json.dumps(doc))
    d = _load_unibo(tmp_path)[0]
    assert "\r" not in d.text
    hit = [i for i, g in enumerate(d.gold) if g]
    assert len(hit) == 1 and "weight" in d.text[d.spans[hit[0]][0]:d.spans[hit[0]][1]]
    assert d.gold[hit[0]] == {Claim(kind="flow", category="Health and fitness")}


def test_parse_codes():
    got = parse_codes("health and fitness:collect; Location:share:not:sp; deletion; automated_decisions:not")
    assert got == {
        Claim(kind="flow", category="Health and fitness", action="collect"),
        Claim(kind="flow", category="Location", action="share", polarity="deny"),
        Claim(kind="practice", practice="deletion"),
        Claim(kind="practice", practice="automated_decisions", polarity="deny"),
    }
