"""Gold-standard corpora for training the baseline and measuring both extractors.

Each corpus is loaded into the same shape (GoldDoc): the text, its segments, and per segment the
set of true claims, translated into MAPPA's vocabulary. Each corpus only labels some things, so
each also has a SCOPE: predictions outside it count as neither right nor wrong.

  app350   350 Android app policies, sentence labels for contact data, contacts, location and device
           IDs, with affirm/deny. Official splits. https://usableprivacy.org/data (research-only licence)
  opp115   115 website policies, paragraph labels for data types (incl. health, financial) and practices
           (retention, security, deletion, choice, children...). Split here by hashing policy ids.
  unibo    30 policies (4 health) with spans marking which data categories are mentioned. No action or
           polarity labels, so it only measures "is category X mentioned". Downloaded at runtime (no licence).
  indomain hand-labelled Australian health app policies (see evaluate.annotate / evaluate.kappa).

Nothing here is ever written into the repository: everything lives under <data>/external.
"""

import csv
import hashlib
import json
import re
import zipfile
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path

import requests
import yaml

from .text import segment
from .vocab import CATEGORIES, GENERIC, PRACTICES, Claim

SOURCES = {
    "app350": ("https://usableprivacy.org/static/data/APP-350_v1.1.zip", "APP-350_v1.1"),
    "opp115": ("https://usableprivacy.org/static/data/OPP-115_v1_0.zip", "OPP-115"),
}
UNIBO_URL = "https://raw.githubusercontent.com/nlp-unibo/Privacy-Policies-Compliance/main/cat/data/{}.json"
UNIBO_NAMES = (
    "AdobePhotoshop", "Airbnb", "Amazon", "Apple", "Ava", "Blizzard", "Deliveroo", "Dropbox", "Ebay", "Fitbit",
    "Groupon", "JustEat", "Kardia", "LinkedIn", "MyHeritage", "MySugr", "Oculus", "Paypal", "Rovio", "Snapchat",
    "TikTok", "Tinder", "Twitch", "Uber", "Ubisoft", "Western Union", "X_Twitter", "Yahoo", "Yelp", "Zoom",
)
UNIBO_HEALTH = ("Ava", "Fitbit", "Kardia", "MySugr")


@dataclass
class GoldDoc:
    corpus: str
    doc_id: str
    split: str  # train | dev | test
    text: str
    spans: list[tuple[int, int]]  # segments
    gold: list[set[Claim]]  # true claims per segment (labels only: no quotes or offsets)
    quotes: list[str] = field(default_factory=list)  # annotated sentences (APP-350), to test the quote matcher
    name: str = ""  # the app or website the policy belongs to


# ---------------------------------------------------------------- label translations
# APP-350 data types -> (Play category, subtype). None = no clean Play equivalent: left out of scoring.
APP350_TYPES: dict[str, tuple[str, str | None] | None] = {
    "Contact": ("Personal info", None),  # generic contact data, e.g. "your name"
    "Contact_E_Mail_Address": ("Personal info", "Email address"),
    "Contact_Phone_Number": ("Personal info", "Phone number"),
    "Contact_Postal_Address": ("Personal info", "Address"),
    "Contact_City": ("Personal info", "Address"),
    "Contact_ZIP": ("Personal info", "Address"),
    "Contact_Password": None,  # a credential, not a Play data type
    "Demographic": ("Personal info", "Other info"),
    "Demographic_Age": ("Personal info", "Other info"),
    "Demographic_Gender": ("Personal info", "Other info"),
    "Contact_Address_Book": ("Contacts", "Contacts"),
    "Location": ("Location", None),
    "Location_GPS": ("Location", "Precise location"),  # APP-350's guide: fine-grained = GPS, WiFi, Bluetooth
    "Location_WiFi": ("Location", "Precise location"),
    "Location_Bluetooth": ("Location", "Precise location"),
    "Location_Cell_Tower": ("Location", "Approximate location"),  # coarse = cell tower, IP
    "Location_IP_Address": ("Location", "Approximate location"),
    "Identifier_Mobile_Carrier": None,  # network information, not an identifier in Play's sense
    "Identifier_SSID_BSSID": None,  # could be an ID or a location signal: ambiguous
    "SSO": None,  # a sign-in route, not a data type
    "Facebook_SSO": None,
}
for _t in ("Identifier", "Identifier_Ad_ID", "Identifier_Cookie_or_similar_Tech", "Identifier_Device_ID",
           "Identifier_IMEI", "Identifier_IMSI", "Identifier_IP_Address", "Identifier_MAC", "Identifier_SIM_Serial"):
    APP350_TYPES[_t] = ("Device or other IDs", "Device or other IDs")
APP350_PARTY = {"1stParty": "collect", "3rdParty": "share"}
APP350_SPLIT = {"TRAINING": "train", "VALIDATION": "dev", "TEST": "test"}

OPP115_TYPES = {
    "Contact": "Personal info", "Demographic": "Personal info", "User profile": "Personal info",
    "Personal identifier": "Personal info", "Health": "Health and fitness", "Financial": "Financial info",
    "Location": "Location", "IP address and device IDs": "Device or other IDs",
    "Cookies and tracking elements": "Device or other IDs", "User online activities": "App activity",
    "Generic personal information": GENERIC,
}  # Computer information, Social media data, Survey data, Other, Unspecified: no clean Play equivalent
OPP115_PURPOSES = {
    "Basic service/feature": "App functionality", "Additional service/feature": "App functionality",
    "Advertising": "Advertising or marketing", "Marketing": "Advertising or marketing",
    "Analytics/Research": "Analytics", "Personalization/Customization": "Personalization",
    "Service operation and security": "Fraud prevention, security, and compliance",
    "Legal requirement": "Fraud prevention, security, and compliance",
}
UNIBO_KINDS = {
    "GeoInfo": "Location", "BasicAccountInfo": "Personal info", "ContactInfo": "Personal info",
    "UserProfileInfo": "Personal info", "Demographic": "Personal info", "IdentityVerificationInfo": "Personal info",
    "Payment": "Financial info", "Purchase": "Financial info", "Financial": "Financial info",
    "HealthFitness": "Health and fitness", "CommunicationProv": "Messages", "Images": "Photos and videos",
    "AudioTyping": "Audio", "ContactList": "Contacts", "ListFriendsInfo": "Contacts",
    "UsageData": "App activity", "UserGenerated": "App activity", "SocialInteraction": "App activity",
    "ContentPreferences": "App activity", "InternetHistory": "Web browsing", "Performance": "App info and performance",
    "DeviceInfo": "Device or other IDs", "Gen": GENERIC,
}  # Metadata, Settings, Gov, Deidentified, ... : ambiguous or exempt, left out

# What each corpus labels. A scope function returns the claim as the corpus can judge it, or None.
APP350_CATEGORIES = {"Personal info", "Contacts", "Location", "Device or other IDs"}
APP350_SUBTYPES = {sub for v in APP350_TYPES.values() if v for sub in [v[1]] if sub}
OPP115_CATEGORIES = set(OPP115_TYPES.values())
OPP115_PRACTICES = {"retention", "security_measures", "encrypted_in_transit", "deletion", "access_correction",
                    "user_choice", "children", "policy_change", "privacy_contact"}


def _scope_app350(c: Claim) -> Claim | None:
    if c.kind != "flow" or c.category not in APP350_CATEGORIES:
        return None
    return c if c.subtype in APP350_SUBTYPES or c.subtype is None else replace(c, subtype=None)


OPP115_PURPOSE_SET = set(OPP115_PURPOSES.values())


def _scope_opp115(c: Claim) -> Claim | None:
    # OPP-115 never marks a practice as denied, so it cannot judge practice polarity; it has no subtypes, and
    # only five of Play's purposes. Purposes are scored only for data types OPP-115 can map (see UNMAPPED).
    if c.kind == "practice":
        return replace(c, polarity="affirm") if c.practice in OPP115_PRACTICES else None
    if c.category not in OPP115_CATEGORIES:
        return None
    return replace(c, subtype=None, purposes=tuple(p for p in c.purposes if p in OPP115_PURPOSE_SET))


def _scope_unibo(c: Claim) -> Claim | None:
    return c if c.kind == "flow" and c.category in set(UNIBO_KINDS.values()) else None


SCOPE = {"app350": _scope_app350, "opp115": _scope_opp115, "unibo": _scope_unibo, "indomain": lambda c: c}


# ---------------------------------------------------------------- download
def ensure(name: str, root: Path) -> Path:
    """Download and unpack a corpus once, into <root>/external. Prints the archive's SHA-256."""
    ext = Path(root) / "external"
    ext.mkdir(parents=True, exist_ok=True)
    if name == "unibo":
        folder = ext / "unibo"
        folder.mkdir(exist_ok=True)
        for n in UNIBO_NAMES:
            path = folder / f"{n}.json"
            if not path.exists():
                r = requests.get(UNIBO_URL.format(n.replace(" ", "%20")), timeout=60)
                r.raise_for_status()
                path.write_bytes(r.content)
        return folder
    url, folder_name = SOURCES[name]
    folder = ext / folder_name
    if not folder.exists():
        archive = ext / url.rsplit("/", 1)[1]
        if not archive.exists():
            print(f"downloading {url}", flush=True)
            r = requests.get(url, timeout=600)
            r.raise_for_status()
            archive.write_bytes(r.content)
        print(f"{archive.name} sha256 {hashlib.sha256(archive.read_bytes()).hexdigest()}")
        with zipfile.ZipFile(archive) as z:
            z.extractall(ext, members=[m for m in z.namelist() if not m.startswith("__MACOSX")])
    return folder


def load(corpus: str, root: Path, split: str | None = None, path: Path | None = None) -> list[GoldDoc]:
    """All documents of a corpus (optionally one split), sorted by id."""
    if corpus == "indomain":
        docs = load_indomain(path)
    else:
        docs = {"app350": _load_app350, "opp115": _load_opp115, "unibo": _load_unibo}[corpus](ensure(corpus, root))
    return sorted((d for d in docs if split is None or d.split == split), key=lambda d: d.doc_id)


def _join(pieces: list[str], sep: str = "\n\n") -> tuple[str, list[tuple[int, int]]]:
    spans, pos = [], 0
    for p in pieces:
        spans.append((pos, pos + len(p)))
        pos += len(p) + len(sep)
    return sep.join(pieces), spans


# ---------------------------------------------------------------- APP-350
def _app350_claim(practice: str, modality: str) -> Claim | None:
    for party, action in APP350_PARTY.items():
        if practice.endswith("_" + party):
            mapped = APP350_TYPES.get(practice[: -len(party) - 1])
            if mapped is None:
                return None
            polarity = "deny" if modality == "NOT_PERFORMED" else "affirm"
            return Claim(kind="flow", category=mapped[0], subtype=mapped[1], action=action, polarity=polarity)
    return None  # SSO labels carry no party


def _load_app350(folder: Path) -> list[GoldDoc]:
    docs = []
    for f in sorted((folder / "annotations").glob("policy_*.yml")):
        raw = yaml.load(f.read_text(), Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader))
        segments = raw["segments"]
        text, spans = _join([" ".join(s["segment_text"].split()) for s in segments])
        gold, quotes = [], []
        for s in segments:
            labels = [a for sent in s.get("sentences") or [] for a in sent.get("annotations") or []]
            labels += s.get("annotations") or []
            gold.append({c for a in labels if (c := _app350_claim(a["practice"], a["modality"]))})
            quotes += [sent["sentence_text"] for sent in s.get("sentences") or [] if sent.get("annotations")]
        docs.append(GoldDoc("app350", str(raw["policy_id"]), APP350_SPLIT[raw["policy_type"]], text, spans, gold, quotes,
                            name=str(raw.get("policy_name", ""))))
    return docs


# ---------------------------------------------------------------- OPP-115
def opp115_split(policy_id: str) -> str:
    """Fixed split: policies ordered by a hash of their id; first 40 test, next 10 dev, rest train."""
    return _OPP115_ORDER.get(policy_id, "train")


_OPP115_ORDER: dict[str, str] = {}


# OPP-115 spells some values differently in different files ("User profile" / "User Profile"): match without case.
_OPP115_TYPES_CF = {k.casefold(): v for k, v in OPP115_TYPES.items()}
_OPP115_PURPOSES_CF = {k.casefold(): v for k, v in OPP115_PURPOSES.items()}
UNMAPPED: Counter = Counter()  # OPP-115 values left out of scoring, reported when the corpus loads


def _opp115_claims(category: str, attrs: dict) -> list[Claim]:
    value = lambda name: (attrs.get(name) or {}).get("value")
    if category in ("First Party Collection/Use", "Third Party Sharing/Collection"):
        data_type = value("Personal Information Type") or ""
        cat = _OPP115_TYPES_CF.get(data_type.casefold())
        if cat is None:
            UNMAPPED[f"data type {data_type!r}"] += 1
            return []
        purpose = _OPP115_PURPOSES_CF.get((value("Purpose") or "").casefold())
        return [Claim(kind="flow", category=cat,
                      action="collect" if category.startswith("First") else "share",
                      polarity="deny" if value("Does/Does Not") == "Does Not" else "affirm",
                      purposes=(purpose,) if purpose else ())]
    practice = None
    if category == "Data Retention":
        practice = "retention"
    elif category == "Data Security":
        practice = "encrypted_in_transit" if value("Security Measure") == "Secure data transfer" else "security_measures"
    elif category == "User Access, Edit and Deletion":
        access = value("Access Type") or ""
        practice = "deletion" if access.startswith("Delete") else (
            "access_correction" if access in ("Edit information", "View", "Export") else None)
    elif category == "User Choice/Control":
        practice = "user_choice"
    elif category == "International and Specific Audiences":
        practice = "children" if value("Audience Type") == "Children" else None
    elif category == "Policy Change":
        practice = "policy_change"
    elif category == "Other":
        practice = "privacy_contact" if value("Other Type") == "Privacy contact information" else None
    return [Claim(kind="practice", practice=practice)] if practice else []


def _clean_segment(html: str) -> str:
    return " ".join(re.sub(r"<[^>]+>", " ", html.replace("&nbsp;", " ").replace("&amp;", "&")).split())


def _load_opp115(folder: Path) -> list[GoldDoc]:
    files = sorted((folder / "sanitized_policies").glob("*.html"))
    ids = sorted((f.stem.split("_", 1)[0] for f in files), key=lambda i: hashlib.sha1(i.encode()).hexdigest())
    _OPP115_ORDER.update({i: "test" if n < 40 else "dev" if n < 50 else "train" for n, i in enumerate(ids)})
    labels_dir = folder / "consolidation" / "threshold-0.75-overlap-similarity"
    docs = []
    for f in files:
        policy_id = f.stem.split("_", 1)[0]
        segments = [_clean_segment(s) for s in f.read_text(encoding="utf-8", errors="replace").split("|||")]
        gold = [set() for _ in segments]
        with open(labels_dir / f"{f.stem}.csv", newline="", encoding="utf-8") as fh:
            for row in csv.reader(fh):
                seg = int(row[4])
                if seg < len(gold):
                    gold[seg].update(_opp115_claims(row[5], json.loads(row[6])))
        keep = [i for i, s in enumerate(segments) if s]
        text, spans = _join([segments[i] for i in keep])
        docs.append(GoldDoc("opp115", policy_id, opp115_split(policy_id), text, spans, [gold[i] for i in keep],
                            name=f.stem.split("_", 1)[1]))
    if UNMAPPED:
        print("OPP-115 annotations left out (no Play equivalent): "
              + ", ".join(f"{k} x{n}" for k, n in UNMAPPED.most_common()))
        UNMAPPED.clear()
    return docs


# ---------------------------------------------------------------- unibo
def _load_unibo(folder: Path) -> list[GoldDoc]:
    docs = []
    for name in UNIBO_NAMES:
        raw = json.loads((folder / f"{name}.json").read_text(encoding="utf-8"))
        plain = raw["document"]["plainText"]
        # Offsets refer to the text with Windows line ends; our text uses "\n", so shift them.
        crs = [i for i, ch in enumerate(plain) if ch == "\r"]
        text = plain.replace("\r", "")
        shift = lambda pos: pos - bisect_left(crs, pos)
        spans = segment(text)
        starts = [s for s, _ in spans]
        gold = [set() for _ in spans]
        seen = set()
        for a in raw["annotations"]:
            key = (a["start"], a["end"], a["type"])
            if a["type"] not in ("Category", "SubCategory") or key in seen:
                continue
            seen.add(key)
            cat = UNIBO_KINDS.get((a.get("attributes") or {}).get("Kind"))
            if cat is None:
                continue
            seg = max(0, bisect_left(starts, shift(a["start"]) + 1) - 1)
            gold[seg].add(Claim(kind="flow", category=cat))
        docs.append(GoldDoc("unibo", name, "test", text, spans, gold, name=name))
    return docs


# ---------------------------------------------------------------- hand-labelled Australian set
def parse_codes(codes: str) -> set[Claim]:
    """'Health and fitness:collect; Location:share:not; deletion' -> claims. ':sp' (service provider) is ignored here."""
    out = set()
    lookup = {k.casefold(): k for k in (*CATEGORIES, GENERIC)}
    for code in filter(None, (c.strip() for c in (codes or "").split(";"))):
        parts = [p.strip() for p in code.split(":")]
        polarity = "deny" if "not" in parts[1:] else "affirm"
        if parts[0] in PRACTICES:
            out.add(Claim(kind="practice", practice=parts[0], polarity=polarity))
        elif parts[0].casefold() in lookup and len(parts) > 1 and parts[1] in ("collect", "share", "sell"):
            out.add(Claim(kind="flow", category=lookup[parts[0].casefold()], action=parts[1], polarity=polarity))
        else:
            raise ValueError(f"cannot read code {code!r}")
    return out


def load_indomain(path: Path | None) -> list[GoldDoc]:
    """A filled-in annotation sheet (see evaluate.annotate) -> one GoldDoc per app."""
    if path is None:
        raise SystemExit("the indomain corpus needs --gold <adjudicated sheet.csv>")
    by_app: dict[str, list[dict]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            by_app.setdefault(row["app_id"], []).append(row)
    docs = []
    for app_id, rows in by_app.items():
        rows.sort(key=lambda r: int(r["seg"]))
        text, spans = _join([r["text"] for r in rows], sep="\n")
        docs.append(GoldDoc("indomain", app_id, "test", text, spans, [parse_codes(r["codes"]) for r in rows]))
    return docs
