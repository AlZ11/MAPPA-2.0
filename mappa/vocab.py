"""Shared vocabulary: Google Play's Data Safety terms and MAPPA's one row shape.

Every evidence source (label and policy now; code and traffic in Build 2) writes rows in these
terms, so comparing two sources is a plain query instead of custom glue code per pair.
"""

import re
from dataclasses import dataclass

# Play's 14 data categories and their 38 data types, spelled exactly as the Data Safety page does.
CATEGORIES: dict[str, tuple[str, ...]] = {
    "Location": ("Approximate location", "Precise location"),
    "Personal info": (
        "Name", "Email address", "User IDs", "Address", "Phone number", "Race and ethnicity",
        "Political or religious beliefs", "Sexual orientation", "Other info",
    ),
    "Financial info": ("User payment info", "Purchase history", "Credit score", "Other financial info"),
    "Health and fitness": ("Health info", "Fitness info"),
    "Messages": ("Emails", "SMS or MMS", "Other in-app messages"),
    "Photos and videos": ("Photos", "Videos"),
    "Audio": ("Voice or sound recordings", "Music files", "Other audio files"),  # verified on a live label
    "Files and docs": ("Files and docs",),
    "Calendar": ("Calendar events",),
    "Contacts": ("Contacts",),
    "App activity": (
        "App interactions", "In-app search history", "Installed apps",
        "Other user-generated content", "Other actions",
    ),
    "Web browsing": ("Web browsing history",),
    "App info and performance": ("Crash logs", "Diagnostics", "Other app performance data"),
    "Device or other IDs": ("Device or other IDs",),
}
SUBTYPE_PARENT = {sub: cat for cat, subs in CATEGORIES.items() for sub in subs}

# Policies often say "we collect personal information" without naming a type. That is real
# evidence (a vague disclosure) but it fits no Play category, so it gets its own bucket.
GENERIC = "Generic personal data"

PURPOSES: tuple[str, ...] = (
    "App functionality",
    "Analytics",
    "Developer communications",
    "Advertising or marketing",
    "Fraud prevention, security, and compliance",
    "Personalization",
    "Account management",
)

ACTIONS = ("collect", "share", "sell")
POLARITIES = ("affirm", "deny")
SOURCES = ("label", "policy")  # Build 2 adds "code" and "traffic"

# Who receives shared data. Play's "shared" excludes service providers, legal requests and
# anonymised data, so a policy saying "we share with our hosting provider" does NOT contradict a
# label saying "no data shared". Keeping service_provider separate is what lets Build 2 tell them apart.
RECIPIENTS = ("service_provider", "affiliate", "authorities", "third_party", "unspecified")

# Statements that are not "we collect/share X". One line each: the LLM prompt, the annotation
# codebook and this file all use the same definitions.
PRACTICES: dict[str, str] = {
    "encrypted_in_transit": "Data is encrypted while sent over the network (TLS/SSL, HTTPS).",
    "security_measures": "Other safeguards: encryption at rest, access controls, audits, security testing.",
    "retention": "How long personal data is kept, or when it is deleted.",
    "deletion": "Users can delete their data or account, or ask for it to be deleted.",
    "access_correction": "Users can access, export or correct their personal data.",
    "user_choice": "Users can opt in or out of a collection, use or sharing (consent, marketing, ads).",
    "children": "Statements about children or minors: age limits, parental consent, not directed at children.",
    "policy_change": "How users are told when the privacy policy changes.",
    "privacy_contact": "How to contact the developer about privacy (email, address, officer).",
    "breach_notification": "Notifying users or a regulator after a data breach.",
    "automated_decisions": (
        "Personal information is used by a computer program to make, or substantially help make, "
        "decisions that could significantly affect a person (Australian APP 1.7-1.9)."
    ),
    "ai_processing": "Personal data is processed by AI features or sent to an AI model provider.",
    "overseas_disclosure": "Personal data is sent to, stored in or accessed from other countries (APP 8).",
}

# Play's "Security practices" lines -> (practice, polarity). Keys are normalised by label_key().
LABEL_PRACTICES: dict[str, tuple[str, str]] = {
    "data is encrypted in transit": ("encrypted_in_transit", "affirm"),
    "data isn't encrypted": ("encrypted_in_transit", "deny"),
    "you can request that data be deleted": ("deletion", "affirm"),
    "data can't be deleted": ("deletion", "deny"),
    "committed to follow the play families policy": ("families_policy", "affirm"),
    "independent security review": ("independent_review", "affirm"),
}

# Data Safety section titles -> action. An "empty" title is the developer's explicit "none".
SECTION_TITLES = {"data shared": "share", "data collected": "collect"}
EMPTY_TITLES = {"no data shared with third parties": "share", "no data collected": "collect"}

# Keyword rules: the baseline for practices that no public dataset labels.
RULES: dict[str, re.Pattern] = {
    "automated_decisions": re.compile(
        r"automated[- ](?:decision|processing|means)|solely automated|profiling"
        r"|algorithm(?:ic|s)? (?:to )?(?:decide|decision)",
        re.I,
    ),
    "ai_processing": re.compile(
        r"artificial intelligence|machine learning|\bA\.?I\.?\b|large language model|\bLLMs?\b"
        r"|OpenAI|ChatGPT|\bGPT\b|Anthropic|Gemini|chatbot",
    ),
    "overseas_disclosure": re.compile(
        r"overseas|outside (?:of )?(?:australia|your (?:country|jurisdiction)|the country)"
        r"|cross[- ]border|international(?:ly)? transfer|transferr?ed to (?:other )?countries",
        re.I,
    ),
    "breach_notification": re.compile(r"(?:data|security) breach|notifiable", re.I),
}

# No public corpus has enough health-data labels to train on (OPP-115's are almost all in one hospital
# policy, which sits in its test split), so the baseline finds health data flows with keywords.
HEALTH_TERMS = re.compile(
    r"\bhealth|\bmedical|\bfitness|\bsymptom|\bdiagnos(?:is|es|ed|e|ing)\b|\bmedication|\bprescription|heart rate"
    r"|blood pressure|\bglucose|blood sugar|\bweight\b|\bBMI\b|step count|\bsleep|\bmenstrua|\bpregnan|\bfertility"
    r"|\bovulation|\bmood\b|\bmental\b|\bcalorie|\bnutrition|\bworkout|\bbiometric"
    r"|\bexercis(?:e|es|ing)\b(?!\s+(?:of\s+)?(?:your|their|his|her|my|our|any|these|those|such|this|the|a|certain)\b)",
    re.I,
)
DATA_VERBS = re.compile(
    r"\b(?:collect(?:s|ed|ing)?|gather(?:s|ed|ing)?|obtain(?:s|ed|ing)?|receiv(?:e|es|ed|ing)|record(?:s|ed|ing)?"
    r"|stor(?:e|es|ed|ing|age)|track(?:s|ed|ing)?|process(?:es|ed|ing)?|use[sd]?|using|shar(?:e|es|ed|ing)"
    r"|disclos(?:e|es|ed|ing|ure)|transfer(?:s|red|ring)?|sell(?:s|ing)?|sold|provid(?:e|es|ed|ing)|access(?:es|ed|ing)?)\b",
    re.I,
)
SHARE_WORDS = re.compile(r"\b(?:shar(?:e|es|ed|ing)|disclos(?:e|es|ed|ing|ure)|transfer(?:s|red|ring)?|sell(?:s|ing)?|sold"
                         r"|third[- ]part(?:y|ies)|partners?)\b", re.I)


@dataclass(frozen=True)
class Claim:
    """One row of evidence, from any source.

    kind="flow":     "<we> <action> <category/subtype> [to <recipient>] [for <purposes>]".
    kind="practice": "<practice>" (e.g. deletion), affirmed or denied.
    polarity="deny" turns either into "we do NOT ...": the most useful thing a policy can say,
    because a denial plus code or traffic that does the thing is a direct contradiction.
    """

    kind: str
    polarity: str = "affirm"
    category: str | None = None
    subtype: str | None = None
    action: str | None = None
    practice: str | None = None
    recipient: str | None = None
    detail: str | None = None  # a company, period, age or AI provider, copied from the quote
    purposes: tuple[str, ...] = ()
    optional: bool | None = None  # label only: the user can choose not to provide it
    quote: str | None = None  # verbatim policy text, or a pointer into the label
    seg: int | None = None  # which policy segment the quote was found in
    start: int | None = None  # character offsets of the quote in the policy text
    end: int | None = None


def label_key(s: str) -> str:
    """Normalise a Play label string for lookup: curly apostrophes, case, spacing."""
    return " ".join(s.replace("’", "'").casefold().split())


def split_purposes(s: str) -> tuple[tuple[str, ...], str]:
    """'App functionality, Fraud prevention, security, and compliance' -> both purposes.

    Splitting on commas would break the purpose that contains commas, so this matches the known
    purposes longest-first and returns whatever is left over, so that nothing is silently lost.
    """
    found, rest = set(), s
    for purpose in sorted(PURPOSES, key=len, reverse=True):
        if purpose in rest:
            found.add(purpose)
            rest = rest.replace(purpose, " ")
    pieces = (re.sub(r"^and\b\s*", "", piece.strip()) for piece in rest.split(","))
    return tuple(p for p in PURPOSES if p in found), ", ".join(p for p in pieces if p)
