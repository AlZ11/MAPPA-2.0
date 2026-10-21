"""The row form the policy reader outputs, and the only values each field may take."""

from typing import Literal, get_args

from pydantic import BaseModel, field_validator

# Google Play Data Safety's 14 data categories.
DataType = Literal[
    "Location",
    "Personal info",
    "Financial info",
    "Health and fitness",
    "Messages",
    "Photos and videos",
    "Audio files",
    "Files and docs",
    "Calendar",
    "Contacts",
    "App activity",
    "Web browsing",
    "App info and performance",
    "Device or other IDs",
]

Action = Literal["collect", "share", "sell"]

# Play's purposes, plus "other" for anything the policy states that doesn't fit.
Purpose = Literal[
    "app_functionality",
    "analytics",
    "developer_communications",
    "advertising_or_marketing",
    "fraud_prevention_security_compliance",
    "personalization",
    "account_management",
    "other",
    "unspecified",
]

Condition = Literal["required", "optional", "with_consent", "unspecified"]

# recipient is the one open field: either one of these two fixed values, or a company name.
FIRST_PARTY = "first_party"
UNNAMED_THIRD_PARTY = "unnamed_third_party"

DATA_TYPES: tuple[str, ...] = get_args(DataType)
ACTIONS: tuple[str, ...] = get_args(Action)
PURPOSES: tuple[str, ...] = get_args(Purpose)
CONDITIONS: tuple[str, ...] = get_args(Condition)


class Row(BaseModel):
    """One claim from a policy: 'we <action> <data_type> to <recipient> for <purpose>'.

    negated=True flips it: 'we do NOT <action> <data_type>'. Those denials are what the
    consistency engine later checks against the Play label, the APK and the traffic.
    """

    data_type: DataType
    action: Action
    recipient: str
    purpose: Purpose
    condition: Condition
    quote: str
    negated: bool = False

    @field_validator("recipient")
    @classmethod
    def recipient_is_sane(cls, v: str) -> str:
        v = v.strip()
        if not v or len(v) > 80:
            raise ValueError("recipient must be a short non-empty name")
        return v

    @field_validator("quote")
    @classmethod
    def quote_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("quote too short to be evidence")
        return v

    @property
    def is_third_party(self) -> bool:
        return self.recipient != FIRST_PARTY


# ---------------------------------------------------------------- non-data claims
# Everything a policy says that isn't "we collect/share/sell X": the spec's later forms.
# Each topic has its own fixed list of values. Denials get their own value ("no_...") because
# for these topics the denial is usually a distinct statement, not a flipped practice.
CLAIM_VALUES: dict[str, tuple[str, ...]] = {
    "security": (
        "encryption_in_transit", "encryption_at_rest", "access_controls", "security_testing",
        "general_security_measures", "no_security_guarantee",
    ),
    "retention": (
        "fixed_period",            # detail = the period, e.g. "30 days"
        "until_account_deleted", "as_long_as_necessary", "anonymised_after_use",
        "retained_indefinitely",
    ),
    "user_rights": (
        "access", "correction", "deletion", "portability", "opt_out_marketing",
        "opt_out_tracking_or_ads", "withdraw_consent", "complain_to_regulator",
    ),
    "breach_and_contact": (
        "privacy_contact",         # detail = the email / address / role given
        "notify_users_of_breach", "notify_regulator_of_breach",
    ),
    "children": (
        "not_directed_at_children", "minimum_age",  # detail = the age, e.g. "13"
        "parental_consent", "deletes_child_data_found", "directed_at_children",
    ),
    "automated_decisions": (
        "solely_automated_decisions", "automated_assistance_in_decisions", "profiling",
        "data_used_in_decisions_described", "human_review_available", "no_automated_decisions",
    ),
    "ai": (
        "uses_ai_features", "data_sent_to_ai_provider",  # detail = provider name if given
        "data_used_to_train_ai", "data_not_used_to_train_ai",
    ),
}

CLAIM_TOPICS: tuple[str, ...] = tuple(CLAIM_VALUES)


class Claim(BaseModel):
    """A non-data statement: '<topic>: <value> (<detail>)', backed by a verbatim quote."""

    topic: str
    value: str
    detail: str = ""  # short text copied from the quote (a period, an age, a company, an email)
    quote: str

    @field_validator("topic")
    @classmethod
    def topic_allowed(cls, v: str) -> str:
        if v not in CLAIM_VALUES:
            raise ValueError(f"unknown topic {v!r}")
        return v

    @field_validator("value")
    @classmethod
    def value_allowed_for_topic(cls, v: str, info) -> str:
        topic = info.data.get("topic")  # missing if the topic itself already failed
        if topic is None or v not in CLAIM_VALUES[topic]:
            raise ValueError(f"value {v!r} not allowed for topic {topic!r}")
        return v

    @field_validator("detail")
    @classmethod
    def detail_short(cls, v: str) -> str:
        v = (v or "").strip()
        if len(v) > 120:
            raise ValueError("detail must be short")
        return v

    @field_validator("quote")
    @classmethod
    def quote_not_empty(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("quote too short to be evidence")
        return v
