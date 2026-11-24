from __future__ import annotations

from datetime import date
from enum import IntEnum
from typing import Literal

from pydantic import BaseModel

Certainty = Literal["stated", "proposed", "uncertain", "absent"]
AppliesTo = Literal["this_customer", "generic", "other_customer"]
SourceType = Literal[
    "customer_request", "questionnaire", "internal_chat", "customer_config_draft",
    "generic_doc", "prior_customer", "evidence", "guidance", "people", "case_header",
]


class Rank(IntEnum):
    OTHER_CUSTOMER = 1
    GENERIC = 2
    CUSTOMER = 3
    EVIDENCE = 4


class Fact(BaseModel):
    id: str
    file: str
    line: int | None
    topic: str
    claim: str
    value: str | None
    applies_to: AppliesTo
    certainty: Certainty
    date: str | None
    speaker: str | None
    quote: str
    valid_until: date | None
    rank: Rank
    source_type: SourceType
    case: str
    customer: str
    valid_until_raw: str | None = None  # written expiry that could not be parsed (fails closed)


class Requirement(BaseModel):
    id: str
    text: str
    requester: str | None
    topics: list[str]
    source_file: str
    quote: str | None
    draft_answer: str | None
    from_customer: bool
    group: str | None
    section: str | None = None
    note: str | None = None


class Dependency(BaseModel):
    id: str
    kind: Literal["blocked_until", "milestone"]
    subject: str
    depends_on_topics: list[str]
    date: date | None
    quote: str
    file: str


class OpenDecision(BaseModel):
    id: str
    text: str
    decider: Literal["customer", "internal"]
    affected_topics: list[str]
    quote: str
    file: str


class Verdict(BaseModel):
    topic: str
    status: Literal["VERIFIED", "PROPOSED", "CONFLICT", "UNKNOWN"]
    value: str | None
    used: list[str]
    excluded: list[tuple[str, str]]
    hints: list[str]
    rule: str
    caveats: list[str]
    sendable: bool


class Topic(BaseModel):
    id: str
    description: str
    scope: Literal["platform_wide", "per_customer"]
    area: str
    keywords: list[str]


class Reconciliation(BaseModel):
    """Per-topic meaning labels from the reconciler. Code decides what they change.

    off_topic: fact ids not about the topic. agree / contradict: unordered pairs of fact ids
    (from different files) whose values say the same thing / contradict each other.
    """
    off_topic: set[str] = set()
    agree: set[frozenset[str]] = set()
    contradict: set[frozenset[str]] = set()
