from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

from .models import Dependency, Verdict
from .requirements import RequirementResult
from .topics import get_topic

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


class Blocker(BaseModel):
    item: str  # topic id
    reason: Literal["explicit_dependency", "caveat_before_milestone", "asked_hard_gap"]
    chain: list[str]
    quote: str | None = None
    subject: str | None = None  # explicit_dependency: topic id of what is blocked ("dpa")


def _expiry(caveat: str) -> tuple[bool, date | None]:
    """(is an expiry caveat, its date if one can be parsed)."""
    if not caveat.strip().lower().startswith(("expires", "expired")):
        return False, None
    if "unreadable date" in caveat:  # fail closed: never trust a date inside unreadable text
        return True, None
    m = _ISO.search(caveat)
    if m:
        try:
            return True, date.fromisoformat(m.group())
        except ValueError:
            pass
    return True, None


def _status(verdicts: dict[str, Verdict], topic: str) -> str:
    v = verdicts.get(topic)
    return v.status if v else "UNKNOWN"


def find_blockers(
    verdicts: dict[str, Verdict],
    results: list[RequirementResult],
    dependencies: list[Dependency],
    as_of: date,
) -> list[Blocker]:
    milestones = [d for d in dependencies if d.kind == "milestone"]
    dated = sorted((m for m in milestones if m.date), key=lambda m: m.date)  # stable
    primary = dated[0] if dated else (milestones[0] if milestones else None)
    out: list[Blocker] = []

    for dep in dependencies:
        if dep.kind != "blocked_until":
            continue
        for t in dep.depends_on_topics:
            st = _status(verdicts, t)
            if st == "VERIFIED":
                continue
            chain = [f"{t} {st}", dep.subject] + ([primary.subject] if primary else [])
            out.append(Blocker(item=t, reason="explicit_dependency", chain=chain, quote=dep.quote,
                               subject=get_topic(dep.subject).id))

    for t, v in verdicts.items():
        for cav in v.caveats:
            is_exp, exp = _expiry(cav)
            if not is_exp:
                continue
            hits = [m for m in milestones if m.date is None or exp is None or m.date >= exp]
            if not hits:
                continue
            hits.sort(key=lambda m: (m.date is None, m.date or date.max))
            out.append(Blocker(item=t, reason="caveat_before_milestone",
                               chain=[f"{t} {cav}", hits[0].subject], quote=cav))

    for res in results:
        if not res.req.from_customer:
            continue
        for t in res.req.topics:
            st = _status(verdicts, t)
            if st in ("UNKNOWN", "CONFLICT"):
                chain = [f"{t} {st}"] + ([primary.subject] if primary else [])
                out.append(Blocker(item=t, reason="asked_hard_gap", chain=chain,
                                   quote=res.req.quote))

    seen: set[tuple[str, str]] = set()
    uniq: list[Blocker] = []
    for b in out:
        k = (b.item, b.reason)
        if k not in seen:
            seen.add(k)
            uniq.append(b)
    return uniq


def group_blockers(blockers: list[Blocker]) -> dict[str, list[Blocker]]:
    """Blockers grouped by item (sorted), each group sorted by reason."""
    out: dict[str, list[Blocker]] = {}
    for b in sorted(blockers, key=lambda b: (b.item, b.reason)):
        out.setdefault(b.item, []).append(b)
    return out


# --- case-level verdicts derived from the run (deterministic; never VERIFIED) -------------

DERIVED_DEPENDENCY = "derived from dependency"
DERIVED_BLOCKERS = "derived from blocker list"
DERIVED_REVIEW = "derived from review policy"
DERIVED_RULES = (DERIVED_DEPENDENCY, DERIVED_BLOCKERS, DERIVED_REVIEW)
APPROVAL_TOPIC = "approval_blockers"
SUBMISSION_TOPIC = "submission_approval"


def dependencies_for(topic: str, dependencies: list[Dependency]) -> list[Dependency]:
    """`blocked_until` dependencies whose subject names this topic ("DPA" -> dpa)."""
    return [d for d in dependencies
            if d.kind == "blocked_until" and get_topic(d.subject).id == topic]


def derive_from_dependencies(verdicts: dict[str, Verdict],
                             dependencies: list[Dependency]) -> dict[str, Verdict]:
    """An UNKNOWN topic that is the subject of a blocked_until dependency is answered by that
    dependency: PROPOSED "blocked until <still-open topics> are confirmed", or "no longer
    blocked" once every depended-on topic is VERIFIED."""
    out = dict(verdicts)
    for t, v in verdicts.items():
        if v.status != "UNKNOWN":
            continue
        hits = dependencies_for(t, dependencies)
        on: list[str] = []
        for d in hits:
            on += [x for x in d.depends_on_topics if x != t and x not in on]
        if not on:
            continue
        still = [x for x in on if (w := verdicts.get(x)) is None or w.status != "VERIFIED"]
        verb = "is" if len(still) == 1 else "are"
        value = (f"blocked until {', '.join(still)} {verb} confirmed" if still
                 else f"no longer blocked: {', '.join(on)} are confirmed")
        out[t] = v.model_copy(update={
            "status": "PROPOSED", "value": value,
            "rule": DERIVED_DEPENDENCY, "hints": v.hints + [f"{d.quote} ({d.file})" for d in hits],
            "sendable": False})
    return out


def derive_approval_blockers(verdicts: dict[str, Verdict],
                             blockers: list[Blocker]) -> dict[str, Verdict]:
    """The approval-blockers topic (if asked) is answered from the blocker list."""
    v = verdicts.get(APPROVAL_TOPIC)
    if v is None:
        return verdicts
    items = sorted({b.item for b in blockers if b.item != APPROVAL_TOPIC})
    value = (f"{len(items)} items may delay approval: {', '.join(items)}" if items
             else "no blockers found in the case files")
    out = dict(verdicts)
    out[APPROVAL_TOPIC] = v.model_copy(update={
        "status": "PROPOSED", "value": value, "rule": DERIVED_BLOCKERS, "used": [],
        "sendable": False})
    return out


def derive_submission_approval(verdicts: dict[str, Verdict]) -> dict[str, Verdict]:
    """Which answers need sign-off is product policy, not case content: every external answer
    does (nothing is sent automatically). The draft lists them by readiness."""
    v = verdicts.get(SUBMISSION_TOPIC)
    if v is None:
        return verdicts
    out = dict(verdicts)
    out[SUBMISSION_TOPIC] = v.model_copy(update={
        "status": "PROPOSED", "value": "every response needs human approval before submission",
        "rule": DERIVED_REVIEW, "used": [], "sendable": False})
    return out
