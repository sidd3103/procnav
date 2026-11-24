from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from .models import OpenDecision, Requirement, Verdict
from .resolve import normalize_value

if TYPE_CHECKING:
    from .blockers import Blocker

Status = Literal["ready_for_review", "needs_confirmation", "blocked", "unknown"]
Readiness = Literal["ready", "confirm", "missing"]
_SEVERITY = {"ready_for_review": 0, "needs_confirmation": 1, "unknown": 2}


class RequirementResult(BaseModel):
    req: Requirement
    status: Status
    topic_verdicts: list[str]
    open_assumptions: list[str]
    stale_draft: str | None
    depends_on_decisions: list[str]


def _template_key(r: Requirement) -> str:
    return re.sub(r"\d+", "", r.text).strip().lower()


def template_flags(questionnaire_reqs: list[Requirement]) -> list[bool]:
    """Per row: True when it is a repeated template row (text minus digits occurs >= 3 times)."""
    counts: dict[str, int] = {}
    for r in questionnaire_reqs:
        counts[_template_key(r)] = counts.get(_template_key(r), 0) + 1
    return [counts[_template_key(r)] >= 3 for r in questionnaire_reqs]


def build_requirements(
    extracted_reqs: list[Requirement], questionnaire_reqs: list[Requirement]
) -> list[Requirement]:
    """Collapse repeated template questionnaire rows to one representative per section.

    A row is a template row when its text, digits removed, occurs in >= 3 rows. The
    representative (first template row of its section) carries the union of the topics of
    the section's non-template rows. Extracted requirements are appended after.
    """
    is_tmpl = template_flags(questionnaire_reqs)

    section_topics: dict[str | None, list[str]] = {}
    for r, t in zip(questionnaire_reqs, is_tmpl):
        if not t:
            lst = section_topics.setdefault(r.section, [])
            lst.extend(x for x in r.topics if x not in lst)

    out: list[Requirement] = []
    seen_sections: set[str | None] = set()
    for r, t in zip(questionnaire_reqs, is_tmpl):
        if not t:
            out.append(r)
        elif r.section not in seen_sections:
            seen_sections.add(r.section)
            topics = list(section_topics.get(r.section, [])) or list(r.topics)
            out.append(r.model_copy(update={"group": f"template:{r.section}", "topics": topics}))
    return out + list(extracted_reqs)


def _topic_status(v: Verdict | None) -> Status:
    if v is None or v.status in ("UNKNOWN", "CONFLICT"):
        return "unknown"
    if v.status == "VERIFIED" and v.sendable:
        return "ready_for_review"
    return "needs_confirmation"


def _stale_draft(req: Requirement, verdicts: dict[str, Verdict]) -> str | None:
    if not req.draft_answer or not req.draft_answer.strip():
        return None
    msg = f"unsourced draft '{req.draft_answer}'"
    draft = normalize_value(req.draft_answer)
    contra = [f"{t}='{v.value}'" for t in req.topics
              if (v := verdicts.get(t)) is not None and v.value
              and normalize_value(v.value) not in draft]
    return f"{msg}; contradicts {', '.join(contra)}" if contra else msg


def evaluate(
    reqs: list[Requirement], verdicts: dict[str, Verdict], decisions: list[OpenDecision]
) -> list[RequirementResult]:
    results: list[RequirementResult] = []
    for req in reqs:
        status: Status = "unknown" if not req.topics else "ready_for_review"
        tv: list[str] = []
        open_a: list[str] = []
        for t in req.topics:
            v = verdicts.get(t)
            st = _topic_status(v)
            if _SEVERITY[st] > _SEVERITY[status]:
                status = st
            if v is None:
                tv.append(f"{t}: UNKNOWN")
            else:
                tv.append(f"{t}: {v.status}" + (f" {v.value}" if v.value else ""))
            if v is None or v.status in ("UNKNOWN", "CONFLICT"):
                open_a.append(f"{t}: {v.status.lower() if v else 'unknown'}")
        deps = [d.id for d in decisions if set(d.affected_topics) & set(req.topics)]
        results.append(RequirementResult(
            req=req, status=status, topic_verdicts=tv, open_assumptions=open_a,
            stale_draft=_stale_draft(req, verdicts), depends_on_decisions=deps))
    return results


DECISION_CAVEAT = "open decision"


def apply_decision_caveats(verdicts: dict[str, Verdict],
                           decisions: list[OpenDecision]) -> dict[str, Verdict]:
    """A pending decision on a topic is a caveat: the value may still change, so the topic is
    never sendable and its answers need confirmation. UNKNOWN and derived topics are left as is."""
    out = dict(verdicts)
    for d in decisions:
        cav = f"{DECISION_CAVEAT} ({d.decider}): {d.text}"
        for t in d.affected_topics:
            v = out.get(t)
            if v is None or v.status == "UNKNOWN" or v.rule.startswith("derived from") \
                    or cav in v.caveats:
                continue
            out[t] = v.model_copy(update={"caveats": [*v.caveats, cav], "sendable": False})
    return out


def readiness(topics: list[str], verdicts: dict[str, Verdict]) -> Readiness:
    """Can we write a true answer now? Separate from deal risk (blocked or not).

    ready: every topic is VERIFIED evidence (an expiry caveat is part of the answer; an open
    decision is not, it means the value may change),
    or any topic is derived by code from the case (code then writes the whole answer). missing: no topic has anything usable. confirm: the
    rest (proposed values, conflicts, or partly known).
    """
    vs = [verdicts.get(t) for t in topics]
    known = [v for v in vs if v is not None and v.status != "UNKNOWN"]
    if not known:
        return "missing"
    if any(v.rule.startswith("derived from") for v in known):
        return "ready"
    if len(known) == len(vs) and all(
            v.status == "VERIFIED" and not any(c.startswith(DECISION_CAVEAT) for c in v.caveats)
            for v in known):
        return "ready"
    return "confirm"


def apply_blockers(
    results: list[RequirementResult], blockers: list[Blocker]
) -> list[RequirementResult]:
    """Return copies with status 'blocked' for requirements touching a blocked topic, or the
    subject of an explicit dependency that is still blocked (e.g. a question about the DPA)."""
    items = {b.item for b in blockers} | {b.subject for b in blockers if b.subject}
    return [
        r.model_copy(update={"status": "blocked"}) if items & set(r.req.topics) else r
        for r in results
    ]
