"""Chase messages: one drafted message per NextStep; the LLM writes, code checks citations.

Only `used` facts of VERIFIED/PROPOSED verdicts are citable. Labelled hints are shown as
starting points, never as evidence.
"""
from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from .answers import (
    _topic_lines,
    as_of_line,
    cached_complete,
    check_citations,
    fact_lines,
    with_caveats,
)
from .cache import JsonCache, cache_key
from .extract import PROMPT_VERSION, LLMOut, _short
from .llm import LLMClient
from .models import Fact, Verdict

if TYPE_CHECKING:
    from .pipeline import NextStep

LLM_ERROR_MESSAGE = "Draft unavailable (LLM error)"


class ChaseOut(LLMOut):
    message: str
    cited_fact_ids: list[str] = []


class ChaseMessage(BaseModel):
    owner: str
    kind: Literal["internal", "customer_decision"]
    message: str
    cited: list[str]
    evidence_quotes: list[str]
    citation_errors: list[str]


CHASE_SYSTEM = """You draft ONE short, consolidated message to a single owner, chasing the open \
items on a vendor-procurement case. You are given the owner's open ITEMS, the resolved state of \
the related topics, the ALLOWED FACTS (the only evidence you may use) and optional STARTING \
POINTS.

Rules:
- Write from the allowed facts only. Do not add claims, values, dates or names that are not given.
- Say what is already known (from the allowed facts; label PROPOSED values as unconfirmed), then \
state exactly what is needed from this owner for each open item.
- STARTING POINTS are not evidence: you may mention them as leads to check, never as facts.
- Never claim approval, sign-off or compliance. Status is decided elsewhere.
- If the message is a customer decision, ask the customer to decide; do not decide for them.
- Caveats are binding: an expired item must be described as expired, never as current. \
Judge dates against today's (as-of) date given in the request.
- Return the ids of the allowed facts you relied on in cited_fact_ids. Cite only allowed ids.
- Keep it short."""


def _allowed_ids(topics: list[str], verdicts: dict[str, Verdict]) -> list[str]:
    ids: list[str] = []
    for t in topics:
        v = verdicts.get(t)
        if v is not None and v.status in ("VERIFIED", "PROPOSED"):
            ids.extend(i for i in v.used if i not in ids)
    return ids


def _hints(topics: list[str], verdicts: dict[str, Verdict]) -> list[str]:
    out: list[str] = []
    for t in topics:
        v = verdicts.get(t)
        out += [f"- {t}: {h}" for h in (v.hints if v else [])]
    return out


def _user_prompt(step: NextStep, verdicts: dict[str, Verdict], allowed: list[Fact],
                 hints: list[str], as_of: date | None = None) -> str:
    lines = [f"KEY: chase:{step.kind}:{step.owner}", *as_of_line(as_of), f"OWNER: {step.owner}",
             f"KIND: {step.kind}"]
    if step.kind == "customer_decision":
        lines.append("This is a customer decision: ask the customer to decide the open items.")
        lines.append("This goes to the customer: do not mention other customers, internal file "
                     "names or internal people.")
    lines += ["", "OPEN ITEMS:"] + [f"- {i}" for i in step.items]
    lines += ["", "TOPICS:"] + _topic_lines(step.topics, verdicts)
    lines += ["", "ALLOWED FACTS:"] + fact_lines(allowed)
    lines += ["", "STARTING POINTS (not evidence):"] + (hints or ["(none)"])
    return "\n".join(lines)


def draft_chase(
    next_step: NextStep,
    verdicts: dict[str, Verdict],
    facts_by_id: dict[str, Fact],
    llm: LLMClient,
    cache: JsonCache,
    as_of: date | None = None,
) -> ChaseMessage:
    def make(message: str, cited: list[str], quotes: list[str], errors: list[str]):
        # customer-facing messages get no caveat line: caveat text carries internal wording
        if next_step.kind != "customer_decision":
            message = with_caveats(message, next_step.topics, verdicts)
        return ChaseMessage(owner=next_step.owner, kind=next_step.kind, message=message,
                            cited=cited, evidence_quotes=quotes, citation_errors=errors)

    allowed_ids = [i for i in _allowed_ids(next_step.topics, verdicts) if i in facts_by_id]
    allowed = [facts_by_id[i] for i in allowed_ids]
    # customer-facing messages get no starting points: hints name other customers and files
    hints = [] if next_step.kind == "customer_decision" else _hints(next_step.topics, verdicts)
    key = cache_key("chase", PROMPT_VERSION, llm.model, CHASE_SYSTEM, next_step.owner,
                    next_step.kind,
                    next_step.items, sorted((f.id, f.value, f.quote) for f in allowed), hints,
                    _topic_lines(next_step.topics, verdicts), as_of_line(as_of))

    out, err = cached_complete(llm, cache, key, CHASE_SYSTEM,
                               lambda: _user_prompt(next_step, verdicts, allowed, hints, as_of),
                               ChaseOut)
    if out is None:
        return make(LLM_ERROR_MESSAGE, [], [], [f"llm_failed: {_short(err)}"])
    cited, errors = check_citations(out.cited_fact_ids, allowed_ids)
    return make(out.message, cited, [facts_by_id[i].quote for i in cited], errors)
