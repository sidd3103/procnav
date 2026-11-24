"""Draft answers: the LLM writes prose from allowed facts only; code checks the citations.

Status, owner and open assumptions come from deterministic code, never from the LLM.
"""
from __future__ import annotations

from collections.abc import Callable
from datetime import date
from typing import TypeVar

from pydantic import BaseModel

from .cache import JsonCache, cache_key
from .extract import PROMPT_VERSION, LLMOut, _complete, _short
from .llm import LLMClient
from .models import Fact, Verdict
from .requirements import RequirementResult, Status

T = TypeVar("T", bound=BaseModel)

LLM_ERROR_ANSWER = "Draft unavailable (LLM error)"


class AnswerOut(LLMOut):
    answer: str
    cited_fact_ids: list[str] = []


class DraftAnswer(BaseModel):
    req_id: str
    answer: str
    cited: list[str]
    evidence_quotes: list[str]
    owner: str
    open_assumptions: list[str]
    status: Status
    citation_errors: list[str]


ANSWER_SYSTEM = """You draft one answer to a vendor-security / procurement requirement for a \
software supplier. You are given the requirement, the resolved state of each related topic, the \
ALLOWED FACTS (the only evidence you may use) and the OPEN ASSUMPTIONS.

Rules:
- Answer ONLY from the allowed facts. Do not use outside knowledge or invent values.
- A PROPOSED value is not confirmed: label it clearly as unconfirmed / proposed.
- State unknowns and open assumptions plainly, e.g. "inference location: not yet confirmed".
- Never claim approval, sign-off, compliance or any status. Status is decided elsewhere.
- Caveats are binding: an expired item must be described as expired, never as current. \
Judge dates against today's (as-of) date given in the request.
- Return the ids of the allowed facts you relied on in cited_fact_ids. Cite only allowed ids.
- Keep the answer short and factual."""


def cached_complete(llm: LLMClient, cache: JsonCache, key: str, system: str,
                    build_user: Callable[[], str], schema: type[T]) -> tuple[T | None, Exception | None]:
    """Cache hit, else one LLM call (retry once). Failures are returned, never cached."""
    cached = cache.get(key)
    if cached is not None:
        try:
            return schema.model_validate(cached), None
        except ValueError:
            pass
    out, err = _complete(llm, system, build_user(), schema)
    if out is not None:
        cache.put(key, out.model_dump())
    return out, err


def check_citations(cited_ids: list[str], allowed_ids: list[str]) -> tuple[list[str], list[str]]:
    """(valid cited ids, errors): ids outside the allowed set are errors; no valid id is 'uncited'."""
    allowed_set = set(allowed_ids)
    cited: list[str] = []
    errors: list[str] = []
    for i in cited_ids:
        if i in allowed_set:
            if i not in cited:
                cited.append(i)
        else:
            errors.append(f"unknown fact id: {i}")
    if allowed_set and not cited:
        errors.append("uncited")
    return cited, errors


def fact_lines(allowed: list[Fact]) -> list[str]:
    if not allowed:
        return ["(none)"]
    return [f'- id={f.id} value={f.value!r} certainty={f.certainty} quote="{f.quote}"'
            for f in allowed]


def _allowed_ids(result: RequirementResult, verdicts: dict[str, Verdict]) -> list[str]:
    ids: list[str] = []
    for t in result.req.topics:
        v = verdicts.get(t)
        if v is not None and v.status in ("VERIFIED", "PROPOSED"):
            ids.extend(i for i in v.used if i not in ids)
    return ids


def _open_assumptions(
    result: RequirementResult, verdicts: dict[str, Verdict], facts_by_id: dict[str, Fact]
) -> list[str]:
    out = list(result.open_assumptions)
    for t in result.req.topics:
        v = verdicts.get(t)
        if v is None or v.status != "CONFLICT":
            continue
        vals = sorted({str(facts_by_id[i].value) for i in v.used
                       if i in facts_by_id and facts_by_id[i].value is not None})
        msg = f"{t}: conflict" + (f" between {' vs '.join(vals)}" if vals else "")
        plain = f"{t}: conflict"
        out = [msg if a == plain else a for a in out]
        if msg not in out:
            out.append(msg)
    return out


def as_of_line(as_of: date | None) -> list[str]:
    return [f"Today (as-of) is {as_of.isoformat()}"] if as_of is not None else []


def with_caveats(text: str, topics: list[str], verdicts: dict[str, Verdict]) -> str:
    """Append a deterministic 'Caveats: ...' line (every caveat of the topics, sorted, unique),
    so LLM prose can never hide an expiry or a missing piece of evidence."""
    cav = sorted({c for t in topics if (v := verdicts.get(t)) is not None for c in v.caveats})
    return f"{text}\nCaveats: {'; '.join(cav)}" if cav else text


def _topic_lines(topics: list[str], verdicts: dict[str, Verdict]) -> list[str]:
    out: list[str] = []
    for t in topics:
        v = verdicts.get(t)
        if v is None:
            out.append(f"- {t}: UNKNOWN")
            continue
        cav = f" caveats: {'; '.join(v.caveats)}" if v.caveats else ""
        out.append(f"- {t}: {v.status} value={v.value!r}{cav}")
    return out


def _user_prompt(
    result: RequirementResult, verdicts: dict[str, Verdict],
    allowed: list[Fact], open_a: list[str], as_of: date | None = None,
) -> str:
    req = result.req
    lines = [f"KEY: answer:{req.id}", *as_of_line(as_of), f"REQUIREMENT: {req.text}", "",
             "TOPICS:"]
    lines += _topic_lines(req.topics, verdicts)
    lines += ["", "ALLOWED FACTS:"]
    lines += fact_lines(allowed)
    lines += ["", "OPEN ASSUMPTIONS:"]
    lines += [f"- {a}" for a in open_a] or ["(none)"]
    return "\n".join(lines)


def draft_answer(
    result: RequirementResult,
    verdicts: dict[str, Verdict],
    facts_by_id: dict[str, Fact],
    owner: str,
    llm: LLMClient,
    cache: JsonCache,
    as_of: date | None = None,
) -> DraftAnswer:
    req = result.req
    open_a = _open_assumptions(result, verdicts, facts_by_id)

    def make(answer: str, cited: list[str], quotes: list[str], errors: list[str]) -> DraftAnswer:
        answer = with_caveats(answer, req.topics, verdicts)
        return DraftAnswer(req_id=req.id, answer=answer, cited=cited, evidence_quotes=quotes,
                           owner=owner, open_assumptions=open_a, status=result.status,
                           citation_errors=errors)

    if all((verdicts.get(t) is None or verdicts[t].status == "UNKNOWN") for t in req.topics):
        detail = "; ".join(open_a) or "no information in case files"
        return make(f"Not yet known. Owner: {owner}. Open: {detail}", [], [], [])

    allowed_ids = [i for i in _allowed_ids(result, verdicts) if i in facts_by_id]
    allowed = [facts_by_id[i] for i in allowed_ids]
    key = cache_key("answer", PROMPT_VERSION, llm.model, ANSWER_SYSTEM, req.id, req.text,
                    sorted((f.id, f.value, f.quote) for f in allowed), open_a,
                    _topic_lines(req.topics, verdicts), as_of_line(as_of))

    out, err = cached_complete(llm, cache, key, ANSWER_SYSTEM,
                               lambda: _user_prompt(result, verdicts, allowed, open_a, as_of),
                               AnswerOut)
    if out is None:
        return make(LLM_ERROR_ANSWER, [], [], [f"llm_failed: {_short(err)}"])
    cited, errors = check_citations(out.cited_fact_ids, allowed_ids)
    return make(out.answer, cited, [facts_by_id[i].quote for i in cited], errors)
