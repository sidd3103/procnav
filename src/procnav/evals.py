"""Score a saved run against a hand-labelled answer key. Pure; no LLM, no I/O.

The key (JSON, case-specific, lives outside src/) lists:
- questions: by questionnaire `id`, or customer asks by `match` (all substrings in the ask
  text): allowed statuses, substrings the draft must / must not include.
- topics: allowed verdict statuses, value substrings, and source files / quotes the verdict
  must not be built from.
- answers_must_not_include: substrings banned from every draft answer.
A must_include entry is a string, or a list of alternatives (any one is enough). Matching is
case-insensitive.
"""
from __future__ import annotations

from pydantic import BaseModel

from .pipeline import Snapshot

Needle = str | list[str]


class QuestionKey(BaseModel):
    id: str | None = None
    match: list[str] = []
    statuses: list[str] = []
    must_include: list[Needle] = []
    must_not_include: list[str] = []
    note: str = ""


class TopicKey(BaseModel):
    topic: str
    statuses: list[str] = []
    value_includes: list[Needle] = []
    used_files_exclude: list[str] = []
    used_quotes_exclude: list[str] = []
    note: str = ""


class AnswerKey(BaseModel):
    case: str
    as_of: str
    questions: list[QuestionKey] = []
    topics: list[TopicKey] = []
    answers_must_not_include: list[str] = []


class Check(BaseModel):
    name: str
    passed: bool
    detail: str = ""


def _has(text: str, needle: Needle) -> bool:
    alts = [needle] if isinstance(needle, str) else needle
    low = text.casefold()
    return any(a.casefold() in low for a in alts)


def _needle_name(needle: Needle) -> str:
    return repr(needle) if isinstance(needle, str) else f"one of {needle!r}"


def _short(text: str, n: int = 160) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 3] + "..."


def _question_checks(s: Snapshot, q: QuestionKey) -> list[Check]:
    if q.id is not None:
        label = q.id
        res = next((r for r in s.required if r.req.id == q.id), None)
    else:
        label = f"customer ask {q.match!r}"
        res = next((r for r in s.required if r.req.from_customer
                    and all(m.casefold() in r.req.text.casefold() for m in q.match)), None)
    if res is None:
        return [Check(name=f"{label} present", passed=False, detail="question not found in run")]
    ans = next((a for a in s.answers if a.req_id == res.req.id), None)
    text = ans.answer if ans is not None else ""
    out: list[Check] = []
    if q.statuses:
        out.append(Check(name=f"{label} status", passed=res.status in q.statuses,
                         detail=f"got {res.status}, allowed {q.statuses}"))
    for n in q.must_include:
        out.append(Check(name=f"{label} includes {_needle_name(n)}", passed=_has(text, n),
                         detail=_short(text)))
    for n in q.must_not_include:
        out.append(Check(name=f"{label} excludes {n!r}", passed=not _has(text, n),
                         detail=_short(text)))
    return out


def _topic_checks(s: Snapshot, t: TopicKey) -> list[Check]:
    label = f"topic {t.topic}"
    v = s.verdicts.get(t.topic)
    facts = {f.id: f for f in s.facts}
    used = [facts[i] for i in (v.used if v else []) if i in facts]
    out: list[Check] = []
    if t.statuses:
        got = v.status if v else "missing"
        out.append(Check(name=f"{label} status", passed=v is not None and v.status in t.statuses,
                         detail=f"got {got}, allowed {t.statuses}"))
    for n in t.value_includes:
        val = (v.value or "") if v else ""
        out.append(Check(name=f"{label} value includes {_needle_name(n)}",
                         passed=bool(val) and _has(val, n), detail=f"value {val!r}"))
    for sub in t.used_files_exclude:
        bad = [f.file for f in used if sub.casefold() in f.file.casefold()]
        out.append(Check(name=f"{label} not from {sub!r}", passed=not bad,
                         detail=", ".join(bad)))
    for sub in t.used_quotes_exclude:
        bad = [f.quote for f in used if sub.casefold() in f.quote.casefold()]
        out.append(Check(name=f"{label} quotes exclude {sub!r}", passed=not bad,
                         detail=" | ".join(_short(q, 80) for q in bad)))
    return out


def score(s: Snapshot, key: AnswerKey) -> list[Check]:
    out: list[Check] = []
    for q in key.questions:
        out += _question_checks(s, q)
    for t in key.topics:
        out += _topic_checks(s, t)
    for n in key.answers_must_not_include:
        bad = [a.req_id for a in s.answers if _has(a.answer, n)]
        out.append(Check(name=f"answers exclude {n!r}", passed=not bad, detail=", ".join(bad)))
    return out


def render_checks(checks: list[Check]) -> str:
    lines = [f"FAIL {c.name}" + (f" — {c.detail}" if c.detail else "")
             for c in checks if not c.passed]
    passed = sum(c.passed for c in checks)
    lines.append(f"passed {passed}/{len(checks)}")
    return "\n".join(lines)
