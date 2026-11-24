from __future__ import annotations

import re
from datetime import date
from typing import Literal

from pydantic import BaseModel

from .models import Dependency, Verdict
from .requirements import RequirementResult

_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


class Blocker(BaseModel):
    item: str  # topic id
    reason: Literal["explicit_dependency", "caveat_before_milestone", "asked_hard_gap"]
    chain: list[str]
    quote: str | None = None


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
            out.append(Blocker(item=t, reason="explicit_dependency", chain=chain, quote=dep.quote))

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
