"""Deterministic resolver: gives each topic a verdict. Pure; no LLM, no I/O."""
from __future__ import annotations

import re
from datetime import date, timedelta

from .models import Fact, Rank, Verdict
from .topics import get_topic

EXPIRY_WINDOW_DAYS = 30


def normalize_value(v: str) -> str:
    s = re.sub(r"\s+", " ", v.lower()).strip()
    return re.sub(r"\b(\d+)\s*d\b", r"\1 days", s)


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    try:
        return date.fromisoformat(s)
    except ValueError:
        return None


def _val(f: Fact) -> str:
    """Display text only (hints); candidates always carry a value of their own."""
    return f.value if f.value else f.claim


def _has_value(f: Fact) -> bool:
    return bool(f.value and f.value.strip())


def _pick(values: list[str]) -> str:
    """Value-choice rule: the longest original text, then lexicographic (deterministic)."""
    return min(values, key=lambda v: (-len(v), v))


def _merge_by_file(cands: list[Fact]) -> dict[str, tuple[str, str]]:
    """R26: facts from one file are complementary aspects of one statement.

    Per file: facts with the same normalised value collapse (text by the value-choice rule);
    the distinct aspects are joined with "; " in line order. Returns
    {file: (normalised merged value, display merged value)}. Input must be sorted by
    (file, line, id), which makes the result independent of input order.
    """
    out: dict[str, tuple[str, str]] = {}
    by_file: dict[str, dict[str, list[str]]] = {}
    for f in cands:
        aspects = by_file.setdefault(f.file, {})  # dict keeps first-seen (line) order
        aspects.setdefault(normalize_value(_val(f)), []).append(_val(f))
    for file, aspects in by_file.items():
        out[file] = ("; ".join(aspects), "; ".join(_pick(vs) for vs in aspects.values()))
    return out


def _caveats_for(used: list[Fact], as_of: date) -> list[str]:
    out: list[str] = []
    limit = as_of + timedelta(days=EXPIRY_WINDOW_DAYS)
    for f in used:
        if f.valid_until is None:
            if f.valid_until_raw:  # fail closed: a written but unreadable expiry
                c = f"expires on unreadable date '{f.valid_until_raw}'"
                if c not in out:
                    out.append(c)
            continue
        if f.valid_until < as_of:
            c = f"expired {f.valid_until.isoformat()}"
        elif f.valid_until <= limit:
            c = f"expires {f.valid_until.isoformat()}"
        else:
            continue
        if c not in out:
            out.append(c)
    return out


def resolve_topic(topic_id: str, facts: list[Fact], as_of: date) -> Verdict:
    facts = sorted(facts, key=lambda f: (f.file, f.line or 0, f.id))
    topic = get_topic(topic_id)
    per_customer = topic.scope == "per_customer"
    excluded: list[tuple[str, str]] = []
    hints: list[str] = []
    caveats: list[str] = []
    cands: list[Fact] = []
    absents: list[Fact] = []

    for f in facts:
        if f.applies_to == "other_customer" or f.rank == Rank.OTHER_CUSTOMER:
            hints.append(f"other customer: {_val(f)} ({f.file})")
            excluded.append((f.id, "other customer"))
        elif f.certainty == "absent":
            absents.append(f)
        elif not _has_value(f):
            hints.append(f"open: {f.claim} ({f.file})")
            excluded.append((f.id, "no value stated"))
        elif per_customer and f.rank < Rank.CUSTOMER:
            hints.append(f"generic: {_val(f)} ({f.file})")
            excluded.append((f.id, "generic source on per-customer topic"))
        else:
            cands.append(f)

    def verdict(status, value, used, rule):
        # An absence ("no remediation evidence") is superseded when the single winning group
        # holds a stated fact of at least the absence's rank, unless the absence is dated
        # strictly newer than every such fact.
        for a in absents:
            # a file that says it lacks X cannot itself be the evidence for X
            support = [f for f in used if f.certainty == "stated" and f.rank >= a.rank
                       and f.file != a.file]
            a_date = _parse_date(a.date)
            s_dates = [_parse_date(f.date) for f in support]
            newer = a_date is not None and all(d is not None and d < a_date for d in s_dates)
            if status in ("VERIFIED", "PROPOSED") and support and not newer:
                excluded.append((a.id, "absence superseded by evidence"))
            else:
                caveats.append(f"no evidence: {a.claim}")
                excluded.append((a.id, f"absence: {a.claim}"))
        cav = caveats + _caveats_for(used, as_of)
        return Verdict(
            topic=topic_id, status=status, value=value, used=[f.id for f in used],
            excluded=excluded, hints=hints, rule=rule, caveats=cav,
            sendable=status == "VERIFIED" and not cav,
        )

    if not cands:
        rule = "no usable this-customer facts" if per_customer else "no usable facts"
        return verdict("UNKNOWN", None, [], rule)

    top = max(f.rank for f in cands)
    for f in cands:
        if f.rank < top:
            excluded.append((f.id, "outranked by higher rank"))
    cands = [f for f in cands if f.rank == top]

    dates = {f.id: _parse_date(f.date) for f in cands}
    dated = [d for d in dates.values() if d is not None]
    if dated:
        newest = max(dated)
        kept = []
        for f in cands:
            d = dates[f.id]
            if d is not None and d < newest:
                excluded.append((f.id, "superseded by newer"))
            else:
                kept.append(f)
        cands = kept

    # CONFLICT only between different files whose (merged) values differ
    merged = _merge_by_file(cands)
    if len({norm for norm, _ in merged.values()}) > 1:
        return verdict("CONFLICT", None, cands, "conflicting values at same rank")

    value = _pick([shown for _, shown in merged.values()])

    if top == Rank.EVIDENCE and all(f.certainty == "stated" for f in cands):
        return verdict("VERIFIED", value, cands, "evidence, stated")
    if top == Rank.EVIDENCE:
        return verdict("PROPOSED", value, cands, "evidence, not stated")
    if not per_customer and top == Rank.GENERIC:
        return verdict("PROPOSED", value, cands, "generic platform fact: needs confirmation")
    return verdict("PROPOSED", value, cands, "unverified source")


def resolve_all(facts: list[Fact], as_of: date) -> dict[str, Verdict]:
    by_topic: dict[str, list[Fact]] = {}
    for f in facts:
        by_topic.setdefault(f.topic, []).append(f)
    return {t: resolve_topic(t, fs, as_of) for t, fs in by_topic.items()}
