"""Pure view helpers shared by the CLI and the Streamlit app (no Streamlit import here).

Every row is a plain dict so it can be printed, shown as a table, or tested directly.
Owners are suggestions only (routing is a human decision).
"""
from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .models import Rank, Verdict
from .pipeline import Snapshot, group_blockers
from .requirements import RequirementResult, readiness
from .topics import get_topic

Row = dict[str, Any]

# Answer readiness ("can we write a true answer now?") -> (icon, label). Deal risk (blocks the
# milestone or not) is a separate flag on each row.
READINESS: dict[str, tuple[str, str]] = {
    "ready": ("✅", "Ready to draft"),
    "confirm": ("🟡", "Needs someone to confirm"),
    "missing": ("❔", "Not in the files"),
}


def list_cases(root: str | Path) -> list[str]:
    """Case folders (non-hidden subdirectories) under root, sorted; [] when root is missing."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def latest_two(runs_dir: str | Path, case: str) -> tuple[Snapshot | None, Snapshot | None]:
    """(latest, previous) readable saved snapshots of a case; unreadable files are skipped."""
    d = Path(runs_dir) / case
    found: list[Snapshot] = []
    if d.is_dir():
        for p in sorted(d.glob("*.json"), reverse=True):
            try:
                found.append(Snapshot.model_validate_json(p.read_text(encoding="utf-8")))
            except (ValueError, OSError):
                continue
            if len(found) == 2:
                break
    found += [None, None]
    return found[0], found[1]


def _known(s: Snapshot) -> list[Verdict]:
    return [v for v in s.verdicts.values() if v.status in ("VERIFIED", "PROPOSED")]


def _unresolved(s: Snapshot) -> list[Verdict]:
    return [v for v in s.verdicts.values() if not v.sendable]


def summary_counts(s: Snapshot) -> dict[str, int]:
    return {
        "required": len(s.required),
        "known": len(_known(s)),
        "unresolved": len(_unresolved(s)),
        "blocking": len(group_blockers(s.blockers)),
    }


def owner(s: Snapshot, topic: str) -> str:
    return s.owners.get(topic, "?")


def known_rows(s: Snapshot) -> list[Row]:
    return [{"topic": v.topic, "status": v.status, "value": v.value, "sendable": v.sendable,
             "rule": v.rule, "used": list(v.used), "caveats": list(v.caveats)}
            for v in _known(s)]


def unresolved_rows(s: Snapshot) -> list[Row]:
    return [{"topic": v.topic, "status": v.status, "value": v.value, "owner": owner(s, v.topic),
             "rule": v.rule, "caveats": list(v.caveats), "hints": list(v.hints)}
            for v in _unresolved(s)]


def blocking_rows(s: Snapshot) -> list[Row]:
    return [{"item": item, "reasons": [b.reason for b in group], "owner": owner(s, item),
             "chains": [" → ".join(b.chain) for b in group],
             "questions": [r.req.id for r in s.required if item in r.req.topics]}
            for item, group in group_blockers(s.blockers).items()]


def next_step_rows(s: Snapshot) -> list[Row]:
    rows: list[Row] = []
    for st in s.next_steps:
        ch = next((c for c in s.chases if (c.owner, c.kind) == (st.owner, st.kind)), None)
        rows.append({
            "owner": st.owner, "kind": st.kind,
            "label": "open decision" if st.kind == "customer_decision" else "to chase",
            "items": list(st.items), "topics": list(st.topics),
            "chase": None if ch is None else {"message": ch.message, "cited": list(ch.cited),
                                              "citation_errors": list(ch.citation_errors)},
        })
    return rows


def fact_detail(s: Snapshot, fact_id: str) -> Row | None:
    f = next((x for x in s.facts if x.id == fact_id), None)
    if f is None:
        return None
    return {"id": f.id, "where": f"{f.file}:{f.line}" if f.line else f.file, "quote": f.quote,
            "claim": f.claim, "value": f.value, "certainty": f.certainty,
            "applies_to": f.applies_to, "source_type": f.source_type, "date": f.date,
            "speaker": f.speaker}


def _fact_or_stub(s: Snapshot, fact_id: str) -> Row:
    return fact_detail(s, fact_id) or {"id": fact_id, "where": "?", "quote": ""}


def verdict_detail(s: Snapshot, topic: str) -> Row | None:
    v = s.verdicts.get(topic)
    if v is None:
        return None
    return {
        "topic": v.topic, "status": v.status, "value": v.value, "sendable": v.sendable,
        "rule": v.rule, "caveats": list(v.caveats), "hints": list(v.hints),
        "owner": owner(s, topic),
        "used": [_fact_or_stub(s, i) for i in v.used],
        "excluded": [{**_fact_or_stub(s, i), "reason": r} for i, r in v.excluded],
    }


# ---------------------------------------------------------------- deal brief


def title(topic: str) -> str:
    return get_topic(topic).title or topic


def _caveat(v: Verdict) -> str:
    return "; ".join(c.removeprefix("no evidence: ").strip() for c in v.caveats)


_TOPIC_ORDER = {"UNKNOWN": 0, "CONFLICT": 1, "PROPOSED": 2, "VERIFIED": 3}


def topic_phrase(s: Snapshot, topic: str) -> str:
    """One plain clause about a topic's state, e.g. 'Where data is stored: “AWS…” (unconfirmed)'."""
    v = s.verdicts.get(topic)
    t = title(topic)
    if v is None or v.status == "UNKNOWN":
        return f"{t}: not in the files"
    if v.status == "CONFLICT":
        return f"{t}: sources disagree"
    if v.rule.startswith("derived from"):
        return f"{t}: worked out from the case"
    if v.status == "PROPOSED":
        return f"{t}: “{v.value}” (unconfirmed)"
    cav = _caveat(v)
    return f"{t}: “{v.value}” (evidence{'; ' + cav if cav else ''})"


def answer_why(s: Snapshot, topics: list[str], limit: int = 2) -> str:
    """Gaps first, then unconfirmed, then evidence; at most `limit` clauses."""
    if not topics:
        return "not linked to any topic"
    order = sorted(topics, key=lambda t: _TOPIC_ORDER.get(
        s.verdicts[t].status if t in s.verdicts else "UNKNOWN", 0))
    out = "; ".join(topic_phrase(s, t) for t in order[:limit])
    return out + (f" (+{len(order) - limit} more)" if len(order) > limit else "")


def _answer(s: Snapshot, req_id: str):  # -> DraftAnswer | None
    return next((a for a in s.answers if a.req_id == req_id), None)


def _is_template(r: RequirementResult) -> bool:
    return (r.req.group or "").startswith("template:")


def label_id(r: RequirementResult) -> str:
    """Questionnaire id as is; a customer ask by who asked (extracted ids are internal)."""
    if r.req.from_customer:
        return f"Email ask · {r.req.requester or 'customer'}"
    return r.req.id


def answer_rows(s: Snapshot, templates: bool = False) -> list[Row]:
    """One row per question/ask, grouped by readiness (ready, confirm, missing), original order
    within a group. `blocked` is the separate deal-risk flag."""
    blocked_q = {r.req.id for r in s.required if r.status == "blocked"}
    decisions = {d.id: d for d in s.open_decisions}
    rows: list[Row] = []
    for r in s.required:
        if _is_template(r) and not templates:
            continue
        a = _answer(s, r.req.id)
        rd = readiness(r.req.topics, s.verdicts)
        icon, label = READINESS[rd]
        rows.append({
            "id": r.req.id, "label_id": label_id(r), "question": r.req.text, "readiness": rd, "icon": icon,
            "label": label, "blocked": r.req.id in blocked_q,
            "why": answer_why(s, r.req.topics), "owner": a.owner if a else "?",
            "asked_by": r.req.requester if r.req.from_customer and r.req.requester
            else "questionnaire",
            "source_file": r.req.source_file, "ask_quote": r.req.quote,
            "template": _is_template(r),
            "answer": a.answer if a else "", "quotes": list(a.evidence_quotes) if a else [],
            "cited": list(a.cited) if a else [],
            "citation_errors": list(a.citation_errors) if a else [],
            "stale_draft": r.stale_draft, "topics": list(r.req.topics),
            "decisions": [f"{decisions[i].decider} decides: {decisions[i].text}"
                          for i in r.depends_on_decisions if i in decisions],
        })
    keys = list(READINESS)
    return sorted(rows, key=lambda row: keys.index(row["readiness"]))


def milestone(s: Snapshot) -> str | None:
    ms = [d for d in s.dependencies if d.kind == "milestone"]
    dated = sorted((m for m in ms if m.date), key=lambda m: m.date)
    first = dated[0] if dated else (ms[0] if ms else None)
    return first.subject if first else None


def brief(s: Snapshot) -> Row:
    rows = answer_rows(s)
    n = {k: sum(r["readiness"] == k for r in rows) for k in READINESS}
    n_block = len(group_blockers(s.blockers))
    people = len({st.owner for st in s.next_steps if st.kind == "internal"})
    ms = milestone(s) or "launch"
    sentence = (f"{n['ready']} answers ready to draft · {n['confirm']} need someone to confirm · "
                f"{n['missing']} not in the files. {n_block} things block the {ms}. "
                f"{people} people to chase.")
    return {**n, "blockers": n_block, "milestone": ms, "people": people, "sentence": sentence}


_REASON_ORDER = {"explicit_dependency": 0, "caveat_before_milestone": 1, "asked_hard_gap": 2}


def _blocker_why(s: Snapshot, b) -> str:
    if b.reason == "explicit_dependency":
        return f"{b.chain[1]} waits on it"
    if b.reason == "caveat_before_milestone":
        return f"{b.quote}, before {b.chain[-1]}"
    v = s.verdicts.get(b.item)
    gap = "sources disagree" if v is not None and v.status == "CONFLICT" else "not in the files"
    return f"the customer asked; {gap}"


def blocker_actions(s: Snapshot) -> list[Row]:
    """What blocks the milestone, as plain sentences with the suggested owner."""
    rows: list[Row] = []
    for item, group in group_blockers(s.blockers).items():
        group = sorted(group, key=lambda b: _REASON_ORDER[b.reason])
        rows.append({
            "topic": item, "title": title(item), "owner": owner(s, item),
            "why": "; ".join(_blocker_why(s, b) for b in group),
            "rank": _REASON_ORDER[group[0].reason],
            "questions": list(dict.fromkeys(label_id(r) for r in s.required
                                            if item in r.req.topics and not _is_template(r))),
        })
    return sorted(rows, key=lambda r: (r["rank"], r["title"]))


def _chase_line(s: Snapshot, topic: str, blocked: set[str], ms: str) -> str:
    v = s.verdicts.get(topic)
    t = title(topic)
    if v is None or v.status == "UNKNOWN":
        line = f"{t}: not in the files"
    elif v.status == "CONFLICT":
        line = f"{t}: sources disagree, settle which is right"
    elif v.status == "PROPOSED":
        line = f"{t}: confirm “{v.value}”"
    else:
        line = f"{t}: {_caveat(v) or 'confirm for external use'}"
    return line + (f" — blocks the {ms}" if topic in blocked else "")


def chase_cards(s: Snapshot) -> list[Row]:
    """One card per suggested owner: plain to-do lines plus the drafted message."""
    blocked = {b.item for b in s.blockers}
    ms = milestone(s) or "launch"
    cards: list[Row] = []
    for row in next_step_rows(s):
        lines = [] if row["kind"] == "customer_decision" else [
            _chase_line(s, t, blocked, ms) for t in row["topics"]
            if not (t in s.verdicts and s.verdicts[t].rule.startswith("derived from"))]
        lines += ["Decide: " + re.sub(r" \(affects [^)]*\)$", "", i.split("): ", 1)[1])
                  for i in row["items"] if i.startswith("decide (")]
        ch = row["chase"] or {}
        cards.append({"owner": row["owner"], "kind": row["kind"], "lines": lines,
                      "message": ch.get("message", ""), "cited": ch.get("cited", []),
                      "citation_errors": ch.get("citation_errors", [])})
    return cards


def conflict_rows(s: Snapshot) -> list[Row]:
    """CONFLICT topics with each side's value and source."""
    out: list[Row] = []
    for t, v in sorted(s.verdicts.items()):
        if v.status != "CONFLICT":
            continue
        sides = [{"value": f["value"] if f.get("value") is not None else "", "where": f["where"],
                  "quote": f["quote"]} for f in (_fact_or_stub(s, i) for i in v.used)]
        out.append({"topic": t, "title": title(t), "owner": owner(s, t), "sides": sides,
                    "hints": list(v.hints)})
    return out


# ---------------------------------------------------------------- sources & decisions

RANK_ORDER = ["evidence", "customer-specific", "generic", "other customer", "—"]
_RANK_NAME = {Rank.EVIDENCE: "evidence", Rank.CUSTOMER: "customer-specific",
              Rank.GENERIC: "generic", Rank.OTHER_CUSTOMER: "other customer"}


def source_rows(s: Snapshot) -> list[Row]:
    """One row per loaded file: how extraction judged it (source type, rank, applicability).

    Files with no facts (parsed in code, or nothing extracted) show "—" and sort last.
    """
    by_file: dict[str, list] = {}
    for f in s.facts:
        by_file.setdefault(f.file, []).append(f)
    failed = dict(s.failed_files)
    counts: dict[str, Counter] = {}
    for r in s.required:
        counts.setdefault(r.req.source_file, Counter())["asks" if r.req.from_customer
                                                       else "questions"] += 1
    for d in s.dependencies:
        counts.setdefault(d.file, Counter())[
            "milestones" if d.kind == "milestone" else "dependencies"] += 1
    for o in s.open_decisions:
        counts.setdefault(o.file, Counter())["decisions"] += 1
    rows: list[Row] = []
    for path in sorted(s.files):
        fs = by_file.get(path, [])
        c = Counter({"facts": len(fs)}) + counts.get(path, Counter())
        parts = [f"{n} {k if n > 1 else k.removesuffix('s').replace('dependencie', 'dependency')}"
                 for k in ("facts", "asks", "questions", "dependencies", "milestones",
                           "decisions") if (n := c[k])]
        types = Counter(f.source_type for f in fs)
        rank = _RANK_NAME[max(f.rank for f in fs)] if fs else "—"
        rows.append({
            "file": path,
            "source_type": types.most_common(1)[0][0] if fs else "—",
            "rank": rank,
            "facts": len(fs),
            "extracted": ", ".join(parts) or "nothing",
            "applies_to": ", ".join(sorted({f.applies_to for f in fs})),
            "note": f"failed: {failed[path]}" if path in failed else "",
        })
    rows.sort(key=lambda r: (RANK_ORDER.index(r["rank"]), r["file"]))
    return rows


def decision_rows(s: Snapshot) -> list[Row]:
    """Open decisions: what is undecided, who decides, which topics wait on it."""
    return [{"text": d.text, "decider": d.decider, "topics": list(d.affected_topics),
             "quote": d.quote, "file": d.file} for d in s.open_decisions]
