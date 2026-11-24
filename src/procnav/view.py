"""Pure view helpers shared by the CLI and the Streamlit app (no Streamlit import here).

Every row is a plain dict so it can be printed, shown as a table, or tested directly.
Owners are suggestions only (routing is a human decision).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .models import Rank, Verdict
from .pipeline import Snapshot, group_blockers
from .requirements import RequirementResult, _topic_status

Row = dict[str, Any]

# Backend requirement status -> (icon, plain-English label), most severe first.
STATUS_LABELS: dict[str, tuple[str, str]] = {
    "blocked": ("⛔", "Blocks launch"),
    "unknown": ("❔", "Missing information"),
    "needs_confirmation": ("🟡", "Drafted, needs sign-off"),
    "ready_for_review": ("✅", "Ready for Legal"),
}
_SEVERITY = {k: i for i, k in enumerate(STATUS_LABELS)}


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


# ---------------------------------------------------------------- question-first view


def _words(topic: str) -> str:
    return topic.removeprefix("other:").replace("_", " ")


def _caveat(v: Verdict | None) -> str | None:
    if v is None or not v.caveats:
        return None
    return v.caveats[0].removeprefix("no evidence: ").strip()


def _topic_level(s: Snapshot, topic: str, blocked: set[str]) -> str:
    """The question status this one topic would cause on its own."""
    return "blocked" if topic in blocked else _topic_status(s.verdicts.get(topic))


def _topic_reason(s: Snapshot, level: str, topic: str) -> str:
    v = s.verdicts.get(topic)
    words = _words(topic)
    if level == "blocked":
        chain = list(group_blockers(s.blockers)[topic][0].chain)
        chain[0] = chain[0].replace(topic, words, 1)
        return " → ".join(chain)
    if (cav := _caveat(v)) is not None:
        return cav
    if level == "unknown":
        if v is not None and v.status == "CONFLICT":
            return f"sources disagree on {words}"
        return f"nobody has stated {words}"
    if level == "needs_confirmation":
        return f"{words}: {v.value if v else '?'} (unconfirmed)"
    return f"evidence: {words} = {v.value if v else '?'}"


def question_reason(r: RequirementResult, s: Snapshot) -> str:
    """One plain line explaining a question's status, from its worst topic."""
    topics = r.req.topics
    if not topics:
        reason = "not linked to any topic"
    else:
        blocked = {b.item for b in s.blockers}
        levels = {t: _topic_level(s, t, blocked) for t in topics}
        worst = min(levels.values(), key=_SEVERITY.__getitem__)
        at_worst = [t for t in topics if levels[t] == worst]
        reason = _topic_reason(s, worst, at_worst[0])
        if len(at_worst) > 1:
            reason += f" (+{len(at_worst) - 1} more)"
    if r.depends_on_decisions:
        reason += " · waiting on decision"
    return reason


def _answer(s: Snapshot, req_id: str):  # -> DraftAnswer | None
    return next((a for a in s.answers if a.req_id == req_id), None)


def _question_row(s: Snapshot, r: RequirementResult) -> Row:
    req, a = r.req, _answer(s, r.req.id)
    icon, label = STATUS_LABELS[r.status]
    return {
        "id": req.id, "question": req.text, "status": r.status, "icon": icon, "label": label,
        "reason": question_reason(r, s), "owner": a.owner if a else "?",
        "asked_by": req.requester if req.from_customer and req.requester else "questionnaire",
        "template": (req.group or "").startswith("template:"),
        "answer": a.answer if a else "",
    }


def question_rows(s: Snapshot) -> list[Row]:
    """One row per question, most severe status first, original order within a status."""
    rows = [_question_row(s, r) for r in s.required]
    return sorted(rows, key=lambda row: _SEVERITY[row["status"]])


def question_counts(s: Snapshot) -> dict[str, int]:
    """Questions per plain-English label (all four labels, severity order)."""
    counts = {label: 0 for _, label in STATUS_LABELS.values()}
    for r in s.required:
        counts[STATUS_LABELS[r.status][1]] += 1
    return counts


def filter_questions(rows: list[Row], labels: list[str], owner_: str, text: str) -> list[Row]:
    """Keep rows whose label is in labels (empty = any), owner matches ("All" = any) and
    whose id, question or reason contains text (case-insensitive)."""
    needle = text.strip().lower()
    return [r for r in rows
            if (not labels or r["label"] in labels)
            and (owner_ == "All" or r["owner"] == owner_)
            and (not needle or needle in f"{r['id']} {r['question']} {r['reason']}".lower())]


def question_detail(s: Snapshot, req_id: str) -> Row | None:
    r = next((x for x in s.required if x.req.id == req_id), None)
    if r is None:
        return None
    a = _answer(s, req_id)
    decisions = [{"id": d.id, "text": d.text, "decider": d.decider}
                 for d in s.open_decisions if d.id in r.depends_on_decisions]
    decisions += [{"id": i, "text": i, "decider": "?"} for i in r.depends_on_decisions
                  if i not in {d["id"] for d in decisions}]
    return {
        **_question_row(s, r),
        "cited": list(a.cited) if a else [],
        "evidence_quotes": list(a.evidence_quotes) if a else [],
        "open_assumptions": list(a.open_assumptions) if a else [],
        "citation_errors": list(a.citation_errors) if a else [],
        "stale_draft": r.stale_draft, "quote": r.req.quote, "source_file": r.req.source_file,
        "decisions": decisions, "topics": list(r.req.topics),
    }


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
