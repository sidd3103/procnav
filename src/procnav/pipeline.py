"""End-to-end run over one case folder, plus snapshot save/load.

Order: load -> identify -> parse questionnaire/people in code -> extract every other file
-> map questionnaire topics -> build requirements -> map new topics -> reconcile -> resolve -> evaluate -> blockers ->
owners/next steps -> draft answers. The Snapshot holds no timestamps, so an unchanged
folder produces an identical Snapshot; the run time lives only in the saved filename.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from .answers import DraftAnswer, draft_answer
from .chase import ChaseMessage, draft_chase
from .blockers import (
    Blocker,
    derive_approval_blockers,
    derive_from_dependencies,
    derive_submission_approval,
    find_blockers,
    group_blockers,
)
from .cache import JsonCache
from .extract import (
    CaseInfo,
    extract_file,
    identify_case,
    map_new_topics,
    map_questions_checked,
)
from .llm import LLMClient
from .loader import CaseFile, Person, load_case, parse_people, parse_questionnaire
from .models import Dependency, Fact, OpenDecision, Rank, Requirement, Verdict
from .owners import assign_decision_owner, assign_owner
from .reconcile import reconcile_all
from .requirements import (
    RequirementResult,
    apply_blockers,
    build_requirements,
    evaluate,
    template_flags,
)
from .resolve import resolve_all, resolve_topic
from .topics import get_topic


class NextStep(BaseModel):
    owner: str
    items: list[str]
    kind: Literal["internal", "customer_decision"]
    topics: list[str] = []


class Snapshot(BaseModel):
    case: str
    as_of: date
    customer: str
    required: list[RequirementResult]
    verdicts: dict[str, Verdict]
    facts: list[Fact]
    blockers: list[Blocker]
    next_steps: list[NextStep]
    answers: list[DraftAnswer]
    failed_files: list[tuple[str, str]]
    skipped_files: list[str]
    dropped_quotes: list[str]
    supplier: str = ""
    files: dict[str, str] = {}  # relative path -> sha256 of every loaded file
    owners: dict[str, str] = {}
    dependencies: list[Dependency] = []
    open_decisions: list[OpenDecision] = []
    chases: list[ChaseMessage] = []
    new_topics: list[str] = []  # `other:*` topics left after mapping ("new topic discovered")


def _verdict_label(v: Verdict) -> str:
    return v.status + (f" {v.value}" if v.value else "")


def _add(by_owner: dict[str, list[str]], owner: str, topics: list[str]) -> None:
    lst = by_owner.setdefault(owner, [])
    lst += [t for t in topics if t not in lst]


def _next_steps(
    verdicts: dict[str, Verdict],
    blockers: list[Blocker],
    reqs: list[Requirement],
    decisions: list[OpenDecision],
    owners: dict[str, str],
    people: list[Person],
) -> list[NextStep]:
    needed_by: dict[str, list[str]] = {}
    for r in reqs:
        for t in r.topics:
            needed_by.setdefault(t, [])
            if r.id not in needed_by[t]:
                needed_by[t].append(r.id)

    items: dict[str, list[str]] = {}
    topics: dict[str, list[str]] = {}
    cust_topics: dict[str, list[str]] = {}
    n_block: dict[str, int] = {}
    for item, group in group_blockers(blockers).items():
        owner = owners[item]
        reasons = ", ".join(b.reason for b in group)
        chains = " | ".join(" → ".join(b.chain) for b in group)
        _add(topics, owner, [item])
        items.setdefault(owner, []).append(f"BLOCKER {item} ({reasons}): {chains}")
        n_block[owner] = n_block.get(owner, 0) + 1  # distinct blocked items per owner
    for t in sorted(verdicts):
        v = verdicts[t]
        if v.status == "VERIFIED" and v.sendable:
            continue
        _add(topics, owners[t], [t])
        line = f"{t}: {_verdict_label(v)}"
        if v.caveats:
            line += f" [caveats: {'; '.join(v.caveats)}]"
        if needed_by.get(t):
            line += f" — needed by {', '.join(needed_by[t])}"
        items.setdefault(owners[t], []).append(line)

    # open decisions: internal ones join the owner's internal entry, customer ones are separate
    cust_items: dict[str, list[str]] = {}
    for d in sorted(decisions, key=lambda d: d.id):
        topic = get_topic(d.affected_topics[0]) if d.affected_topics else None
        owner = assign_decision_owner(d.decider, people, topic)
        affects = f" (affects {', '.join(d.affected_topics)})" if d.affected_topics else ""
        line = f"decide ({d.decider}): {d.text}{affects}"
        target = cust_items if d.decider == "customer" else items
        _add(cust_topics if d.decider == "customer" else topics, owner, d.affected_topics)
        target.setdefault(owner, []).append(line)

    steps = [NextStep(owner=o, items=its, kind="internal", topics=topics.get(o, [])) for o, its in items.items()]
    steps.sort(key=lambda s: (-n_block.get(s.owner, 0), s.owner))
    steps += [NextStep(owner=o, items=cust_items[o], kind="customer_decision",
                       topics=cust_topics.get(o, []))
              for o in sorted(cust_items)]
    return steps


def _other_examples(facts: list[Fact], reqs: list[Requirement], deps: list[Dependency],
                    decisions: list[OpenDecision]) -> dict[str, list[str]]:
    """Every `other:*` topic id in the run, with up to 2 example texts (facts first)."""
    out: dict[str, list[str]] = {}

    def add(topics: list[str], text: str) -> None:
        for t in topics:
            if t.startswith("other:"):
                ex = out.setdefault(t, [])
                if len(ex) < 2 and text and text not in ex:
                    ex.append(text)

    for f in facts:
        add([f.topic], f.claim or f.quote)
    for r in reqs:
        add(r.topics, r.text)
    for d in deps:
        add(d.depends_on_topics, d.subject)
    for o in decisions:
        add(o.affected_topics, o.text)
    return out


def _remap(topics: list[str], mapping: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for t in topics:
        out += [m for m in mapping.get(t, [t]) if m not in out]
    return out


def _apply_topic_map(mapping: dict[str, list[str]], facts: list[Fact], reqs: list[Requirement],
                     deps: list[Dependency], decisions: list[OpenDecision]):
    """Rewrite topics via the mapping. Topic lists take every mapped seed id; a fact (one
    value for one topic) is moved only when its topic maps to exactly one seed id."""
    if not mapping:
        return facts, reqs, deps, decisions
    facts = [f.model_copy(update={"topic": mapping[f.topic][0]})
             if len(mapping.get(f.topic, [])) == 1 else f for f in facts]
    reqs = [r.model_copy(update={"topics": _remap(r.topics, mapping)}) for r in reqs]
    deps = [d.model_copy(update={"depends_on_topics": _remap(d.depends_on_topics, mapping)})
            for d in deps]
    decisions = [o.model_copy(update={"affected_topics": _remap(o.affected_topics, mapping)})
                 for o in decisions]
    return facts, reqs, deps, decisions


SERVICE_TOPIC = "service_description"


def _service_fact(info: CaseInfo, files: list[CaseFile], case: str) -> Fact | None:
    """One fact for the proposed service, only when identify's quote is verbatim in a file."""
    q = info.service_quote
    if not info.service or not q:
        return None
    for f in sorted(files, key=lambda f: f.path):
        if q in f.text:
            line = f.text[:f.text.index(q)].count("\n") + 1
            return Fact(
                id=f"{Path(f.path).with_suffix('').as_posix()}#S1", file=f.path, line=line,
                topic=SERVICE_TOPIC, claim=f"Proposed service: {info.service}",
                value=info.service, applies_to="this_customer", certainty="stated", date=None,
                speaker=None, quote=q, valid_until=None, rank=Rank.CUSTOMER,
                source_type="case_header", case=case, customer=info.customer)
    return None


def run_case(case_dir: str | Path, as_of: date, llm: LLMClient,
             cache_dir: str | Path) -> Snapshot:
    case_dir = Path(case_dir)
    if not case_dir.is_dir():
        raise NotADirectoryError(f"case folder not found: {case_dir}")
    case = case_dir.resolve().name
    cache = JsonCache(cache_dir)

    files, skipped = load_case(case_dir)
    info = identify_case(files, llm, cache, case=case)  # raises RuntimeError on failure

    questionnaire: list[Requirement] = []
    people: list[Person] = []
    facts: list[Fact] = []
    extracted_reqs: list[Requirement] = []
    deps: list[Dependency] = []
    decisions: list[OpenDecision] = []
    failed: list[tuple[str, str]] = []
    dropped: list[str] = []
    svc = _service_fact(info, files, case)
    if svc is not None:
        facts.append(svc)
    for f in files:
        q = parse_questionnaire(f)
        p = parse_people(f)
        if q is not None or p is not None:
            questionnaire += q or []
            people += p or []
            continue
        ef = extract_file(f, info, llm, cache)
        if ef.failed is not None:
            failed.append((ef.file, ef.failed))
            continue
        facts += ef.facts
        extracted_reqs += ef.requirements
        deps += ef.dependencies
        decisions += ef.open_decisions
        dropped += [f"{ef.file}: {d}" for d in ef.dropped]

    # topics on real rows first (templates are not sent), so template representatives can
    # inherit their section's topics in build_requirements
    flags = template_flags(questionnaire)
    real = [r for r, t in zip(questionnaire, flags) if not t]
    mapped, map_err = map_questions_checked(real, llm, cache)
    if map_err is not None:
        failed.append(("map_questions", map_err))
    by_id = {r.id: r for r in mapped}
    questionnaire = [by_id.get(r.id, r) if not t else r for r, t in zip(questionnaire, flags)]
    reqs = build_requirements(extracted_reqs, questionnaire)

    # map topics the extractor invented onto seed topics (separate cached step)
    mapping, topics_err = map_new_topics(_other_examples(facts, reqs, deps, decisions), llm, cache)
    if topics_err is not None:
        failed.append(("map_topics", topics_err))
    facts, reqs, deps, decisions = _apply_topic_map(mapping, facts, reqs, deps, decisions)

    # cross-file meaning labels per topic (LLM); resolve decides what they change
    recon, recon_failed = reconcile_all(facts, llm, cache)
    failed += recon_failed

    verdicts = resolve_all(facts, as_of, recon)
    referenced = {t for r in reqs for t in r.topics}
    referenced |= {t for d in deps for t in d.depends_on_topics}
    referenced |= {t for d in decisions for t in d.affected_topics}
    for t in referenced - verdicts.keys():
        verdicts[t] = resolve_topic(t, [], as_of)
    verdicts = dict(sorted(verdicts.items()))

    # case-level answers derived deterministically from dependencies and the blocker list
    verdicts = derive_submission_approval(derive_from_dependencies(verdicts, deps))
    results = evaluate(reqs, verdicts, decisions)
    blockers = find_blockers(verdicts, results, deps, as_of)
    verdicts = derive_approval_blockers(verdicts, blockers)
    results = evaluate(reqs, verdicts, decisions)
    blockers = find_blockers(verdicts, results, deps, as_of)
    results = apply_blockers(results, blockers)

    owners = {t: assign_owner(get_topic(t), people) for t in verdicts}
    next_steps = _next_steps(verdicts, blockers, reqs, decisions, owners, people)

    facts_by_id = {fa.id: fa for fa in facts}
    answers = [
        draft_answer(r, verdicts, facts_by_id,
                     owners[r.req.topics[0]] if r.req.topics else "Unassigned", llm, cache,
                     as_of=as_of, blockers=blockers, dependencies=deps, results=results)
        for r in results
    ]

    chases = sorted((draft_chase(st, verdicts, facts_by_id, llm, cache, as_of=as_of) for st in next_steps),
                    key=lambda c: (c.owner, c.kind))

    return Snapshot(
        case=case, as_of=as_of, customer=info.customer, supplier=info.supplier,
        files={f.path: f.sha256 for f in files},
        required=results, verdicts=verdicts, facts=sorted(facts, key=lambda x: x.id),
        blockers=sorted(blockers, key=lambda b: (b.item, b.reason)), next_steps=next_steps,
        answers=answers, failed_files=sorted(failed), skipped_files=sorted(skipped),
        dropped_quotes=sorted(dropped), owners=owners,
        dependencies=sorted(deps, key=lambda d: d.id),
        open_decisions=sorted(decisions, key=lambda d: d.id), chases=chases,
        new_topics=[t for t in verdicts if t.startswith("other:")],
    )


def save_snapshot(s: Snapshot, runs_dir: str | Path) -> Path:
    d = Path(runs_dir) / s.case
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.json"
    p.write_text(s.model_dump_json(indent=2), encoding="utf-8")
    return p


def load_latest(runs_dir: str | Path, case: str) -> Snapshot | None:
    d = Path(runs_dir) / case
    if not d.is_dir():
        return None
    for p in sorted(d.glob("*.json"), reverse=True):
        try:
            return Snapshot.model_validate_json(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
    return None
