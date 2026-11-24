"""Per-topic reconciliation: the LLM labels what facts from different files MEAN.

Per-file extraction cannot see other files, and code can only compare strings ("AWS Canada"
vs "Canada"). For each topic whose facts come from two or more files, one LLM call labels:
  - per fact: is it actually about this topic (yes / no / unclear)
  - per cross-file pair of valued facts: same / more_specific / compatible / contradicts /
    unclear.
Code turns the labels into a `Reconciliation` and `resolve.resolve_topic` decides what they
change (asymmetric trust: raising a conflict is applied directly, clearing one is capped at
PROPOSED). Missing or unknown labels mean "no label" (fail closed).
"""
from __future__ import annotations

from .answers import cached_complete
from .cache import JsonCache, cache_key
from .extract import LLMOut, _short
from .llm import LLMClient
from .models import Fact, Reconciliation
from .topics import get_topic

PROMPT_VERSION = "r1"

_AGREE = {"same", "more_specific", "compatible"}

SYSTEM = """You compare facts about ONE topic that were extracted separately from different \
files of one procurement case. You only label meaning; you never decide what is true or \
which answer to send.

For each fact, say whether it is about the topic as described (about_topic: yes / no / \
unclear). A fact about a neighbouring subject (e.g. whether payloads are captured, filed under \
how long traces are kept) is "no".

For each listed pair (a, b) of facts from different files, label how their values relate:
- same: they state the same thing in different words.
- more_specific: one states the other with more detail and no contradiction \
(e.g. "AWS Canada" vs "Canada").
- compatible: different aspects that can both be true at once.
- contradicts: they cannot both be true, or they differ in a condition, scope, period or \
qualifier that changes the answer (e.g. "30 days" vs "30 days, this customer only"; \
"OpenAI" vs "OpenAI and Anthropic").
- unclear: you cannot tell.
When in doubt, use unclear. Use only the ids given."""


class FactLabel(LLMOut):
    id: str = ""
    about_topic: str = "unclear"


class PairLabel(LLMOut):
    a: str = ""
    b: str = ""
    relation: str = "unclear"


class ReconcileOut(LLMOut):
    facts: list[FactLabel] = []
    pairs: list[PairLabel] = []


def _relevant(facts: list[Fact]) -> list[Fact]:
    return sorted((f for f in facts if f.applies_to != "other_customer"),
                  key=lambda f: (f.file, f.line or 0, f.id))


def needs_reconcile(facts: list[Fact]) -> bool:
    return len({f.file for f in _relevant(facts)}) >= 2


def _valued_pairs(facts: list[Fact]) -> list[tuple[Fact, Fact]]:
    valued = [f for f in facts if f.value and f.value.strip()]
    return [(a, b) for i, a in enumerate(valued) for b in valued[i + 1:] if a.file != b.file]


def to_reconciliation(out: ReconcileOut, facts: list[Fact]) -> Reconciliation:
    """Labels -> Reconciliation. Unknown ids, same-file pairs and self-pairs are ignored; a
    pair labelled both agreeing and contradicting counts as contradicting."""
    file_of = {f.id: f.file for f in facts}
    off = {x.id for x in out.facts if x.id in file_of and x.about_topic.strip().lower() == "no"}
    agree: set[frozenset[str]] = set()
    contra: set[frozenset[str]] = set()
    for p in out.pairs:
        if p.a not in file_of or p.b not in file_of or file_of[p.a] == file_of[p.b]:
            continue
        rel = p.relation.strip().lower()
        if rel in _AGREE:
            agree.add(frozenset({p.a, p.b}))
        elif rel == "contradicts":
            contra.add(frozenset({p.a, p.b}))
    return Reconciliation(off_topic=off, agree=agree - contra, contradict=contra)


def _fact_line(f: Fact) -> str:
    return (f"- id={f.id} file={f.file} source_type={f.source_type} applies_to={f.applies_to} "
            f"value={f.value!r} | claim: {f.claim} | quote: {f.quote}")


def _user_prompt(topic_id: str, description: str, facts: list[Fact]) -> str:
    pairs = _valued_pairs(facts)
    lines = [f"KEY: reconcile {topic_id}", f"Topic: {topic_id} — {description}", "", "Facts:"]
    lines += [_fact_line(f) for f in facts]
    lines += ["", "Pairs to label:"]
    lines += [f"- a={a.id} b={b.id}" for a, b in pairs] or ["(none)"]
    return "\n".join(lines)


def reconcile_topic(topic_id: str, facts: list[Fact], llm: LLMClient,
                    cache: JsonCache) -> tuple[Reconciliation | None, str | None]:
    facts = _relevant(facts)
    desc = get_topic(topic_id).description
    key = cache_key("reconcile", PROMPT_VERSION, llm.model, SYSTEM, topic_id, desc,
                    [(f.id, f.file, f.source_type, f.applies_to, f.value, f.claim, f.quote)
                     for f in facts])
    out, err = cached_complete(llm, cache, key, SYSTEM,
                               lambda: _user_prompt(topic_id, desc, facts), ReconcileOut)
    if out is None:
        return None, _short(err)
    return to_reconciliation(out, facts), None


def reconcile_all(facts: list[Fact], llm: LLMClient, cache: JsonCache
                  ) -> tuple[dict[str, Reconciliation], list[tuple[str, str]]]:
    """One call per topic whose facts span >= 2 files. Failures are listed and that topic
    resolves without labels."""
    by_topic: dict[str, list[Fact]] = {}
    for f in facts:
        by_topic.setdefault(f.topic, []).append(f)
    out: dict[str, Reconciliation] = {}
    failed: list[tuple[str, str]] = []
    for t in sorted(by_topic):
        if not needs_reconcile(by_topic[t]):
            continue
        r, err = reconcile_topic(t, by_topic[t], llm, cache)
        if r is None:
            failed.append((f"reconcile:{t}", err or "unknown error"))
        else:
            out[t] = r
    return out, failed
