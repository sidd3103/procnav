from datetime import date

from procnav.blockers import find_blockers
from procnav.models import Dependency, Requirement, Verdict
from procnav.requirements import evaluate

AS_OF = date(2026, 5, 1)


def V(topic, status="VERIFIED", value="x", caveats=None):
    return Verdict(topic=topic, status=status, value=value, used=[], excluded=[], hints=[],
                   rule="r", caveats=caveats or [], sendable=status == "VERIFIED")


def dep(id, kind, subject, topics=(), d=None):
    return Dependency(id=id, kind=kind, subject=subject, depends_on_topics=list(topics),
                      date=d, quote="q " + id, file="f")


def R(id, topics, from_customer=False):
    return Requirement(id=id, text="q", requester=None, topics=topics, source_file="f",
                       quote="asked", draft_answer=None, from_customer=from_customer, group=None)


def run(verdicts, reqs, deps):
    return find_blockers(verdicts, evaluate(reqs, verdicts, []), deps, AS_OF)


def test_dpa_blocked_until_two_unknown_topics():
    deps = [dep("d1", "blocked_until", "DPA", ["subprocessors", "cross_border"]),
            dep("m1", "milestone", "October pilot", d=date(2026, 10, 1))]
    bs = run({"subprocessors": V("subprocessors", "UNKNOWN", None)}, [], deps)
    assert [b.item for b in bs] == ["subprocessors", "cross_border"]
    assert all(b.reason == "explicit_dependency" for b in bs)
    assert bs[0].chain == ["subprocessors UNKNOWN", "DPA", "October pilot"]
    assert bs[1].chain[0] == "cross_border UNKNOWN"
    assert bs[0].quote == "q d1"


def test_verified_dep_topic_not_blocker():
    deps = [dep("d1", "blocked_until", "DPA", ["a"])]
    assert run({"a": V("a")}, [], deps) == []


def test_chain_without_milestone_and_earliest_milestone():
    deps = [dep("d1", "blocked_until", "DPA", ["a"])]
    assert run({}, [], deps)[0].chain == ["a UNKNOWN", "DPA"]
    deps += [dep("m1", "milestone", "Late", d=date(2026, 12, 1)),
             dep("m2", "milestone", "Early", d=date(2026, 9, 1))]
    assert run({}, [], deps)[0].chain[-1] == "Early"
    undated = [dep("d1", "blocked_until", "DPA", ["a"]), dep("m1", "milestone", "First"),
               dep("m2", "milestone", "Second")]
    assert run({}, [], undated)[0].chain[-1] == "First"


def test_insurance_expiry_caveat_with_undated_milestone():
    v = {"insurance": V("insurance", caveats=["expires 2026-09-30"])}
    deps = [dep("m1", "milestone", "October pilot")]
    bs = run(v, [], deps)
    assert len(bs) == 1
    assert bs[0].item == "insurance" and bs[0].reason == "caveat_before_milestone"
    assert bs[0].chain[-1] == "October pilot"


def test_caveat_dated_milestone_before_expiry_is_no_blocker():
    v = {"insurance": V("insurance", caveats=["expires 2026-09-30"])}
    assert run(v, [], [dep("m1", "milestone", "Early", d=date(2026, 8, 1))]) == []
    assert len(run(v, [], [dep("m1", "milestone", "Late", d=date(2026, 10, 1))])) == 1


def test_caveat_without_milestone_no_blocker():
    assert run({"i": V("i", caveats=["expired 2025-01-01"])}, [], []) == []


def test_proposed_asked_by_customer_not_blocker():
    v = {"a": V("a", "PROPOSED", "v")}
    assert run(v, [R("q", ["a"], True)], []) == []


def test_unknown_and_conflict_asked_by_customer_are_blockers():
    v = {"a": V("a", "UNKNOWN", None), "b": V("b", "CONFLICT", "1")}
    bs = run(v, [R("q", ["a", "b", "c"], True), R("n", ["z"], False)], [])
    assert [(b.item, b.reason) for b in bs] == [("a", "asked_hard_gap"), ("b", "asked_hard_gap"),
                                                ("c", "asked_hard_gap")]
    assert bs[0].chain == ["a UNKNOWN"] and bs[0].quote == "asked"


def test_dedup_by_item_and_reason():
    deps = [dep("d1", "blocked_until", "DPA", ["a"]), dep("d2", "blocked_until", "DPA2", ["a"])]
    bs = run({}, [R("q", ["a"], True), R("q2", ["a"], True)], deps)
    assert [(b.item, b.reason) for b in bs] == [("a", "explicit_dependency"),
                                                ("a", "asked_hard_gap")]


def test_unreadable_expiry_still_blocks_dated_milestone():
    v = {"insurance": V("insurance", "VERIFIED", "x",
                        caveats=["expires on unreadable date 'next renewal'"])}
    bs = run(v, [], [dep("m1", "milestone", "Pilot", d=date(2026, 6, 1))])
    assert [(b.item, b.reason) for b in bs] == [("insurance", "caveat_before_milestone")]


def test_valueless_fact_on_asked_topic_still_blocks():
    from procnav.models import Fact, Rank
    from procnav.resolve import resolve_topic
    f = Fact(id="nv", file="chat.md", line=1, topic="retention_traces", claim="unclear",
             value=None, applies_to="this_customer", certainty="uncertain", date=None,
             speaker=None, quote="q", valid_until=None, rank=Rank.CUSTOMER,
             source_type="internal_chat", case="c", customer="A")
    verdicts = {"retention_traces": resolve_topic("retention_traces", [f], AS_OF)}
    assert verdicts["retention_traces"].status == "UNKNOWN"
    bs = run(verdicts, [R("e1", ["retention_traces"], from_customer=True)], [])
    assert [(b.item, b.reason) for b in bs] == [("retention_traces", "asked_hard_gap")]


def test_unreadable_expiry_with_embedded_date_is_still_undated():
    v = {"insurance": V("insurance", "VERIFIED", "x",
                        caveats=["expires on unreadable date 'after 2027-01-01 maybe'"])}
    bs = run(v, [], [dep("m1", "milestone", "Pilot", d=date(2026, 6, 1))])
    assert [(b.item, b.reason) for b in bs] == [("insurance", "caveat_before_milestone")]


def test_dependency_subject_requirement_is_blocked_too():
    from procnav.requirements import apply_blockers
    deps = [dep("d1", "blocked_until", "DPA", ["subprocessors"])]
    verdicts = {"subprocessors": V("subprocessors", "UNKNOWN", None), "dpa": V("dpa", "PROPOSED")}
    reqs = [R("Q-020", ["dpa"]), R("Q-x", ["insurance"])]
    bs = run(verdicts, reqs, deps)
    assert bs[0].subject == "dpa"
    out = apply_blockers(evaluate(reqs, verdicts, []), bs)
    assert [r.status for r in out] == ["blocked", "unknown"]


def test_dependency_subject_not_blocked_once_dependencies_verified():
    deps = [dep("d1", "blocked_until", "DPA", ["subprocessors"])]
    assert run({"subprocessors": V("subprocessors")}, [R("Q-020", ["dpa"])], deps) == []


def test_derived_dependency_value_lists_only_open_topics():
    from procnav.blockers import derive_from_dependencies
    deps = [dep("d1", "blocked_until", "DPA", ["subprocessors", "cross_border"])]
    vs = {"dpa": V("dpa", "UNKNOWN", None), "subprocessors": V("subprocessors"),
          "cross_border": V("cross_border", "UNKNOWN", None)}
    out = derive_from_dependencies(vs, deps)
    assert out["dpa"].value == "blocked until cross_border is confirmed"
    vs["cross_border"] = V("cross_border")
    out = derive_from_dependencies(vs, deps)
    assert out["dpa"].status == "PROPOSED"
    assert out["dpa"].value == "no longer blocked: subprocessors, cross_border are confirmed"
