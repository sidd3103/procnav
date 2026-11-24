from datetime import date

from procnav.blockers import Blocker
from procnav.diff import diff
from procnav.models import Requirement, Verdict
from procnav.pipeline import Snapshot
from procnav.requirements import RequirementResult


def V(topic, status, value=None):
    return Verdict(topic=topic, status=status, value=value, used=[], excluded=[], hints=[],
                   rule="r", caveats=[], sendable=status == "VERIFIED")


def RR(rid, status):
    req = Requirement(id=rid, text="q", requester=None, topics=[], source_file="f.csv",
                      quote=None, draft_answer=None, from_customer=False, group=None)
    return RequirementResult(req=req, status=status, topic_verdicts=[], open_assumptions=[],
                             stale_draft=None, depends_on_decisions=[])


def snap(verdicts, blockers=(), required=(), failed=(), files=None):
    return Snapshot(
        case="C", as_of=date(2026, 9, 25), customer="X", required=list(required),
        verdicts={v.topic: v for v in verdicts},
        facts=[], blockers=[Blocker(item=i, reason="explicit_dependency", chain=[i])
                            for i in blockers],
        next_steps=[], answers=[], failed_files=list(failed), skipped_files=[],
        dropped_quotes=[], files=files or {},
    )


def test_diff_lines():
    old = snap(
        [V("retention_traces", "UNKNOWN"), V("retention_app_logs", "PROPOSED", "14 days"),
         V("soc2", "VERIFIED", "SOC 2")],
        blockers=["subprocessors", "cross_border"],
        required=[RR("Q-004", "unknown"), RR("Q-007", "ready_for_review")],
    )
    new = snap(
        [V("retention_traces", "PROPOSED", "30 days"), V("retention_app_logs", "PROPOSED", "14 days"),
         V("soc2", "VERIFIED", "SOC 2"), V("insurance", "VERIFIED", "CAD 5M")],
        blockers=["cross_border", "insurance"],
        required=[RR("Q-004", "needs_confirmation"), RR("Q-007", "ready_for_review")],
        failed=[("05_evidence/x.txt", "boom")],
    )
    lines = diff(old, new)
    assert "retention_traces: UNKNOWN → PROPOSED 30 days" in lines
    assert "insurance: (none) → VERIFIED CAD 5M" in lines
    assert "+ blocker insurance" in lines
    assert "- blocker subprocessors" in lines
    assert "requirement Q-004: unknown → needs_confirmation" in lines
    assert "+ failed 05_evidence/x.txt: boom" in lines
    assert not any(line.startswith(("retention_app_logs", "soc2", "requirement Q-007"))
                   for line in lines)
    assert not any("cross_border" in line for line in lines)
    assert lines == diff(old, new)  # deterministic


def test_diff_empty_when_equal():
    s = snap([V("soc2", "VERIFIED", "SOC 2")], blockers=["insurance"],
             required=[RR("Q-1", "blocked")], failed=[("a", "b")])
    assert diff(s, s.model_copy(deep=True)) == []


def test_diff_status_only_change():
    lines = diff(snap([V("t", "PROPOSED", "x")]), snap([V("t", "CONFLICT")]))
    assert lines == ["t: PROPOSED x → CONFLICT"]


def test_diff_removed_verdict():
    lines = diff(snap([V("t", "UNKNOWN")]), snap([]))
    assert lines == ["t: UNKNOWN → (none)"]


def test_diff_files_and_cleared_failures():
    old = snap([], files={"a.txt": "1", "b.txt": "2", "c.txt": "3"},
               failed=[("b.txt", "boom"), ("map_questions", "x")])
    new = snap([], files={"a.txt": "1", "b.txt": "9", "d.txt": "4"},
               failed=[("map_questions", "x")])
    lines = diff(old, new)
    assert lines == ["+ file d.txt", "- file c.txt", "~ file b.txt", "- failed b.txt"]
