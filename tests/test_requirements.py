from pathlib import Path

from procnav.loader import load_case, parse_questionnaire
from procnav.models import OpenDecision, Requirement, Verdict
from procnav.requirements import apply_blockers, build_requirements, evaluate, template_flags

ACME = Path(__file__).resolve().parent.parent / "reonsiteinterview" / "CASE_1_Acme_Bank"


def V(topic, status="VERIFIED", value="x", sendable=True, caveats=None):
    return Verdict(topic=topic, status=status, value=value, used=[], excluded=[], hints=[],
                   rule="r", caveats=caveats or [], sendable=sendable)


def R(id, topics, draft=None, from_customer=False, **kw):
    return Requirement(id=id, text="q", requester=None, topics=topics, source_file="f",
                       quote=None, draft_answer=draft, from_customer=from_customer,
                       group=None, **kw)


def _acme_rows():
    files, _ = load_case(ACME)
    for f in files:
        rows = parse_questionnaire(f)
        if rows is not None:
            return rows
    raise AssertionError


def test_acme_rows_become_23_real_plus_5_template_groups():
    rows = _acme_rows()
    out = build_requirements([], rows)
    templ = [r for r in out if r.group and r.group.startswith("template:")]
    assert len(templ) == 5
    assert len(out) - len(templ) == 23
    assert len({r.group for r in templ}) == 5
    assert all(r.group == f"template:{r.section}" for r in templ)


def test_template_representative_unions_section_topics_and_extracted_appended():
    rows = [
        R("a", ["dpa"], section="S"), R("b", ["soc2"], section="S"),
        R("t1", [], section="S"), R("t2", [], section="S"), R("t3", [], section="S"),
    ]
    for r, t in zip(rows[2:], ["Control 1 x", "Control 2 x", "Control 3 x"]):
        r.text = t
    for r in rows[:2]:
        r.text = "unique " + r.id
    ext = R("e1", ["encryption"], from_customer=True)
    out = build_requirements([ext], rows)
    assert [r.id for r in out] == ["a", "b", "t1", "e1"]
    assert out[2].group == "template:S"
    assert out[2].topics == ["dpa", "soc2"]


def test_stale_draft_contradicts():
    r = R("Q-004", ["retention_app_logs"], draft="30 days?")
    [res] = evaluate([r], {"retention_app_logs": V("retention_app_logs", "PROPOSED", "14d", False)}, [])
    assert res.stale_draft and "unsourced draft '30 days?'" in res.stale_draft
    assert "contradicts" in res.stale_draft
    assert res.topic_verdicts == ["retention_app_logs: PROPOSED 14d"]
    assert res.status == "needs_confirmation"


def test_draft_containing_value_does_not_contradict():
    r = R("q", ["t"], draft="14 days")
    [res] = evaluate([r], {"t": V("t", value="14d")}, [])
    assert "contradicts" not in res.stale_draft


def test_no_draft_no_stale():
    [res] = evaluate([R("q", ["t"])], {"t": V("t")}, [])
    assert res.stale_draft is None


def test_all_verified_sendable_is_ready():
    [res] = evaluate([R("q", ["a", "b"])], {"a": V("a"), "b": V("b")}, [])
    assert res.status == "ready_for_review"
    assert res.open_assumptions == []


def test_verified_not_sendable_needs_confirmation():
    [res] = evaluate([R("q", ["a"])], {"a": V("a", sendable=False)}, [])
    assert res.status == "needs_confirmation"


def test_unknown_conflict_and_missing_are_unknown():
    [res] = evaluate([R("q", ["a", "b", "c"])],
                     {"a": V("a"), "b": V("b", "UNKNOWN", None, False)}, [])
    assert res.status == "unknown"
    assert "b: unknown" in res.open_assumptions and "c: unknown" in res.open_assumptions
    [res] = evaluate([R("q", ["a"])], {"a": V("a", "CONFLICT", "1", False)}, [])
    assert res.status == "unknown" and res.open_assumptions == ["a: conflict"]


def test_no_topics_is_unknown():
    [res] = evaluate([R("q", [])], {}, [])
    assert res.status == "unknown"


def test_depends_on_decisions():
    d = OpenDecision(id="D1", text="t", decider="internal", affected_topics=["a"], quote="q", file="f")
    d2 = OpenDecision(id="D2", text="t", decider="internal", affected_topics=["z"], quote="q", file="f")
    [res] = evaluate([R("q", ["a"])], {"a": V("a")}, [d, d2])
    assert res.depends_on_decisions == ["D1"]


def test_apply_blockers_sets_blocked():
    from procnav.blockers import Blocker
    res = evaluate([R("q", ["a"]), R("p", ["b"])], {"a": V("a"), "b": V("b")}, [])
    b = Blocker(item="a", reason="asked_hard_gap", chain=["a unknown"], quote=None)
    out = apply_blockers(res, [b])
    assert [r.status for r in out] == ["blocked", "ready_for_review"]
    assert res[0].status == "ready_for_review"  # pure


def test_template_flags_acme():
    rows = _acme_rows()
    flags = template_flags(rows)
    assert len(flags) == len(rows)
    real = [r.id for r, t in zip(rows, flags) if not t]
    assert real == [f"Q-{i:03d}" for i in range(1, 24)]
    assert sum(flags) == 37


def test_stale_draft_lists_every_contradiction():
    r = R("Q-004", ["retention_app_logs", "retention_traces", "retention_source_files"],
          draft="30 days?")
    vs = {"retention_app_logs": V("retention_app_logs", "PROPOSED", "14 days", False),
          "retention_traces": V("retention_traces", "UNKNOWN", None, False),
          "retention_source_files": V("retention_source_files", "PROPOSED", "workspace lifetime",
                                      False)}
    [res] = evaluate([r], vs, [])
    assert res.stale_draft == ("unsourced draft '30 days?'; contradicts "
                               "retention_app_logs='14 days', "
                               "retention_source_files='workspace lifetime'")


def test_readiness_separates_answerable_from_missing():
    from procnav.requirements import readiness
    vs = {"soc2": V("soc2", caveats=["no evidence: 2026 not final"], sendable=False),
          "cur": V("cur", "UNKNOWN", None, False), "st": V("st", "PROPOSED", sendable=False),
          "fb": V("fb", "CONFLICT", None, False),
          "dpa": V("dpa", "PROPOSED", sendable=False).model_copy(
              update={"rule": "derived from dependency"})}
    assert readiness([], vs) == "missing"
    assert readiness(["soc2"], vs) == "ready"           # evidence with a caveat is answerable
    assert readiness(["dpa"], vs) == "ready"            # derived by code from the case
    assert readiness(["dpa", "cur"], vs) == "ready"     # code writes the whole answer
    assert readiness(["soc2", "cur"], vs) == "confirm"  # partly known
    assert readiness(["st"], vs) == "confirm"
    assert readiness(["fb"], vs) == "confirm"           # someone must settle the conflict
    assert readiness(["cur", "nope"], vs) == "missing"


def _dec(topics, text="Is the fallback disabled?", decider="internal"):
    return OpenDecision(id="o1", text=text, decider=decider, affected_topics=topics,
                        quote="q", file="chat.txt")


def test_open_decision_adds_caveat_and_blocks_sendable():
    from procnav.requirements import apply_decision_caveats
    vs = {"mp": V("mp"), "un": V("un", "UNKNOWN", None, False),
          "dpa": V("dpa", "PROPOSED", sendable=False).model_copy(
              update={"rule": "derived from dependency"})}
    out = apply_decision_caveats(vs, [_dec(["mp", "un", "dpa", "missing"])])
    assert out["mp"].caveats == ["open decision (internal): Is the fallback disabled?"]
    assert out["mp"].status == "VERIFIED" and not out["mp"].sendable
    assert out["un"] == vs["un"] and out["dpa"] == vs["dpa"]  # nothing to caveat
    assert "missing" not in out
    assert apply_decision_caveats(vs, []) == vs


def test_readiness_open_decision_needs_confirmation():
    from procnav.requirements import apply_decision_caveats, readiness
    vs = apply_decision_caveats({"mp": V("mp")}, [_dec(["mp"])])
    assert readiness(["mp"], vs) == "confirm"
    expiring = {"ins": V("ins", caveats=["expires 2026-10-01"], sendable=False)}
    assert readiness(["ins"], expiring) == "ready"  # an expiry is part of the answer
