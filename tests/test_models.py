from datetime import date

from procnav.models import (Dependency, Fact, OpenDecision, Rank, Requirement,
                            Topic, Verdict)


def make_fact(**kw):
    base = dict(
        id="f1", file="a.txt", line=3, topic="encryption", claim="c", value="AES-256",
        applies_to="this_customer", certainty="stated", date="2025-01-02",
        speaker=None, quote="q", valid_until=date(2026, 1, 1), rank=Rank.CUSTOMER,
        source_type="internal_chat", case="case1", customer="Cust",
    )
    base.update(kw)
    return Fact(**base)


def test_fact_roundtrip():
    f = make_fact()
    assert Fact.model_validate_json(f.model_dump_json()) == f


def test_rank_order():
    assert Rank.OTHER_CUSTOMER < Rank.GENERIC < Rank.CUSTOMER < Rank.EVIDENCE
    assert [r.value for r in Rank] == [1, 2, 3, 4]


def test_requirement_extras_default_none():
    r = Requirement(id="r1", text="t", requester=None, topics=["dpa"], source_file="x",
                    quote=None, draft_answer=None, from_customer=False, group=None)
    assert r.section is None and r.note is None


def test_other_models_construct():
    Dependency(id="d", kind="blocked_until", subject="s", depends_on_topics=["dpa"],
               date=None, quote="q", file="f")
    OpenDecision(id="o", text="t", decider="customer", affected_topics=[], quote="q", file="f")
    v = Verdict(topic="dpa", status="UNKNOWN", value=None, used=[],
                excluded=[("f1", "reason")], hints=[], rule="r", caveats=[], sendable=False)
    assert Verdict.model_validate_json(v.model_dump_json()) == v
    Topic(id="x", description="d", scope="per_customer", area="a", keywords=[])
