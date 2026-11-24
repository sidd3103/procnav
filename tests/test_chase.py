from datetime import date

from conftest import ACME_DIR

from procnav.cache import JsonCache
from procnav.chase import ChaseMessage, ChaseOut, draft_chase
from procnav.llm import FakeLLM
from procnav.models import Fact, Rank, Verdict
from procnav.pipeline import NextStep, run_case


def F(id, value="ca-central-1", quote="we use ca-central-1", topic="storage"):
    return Fact(id=id, file="f.md", line=1, topic=topic, claim="c", value=value,
                applies_to="this_customer", certainty="stated", date=None, speaker=None,
                quote=quote, valid_until=None, rank=Rank.CUSTOMER, source_type="internal_chat",
                case="c", customer="Acme")


def V(topic, status="VERIFIED", value="x", used=None, hints=None):
    return Verdict(topic=topic, status=status, value=value, used=used or [], excluded=[],
                   hints=hints or [], rule="r", caveats=[], sendable=True)


def NS(kind="internal", topics=("storage", "inference")):
    return NextStep(owner="Bob", kind=kind, topics=list(topics),
                    items=["storage: PROPOSED ca-central-1", "inference: UNKNOWN"])


def setup(tmp_path):
    facts = {"f1": F("f1"), "f2": F("f2", value="us-east-1", quote="other customer used us-east-1")}
    verdicts = {
        "storage": V("storage", "PROPOSED", "ca-central-1", ["f1"]),
        "inference": V("inference", "UNKNOWN", None, hints=["other customer: us-east-1 (f2)"]),
    }
    return facts, verdicts, JsonCache(tmp_path)


def llm_for(out):
    return FakeLLM({"chase:internal:Bob": out, "chase:customer_decision:Bob": out})


def test_valid_citation_passes(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(ChaseOut(message="Please confirm inference location.", cited_fact_ids=["f1"]))
    c = draft_chase(NS(), verdicts, facts, llm, cache)
    assert isinstance(c, ChaseMessage)
    assert c.cited == ["f1"] and c.evidence_quotes == ["we use ca-central-1"]
    assert c.citation_errors == [] and c.owner == "Bob" and c.kind == "internal"


def test_invented_id_flagged(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(ChaseOut(message="x", cited_fact_ids=["f1", "zzz"]))
    c = draft_chase(NS(), verdicts, facts, llm, cache)
    assert c.cited == ["f1"] and "unknown fact id: zzz" in c.citation_errors


def test_uncited_flagged(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    c = draft_chase(NS(), verdicts, facts, llm_for(ChaseOut(message="x", cited_fact_ids=[])), cache)
    assert c.citation_errors == ["uncited"]


def test_hint_value_not_citable(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(ChaseOut(message="x", cited_fact_ids=["f2"]))
    c = draft_chase(NS(), verdicts, facts, llm, cache)
    assert c.cited == [] and "unknown fact id: f2" in c.citation_errors


def test_llm_failure_not_raised_not_cached(tmp_path):
    facts, verdicts, cache = setup(tmp_path)

    def boom(system, user, schema):
        raise RuntimeError("down")

    c = draft_chase(NS(), verdicts, facts, FakeLLM(boom), cache)
    assert c.message == "Draft unavailable (LLM error)" and c.cited == []
    assert c.citation_errors and c.citation_errors[0].startswith("llm_failed")
    ok = llm_for(ChaseOut(message="fine", cited_fact_ids=["f1"]))
    c2 = draft_chase(NS(), verdicts, facts, ok, cache)
    assert ok.calls == 1 and c2.message == "fine"


def test_cache_hit_on_rerun(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(ChaseOut(message="m", cited_fact_ids=["f1"]))
    a = draft_chase(NS(), verdicts, facts, llm, cache)
    b = draft_chase(NS(), verdicts, facts, llm, cache)
    assert llm.calls == 1 and a == b
    draft_chase(NS(kind="customer_decision"), verdicts, facts, llm, cache)
    assert llm.calls == 2  # kind is part of the prompt, so part of the key


def test_prompt_contains_items(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    seen = []

    def spy(system, user, schema):
        seen.append(user)
        return schema.model_validate({"message": "m", "cited_fact_ids": ["f1"]})

    draft_chase(NS(), verdicts, facts, FakeLLM(spy), cache)
    u = seen[0]
    assert u.startswith("KEY: chase:internal:Bob\n")
    assert "inference: UNKNOWN" in u and "storage: PROPOSED ca-central-1" in u
    assert "id=f1" in u and "id=f2" not in u
    assert "starting points (not evidence)" in u.lower() and "us-east-1" in u

    seen.clear()
    draft_chase(NS(kind="customer_decision"), verdicts, facts, FakeLLM(spy), cache)
    assert "decide" in seen[0].lower()


def test_acme_one_chase_per_next_step(tmp_path, acme_llm):
    snap = run_case(ACME_DIR, date(2026, 9, 25), acme_llm(), tmp_path / "c")
    assert snap.next_steps and len(snap.chases) == len(snap.next_steps)
    assert [c.owner for c in snap.chases] == sorted(c.owner for c in snap.chases)
    assert {(c.owner, c.kind) for c in snap.chases} == {(s.owner, s.kind) for s in snap.next_steps}
    assert all(s.topics for s in snap.next_steps)
    assert all(c.message and not c.citation_errors for c in snap.chases)


def test_customer_decision_prompt_has_no_hints_and_customer_rule(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    seen = []

    def spy(system, user, schema):
        seen.append(user)
        return schema.model_validate({"message": "m", "cited_fact_ids": ["f1"]})

    draft_chase(NS(kind="customer_decision"), verdicts, facts, FakeLLM(spy), cache)
    u = seen[0]
    assert u.startswith("KEY: chase:customer_decision:Bob\n")
    assert "us-east-1" not in u and "other customer:" not in u
    assert ("This goes to the customer: do not mention other customers, internal file names "
            "or internal people.") in u


def test_chase_system_prompt_is_part_of_cache_key(tmp_path, monkeypatch):
    import procnav.chase as ch
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(ChaseOut(message="m", cited_fact_ids=["f1"]))
    draft_chase(NS(), verdicts, facts, llm, cache)
    monkeypatch.setattr(ch, "CHASE_SYSTEM", ch.CHASE_SYSTEM + "\nchanged")
    draft_chase(NS(), verdicts, facts, llm, cache)
    assert llm.calls == 2


def test_chase_out_defaults():
    assert ChaseOut.model_validate({"message": 5}).model_dump() == {"message": "5",
                                                                     "cited_fact_ids": []}


# ---- R30: as_of in prompt and cache key; caveats appended in code

AS_OF = date(2026, 11, 24)


def test_chase_caveats_line_and_as_of(tmp_path):
    from procnav.chase import CHASE_SYSTEM
    facts, verdicts, cache = setup(tmp_path)
    verdicts["storage"] = verdicts["storage"].model_copy(
        update={"caveats": ["expired 2026-10-01"]})
    seen = []

    def spy(system, user, schema):
        seen.append(user)
        return ChaseOut(message="Coverage is current.", cited_fact_ids=["f1"])

    c = draft_chase(NS(), verdicts, facts, FakeLLM(spy), cache, as_of=AS_OF)
    assert c.message == "Coverage is current.\nCaveats: expired 2026-10-01"
    assert "Today (as-of) is 2026-11-24" in seen[0]
    assert ("Caveats are binding: an expired item must be described as expired, never as "
            "current.") in CHASE_SYSTEM


def test_chase_as_of_change_is_cache_miss(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({"chase:internal:Bob": [ChaseOut(message="a", cited_fact_ids=["f1"]),
                                          ChaseOut(message="b", cited_fact_ids=["f1"])]})
    draft_chase(NS(), verdicts, facts, llm, cache, as_of=AS_OF)
    draft_chase(NS(), verdicts, facts, llm, cache, as_of=AS_OF)
    assert llm.calls == 1
    c = draft_chase(NS(), verdicts, facts, llm, cache, as_of=date(2026, 12, 1))
    assert llm.calls == 2 and c.message == "b"


def test_pipeline_passes_as_of_into_drafts(tmp_path, acme_llm):
    seen = []
    inner = acme_llm()

    class Spy:
        model = inner.model

        def complete_json(self, system, user, schema):
            if user.startswith(("KEY: answer:", "KEY: chase:")):
                seen.append(user)
            return inner.complete_json(system, user, schema)

    snap = run_case(ACME_DIR, AS_OF, Spy(), tmp_path / "c")
    assert seen and all("Today (as-of) is 2026-11-24" in u for u in seen)
    ins = [a for a in snap.answers if "expired 2026-10-01" in a.answer]
    assert ins and all("\nCaveats: " in a.answer for a in ins)


def test_customer_decision_chase_gets_no_caveats_line(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    verdicts["storage"] = verdicts["storage"].model_copy(
        update={"caveats": ["no evidence: This file contains no remediation evidence."]})
    llm = llm_for(ChaseOut(message="Please decide.", cited_fact_ids=["f1"]))
    c = draft_chase(NS(kind="customer_decision"), verdicts, facts, llm, cache, as_of=AS_OF)
    assert c.message == "Please decide." and "Caveats:" not in c.message
    internal = draft_chase(NS(), verdicts, facts, llm, cache, as_of=AS_OF)
    assert internal.message.endswith("\nCaveats: no evidence: This file contains no remediation "
                                     "evidence.")
