from procnav.answers import AnswerOut, DraftAnswer, draft_answer
from procnav.cache import JsonCache
from procnav.llm import FakeLLM
from procnav.models import Fact, Rank, Requirement, Verdict
from procnav.requirements import RequirementResult


def F(id, value="ca-central-1", quote="we use ca-central-1", topic="storage"):
    return Fact(id=id, file="f.md", line=1, topic=topic, claim="c", value=value,
                applies_to="this_customer", certainty="stated", date=None, speaker=None,
                quote=quote, valid_until=None, rank=Rank.CUSTOMER, source_type="internal_chat",
                case="c", customer="Acme")


def V(topic, status="VERIFIED", value="x", used=None, caveats=None):
    return Verdict(topic=topic, status=status, value=value, used=used or [], excluded=[],
                   hints=[], rule="r", caveats=caveats or [], sendable=True)


def RR(topics, status="needs_confirmation", open_a=None):
    req = Requirement(id="Q-1", text="Where is data stored?", requester=None, topics=topics,
                      source_file="f", quote=None, draft_answer=None, from_customer=False,
                      group=None)
    return RequirementResult(req=req, status=status, topic_verdicts=[],
                             open_assumptions=open_a or [], stale_draft=None,
                             depends_on_decisions=[])


def setup(tmp_path):
    facts = {"f1": F("f1"), "f2": F("f2", value="us-east-1", quote="maybe us-east-1")}
    verdicts = {"storage": V("storage", "VERIFIED", "ca-central-1", ["f1"])}
    return facts, verdicts, JsonCache(tmp_path)


def llm_for(out):
    return FakeLLM({"answer:Q-1": out})


def test_valid_citation_passes(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="Stored in ca-central-1.", cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert isinstance(d, DraftAnswer)
    assert d.cited == ["f1"] and d.evidence_quotes == ["we use ca-central-1"]
    assert d.citation_errors == [] and d.owner == "Bob" and d.req_id == "Q-1"


def test_invented_id_flagged(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="x", cited_fact_ids=["f1", "f2", "zzz"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert d.cited == ["f1"]
    assert "unknown fact id: f2" in d.citation_errors  # exists but not allowed
    assert "unknown fact id: zzz" in d.citation_errors


def test_uncited_flagged(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="x", cited_fact_ids=[]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert d.citation_errors == ["uncited"]


def test_all_unknown_skips_llm(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({})
    rr = RR(["a", "b"], "unknown", ["a: unknown"])
    d = draft_answer(rr, {"a": V("a", "UNKNOWN", None)}, facts, "Bob", llm, cache)
    assert llm.calls == 0
    assert d.answer == "Not yet known. Owner: Bob. Open: a: unknown"
    assert d.status == "unknown" and d.citation_errors == [] and d.cited == []


def test_no_topics_and_no_open(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({})
    d = draft_answer(RR([], "unknown"), verdicts, facts, "Bob", llm, cache)
    assert llm.calls == 0
    assert d.answer == "Not yet known. Owner: Bob. Open: no information in case files"


def test_status_from_result_not_llm(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="This is approved and ready.", cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage"], "blocked"), verdicts, facts, "Bob", llm, cache)
    assert d.status == "blocked"


def test_cached_on_rerun(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="ok", cited_fact_ids=["f1"]))
    d1 = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    d2 = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert llm.calls == 1 and d1 == d2


def test_evidence_change_invalidates_cache(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({"answer:Q-1": [AnswerOut(answer="a", cited_fact_ids=["f1"]),
                                  AnswerOut(answer="b", cited_fact_ids=["f1"])]})
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    facts2 = {**facts, "f1": F("f1", quote="we now use somewhere else")}
    d = draft_answer(RR(["storage"]), verdicts, facts2, "Bob", llm, cache)
    assert llm.calls == 2 and d.answer == "b"


def test_conflict_not_allowed_but_open_assumption(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    verdicts["loc"] = V("loc", "CONFLICT", None, ["f2"])
    seen = {}

    def fn(system, user, schema):
        seen["user"] = user
        return AnswerOut(answer="x", cited_fact_ids=["f1", "f2"])

    d = draft_answer(RR(["storage", "loc"]), verdicts, facts, "Bob", FakeLLM(fn), cache)
    assert d.cited == ["f1"] and "unknown fact id: f2" in d.citation_errors
    assert any("loc" in a and "us-east-1" in a for a in d.open_assumptions)
    assert seen["user"].startswith("KEY: answer:Q-1")
    assert "maybe us-east-1" not in seen["user"]  # conflict quote not offered as evidence


def test_proposed_allowed(tmp_path):
    facts, _, cache = setup(tmp_path)
    verdicts = {"storage": V("storage", "PROPOSED", "ca-central-1", ["f1"])}
    llm = llm_for(AnswerOut(answer="proposed", cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert d.cited == ["f1"]


def test_llm_failure_not_cached(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    bad = FakeLLM({"answer:Q-1": [RuntimeError("boom"), RuntimeError("boom")]})
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", bad, cache)
    assert d.answer == "Draft unavailable (LLM error)"
    assert d.citation_errors and d.citation_errors[0].startswith("llm_failed: ")
    assert bad.calls == 2
    good = llm_for(AnswerOut(answer="ok", cited_fact_ids=["f1"]))
    d2 = draft_answer(RR(["storage"]), verdicts, facts, "Bob", good, cache)
    assert good.calls == 1 and d2.answer == "ok"


def test_only_invalid_citations_also_uncited(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="x", cited_fact_ids=["zzz"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert d.cited == []
    assert "unknown fact id: zzz" in d.citation_errors and "uncited" in d.citation_errors


def test_caveat_change_invalidates_cache(tmp_path):
    facts, _, cache = setup(tmp_path)
    llm = FakeLLM({"answer:Q-1": [AnswerOut(answer="a", cited_fact_ids=["f1"]),
                                  AnswerOut(answer="b", cited_fact_ids=["f1"])]})
    v1 = {"storage": V("storage", "PROPOSED", "ca-central-1", ["f1"], ["draft only"])}
    v2 = {"storage": V("storage", "PROPOSED", "ca-central-1", ["f1"], ["pending legal"])}
    draft_answer(RR(["storage"]), v1, facts, "Bob", llm, cache)
    d = draft_answer(RR(["storage"]), v2, facts, "Bob", llm, cache)
    assert llm.calls == 2 and d.answer == "b\nCaveats: pending legal"  # R30 caveat line


def test_prompt_has_proposed_status_and_caveat(tmp_path):
    facts, _, cache = setup(tmp_path)
    verdicts = {"storage": V("storage", "PROPOSED", "ca-central-1", ["f1"], ["draft only"])}
    seen = {}

    def fn(system, user, schema):
        seen["user"] = user
        return AnswerOut(answer="x", cited_fact_ids=["f1"])

    draft_answer(RR(["storage"]), verdicts, facts, "Bob", FakeLLM(fn), cache)
    assert "PROPOSED" in seen["user"] and "draft only" in seen["user"]


def test_answer_system_prompt_is_part_of_cache_key(tmp_path, monkeypatch):
    import procnav.answers as an
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({"answer:Q-1": [AnswerOut(answer="a", cited_fact_ids=["f1"])] * 2})
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert llm.calls == 1
    monkeypatch.setattr(an, "ANSWER_SYSTEM", an.ANSWER_SYSTEM + "\nchanged")
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache)
    assert llm.calls == 2


def test_answer_out_defaults():
    assert AnswerOut.model_validate({"answer": "a"}).cited_fact_ids == []


# ---- R30: as_of in prompt and cache key; caveats appended in code

from datetime import date  # noqa: E402

AS_OF = date(2026, 11, 24)


def test_expired_caveat_line_appended_even_if_llm_omits_it(tmp_path):
    facts, _, cache = setup(tmp_path)
    verdicts = {"storage": V("storage", "VERIFIED", "CAD 5M", ["f1"],
                             ["expired 2026-10-01", "a earlier caveat"])}
    llm = llm_for(AnswerOut(answer="Coverage is current and expires on October 1, 2026.",
                            cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache, as_of=AS_OF)
    assert d.answer.startswith("Coverage is current")
    assert d.answer.endswith("\nCaveats: a earlier caveat; expired 2026-10-01")


def test_caveats_line_collects_all_topics_sorted_unique(tmp_path):
    facts, _, cache = setup(tmp_path)
    verdicts = {"storage": V("storage", "VERIFIED", "x", ["f1"], ["z last", "expired 2026-10-01"]),
                "loc": V("loc", "PROPOSED", "y", ["f1"], ["expired 2026-10-01"])}
    llm = llm_for(AnswerOut(answer="text", cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage", "loc"]), verdicts, facts, "Bob", llm, cache, as_of=AS_OF)
    assert d.answer == "text\nCaveats: expired 2026-10-01; z last"


def test_no_caveats_no_line(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = llm_for(AnswerOut(answer="text", cited_fact_ids=["f1"]))
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache, as_of=AS_OF)
    assert d.answer == "text"


def test_all_unknown_answer_gets_caveats(tmp_path):
    facts, _, cache = setup(tmp_path)
    verdicts = {"a": V("a", "UNKNOWN", None, caveats=["no evidence: audit not final"])}
    d = draft_answer(RR(["a"], "unknown", ["a: unknown"]), verdicts, facts, "Bob", FakeLLM({}),
                     cache, as_of=AS_OF)
    assert d.answer == ("Not yet known. Owner: Bob. Open: a: unknown\n"
                        "Caveats: no evidence: audit not final")


def test_as_of_in_prompt_and_binding_caveat_rule(tmp_path):
    from procnav.answers import ANSWER_SYSTEM
    facts, verdicts, cache = setup(tmp_path)
    seen = {}

    def fn(system, user, schema):
        seen["user"] = user
        return AnswerOut(answer="x", cited_fact_ids=["f1"])

    draft_answer(RR(["storage"]), verdicts, facts, "Bob", FakeLLM(fn), cache, as_of=AS_OF)
    assert "Today (as-of) is 2026-11-24" in seen["user"]
    assert ("Caveats are binding: an expired item must be described as expired, never as "
            "current.") in ANSWER_SYSTEM


def test_as_of_change_is_cache_miss(tmp_path):
    facts, verdicts, cache = setup(tmp_path)
    llm = FakeLLM({"answer:Q-1": [AnswerOut(answer="a", cited_fact_ids=["f1"]),
                                  AnswerOut(answer="b", cited_fact_ids=["f1"])]})
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache, as_of=AS_OF)
    draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache, as_of=AS_OF)
    assert llm.calls == 1
    d = draft_answer(RR(["storage"]), verdicts, facts, "Bob", llm, cache,
                     as_of=date(2026, 12, 1))
    assert llm.calls == 2 and d.answer == "b"
