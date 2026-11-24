"""Topic mapping: `other:*` ids produced by extraction are mapped onto seed topics (or kept)."""
import re
from datetime import date

import pytest

from procnav.cache import JsonCache
from procnav.extract import map_new_topics
from procnav.llm import FakeLLM
from procnav.pipeline import run_case

AS_OF = date(2026, 11, 24)

CHAT = "Dev: External actions disabled for the pilot.\nOps: retention by data type is 30 days.\n"
REQUEST = "Customer: Please confirm which tool actions the assistant can take.\n"


def _fact(topic, claim, value, quote):
    return {"topic": topic, "claim": claim, "value": value, "applies_to": "this_customer",
            "certainty": "stated", "date": None, "speaker": "Dev", "quote": quote,
            "valid_until": None}


EXTRACTIONS = {
    "chat.txt": {
        "source_type": "internal_chat", "customer_scope": "this_customer",
        "facts": [
            _fact("external_actions", "External actions are disabled for the pilot.",
                  "disabled", "External actions disabled for the pilot."),
            _fact("retention_by_data_type", "Retention by data type is 30 days.", "30 days",
                  "retention by data type is 30 days."),
        ],
    },
    "request.txt": {
        "source_type": "customer_request", "customer_scope": "this_customer",
        "requirements": [{"text": "Confirm which tool actions the assistant can take.",
                          "requester": "Customer", "topics": ["tool_actions"],
                          "quote": "Please confirm which tool actions the assistant can take."},
                         {"text": "Confirm retention by data type.", "requester": "Customer",
                          "topics": ["retention_by_data_type"],
                          "quote": "Please confirm which tool actions"}],
    },
}


def responder(mapping=None, fail=False, seen=None):
    """mapping: {other_id: [answers]}; unmapped ids answer "new"."""
    mapping = mapping or {}

    def respond(system, user, schema):
        key = user.split("\n", 1)[0].removeprefix("KEY:").strip()
        if seen is not None:
            seen.setdefault(key, []).append(user)
        if key == "identify_case":
            return schema.model_validate({"customer": "Cust Co", "supplier": "Sup Co"})
        if key == "map_topics":
            if fail:
                raise RuntimeError("simulated map failure")
            ids = re.findall(r"^- (other:\S+?):", user, flags=re.MULTILINE)
            return schema.model_validate({"mappings": [
                {"topic_id": i, "map_to": mapping.get(i, ["new"])} for i in ids]})
        if key.startswith("answer:"):
            return schema.model_validate({"answer": "Draft.", "cited_fact_ids": []})
        if key.startswith("chase:"):
            return schema.model_validate({"message": "Please review.", "cited_fact_ids": []})
        return schema.model_validate(EXTRACTIONS[key])

    return respond


@pytest.fixture
def case(tmp_path):
    d = tmp_path / "CASE_X"
    d.mkdir()
    (d / "chat.txt").write_text(CHAT, encoding="utf-8")
    (d / "request.txt").write_text(REQUEST, encoding="utf-8")
    return d


def test_without_mapping_topic_stays_other_and_seed_is_unknown(case, tmp_path):
    snap = run_case(case, AS_OF, FakeLLM(responder()), tmp_path / "c")
    assert snap.verdicts["tool_actions"].status == "UNKNOWN"
    assert snap.verdicts["other:external_actions"].status == "PROPOSED"
    assert snap.new_topics == ["other:external_actions", "other:retention_by_data_type"]


def test_mapping_rewrites_fact_topic_and_verdict_becomes_proposed(case, tmp_path):
    llm = FakeLLM(responder({"other:external_actions": ["tool_actions"]}))
    snap = run_case(case, AS_OF, llm, tmp_path / "c")
    v = snap.verdicts["tool_actions"]
    assert v.status == "PROPOSED" and v.value == "disabled"
    assert "other:external_actions" not in snap.verdicts
    assert snap.new_topics == ["other:retention_by_data_type"]
    assert not any(b.item == "tool_actions" for b in snap.blockers)


def test_requirement_mapped_to_several_seed_topics(case, tmp_path):
    llm = FakeLLM(responder({"other:retention_by_data_type":
                             ["retention_app_logs", "retention_traces"]}))
    snap = run_case(case, AS_OF, llm, tmp_path / "c")
    req = next(r.req for r in snap.required if r.req.text.startswith("Confirm retention"))
    assert req.topics == ["retention_app_logs", "retention_traces"]
    # a fact is one value for one topic: a multi-topic mapping does not copy it everywhere
    fact = next(f for f in snap.facts if "data type" in f.claim)
    assert fact.topic == "other:retention_by_data_type"


def test_invalid_mapping_answer_is_ignored(case, tmp_path):
    llm = FakeLLM(responder({"other:external_actions": ["made_up_topic"]}))
    snap = run_case(case, AS_OF, llm, tmp_path / "c")
    assert snap.verdicts["tool_actions"].status == "UNKNOWN"
    assert "other:external_actions" in snap.new_topics


def test_mapping_failure_recorded_and_run_completes(case, tmp_path):
    snap = run_case(case, AS_OF, FakeLLM(responder(fail=True)), tmp_path / "c")
    assert [p for p, _ in snap.failed_files] == ["map_topics"]
    assert "simulated map failure" in snap.failed_files[0][1]
    assert "other:external_actions" in snap.verdicts
    assert snap.answers


def test_mapping_is_cached(case, tmp_path):
    mapping = {"other:external_actions": ["tool_actions"]}
    run_case(case, AS_OF, FakeLLM(responder(mapping)), tmp_path / "c")
    llm = FakeLLM(responder(mapping))
    snap = run_case(case, AS_OF, llm, tmp_path / "c")
    assert llm.calls == 0
    assert snap.verdicts["tool_actions"].status == "PROPOSED"


def test_prompt_lists_seed_topics_and_examples(case, tmp_path):
    seen = {}
    run_case(case, AS_OF, FakeLLM(responder(seen=seen)), tmp_path / "c")
    (user,) = seen["map_topics"]
    assert "- tool_actions:" in user
    assert "External actions are disabled for the pilot." in user


def test_map_new_topics_unit(tmp_path):
    cache = JsonCache(tmp_path / "c")
    llm = FakeLLM(responder({"other:a": ["SOC 2", "dpa", "bogus"], "other:b": ["bogus"]}))
    out, err = map_new_topics({"other:a": ["x"], "other:b": ["y"]}, llm, cache)
    assert err is None
    assert out == {"other:a": ["soc2", "dpa"]}


def test_map_new_topics_empty_makes_no_call(tmp_path):
    llm = FakeLLM(responder())
    assert map_new_topics({}, llm, JsonCache(tmp_path / "c")) == ({}, None)
    assert llm.calls == 0


def test_fact_without_claim_uses_its_quote_as_example(case, tmp_path, monkeypatch):
    fact = EXTRACTIONS["chat.txt"]["facts"][0]
    monkeypatch.setitem(fact, "claim", "")
    seen = {}
    run_case(case, AS_OF, FakeLLM(responder(seen=seen)), tmp_path / "c")
    (user,) = seen["map_topics"]
    line = next(ln for ln in user.splitlines() if ln.startswith("- other:external_actions:"))
    assert "External actions disabled for the pilot." in line
