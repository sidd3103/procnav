from procnav.cache import JsonCache
from procnav.llm import FakeLLM
from procnav.models import Fact, Rank
from procnav.reconcile import (
    PROMPT_VERSION,
    ReconcileOut,
    needs_reconcile,
    reconcile_all,
    to_reconciliation,
)


def F(fid, file, value="v", topic="storage_location", applies="this_customer"):
    return Fact(id=fid, file=file, line=1, topic=topic, claim=f"claim {fid}", value=value,
                applies_to=applies, certainty="stated", date=None, speaker=None,
                quote=f"quote {fid}", valid_until=None, rank=Rank.CUSTOMER,
                source_type="internal_chat", case="c", customer="X")


def test_trigger_needs_two_files_excluding_other_customer():
    assert not needs_reconcile([F("a", "1.md"), F("b", "1.md")])
    assert not needs_reconcile([F("a", "1.md"), F("b", "2.md", applies="other_customer")])
    assert needs_reconcile([F("a", "1.md"), F("b", "2.md")])


def test_to_reconciliation_maps_labels_and_fails_closed():
    facts = [F("a", "1.md", "AWS Canada"), F("b", "2.md", "Canada"), F("c", "3.md", "US")]
    out = ReconcileOut.model_validate({
        "facts": [{"id": "a", "about_topic": "yes"}, {"id": "b", "about_topic": "unclear"},
                  {"id": "c", "about_topic": "no"}, {"id": "zz", "about_topic": "no"}],
        "pairs": [{"a": "a", "b": "b", "relation": "more_specific"},
                  {"a": "a", "b": "c", "relation": "contradicts"},
                  {"a": "b", "b": "c", "relation": "unclear"},
                  {"a": "a", "b": "zz", "relation": "same"},
                  {"a": "a", "b": "a", "relation": "same"},
                  {"a": "b", "b": "c", "relation": "bogus"}],
    })
    r = to_reconciliation(out, facts)
    assert r.off_topic == {"c"}
    assert r.agree == {frozenset({"a", "b"})}
    assert r.contradict == {frozenset({"a", "c"})}


def test_conflicting_labels_for_one_pair_count_as_contradiction():
    facts = [F("a", "1.md"), F("b", "2.md")]
    out = ReconcileOut.model_validate({"pairs": [
        {"a": "a", "b": "b", "relation": "same"}, {"a": "b", "b": "a", "relation": "contradicts"}]})
    r = to_reconciliation(out, facts)
    assert r.agree == set() and r.contradict == {frozenset({"a", "b"})}


def test_same_file_pair_ignored():
    facts = [F("a", "1.md"), F("b", "1.md"), F("c", "2.md")]
    out = ReconcileOut.model_validate({"pairs": [{"a": "a", "b": "b", "relation": "same"}]})
    assert to_reconciliation(out, facts).agree == set()


def test_reconcile_all_calls_only_multi_file_topics_and_caches(tmp_path):
    facts = [F("a", "1.md", "AWS Canada"), F("b", "2.md", "Canada"),
             F("x", "1.md", topic="dpa"), F("y", "1.md", topic="dpa")]
    seen = []

    def respond(system, user, schema):
        seen.append(user)
        return schema.model_validate(
            {"pairs": [{"a": "a", "b": "b", "relation": "same"}]})

    cache = JsonCache(tmp_path)
    llm = FakeLLM(respond)
    recon, failed = reconcile_all(facts, llm, cache)
    assert llm.calls == 1 and failed == []
    assert seen[0].startswith("KEY: reconcile storage_location\n")
    assert "id=a" in seen[0] and "file=2.md" in seen[0] and "quote b" in seen[0]
    assert "- a=a b=b" in seen[0]
    assert recon["storage_location"].agree == {frozenset({"a", "b"})}
    assert "dpa" not in recon

    llm2 = FakeLLM(respond)
    recon2, _ = reconcile_all(facts, llm2, cache)
    assert llm2.calls == 0 and recon2 == recon


def test_reconcile_llm_error_gives_failed_entry_and_no_labels(tmp_path):
    facts = [F("a", "1.md"), F("b", "2.md")]
    llm = FakeLLM({"reconcile storage_location": [RuntimeError("boom"), RuntimeError("boom")]})
    recon, failed = reconcile_all(facts, llm, JsonCache(tmp_path))
    assert recon == {}
    assert [k for k, _ in failed] == ["reconcile:storage_location"] and "boom" in failed[0][1]


def test_other_customer_facts_not_sent(tmp_path):
    facts = [F("a", "1.md"), F("b", "2.md"), F("o", "3.md", applies="other_customer")]
    seen = []

    def respond(system, user, schema):
        seen.append(user)
        return schema.model_validate({})

    reconcile_all(facts, FakeLLM(respond), JsonCache(tmp_path))
    assert "id=o" not in seen[0]


def test_prompt_version_is_set():
    assert PROMPT_VERSION
