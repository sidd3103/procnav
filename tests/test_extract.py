import hashlib

import pytest
from pydantic import ValidationError

from procnav.cache import JsonCache
from procnav.extract import (
    PROMPT_VERSION,
    SYSTEM_PROMPT,
    CaseInfo,
    CaseOut,
    DecOut,
    DepOut,
    ExtractionOut,
    FactOut,
    MapItem,
    MapOut,
    ReqOut,
    extract_file,
    identify_case,
    map_questions,
    map_questions_checked,
    rank_for,
)
from procnav.llm import FakeLLM
from procnav.loader import CaseFile
from procnav.models import Rank, Requirement

PATH = "03_internal/thread.md"
TEXT = (
    "Dana: Storage is in AWS ca-central-1 for this customer.\n"
    "Lee: Retention for app logs is probably 30 days.\n"
    "Dana: DPA is blocked until subprocessors are confirmed.\n"
    "Lee: Fallback provider is not finalized; customer decides.\n"
)
INFO = CaseInfo(customer="Acme Bank", supplier="Vendor Co", case="CASE_1")


def mkfile(text=TEXT, path=PATH):
    return CaseFile(path=path, text=text, sha256=hashlib.sha256(text.encode()).hexdigest())


def fact(**kw):
    base = dict(topic="storage_location", claim="Data stored in ca-central-1",
                value="AWS ca-central-1", applies_to="this_customer", certainty="stated",
                date="2026-10-01", speaker="Dana",
                quote="Storage is in AWS ca-central-1 for this customer.", valid_until=None)
    base.update(kw)
    return FactOut(**base)


def out(source_type="internal_chat", facts=None, requirements=None, deps=None, decs=None):
    return ExtractionOut(
        source_type=source_type, customer_scope="this_customer", date=None, is_draft=False,
        facts=facts if facts is not None else [fact()],
        requirements=requirements or [], dependencies=deps or [], open_decisions=decs or [],
    )


@pytest.fixture
def cache(tmp_path):
    return JsonCache(tmp_path / "cache")


def test_prompt_version():
    assert PROMPT_VERSION == "v6"


def test_valid_fact_is_built(cache):
    llm = FakeLLM({PATH: out()})
    ef = extract_file(mkfile(), INFO, llm, cache)
    assert ef.failed is None
    assert ef.source_type == "internal_chat"
    [f] = ef.facts
    assert f.id == "03_internal/thread#F1"
    assert f.file == PATH and f.line == 1
    assert f.topic == "storage_location"
    assert f.rank == Rank.CUSTOMER
    assert f.case == "CASE_1" and f.customer == "Acme Bank"
    assert f.source_type == "internal_chat"


def test_prompt_starts_with_key_and_has_context(cache):
    seen = {}

    def responder(system, user, schema):
        seen["system"], seen["user"] = system, user
        return out()

    extract_file(mkfile(), INFO, FakeLLM(responder), cache)
    assert seen["user"].splitlines()[0] == f"KEY: {PATH}"
    assert "Acme Bank" in seen["user"] and "Vendor Co" in seen["user"]
    assert TEXT in seen["user"]
    assert "Acme" not in seen["system"]


def test_invented_quote_dropped_and_listed(cache):
    bad = fact(quote="We store everything in Frankfurt.")
    llm = FakeLLM({PATH: out(facts=[fact(), bad])})
    ef = extract_file(mkfile(), INFO, llm, cache)
    assert len(ef.facts) == 1
    assert len(ef.dropped) == 1
    assert "Frankfurt" in ef.dropped[0]


def test_probably_quote_becomes_proposed(cache):
    f = fact(topic="retention_app_logs", value="30 days", certainty="stated",
             quote="Retention for app logs is probably 30 days.")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[f])}), cache)
    assert ef.facts[0].certainty == "proposed"
    assert ef.facts[0].line == 2


def test_rank_for():
    assert rank_for("evidence", "this_customer") == Rank.EVIDENCE
    assert rank_for("internal_chat", "other_customer") == Rank.OTHER_CUSTOMER
    assert rank_for("evidence", "other_customer") == Rank.OTHER_CUSTOMER
    assert rank_for("generic_doc", "generic") == Rank.GENERIC
    assert rank_for("internal_chat", "generic") == Rank.GENERIC
    assert rank_for("internal_chat", "this_customer") == Rank.CUSTOMER
    assert rank_for("prior_customer", "this_customer") == Rank.OTHER_CUSTOMER
    # evidence about the platform is still evidence (checked before generic)
    assert rank_for("evidence", "generic") == Rank.EVIDENCE


def test_prompt_valid_until_is_expiry_only():
    assert "never an audit or report period" in SYSTEM_PROMPT
    assert "report period end" not in SYSTEM_PROMPT


def test_ids_use_path_without_extension(cache):
    a = mkfile(path="05_evidence/notes.txt")
    b = mkfile(path="03_internal/notes.md")
    llm = FakeLLM({a.path: out(), b.path: out()})
    ida = extract_file(a, INFO, llm, cache).facts[0].id
    idb = extract_file(b, INFO, llm, cache).facts[0].id
    assert ida == "05_evidence/notes#F1" and idb == "03_internal/notes#F1"


def test_unknown_topic_becomes_other(cache):
    f = fact(topic="Quantum Key Escrow")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[f])}), cache)
    assert ef.facts[0].topic == "other:quantum_key_escrow"


def test_second_call_uses_cache(cache):
    llm = FakeLLM({PATH: [out(), out()]})
    a = extract_file(mkfile(), INFO, llm, cache)
    b = extract_file(mkfile(), INFO, llm, cache)
    assert llm.calls == 1
    assert a == b


def test_cache_key_includes_customer(cache):
    llm = FakeLLM({PATH: [out(), out()]})
    extract_file(mkfile(), INFO, llm, cache)
    extract_file(mkfile(), CaseInfo(customer="Other", supplier="Vendor Co"), llm, cache)
    assert llm.calls == 2


def test_raising_twice_sets_failed(cache):
    llm = FakeLLM({PATH: [RuntimeError("boom"), RuntimeError("boom again")]})
    ef = extract_file(mkfile(), INFO, llm, cache)
    assert ef.failed and "boom" in ef.failed
    assert ef.facts == [] and ef.requirements == []
    assert llm.calls == 2


def test_failure_not_cached(cache):
    llm = FakeLLM({PATH: [RuntimeError("a"), RuntimeError("b"), out()]})
    assert extract_file(mkfile(), INFO, llm, cache).failed
    assert extract_file(mkfile(), INFO, llm, cache).failed is None


def test_invalid_then_valid_retries_once(cache):
    try:
        ExtractionOut.model_validate({"source_type": "nonsense"})
    except ValidationError as e:
        err = e
    llm = FakeLLM({PATH: [err, out()]})
    ef = extract_file(mkfile(), INFO, llm, cache)
    assert ef.failed is None
    assert len(ef.facts) == 1
    assert llm.calls == 2


@pytest.mark.parametrize("st", ["guidance", "people", "case_header", "questionnaire"])
def test_non_fact_sources_yield_no_facts(cache, st):
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(source_type=st)}), cache)
    assert ef.facts == []
    assert ef.source_type == st


def test_requirements_only_from_customer_request(cache, tmp_path):
    req = ReqOut(text="Where is data stored?", requester="Dana", topics=["storage_location", "Weird Thing"],
                 quote="Storage is in AWS ca-central-1 for this customer.")
    chat = extract_file(mkfile(), INFO, FakeLLM({PATH: out(requirements=[req])}), cache)
    assert chat.requirements == []
    f2 = mkfile(path="01_customer/email.md")
    ef = extract_file(f2, INFO, FakeLLM({"01_customer/email.md": out("customer_request", requirements=[req])}),
                      JsonCache(tmp_path / "cache2"))
    [r] = ef.requirements
    assert r.id == "01_customer/email#R1"
    assert r.from_customer is True
    assert r.source_file == "01_customer/email.md"
    assert r.topics == ["storage_location", "other:weird_thing"]


def test_dependencies_and_decisions(cache):
    dep = DepOut(kind="blocked_until", subject="DPA", depends_on_topics=["subprocessors"],
                 date="2026-10-15", quote="DPA is blocked until subprocessors are confirmed.")
    dep2 = DepOut(kind="milestone", subject="Pilot", depends_on_topics=[], date="October",
                  quote="DPA is blocked until subprocessors are confirmed.")
    dec = DecOut(text="Fallback provider", decider="customer", affected_topics=["model_fallback"],
                 quote="Fallback provider is not finalized; customer decides.")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(deps=[dep, dep2], decs=[dec])}), cache)
    d1, d2 = ef.dependencies
    assert d1.id == "03_internal/thread#D1" and d1.date.isoformat() == "2026-10-15" and d1.file == PATH
    assert d2.date is None
    [o] = ef.open_decisions
    assert o.id == "03_internal/thread#O1" and o.decider == "customer" and o.affected_topics == ["model_fallback"]


def test_prior_customer_drops_deps_and_forces_other(cache):
    dep = DepOut(kind="blocked_until", subject="DPA", depends_on_topics=[], date=None,
                 quote="DPA is blocked until subprocessors are confirmed.")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out("prior_customer", deps=[dep])}), cache)
    assert ef.dependencies == []
    assert ef.facts[0].applies_to == "other_customer"
    assert ef.facts[0].rank == Rank.OTHER_CUSTOMER


def test_valid_until_parsed(cache):
    f = fact(valid_until="2027-03-31")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[f])}), cache)
    assert ef.facts[0].valid_until.isoformat() == "2027-03-31"
    f = fact(valid_until="next spring")
    ef = extract_file(mkfile(), CaseInfo(customer="X", supplier="Y"), FakeLLM({PATH: out(facts=[f])}), cache)
    assert ef.facts[0].valid_until is None


def test_identify_case(cache):
    files = [mkfile(), mkfile(text="Customer: Acme Bank\n", path="00_START.txt")]
    seen = {}

    def responder(system, user, schema):
        seen["user"] = user
        return CaseOut(customer="Acme Bank", supplier="Vendor Co")

    llm = FakeLLM(responder)
    info = identify_case(files, llm, cache, case="CASE_1")
    assert info == CaseInfo(customer="Acme Bank", supplier="Vendor Co", case="CASE_1")
    assert seen["user"].splitlines()[0] == "KEY: identify_case"
    assert "00_START.txt" in seen["user"]
    identify_case(files, llm, cache, case="CASE_1")
    assert llm.calls == 1


def test_identify_case_returns_service_and_quote(cache):
    files = [mkfile(text="Service: AI assistant for support\n", path="00_START.txt")]
    out = CaseOut(customer="Acme Bank", supplier="Vendor Co", service="AI assistant for support",
                  service_quote="Service: AI assistant for support")
    info = identify_case(files, FakeLLM({"identify_case": out}), cache, case="CASE_1")
    assert info.service == "AI assistant for support"
    assert info.service_quote == "Service: AI assistant for support"
    # cached path keeps the service too
    again = identify_case(files, FakeLLM({}), cache, case="CASE_1")
    assert again == info


def test_identify_prompt_asks_for_service_with_verbatim_quote():
    from procnav.extract import IDENTIFY_SYSTEM
    low = IDENTIFY_SYSTEM.lower()
    assert "service" in low and "service_quote" in low and "verbatim" in low


def test_identify_service_defaults_empty():
    co = CaseOut.model_validate({"customer": "A", "supplier": "B"})
    assert co.service == "" and co.service_quote == ""


def _req(id_, text, topics=None):
    return Requirement(id=id_, text=text, requester=None, topics=topics or [], source_file="q.csv",
                       quote=None, draft_answer=None, from_customer=False, group=None)


def test_map_questions(cache):
    reqs = [_req("Q1", "Where is data stored?"), _req("Q2", "Do you have SOC 2?"), _req("Q3", "Other?")]
    seen = {}

    def responder(system, user, schema):
        seen["user"] = user
        return MapOut(mappings=[MapItem(req_id="Q1", topics=["storage_location", "inference_location"]),
                                MapItem(req_id="Q2", topics=["soc2", "Brand New Thing"])])

    llm = FakeLLM(responder)
    res = map_questions(reqs, llm, cache)
    assert seen["user"].splitlines()[0] == "KEY: map_questions"
    assert "storage_location" in seen["user"]
    assert res[0].topics == ["storage_location", "inference_location"]
    assert res[1].topics == ["soc2", "other:brand_new_thing"]
    assert res[2].topics == []
    assert reqs[0].topics == []  # originals untouched
    map_questions(reqs, llm, cache)
    assert llm.calls == 1


def test_map_questions_checked_reports_failure(cache):
    reqs = [_req("Q1", "Where?")]
    llm = FakeLLM({"map_questions": [RuntimeError("x"), RuntimeError("y")]})
    res, err = map_questions_checked(reqs, llm, cache)
    assert res[0].topics == [] and err and "RuntimeError" in err
    ok = FakeLLM({"map_questions": MapOut(mappings=[MapItem(req_id="Q1", topics=["soc2"])])})
    res, err = map_questions_checked(reqs, ok, cache)
    assert err is None and res[0].topics == ["soc2"]
    assert map_questions_checked([], ok, cache) == ([], None)


def test_map_questions_failure_returns_unchanged(cache):
    reqs = [_req("Q1", "Where?")]
    llm = FakeLLM({"map_questions": [RuntimeError("x"), RuntimeError("y")]})
    assert map_questions(reqs, llm, cache)[0].topics == []


# ---------------------------------------------------------------- final fix wave

@pytest.mark.parametrize("raw", ["Oct 1, 2026", "October 1, 2026", "2026-10-01T00:00:00Z",
                                 "1 October 2026", "1 Oct 2026", "2026/10/01", " Oct 1, 2026. "])
def test_valid_until_lenient_formats(cache, raw):
    from datetime import date as _d
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[fact(valid_until=raw)])}), cache)
    assert ef.facts[0].valid_until == _d(2026, 10, 1)
    assert ef.facts[0].valid_until_raw is None


def test_valid_until_written_as_words_gives_expired_caveat(cache):
    from datetime import date as _d

    from procnav.resolve import resolve_topic
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[fact(valid_until="Oct 1, 2026")])}),
                      cache)
    v = resolve_topic("storage_location", ef.facts, _d(2026, 11, 1))
    assert "expired 2026-10-01" in v.caveats and not v.sendable


def test_garbage_valid_until_kept_raw_and_not_sendable(cache):
    from datetime import date as _d

    from procnav.resolve import resolve_topic
    f = fact(valid_until="end of the renewal cycle")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[f])}), cache)
    [fa] = ef.facts
    assert fa.valid_until is None and fa.valid_until_raw == "end of the renewal cycle"
    v = resolve_topic("storage_location", [fa], _d(2026, 11, 1))
    assert v.caveats == ["expires on unreadable date 'end of the renewal cycle'"]
    assert not v.sendable


def test_fact_and_dependency_dates_lenient(cache):
    f = fact(date="Sep 3, 2026")
    dep = DepOut(kind="milestone", subject="Pilot", depends_on_topics=[], date="October 15, 2026",
                 quote="DPA is blocked until subprocessors are confirmed.")
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: out(facts=[f], deps=[dep])}), cache)
    assert ef.facts[0].date == "2026-09-03"
    assert ef.dependencies[0].date.isoformat() == "2026-10-15"


def test_extraction_out_defaults_and_coercion():
    eo = ExtractionOut.model_validate({
        "source_type": "Internal_Chat", "customer_scope": " This_Customer ",
        "facts": [{"topic": "retention_app_logs", "claim": "c", "value": 14,
                   "applies_to": "This_Customer", "certainty": "Stated", "quote": "q"}],
        "requirements": [{"text": "t", "quote": "q"}],
    })
    assert eo.source_type == "internal_chat" and eo.customer_scope == "this_customer"
    assert eo.date is None and eo.is_draft is False
    assert eo.dependencies == [] and eo.open_decisions == []
    [fo] = eo.facts
    assert fo.value == "14" and fo.certainty == "stated" and fo.applies_to == "this_customer"
    assert fo.speaker is None and fo.valid_until is None and fo.date is None
    [ro] = eo.requirements
    assert ro.requester is None and ro.topics == []
    d = DepOut.model_validate({"kind": "Milestone", "subject": "s", "quote": "q"})
    assert d.kind == "milestone" and d.date is None and d.depends_on_topics == []
    o = DecOut.model_validate({"text": "t", "decider": "Customer", "quote": "q"})
    assert o.decider == "customer" and o.affected_topics == []
    assert MapOut.model_validate({}).mappings == []
    assert MapItem.model_validate({"req_id": 7}).req_id == "7"


def test_prompt_v3_no_case_content_and_value_rules():
    assert PROMPT_VERSION == "v6"
    assert "ca-central-1" not in SYSTEM_PROMPT
    assert "never add a region, number or name that is not written" in SYSTEM_PROMPT
    assert "'enabled' or 'disabled'" in SYSTEM_PROMPT


def test_identify_cache_ignores_text_beyond_preview(cache):
    from procnav.extract import IDENTIFY_PREVIEW_CHARS
    long = "Customer: Acme Bank\n" + "x" * IDENTIFY_PREVIEW_CHARS
    llm = FakeLLM({"identify_case": [CaseOut(customer="Acme Bank", supplier="V")] * 2})
    identify_case([mkfile(text=long, path="a.txt")], llm, cache)
    identify_case([mkfile(text=long + "\nappended line\n", path="a.txt")], llm, cache)
    assert llm.calls == 1
    identify_case([mkfile(text="changed " + long, path="a.txt")], llm, cache)
    assert llm.calls == 2


def test_extraction_cache_customer_normalised(cache):
    llm = FakeLLM({PATH: [out(), out()]})
    extract_file(mkfile(), INFO, llm, cache)
    extract_file(mkfile(), CaseInfo(customer="  ACME   bank ", supplier="Vendor Co"), llm, cache)
    assert llm.calls == 1


def test_system_prompts_are_part_of_cache_keys(cache, monkeypatch):
    import procnav.extract as ex
    llm = FakeLLM({PATH: [out(), out()],
                   "identify_case": [CaseOut(customer="A", supplier="V")] * 2,
                   "map_questions": [MapOut(mappings=[])] * 2})
    reqs = [_req("Q1", "Where?")]
    extract_file(mkfile(), INFO, llm, cache)
    identify_case([mkfile()], llm, cache)
    map_questions(reqs, llm, cache)
    assert llm.calls == 3
    monkeypatch.setattr(ex, "SYSTEM_PROMPT", ex.SYSTEM_PROMPT + "\nchanged")
    monkeypatch.setattr(ex, "IDENTIFY_SYSTEM", ex.IDENTIFY_SYSTEM + "\nchanged")
    monkeypatch.setattr(ex, "MAP_SYSTEM", ex.MAP_SYSTEM + "\nchanged")
    extract_file(mkfile(), INFO, llm, cache)
    assert llm.calls == 4
    identify_case([mkfile()], llm, cache)
    assert llm.calls == 5
    map_questions(reqs, llm, cache)
    assert llm.calls == 6


# ---------------------------------------------------------------- live run 1 fixes

EV_PATH = "05_evidence/status.txt"
EV_TEXT = (
    "Coverage: CAD 1,000,000. Current certificate expires Oct 1, 2026. Renewal pending.\n"
    "Policy valid through 2027-03-31.\n"
    "Retention period: 2026-01-15.\n"
    "This file contains no remediation evidence.\n"
    "2026 audit is not yet final.\n"
    "Retention TBD.\n"
)


def _ev(facts, source_type="evidence"):
    f = mkfile(text=EV_TEXT, path=EV_PATH)
    eo = ExtractionOut(source_type=source_type, customer_scope="generic", facts=facts)
    return f, FakeLLM({EV_PATH: eo})


# R25: a missing field on one item never fails the file

def test_open_decision_missing_quote_validates_and_is_dropped(cache):
    raw = {
        "source_type": "internal_chat", "customer_scope": "this_customer",
        "facts": [fact().model_dump()],
        "dependencies": [{"kind": "blocked_until", "subject": "DPA",
                          "depends_on_topics": ["subprocessors"],
                          "quote": "DPA is blocked until subprocessors are confirmed."}],
        "open_decisions": [{"text": "Which fallback provider?", "decider": "customer",
                            "affected_topics": ["model_fallback"]}],
    }
    eo = ExtractionOut.model_validate(raw)  # used to raise "open_decisions.0.quote Field required"
    assert eo.open_decisions[0].quote == ""
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: eo}), cache)
    assert ef.failed is None
    assert ef.open_decisions == []
    assert any(d.startswith("open_decision:") and "Which fallback provider?" in d
               for d in ef.dropped)
    assert len(ef.facts) == 1 and len(ef.dependencies) == 1


def test_missing_quote_from_raw_json_does_not_fail_file(cache):
    import json

    def responder(system, user, schema):  # mimics OpenAIClient: validate the raw JSON text
        return schema.model_validate_json(json.dumps({
            "source_type": "internal_chat", "customer_scope": "this_customer",
            "facts": [fact().model_dump(), {"topic": "retention_app_logs", "claim": "c",
                                            "value": "30 days", "applies_to": "this_customer",
                                            "certainty": "stated"}],
            "requirements": [{"text": "t"}],
            "dependencies": [{"kind": "milestone", "subject": "Pilot"}],
            "open_decisions": [{"text": "t", "decider": "internal"}],
        }))

    ef = extract_file(mkfile(), INFO, FakeLLM(responder), cache)
    assert ef.failed is None
    assert [f.topic for f in ef.facts] == ["storage_location"]
    assert ef.dependencies == [] and ef.open_decisions == []
    kinds = sorted(d.split(":", 1)[0] for d in ef.dropped)
    assert kinds == ["dependency", "fact", "open_decision"]
    assert all("quote missing" in d for d in ef.dropped)


def test_every_required_string_has_a_safe_default():
    fo = FactOut.model_validate({"applies_to": "generic", "certainty": "stated"})
    assert fo.topic == "" and fo.claim == "" and fo.quote == ""
    ro = ReqOut.model_validate({})
    assert ro.text == "" and ro.quote == ""
    do = DepOut.model_validate({"kind": "milestone"})
    assert do.subject == "" and do.quote == ""
    oo = DecOut.model_validate({"decider": "internal"})
    assert oo.text == "" and oo.quote == ""


def test_item_with_invalid_enum_dropped_not_file(cache):
    eo = ExtractionOut.model_validate({
        "source_type": "internal_chat", "customer_scope": "this_customer",
        "facts": [fact().model_dump(), {"topic": "x", "certainty": "certain-ish", "quote": "q",
                                        "applies_to": "this_customer"}],
        "open_decisions": [{"text": "t", "decider": "vendor", "quote": "q"}],
        "dependencies": None,
    })
    assert len(eo.facts) == 1 and eo.open_decisions == [] and eo.dependencies == []
    assert len(eo.invalid_items) == 2
    assert "invalid_items" not in ExtractionOut.model_json_schema()["properties"]
    ef = extract_file(mkfile(), INFO, FakeLLM({PATH: eo}), cache)
    assert ef.failed is None and len(ef.facts) == 1
    assert sum("invalid" in d for d in ef.dropped) == 2
    # the notes survive a cache round trip
    again = extract_file(mkfile(), INFO, FakeLLM({}), cache)
    assert again.dropped == ef.dropped


# R27 (a): an expiry written in value/quote fails closed into valid_until

def test_expiry_date_in_value_moves_to_valid_until(cache):
    from datetime import date as _d
    fo = fact(topic="insurance", value="2026-10-01", valid_until=None,
              quote="Current certificate expires Oct 1, 2026.")
    f, llm = _ev([fo])
    [fa] = extract_file(f, INFO, llm, cache).facts
    assert fa.valid_until == _d(2026, 10, 1)


def test_expiry_prefers_date_from_quote(cache):
    from datetime import date as _d
    fo = fact(topic="insurance", value="Oct 2, 2026", quote="Current certificate expires Oct 1, 2026.")
    f, llm = _ev([fo])
    assert extract_file(f, INFO, llm, cache).facts[0].valid_until == _d(2026, 10, 1)
    fo = fact(topic="insurance", value="March 31, 2027", quote="Policy valid through 2027-03-31.")
    f, llm = _ev([fo])
    assert extract_file(f, CaseInfo(customer="X", supplier="Y"), llm, cache).facts[0].valid_until \
        == _d(2027, 3, 31)


def test_date_value_without_expiry_wording_is_not_an_expiry(cache):
    fo = fact(topic="retention_app_logs", value="2026-01-15", quote="Retention period: 2026-01-15.")
    f, llm = _ev([fo])
    assert extract_file(f, INFO, llm, cache).facts[0].valid_until is None


def test_expiry_wording_with_non_date_value_unchanged(cache):
    fo = fact(topic="insurance", value="CAD 1,000,000",
              quote="Current certificate expires Oct 1, 2026.")
    f, llm = _ev([fo])
    assert extract_file(f, INFO, llm, cache).facts[0].valid_until is None


def test_written_valid_until_is_kept(cache):
    from datetime import date as _d
    fo = fact(topic="insurance", value="2026-10-01", valid_until="2026-12-31",
              quote="Current certificate expires Oct 1, 2026.")
    f, llm = _ev([fo])
    assert extract_file(f, INFO, llm, cache).facts[0].valid_until == _d(2026, 12, 31)


# R27 (b): placeholder values mean "no value stated"

@pytest.mark.parametrize("val", ["TBD", "tbd.", " To Be Determined ", "unknown", "N/A",
                                 "Pending", "none", "Not decided"])
def test_placeholder_value_becomes_none(cache, val):
    fo = fact(topic="retention_source_files", value=val, certainty="proposed", quote="Retention TBD.")
    f, llm = _ev([fo], source_type="customer_config_draft")
    [fa] = extract_file(f, INFO, llm, cache).facts
    assert fa.value is None


def test_real_values_are_not_placeholders(cache):
    for val in ["none of the above vendors", "30 days", "no"]:
        fo = fact(topic="retention_source_files", value=val, certainty="proposed",
                  quote="Retention TBD.")
        f, llm = _ev([fo], source_type="customer_config_draft")
        [fa] = extract_file(f, CaseInfo(customer=val, supplier="Y"), llm, cache).facts
        assert fa.value == val


# R27 backstop applied in extraction

def test_stated_absence_wording_becomes_absent(cache):
    a = fact(topic="pen_test_remediation", value="no remediation evidence", certainty="stated",
             quote="This file contains no remediation evidence.")
    b = fact(topic="soc2_current", value="not final for 2026", certainty="stated",
             quote="2026 audit is not yet final.")
    f, llm = _ev([a, b])
    assert [x.certainty for x in extract_file(f, INFO, llm, cache).facts] == ["absent", "absent"]


# R27 (c): prompt v4

def test_prompt_v4_absence_and_expiry_examples():
    assert PROMPT_VERSION == "v6"
    p = SYSTEM_PROMPT
    assert "certainty \"absent\"" in p
    for phrase in ["contains no", "is not yet final", "has been provided"]:
        assert phrase in p, phrase
    assert "never put an expiry or renewal date in value" in p
    assert "TBD" in p


# ---- R29: quote fallback to the item's own text

CORR_PATH = "01_customer/mail.txt"
CORR_TEXT = (
    "Sep 10 - Sponsor: Spring production pilot is agreed in principle, subject to onboarding.\n"
    "Sep 21 - Privacy: Clarify personal-information scope and retention by data type.\n"
    "Sep 22 - Sponsor: The team is deciding whether the pilot uses real data.\n"
)


def _corr(**kw):
    f = mkfile(text=CORR_TEXT, path=CORR_PATH)
    eo = ExtractionOut(source_type="customer_request", customer_scope="this_customer", **kw)
    return f, FakeLLM({CORR_PATH: eo})


def test_empty_quote_dependency_falls_back_to_subject(cache):
    dep = DepOut(kind="milestone", subject="Spring production pilot", depends_on_topics=[])
    f, llm = _corr(dependencies=[dep])
    ef = extract_file(f, INFO, llm, cache)
    [d] = ef.dependencies
    assert d.quote == "Spring production pilot" and ef.dropped == []
    assert d.id == "01_customer/mail#D1"


def test_fallback_gives_right_line_for_each_item_kind(cache):
    req = ReqOut(text="Clarify personal-information scope and retention by data type.")
    dec = DecOut(text="whether the pilot uses real data", decider="customer", quote="invented")
    fo = FactOut(topic="other:pilot", claim="nope", value="Spring production pilot",
                 applies_to="this_customer", certainty="proposed")
    f, llm = _corr(requirements=[req], open_decisions=[dec], facts=[fo])
    ef = extract_file(f, INFO, llm, cache)
    assert ef.requirements[0].quote == req.text
    assert ef.open_decisions[0].quote == "whether the pilot uses real data"
    assert ef.facts[0].quote == "Spring production pilot" and ef.facts[0].line == 1
    assert ef.dropped == []


def test_fallback_uses_claim_before_value_and_reports_line(cache):
    fo = FactOut(topic="other:x", claim="the pilot uses real data", value="Clarify",
                 applies_to="this_customer", certainty="proposed")
    f, llm = _corr(facts=[fo])
    [fa] = extract_file(f, INFO, llm, cache).facts
    assert fa.quote == "the pilot uses real data" and fa.line == 3


def test_empty_quote_with_paraphrased_text_still_dropped(cache):
    dep = DepOut(kind="milestone", subject="October production pilot")  # not in file
    req = ReqOut(text="Complete the supplier profile for the configuration they would receive.")
    dec = DecOut(text="Whether the pilot uses fake data", decider="customer")  # not in file
    f, llm = _corr(dependencies=[dep], requirements=[req], open_decisions=[dec])
    ef = extract_file(f, INFO, llm, cache)
    assert ef.dependencies == [] and ef.requirements == [] and ef.open_decisions == []
    assert len(ef.dropped) == 3


def test_prompt_v5_requires_quotes_on_every_item():
    assert PROMPT_VERSION == "v6"
    assert ("Every fact, requirement, dependency and open decision MUST include a verbatim quote"
            in SYSTEM_PROMPT)


def test_fallback_needs_at_least_three_words(cache):
    # a one- or two-word text would match almost any file by accident
    fo = FactOut(topic="other:x", claim="pilot", value="Sep 10",
                 applies_to="this_customer", certainty="proposed")
    f, llm = _corr(facts=[fo])
    ef = extract_file(f, INFO, llm, cache)
    assert ef.facts == [] and ef.dropped == ["fact: quote missing: 'pilot'"]


def test_fallback_matches_case_insensitively_with_original_line_and_span(cache):
    dec = DecOut(text="Whether the pilot uses real data.", decider="customer")
    f, llm = _corr(open_decisions=[dec])
    ef = extract_file(f, INFO, llm, cache)
    [o] = ef.open_decisions
    assert o.text == "Whether the pilot uses real data."
    assert o.quote == "whether the pilot uses real data."  # verbatim from the file
    assert o.id == "01_customer/mail#O1" and ef.dropped == []
    fo = FactOut(topic="other:x", claim="THE PILOT USES REAL DATA", applies_to="this_customer",
                 certainty="proposed")
    f, llm = _corr(facts=[fo])
    [fa] = extract_file(f, CaseInfo(customer="X", supplier="Y"), llm, cache).facts
    assert fa.line == 3 and fa.quote == "the pilot uses real data"


def test_llm_quote_check_stays_case_sensitive(cache):
    fo = FactOut(topic="other:x", claim="c", applies_to="this_customer", certainty="proposed",
                 quote="THE PILOT USES REAL DATA")
    f, llm = _corr(facts=[fo])
    ef = extract_file(f, INFO, llm, cache)
    assert ef.facts == [] and ef.dropped == ["fact: quote not found: 'THE PILOT USES REAL DATA'"]


def test_prompt_v6_values_keep_qualifiers_and_decider_rule():
    p = SYSTEM_PROMPT
    assert "keep its qualifiers" in p
    assert '"SOC 2 Type II, Jan-Dec 2025, final", not "final"' in p
    assert "'enabled' or 'disabled'" in p and "followed by any condition" in p
    assert "decider: customer ONLY when the text says the customer decides or approves" in p
