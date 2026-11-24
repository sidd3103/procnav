import shutil
from datetime import date

import pytest

from conftest import (
    ACME_DIR,
    INFERENCE_EDIT_LINE,
    RETEST_LETTER_PATH,
    RETEST_LETTER_TEXT,
    acme_responder,
)
from procnav.diff import diff
from procnav.llm import FakeLLM, OpenAIClient
from procnav.pipeline import NextStep, Snapshot, load_latest, run_case, save_snapshot
from procnav.resolve import normalize_value

AS_OF = date(2026, 9, 25)
PRIOR = "04_knowledge/prior_customer_answers_CONTEXT_ONLY.md"


@pytest.fixture
def acme(tmp_path, acme_llm):
    llm = acme_llm()
    snap = run_case(ACME_DIR, AS_OF, llm, tmp_path / "cache")
    return snap, llm


def _prior_fact_ids(snap):
    return {f.id for f in snap.facts if f.file == PRIOR}


def test_acme_golden(acme):
    snap, _ = acme
    v = snap.verdicts
    assert snap.case == "CASE_1_Acme_Bank"
    assert snap.customer == "Acme National Bank"
    assert snap.as_of == AS_OF

    assert v["retention_app_logs"].status == "PROPOSED"
    assert v["retention_app_logs"].value == "14 days"
    assert v["retention_traces"].status == "UNKNOWN"
    assert v["inference_location"].status == "UNKNOWN"
    assert v["model_fallback"].status == "PROPOSED"
    # the 2025 report is verified; "2026 audit not final" folds in as a caveat (not sendable)
    assert v["soc2"].status == "VERIFIED" and not v["soc2"].sendable
    assert any(c.startswith("no evidence:") and "2026" in c for c in v["soc2"].caveats)
    assert v["pen_test"].status == "VERIFIED" and v["pen_test"].sendable
    assert not v["insurance"].sendable
    assert v["encryption_in_transit"].status == "PROPOSED"
    assert v["encryption_at_rest"].status == "PROPOSED"

    prior = _prior_fact_ids(snap)
    assert prior, "prior-customer facts should be extracted (as hints only)"
    for verdict in v.values():
        assert not prior & set(verdict.used), verdict.topic

    items = {b.item for b in snap.blockers}
    assert {"subprocessors", "cross_border", "insurance"} <= items

    q4 = next(r for r in snap.required if r.req.id == "Q-004")
    assert q4.stale_draft
    assert "retention_app_logs='14 days'" in q4.stale_draft
    assert snap.dropped_quotes == []
    assert snap.failed_files == []


def test_acme_structure(acme):
    snap, _ = acme
    ids = [r.req.id for r in snap.required]
    assert all(f"Q-{i:03d}" in ids for i in range(1, 24))
    assert sum(1 for r in snap.required if (r.req.group or "").startswith("template:")) == 5
    assert any(r.req.from_customer for r in snap.required)
    assert [a.req_id for a in snap.answers] == ids
    assert "02_questionnaires/vendor_risk_questionnaire.csv" not in {f.file for f in snap.facts}
    for r in snap.required:
        for t in r.req.topics:
            assert t in snap.verdicts
    # ids carry the relative path, so same-stem files cannot collide
    assert "05_evidence/soc2_status#F1" in snap.verdicts["soc2"].used
    # every loaded file is fingerprinted
    assert snap.files["03_internal/internal_messages.txt"]
    assert "07_people/ownership.csv" in snap.files


def test_map_questions_gets_only_real_rows(tmp_path):
    seen = {}
    base = acme_responder()

    def spy(system, user, schema):
        if user.startswith("KEY: map_questions"):
            seen["user"] = user
        return base(system, user, schema)

    run_case(ACME_DIR, AS_OF, FakeLLM(spy), tmp_path / "cache")
    assert "- Q-023:" in seen["user"]
    assert "Q-024" not in seen["user"] and "Control" not in seen["user"]


def test_map_questions_failure_recorded(tmp_path, acme_llm):
    snap = run_case(ACME_DIR, AS_OF, acme_llm(fail_paths=("map_questions",)), tmp_path / "c")
    assert [p for p, _ in snap.failed_files] == ["map_questions"]
    assert "simulated LLM failure" in snap.failed_files[0][1]
    q4 = next(r for r in snap.required if r.req.id == "Q-004")
    assert q4.req.topics == [] and q4.status == "unknown"


def test_next_steps(acme):
    snap, _ = acme
    internal = [s for s in snap.next_steps if s.kind == "internal"]
    owners = [s.owner for s in internal]
    assert len(owners) == len(set(owners)), "one internal entry per owner"
    assert "Customer Procurement" not in owners and "Customer Risk Teams" not in owners

    def nblockers(s: NextStep):
        return sum(1 for i in s.items if i.startswith("BLOCKER"))

    keys = [(-nblockers(s), s.owner) for s in internal]
    assert keys == sorted(keys)
    all_items = [i for s in internal for i in s.items]
    assert any(i.startswith("retention_traces: UNKNOWN") and "Q-004" in i for i in all_items)
    assert not any(i.startswith("pen_test:") for i in all_items), "sendable topics are resolved"
    # one BLOCKER line per item, even when several rules fire for it
    blk = [i.split()[1] for i in all_items if i.startswith("BLOCKER")]
    assert len(blk) == len(set(blk))
    assert set(blk) == {b.item for b in snap.blockers}
    cb = next(i for i in all_items if i.startswith("BLOCKER cross_border"))
    assert "explicit_dependency" in cb and "asked_hard_gap" in cb

    # internal decisions fold into the owner's internal entry; customer ones are separate
    assert any("decide" in i and "Anthropic fallback" in i for i in all_items)
    decisions = [s for s in snap.next_steps if s.kind == "customer_decision"]
    assert decisions and all(s.owner.startswith("Customer") for s in decisions)
    text = " ".join(i for s in decisions for i in s.items)
    assert "synthetic-only" in text and "Anthropic" not in text
    assert len({s.owner for s in decisions}) == len(decisions)


def test_snapshot_save_load_roundtrip_and_deterministic(tmp_path, acme_llm):
    a = run_case(ACME_DIR, AS_OF, acme_llm(), tmp_path / "c1")
    b = run_case(ACME_DIR, AS_OF, acme_llm(), tmp_path / "c2")
    assert a.model_dump_json() == b.model_dump_json()
    runs = tmp_path / "runs"
    assert load_latest(runs, a.case) is None
    p = save_snapshot(a, runs)
    assert p.parent == runs / a.case and p.suffix == ".json"
    loaded = load_latest(runs, a.case)
    assert isinstance(loaded, Snapshot)
    assert loaded.model_dump_json() == a.model_dump_json()
    assert diff(loaded, a) == []


def test_rerun_unchanged_zero_calls_and_empty_diff(tmp_path, acme_llm):
    cache = tmp_path / "cache"
    first_llm = acme_llm()
    first = run_case(ACME_DIR, AS_OF, first_llm, cache)
    assert first_llm.calls > 0
    llm = acme_llm()
    second = run_case(ACME_DIR, AS_OF, llm, cache)
    assert llm.calls == 0
    assert diff(first, second) == []


def _copy_case(tmp_path):
    case = tmp_path / "CASE_1_Acme_Bank"
    shutil.copytree(ACME_DIR, case)
    return case


def test_edit_file_changes_verdict(tmp_path, acme_llm):
    case = _copy_case(tmp_path)
    cache = tmp_path / "cache"
    before = run_case(case, AS_OF, acme_llm(), cache)
    assert before.verdicts["inference_location"].status == "UNKNOWN"

    chat = case / "03_internal" / "internal_messages.txt"
    chat.write_text(chat.read_text(encoding="utf-8") + "\n" + INFERENCE_EDIT_LINE + "\n",
                    encoding="utf-8")
    llm = acme_llm()
    after = run_case(case, AS_OF, llm, cache)
    assert llm.calls >= 1
    assert after.dropped_quotes == []
    lines = diff(before, after)
    assert "~ file 03_internal/internal_messages.txt" in lines
    assert "inference_location: UNKNOWN → PROPOSED OpenAI US region" in lines
    assert after.verdicts["inference_location"].status == "PROPOSED"


def test_new_evidence_file_clears_remediation(tmp_path, acme_llm):
    case = _copy_case(tmp_path)
    cache = tmp_path / "cache"
    before = run_case(case, AS_OF, acme_llm(), cache)
    rem = before.verdicts["pen_test_remediation"]
    assert rem.status == "PROPOSED" and not rem.sendable
    assert any(c.startswith("no evidence") for c in rem.caveats)

    (case / RETEST_LETTER_PATH).write_text(RETEST_LETTER_TEXT, encoding="utf-8")
    after = run_case(case, AS_OF, acme_llm(), cache)
    assert after.dropped_quotes == [] and after.failed_files == []
    rem2 = after.verdicts["pen_test_remediation"]
    assert rem2.status == "VERIFIED" and rem2.value == "remediated"
    assert rem2.caveats == [] and rem2.sendable
    assert ("05_evidence/pen_test_summary#F2", "absence superseded by evidence") in rem2.excluded
    lines = diff(before, after)
    assert f"+ file {RETEST_LETTER_PATH}" in lines
    assert any(line.startswith("pen_test_remediation: ") and line.endswith("→ VERIFIED remediated")
               for line in lines), lines


def test_failed_file_reported(tmp_path, acme_llm):
    bad = "05_evidence/soc2_status.txt"
    snap = run_case(ACME_DIR, AS_OF, acme_llm(fail_paths=(bad,)), tmp_path / "cache")
    assert [p for p, _ in snap.failed_files] == [bad]
    assert "simulated LLM failure" in snap.failed_files[0][1]
    assert snap.verdicts["soc2"].status == "UNKNOWN"  # no silent fallback
    assert snap.verdicts["insurance"].status == "VERIFIED"  # rest of the run completed


def test_identify_failure_propagates(tmp_path):
    def boom(system, user, schema):
        raise RuntimeError("down")

    with pytest.raises(RuntimeError, match="identify_case"):
        run_case(ACME_DIR, AS_OF, FakeLLM(boom), tmp_path / "cache")


def test_missing_case_dir_raises_oserror(tmp_path, acme_llm):
    llm = acme_llm()
    with pytest.raises(OSError):
        run_case(tmp_path / "nope", AS_OF, llm, tmp_path / "cache")
    assert llm.calls == 0


@pytest.mark.live
def test_live_acme(tmp_path):
    snap = run_case(ACME_DIR, AS_OF, OpenAIClient.from_env(), tmp_path / "cache")
    v = snap.verdicts
    assert snap.failed_files == []
    apl = v.get("retention_app_logs")
    assert apl is not None and apl.value and normalize_value(apl.value) == "14 days"
    assert apl.status != "VERIFIED"
    assert v["retention_traces"].status in ("UNKNOWN", "PROPOSED")
    assert v["inference_location"].status in ("UNKNOWN", "CONFLICT")
    assert v["model_fallback"].status in ("PROPOSED", "CONFLICT")
    # the 2025 report is verified; "2026 audit not final" folds in as a caveat (not sendable)
    assert v["soc2"].status == "VERIFIED" and not v["soc2"].sendable
    assert any(c.startswith("no evidence:") and "2026" in c for c in v["soc2"].caveats)
    assert v["pen_test"].status == "VERIFIED" and v["pen_test"].sendable
    assert not v["insurance"].sendable
    prior = _prior_fact_ids(snap)
    for verdict in v.values():
        assert not prior & set(verdict.used), verdict.topic
    items = {b.item for b in snap.blockers}
    assert {"subprocessors", "cross_border"} <= items
    q4 = next(r for r in snap.required if r.req.id == "Q-004")
    assert q4.stale_draft


def test_append_line_reextracts_one_file_and_skips_identify(tmp_path):
    case = _copy_case(tmp_path)
    cache = tmp_path / "cache"
    run_case(case, AS_OF, FakeLLM(acme_responder()), cache)

    calls = []
    base = acme_responder()

    def spy(system, user, schema):
        calls.append(user.split("\n", 1)[0].removeprefix("KEY:").strip())
        return base(system, user, schema)

    ref = case / "03_internal" / "internal_messages.txt"  # longer than the identify preview
    ref.write_text(ref.read_text(encoding="utf-8") + "\nAppended note line.\n", encoding="utf-8")
    run_case(case, AS_OF, FakeLLM(spy), cache)
    other = ("identify_case", "map_questions", "answer:", "chase:")
    extraction = [k for k in calls if not k.startswith(other)]
    assert extraction == ["03_internal/internal_messages.txt"]
    assert "identify_case" not in calls and "map_questions" not in calls


# --- reconciliation in the pipeline -------------------------------------------------------

def _recon_spy(inner, labels: dict[str, dict]):
    calls = []

    class Spy:
        model = inner.model

        def complete_json(self, system, user, schema):
            key = user.split("\n", 1)[0].removeprefix("KEY:").strip()
            if key.startswith("reconcile "):
                calls.append(key)
                topic = key.removeprefix("reconcile ")
                if topic in labels:
                    return schema.model_validate(labels[topic])
            return inner.complete_json(system, user, schema)

    return Spy(), calls


def test_pipeline_reconciles_multi_file_topics(tmp_path, acme_llm):
    chat, draft = ("03_internal/internal_messages#F1",
                   "04_knowledge/customer_specific_configuration_DRAFT#F4")
    spy, calls = _recon_spy(acme_llm(), {"retention_app_logs": {"pairs": [
        {"a": chat, "b": draft, "relation": "contradicts"}]}})
    snap = run_case(ACME_DIR, date(2026, 9, 25), spy, tmp_path / "c")
    assert "reconcile retention_app_logs" in calls and "reconcile storage_location" in calls
    assert "reconcile dpa" not in calls  # single-file topics are not reconciled
    v = snap.verdicts["retention_app_logs"]
    assert v.status == "CONFLICT" and v.rule == "sources contradict (reconciled)"


def test_pipeline_reconcile_failure_is_listed_and_topic_unchanged(tmp_path, acme_llm):
    base = run_case(ACME_DIR, date(2026, 9, 25), acme_llm(), tmp_path / "a")
    snap = run_case(ACME_DIR, date(2026, 9, 25),
                    acme_llm(fail_paths=("reconcile storage_location",)), tmp_path / "b")
    assert [p for p, _ in snap.failed_files] == ["reconcile:storage_location"]
    assert snap.verdicts["storage_location"] == base.verdicts["storage_location"]


# --- case-level answers derived from the run ----------------------------------------------

DPA_QUOTE = "DPA blocked until subprocessors and cross-border transfers are confirmed."


def _answer_keys_spy(inner):
    keys = []

    class Spy:
        model = inner.model

        def complete_json(self, system, user, schema):
            keys.append(user.split("\n", 1)[0])
            return inner.complete_json(system, user, schema)

    return Spy(), keys


def test_dependency_subject_gets_derived_verdict_and_deterministic_draft(tmp_path, acme_llm):
    spy, keys = _answer_keys_spy(acme_llm())
    snap = run_case(ACME_DIR, date(2026, 9, 25), spy, tmp_path / "c")
    v = snap.verdicts["dpa"]
    assert v.status == "PROPOSED" and v.rule == "derived from dependency" and not v.sendable
    assert v.value == "blocked until subprocessors, cross_border are confirmed"
    assert any(DPA_QUOTE in h and "internal_messages" in h for h in v.hints)
    a = next(a for a in snap.answers if a.req_id == "Q-020")
    assert "KEY: answer:Q-020" not in keys
    assert "blocked until subprocessors, cross_border are confirmed" in a.answer
    assert "subprocessors: UNKNOWN" in a.answer and "cross_border: UNKNOWN" in a.answer
    assert a.evidence_quotes == [DPA_QUOTE]
    # the depended-on topics still block
    assert {"subprocessors", "cross_border"} <= {b.item for b in snap.blockers}
    assert "dpa" not in {b.item for b in snap.blockers}


def test_approval_blockers_answered_from_blocker_list(tmp_path, acme_llm):
    spy, keys = _answer_keys_spy(acme_llm())
    snap = run_case(ACME_DIR, date(2026, 9, 25), spy, tmp_path / "c")
    items = sorted({b.item for b in snap.blockers})
    assert items and "approval_blockers" not in items
    v = snap.verdicts["approval_blockers"]
    assert v.status == "PROPOSED" and v.rule == "derived from blocker list"
    assert v.value.startswith(f"{len(items)} items may delay approval")
    a = next(a for a in snap.answers if a.req_id == "Q-022")
    assert "KEY: answer:Q-022" not in keys
    for item in items:
        assert item in a.answer
    assert DPA_QUOTE in a.evidence_quotes
    r = next(r for r in snap.required if r.req.id == "Q-022")
    assert r.status == "needs_confirmation"


# --- service description from the case header ---------------------------------------------

def test_service_fact_created_from_verbatim_header_quote(tmp_path, acme_llm):
    snap = run_case(ACME_DIR, date(2026, 9, 25), acme_llm(), tmp_path / "c")
    svc = [f for f in snap.facts if f.topic == "service_description"]
    assert len(svc) == 1
    f = svc[0]
    assert f.source_type == "case_header" and f.applies_to == "this_customer"
    assert f.file == "00_START_HERE.txt" and f.certainty == "stated"
    assert f.quote == "Service: AI operations assistant for customer-service workflows"
    v = snap.verdicts["service_description"]
    assert v.status == "PROPOSED"
    assert v.value == "AI operations assistant for customer-service workflows"


def test_service_fact_not_created_when_quote_not_verbatim(tmp_path, acme_llm):
    inner = acme_llm()

    class Spy:
        model = inner.model

        def complete_json(self, system, user, schema):
            out = inner.complete_json(system, user, schema)
            if user.startswith("KEY: identify_case"):
                out = out.model_copy(update={"service_quote": "Service: an invented line"})
            return out

    snap = run_case(ACME_DIR, date(2026, 9, 25), Spy(), tmp_path / "c")
    assert not [f for f in snap.facts if f.topic == "service_description"]


def test_submission_approval_answered_from_review_policy(tmp_path, acme_llm):
    spy, keys = _answer_keys_spy(acme_llm())
    snap = run_case(ACME_DIR, date(2026, 9, 25), spy, tmp_path / "c")
    v = snap.verdicts["submission_approval"]
    assert v.status == "PROPOSED" and v.rule == "derived from review policy"
    assert "KEY: answer:Q-023" not in keys
    a = next(a for a in snap.answers if a.req_id == "Q-023")
    assert a.answer.startswith("Every response needs human approval before external submission")
    lines = a.answer.splitlines()
    unconfirmed = next(l for l in lines if l.startswith("Confirm first"))
    missing = next(l for l in lines if l.startswith("Not answerable yet"))
    evidenced = next(l for l in lines if l.startswith("Evidence-backed"))
    assert "Q-012" in unconfirmed and "Q-010" in missing and "Q-021" in evidenced
    assert "Q-023" not in a.answer.split("\n", 1)[1]  # does not list itself
    assert "Q-024" not in a.answer                    # template rows are not listed


def test_open_decisions_become_caveats_end_to_end(tmp_path, acme_llm):
    snap = run_case(ACME_DIR, date(2026, 9, 25), acme_llm(), tmp_path / "c")
    affected = {t for d in snap.open_decisions for t in d.affected_topics}
    caveated = {t for t, v in snap.verdicts.items()
                if any(c.startswith("open decision (") for c in v.caveats)}
    assert caveated and caveated <= affected
    assert all(not snap.verdicts[t].sendable for t in caveated)
    # the decision reaches the draft through the code-appended caveats line
    a = next(a for a in snap.answers if a.req_id == "Q-012")
    assert "Caveats: " in a.answer and "open decision" in a.answer
