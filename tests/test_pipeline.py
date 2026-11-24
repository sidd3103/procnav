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
    assert v["soc2"].status == "VERIFIED" and v["soc2"].sendable
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
    assert not any(i.startswith("soc2:") for i in all_items), "sendable topics are resolved"
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
    assert v["soc2"].status == "VERIFIED" and v["soc2"].sendable
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
