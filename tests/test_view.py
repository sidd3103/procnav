from datetime import date

import pytest

from conftest import ACME_DIR
from procnav import cli, view
from procnav.pipeline import run_case

AS_OF = date(2026, 9, 25)


@pytest.fixture(scope="module")
def snap(tmp_path_factory, request):
    from conftest import acme_responder
    from procnav.llm import FakeLLM
    return run_case(ACME_DIR, AS_OF, FakeLLM(acme_responder()),
                    tmp_path_factory.mktemp("cache"))


def test_list_cases(tmp_path):
    for name in ("CASE_B", "CASE_A", ".hidden"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.pdf").write_text("x")
    assert view.list_cases(tmp_path) == ["CASE_A", "CASE_B"]
    assert view.list_cases(tmp_path / "missing") == []


def test_summary_counts_match_cli_headers(snap):
    c = view.summary_counts(snap)
    out = cli.render(snap, None, 0)
    assert f"== Required ({c['required']}) ==" in out
    assert f"== Known / evidenced ({c['known']}) ==" in out
    assert f"== Unresolved ({c['unresolved']}) ==" in out
    assert f"== Blocking ({c['blocking']}) ==" in out
    assert c["required"] == len(snap.required) and c["blocking"] >= 3


def test_known_rows(snap):
    rows = view.known_rows(snap)
    soc2 = next(r for r in rows if r["topic"] == "soc2")
    assert soc2["status"] == "VERIFIED" and soc2["sendable"]
    assert all(r["status"] in ("VERIFIED", "PROPOSED") for r in rows)
    assert len(rows) == view.summary_counts(snap)["known"]


def test_unresolved_rows_have_suggested_owner(snap):
    rows = view.unresolved_rows(snap)
    assert not any(r["topic"] == "soc2" for r in rows)
    logs = next(r for r in rows if r["topic"] == "retention_app_logs")
    assert logs["owner"] == "Anthony"
    assert logs["value"] == "14 days"


def test_blocking_rows(snap):
    rows = view.blocking_rows(snap)
    assert {"subprocessors", "cross_border", "insurance"} <= {r["item"] for r in rows}
    ins = next(r for r in rows if r["item"] == "insurance")
    assert ins["reasons"] and ins["chains"] and ins["owner"] == "Sam"


def test_next_step_rows_pair_chase_drafts(snap):
    rows = view.next_step_rows(snap)
    assert [(r["owner"], r["kind"]) for r in rows] == [(s.owner, s.kind) for s in snap.next_steps]
    assert all(r["chase"] is not None for r in rows)
    assert rows[0]["chase"]["message"]


def test_verdict_and_fact_detail(snap):
    d = view.verdict_detail(snap, "soc2")
    assert d["status"] == "VERIFIED"
    used = d["used"][0]
    assert used["id"] == "05_evidence/soc2_status#F1"
    assert used["where"].startswith("05_evidence/soc2_status.txt:")
    assert used["quote"]
    fd = view.fact_detail(snap, used["id"])
    assert fd["quote"] == used["quote"]
    assert view.fact_detail(snap, "nope") is None
    # excluded facts carry their reason
    det = view.verdict_detail(snap, "retention_app_logs")
    assert all(e["reason"] for e in det["excluded"])


def test_cli_prints_suggested_owner_and_new_topics(snap):
    out = cli.render(snap.model_copy(update={"new_topics": ["other:launch_date"]}), None, 0)
    assert "suggested owner: Anthony" in out
    assert "new topic discovered: other:launch_date" in out


def test_latest_two_runs(tmp_path, snap):
    from procnav.pipeline import save_snapshot
    assert view.latest_two(tmp_path, snap.case) == (None, None)
    save_snapshot(snap, tmp_path)
    latest, prev = view.latest_two(tmp_path, snap.case)
    assert latest == snap and prev is None
    changed = snap.model_copy(update={"customer": "Changed"})
    save_snapshot(changed, tmp_path)
    (tmp_path / snap.case / "zzz-broken.json").write_text("{", encoding="utf-8")
    latest, prev = view.latest_two(tmp_path, snap.case)
    assert latest.customer == "Changed" and prev == snap


# ---------------------------------------------------------------- question-first view

BACKEND_STATUSES = ["blocked", "unknown", "needs_confirmation", "ready_for_review"]


def _row(rows, qid):
    return next(r for r in rows if r["id"] == qid)


def test_status_labels_cover_backend_statuses_in_severity_order():
    assert list(view.STATUS_LABELS) == BACKEND_STATUSES
    assert view.STATUS_LABELS["blocked"] == ("⛔", "Blocks launch")
    assert view.STATUS_LABELS["unknown"] == ("❔", "Missing information")
    assert view.STATUS_LABELS["needs_confirmation"] == ("🟡", "Drafted, needs sign-off")
    assert view.STATUS_LABELS["ready_for_review"] == ("✅", "Ready for Legal")


def test_question_counts_match_backend(snap):
    c = view.question_counts(snap)
    assert list(c) == [lbl for _, lbl in view.STATUS_LABELS.values()]
    assert sum(c.values()) == len(snap.required)
    for status, (_, lbl) in view.STATUS_LABELS.items():
        assert c[lbl] == sum(r.status == status for r in snap.required)


def test_question_rows_fields_and_order(snap):
    rows = view.question_rows(snap)
    assert len(rows) == len(snap.required)
    sev = [BACKEND_STATUSES.index(r["status"]) for r in rows]
    assert sev == sorted(sev)
    q21 = _row(rows, "Q-021")
    assert (q21["icon"], q21["label"]) == ("⛔", "Blocks launch")
    assert q21["owner"] == next(a.owner for a in snap.answers if a.req_id == "Q-021")
    assert q21["answer"] and q21["asked_by"] == "questionnaire" and not q21["template"]
    assert _row(rows, "Q-024")["template"]
    r2 = _row(rows, "01_customer/customer_correspondence#R2")
    assert r2["asked_by"] == "Elena, Privacy"
    # original order kept within a status
    blocked = [r["id"] for r in rows if r["status"] == "blocked"]
    assert blocked == [r.req.id for r in snap.required if r.status == "blocked"]


def test_reason_blocked_uses_blocker_chain(snap):
    reason = _row(view.question_rows(snap), "Q-021")["reason"]
    assert "insurance" in reason and "2026-10-01" in reason
    assert "October production pilot" in reason and "→" in reason


def test_reason_prefers_caveat_without_prefix(snap):
    q7 = _row(view.question_rows(snap), "Q-007")
    assert q7["label"] == view.STATUS_LABELS[q7["status"]][1]
    assert q7["reason"].startswith("The 2026 SOC 2 audit is not yet final.")
    assert "no evidence" not in q7["reason"]


def test_reason_unknown_and_more_count(snap):
    rows = view.question_rows(snap)
    assert _row(rows, "Q-010")["reason"] == "nobody has stated incident response"
    assert _row(rows, "Q-001")["reason"] == ("service description: AI operations assistant for "
                                             "customer-service workflows (unconfirmed)")
    q9 = _row(rows, "Q-009")["reason"]  # iam_access UNKNOWN; encryption topics are PROPOSED
    assert q9 == "nobody has stated iam access"
    q13 = _row(rows, "Q-026")["reason"]
    assert "(+" in q13 and "more)" in q13


def test_reason_needs_signoff_and_decision(snap):
    rows = view.question_rows(snap)
    assert _row(rows, "Q-016")["reason"] == (
        "human oversight: human confirms customer-facing actions (unconfirmed)")
    assert _row(rows, "Q-012")["reason"] == "model providers: OpenAI (unconfirmed) · waiting on decision"


def test_question_reason_edge_cases(snap):
    from procnav.models import Verdict
    r10 = next(r for r in snap.required if r.req.id == "Q-010")
    no_topics = r10.model_copy(update={"req": r10.req.model_copy(update={"topics": []})})
    assert view.question_reason(no_topics, snap) == "not linked to any topic"
    conflict = Verdict(topic="incident_response", status="CONFLICT", value=None, used=[],
                       excluded=[], hints=[], rule="r", caveats=[], sendable=False)
    s2 = snap.model_copy(update={"verdicts": {**snap.verdicts, "incident_response": conflict}})
    assert view.question_reason(r10, s2) == "sources disagree on incident response"
    ready = next(r for r in snap.required if r.status == "ready_for_review")
    assert view.question_reason(ready, snap).startswith("evidence: soc2 = SOC 2 Type II")


def test_filter_questions(snap):
    rows = view.question_rows(snap)
    only = view.filter_questions(rows, ["Blocks launch"], "All", "")
    assert only and all(r["status"] == "blocked" for r in only)
    assert view.filter_questions(rows, [], "All", "") == rows
    mina = view.filter_questions(rows, [], "Mina", "")
    assert mina and all(r["owner"] == "Mina" for r in mina)
    hit = view.filter_questions(rows, [], "All", "INSURANCE EVIDENCE")
    assert [r["id"] for r in hit] == ["Q-021"]
    assert view.filter_questions(rows, [], "All", "q-021")[0]["id"] == "Q-021"
    assert view.filter_questions(rows, [], "All", "pilot")  # matches reason text


def test_question_detail(snap):
    d = view.question_detail(snap, "Q-021")
    a = next(a for a in snap.answers if a.req_id == "Q-021")
    assert d["id"] == "Q-021" and d["label"] == "Blocks launch"
    assert d["answer"] == a.answer and d["cited"] == a.cited
    assert d["evidence_quotes"] == a.evidence_quotes
    assert d["open_assumptions"] == a.open_assumptions
    assert d["citation_errors"] == a.citation_errors
    assert d["topics"] == ["insurance"]
    assert "stale_draft" in d and "quote" in d and "source_file" in d
    r2 = view.question_detail(snap, "01_customer/customer_correspondence#R2")
    assert r2["decisions"] and r2["decisions"][0]["text"].startswith("Does the pilot")
    assert view.question_detail(snap, "nope") is None


def test_blocking_rows_list_blocked_questions(snap):
    ins = next(r for r in view.blocking_rows(snap) if r["item"] == "insurance")
    assert {"Q-021", "Q-024"} <= set(ins["questions"])


# --- sources and open decisions -----------------------------------------------------------

def test_source_rows_show_how_each_file_was_judged(snap):
    rows = {r["file"]: r for r in view.source_rows(snap)}
    assert set(rows) == set(snap.files)
    prior = rows["04_knowledge/prior_customer_answers_CONTEXT_ONLY.md"]
    assert prior["rank"] == "other customer" and prior["facts"] > 0
    assert "other_customer" in prior["applies_to"]
    assert rows["05_evidence/insurance_status.txt"]["rank"] == "evidence"
    mail = rows["01_customer/customer_correspondence.txt"]
    assert "ask" in mail["extracted"] and "milestone" in mail["extracted"]
    assert "facts" in rows["03_internal/internal_messages.txt"]["extracted"]
    assert rows["06_reference/customer_review_guidance.txt"]["extracted"] == "nothing"
    q = rows["02_questionnaires/vendor_risk_questionnaire.csv"]
    assert q["facts"] == 0 and q["source_type"] == "—"
    ranks = [view.RANK_ORDER.index(r["rank"]) for r in view.source_rows(snap)]
    assert ranks == sorted(ranks)  # evidence first, files without facts last


def test_source_rows_mark_failed_files(snap):
    s = snap.model_copy(update={"failed_files": [("03_internal/internal_messages.txt", "boom")]})
    row = next(r for r in view.source_rows(s) if r["file"] == "03_internal/internal_messages.txt")
    assert row["note"] == "failed: boom"


def test_decision_rows(snap):
    rows = view.decision_rows(snap)
    assert len(rows) == len(snap.open_decisions) > 0
    r = rows[0]
    assert set(r) >= {"text", "decider", "topics", "quote", "file"}
    assert r["decider"] in {"customer", "internal"}
