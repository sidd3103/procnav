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
    pen = next(r for r in rows if r["topic"] == "pen_test")
    assert pen["status"] == "VERIFIED" and pen["sendable"]
    soc2 = next(r for r in rows if r["topic"] == "soc2")
    assert soc2["status"] == "VERIFIED" and not soc2["sendable"]
    assert all(r["status"] in ("VERIFIED", "PROPOSED") for r in rows)
    assert len(rows) == view.summary_counts(snap)["known"]


def test_unresolved_rows_have_suggested_owner(snap):
    rows = view.unresolved_rows(snap)
    assert not any(r["topic"] == "pen_test" for r in rows)
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


# ---------------------------------------------------------------- deal brief

def _row(rows, qid):
    return next(r for r in rows if r["id"] == qid)


def test_readiness_labels():
    assert list(view.READINESS) == ["ready", "confirm", "missing"]
    assert view.READINESS["missing"] == ("❔", "Not in the files")


def test_answer_rows_group_by_readiness_with_risk_separate(snap):
    rows = view.answer_rows(snap)
    assert all(not r["template"] for r in rows)
    assert len(view.answer_rows(snap, templates=True)) == len(snap.required)
    q21 = _row(rows, "Q-021")
    assert q21["readiness"] == "ready" and q21["blocked"]  # answerable, yet a deal risk
    assert _row(rows, "Q-010")["readiness"] == "missing"
    assert _row(rows, "Q-012")["readiness"] == "confirm"
    assert _row(rows, "Q-020")["readiness"] == "ready"
    order = [list(view.READINESS).index(r["readiness"]) for r in rows]
    assert order == sorted(order)
    r2 = _row(rows, "01_customer/customer_correspondence#R2")
    assert r2["asked_by"] == "Elena, Privacy" and r2["label_id"] == "Email ask · Elena, Privacy"
    assert q21["label_id"] == "Q-021"
    a = next(a for a in snap.answers if a.req_id == "Q-021")
    assert q21["answer"] == a.answer and q21["quotes"] == a.evidence_quotes


def test_answer_why_uses_plain_titles(snap):
    rows = view.answer_rows(snap)
    assert _row(rows, "Q-010")["why"] == "Incident response: not in the files"
    q3 = _row(rows, "Q-003")["why"]
    assert q3.startswith("Where inference runs: not in the files")
    assert "Where data is stored: “AWS ca-central-1”" in q3 and "(unconfirmed)" in q3
    q21 = _row(rows, "Q-021")["why"]
    assert q21.startswith("Insurance: “") and "evidence" in q21 and "2026-10-01" in q21
    assert "_" not in _row(rows, "Q-004")["why"]
    assert "(+" in _row(rows, "Q-004")["why"]


def test_brief_sentence(snap):
    b = view.brief(snap)
    rows = view.answer_rows(snap)
    for k in view.READINESS:
        assert b[k] == sum(r["readiness"] == k for r in rows)
    assert b["milestone"] == "October production pilot"
    assert b["blockers"] == len({x.item for x in snap.blockers})
    assert f"{b['ready']} answers ready to draft" in b["sentence"]
    assert f"{b['blockers']} things block the October production pilot" in b["sentence"]
    assert "people to chase" in b["sentence"]


def test_blocker_actions_are_plain_sentences(snap):
    acts = {a["topic"]: a for a in view.blocker_actions(snap)}
    assert set(acts) == {b.item for b in snap.blockers}
    ins = acts["insurance"]
    assert ins["title"] == "Insurance" and "2026-10-01" in ins["why"]
    assert "before October production pilot" in ins["why"]
    sub = acts["subprocessors"]
    assert "DPA waits on it" in sub["why"] and sub["owner"]
    assert any(q.startswith("Email ask · ") for q in sub["questions"])
    assert len(sub["questions"]) == len(set(sub["questions"]))
    assert "Q-005" in sub["questions"] and not any(q.startswith("Q-02") and q > "Q-023"
                                                    for q in sub["questions"])
    assert "→" not in " ".join(a["why"] for a in acts.values())


def test_chase_cards_plain_items(snap):
    cards = view.chase_cards(snap)
    assert [c["owner"] for c in cards] == [s.owner for s in snap.next_steps]
    internal = [c for c in cards if c["kind"] == "internal"]
    lines = " ".join(l for c in internal for l in c["lines"])
    assert "_" not in lines.replace("other:", "")
    assert "Subprocessors: not in the files — blocks the October production pilot" in lines
    cust = next(c for c in cards if c["kind"] == "customer_decision")
    assert cust["lines"] and all(l.startswith("Decide: ") for l in cust["lines"])
    assert all("message" in c for c in cards)


def test_conflict_rows(snap):
    from procnav.models import Verdict
    v = snap.verdicts["model_providers"]
    c = Verdict(**{**v.model_dump(), "status": "CONFLICT", "value": None})
    s2 = snap.model_copy(update={"verdicts": {**snap.verdicts, "model_providers": c}})
    rows = view.conflict_rows(s2)
    mp = next(r for r in rows if r["topic"] == "model_providers")
    assert mp["title"] == "Model providers" and mp["sides"] and mp["owner"]
    assert all(set(x) == {"value", "where", "quote"} for x in mp["sides"])


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
