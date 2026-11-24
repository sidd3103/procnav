import json
from datetime import date

from procnav import cli
from procnav.answers import DraftAnswer
from procnav.evals import AnswerKey, score
from procnav.models import Fact, Rank, Requirement, Verdict
from procnav.pipeline import Snapshot, save_snapshot
from procnav.requirements import RequirementResult


def _fact(fid, file, quote):
    return Fact(id=fid, file=file, line=1, topic="t", claim="c", value="v",
                applies_to="this_customer", certainty="stated", date=None, speaker=None,
                quote=quote, valid_until=None, rank=Rank.CUSTOMER, source_type="internal_chat",
                case="C", customer="X")


def _req(rid, text, from_customer=False):
    return Requirement(id=rid, text=text, requester=None, topics=["storage_location"],
                       source_file="q.csv", quote=None, draft_answer=None,
                       from_customer=from_customer, group=None)


def _verdict(topic, status, value=None, used=()):
    return Verdict(topic=topic, status=status, value=value, used=list(used), excluded=[],
                   hints=[], rule="r", caveats=[], sendable=False)


def _ans(rid, text):
    return DraftAnswer(req_id=rid, answer=text, cited=[], evidence_quotes=[], owner="o",
                       open_assumptions=[], status="needs_confirmation", citation_errors=[])


def snapshot():
    reqs = [_req("Q-001", "Where is data stored?"),
            _req("01_customer/mail#R3", "Confirm retention by data type.", from_customer=True)]
    return Snapshot(
        case="C", as_of=date(2026, 9, 25), customer="X",
        required=[RequirementResult(req=r, status="needs_confirmation", topic_verdicts=[],
                                    open_assumptions=[], stale_draft=None,
                                    depends_on_decisions=[]) for r in reqs],
        verdicts={"storage_location": _verdict("storage_location", "PROPOSED", "AWS Canada",
                                               ["a"]),
                  "retention_traces": _verdict("retention_traces", "PROPOSED", "redacted",
                                               ["p"])},
        facts=[_fact("a", "04_knowledge/config.md", "Storage in AWS Canada"),
               _fact("p", "03_internal/chat.txt", "payloads are redacted in traces")],
        blockers=[], next_steps=[],
        answers=[_ans("Q-001", "Data is stored in AWS Canada (proposed)."),
                 _ans("01_customer/mail#R3", "Log retention is 14 days; findings remain open.")],
        failed_files=[], skipped_files=[], dropped_quotes=[])


KEY = {
    "case": "C", "as_of": "2026-09-25",
    "answers_must_not_include": ["remain open"],
    "questions": [
        {"id": "Q-001", "statuses": ["needs_confirmation", "blocked"],
         "must_include": ["AWS Canada", ["proposed", "unconfirmed"]],
         "must_not_include": ["confirmed in writing"]},
        {"match": ["retention", "data type"], "statuses": ["needs_confirmation"],
         "must_include": ["retention"]},
    ],
    "topics": [
        {"topic": "storage_location", "statuses": ["PROPOSED", "VERIFIED"],
         "value_includes": [["Canada", "ca-central-1"]], "used_files_exclude": ["general"]},
        {"topic": "retention_traces", "statuses": ["PROPOSED", "UNKNOWN"],
         "used_quotes_exclude": ["payload"]},
        {"topic": "dpa", "statuses": ["PROPOSED"]},
    ],
}


def _by_name(checks):
    return {c.name: c for c in checks}


def test_score_pass_and_fail():
    checks = score(snapshot(), AnswerKey.model_validate(KEY))
    got = _by_name(checks)
    assert got["Q-001 status"].passed
    assert got["Q-001 includes 'AWS Canada'"].passed
    assert got["Q-001 includes one of ['proposed', 'unconfirmed']"].passed
    assert got["Q-001 excludes 'confirmed in writing'"].passed
    ask = "customer ask ['retention', 'data type']"
    assert got[f"{ask} status"].passed and got[f"{ask} includes 'retention'"].passed
    assert not got["answers exclude 'remain open'"].passed
    assert "01_customer/mail#R3" in got["answers exclude 'remain open'"].detail
    assert got["topic storage_location status"].passed
    assert got["topic storage_location value includes one of ['Canada', 'ca-central-1']"].passed
    assert got["topic storage_location not from 'general'"].passed
    assert not got["topic retention_traces quotes exclude 'payload'"].passed
    assert not got["topic dpa status"].passed  # no verdict -> fails
    assert len(checks) == len(got)


def test_missing_question_and_unmatched_ask_fail():
    key = AnswerKey.model_validate({"case": "C", "as_of": "2026-09-25", "questions": [
        {"id": "Q-999", "statuses": ["blocked"]},
        {"match": ["insurance"], "statuses": ["blocked"]}]})
    checks = score(snapshot(), key)
    assert [c.passed for c in checks] == [False, False]
    assert "not found" in checks[0].detail and "not found" in checks[1].detail


def test_cli_eval_prints_failures_and_score(tmp_path, capsys):
    save_snapshot(snapshot(), tmp_path / "runs")
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps(KEY))
    rc = cli.main(["eval", str(tmp_path / "runs"), "C", "--key", str(kp)])
    out = capsys.readouterr().out
    total = len(score(snapshot(), AnswerKey.model_validate(KEY)))
    assert f"passed {total - 3}/{total}" in out
    assert "FAIL answers exclude 'remain open'" in out
    assert rc == 1


def test_cli_eval_no_run_is_error(tmp_path, capsys):
    kp = tmp_path / "key.json"
    kp.write_text(json.dumps(KEY))
    rc = cli.main(["eval", str(tmp_path / "runs"), "C", "--key", str(kp)])
    assert rc != 0 and "no run" in capsys.readouterr().err
