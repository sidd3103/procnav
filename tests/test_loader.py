from pathlib import Path

from procnav.loader import CaseFile, load_case, parse_people, parse_questionnaire

CASE = Path(__file__).resolve().parent.parent / "reonsiteinterview" / "CASE_1_Acme_Bank"


def _by_path(files):
    return {f.path: f for f in files}


def test_loads_14_files_and_skips_ds_store():
    files, skipped = load_case(CASE)
    assert len(files) == 14
    assert ".DS_Store" in skipped
    paths = [f.path for f in files]
    assert paths == sorted(paths)
    assert all("\\" not in p and not p.startswith("/") for p in paths)
    assert all(len(f.sha256) == 64 for f in files)


def test_binary_and_non_utf8_skipped(tmp_path):
    (tmp_path / "a.txt").write_text("hello")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.bin").write_bytes(b"\xff\xfe\x00\x81")
    files, skipped = load_case(tmp_path)
    assert [f.path for f in files] == ["a.txt"]
    assert skipped == ["sub/b.bin"]


def test_questionnaire_parses_60_rows():
    files, _ = load_case(CASE)
    q = [r for f in files if (r := parse_questionnaire(f)) is not None]
    assert len(q) == 1
    reqs = q[0]
    assert len(reqs) == 60
    by_id = {r.id: r for r in reqs}
    assert by_id["Q-004"].draft_answer == "30 days?"
    r = by_id["Q-001"]
    assert r.section == "Supplier" and r.from_customer is False
    assert r.topics == [] and r.requester is None and r.quote is None and r.group is None
    assert r.source_file.endswith("vendor_risk_questionnaire.csv")


def test_people_parses_7():
    files, _ = load_case(CASE)
    p = [r for f in files if (r := parse_people(f)) is not None]
    assert len(p) == 1
    people = p[0]
    assert len(people) == 7
    mina = next(x for x in people if x.name == "Mina")
    assert "subprocessors" in mina.knowledge


def test_non_questionnaire_csv_is_none():
    f = CaseFile(path="x.csv", text="a,b\n1,2\n", sha256="0" * 64)
    assert parse_questionnaire(f) is None
    assert parse_people(f) is None


def test_non_csv_is_none():
    f = CaseFile(path="x.txt", text="question\nfoo", sha256="0" * 64)
    assert parse_questionnaire(f) is None


def test_ragged_csv_row_does_not_crash():
    text = "id,section,question,draft_answer,status,note\nQ-1,S,What?,a,UNKNOWN,n,EXTRA,MORE\n"
    f = CaseFile(path="q.csv", text=text, sha256="0" * 64)
    reqs = parse_questionnaire(f)
    assert reqs is not None and reqs[0].id == "Q-1" and reqs[0].text == "What?"
    p = CaseFile(path="p.csv", text="person,likely_knowledge\nA,x,extra\n", sha256="0" * 64)
    assert parse_people(p)[0].name == "A"


def test_bom_csv_detected(tmp_path):
    (tmp_path / "q.csv").write_bytes("﻿id,question\nQ-1,Hi?\n".encode("utf-8"))
    files, skipped = load_case(tmp_path)
    assert skipped == []
    reqs = parse_questionnaire(files[0])
    assert reqs is not None and reqs[0].id == "Q-1"


def test_unreadable_file_skipped(tmp_path):
    (tmp_path / "ok.txt").write_text("x")
    bad = tmp_path / "bad.txt"
    bad.write_text("y")
    bad.chmod(0)
    try:
        files, skipped = load_case(tmp_path)
    finally:
        bad.chmod(0o644)
    assert [f.path for f in files] == ["ok.txt"]
    assert skipped == ["bad.txt"]
