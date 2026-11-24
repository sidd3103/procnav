from conftest import ACME_DIR
from procnav import cli

SECTIONS = ["Required", "Known / evidenced", "Unresolved", "Blocking", "Next steps"]


def _run(tmp_path, capsys):
    rc = cli.main(["run", str(ACME_DIR), "--as-of", "2026-09-25",
                   "--cache", str(tmp_path / "cache"), "--runs", str(tmp_path / "runs")])
    return rc, capsys.readouterr().out


def test_cli_run_twice(tmp_path, capsys, monkeypatch, acme_llm):
    monkeypatch.setattr(cli.OpenAIClient, "from_env", classmethod(lambda cls: acme_llm()))
    rc, out = _run(tmp_path, capsys)
    assert rc == 0
    pos = [out.index(f"== {s}") for s in SECTIONS]
    assert pos == sorted(pos)
    assert "ready for Legal review" in out
    assert "Draft answers" in out
    assert "Changes since last run" in out
    assert "LLM calls: 0" not in out
    assert len(list((tmp_path / "runs" / "CASE_1_Acme_Bank").glob("*.json"))) == 1

    rc2, out2 = _run(tmp_path, capsys)
    assert rc2 == 0
    assert "No changes" in out2
    assert "LLM calls: 0" in out2
    assert len(list((tmp_path / "runs" / "CASE_1_Acme_Bank").glob("*.json"))) == 2


def test_cli_missing_key_is_clean_error(tmp_path, capsys, monkeypatch):
    def boom(cls):
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY")

    monkeypatch.setattr(cli.OpenAIClient, "from_env", classmethod(boom))
    rc = cli.main(["run", str(ACME_DIR), "--cache", str(tmp_path / "c"),
                   "--runs", str(tmp_path / "r")])
    err = capsys.readouterr().err
    assert rc != 0
    assert "OPENAI_API_KEY" in err


def test_cli_identify_failure_is_clean_error(tmp_path, capsys, monkeypatch):
    from procnav.llm import FakeLLM

    def fail(system, user, schema):
        raise ValueError("bad json")

    monkeypatch.setattr(cli.OpenAIClient, "from_env", classmethod(lambda cls: FakeLLM(fail)))
    rc = cli.main(["run", str(ACME_DIR), "--cache", str(tmp_path / "c"),
                   "--runs", str(tmp_path / "r")])
    err = capsys.readouterr().err
    assert rc != 0
    assert "identify_case failed" in err
    assert not (tmp_path / "r").exists()


def test_cli_bad_case_dir_is_clean_error(tmp_path, capsys, monkeypatch, acme_llm):
    monkeypatch.setattr(cli.OpenAIClient, "from_env", classmethod(lambda cls: acme_llm()))
    rc = cli.main(["run", str(tmp_path / "missing"), "--cache", str(tmp_path / "c"),
                   "--runs", str(tmp_path / "r")])
    err = capsys.readouterr().err
    assert rc != 0
    assert err.startswith("error:") and "missing" in err
    assert not (tmp_path / "r").exists()
