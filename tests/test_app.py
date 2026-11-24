"""Smoke tests for the Streamlit app (app.py) via streamlit's AppTest; FakeLLM only."""
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ACME_DIR, acme_responder
from procnav.llm import FakeLLM, OpenAIClient
from procnav.pipeline import run_case, save_snapshot

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TABS = ["Questions", "Blocks launch", "Who to chase", "Evidence by topic"]
METRICS = ["Ready for Legal", "Drafted, needs sign-off", "Missing information", "Blocks launch"]
CASE = ACME_DIR.name


@pytest.fixture
def env(tmp_path, monkeypatch):
    runs, cases = tmp_path / "runs", tmp_path / "cases"
    (cases / CASE).mkdir(parents=True)
    monkeypatch.setenv("PROCNAV_RUNS", str(runs))
    monkeypatch.setenv("PROCNAV_CASES", str(cases))
    monkeypatch.setenv("PROCNAV_CACHE", str(tmp_path / "cache"))
    return runs, cases


def _app():
    return AppTest.from_file(APP, default_timeout=30)


def _texts(at):
    return " ".join(str(e.value) for kind in ("markdown", "caption", "info", "error", "success")
                    for e in getattr(at, kind))


@pytest.fixture
def saved(env, tmp_path):
    runs, _ = env
    snap = run_case(ACME_DIR, date(2026, 9, 25), FakeLLM(acme_responder()), tmp_path / "c")
    save_snapshot(snap, runs)
    return snap


def test_shows_saved_snapshot_with_all_tabs(saved):
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == TABS
    assert "ready for Legal review" in _texts(at)
    assert [m.label.split(" ", 1)[1] for m in at.metric] == METRICS
    n = len(saved.required)
    assert all(m.value.endswith(f"/ {n}") for m in at.metric)


def test_selecting_question_shows_its_draft_answer(saved):
    at = _app().run()
    at.selectbox(key="open_q").set_value("Q-021").run()
    assert not at.exception
    answer = next(a.answer for a in saved.answers if a.req_id == "Q-021")
    assert any(c.value == answer for c in at.code)
    assert "October production pilot" in _texts(at)


def test_filter_to_blocks_launch(saved):
    at = _app().run()
    at.multiselect(key="q_status").set_value(["Blocks launch"]).run()
    assert not at.exception
    df = at.dataframe[0].value
    assert len(df) and all("Blocks launch" in v for v in df["Status"])


def test_template_rows_hidden_until_toggled(saved):
    at = _app().run()
    ids = list(at.dataframe[0].value["ID"])
    assert "Q-024" not in ids
    at.toggle(key="q_templates").set_value(True).run()
    assert "Q-024" in list(at.dataframe[0].value["ID"])


def test_no_saved_run_shows_hint(env):
    at = _app().run()
    assert not at.exception
    assert "No saved run" in _texts(at)


def test_missing_key_is_clean_error(env, monkeypatch):
    def boom(cls):
        raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY")

    monkeypatch.setattr(OpenAIClient, "from_env", classmethod(boom))
    at = _app().run()
    at.button(key="rerun").click().run()
    assert not at.exception
    assert any("OPENAI_API_KEY" in e.value for e in at.error)


def test_rerun_saves_snapshot_and_shows_changes(env, monkeypatch):
    runs, _ = env
    monkeypatch.setattr(OpenAIClient, "from_env",
                        classmethod(lambda cls: FakeLLM(acme_responder())))
    at = _app().run()
    at.date_input(key="as_of").set_value(date(2026, 9, 25)).run()
    at.button(key="rerun").click().run()
    assert not at.exception
    assert len(list((runs / CASE).glob("*.json"))) == 1
    assert "LLM calls:" in _texts(at)
    at.button(key="rerun").click().run()
    assert len(list((runs / CASE).glob("*.json"))) == 2
    assert "No changes" in _texts(at)
    assert "LLM calls: 0" in _texts(at)
