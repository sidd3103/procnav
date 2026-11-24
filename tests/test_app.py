"""Smoke tests for the Streamlit app (app.py) via streamlit's AppTest; FakeLLM only."""
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ACME_DIR, acme_responder
from procnav.llm import FakeLLM, OpenAIClient
from procnav.pipeline import run_case, save_snapshot

APP = str(Path(__file__).resolve().parents[1] / "app.py")
TABS = ["Deal brief", "How we know"]
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


def _labels(at):
    return [e.label for e in at.expander]


def test_shows_deal_brief_and_how_we_know(saved):
    at = _app().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == TABS
    from procnav import view
    b = view.brief(saved)
    assert any(b["sentence"].split(".")[0].split("·")[0].strip() in m.value
               for m in at.markdown)
    heads = [h.value for h in at.header]
    assert heads[:3] == ["① What blocks the October production pilot", "② Answers",
                         "③ Who to chase"]
    assert {"Conflicts", "Evidence by topic", "Open decisions", "Run log"} <= set(heads)
    subs = [h.value for h in at.subheader]
    assert any(x.startswith("✅ Ready to draft (") for x in subs)
    assert any(x.startswith("❔ Not in the files (") for x in subs)


def test_blockers_read_as_plain_actions(saved):
    t = _texts(_app().run())
    assert "Insurance" in t and "before October production pilot" in t
    assert "→ ask" in t and "asked\\_hard\\_gap" not in t and "asked_hard_gap" not in t


def test_answer_opens_in_place_with_its_draft(saved):
    at = _app().run()
    assert any(lbl.startswith("⛔ Q-021 — ") for lbl in _labels(at))  # ready, yet blocks
    answer = next(a.answer for a in saved.answers if a.req_id == "Q-021")
    assert any(c.value == answer for c in at.code)


def test_template_rows_hidden_until_toggled(saved):
    at = _app().run()
    assert not any("Q-024 — " in lbl for lbl in _labels(at))
    at.toggle(key="templates").set_value(True).run()
    assert any("Q-024 — " in lbl for lbl in _labels(at))


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
    assert "No changes since last run" in _texts(at)
    assert "LLM calls: 0" in _texts(at)


def test_how_we_know_shows_sources_and_decisions(saved):
    at = _app().run()
    src = next(d.value for d in at.dataframe if "Rank" in d.value.columns)
    ranks = dict(zip(src["File"], src["Rank"]))
    assert ranks["04_knowledge/prior_customer_answers_CONTEXT_ONLY.md"] == "other customer"
    assert "Anthropic fallback" in _texts(at)
    assert any(lbl.startswith("✅ Insurance: VERIFIED") for lbl in _labels(at))


def test_as_of_defaults_to_latest_run(saved):
    at = _app().run()
    assert at.date_input(key="as_of").value == date(2026, 9, 25)
