import json
import os
import re
from pathlib import Path, PurePosixPath

import pytest

from procnav.llm import FakeLLM

ACME_DIR = Path(__file__).resolve().parents[1] / "reonsiteinterview" / "CASE_1_Acme_Bank"
ACME_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "acme"
# Line appended by the edit test; when the chat text contains it, the edited fixture is used (R9).
INFERENCE_EDIT_LINE = "Anthony: inference runs in OpenAI US region"
# New evidence file written by the new-file test (fixture: pen_test_retest_letter.json).
RETEST_LETTER_PATH = "05_evidence/pen_test_retest_letter.txt"
RETEST_LETTER_TEXT = (
    "Penetration test retest letter - Jun 15, 2026.\n"
    "Retest of the Feb 2026 external penetration test.\n"
    "Both Medium findings (M-01, M-02) were remediated and confirmed closed on retest.\n"
)


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PROCNAV_LIVE") == "1":
        return
    skip = pytest.mark.skip(reason="live test; set PROCNAV_LIVE=1")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


def _load(name: str) -> dict:
    return json.loads((ACME_FIXTURES / name).read_text(encoding="utf-8"))


def acme_responder(fail_paths: tuple[str, ...] = ()):
    """FakeLLM callable dispatching on the `KEY:` line (R4) and the requested schema."""

    def respond(system: str, user: str, schema):
        key = user.split("\n", 1)[0].removeprefix("KEY:").strip()
        if key in fail_paths:
            raise RuntimeError(f"simulated LLM failure for {key}")
        if key == "identify_case":
            return schema.model_validate(_load("identify.json"))
        if key == "map_questions":
            return schema.model_validate(_load("map_questions.json"))
        if key == "map_topics":  # default: every new topic stays new
            ids = re.findall(r"^- (other:\S+?):", user, flags=re.MULTILINE)
            return schema.model_validate(
                {"mappings": [{"topic_id": i, "map_to": ["new"]} for i in ids]})
        if key.startswith("answer:"):
            ids = re.findall(r"^- id=(\S+) ", user, flags=re.MULTILINE)
            return schema.model_validate({
                "answer": "Draft answer based on the cited case facts; unconfirmed items are "
                          "labelled as such.",
                "cited_fact_ids": ids[:1],
            })
        if key.startswith("chase:"):
            ids = re.findall(r"^- id=(\S+) ", user, flags=re.MULTILINE)
            return schema.model_validate({
                "message": "Please review the open items listed and confirm what is needed; "
                           "unconfirmed items are labelled as such.",
                "cited_fact_ids": ids[:1],
            })
        stem = PurePosixPath(key).stem
        name = f"{stem}.json"
        if INFERENCE_EDIT_LINE in user and (ACME_FIXTURES / f"{stem}.edited.json").exists():
            name = f"{stem}.edited.json"
        return schema.model_validate(_load(name))

    return respond


@pytest.fixture
def acme_llm():
    """Factory: acme_llm(fail_paths=(...)) -> fresh FakeLLM backed by the Acme fixtures."""
    return lambda fail_paths=(): FakeLLM(acme_responder(tuple(fail_paths)))
