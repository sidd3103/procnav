"""Regression tests built from REAL model output (LIVE RUN 1, cached ExtractionOut dumps).

Each fixture in tests/fixtures/live/ holds the source file text and the verbatim validated
model output for it. They run through the normal post-processing and resolver with FakeLLM.
"""
import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from procnav.cache import JsonCache
from procnav.extract import CaseInfo, ExtractionOut, extract_file
from procnav.llm import FakeLLM
from procnav.loader import CaseFile
from procnav.resolve import resolve_all

LIVE = Path(__file__).resolve().parent / "fixtures" / "live"
AS_OF = date(2026, 11, 24)
INFO = CaseInfo(customer="Acme National Bank", supplier="Lantern AI Technologies Inc.", case="CASE_1_Acme_Bank")
NAMES = ["insurance_status", "pen_test_summary", "soc2_status",
         "customer_specific_configuration_DRAFT"]


def _extract(tmp_path):
    files, responses = [], {}
    for name in NAMES:
        doc = json.loads((LIVE / f"{name}.json").read_text(encoding="utf-8"))
        text = doc["text"]
        files.append(CaseFile(path=doc["path"], text=text,
                              sha256=hashlib.sha256(text.encode()).hexdigest()))
        responses[doc["path"]] = ExtractionOut.model_validate(doc["out"])
    llm, cache = FakeLLM(responses), JsonCache(tmp_path / "cache")
    return [extract_file(f, INFO, llm, cache) for f in files]


@pytest.fixture
def verdicts(tmp_path):
    efs = _extract(tmp_path)
    assert all(ef.failed is None and ef.dropped == [] for ef in efs)
    return resolve_all([f for ef in efs for f in ef.facts], AS_OF)


def test_insurance_verified_merged_with_expired_caveat(verdicts):
    v = verdicts["insurance"]
    assert v.status == "VERIFIED"
    assert v.value == "CAD 5,000,000; 2026-10-01"
    assert "expired 2026-10-01" in v.caveats
    assert not v.sendable


def test_pen_test_verified_merged(verdicts):
    v = verdicts["pen_test"]
    assert v.status == "VERIFIED" and v.sendable
    assert v.value == "external penetration test Feb 2026; no critical or high findings"
    assert len(v.used) == 2


def test_pen_test_remediation_not_sendable_with_no_evidence_caveat(verdicts):
    v = verdicts["pen_test_remediation"]
    assert v.status != "CONFLICT"
    assert not v.sendable
    assert any(c.startswith("no evidence:") for c in v.caveats)


def test_soc2_not_sendable_while_2026_audit_open(verdicts):
    # the model's legacy "soc2_current" topic is aliased onto soc2
    assert "soc2_current" not in verdicts
    v = verdicts["soc2"]
    assert v.status == "VERIFIED"  # the final 2025 report is untouched
    assert not v.sendable
    assert any(c.startswith("no evidence:") for c in v.caveats)


def test_retention_source_files_proposed_workspace_lifetime(verdicts):
    v = verdicts["retention_source_files"]
    assert v.status == "PROPOSED" and v.value == "workspace lifetime"
    assert any(r == "no value stated" for _, r in v.excluded)
    assert any(h.startswith("open: ") for h in v.hints)


def test_live_facts_postprocessed_fail_closed(tmp_path):
    by_topic = {}
    for ef in _extract(tmp_path):
        for f in ef.facts:
            by_topic.setdefault(f.topic, []).append(f)
    expiry = [f for f in by_topic["insurance"] if f.value == "2026-10-01"]
    assert [f.valid_until for f in expiry] == [date(2026, 10, 1)]
    assert {f.certainty for f in by_topic["soc2"]} == {"stated", "absent"}
    rem = {f.value: f.certainty for f in by_topic["pen_test_remediation"]}
    assert rem["no remediation evidence"] == "absent"
    assert [f.value for f in by_topic["retention_source_files"]] == ["workspace lifetime", None]
