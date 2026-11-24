import pytest

from procnav.topics import SEED, get_topic

PLATFORM = ["encryption_in_transit", "encryption_at_rest", "soc2", "pen_test", "insurance", "incident_response", "iam_access",
            "pen_test_remediation"]
PER_CUSTOMER = [
    "retention_app_logs", "model_providers", "inference_location", "subprocessors",
    "rto_rpo", "model_fallback", "storage_location", "retention_traces",
    "retention_source_files", "retention_prompts_responses", "cross_border",
    "personal_info_scope", "tool_actions", "human_oversight",
    "provider_training_retention", "dpa", "evaluation", "prompt_injection",
    "provenance", "soc2_current",
    "trace_payload_capture", "service_description", "submission_approval", "approval_blockers",
]


@pytest.mark.parametrize("tid", PLATFORM)
def test_platform_wide(tid):
    assert SEED[tid].scope == "platform_wide"


@pytest.mark.parametrize("tid", PER_CUSTOMER)
def test_per_customer(tid):
    assert SEED[tid].scope == "per_customer"


def test_seed_complete_and_well_formed():
    assert set(SEED) == set(PLATFORM + PER_CUSTOMER)
    assert "pilot_data_scope" not in SEED
    for tid, t in SEED.items():
        assert t.id == tid and t.description and t.area and t.keywords


def test_get_topic_known():
    assert get_topic("encryption_at_rest") is SEED["encryption_at_rest"]


def test_encryption_split():
    assert "encryption" not in SEED
    assert get_topic("encryption").id == "other:encryption"


def test_get_topic_unknown():
    t = get_topic("eu_ai_act")
    assert t.id == "other:eu_ai_act"
    assert t.scope == "per_customer"
    assert t.area == "other"


@pytest.mark.parametrize("raw,expected", [
    ("SOC2", "soc2"), ("SOC 2", "soc2"), ("soc-2", "soc2"), ("Retention_Traces", "retention_traces"),
    ("other:retention_traces", "retention_traces"), ("Pen Test", "pen_test"),
    ("", "other:unspecified"), ("  ", "other:unspecified"), ("other:", "other:unspecified"),
    ("EU AI Act", "other:eu_ai_act"), ("other:EU AI Act", "other:eu_ai_act"),
])
def test_get_topic_normalises_before_seed_match(raw, expected):
    assert get_topic(raw).id == expected


def test_retention_traces_excludes_payload_capture():
    d = SEED["retention_traces"].description.lower()
    assert "how long" in d and "not whether payloads are captured" in d


def test_human_oversight_excludes_questionnaire_signoff():
    d = SEED["human_oversight"].description.lower()
    assert "inside the product" in d and "questionnaire" in d


@pytest.mark.parametrize("tid,area", [
    ("trace_payload_capture", "ai"), ("service_description", "commercial"),
    ("submission_approval", "legal"), ("approval_blockers", "procurement"),
])
def test_new_case_level_topics(tid, area):
    assert SEED[tid].area == area


def test_every_seed_topic_has_a_plain_title():
    for tid, t in SEED.items():
        assert t.title and "_" not in t.title, tid
    assert SEED["inference_location"].title == "Where inference runs"


def test_unknown_topic_title_is_humanised():
    assert get_topic("other:eu_ai_act").title == "Eu ai act"
