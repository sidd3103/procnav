from __future__ import annotations

import re

from .models import Topic

PW, PC = "platform_wide", "per_customer"

# (id, scope, area, description, keywords)
_ROWS = [
    ("encryption_in_transit", PW, "security", "Encryption of data in transit (e.g. TLS).",
     ["encryption", "security", "tls", "transit", "cryptography", "infrastructure"]),
    ("encryption_at_rest", PW, "security",
     "Encryption of stored data at rest (databases, object storage, keys).",
     ["encryption", "security", "keys", "rest", "storage", "cryptography", "infrastructure"]),
    ("soc2", PW, "security", "SOC 2 attestation status and report availability.",
     ["soc2", "security", "evidence", "audit", "compliance", "attestation"]),
    ("pen_test", PW, "security", "Penetration tests: each test's date, scope and findings as reported, and the "
     "testing program and cadence.",
     ["pen test", "security", "evidence", "vulnerability", "testing"]),
    ("insurance", PW, "legal", "Cyber and liability insurance coverage.",
     ["insurance", "evidence", "coverage", "legal", "finance", "security"]),
    ("incident_response", PW, "security", "Incident response and breach notification process.",
     ["incident", "security", "breach", "response", "operations"]),
    ("iam_access", PW, "security", "Identity, access management and least privilege.",
     ["iam", "access", "security", "authentication", "sso", "infrastructure"]),
    ("model_providers", PC, "ai", "Which LLM providers and models process customer data.",
     ["model", "providers", "ai", "llm", "architecture", "engineering"]),
    ("model_fallback", PC, "ai", "Fallback model or provider used when the primary is unavailable.",
     ["fallback", "model", "ai", "providers", "architecture", "engineering"]),
    ("inference_location", PC, "ai", "Region or location where model inference runs.",
     ["inference", "region", "location", "ai", "architecture", "infrastructure"]),
    ("storage_location", PC, "resilience", "Region or location where customer data is stored.",
     ["storage", "region", "location", "infrastructure", "data residency"]),
    ("retention_app_logs", PC, "resilience", "Retention period for application logs.",
     ["retention", "logging", "logs", "infrastructure", "operations"]),
    ("retention_traces", PC, "resilience",
     "How long traces/telemetry are kept. Not whether payloads are captured or redacted.",
     ["retention", "traces", "logging", "observability", "infrastructure"]),
    ("retention_source_files", PC, "resilience", "Retention period for uploaded source files.",
     ["retention", "files", "storage", "infrastructure", "data"]),
    ("retention_prompts_responses", PC, "ai", "Retention of prompts and model responses.",
     ["retention", "prompts", "responses", "ai", "logging", "engineering"]),
    ("subprocessors", PC, "privacy", "Subprocessors that handle customer data.",
     ["subprocessors", "privacy", "vendors", "third parties", "data protection"]),
    ("cross_border", PC, "privacy", "Cross-border data transfers and transfer mechanisms.",
     ["cross-border", "transfers", "privacy", "legal", "data protection", "region"]),
    ("personal_info_scope", PC, "privacy", "Scope of personal information processed.",
     ["personal information", "pii", "privacy", "data", "scope", "data protection", "dpa",
      "legal"]),
    ("tool_actions", PC, "ai", "Actions and tools the AI system may invoke.",
     ["tools", "actions", "agent", "ai", "architecture", "engineering"]),
    ("human_oversight", PC, "ai",
     "Human review of AI outputs/actions inside the product (not sign-off on our "
     "questionnaire answers).",
     ["human", "oversight", "review", "ai", "governance", "product"]),
    ("rto_rpo", PC, "resilience", "Recovery time and recovery point objectives.",
     ["rto", "rpo", "recovery", "resilience", "disaster", "infrastructure", "backup"]),
    ("provider_training_retention", PC, "ai",
     "Whether model providers train on or retain customer data.",
     ["training", "provider", "retention", "ai", "privacy", "zero retention"]),
    ("dpa", PC, "legal", "Data processing agreement status and terms.",
     ["dpa", "legal", "privacy", "contract", "data protection", "agreement"]),
    ("evaluation", PC, "ai", "Evaluation and testing of AI quality and safety.",
     ["evaluation", "testing", "ai", "quality", "engineering", "benchmarks"]),
    ("prompt_injection", PC, "ai", "Defenses against prompt injection attacks.",
     ["prompt injection", "ai", "security", "guardrails", "engineering"]),
    ("provenance", PC, "ai", "Provenance and citation of sources behind AI outputs.",
     ["provenance", "citations", "sources", "ai", "traceability", "engineering"]),
    ("pen_test_remediation", PW, "security",
     "Whether findings were fixed after the test (retest, remediation evidence). The "
     "findings' status in the original report belongs to pen_test.",
     ["pen test", "remediation", "security", "findings", "evidence"]),
    ("trace_payload_capture", PC, "ai",
     "Whether observability/tracing captures or redacts request and response payloads.",
     ["traces", "payloads", "capture", "redaction", "observability", "architecture",
      "infrastructure"]),
    ("service_description", PC, "commercial",
     "The proposed service and its intended use for this customer.",
     ["service", "intended use", "proposal", "commercial", "deal", "context"]),
    ("submission_approval", PC, "legal",
     "Which of our questionnaire answers need internal sign-off before they are sent.",
     ["sign-off", "approval", "legal", "externally", "safe", "claims"]),
    ("approval_blockers", PC, "procurement",
     "What blocks or may delay approval of the case overall; answered from the blocker list.",
     ["blockers", "approval", "delay", "procurement", "deal", "process"]),
]

TITLES = {
    "encryption_in_transit": "Encryption in transit",
    "encryption_at_rest": "Encryption at rest",
    "soc2": "SOC 2 report",
    "pen_test": "Penetration test",
    "insurance": "Insurance",
    "incident_response": "Incident response",
    "iam_access": "Access control (IAM)",
    "model_providers": "Model providers",
    "model_fallback": "Fallback model",
    "inference_location": "Where inference runs",
    "storage_location": "Where data is stored",
    "retention_app_logs": "App log retention",
    "retention_traces": "Trace retention",
    "retention_source_files": "Source file retention",
    "retention_prompts_responses": "Prompt and response retention",
    "subprocessors": "Subprocessors",
    "cross_border": "Cross-border transfers",
    "personal_info_scope": "Personal information in scope",
    "tool_actions": "Tools and actions the AI can take",
    "human_oversight": "Human oversight",
    "rto_rpo": "Recovery objectives (RTO/RPO)",
    "provider_training_retention": "Provider training and retention",
    "dpa": "Data processing agreement (DPA)",
    "evaluation": "AI evaluation",
    "prompt_injection": "Prompt-injection defences",
    "provenance": "Sources and citations in outputs",
    "pen_test_remediation": "Pen test remediation",
    "trace_payload_capture": "Payload capture in traces",
    "service_description": "Proposed service",
    "submission_approval": "Sign-off before submission",
    "approval_blockers": "What may delay approval",
}

SEED: dict[str, Topic] = {
    id_: Topic(id=id_, description=desc, scope=scope, area=area, keywords=kw,
               title=TITLES[id_])
    for id_, scope, area, desc, kw in _ROWS
}


_SEED_NO_UNDERSCORE = {k.replace("_", ""): k for k in SEED}

# retired seed ids (and their spellings) that now belong to another seed topic;
# "current" is a property of the SOC 2 report (period + caveats), not its own subject
_ALIASES = {"soc2current": "soc2"}  # keyed without underscores


def get_topic(topic_id: str) -> Topic:
    """Seed topic for any spelling of a seed id ("SOC 2", "Retention_Traces"), else other:<slug>."""
    raw = topic_id.strip().lower().removeprefix("other:")
    slug = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    slug = _ALIASES.get(slug.replace("_", ""), slug)
    if slug in SEED:
        return SEED[slug]
    alias = _SEED_NO_UNDERSCORE.get(slug.replace("_", ""))
    if alias is not None:
        return SEED[alias]
    slug = slug or "unspecified"
    return Topic(id=f"other:{slug}", description=topic_id, scope="per_customer",
                 area="other", keywords=[], title=slug.replace("_", " ").capitalize())
