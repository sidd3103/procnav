"""LLM-facing extraction layer.

The LLM reads one case file and returns structured items (facts, requirements,
dependencies, open decisions). Everything after that is deterministic code:
quote verification, hedge downgrading, topic normalisation, ranking and ids.
The raw validated LLM output is cached; post-processing always re-runs.
"""
from __future__ import annotations

import re
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator, model_validator
from pydantic.json_schema import SkipJsonSchema

from .cache import JsonCache, cache_key
from .checks import apply_absence, apply_hedges, find_quote, match_quote
from .llm import LLMClient
from .loader import CaseFile
from .models import (
    AppliesTo,
    Certainty,
    Dependency,
    Fact,
    OpenDecision,
    Rank,
    Requirement,
    SourceType,
)
from .topics import SEED, get_topic

PROMPT_VERSION = "v5"

MIN_FALLBACK_WORDS = 3  # shorter texts would match almost any file by accident
NO_FACT_SOURCES = {"guidance", "people", "case_header", "questionnaire"}
IDENTIFY_PREVIEW_CHARS = 300


# ---------------------------------------------------------------- data types

class CaseInfo(BaseModel):
    customer: str
    supplier: str
    case: str = ""
    service: str = ""
    service_quote: str = ""


def _lower(v: object) -> object:
    return v.strip().lower() if isinstance(v, str) else v


class LLMOut(BaseModel):
    """Base for every schema the LLM fills: numbers are accepted where strings are expected."""
    model_config = ConfigDict(coerce_numbers_to_str=True)


# Every free-text field has a safe default (R25): one item missing a field must never fail
# the whole file. An item without a quote is dropped (and listed) in post-processing.

class FactOut(LLMOut):
    topic: str = ""
    claim: str = ""
    value: str | None = None
    applies_to: AppliesTo
    certainty: Certainty
    date: str | None = None
    speaker: str | None = None
    quote: str = ""
    valid_until: str | None = None

    _norm = field_validator("applies_to", "certainty", mode="before")(_lower)


class ReqOut(LLMOut):
    text: str = ""
    requester: str | None = None
    topics: list[str] = []
    quote: str = ""


class DepOut(LLMOut):
    kind: Literal["blocked_until", "milestone"]
    subject: str = ""
    depends_on_topics: list[str] = []
    date: str | None = None
    quote: str = ""

    _norm = field_validator("kind", mode="before")(_lower)


class DecOut(LLMOut):
    text: str = ""
    decider: Literal["customer", "internal"]
    affected_topics: list[str] = []
    quote: str = ""

    _norm = field_validator("decider", mode="before")(_lower)


class ExtractionOut(LLMOut):
    source_type: SourceType
    customer_scope: AppliesTo
    date: str | None = None
    is_draft: bool = False
    facts: list[FactOut] = []
    requirements: list[ReqOut] = []
    dependencies: list[DepOut] = []
    open_decisions: list[DecOut] = []
    # filled by code, never by the LLM (kept out of the JSON schema; kept in the cache)
    invalid_items: SkipJsonSchema[list[str]] = []

    _norm = field_validator("source_type", "customer_scope", mode="before")(_lower)

    @model_validator(mode="before")
    @classmethod
    def _drop_invalid_items(cls, data: Any) -> Any:
        """An item that still fails validation is dropped and noted; the file survives."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        notes = list(data.get("invalid_items") or [])
        for key, kind, item_cls in _ITEM_LISTS:
            items = data.get(key)
            if items is None:
                data[key] = []
                continue
            if not isinstance(items, list):
                continue
            kept = []
            for i, item in enumerate(items):
                try:
                    item_cls.model_validate(item)
                except ValidationError as e:
                    errs = "; ".join(
                        f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                        for err in e.errors()
                    )
                    notes.append(f"{kind}: invalid item {i} dropped: {errs}")
                    continue
                kept.append(item)
            data[key] = kept
        data["invalid_items"] = notes
        return data


_ITEM_LISTS = (("facts", "fact", FactOut), ("requirements", "requirement", ReqOut),
               ("dependencies", "dependency", DepOut), ("open_decisions", "open_decision", DecOut))


class CaseOut(LLMOut):
    customer: str
    supplier: str
    service: str = ""
    service_quote: str = ""


class MapItem(LLMOut):
    req_id: str
    topics: list[str] = []


class MapOut(LLMOut):
    mappings: list[MapItem] = []


class TopicMapItem(LLMOut):
    topic_id: str
    map_to: list[str] = []


class TopicMapOut(LLMOut):
    mappings: list[TopicMapItem] = []


class ExtractedFile(BaseModel):
    file: str
    source_type: SourceType | None
    facts: list[Fact] = []
    requirements: list[Requirement] = []
    dependencies: list[Dependency] = []
    open_decisions: list[OpenDecision] = []
    dropped: list[str] = []
    failed: str | None = None


# ---------------------------------------------------------------- prompts

def _seed_lines() -> str:
    return "\n".join(f"- {t.id}: {t.description}" for t in SEED.values())


SYSTEM_PROMPT = f"""You are a careful analyst for a software supplier answering a customer's \
vendor-security and procurement review. You read ONE file from a case folder and extract \
structured, verifiable items from it. Downstream code verifies every quote against the file \
and decides all statuses itself, so precision matters more than coverage: never guess, never \
infer values that are not written down, and never summarise across files.

FILE CLASSIFICATION
- source_type, choose one:
  - customer_request: correspondence written by the customer (or forwarded from them) that asks \
for information, states their requirements, or answers our questions.
  - questionnaire: a structured question list or form.
  - internal_chat: the supplier's own internal discussion (chat, email thread, meeting notes).
  - customer_config_draft: a draft architecture, configuration or deployment plan for this \
specific customer.
  - generic_doc: product or policy documentation that is not specific to this customer, or an \
older/standard reference document.
  - prior_customer: material from another customer's review (their answers, contract, config).
  - evidence: a formal artefact proving something (audit report, attestation, pen-test report, \
certificate, insurance certificate, scan result).
  - guidance: instructions about how to do the review or how to work (process notes, task \
descriptions); not a source of facts about the system.
  - people: a directory of people, roles or who-knows-what.
  - case_header: an overview or cover note describing the case itself.
- customer_scope: this_customer if the file is about the customer named in the user message; \
other_customer if it is about a different customer; generic if it applies to no customer in \
particular.
- date: the file's own date as YYYY-MM-DD if stated, else null.
- is_draft: true if the file presents itself as a draft, proposal or work in progress.

FACTS (statements about the supplier's system, contract or posture)
- Atomic: one topic and one value per fact. Split compound sentences into several facts.
- topic: use one of the seed topic ids below when it fits. Otherwise invent a short new \
snake_case id (e.g. backup_frequency). Do not force a fact into a seed topic it does not match.
- value: canonical and short, e.g. "30 days", "<provider> <region> exactly as written", "yes", \
"no". claim: one plain sentence restating the fact.
- value must not be more specific than the quote; never add a region, number or name that is \
not written.
- For on/off settings use 'enabled' or 'disabled' (with any condition kept in claim).
- Distinguish storage location (where data is kept at rest) from inference location (where \
models process requests); these are different topics even when stated in one sentence.
- Distinguish retention of app logs, traces, source files, and prompts/responses; each is its \
own topic.
- applies_to is relative to the customer named in the user message: this_customer if the \
statement is about that customer's deployment or contract; generic if it is a standard \
platform-wide statement or comes from a generic or older reference document; other_customer if \
it describes another customer's arrangement or answers (even if it appears in an internal chat).
- certainty:
  - stated: asserted as current fact without hedging.
  - proposed: planned, intended, recommended, drafted, "should be", "probably", pending approval.
  - uncertain: the writer says they do not know, asks a question, or contradicts themselves.
  - absent: the file explicitly says something does NOT exist, is missing, is not final or has \
not been provided. Record these; value describes what is missing (e.g. "no evidence"). Never \
tag such a sentence stated: a sentence saying evidence is missing is not evidence.
- ABSENCE EXAMPLES (use certainty "absent", value = what is missing):
  - "This document contains no X evidence." -> certainty "absent", value "no X evidence".
  - "The X report is not yet final." -> certainty "absent", value "X report not final".
  - "No X has been provided." -> certainty "absent", value "X not provided".
- value must be a real value. If the file only says a value is TBD, to be determined, unknown, \
pending or not decided, set value to null and record the open point as an open decision.
- date: when the statement was made or takes effect, as YYYY-MM-DD if known, else null.
- speaker: who said it (name or role as written), else null.
- valid_until: ONLY the written expiry date of a certificate, policy, attestation or insurance \
coverage ("expires", "valid until", "renewal due"), as YYYY-MM-DD; never an audit or report \
period (a report covering Jan-Dec does not expire in December). Else null. An expiry belongs \
in valid_until of the fact it limits; never put an expiry or renewal date in value. Example: \
"Coverage: <amount>. Certificate expires Mar 3, 2027." -> one fact with value "<amount>" and \
valid_until "2027-03-03".
- Files of type guidance, people, case_header or questionnaire contain no facts: return an \
empty facts list for them.

REQUIREMENTS
- Only from customer_request files: each distinct thing the customer asks us to provide, \
confirm or comply with. For every other source type return an empty requirements list.
- text: the request in one sentence. requester: who asked. topics: the topic ids it concerns \
(seed ids or new snake_case ids).

DEPENDENCIES
- blocked_until: "X is blocked until Y" or "X cannot proceed until Y". subject is X; \
depends_on_topics are the topic ids of Y.
- milestone: a dated launch, pilot, go-live or deadline. subject names it; depends_on_topics \
lists any topics it is conditional on.
- date: YYYY-MM-DD if a specific date is written, else null (do not invent a day for "October").

OPEN DECISIONS
- Something a stakeholder says is still undecided, not finalized, or awaiting a choice, where \
the outcome would change answers. text: the open question. decider: customer if the customer \
must decide, internal if the supplier must. affected_topics: topic ids whose answers depend \
on it.

QUOTES (all items)
- Every fact, requirement, dependency and open decision MUST include a verbatim quote from the \
file. Never leave quote empty; an item without a matching quote is discarded.
- quote must be copied VERBATIM from the file: an exact contiguous substring, same words, \
same spelling, same punctuation. Do not paraphrase, fix typos, join separate lines or add \
ellipses. Keep it short: the smallest span that supports the item. Items whose quote is not \
found in the file are discarded.

SEED TOPICS
{_seed_lines()}
"""

IDENTIFY_SYSTEM = """You identify the parties in a vendor-security / procurement review case \
folder. You are given each file's path and its opening text. Return:
- customer: the organisation whose review this is (the buyer asking the questions).
- supplier: the organisation answering the review (the vendor).
- service: the proposed service / product under review and its intended use, as written \
(empty if no file states it).
- service_quote: the exact line stating the service, copied verbatim from one file (empty if \
none).
Use the names as written in the files. Folder and file names are hints only; prefer content."""

MAP_SYSTEM = """You map questionnaire questions to topic ids. For each requirement id, list \
the topic ids the question is about (usually one, sometimes two or three). Prefer the given \
topic ids; if none fits, propose one short new snake_case id. Return every requirement id \
exactly as given."""

MAP_TOPICS_SYSTEM = """You map newly discovered topic ids onto a fixed list of known topic ids. \
Each new topic comes with example statements or requests that used it. For each new topic id, \
return in map_to the known topic id(s) whose subject is the same as the subject named by the \
new id:
- Judge by the subject the new id names. The examples are context only: a request may mention \
several subjects at once; map only the one the new id is about.
- A different wording, a synonym, or a narrower/broader name for the same subject IS the same \
subject (e.g. "data_encryption_stored" -> encryption_at_rest; "outbound_calls_by_agent" -> \
tool_actions; "malicious_document_input" -> prompt_injection).
- Use several known ids only when the new id bundles several known subjects (e.g. retention \
for several data types -> each matching retention topic).
- If no known topic covers the subject (e.g. a review meeting, a launch date, a team's \
approval), return ["new"].
Return every new topic id exactly as given."""


def _extract_user(f: CaseFile, info: CaseInfo) -> str:
    return (
        f"KEY: {f.path}\n"
        f"Customer under review: {info.customer}\n"
        f"Supplier (us): {info.supplier}\n"
        f"File path (a hint only; classify by content): {f.path}\n"
        f"----- FILE CONTENT START -----\n"
        f"{f.text}\n"
        f"----- FILE CONTENT END -----\n"
    )


# ---------------------------------------------------------------- helpers

def _complete(llm: LLMClient, system: str, user: str, schema: type[BaseModel]):
    """Call once, retry once on any error. Returns (result, None) or (None, error)."""
    err: Exception | None = None
    for _ in range(2):
        try:
            res = llm.complete_json(system, user, schema)
            return schema.model_validate(res.model_dump()), None
        except Exception as e:  # noqa: BLE001 - retry on anything, incl. validation
            err = e
    return None, err


def _short(e: Exception | None) -> str:
    msg = f"{type(e).__name__}: {e}"
    return msg if len(msg) <= 200 else msg[:197] + "..."


_DATE_FORMATS = ("%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y", "%Y/%m/%d")


def _iso(s: str | None) -> date | None:
    """Lenient date reader: ISO (date part of a datetime too) or a few written forms."""
    if not s:
        return None
    s = s.strip().removesuffix(".").strip()
    try:
        return date.fromisoformat(s[:10])
    except ValueError:
        pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


def _iso_str(s: str | None) -> str | None:
    """ISO string when readable; otherwise the raw text (the resolver treats it as undated)."""
    d = _iso(s)
    return d.isoformat() if d is not None else s


_EXPIRY_RE = re.compile(r"expir|valid until|valid through|valid to|renewal due", re.IGNORECASE)
_DATE_IN_TEXT_RE = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{4}/\d{2}/\d{2}"
    r"|[A-Za-z]{3,9}\.? \d{1,2}, \d{4}|\d{1,2} [A-Za-z]{3,9} \d{4}"
)
_PLACEHOLDERS = {"tbd", "tbc", "to be determined", "to be confirmed", "unknown", "n/a", "na",
                 "pending", "none", "not decided", "undecided"}


def _date_in(text: str, start: int = 0) -> date | None:
    """First readable date written in text at or after start."""
    for m in _DATE_IN_TEXT_RE.finditer(text, start):
        d = _iso(m.group(0).replace(".", ""))
        if d is not None:
            return d
    return None


def _expiry_from(value: str | None, quote: str) -> date | None:
    """R27(a): a date-valued fact whose quote speaks of expiry is an expiry (fails closed).
    The date written in the quote wins over the value's date."""
    value_date = _iso(value)
    m = _EXPIRY_RE.search(quote)
    if value_date is None or m is None:
        return None
    return _date_in(quote, m.start()) or _date_in(quote) or value_date


def _is_placeholder(value: str | None) -> bool:
    if value is None:
        return False
    v = " ".join(value.lower().split()).strip(" .:;!-")
    return v in _PLACEHOLDERS


def _topics(ids: list[str]) -> list[str]:
    out: list[str] = []
    for t in ids:
        tid = get_topic(t).id
        if tid not in out:
            out.append(tid)
    return out


def rank_for(source_type: str, applies_to: str) -> Rank:
    if applies_to == "other_customer" or source_type == "prior_customer":
        return Rank.OTHER_CUSTOMER
    if source_type == "evidence":
        return Rank.EVIDENCE
    if applies_to == "generic" or source_type == "generic_doc":
        return Rank.GENERIC
    return Rank.CUSTOMER


# ---------------------------------------------------------------- public API

def _case_info(co: CaseOut, case: str) -> CaseInfo:
    return CaseInfo(customer=co.customer, supplier=co.supplier, case=case,
                    service=co.service.strip(), service_quote=co.service_quote.strip())


def identify_case(files: list[CaseFile], llm: LLMClient, cache: JsonCache,
                  case: str = "") -> CaseInfo:
    # keyed on exactly what the prompt shows, so edits past the preview do not re-identify
    pairs = sorted((f.path, f.text[:IDENTIFY_PREVIEW_CHARS]) for f in files)
    key = cache_key("identify", PROMPT_VERSION, llm.model, IDENTIFY_SYSTEM, pairs)
    cached = cache.get(key)
    if cached is not None:
        try:
            return _case_info(CaseOut.model_validate(cached), case)
        except ValueError:
            pass
    parts = ["KEY: identify_case"]
    for f in sorted(files, key=lambda f: f.path):
        parts.append(f"=== {f.path} ===\n{f.text[:IDENTIFY_PREVIEW_CHARS]}")
    co, err = _complete(llm, IDENTIFY_SYSTEM, "\n".join(parts), CaseOut)
    if co is None:
        raise RuntimeError(f"identify_case failed: {_short(err)}")
    cache.put(key, co.model_dump())
    return _case_info(co, case)


def extract_file(f: CaseFile, info: CaseInfo, llm: LLMClient, cache: JsonCache) -> ExtractedFile:
    customer = " ".join(info.customer.casefold().split())
    key = cache_key(f.sha256, PROMPT_VERSION, llm.model, customer, SYSTEM_PROMPT)
    out: ExtractionOut | None = None
    cached = cache.get(key)
    if cached is not None:
        try:
            out = ExtractionOut.model_validate(cached)
        except ValueError:
            out = None
    if out is None:
        out, err = _complete(llm, SYSTEM_PROMPT, _extract_user(f, info), ExtractionOut)
        if out is None:
            return ExtractedFile(file=f.path, source_type=None, failed=_short(err))
        cache.put(key, out.model_dump())
    return _postprocess(f, info, out)


def _postprocess(f: CaseFile, info: CaseInfo, out: ExtractionOut) -> ExtractedFile:
    st = out.source_type
    stem = PurePosixPath(f.path).with_suffix("").as_posix()  # unique per file
    dropped: list[str] = list(out.invalid_items)
    force_other = st == "prior_customer" or out.customer_scope == "other_customer"

    def locate(kind: str, quote: str, label: str,
               fallbacks: tuple[str | None, ...] = ()) -> tuple[int, str] | None:
        """(line, quote actually used). An empty or unmatched quote falls back to the item's
        own text when the file contains it, ignoring whitespace and case; the stored quote is
        then the file's own span (R29). Otherwise the item is dropped."""
        if quote.strip():
            line = find_quote(f.text, quote)
            if line is not None:
                return line, quote
        for alt in fallbacks:
            if alt and len(alt.split()) >= MIN_FALLBACK_WORDS:
                hit = match_quote(f.text, alt, ignore_case=True)
                if hit is not None:
                    return hit
        if not quote.strip():
            dropped.append(f"{kind}: quote missing: {label!r}")
        else:
            dropped.append(f"{kind}: quote not found: {quote!r}")
        return None

    facts: list[Fact] = []
    if st not in NO_FACT_SOURCES:
        for fo in out.facts:
            hit = locate("fact", fo.quote, fo.claim or fo.topic, (fo.claim, fo.value))
            if hit is None:
                continue
            line, quote = hit
            applies: AppliesTo = "other_customer" if force_other else fo.applies_to
            valid_until = _iso(fo.valid_until)
            written = (fo.valid_until or "").strip()
            # an expiry we cannot read must fail closed, not vanish
            raw_expiry = written if written and valid_until is None else None
            if not written:  # an expiry date put in value instead of valid_until
                valid_until = _expiry_from(fo.value, quote)
            # a placeholder ("TBD", "unknown") is no value; the resolver lists it as open
            value = None if _is_placeholder(fo.value) else fo.value
            certainty = apply_absence(apply_hedges(fo.certainty, quote), quote)
            facts.append(Fact(
                id=f"{stem}#F{len(facts) + 1}", file=f.path, line=line,
                topic=get_topic(fo.topic).id, claim=fo.claim, value=value,
                applies_to=applies, certainty=certainty,
                date=_iso_str(fo.date or out.date), speaker=fo.speaker, quote=quote,
                valid_until=valid_until, rank=rank_for(st, applies),
                source_type=st, case=info.case, customer=info.customer,
                valid_until_raw=raw_expiry,
            ))

    reqs: list[Requirement] = []
    if st == "customer_request":
        for ro in out.requirements:
            hit = locate("requirement", ro.quote, ro.text, (ro.text,))
            if hit is None:
                continue
            reqs.append(Requirement(
                id=f"{stem}#R{len(reqs) + 1}", text=ro.text, requester=ro.requester,
                topics=_topics(ro.topics), source_file=f.path, quote=hit[1],
                draft_answer=None, from_customer=True, group=None,
            ))

    deps: list[Dependency] = []
    decs: list[OpenDecision] = []
    if st != "prior_customer":
        for do in out.dependencies:
            hit = locate("dependency", do.quote, do.subject, (do.subject,))
            if hit is None:
                continue
            deps.append(Dependency(
                id=f"{stem}#D{len(deps) + 1}", kind=do.kind, subject=do.subject,
                depends_on_topics=_topics(do.depends_on_topics), date=_iso(do.date),
                quote=hit[1], file=f.path,
            ))
        for oo in out.open_decisions:
            hit = locate("open_decision", oo.quote, oo.text, (oo.text,))
            if hit is None:
                continue
            decs.append(OpenDecision(
                id=f"{stem}#O{len(decs) + 1}", text=oo.text, decider=oo.decider,
                affected_topics=_topics(oo.affected_topics), quote=hit[1], file=f.path,
            ))

    return ExtractedFile(file=f.path, source_type=st, facts=facts, requirements=reqs,
                         dependencies=deps, open_decisions=decs, dropped=dropped)


def map_new_topics(
    items: dict[str, list[str]], llm: LLMClient, cache: JsonCache
) -> tuple[dict[str, list[str]], str | None]:
    """Map `other:*` topic ids (each with example texts) onto seed topic ids.

    Returns ({other id: [seed ids]}, failure reason or None). Only answers that are existing
    seed ids are kept; an id with no valid answer (incl. "new") is left out, so it stays
    `other:`. One batched, cached call; nothing to map makes no call.
    """
    if not items:
        return {}, None
    pairs = sorted((k, tuple(v[:2])) for k, v in items.items())
    key = cache_key("map_topics", PROMPT_VERSION, llm.model, MAP_TOPICS_SYSTEM, _seed_lines(),
                    pairs)
    out: TopicMapOut | None = None
    cached = cache.get(key)
    if cached is not None:
        try:
            out = TopicMapOut.model_validate(cached)
        except ValueError:
            out = None
    if out is None:
        lines = ["KEY: map_topics", "Known topics:", _seed_lines(), "", "New topics:"]
        for tid, examples in pairs:
            ex = " | ".join(f'"{e}"' for e in examples)
            lines.append(f"- {tid}: e.g. {ex}" if ex else f"- {tid}:")
        out, err = _complete(llm, MAP_TOPICS_SYSTEM, "\n".join(lines), TopicMapOut)
        if out is None:
            return {}, _short(err)
        cache.put(key, out.model_dump())
    result: dict[str, list[str]] = {}
    for m in out.mappings:
        tid = get_topic(m.topic_id).id
        if tid not in items:
            continue
        seeds = [s for s in _topics(m.map_to) if s in SEED]
        if seeds:
            result[tid] = seeds
    return result, None


def map_questions(reqs: list[Requirement], llm: LLMClient, cache: JsonCache) -> list[Requirement]:
    return map_questions_checked(reqs, llm, cache)[0]


def map_questions_checked(
    reqs: list[Requirement], llm: LLMClient, cache: JsonCache
) -> tuple[list[Requirement], str | None]:
    """Like map_questions, but also returns the failure reason (None on success).

    On failure the requirements come back unchanged (no topics), so callers can surface it.
    """
    if not reqs:
        return [], None
    # ids are included with the texts because the cached mapping is keyed by id
    key = cache_key("map", PROMPT_VERSION, llm.model, MAP_SYSTEM, [(r.id, r.text) for r in reqs])
    mo: MapOut | None = None
    cached = cache.get(key)
    if cached is not None:
        try:
            mo = MapOut.model_validate(cached)
        except ValueError:
            mo = None
    if mo is None:
        lines = ["KEY: map_questions", "Topics:", _seed_lines(), "", "Requirements:"]
        for r in reqs:
            ctx = f" [section: {r.section}]" if r.section else ""
            lines.append(f"- {r.id}: {r.text}{ctx}")
        mo, err = _complete(llm, MAP_SYSTEM, "\n".join(lines), MapOut)
        if mo is None:
            return [r.model_copy(deep=True) for r in reqs], _short(err)
        cache.put(key, mo.model_dump())
    by_id = {m.req_id: _topics(m.topics) for m in mo.mappings}
    return [
        r.model_copy(update={"topics": by_id[r.id]}, deep=True) if by_id.get(r.id)
        else r.model_copy(deep=True)
        for r in reqs
    ], None
