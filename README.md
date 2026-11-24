# Procurement Navigator (`procnav`)

An evidence ledger for vendor security and procurement reviews. Point it at a case folder (emails, chat exports, drafts, evidence files, a questionnaire CSV, a team CSV) and it answers five questions: **what's required, what's known, what's unresolved, what's blocking, and what to do next**. Every verdict traces back to a verbatim quote with its file and line.

**Core principle: the LLM judges what text means, code owns the status, and a human approves.** Nothing is sent automatically.

## Quick start

```bash
uv sync
cp .env.example .env   # or create .env with the two lines below
# OPENROUTER_API_KEY=…
# PROCNAV_MODEL=anthropic/claude-haiku-5.5

uv run pytest -q       # 402 tests, fake LLM, no network
uv run procnav run reonsiteinterview/CASE_1_Acme_Bank --as-of 2026-09-25
uv run procnav eval runs CASE_1_Acme_Bank --key evals/answer_key/CASE_1_Acme_Bank.json
uv run streamlit run app.py
```

`--as-of` sets "today", so the same files always give the same verdicts. It matters for expiry dates.

## How it works

1. **Load** (`loader.py`): reads files and hashes each one. The questionnaire and people CSVs are parsed in code, with header aliases (for example `Requirement` → question, `Vendor response` → draft answer).
2. **Identify the case** (`pipeline.py`, LLM): the customer, supplier and service, each with a supporting quote.
3. **Extract** (`extract.py`, LLM): one call per file. It returns facts (topic, value, certainty, source type, quote), requirements, dependencies and open decisions.
4. **Check** (`checks.py`, code): quotes must appear verbatim in the file. Code also applies hedge words, absences, expiry dates and source rank.
5. **Map topics** (LLM): questions and new facts are mapped onto seed topics (`topics.py`) or onto new `other:` topics.
6. **Reconcile** (`reconcile.py`, LLM): one call per topic whose facts come from two or more files. It labels each fact on-topic or off-topic, and each pair of values as the same, more specific or contradicting. Code applies the labels with **asymmetric trust**:
   - "Contradicts" and "off-topic" labels apply directly.
   - "Same" and "more specific" labels are capped at PROPOSED.
   - A missing or unclear label changes nothing.
7. **Resolve** (`resolve.py`, code):
   - Facts are ranked evidence > customer > generic > other customer, then by date.
   - Another customer's facts are shown only as hints.
   - Each topic gets one status: **VERIFIED / PROPOSED / CONFLICT / UNKNOWN**. *Sendable* means VERIFIED with no caveats.
8. **Blockers** (`blockers.py`, code): an item blocks only through one of three reasons:
   - an explicit dependency;
   - an expiry before a milestone;
   - a gap the customer explicitly asked about.
9. **Answers and chase messages** (`answers.py`, `chase.py`, LLM): drafts use only the facts allowed for that topic, and every citation ID is checked against those facts.
10. **Snapshot** (`runs/`), **diff** against the last run, **CLI** and the **Streamlit app**.

Each question gets a readiness of **ready / needs confirmation / unknown / blocked**. The worst of its topics wins, and a blocker overrides everything.

LLM calls run at temperature 0. They're cached in `.procnav_cache/` (override with `PROCNAV_CACHE`), keyed by the file hash, prompt version, model, customer and prompt text. A re-run with unchanged files makes 0 calls; adding or editing a file re-extracts only that file.

## Layout

| Path | What |
|---|---|
| `src/procnav/` | pipeline code (no customer, person or file names) |
| `app.py` | Streamlit UI: deal brief, then "How we know" |
| `reonsiteinterview/CASE_*` | case folders. Case 3 (Cedar Valley Health) is synthetic, written by us |
| `evals/answer_key/` | hand-labelled answer keys, one per case |
| `runs/` | saved snapshots (load offline) |
| `tests/` | pytest on a fake LLM, no network |
| `DECISIONS.md` | every design decision and why |
| `docs/interview_prep.md` | architecture diagram, tradeoffs, failure modes |
| `docs/demo_runbook.md` | 10-minute demo script |

## Evals

`procnav eval` is pure code. It checks each question's allowed readiness values and required or banned phrases, each topic's status and value, the files that must never be used, and who decides each open decision.

| Case | Score |
|---|---|
| Acme Bank (as of 2026-09-25) | 93 / 94 |
| Northstar Telecom (as of 2026-09-25) | 91 / 91 |
| Cedar Valley Health, synthetic (as of 2026-12-07) | 53 / 56 |

The known misses are listed in `docs/interview_prep.md` §6.

## Known limitations

- The LLM assigns each fact's source type, so it decides the fact's rank. In production, authority should come from source metadata.
- An expiry is only flagged within 30 days of the as-of date.
- ISO 27001 has no seed topic yet.
- Files go to the model whole, one call per file. That suits small case folders, not 300-page reports.
