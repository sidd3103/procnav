# Decisions & Tradeoffs

## Framing
1. **Evidence ledger, not a questionnaire auto-filler.** The data is full of stale and unverified claims, so auto-filling means false statements to a bank.
   *Alt:* auto-fill. Faster demo, but it invents answers.
2. **The user is Jai (commercial lead). One screen, five sections: Required, Known, Unresolved, Blocking, Next steps.** These map one-to-one to the brief's minimum deliverables. Who-to-chase and safe-to-send sit inside them.
   *Alt:* a workbench for Parsa or Mina. Deeper per answer, worse for moving the deal.
3. **Required = every customer source, not just the questionnaire.** Some asks exist only in email, and there's no authoritative checklist.
4. **No logic for specific messages, people or customers.** Case content enters only through extraction, so Northstar runs with no code changes.
   *Alt:* special features such as a synthetic/real switch. Good demo moment, but overfitted.

## Architecture
5. **The LLM extracts, code decides, a human approves.** Verdicts are explainable, repeatable and testable, and the split is clear.
   *Alt:* one big LLM call (varies between runs, can invent answers) or pure rules (break on new cases).
6. **Python, Streamlit, uv. OpenAI SDK, with OpenRouter on the same client. JSON files, no database.** Fastest path in 5–6 hours.
7. **Model tiers.** A cheap model classifies files and maps topics and owners. A strong model extracts and drafts. Mapping is choosing from a list, while extraction needs nuance.
8. **Topic registry: a seed list that grows. New topics are per-customer by default, and only a human can mark one platform-wide.** Consistent names make conflict detection work, and the platform-wide setting is the risky one.
   *Alt:* fixed list (breaks on new cases) or free-form topics (conflicts go unnoticed).
9. **Prototype = full extraction per file, cached. Production adds connectors, retrieval and a shared fact store.** Fifteen small files don't need retrieval. The ledger and rules are the same either way.

## Trust rules
10. **Source ranking: evidence > customer-specific (draft or chat) > generic > other customer.** Newer wins within a rank. Disagreement with no tie-breaker becomes a visible CONFLICT, never a silent pick.
11. **Generic facts only count for platform-wide topics, and are capped at "needs confirmation".** Encryption is the same for every customer, but retention and model providers aren't.
    *Alt:* customer-specific sources only. Safest, but everything lands on Anthony.
12. **Prior-customer and generic facts become labelled hints, never answers.** The case says old answers are "starting points", not evidence.
13. **Problems with evidence demote the broad claim and keep the narrow fact.** "SOC 2 Type II, Jan-Dec 2025, final" stays VERIFIED, but its "2026 audit not final" caveat keeps it out of "ready" (one `soc2` topic; "current" is a property of the report, not a separate topic), and an expired certificate is never proof.
14. **Only VERIFIED facts become "ready for Legal review". Nothing is sent automatically.** "Don't guess in customer paperwork."
15. **Blocking ≠ unresolved.** An item blocks only through an explicit dependency, an evidence problem that hits a dated milestone, or a gap the customer explicitly asked about. Otherwise everything is blocking.
16. **The questionnaire's existing draft answers ("30 days?") are flagged as stale, never used as facts.** They have no source.

## Extraction
17. **One strong-model pass per whole file. Facts are atomic and carry free-text topic hints, which are mapped to the registry separately.** Whole-file reading keeps chat context, and separate mapping keeps the cache valid as topics grow.
18. **Code checks the LLM.** Quotes must appear word for word in the file, hedge words force "proposed", and bad output gets one retry and then a visible failure.
19. **Temperature 0, cache keyed by file hash + prompt version + model.** Stable runs, a clean "what changed" view, and only edited files are re-read.

20. **Provenance runs end to end: file version and line → fact → verdict (used and excluded, with the rule) → blocker → chase message → diff.** Every claim on screen is traceable. Chase messages must cite fact IDs, and code checks them, which closes the last gap where the LLM could invent something.

## Demo
21. **New information = editing files, then Re-run, with scripted edits ready.** The same path handles unscripted edits.
22. **"As-of" date is a setting.** The case's "today" is ambiguous, and insurance expiry depends on it.
23. **Every fact shows applicability, freshness, authority, provenance and scope.** These are the graders' stated criteria.

## Cross-file reasoning (added after the model audit, 2026-11-24)
24. **Sharper boundary: the LLM judges what text *means*; code owns *status*.** "The LLM never reasons" was never true, because extraction already decides applies_to, certainty and topic. Code fails at meaning: exact string comparison turned "AWS Canada" vs "Canada" into a CONFLICT and filed a payload-capture line under trace retention. Code keeps policy (ranking, freshness, expiry, VERIFIED/sendable, verbatim quotes), because that has to be deterministic and auditable.
25. **Per-topic reconciliation: one LLM call per topic whose facts come from ≥ 2 files.** It labels each fact on-topic (yes/no/unclear) and each cross-file pair same / more specific / compatible / contradicts / unclear. Code turns the labels into a `Reconciliation`, and `resolve_topic` decides what changes. Missing or unclear labels change nothing (fail closed). A failed call is listed and the topic resolves as before. About 6–10 calls per case, all cached.
    *Alt:* word containment ("Canada" ⊂ "AWS Canada"). Cheap, but "OpenAI" ⊂ "OpenAI and Anthropic" is a real contradiction.
26. **Asymmetric trust.** A label that makes the outcome *more* cautious applies directly: "contradicts" on equal-looking values raises a CONFLICT, even over evidence. A label that makes it *less* cautious is capped and shown: clearing a string conflict gives at most PROPOSED ("sources agree (reconciled)", with both wordings in the hints), and an off-topic fact is listed under excluded. The reconciler can never produce VERIFIED or sendable, so its worst mistake sends an item to human review.
27. **Case-level questions are answered by code from the run.** "What blocks the DPA" comes from the explicit dependency, "what will delay approval" from the blocker list, and both drafts are deterministic with the source quotes. The proposed service comes from the cover file via the identify step, kept only if its quote is verbatim in a file.
28. **Answer key + scorer.** Hand-labelled per case (`evals/answer_key/`): allowed statuses, must / must-not substrings per question, expected topic statuses, and regression checks (no "remain open", no prior-customer names, trace retention never built from a payload line). `procnav eval runs <case> --key …` prints the failures and passed/total.

## Model
29. **Claude Haiku 5.5 via OpenRouter (`PROCNAV_MODEL`).** On real runs of both cases it beat gpt-4.1-mini and Sonnet 5.5 on the review checklist at about ¼ of mini's cost. It was stable across runs except for one Northstar flip, which reconciliation now targets. The code default stays unchanged, and the model is a setting.
    *Answer-key scores after this change (Haiku 5.5, as-of 2026-09-25, fresh cache):* Acme 90/94, Northstar 89/91. Earlier saved runs from the previous pipeline score 75–80/94 and 77/91 (as-of 2026-11-24, so not strictly comparable). Still failing: Acme `model_fallback` is a CONFLICT, because the reconciler left "disable if residency requires" vs "OFF unless approved" unlabelled and failed closed. `pen_test_remediation` shows VERIFIED "open in original report" (accurate as a past state, but it reads as remediation being verified). One Northstar Q-008 draft drops the "Feb 2026" date.

## Demo hardening (2026-11-24)
30. **Topic descriptions decide where facts land, so they name the boundary.** `pen_test` covers each test's date, scope and findings as reported. `pen_test_remediation` covers only whether findings were fixed afterwards (retest, remediation evidence). The old wording filed "open in the original report" under remediation as VERIFIED. In Case 3 the reconciler also dropped every pen-test finding as "not about program and cadence". `soc2_current` was folded into `soc2`; an alias keeps old ids and LLM spellings landing on `soc2`.
31. **Prompt v7: an open decision is a choice between options that changes answers.** "Renewal pending" and "verify X" are not decisions, and a pending renewal becomes an absent fact on its topic, so it still shows as a caveat. The decider is the customer when a customer-side person says their team is deciding. A date in a file's heading qualifies the facts under it ("Feb 2026" test).
32. **Questionnaire and people CSVs are found by generic header aliases** (`Requirement` / `Control description` → question, `Ref` → id, `Vendor response` → draft answer, `Name` / `Expertise` → person / knowledge). An exact header always wins. Before this, a questionnaire without a `question` column was silently ignored.
33. **A synthetic third case tests adaptability: `CASE_3_SYNTHETIC_Cedar_Valley_Health`.** It is a fictional hospital (HIPAA) with a different folder layout, file names and questionnaire shape, and it is clearly marked as written by us. Its answer key is written from the planted facts. It runs with no case-specific code. What it exposed, and what is still open: ISO 27001 has no seed topic, so it lands in `other:`. The question maps to `soc2` only, and the expiry blocker has no owner. The chat's "BAA can't be signed until …" doesn't derive the `dpa` answer, because its subject isn't spelled as a topic.
    *Scores (Haiku 5.5):* Acme 93/94 and Northstar 91/91 at 2026-09-25; Case 3 53/56 at 2026-12-07 (48/54 on its first run, before the `pen_test` fix and one corrected key needle). The answer key also checks deciders now (synthetic vs real transcripts → customer).

## Deliberately not built
- **Request kinds / deliverables model.** Questions map to topics well enough. A second taxonomy doubles the mapping surface for little gain in the demo.
- **Passed-milestone detection.** The demo runs as-of 2026-09-25, before every milestone. It needs a policy decision (what a passed pilot date means) rather than code.
- **Draft checker (second LLM grading drafts).** The scorer plus code-side citation checks cover it. A checker LLM adds cost and its own errors.
- **Double extraction / voting.** Doubles cost. Instability is better handled where it shows (reconciliation, answer-key regressions).
- **LangGraph / agents.** The pipeline is a fixed DAG and has no branching for an agent to decide. The natural next stage is a chase-and-reply loop: send chase messages, ingest replies as new files, re-run.
