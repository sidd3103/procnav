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
13. **Problems with evidence demote the broad claim and keep the narrow fact.** "SOC 2 2025 final" can be sent, "current assurance" can't, and an expired certificate is never proof.
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
