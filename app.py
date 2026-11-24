"""Procurement Navigator: one shared, read-only page per case.

Run:  uv run streamlit run app.py --server.address 0.0.0.0
Shows the latest saved run of a case (runs/<case>/*.json). "Re-run" processes the case folder
again (cached, so only changed files call the LLM), saves a new snapshot and shows what changed.
All logic lives in procnav; this file only lays out what procnav.view returns.
"""
from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import streamlit as st

from procnav import view
from procnav.cli import CountingLLM
from procnav.diff import diff
from procnav.llm import OpenAIClient
from procnav.pipeline import Snapshot, run_case, save_snapshot

ROOT = Path(__file__).resolve().parent
CASES = Path(os.environ.get("PROCNAV_CASES") or ROOT / "reonsiteinterview")
RUNS = Path(os.environ.get("PROCNAV_RUNS") or ROOT / "runs")
CACHE = Path(os.environ.get("PROCNAV_CACHE") or ROOT / ".procnav_cache")

STATUS_ICON = {"VERIFIED": "✅", "PROPOSED": "🟡", "CONFLICT": "⚠️", "UNKNOWN": "❔"}


def esc(text: object) -> str:
    """Escape Markdown so quotes from case files render literally."""
    s = "" if text is None else str(text)
    for ch in "\\`*_{}[]<>()#+-.!|$~":
        s = s.replace(ch, "\\" + ch)
    return s


def verdict_body(s: Snapshot, topic: str) -> None:
    d = view.verdict_detail(s, topic)
    if d is None:
        st.caption(f"No verdict for {topic}")
        return
    st.markdown(f"**Value:** {esc(d['value']) or '—'} · **Status:** {d['status']} · "
                f"**Rule:** {esc(d['rule'])} · **Suggested owner:** {esc(d['owner'])}")
    if d["sendable"]:
        st.markdown(":green-badge[ready for Legal review]")
    for c in d["caveats"]:
        st.markdown(f":orange[caveat:] {esc(c)}")
    if d["used"]:
        st.markdown("**Used facts**")
        for u in d["used"]:
            st.markdown(f"- `{u['where']}` — “{esc(u['quote'])}”")
    if d["excluded"]:
        st.markdown("**Excluded facts**")
        for e in d["excluded"]:
            st.markdown(f"- `{e['where']}` — “{esc(e['quote'])}” · *{esc(e['reason'])}*")
    for h in d["hints"]:
        st.markdown(f"- :gray[hint (not evidence):] {esc(h)}")


def verdict_expander(s: Snapshot, topic: str) -> None:
    v = s.verdicts[topic]
    value = f" — {v.value}" if v.value else ""
    ready = " · ready for Legal review" if v.sendable else ""
    with st.expander(f"{STATUS_ICON.get(v.status, '')} {topic}: {v.status}{value}{ready}"):
        verdict_body(s, topic)


# ---------------------------------------------------------------- sidebar

st.set_page_config(page_title="Procurement Navigator", layout="wide")
st.sidebar.title("Procurement Navigator")
cases = view.list_cases(CASES)
if not cases:
    st.error(f"No case folders found under {CASES}")
    st.stop()
case = st.sidebar.selectbox("Case", cases, key="case")
s, prev = view.latest_two(RUNS, case)
# default to the date the saved run used, so a Re-run reproduces it (e.g. the demo's 2026-09-25)
as_of = st.sidebar.date_input("As of", value=s.as_of if s else date.today(), key="as_of")
calls: dict[str, int] = st.session_state.setdefault("llm_calls", {})

if st.sidebar.button("Re-run", key="rerun", type="primary"):
    try:
        llm = CountingLLM(OpenAIClient.from_env())
        with st.spinner(f"Processing {case}…"):
            snap = run_case(CASES / case, as_of, llm, CACHE)
        save_snapshot(snap, RUNS)
        calls[case] = llm.calls
        s, prev = view.latest_two(RUNS, case)
    except Exception as e:  # noqa: BLE001 - shown to the user, never a traceback
        st.error(f"Re-run failed: {e}")
if case in calls:
    st.sidebar.caption(f"LLM calls: {calls[case]} (last re-run)")

if s is None:
    st.info("No saved run for this case yet. Press Re-run in the sidebar.")
    st.stop()

# ---------------------------------------------------------------- header

st.title(s.customer or s.case)
st.caption(f"Case {s.case} · supplier {s.supplier} · as of {s.as_of.isoformat()}")
qcounts = view.question_counts(s)
total = len(s.required)
for col, status in zip(st.columns(4), reversed(view.STATUS_LABELS)):
    icon, label = view.STATUS_LABELS[status]
    col.metric(f"{icon} {label}", f"{qcounts[label]} / {total}")
st.caption("**Required:** every question and customer ask (Required tab) · "
           "**Known:** ✅ Ready for Legal + 🟡 Drafted, needs sign-off · "
           "**Unresolved:** ❔ Missing information + ⛔ Blocks launch · "
           "**Blocking:** Blocking tab · **Next:** Next steps tab")

with st.expander("What changed since last run", expanded=prev is not None):
    if prev is None:
        st.markdown("No previous run")
    else:
        changes = diff(prev, s)
        st.markdown("\n".join(f"- {esc(c)}" for c in changes) if changes else "No changes")

src_rows = view.source_rows(s)
with st.expander(f"Sources ({len(src_rows)} files): how each file was judged"):
    st.dataframe(
        [{"File": r["file"], "Source type": r["source_type"], "Rank": r["rank"],
          "Extracted": r["extracted"], "Applies to": r["applies_to"], "Note": r["note"]}
         for r in src_rows], hide_index=True, width="stretch")
    st.caption("Rank decides which facts can answer a topic: evidence > customer-specific > "
               "generic > other customer (hints only). Questionnaire and people files are "
               "parsed in code; guidance files hold nothing to extract.")

# ---------------------------------------------------------------- tabs

t_q, t_block, t_chase, t_ev = st.tabs(
    ["Required", "Blocking", "Next steps", "Known: evidence by topic"])

with t_q:
    rows = view.question_rows(s)
    labels = [label for _, label in view.STATUS_LABELS.values()]
    c1, c2, c3 = st.columns([3, 1, 2])
    pick = c1.multiselect("Status", labels, default=labels, key="q_status")
    owners = ["All"] + sorted({r["owner"] for r in rows})
    who = c2.selectbox("Suggested owner", owners, key="q_owner")
    text = c3.text_input("Search", key="q_text", placeholder="ID, question or reason")
    n_tmpl = sum(r["template"] for r in rows)
    show_tmpl = st.toggle(f"show template rows ({n_tmpl})", key="q_templates")
    shown = [r for r in view.filter_questions(rows, pick, who, text)
             if show_tmpl or not r["template"]]
    st.dataframe(
        [{"Status": f"{r['icon']} {r['label']}", "ID": r["id"], "Question": r["question"],
          "Reason": r["reason"], "Suggested owner": r["owner"]} for r in shown],
        hide_index=True, width="stretch")
    if not shown:
        st.info("No questions match these filters.")
    else:
        by_id = {r["id"]: r for r in shown}
        qid = st.selectbox("Open question", list(by_id), key="open_q",
                           format_func=lambda i: f"{by_id[i]['icon']} {i} — {by_id[i]['question']}")
        d = view.question_detail(s, qid)
        st.subheader(f"{d['icon']} {d['label']}")
        st.markdown(f"**Why:** {esc(d['reason'])}")
        st.caption(f"Asked by {d['asked_by']} · `{d['source_file']}` · "
                   f"suggested owner: {d['owner']}")
        if d["quote"]:
            st.markdown(f"> {esc(d['quote'])}")
        st.markdown("**Draft answer** (copy, edit, send yourself)")
        st.code(d["answer"] or "(no draft)", language=None, wrap_lines=True)
        if d["cited"]:
            st.caption(f"cites: {', '.join(d['cited'])}")
        for q in d["evidence_quotes"]:
            st.markdown(f"> {esc(q)}")
        if d["open_assumptions"]:
            st.markdown(f":orange[open:] {esc('; '.join(d['open_assumptions']))}")
        for e in d["citation_errors"]:
            st.markdown(f":red[citation issue:] {esc(e)}")
        if d["stale_draft"]:
            st.markdown(f":orange[existing questionnaire draft:] {esc(d['stale_draft'])}")
        for dec in d["decisions"]:
            st.markdown(f"Waiting on {dec['decider']} decision: {esc(dec['text'])}")
        for t in d["topics"]:
            if t in s.verdicts:
                verdict_expander(s, t)
            else:
                st.caption(f"No verdict for {t}")

with t_block:
    rows = view.blocking_rows(s)
    if not rows:
        st.markdown("Nothing blocking")
    for b in rows:
        with st.expander(f"⛔ {b['item']} [{', '.join(b['reasons'])}] · suggested owner: "
                         f"{b['owner']}"):
            for ch in b["chains"]:
                st.markdown(f"- {esc(ch)}")
            if b["questions"]:
                st.markdown(f"**Blocks questions:** {esc(', '.join(b['questions']))}")
            if b["item"] in s.verdicts:
                verdict_body(s, b["item"])

with t_chase:
    for row in view.next_step_rows(s):
        st.subheader(f"Suggested owner: {row['owner']}")
        st.caption(row["label"])
        st.markdown("\n".join(f"- {esc(i)}" for i in row["items"]))
        ch = row["chase"]
        if ch is not None:
            st.markdown("**Draft message**")
            st.code(ch["message"], language=None, wrap_lines=True)
            if ch["cited"]:
                st.caption(f"cites: {', '.join(ch['cited'])}")
            for e in ch["citation_errors"]:
                st.markdown(f":red[citation issue:] {esc(e)}")
    st.subheader("Open decisions")
    decs = view.decision_rows(s)
    if not decs:
        st.markdown("None")
    for d in decs:
        topics = f" · affects {esc(', '.join(d['topics']))}" if d["topics"] else ""
        st.markdown(f"- **{d['decider']} decides:** {esc(d['text'])}{topics}  \n"
                    f"  :gray[“{esc(d['quote'])}” — `{d['file']}`]")

with t_ev:
    order = list(STATUS_ICON)
    statuses = st.multiselect("Topic status", order, default=order, key="ev_status")
    topics = sorted((v for v in s.verdicts.values() if not statuses or v.status in statuses),
                    key=lambda v: (order.index(v.status), v.topic))
    st.caption(f"{len(topics)} of {len(s.verdicts)} topics")
    for v in topics:
        verdict_expander(s, v.topic)

# ---------------------------------------------------------------- footer

st.divider()
st.markdown("**Failures / skipped / dropped / new topics**")
foot = ([f"- failed: `{p}`: {esc(r)}" for p, r in s.failed_files]
        + [f"- skipped: `{p}`" for p in s.skipped_files]
        + [f"- dropped quote: {esc(d)}" for d in s.dropped_quotes]
        + [f"- new topic discovered: `{t}`" for t in s.new_topics])
st.markdown("\n".join(foot) if foot else "None")
