"""Procurement Navigator: a deal brief per case, read top to bottom.

Run:  uv run streamlit run app.py --server.address 0.0.0.0
Shows the latest saved run of a case (runs/<case>/*.json). "Re-run" processes the case folder
again (cached, so only changed files call the LLM), saves a new snapshot and shows what changed.
Tab 1 answers the brief's questions in order (what blocks, answers by readiness, who to chase);
tab 2 shows how the system knows (conflicts, evidence by topic, sources, decisions, run log).
All logic lives in procnav.view; this file only lays it out.
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
        st.caption("No verdict")
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

b = view.brief(s)
st.title(s.customer or s.case)
st.caption(f"{s.supplier} → {s.customer} · {b['milestone']} · as of {s.as_of:%b %d, %Y}")
st.markdown(f"#### {esc(b['sentence'])}")

changes = diff(prev, s) if prev is not None else None
if changes:
    st.info("**Since last run:** " + esc("; ".join(changes[:3]))
            + (f" (+{len(changes) - 3} more in How we know)" if len(changes) > 3 else ""))
elif changes is not None:
    st.caption("No changes since last run")

t_brief, t_how = st.tabs(["Deal brief", "How we know"])

# ---------------------------------------------------------------- deal brief

with t_brief:
    st.header(f"① What blocks the {b['milestone']}")
    acts = view.blocker_actions(s)
    if not acts:
        st.markdown("Nothing blocks it.")
    for a in acts:
        qs = f" · questions {', '.join(a['questions'])}" if a["questions"] else ""
        st.markdown(f"⛔ **{esc(a['title'])}**: {esc(a['why'])} → ask **{esc(a['owner'])}**"
                    f":gray[{esc(qs)}]")

    st.header("② Answers")
    show_tmpl = st.toggle("include repeated template controls", key="templates")
    rows = view.answer_rows(s, templates=show_tmpl)
    for key, (icon, label) in view.READINESS.items():
        group = [r for r in rows if r["readiness"] == key]
        st.subheader(f"{icon} {label} ({len(group)})")
        for r in group:
            risk = "⛔ " if r["blocked"] else ""
            with st.expander(f"{risk}{r['label_id']} — {r['question']}"):
                st.markdown(f"**Why:** {esc(r['why'])}")
                st.caption(f"Asked by {r['asked_by']} · `{r['source_file']}` · "
                           f"suggested owner: {r['owner']}"
                           + (" · blocks the milestone" if r["blocked"] else ""))
                if r["ask_quote"]:
                    st.markdown(f"> {esc(r['ask_quote'])}")
                st.markdown("**Draft answer** (copy, edit, send yourself)")
                st.code(r["answer"] or "(no draft)", language=None, wrap_lines=True)
                for q in r["quotes"]:
                    st.markdown(f"> {esc(q)}")
                for d in r["decisions"]:
                    st.markdown(f":orange[waiting on:] {esc(d)}")
                if r["stale_draft"]:
                    st.markdown(f":orange[existing questionnaire draft:] {esc(r['stale_draft'])}")
                for e in r["citation_errors"]:
                    st.markdown(f":red[citation issue:] {esc(e)}")
                st.caption("Topics: " + " · ".join(view.topic_phrase(s, t) for t in r["topics"])
                           + " — details in How we know")

    st.header("③ Who to chase")
    for c in view.chase_cards(s):
        who = f"{c['owner']} (customer decision)" if c["kind"] == "customer_decision" \
            else c["owner"]
        st.subheader(who)
        st.markdown("\n".join(f"- {esc(line)}" for line in c["lines"]) or "-")
        if c["message"]:
            with st.expander("Draft message"):
                st.code(c["message"], language=None, wrap_lines=True)
                for e in c["citation_errors"]:
                    st.markdown(f":red[citation issue:] {esc(e)}")

# ---------------------------------------------------------------- how we know

with t_how:
    st.header("Conflicts")
    conf = view.conflict_rows(s)
    if not conf:
        st.markdown("None: no two sources disagree.")
    for c in conf:
        st.markdown(f"⚠️ **{esc(c['title'])}** → ask **{esc(c['owner'])}**")
        st.markdown("\n".join(f"- “{esc(x['value'] or x['quote'])}” · `{x['where']}`"
                              for x in c["sides"]))

    st.header("Evidence by topic")
    order = list(STATUS_ICON)
    statuses = st.multiselect("Topic status", order, default=order, key="ev_status")
    topics = sorted((v for v in s.verdicts.values() if not statuses or v.status in statuses),
                    key=lambda v: (order.index(v.status), view.title(v.topic)))
    st.caption(f"{len(topics)} of {len(s.verdicts)} topics")
    for v in topics:
        value = f" — {v.value}" if v.value else ""
        with st.expander(f"{STATUS_ICON[v.status]} {view.title(v.topic)}: {v.status}{value}"):
            st.caption(f"topic id `{v.topic}`")
            verdict_body(s, v.topic)

    src_rows = view.source_rows(s)
    st.header(f"Sources ({len(src_rows)} files)")
    st.dataframe(
        [{"File": r["file"], "Source type": r["source_type"], "Rank": r["rank"],
          "Extracted": r["extracted"], "Applies to": r["applies_to"], "Note": r["note"]}
         for r in src_rows], hide_index=True, width="stretch")
    st.caption("Rank decides which facts can answer a topic: evidence > customer-specific > "
               "generic > other customer (hints only). Questionnaire and people files are "
               "parsed in code; guidance files hold nothing to extract.")

    st.header("Open decisions")
    decs = view.decision_rows(s)
    if not decs:
        st.markdown("None")
    for d in decs:
        st.markdown(f"- **{d['decider']} decides:** {esc(d['text'])}  \n"
                    f"  :gray[“{esc(d['quote'])}” — `{d['file']}`]")

    st.header("Run log")
    log = ([f"- changed: {esc(c)}" for c in changes or []]
           + [f"- failed: `{p}`: {esc(r)}" for p, r in s.failed_files]
           + [f"- skipped: `{p}`" for p in s.skipped_files]
           + [f"- dropped quote: {esc(d)}" for d in s.dropped_quotes]
           + [f"- new topic discovered: `{t}`" for t in s.new_topics])
    st.markdown("\n".join(log) if log else "Nothing to report")
