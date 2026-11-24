"""`procnav run <case_dir>`: run the pipeline and print the five sections as plain text.
`procnav eval <runs_dir> <case> --key <file>`: score the latest saved run against an answer key."""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

from pydantic import BaseModel

from .diff import diff
from .evals import AnswerKey, render_checks, score
from .llm import LLMClient, OpenAIClient
from . import view
from .pipeline import Snapshot, load_latest, run_case, save_snapshot


class CountingLLM:
    """Transparent proxy that counts complete_json calls."""

    def __init__(self, inner: LLMClient):
        self.inner = inner
        self.model = inner.model
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        return self.inner.complete_json(system, user, schema)


def _h(title: str) -> str:
    return f"\n== {title} =="


def render(s: Snapshot, changes: list[str] | None, llm_calls: int) -> str:
    out: list[str] = [f"Case: {s.case}  Customer: {s.customer}  Supplier: {s.supplier}  "
                      f"As of: {s.as_of.isoformat()}"]

    c = view.summary_counts(s)
    out.append(_h(f"Required ({c['required']})"))
    for r in s.required:
        req = r.req
        tag = " [template]" if (req.group or "").startswith("template:") else ""
        src = f" (asked by {req.requester})" if req.from_customer and req.requester else ""
        out.append(f"- {req.id}{tag} [{r.status}] {req.text}{src}")
        if r.topic_verdicts:
            out.append(f"    topics: {'; '.join(r.topic_verdicts)}")
        if r.stale_draft:
            out.append(f"    existing draft: {r.stale_draft}")
        if r.depends_on_decisions:
            out.append(f"    depends on open decision: {', '.join(r.depends_on_decisions)}")

    out.append(_h(f"Known / evidenced ({c['known']})"))
    for x in view.known_rows(s):
        mark = "  ** ready for Legal review **" if x["sendable"] else ""
        out.append(f"- {x['topic']}: {x['status']} {x['value']}{mark}")
        out.append(f"    rule: {x['rule']}; sources: {', '.join(x['used'])}")
        out += [f"    caveat: {cv}" for cv in x["caveats"]]

    out.append(_h(f"Unresolved ({c['unresolved']})"))
    for x in view.unresolved_rows(s):
        val = f" {x['value']}" if x["value"] else ""
        out.append(f"- {x['topic']}: {x['status']}{val} (suggested owner: {x['owner']})")
        out.append(f"    rule: {x['rule']}")
        out += [f"    caveat: {cv}" for cv in x["caveats"]]
        out += [f"    hint (not evidence): {h}" for h in x["hints"]]

    out.append(_h(f"Blocking ({c['blocking']})"))
    for b in view.blocking_rows(s):
        out.append(f"- {b['item']} [{', '.join(b['reasons'])}] (suggested owner: {b['owner']})")
        out += [f"    {ch}" for ch in b["chains"]]

    out.append(_h("Next steps"))
    for st in view.next_step_rows(s):
        out.append(f"- suggested owner: {st['owner']} ({st['label']}):")
        out += [f"    * {i}" for i in st["items"]]
        ch = st["chase"]
        if ch is not None:
            out.append("    draft message:")
            out += [f"      {ln}" for ln in ch["message"].splitlines() or [""]]
            if ch["cited"]:
                out.append(f"      cites: {', '.join(ch['cited'])}")
            out += [f"      citation issue: {e}" for e in ch["citation_errors"]]

    out.append(_h("Draft answers"))
    for a in s.answers:
        out.append(f"- {a.req_id} [{a.status}] suggested owner: {a.owner}")
        out.append(f"    {a.answer}")
        if a.cited:
            out.append(f"    cites: {', '.join(a.cited)}")
        if a.open_assumptions:
            out.append(f"    open: {'; '.join(a.open_assumptions)}")
        for e in a.citation_errors:
            out.append(f"    citation issue: {e}")

    out.append(_h("Changes since last run"))
    if changes is None:
        out.append("No previous run")
    else:
        out += [f"  {c}" for c in changes] if changes else ["No changes"]

    out.append(_h("Failures / skipped / dropped / new topics"))
    out += [f"- failed: {p}: {r}" for p, r in s.failed_files]
    out += [f"- skipped: {p}" for p in s.skipped_files]
    out += [f"- dropped quote: {d}" for d in s.dropped_quotes]
    out += [f"- new topic discovered: {t}" for t in s.new_topics]
    if not (s.failed_files or s.skipped_files or s.dropped_quotes or s.new_topics):
        out.append("None")

    out.append(f"\nLLM calls: {llm_calls}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="procnav")
    sub = ap.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run", help="process a case folder")
    run.add_argument("case_dir")
    run.add_argument("--as-of", type=date.fromisoformat, default=None,
                     help="YYYY-MM-DD (default: today)")
    run.add_argument("--cache", default=".procnav_cache")
    run.add_argument("--runs", default="runs")
    ev = sub.add_parser("eval", help="score the latest saved run against an answer key")
    ev.add_argument("runs_dir")
    ev.add_argument("case")
    ev.add_argument("--key", required=True)
    args = ap.parse_args(argv)

    if args.cmd == "eval":
        return _eval(args.runs_dir, args.case, args.key)

    try:
        llm = CountingLLM(OpenAIClient.from_env())
        snap = run_case(args.case_dir, args.as_of or date.today(), llm, args.cache)
    except (RuntimeError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    prev = load_latest(args.runs, snap.case)  # before saving the new run
    changes = diff(prev, snap) if prev is not None else None
    save_snapshot(snap, args.runs)
    print(render(snap, changes, llm.calls))
    return 0


def _eval(runs_dir: str, case: str, key_path: str) -> int:
    snap = load_latest(runs_dir, case)
    if snap is None:
        print(f"error: no run for {case} in {runs_dir}", file=sys.stderr)
        return 1
    key = AnswerKey.model_validate_json(Path(key_path).read_text(encoding="utf-8"))
    checks = score(snap, key)
    print(render_checks(checks))
    return 0 if all(c.passed for c in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
