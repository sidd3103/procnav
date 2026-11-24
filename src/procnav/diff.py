"""Human-readable changes between two snapshots. An empty list means nothing changed."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Verdict
    from .pipeline import Snapshot

NONE = "(none)"


def _label(v: Verdict | None) -> str:
    if v is None:
        return NONE
    return v.status + (f" {v.value}" if v.value else "")


def diff(old: Snapshot, new: Snapshot) -> list[str]:
    out: list[str] = []
    of_, nf = old.files, new.files
    out += [f"+ file {p}" for p in sorted(nf.keys() - of_.keys())]
    out += [f"- file {p}" for p in sorted(of_.keys() - nf.keys())]
    out += [f"~ file {p}" for p in sorted(of_.keys() & nf.keys()) if of_[p] != nf[p]]

    for t in sorted(old.verdicts.keys() | new.verdicts.keys()):
        a, b = _label(old.verdicts.get(t)), _label(new.verdicts.get(t))
        if a != b:
            out.append(f"{t}: {a} → {b}")

    ob = {b.item for b in old.blockers}
    nb = {b.item for b in new.blockers}
    out += [f"+ blocker {i}" for i in sorted(nb - ob)]
    out += [f"- blocker {i}" for i in sorted(ob - nb)]

    os_ = {r.req.id: r.status for r in old.required}
    ns = {r.req.id: r.status for r in new.required}
    for rid in sorted(os_.keys() | ns.keys()):
        a, b = os_.get(rid, NONE), ns.get(rid, NONE)
        if a != b:
            out.append(f"requirement {rid}: {a} → {b}")

    of = {p for p, _ in old.failed_files}
    out += [f"+ failed {p}: {r}" for p, r in sorted(new.failed_files) if p not in of]
    nfail = {p for p, _ in new.failed_files}
    out += [f"- failed {p}" for p in sorted(of - nfail)]
    return out
