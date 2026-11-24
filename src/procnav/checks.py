from __future__ import annotations

import re

from .models import Certainty

_WORD_HEDGES = ["probably", "tbd", "intended", "draft", "pending"]
_PHRASE_HEDGES = ["should be", "not finalized", "unless approved"]
_WORD_RE = re.compile(r"\b(?:" + "|".join(_WORD_HEDGES) + r")\b", re.IGNORECASE)


def _normalise(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to one space and strip; return text and a map
    from each normalised index to its offset in the original."""
    out: list[str] = []
    idx: list[int] = []
    in_ws = True  # leading whitespace is dropped
    for i, ch in enumerate(text):
        if ch.isspace():
            if not in_ws:
                out.append(" ")
                idx.append(i)
                in_ws = True
        else:
            out.append(ch)
            idx.append(i)
            in_ws = False
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def match_quote(text: str, quote: str, ignore_case: bool = False) -> tuple[int, str] | None:
    """(1-based start line, the matching span of the ORIGINAL text) or None.

    Whitespace runs are treated as one space; case is ignored only when asked."""
    nq, _ = _normalise(quote)
    if not nq:
        return None
    nt, idx = _normalise(text)
    m = re.search(re.escape(nq), nt, re.IGNORECASE if ignore_case else 0)
    if m is None:
        return None
    start, end = idx[m.start()], idx[m.end() - 1] + 1
    return text.count("\n", 0, start) + 1, text[start:end]


def find_quote(text: str, quote: str) -> int | None:
    hit = match_quote(text, quote)
    return hit[0] if hit else None


def apply_hedges(certainty: Certainty, quote: str) -> Certainty:
    if certainty != "stated":
        return certainty
    low = " ".join(quote.lower().split())
    if "?" in quote or _WORD_RE.search(quote) or any(p in low for p in _PHRASE_HEDGES):
        return "proposed"
    return certainty


# Backstop for absences the model tagged "stated" ("contains no X evidence", "X is not yet
# final"): such a quote asserts that something is missing, so it must not become a candidate.
_ABSENCE_RE = re.compile(
    r"\bno\b.*\bevidence\b"
    r"|\bnot (?:yet )?(?:final|finalized|provided|available|confirmed)\b"
    r"|\bcontains no\b",
    re.IGNORECASE,
)


def apply_absence(certainty: Certainty, quote: str) -> Certainty:
    if certainty != "stated":
        return certainty
    low = " ".join(quote.split())
    return "absent" if _ABSENCE_RE.search(low) else certainty
