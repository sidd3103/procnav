from procnav.checks import apply_hedges, find_quote

TEXT = "line one\napp logs yes;  observability\ntraces are kept\nlast"


def test_multiline_quote_maps_to_start_line():
    assert find_quote(TEXT, "app logs yes;  observability\ntraces") == 2


def test_single_line_and_first_line():
    assert find_quote(TEXT, "line one") == 1
    assert find_quote(TEXT, "last") == 4


def test_invented_and_empty_quote():
    assert find_quote(TEXT, "never said this") is None
    assert find_quote(TEXT, "") is None
    assert find_quote(TEXT, "   \n ") is None


def test_hedged_stated_becomes_proposed():
    assert apply_hedges("stated", "globally yes; for Acme probably disable") == "proposed"


def test_final_stays_stated():
    assert apply_hedges("stated", "SOC 2 Type II report issued. FINAL") == "stated"


def test_absent_unchanged_even_with_hedge():
    assert apply_hedges("absent", "is it so?") == "absent"


def test_other_certainties_unchanged():
    assert apply_hedges("uncertain", "TBD") == "uncertain"
    assert apply_hedges("proposed", "TBD") == "proposed"


def test_each_hedge():
    for q in ["30 days?", "TBD", "Intended for Q3", "DRAFT value", "not finalized yet",
              "it Should Be fine", "unless approved by legal", "Pending review"]:
        assert apply_hedges("stated", q) == "proposed", q


def test_word_boundaries():
    assert apply_hedges("stated", "the drafted policy is signed") == "stated"
    assert apply_hedges("stated", "suspending nothing, pendingly") == "stated"


def test_phrase_hedges_across_whitespace():
    assert apply_hedges("stated", "this should\nbe  fine") == "proposed"
    assert apply_hedges("stated", "Not\tFinalized") == "proposed"
    assert apply_hedges("stated", "unless\n approved") == "proposed"


# ---- R27 backstop: clear absence wording in a "stated" quote becomes "absent"

from procnav.checks import apply_absence  # noqa: E402


def test_absence_patterns_turn_stated_into_absent():
    for q in ["This file contains no remediation evidence.",
              "2026 audit is not yet final.",
              "The report is not final.",
              "Evidence has not yet been provided; it is not provided.",
              "A bridge letter is not available.",
              "Coverage is NOT CONFIRMED.",
              "We have no written evidence of this."]:
        assert apply_absence("stated", q) == "absent", q


def test_absence_only_rewrites_stated():
    q = "This file contains no remediation evidence."
    for c in ("proposed", "uncertain", "absent"):
        assert apply_absence(c, q) == c


def test_absence_leaves_plain_statements_alone():
    for q in ["No Critical or High findings.", "SOC 2 Type II, Jan-Dec 2025: FINAL.",
              "Evidence is attached.", "finalized and provided", "nothing contained here",
              "Notably final."]:
        assert apply_absence("stated", q) == "stated", q


# ---- case-insensitive matching for the fallback path only

from procnav.checks import match_quote  # noqa: E402


def test_match_quote_ignore_case_returns_line_and_original_span():
    text = "a\nNina: the team is deciding whether the pilot stays\n  synthetic-only."
    assert match_quote(text, "Whether the pilot stays synthetic-only.") is None
    hit = match_quote(text, "Whether the pilot stays synthetic-only.", ignore_case=True)
    assert hit == (2, "whether the pilot stays\n  synthetic-only.")


def test_find_quote_stays_case_sensitive():
    assert find_quote("whether it is", "Whether it is") is None
