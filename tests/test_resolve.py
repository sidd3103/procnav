from datetime import date

from procnav.models import Fact, Rank
from procnav.resolve import normalize_value, resolve_all, resolve_topic

AS_OF = date(2026, 11, 24)
_n = 0


def F(topic, value, rank=Rank.CUSTOMER, certainty="stated", fdate=None, valid_until=None,
      applies="this_customer", file="f.md", claim=None, fid=None, line=1):
    global _n
    _n += 1
    return Fact(
        id=fid or f"f{_n}", file=file, line=line, topic=topic, claim=claim or f"{topic} claim",
        value=value, applies_to=applies, certainty=certainty, date=fdate, speaker=None,
        quote="q", valid_until=valid_until, rank=rank, source_type="internal_chat",
        case="c", customer="Acme",
    )


def test_normalize_value():
    assert normalize_value("  14D  ") == "14 days"
    assert normalize_value("14d  for  Logs") == "14 days for logs"
    assert normalize_value("30 days") == "30 days"


def test_retention_app_logs_acme_mix():
    chat = F("retention_app_logs", "14d", fid="chat")
    draft = F("retention_app_logs", "14 days", certainty="proposed", fid="draft")
    gen = F("retention_app_logs", "30 days", Rank.GENERIC, applies="generic", file="gen.md", fid="gen")
    mer = F("retention_app_logs", "30 days", Rank.OTHER_CUSTOMER, applies="other_customer",
            file="mer.md", fid="mer")
    v = resolve_topic("retention_app_logs", [chat, draft, gen, mer], AS_OF)
    assert v.status == "PROPOSED"
    assert v.value == "14 days"
    assert v.used == ["chat", "draft"]
    assert ("gen", "generic source on per-customer topic") in v.excluded
    assert ("mer", "other customer") in v.excluded
    assert "other customer: 30 days (mer.md)" in v.hints
    assert "generic: 30 days (gen.md)" in v.hints
    assert not v.sendable
    assert v.rule == "unverified source"


def test_value_is_longest_original_then_lexicographic():
    fs = [F("retention_app_logs", "14d"), F("retention_app_logs", "14 days")]
    assert resolve_topic("retention_app_logs", fs, AS_OF).value == "14 days"
    fs = [F("retention_app_logs", "14 Days"), F("retention_app_logs", "14 days")]
    assert resolve_topic("retention_app_logs", fs, AS_OF).value == "14 Days"


def test_no_facts_unknown():
    v = resolve_topic("retention_traces", [], AS_OF)
    assert v.status == "UNKNOWN" and v.value is None and not v.sendable
    assert v.rule == "no usable this-customer facts"


def test_conflict_same_rank_undated():
    a = F("model_fallback", "OpenAI only", fid="a", file="a.md")
    b = F("model_fallback", "OpenAI + Anthropic", fid="b", file="b.md")
    v = resolve_topic("model_fallback", [a, b], AS_OF)
    assert v.status == "CONFLICT" and v.value is None
    assert v.used == ["a", "b"]
    assert v.rule == "conflicting values at same rank"
    assert not v.sendable


def test_substring_values_conflict():
    for x, y in [("4 days", "14 days"), ("OpenAI", "OpenAI + Anthropic")]:
        v = resolve_topic("retention_app_logs", [F("retention_app_logs", x, file="x.md"),
                                                 F("retention_app_logs", y, file="y.md")], AS_OF)
        assert v.status == "CONFLICT" and v.value is None


def test_equal_after_normalisation_agrees():
    fs = [F("retention_app_logs", "14d"), F("retention_app_logs", "14 days")]
    assert resolve_topic("retention_app_logs", fs, AS_OF).status == "PROPOSED"


def test_input_order_independent():
    import random
    fs = [
        F("retention_app_logs", "14d", file="b.md", fid="x1"),
        F("retention_app_logs", "14 days", file="a.md", fid="x2"),
        F("retention_app_logs", "30 days", Rank.GENERIC, applies="generic", file="g.md", fid="x3"),
        F("retention_app_logs", "90", Rank.OTHER_CUSTOMER, applies="other_customer", file="m.md", fid="x4"),
        F("retention_app_logs", "7 days", fdate="2026-01-01", file="c.md", fid="x5"),
        F("retention_app_logs", None, certainty="absent", claim="c", file="d.md", fid="x6"),
    ]
    base = resolve_topic("retention_app_logs", fs, AS_OF).model_dump()
    for seed in range(10):
        sh = fs[:]
        random.Random(seed).shuffle(sh)
        assert resolve_topic("retention_app_logs", sh, AS_OF).model_dump() == base
    assert resolve_topic("retention_app_logs", fs[::-1], AS_OF).model_dump() == base


def test_soc2_verified_sendable():
    f = F("soc2", "Type II 2025", Rank.EVIDENCE, applies="generic")
    v = resolve_topic("soc2", [f], AS_OF)
    assert v.status == "VERIFIED" and v.sendable and v.value == "Type II 2025"
    assert v.rule == "evidence, stated"


def test_evidence_not_stated_is_proposed():
    f = F("soc2", "x", Rank.EVIDENCE, certainty="uncertain")
    v = resolve_topic("soc2", [f], AS_OF)
    assert v.status == "PROPOSED" and v.rule == "evidence, not stated"


def test_insurance_expiry_caveats():
    f = F("insurance", "$5M", Rank.EVIDENCE, valid_until=date(2026, 10, 1))
    v = resolve_topic("insurance", [f], date(2026, 9, 25))
    assert v.status == "VERIFIED"
    assert v.caveats == ["expires 2026-10-01"]
    assert not v.sendable
    v2 = resolve_topic("insurance", [f], AS_OF)
    assert v2.caveats == ["expired 2026-10-01"] and not v2.sendable
    v3 = resolve_topic("insurance", [f], date(2026, 6, 1))
    assert v3.caveats == [] and v3.sendable


def test_encryption_generic_only_platform_wide():
    f = F("encryption_at_rest", "AES-256", Rank.GENERIC, applies="generic")
    v = resolve_topic("encryption_at_rest", [f], AS_OF)
    assert v.status == "PROPOSED" and v.value == "AES-256"
    assert v.rule == "generic platform fact: needs confirmation"
    assert not v.sendable


def test_platform_wide_other_customer_not_candidate():
    f = F("encryption_in_transit", "x", Rank.OTHER_CUSTOMER, applies="other_customer",
          file="m.md", fid="m")
    v = resolve_topic("encryption_in_transit", [f], AS_OF)
    assert v.status == "UNKNOWN" and v.hints == ["other customer: x (m.md)"]


def test_pen_test_remediation_absent_plus_uncertain():
    ab = F("pen_test_remediation", None, Rank.EVIDENCE, certainty="absent",
           claim="remediation report", fid="ab")
    ch = F("pen_test_remediation", "remediated later", certainty="uncertain", fid="ch")
    v = resolve_topic("pen_test_remediation", [ab, ch], AS_OF)
    assert v.status == "PROPOSED" and v.value == "remediated later"
    assert "no evidence: remediation report" in v.caveats
    assert ("ab", "absence: remediation report") in v.excluded
    assert v.used == ["ch"] and not v.sendable


def test_older_dated_superseded():
    old = F("retention_app_logs", "30 days", fdate="2026-01-01", fid="old")
    new = F("retention_app_logs", "14 days", fdate="2026-03-01", fid="new")
    v = resolve_topic("retention_app_logs", [old, new], AS_OF)
    assert v.status == "PROPOSED" and v.value == "14 days" and v.used == ["new"]
    assert ("old", "superseded by newer") in v.excluded


def test_undated_not_superseded_and_conflicts():
    old = F("retention_app_logs", "30 days", fdate="2026-01-01", fid="old")
    new = F("retention_app_logs", "14 days", fdate="2026-03-01", fid="new")
    und = F("retention_app_logs", "7 days", fid="und", file="und.md")
    v = resolve_topic("retention_app_logs", [old, new, und], AS_OF)
    assert v.status == "CONFLICT" and v.used == ["new", "und"]
    assert ("old", "superseded by newer") in v.excluded


def test_unparseable_date_is_undated():
    a = F("retention_app_logs", "30 days", fdate="last tuesday", fid="a", file="a.md")
    b = F("retention_app_logs", "14 days", fdate="2026-03-01", fid="b", file="b.md")
    v = resolve_topic("retention_app_logs", [a, b], AS_OF)
    assert v.status == "CONFLICT" and v.used == ["a", "b"]


def test_lower_rank_excluded():
    ev = F("soc2", "Type II", Rank.EVIDENCE, fid="ev")
    ch = F("soc2", "maybe", Rank.CUSTOMER, fid="ch")
    v = resolve_topic("soc2", [ev, ch], AS_OF)
    assert v.status == "VERIFIED" and v.used == ["ev"]
    assert [i for i, _ in v.excluded] == ["ch"]


def test_other_topic_generic_only_unknown_with_hint():
    f = F("other:foo", "bar", Rank.GENERIC, applies="generic", file="g.md", fid="g")
    v = resolve_topic("other:foo", [f], AS_OF)
    assert v.status == "UNKNOWN" and v.value is None
    assert v.hints == ["generic: bar (g.md)"]
    assert ("g", "generic source on per-customer topic") in v.excluded


def test_per_customer_generic_only_unknown_with_hint():
    f = F("retention_traces", "30 days", Rank.GENERIC, applies="generic", file="g.md")
    v = resolve_topic("retention_traces", [f], AS_OF)
    assert v.status == "UNKNOWN" and v.hints == ["generic: 30 days (g.md)"]


def test_resolve_all_groups_by_topic():
    fs = [F("soc2", "x", Rank.EVIDENCE), F("retention_app_logs", "14 days")]
    out = resolve_all(fs, AS_OF)
    assert set(out) == {"soc2", "retention_app_logs"}
    assert out["soc2"].status == "VERIFIED"


def test_pure_does_not_mutate_inputs():
    fs = [F("retention_app_logs", "14 days"), F("retention_app_logs", "30 days")]
    snap = [f.model_copy() for f in fs]
    resolve_topic("retention_app_logs", fs, AS_OF)
    assert fs == snap


# ---- absence superseded by evidence

def _absent(rank=Rank.EVIDENCE, fdate=None):
    return F("pen_test_remediation", "no evidence", rank, certainty="absent",
             claim="no remediation evidence", fdate=fdate, fid="ab", file="summary.txt")


def test_absence_superseded_by_stated_evidence():
    ev = F("pen_test_remediation", "remediated", Rank.EVIDENCE, fid="ev", file="retest.txt",
           applies="generic")
    v = resolve_topic("pen_test_remediation", [_absent(), ev], AS_OF)
    assert v.status == "VERIFIED" and v.value == "remediated"
    assert v.caveats == [] and v.sendable
    assert ("ab", "absence superseded by evidence") in v.excluded
    assert not any(r.startswith("absence:") for _, r in v.excluded)


def test_absence_kept_when_winner_lower_rank():
    ch = F("pen_test_remediation", "remediated", Rank.CUSTOMER, fid="ch")
    v = resolve_topic("pen_test_remediation", [_absent(), ch], AS_OF)
    assert v.status == "PROPOSED"
    assert "no evidence: no remediation evidence" in v.caveats
    assert ("ab", "absence: no remediation evidence") in v.excluded


def test_absence_kept_when_winner_not_stated():
    ev = F("pen_test_remediation", "remediated", Rank.EVIDENCE, certainty="proposed", fid="ev")
    v = resolve_topic("pen_test_remediation", [_absent(), ev], AS_OF)
    assert "no evidence: no remediation evidence" in v.caveats


def test_absence_kept_when_strictly_newer():
    ev = F("pen_test_remediation", "remediated", Rank.EVIDENCE, fdate="2026-03-01", fid="ev")
    v = resolve_topic("pen_test_remediation", [_absent(fdate="2026-04-01"), ev], AS_OF)
    assert "no evidence: no remediation evidence" in v.caveats and not v.sendable
    # same date or older absence -> superseded
    v2 = resolve_topic("pen_test_remediation", [_absent(fdate="2026-03-01"), ev], AS_OF)
    assert v2.caveats == [] and v2.sendable
    # undated winner: the absence cannot be shown newer -> superseded
    ev3 = F("pen_test_remediation", "remediated", Rank.EVIDENCE, fid="ev3")
    v3 = resolve_topic("pen_test_remediation", [_absent(fdate="2026-04-01"), ev3], AS_OF)
    assert v3.caveats == []


def test_absence_kept_on_conflict_and_unknown():
    a = F("pen_test_remediation", "remediated", Rank.EVIDENCE, fid="a", file="a.md")
    b = F("pen_test_remediation", "open", Rank.EVIDENCE, fid="b", file="b.md")
    v = resolve_topic("pen_test_remediation", [_absent(), a, b], AS_OF)
    assert v.status == "CONFLICT" and "no evidence: no remediation evidence" in v.caveats
    v2 = resolve_topic("pen_test_remediation", [_absent()], AS_OF)
    assert v2.status == "UNKNOWN" and v2.caveats == ["no evidence: no remediation evidence"]


def test_unreadable_expiry_caveat_not_sendable():
    f = F("insurance", "$5M", Rank.EVIDENCE)
    f = f.model_copy(update={"valid_until_raw": "next renewal"})
    v = resolve_topic("insurance", [f], AS_OF)
    assert v.status == "VERIFIED"
    assert v.caveats == ["expires on unreadable date 'next renewal'"] and not v.sendable


def test_fact_without_value_is_not_a_candidate():
    f = F("retention_traces", None, certainty="uncertain", claim="traces retention unclear",
          file="chat.md", fid="nv")
    blank = F("retention_traces", "  ", certainty="stated", claim="traces kept", file="b.md",
              fid="bl")
    v = resolve_topic("retention_traces", [f, blank], AS_OF)
    assert v.status == "UNKNOWN" and v.used == []
    assert ("nv", "no value stated") in v.excluded and ("bl", "no value stated") in v.excluded
    assert "open: traces retention unclear (chat.md)" in v.hints


# ---- R26: same-file facts are complementary aspects; CONFLICT needs different files

def test_same_file_aspects_merge_into_one_verified_value():
    amount = F("insurance", "CAD 5,000,000", Rank.EVIDENCE, applies="generic", file="ins.txt",
               fid="ins#F1", line=1)
    expiry = F("insurance", "2026-10-01", Rank.EVIDENCE, applies="generic", file="ins.txt",
               fid="ins#F2", line=2, valid_until=date(2026, 10, 1))
    v = resolve_topic("insurance", [expiry, amount], AS_OF)
    assert v.status == "VERIFIED"
    assert v.value == "CAD 5,000,000; 2026-10-01"  # line order, not input order
    assert v.used == ["ins#F1", "ins#F2"]
    assert v.caveats == ["expired 2026-10-01"] and not v.sendable


def test_same_file_aspects_keep_line_order_not_id_order():
    a = F("pen_test", "external test Feb 2026", Rank.EVIDENCE, file="p.txt", fid="p#F2", line=1)
    b = F("pen_test", "no critical or high findings", Rank.EVIDENCE, file="p.txt", fid="p#F1",
          line=2)
    v = resolve_topic("pen_test", [b, a], AS_OF)
    assert v.status == "VERIFIED" and v.sendable
    assert v.value == "external test Feb 2026; no critical or high findings"


def test_same_file_duplicate_values_collapse_before_joining():
    fs = [F("retention_app_logs", "14d", file="c.md", line=1),
          F("retention_app_logs", "enabled", file="c.md", line=2),
          F("retention_app_logs", "14 days", file="c.md", line=3)]
    v = resolve_topic("retention_app_logs", fs, AS_OF)
    assert v.status == "PROPOSED" and v.value == "14 days; enabled"


def test_different_files_different_values_still_conflict():
    a = F("insurance", "CAD 5,000,000", Rank.EVIDENCE, file="a.txt", fid="a")
    b = F("insurance", "CAD 2,000,000", Rank.EVIDENCE, file="b.txt", fid="b")
    v = resolve_topic("insurance", [a, b], AS_OF)
    assert v.status == "CONFLICT" and v.value is None and v.used == ["a", "b"]


def test_files_agree_when_merged_values_match():
    fs = [F("pen_test", "annual", Rank.EVIDENCE, file="a.txt", line=1),
          F("pen_test", "external", Rank.EVIDENCE, file="a.txt", line=2),
          F("pen_test", "Annual", Rank.EVIDENCE, file="b.txt", line=1),
          F("pen_test", "External", Rank.EVIDENCE, file="b.txt", line=5)]
    v = resolve_topic("pen_test", fs, AS_OF)
    assert v.status == "VERIFIED"
    # longest-then-lexicographic over the per-file merged values
    assert v.value == "Annual; External"


def test_merged_file_vs_other_file_subset_conflicts():
    fs = [F("pen_test", "annual", Rank.EVIDENCE, file="a.txt", line=1),
          F("pen_test", "external", Rank.EVIDENCE, file="a.txt", line=2),
          F("pen_test", "annual", Rank.EVIDENCE, file="b.txt", line=1)]
    assert resolve_topic("pen_test", fs, AS_OF).status == "CONFLICT"


def test_merge_is_input_order_independent():
    import random
    fs = [
        F("pen_test", "external Feb 2026", Rank.EVIDENCE, file="a.txt", line=1, fid="a#F1"),
        F("pen_test", "no critical or high", Rank.EVIDENCE, file="a.txt", line=2, fid="a#F2"),
        F("pen_test", "no critical or high", Rank.EVIDENCE, file="a.txt", line=3, fid="a#F3"),
        F("pen_test", "External Feb 2026", Rank.EVIDENCE, file="b.txt", line=4, fid="b#F1"),
        F("pen_test", "No Critical or High", Rank.EVIDENCE, file="b.txt", line=9, fid="b#F2"),
        F("pen_test", "annual", Rank.GENERIC, applies="generic", file="g.md", fid="g#F1"),
    ]
    base = resolve_topic("pen_test", fs, AS_OF).model_dump()
    assert base["status"] == "VERIFIED"
    for seed in range(10):
        sh = fs[:]
        random.Random(seed).shuffle(sh)
        assert resolve_topic("pen_test", sh, AS_OF).model_dump() == base


def test_absence_not_superseded_by_facts_from_its_own_file():
    # a file that says it lacks X cannot itself be the evidence for X
    ab = F("pen_test_remediation", "no remediation evidence", Rank.EVIDENCE, certainty="absent",
           claim="This file contains no remediation evidence.", file="sum.txt", fid="ab", line=4)
    st = F("pen_test_remediation", "two medium findings open", Rank.EVIDENCE, file="sum.txt",
           fid="st", line=2)
    v = resolve_topic("pen_test_remediation", [ab, st], AS_OF)
    assert v.status == "VERIFIED" and v.value == "two medium findings open"
    assert "no evidence: This file contains no remediation evidence." in v.caveats
    assert not v.sendable
