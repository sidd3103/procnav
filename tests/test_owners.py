from pathlib import Path

from procnav.loader import Person, load_case, parse_people
from procnav.owners import assign_decision_owner, assign_owner
from procnav.topics import SEED, get_topic

ACME = Path(__file__).resolve().parent.parent / "reonsiteinterview" / "CASE_1_Acme_Bank"


def people():
    files, _ = load_case(ACME)
    for f in files:
        p = parse_people(f)
        if p is not None:
            return p
    raise AssertionError


def test_real_acme_owners():
    p = people()
    assert assign_owner(SEED["subprocessors"], p) == "Mina"
    assert assign_owner(SEED["retention_app_logs"], p) == "Anthony"
    assert assign_owner(SEED["insurance"], p) == "Sam"
    assert assign_owner(SEED["model_fallback"], p) in {"Parsa", "Anthony"}


def test_real_acme_owners_new_topics():
    p = people()
    assert assign_owner(SEED["trace_payload_capture"], p) in {"Parsa", "Anthony"}
    assert assign_owner(SEED["service_description"], p) == "Jai"
    assert assign_owner(SEED["submission_approval"], p) == "Mina"
    assert assign_owner(SEED["approval_blockers"], p) == "Jai"


def test_tie_goes_to_first_and_zero_overlap_is_unassigned():
    ps = [Person(name="A", role="x", knowledge="widgets"), Person(name="B", role="y", knowledge="widgets")]
    t = get_topic("widgets thing")
    assert assign_owner(t, ps) == "A"
    assert assign_owner(get_topic("nothing common"), ps) == "Unassigned"


def test_unmatched_new_topic_is_not_given_to_first_person():
    p = people()
    assert assign_owner(get_topic("other:launch_date"), p) == "Unassigned"
    assert assign_decision_owner("internal", p, get_topic("other:launch_date")) == "Unassigned"


def test_decision_owner_customer():
    p = people()
    assert assign_decision_owner("customer", p) == "Customer Procurement"
    assert assign_decision_owner("CUSTOMER", p, SEED["dpa"]) == "Customer Procurement"


def test_decision_owner_internal_excludes_customers():
    p = people()
    assert assign_decision_owner("internal", p) == "Jai"
    assert assign_decision_owner("internal", p, SEED["subprocessors"]) == "Mina"
    cust_first = [Person(name="C", role="Customer", knowledge="subprocessors"),
                  Person(name="M", role="Legal", knowledge="privacy")]
    assert assign_decision_owner("internal", cust_first, SEED["subprocessors"]) == "M"


def test_assign_owner_excludes_customer_people():
    ps = [Person(name="C", role="Customer", knowledge="oversight review governance human"),
          Person(name="I", role="Product", knowledge="misc")]
    assert assign_owner(SEED["human_oversight"], ps) == "I"


def test_assign_owner_all_customers_falls_back_to_full_list():
    ps = [Person(name="C1", role="Customer", knowledge="x"),
          Person(name="C2", role="customer", knowledge="oversight")]
    assert assign_owner(SEED["human_oversight"], ps) == "C2"


def test_never_raises_on_missing_people():
    assert assign_owner(SEED["dpa"], []) == "Unassigned"
    assert assign_decision_owner("internal", []) == "Unassigned"
    assert assign_decision_owner("internal", [], SEED["dpa"]) == "Unassigned"
    only_internal = [Person(name="M", role="Legal", knowledge="dpa")]
    assert assign_decision_owner("customer", only_internal) == "Unassigned (customer contact)"
    assert assign_decision_owner("customer", []) == "Unassigned (customer contact)"


def test_ai_topics_go_to_ai_engineering_not_first_person():
    p = people()
    for tid in ("human_oversight", "provider_training_retention", "evaluation", "tool_actions"):
        assert assign_owner(SEED[tid], p) == "Parsa", tid
    assert assign_owner(SEED["human_oversight"], p) != "Jai"
    assert assign_owner(SEED["personal_info_scope"], p) == "Mina"


def test_two_letter_stopwords_do_not_match():
    ps = [Person(name="A", role="x", knowledge="of to or in on an as is it be by we"),
          Person(name="B", role="y", knowledge="ai")]
    assert assign_owner(SEED["evaluation"], ps) == "B"
