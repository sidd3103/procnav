from __future__ import annotations

import re

from .loader import Person
from .models import Topic

_STOP = {"and", "the", "for", "with", "from", "that", "this", "are", "all", "any", "our",
         "of", "to", "or", "in", "on", "an", "as", "is", "it", "be", "by", "we"}


def _tokens(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) >= 2 and t not in _STOP}


def _is_customer(p: Person) -> bool:
    return "customer" in (p.role or "").lower()


def assign_owner(topic: Topic, people: list[Person]) -> str:
    """Person with the largest token overlap between topic words and role + knowledge.

    Customer contacts (role contains "customer") are excluded unless nobody else exists.
    Topic words are keywords + area + the words of the topic id. Ties go to the earliest
    person. Zero overlap, or no people at all, returns "Unassigned" (a suggestion only; a
    human triages it) and never raises.
    """
    if not people:
        return "Unassigned"
    pool = [p for p in people if not _is_customer(p)] or people
    words = _tokens(" ".join([*topic.keywords, topic.area, topic.id.replace("_", " ")]))
    best, best_score = None, 0
    for p in pool:
        score = len(words & _tokens(f"{p.role or ''} {p.knowledge}"))
        if score > best_score:
            best, best_score = p, score
    return best.name if best is not None else "Unassigned"


def assign_decision_owner(
    decider: str, people: list[Person], topic: Topic | None = None
) -> str:
    if decider.lower() == "customer":
        for p in people:
            if _is_customer(p):
                return p.name
        return "Unassigned (customer contact)"
    internal = [p for p in people if not _is_customer(p)]
    if not internal:
        return "Unassigned"
    return assign_owner(topic, internal) if topic is not None else internal[0].name
