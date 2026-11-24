from __future__ import annotations

import csv
import hashlib
import io
from pathlib import Path

from pydantic import BaseModel

from .models import Requirement


class CaseFile(BaseModel):
    path: str  # relative POSIX path
    text: str
    sha256: str


class Person(BaseModel):
    name: str
    role: str | None = None
    knowledge: str = ""
    response_pattern: str | None = None


def load_case(root: str | Path) -> tuple[list[CaseFile], list[str]]:
    root = Path(root)
    files: list[CaseFile] = []
    skipped: list[str] = []
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        if any(part.startswith(".") for part in rel.split("/")):
            skipped.append(rel)
            continue
        try:
            data = p.read_bytes()
        except OSError:
            skipped.append(rel)
            continue
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            skipped.append(rel)
            continue
        if "\x00" in text:
            skipped.append(rel)
            continue
        files.append(CaseFile(path=rel, text=text, sha256=hashlib.sha256(data).hexdigest()))
    return files, skipped


def _rows(f: CaseFile, required: set[str]) -> list[dict[str, str]] | None:
    if not f.path.lower().endswith(".csv"):
        return None
    reader = csv.DictReader(io.StringIO(f.text.lstrip("\ufeff")))
    header = {h.strip().lower() for h in (reader.fieldnames or [])}
    if not required <= header:
        return None
    return [
        {k.strip().lower(): (v or "").strip() for k, v in row.items() if k is not None}
        for row in reader
    ]


def parse_questionnaire(f: CaseFile) -> list[Requirement] | None:
    rows = _rows(f, {"question"})
    if rows is None:
        return None
    return [
        Requirement(
            id=r.get("id") or f"row-{i}",
            text=r["question"],
            requester=None,
            topics=[],
            source_file=f.path,
            quote=None,
            draft_answer=r.get("draft_answer") or None,
            from_customer=False,
            group=None,
            section=r.get("section") or None,
            note=r.get("note") or None,
        )
        for i, r in enumerate(rows, 1)
    ]


def parse_people(f: CaseFile) -> list[Person] | None:
    rows = _rows(f, {"person", "likely_knowledge"})
    if rows is None:
        return None
    return [
        Person(
            name=r["person"],
            role=r.get("role") or None,
            knowledge=r["likely_knowledge"],
            response_pattern=r.get("response_pattern") or None,
        )
        for r in rows
    ]
