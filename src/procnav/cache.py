from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path


def cache_key(*parts: object) -> str:
    h = hashlib.sha256()
    for p in parts:
        b = str(p).encode("utf-8")
        h.update(len(b).to_bytes(8, "big"))
        h.update(b)
    return h.hexdigest()


class JsonCache:
    def __init__(self, dir: str | Path):
        self.dir = Path(dir)

    def get(self, key: str) -> dict | None:
        p = self.dir / f"{key}.json"
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None

    def put(self, key: str, value: dict) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(json.dumps(value))
            os.replace(tmp, self.dir / f"{key}.json")
        except BaseException:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
