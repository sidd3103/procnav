from __future__ import annotations

import copy
import os
from typing import Any, Callable, Protocol

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

OPENROUTER_URL = "https://openrouter.ai/api/v1"


class LLMClient(Protocol):
    model: str

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel: ...


def _require_all(node: Any) -> None:
    if isinstance(node, dict):
        props = node.get("properties")
        if isinstance(props, dict):
            node["required"] = list(props)
        for v in node.values():
            _require_all(v)
    elif isinstance(node, list):
        for v in node:
            _require_all(v)


def all_required_schema(schema: type[BaseModel]) -> dict:
    """JSON schema for the model with EVERY property of every object (incl. $defs) required.

    Asking for all fields makes the model emit quotes etc. instead of omitting them; parsing
    stays tolerant because pydantic still applies the defaults.
    """
    s = copy.deepcopy(schema.model_json_schema())
    _require_all(s)
    return s


class OpenAIClient:
    def __init__(self, api_key: str, model: str, base_url: str | None = None):
        self.model = model
        self.base_url = base_url
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @classmethod
    def from_env(cls) -> "OpenAIClient":
        load_dotenv()
        openai_key = os.environ.get("OPENAI_API_KEY")
        router_key = os.environ.get("OPENROUTER_API_KEY")
        if openai_key:
            key, base_url, default_model = openai_key, None, "gpt-4.1-mini"
        elif router_key:
            key, base_url, default_model = router_key, OPENROUTER_URL, "openai/gpt-4.1-mini"
        else:
            raise RuntimeError("Set OPENAI_API_KEY or OPENROUTER_API_KEY")
        base_url = os.environ.get("PROCNAV_BASE_URL") or base_url
        model = os.environ.get("PROCNAV_MODEL") or default_model
        return cls(key, model, base_url)

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": all_required_schema(schema),
                    "strict": False,
                },
            },
        )
        return schema.model_validate_json(resp.choices[0].message.content or "")


class FakeLLM:
    model = "fake"

    def __init__(self, responses: dict | Callable):
        self.responses = responses
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: type[BaseModel]) -> BaseModel:
        self.calls += 1
        if callable(self.responses):
            return self.responses(system, user, schema)
        first = user.split("\n", 1)[0]
        if not first.startswith("KEY:"):
            raise KeyError(f"user prompt lacks KEY line: {first!r}")
        key = first[4:].strip()
        value = self.responses[key]
        if isinstance(value, list):
            if not value:
                raise KeyError(f"responses for {key!r} exhausted")
            value = value.pop(0)
        if isinstance(value, Exception):
            raise value
        return value
