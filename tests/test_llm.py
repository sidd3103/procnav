import pytest
from pydantic import BaseModel

from procnav import llm
from procnav.llm import FakeLLM, OpenAIClient


class Out(BaseModel):
    x: int


@pytest.fixture
def clean_env(monkeypatch):
    for k in ["OPENAI_API_KEY", "OPENROUTER_API_KEY", "PROCNAV_MODEL", "PROCNAV_BASE_URL"]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(llm, "load_dotenv", lambda *a, **k: False)
    return monkeypatch


def test_openrouter_only(clean_env):
    clean_env.setenv("OPENROUTER_API_KEY", "k")
    c = OpenAIClient.from_env()
    assert c.base_url == "https://openrouter.ai/api/v1"
    assert c.model == "openai/gpt-4.1-mini"


def test_openai_key(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "k")
    clean_env.setenv("OPENROUTER_API_KEY", "k2")
    c = OpenAIClient.from_env()
    assert c.base_url is None
    assert c.model == "gpt-4.1-mini"


def test_overrides(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "k")
    clean_env.setenv("PROCNAV_MODEL", "m")
    clean_env.setenv("PROCNAV_BASE_URL", "http://x")
    c = OpenAIClient.from_env()
    assert (c.model, c.base_url) == ("m", "http://x")


def test_missing_keys(clean_env):
    with pytest.raises(RuntimeError):
        OpenAIClient.from_env()


def test_fake_dict_and_calls():
    f = FakeLLM({"a": Out(x=1), "b": ValueError("boom")})
    assert f.model == "fake"
    assert f.complete_json("s", "KEY: a\nbody", Out) == Out(x=1)
    with pytest.raises(ValueError):
        f.complete_json("s", "KEY: b\nbody", Out)
    assert f.calls == 2


def test_fake_missing_key_raises():
    with pytest.raises(KeyError):
        FakeLLM({}).complete_json("s", "KEY: zzz", Out)


def test_fake_list_consumed_in_order():
    f = FakeLLM({"a": [ValueError("bad"), Out(x=2)]})
    with pytest.raises(ValueError):
        f.complete_json("s", "KEY: a", Out)
    assert f.complete_json("s", "KEY: a", Out) == Out(x=2)
    assert f.calls == 2


def test_fake_callable():
    seen = []

    def fn(system, user, schema):
        seen.append((system, user, schema))
        return Out(x=9)

    f = FakeLLM(fn)
    assert f.complete_json("S", "U", Out) == Out(x=9)
    assert seen == [("S", "U", Out)] and f.calls == 1


# ---- R29: the schema sent to the model marks every property required

def test_all_required_schema_nested_defs():
    from procnav.extract import ExtractionOut
    from procnav.llm import all_required_schema
    s = all_required_schema(ExtractionOut)
    assert set(s["required"]) == set(s["properties"])
    assert {"FactOut", "ReqOut", "DepOut", "DecOut"} <= set(s["$defs"])
    for name, d in s["$defs"].items():
        if "properties" in d:
            assert set(d["required"]) == set(d["properties"]), name
    assert "quote" in s["$defs"]["DecOut"]["required"]
    assert "quote" in s["$defs"]["DepOut"]["required"]
    # parsing stays tolerant
    from procnav.extract import DecOut
    assert DecOut.model_validate({"decider": "internal"}).quote == ""


def test_all_required_schema_does_not_mutate_pydantic_cache():
    from procnav.extract import DecOut
    from procnav.llm import all_required_schema
    all_required_schema(DecOut)
    assert "quote" not in DecOut.model_json_schema().get("required", [])


def test_openai_client_sends_all_required_schema():
    from procnav.extract import ExtractionOut
    seen = {}

    class Msg:
        content = '{"source_type": "guidance", "customer_scope": "generic"}'

    class Resp:
        choices = [type("C", (), {"message": Msg()})()]

    class Completions:
        def create(self, **kw):
            seen.update(kw)
            return Resp()

    c = OpenAIClient("k", "m")
    c._client = type("Cl", (), {"chat": type("Ch", (), {"completions": Completions()})()})()
    out = c.complete_json("s", "u", ExtractionOut)
    assert out.source_type == "guidance"
    schema = seen["response_format"]["json_schema"]["schema"]
    assert "quote" in schema["$defs"]["ReqOut"]["required"]
