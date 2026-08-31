"""The LLM wrapper: schema inlining, validation retries and graceful failure."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from fi_agent.config import LLMSettings
from fi_agent.llm import LLMClient, _inline_refs
from fi_agent.schemas import AnalystResult, TriageResult


def response(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content), prompt_eval_count=10, eval_count=20
    )


@pytest.fixture
def client(tmp_path) -> LLMClient:
    return LLMClient(
        LLMSettings(base_url="http://test:11434", max_retries=2),
        trace_path=tmp_path / "trace.jsonl",
    )


# -- schema handling --------------------------------------------------------------------


def test_inline_refs_removes_defs():
    """Ollama compiles the schema to a grammar and handles $ref inconsistently."""
    schema = _inline_refs(AnalystResult.model_json_schema())
    assert "$defs" not in schema
    assert "$ref" not in str(schema)


def test_inline_refs_preserves_nested_structure():
    schema = _inline_refs(AnalystResult.model_json_schema())
    evidence = schema["properties"]["evidence"]["items"]
    assert set(evidence["properties"]) == {"claim", "source_idx"}


def test_confidence_is_an_enum_not_a_number():
    """A 0-1 float prompted the model to return 100; enums are the fix."""
    schema = _inline_refs(AnalystResult.model_json_schema())
    assert schema["properties"]["confidence"]["enum"] == ["high", "medium", "low"]


# -- generation -------------------------------------------------------------------------


def test_structured_returns_validated_object(client, monkeypatch):
    payload = '{"selected": [{"idx": 0, "relevance": "high", "why": "x"}], ' \
              '"no_material_news": false}'
    monkeypatch.setattr(client._client, "chat", lambda **kw: response(payload))

    result = client.structured(TriageResult, "sys", "user", label="t")
    assert result.selected[0].idx == 0
    assert client.calls == 1


def test_num_ctx_is_always_sent(client, monkeypatch):
    """Ollama defaults num_ctx to 4096, silently truncating article text."""
    seen = {}

    def capture(**kwargs):
        seen.update(kwargs)
        return response('{"selected": [], "no_material_news": true}')

    monkeypatch.setattr(client._client, "chat", capture)
    client.structured(TriageResult, "sys", "user")
    assert seen["options"]["num_ctx"] == 16384
    assert seen["format"], "generation must be schema-constrained"


def test_invalid_json_is_retried_with_the_error(client, monkeypatch):
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs["messages"])
        if len(attempts) == 1:
            return response('{"selected": "not a list"}')
        return response('{"selected": [], "no_material_news": true}')

    monkeypatch.setattr(client._client, "chat", flaky)
    result = client.structured(TriageResult, "sys", "user")

    assert result is not None
    assert len(attempts) == 2
    assert "did not match the required schema" in attempts[1][-1]["content"]


def test_persistent_invalid_json_returns_none(client, monkeypatch):
    monkeypatch.setattr(client._client, "chat", lambda **kw: response("{not json"))
    assert client.structured(TriageResult, "sys", "user") is None
    assert client.calls == 3, "initial attempt plus two retries"


def test_transport_failure_returns_none_rather_than_raising(client, monkeypatch):
    def boom(**kwargs):
        raise ConnectionError("host is down")

    monkeypatch.setattr(client._client, "chat", boom)
    assert client.structured(TriageResult, "sys", "user") is None


def test_calls_are_traced(client, monkeypatch, tmp_path):
    monkeypatch.setattr(
        client._client, "chat",
        lambda **kw: response('{"selected": [], "no_material_news": true}'),
    )
    client.structured(TriageResult, "sys", "user", label="triage:NVDA")

    lines = (tmp_path / "trace.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1 and "triage:NVDA" in lines[0]
