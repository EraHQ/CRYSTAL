"""LongMemEval harness — the agent-surface re-point (LAUNCH_PLAN L7-Q1=A,
ratified 2026-08-28).

The harness is a live-server script under scripts/, not a package
module, so it is loaded here by path. These pins cover the parts that
carry the benchmark's integrity claims and need no server:

  1. ask() hits /v1/agent/messages (the product surface, not the
     retiring proxy), pins temperature=0 (Guarantee #6), sends the
     Bearer key, and reads `final_text` from the agent envelope.
  2. stop_reason="error" raises AgentTurnError — the agent's error
     text is an apology string a judge could mistake for an
     abstention, so it is a harness ERROR, never a graded answer
     (Guarantees #3/#7).
  3. agent_turn_fields() keeps tool NAMES only and flags external
     tools (web_search / web_fetch) — the disclosure column the
     summary uses to disqualify a run (Guarantee #3).
  4. The proxy's --mode flag is gone (there is no passive/active
     split on the agent lane) and the version stamp moved to 3.x so
     2.x rows are never read as comparable.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import httpx
import pytest

_HARNESS_PATH = Path(__file__).parent.parent / "scripts" / "longmemeval_harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("longmemeval_harness", _HARNESS_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _envelope(**overrides):
    body = {
        "id": "chatcmpl-agent-x",
        "model": "claude-haiku-4-5-20251001",
        "final_text": "Paris",
        "stop_reason": "end_turn",
        "iterations": 2,
        "prompt_tokens": 1200,
        "completion_tokens": 30,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "tool_calls": [
            {"tool_name": "knowledge_search", "input": {"q": "x"}, "output": {}},
            {"tool_name": "crystal_recall", "input": {}, "output": {}},
        ],
    }
    body.update(overrides)
    return body


def test_ask_posts_to_agent_door_at_temperature_zero(harness):
    seen: dict = {}

    def handler(req: httpx.Request) -> httpx.Response:
        seen["url"] = str(req.url)
        seen["auth"] = req.headers.get("authorization")
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json=_envelope())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    answer, turn = harness.ask(client, "key-A", "claude-haiku-4-5-20251001", "capital?")

    assert seen["url"].endswith("/v1/agent/messages")
    assert "/v1/chat/completions" not in seen["url"]
    assert seen["auth"] == "Bearer key-A"
    assert seen["body"]["temperature"] == 0  # Guarantee #6
    assert seen["body"]["max_tokens"] == 1024
    assert seen["body"]["model"] == "claude-haiku-4-5-20251001"
    assert seen["body"]["messages"] == [{"role": "user", "content": "capital?"}]

    assert answer == "Paris"
    assert turn["stop_reason"] == "end_turn"
    assert turn["iterations"] == 2
    assert turn["prompt_tokens"] == 1200 and turn["completion_tokens"] == 30
    assert turn["tool_calls"] == ["knowledge_search", "crystal_recall"]
    assert turn["external_tool_used"] is False


def test_ask_raises_on_agent_error_stop_reason(harness):
    """An error turn is counted, never graded (the text is an apology)."""

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_envelope(
            final_text="I hit an error reaching the model: boom. Please try again.",
            stop_reason="error", iterations=1, tool_calls=[],
        ))

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(harness.AgentTurnError) as ei:
        harness.ask(client, "k", "m", "q")
    assert "stop_reason=error" in str(ei.value)


def test_ask_surfaces_http_errors(harness):
    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"detail": "viewer"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(httpx.HTTPStatusError):
        harness.ask(client, "k", "m", "q")


def test_agent_turn_fields_flags_external_tools(harness):
    fields = harness.agent_turn_fields(_envelope(tool_calls=[
        {"tool_name": "knowledge_search", "input": {}, "output": {}},
        {"tool_name": "web_fetch", "input": {"url": "https://x"}, "output": {}},
    ]))
    assert fields["tool_calls"] == ["knowledge_search", "web_fetch"]
    assert fields["external_tool_used"] is True

    fields = harness.agent_turn_fields(_envelope(tool_calls=[
        {"tool_name": "web_search", "input": {"query": "q"}, "output": {}},
    ]))
    assert fields["external_tool_used"] is True

    assert harness.EXTERNAL_TOOLS == frozenset({"web_search", "web_fetch"})


def test_agent_turn_fields_cache_hit_and_malformed_entries(harness):
    """A C2 cache hit is a legitimate zero-iteration turn; malformed
    tool_calls entries are dropped, not crashed on."""
    fields = harness.agent_turn_fields(_envelope(
        stop_reason="cache_hit", iterations=0, prompt_tokens=0,
        completion_tokens=0, tool_calls=[],
    ))
    assert fields["stop_reason"] == "cache_hit"
    assert fields["iterations"] == 0
    assert fields["tool_calls"] == []
    assert fields["external_tool_used"] is False

    fields = harness.agent_turn_fields({"tool_calls": [None, {}, {"tool_name": ""}]})
    assert fields["tool_calls"] == []
    assert fields["stop_reason"] is None


def test_mode_flag_is_gone_and_version_is_3x(harness):
    assert harness.HARNESS_VERSION.startswith("3.")
    with pytest.raises(SystemExit):
        harness.main(["--data", "x.json", "--mode", "passive"])
