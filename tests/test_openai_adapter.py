"""Slice 5a: OpenAI adapter wire translation for the agent loop.

Anthropic message shape is CRYS's internal representation; these tests
pin the translation at the _call_model boundary in both directions:
request out (tools, system, history with tool_use/tool_result) and
response back (shim with Anthropic-shaped block dicts, mapped
stop_reason including the H2 max_tokens guard, and duck-typed usage).

Pure unit tests - no store, no fixtures, no network.

R14 note: these assertions are verified by `pytest`; they describe
expected behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

import json

import pytest

from crystal_cache.agent.adapters.openai import (
    OpenAIChatShim,
    messages_to_openai,
    parse_openai_response,
    tools_to_openai,
)


# ---------------------------------------------------------------------------
# tools_to_openai
# ---------------------------------------------------------------------------

def test_tools_translate_and_cache_control_dropped():
    anthropic_tools = [
        {
            "name": "crystal_recall",
            "description": "Recall knowledge.",
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
        {
            "name": "key_scan",
            "description": "Scan sparse keys.",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": {"type": "ephemeral"},  # Anthropic-only mark
        },
    ]

    out = tools_to_openai(anthropic_tools)

    assert len(out) == 2
    assert out[0] == {
        "type": "function",
        "function": {
            "name": "crystal_recall",
            "description": "Recall knowledge.",
            "parameters": anthropic_tools[0]["input_schema"],
        },
    }
    # The mark never crosses the boundary.
    assert "cache_control" not in json.dumps(out)


def test_tools_empty_becomes_none():
    assert tools_to_openai([]) is None
    assert tools_to_openai(None) is None


# ---------------------------------------------------------------------------
# messages_to_openai
# ---------------------------------------------------------------------------

def test_system_and_plain_user_string():
    out = messages_to_openai("You are CRYS.", [
        {"role": "user", "content": "hello"},
    ])
    assert out == [
        {"role": "system", "content": "You are CRYS."},
        {"role": "user", "content": "hello"},
    ]


def test_assistant_text_and_tool_use_become_tool_calls():
    history = [
        {"role": "user", "content": "count the crystals"},
        {"role": "assistant", "content": [
            {"type": "text", "text": "Let me scan."},
            {"type": "tool_use", "id": "tu_1", "name": "key_scan",
             "input": {"subject_contains": "crystal"}},
        ]},
    ]

    out = messages_to_openai(None, history)

    assistant = out[1]
    assert assistant["role"] == "assistant"
    assert assistant["content"] == "Let me scan."
    assert assistant["tool_calls"] == [{
        "id": "tu_1",
        "type": "function",
        "function": {
            "name": "key_scan",
            "arguments": json.dumps({"subject_contains": "crystal"}),
        },
    }]


def test_assistant_tool_use_only_has_null_content():
    out = messages_to_openai(None, [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "tu_1", "name": "t", "input": {}},
        ]},
    ])
    assert out[0]["content"] is None
    assert out[0]["tool_calls"][0]["function"]["arguments"] == "{}"


def test_tool_result_turn_becomes_role_tool_messages():
    history = [
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "tu_1",
             "content": '{"count": 3}'},
            {"type": "tool_result", "tool_use_id": "tu_2",
             "content": "boom", "is_error": True},
            {"type": "text", "text": "keep going"},
        ]},
    ]

    out = messages_to_openai(None, history)

    # Tool messages first (must directly follow the assistant turn),
    # then the text as a user message.
    assert out[0] == {
        "role": "tool", "tool_call_id": "tu_1", "content": '{"count": 3}',
    }
    assert out[1]["role"] == "tool"
    assert out[1]["tool_call_id"] == "tu_2"
    assert out[1]["content"].startswith("[tool error]")
    assert out[2] == {"role": "user", "content": "keep going"}


def test_cache_control_marks_never_cross():
    history = [
        {"role": "user", "content": [
            {"type": "text", "text": "hi",
             "cache_control": {"type": "ephemeral"}},
        ]},
    ]
    out = messages_to_openai("sys", history)
    assert "cache_control" not in json.dumps(out)


# ---------------------------------------------------------------------------
# parse_openai_response
# ---------------------------------------------------------------------------

def _completion(message: dict, finish: str, usage: dict | None = None) -> dict:
    return {
        "id": "chatcmpl-1",
        "model": "qwen2.5-72b",
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": usage or {},
    }


def test_parse_text_response():
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": "All done."},
        "stop",
        {"prompt_tokens": 100, "completion_tokens": 20,
         "prompt_tokens_details": {"cached_tokens": 60}},
    ))

    assert isinstance(shim, OpenAIChatShim)
    assert shim.content == [{"type": "text", "text": "All done."}]
    assert shim.stop_reason == "end_turn"
    assert shim.usage.input_tokens == 100
    assert shim.usage.output_tokens == 20
    assert shim.usage.cache_read_input_tokens == 60
    assert shim.usage.cache_creation_input_tokens == 0
    assert shim.model == "qwen2.5-72b"


def test_parse_tool_calls_response():
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_abc",
            "type": "function",
            "function": {
                "name": "crystal_recall",
                "arguments": '{"query": "loop tax"}',
            },
        }]},
        "tool_calls",
    ))

    assert shim.stop_reason == "tool_use"
    assert shim.content == [{
        "type": "tool_use",
        "id": "call_abc",
        "name": "crystal_recall",
        "input": {"query": "loop tax"},
    }]


def test_parse_malformed_arguments_degrade_to_empty_input():
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": None, "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "key_scan", "arguments": "{not json"},
        }]},
        "tool_calls",
    ))
    # Dispatchable tool_use with empty input; the tool's own error path
    # informs the model, which can retry.
    assert shim.content[0]["type"] == "tool_use"
    assert shim.content[0]["input"] == {}


def test_parse_length_maps_to_max_tokens_for_h2_guard():
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": "truncated..."},
        "length",
    ))
    assert shim.stop_reason == "max_tokens"


def test_parse_stop_with_tool_calls_forces_tool_use():
    """Some OpenAI-compatible servers report finish stop alongside tool
    calls; the loop keys on stop_reason, so tool_use must win."""
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": "calling", "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "t", "arguments": "{}"},
        }]},
        "stop",
    ))
    assert shim.stop_reason == "tool_use"


def test_parse_missing_tool_call_id_synthesized():
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": None, "tool_calls": [{
            "type": "function",
            "function": {"name": "t", "arguments": "{}"},
        }]},
        "tool_calls",
    ))
    assert shim.content[0]["id"]  # non-empty for tool_result pairing


def test_parse_no_choices_raises():
    with pytest.raises(ValueError):
        parse_openai_response({"choices": []})


def test_shim_content_roundtrips_as_history():
    """Shim blocks are plain dicts, so the loop can append them to the
    working history and this adapter can translate them back out on the
    next iteration - the full multi-turn cycle."""
    shim = parse_openai_response(_completion(
        {"role": "assistant", "content": "Scanning.", "tool_calls": [{
            "id": "call_9",
            "type": "function",
            "function": {"name": "key_scan", "arguments": '{"q": "x"}'},
        }]},
        "tool_calls",
    ))

    history = [
        {"role": "user", "content": "count"},
        {"role": "assistant", "content": shim.content},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "call_9",
             "content": '{"count": 3}'},
        ]},
    ]

    out = messages_to_openai("sys", history)

    assert [m["role"] for m in out] == ["system", "user", "assistant", "tool"]
    assert out[2]["tool_calls"][0]["id"] == "call_9"
    assert out[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
    assert out[3]["tool_call_id"] == "call_9"
