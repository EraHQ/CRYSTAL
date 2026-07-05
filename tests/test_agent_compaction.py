"""C3 — agent-loop compaction tests (cost + parity, 2026-06-17).

Covers `crystal_cache.retrieval.compaction.compact_agent_trajectory` and its
wiring into `Agent.run` behind CC_AGENT_COMPACTION. The weight is on the
boundary slicer: the agent's `working` list interleaves assistant `tool_use`
turns with `user` `tool_result` turns, and the Anthropic Messages API rejects
a `tool_result` whose `tool_use` was dropped — so a naive tail slice would
400. These tests assert the compacted view stays a VALID Anthropic message
sequence (no orphaned tool_result, opens on the user task anchor), that the
running summary accumulates across repeated compactions, that the summary is a
readable progress log rather than stringified block JSON, and that the agent
returns the FULL (uncompacted) trajectory even when the model saw a bounded
view.

R14 note: these assertions are verified by `pytest`; they describe expected
behavior and have not yet been run at authoring time.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache import config
from crystal_cache.agent import Agent
from crystal_cache.agent.agent import _COMPACTION_HEADER
from crystal_cache.retrieval.compaction import (
    _summarize_agent_turns,
    compact_agent_trajectory,
    should_compact,
)


# ---------------------------------------------------------------------------
# Trajectory builders + validator
# ---------------------------------------------------------------------------

def _round(i: int) -> list[dict[str, Any]]:
    """One agent tool round: an assistant tool_use turn + its tool_result.

    Mirrors the real loop shape — assistant carries a text block plus a
    tool_use block; the result is a user-role turn of one tool_result block.
    """
    return [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"Step {i}: doing work."},
                {
                    "type": "tool_use",
                    "id": f"tu_{i}",
                    "name": "read_file",
                    "input": {"path": f"f{i}.py"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": f"tu_{i}",
                    "content": f"result {i}",
                    "is_error": False,
                },
            ],
        },
    ]


def _make_trajectory(
    n_rounds: int, task: str = "Fix the failing test.",
) -> list[dict[str, Any]]:
    """A task turn followed by n complete tool rounds.

    User-turn count == n_rounds + 1 (the task + one per tool_result), so
    should_compact (>= COMPACT_THRESHOLD == 10) fires once n_rounds >= 9.
    """
    msgs: list[dict[str, Any]] = [{"role": "user", "content": task}]
    for i in range(1, n_rounds + 1):
        msgs.extend(_round(i))
    return msgs


def _assert_valid_anthropic_pairs(messages: list[dict[str, Any]]) -> None:
    """Assert the message list satisfies the Messages API tool-pairing rules.

    - opens on a user turn,
    - every assistant `tool_use` id is answered by a `tool_result` in the
      IMMEDIATELY following user turn,
    - every `tool_result` references a `tool_use` in the IMMEDIATELY preceding
      assistant turn (i.e. no orphan from a dropped turn).
    """
    assert messages, "empty message list"
    assert messages[0]["role"] == "user", (
        f"must open on a user turn, got {messages[0]['role']!r}"
    )
    for idx, msg in enumerate(messages):
        content = msg.get("content")
        blocks = content if isinstance(content, list) else []
        if msg["role"] == "assistant":
            use_ids = [
                b["id"] for b in blocks
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]
            if use_ids:
                assert idx + 1 < len(messages), (
                    "assistant tool_use not followed by any turn"
                )
                nxt = messages[idx + 1]
                assert nxt["role"] == "user", (
                    "tool_use must be followed by a user tool_result turn"
                )
                result_ids = {
                    b["tool_use_id"] for b in nxt["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_result"
                }
                for uid in use_ids:
                    assert uid in result_ids, (
                        f"tool_use {uid!r} has no matching tool_result next"
                    )
        elif msg["role"] == "user":
            res_ids = [
                b["tool_use_id"] for b in blocks
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if res_ids:
                assert idx - 1 >= 0, "tool_result with no preceding turn"
                prev = messages[idx - 1]
                assert prev["role"] == "assistant", (
                    "tool_result not preceded by an assistant turn"
                )
                use_ids = {
                    b["id"] for b in prev["content"]
                    if isinstance(b, dict) and b.get("type") == "tool_use"
                }
                for rid in res_ids:
                    assert rid in use_ids, (
                        f"tool_result {rid!r} orphaned — its tool_use was dropped"
                    )


# ---------------------------------------------------------------------------
# Unit — compact_agent_trajectory
# ---------------------------------------------------------------------------

def test_below_threshold_returns_none():
    """Under COMPACT_THRESHOLD user turns, compaction is a no-op (None)."""
    traj = _make_trajectory(5)  # 6 user turns < 10
    assert not should_compact(traj)
    assert compact_agent_trajectory(traj, "cus_x") is None


def test_above_threshold_compacts_and_shrinks():
    """At/over threshold, returns (summary, new_working) shorter than input,
    with the original task turn preserved verbatim as the anchor."""
    traj = _make_trajectory(12)  # 13 user turns >= 10
    assert should_compact(traj)
    result = compact_agent_trajectory(traj, "cus_x")
    assert result is not None
    summary, new_working = result
    assert isinstance(summary, str) and summary
    assert len(new_working) < len(traj)
    # Anchor preserved (identity — same dict object at the head).
    assert new_working[0] is traj[0]
    assert new_working[0]["role"] == "user"


def test_compacted_view_is_valid_anthropic_sequence():
    """The compacted working must not orphan a tool_result and must open on
    the user task anchor — the core hazard C3 exists to avoid."""
    traj = _make_trajectory(20)
    result = compact_agent_trajectory(traj, "cus_x")
    assert result is not None
    _, new_working = result
    _assert_valid_anthropic_pairs(new_working)
    # The window opens on the task (user) then an assistant turn.
    assert new_working[1]["role"] == "assistant"


def test_summary_is_readable_not_stringified_json():
    """The rule-based summary records narrative + tool names, not the
    stringified block dicts `_rule_based_compact` would have produced."""
    traj = _make_trajectory(14)
    result = compact_agent_trajectory(traj, "cus_x")
    assert result is not None
    summary, _ = result
    # Carries the agent's narrative + the tool it called.
    assert "Step" in summary
    assert "read_file" in summary
    # Does NOT carry stringified block-dict noise.
    assert "{'type'" not in summary
    assert "'tool_use'" not in summary
    assert "'tool_result'" not in summary


def test_summary_accumulates_across_compactions():
    """A second compaction extends the prior summary rather than replacing it,
    so turns dropped by the first event aren't forgotten."""
    traj = _make_trajectory(12)
    res1 = compact_agent_trajectory(traj, "cus_x")
    assert res1 is not None
    summary1, working1 = res1

    # Grow the trajectory back over threshold and compact again, feeding the
    # prior summary in (what Agent.run does each iteration).
    grown = list(working1)
    for i in range(13, 18):  # +5 rounds -> back to 10 user turns
        grown.extend(_round(i))
    assert should_compact(grown)
    res2 = compact_agent_trajectory(grown, "cus_x", prior_summary=summary1)
    assert res2 is not None
    summary2, _ = res2

    assert summary2.startswith(summary1)
    assert len(summary2) > len(summary1)


def test_summarize_agent_turns_marks_tool_errors():
    """An errored tool_result in the dropped span is noted in the summary."""
    old = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Trying a risky edit."},
                {"type": "tool_use", "id": "tu_e", "name": "edit_file",
                 "input": {}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tu_e",
                 "content": "boom", "is_error": True},
            ],
        },
    ]
    summary = _summarize_agent_turns(old)
    assert "edit_file" in summary
    assert "error" in summary.lower()


# ---------------------------------------------------------------------------
# Integration — Agent.run wiring (flag off vs on)
# ---------------------------------------------------------------------------

def _script_long_run(fake: Any, n_tool_rounds: int = 10) -> None:
    """Script n knowledge_search rounds (empty bank -> non-error dict) then a
    final text turn. n_tool_rounds=10 -> compaction fires at the iteration-10
    model call when CC_AGENT_COMPACTION is on (the task + 9 prior tool_result
    user turns = 10 user turns, hitting the threshold)."""
    for i in range(1, n_tool_rounds + 1):
        fake.script_tool_use("knowledge_search", {"query": f"q{i}"}, f"tu_{i}")
    fake.script_text("done")


@pytest.mark.asyncio
async def test_compaction_off_when_disabled_sends_full_context(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
    monkeypatch: Any,
):
    """With the flag explicitly OFF, the agent never compacts: the model sees
    the full growing context and no summary header appears in the system
    prompt. (The launch default is ON per the 2026-07-02 flag-stance pass —
    this test pins the disabled behavior.)"""
    monkeypatch.setattr(config.settings, "agent_compaction", False)
    _script_long_run(fake_anthropic, 10)

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "start the task"}],
    )

    assert result["iterations"] == 11
    # Full trajectory returned: task + 10*(assistant+user) + final assistant.
    assert len(result["messages"]) == 22
    # The final model call saw the full, uncompacted context (task + 10 rounds).
    last_call = fake_anthropic.calls[-1]
    assert len(last_call["messages"]) == 21
    # No compaction summary folded into the system prompt.
    assert _COMPACTION_HEADER not in last_call["system"][0]["text"]


@pytest.mark.asyncio
async def test_compaction_on_bounds_model_view_but_returns_full(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
    monkeypatch: Any,
):
    """With the flag on, compaction fires mid-loop: the model-facing message
    list is bounded and the system prompt carries the summary header — but the
    RETURNED trajectory is still the complete, uncompacted history."""
    monkeypatch.setattr(config.settings, "agent_compaction", True)
    _script_long_run(fake_anthropic, 10)

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "start the task"}],
    )

    assert result["iterations"] == 11

    # Compaction fires at the iteration-10 model call: the working list (19
    # msgs pre-compaction) is replaced by a bounded view, and the system
    # prompt gains the summary header.
    compacted_call = fake_anthropic.calls[9]
    assert len(compacted_call["messages"]) < 19
    assert _COMPACTION_HEADER in compacted_call["system"][0]["text"]
    # And the bounded view is still a valid Anthropic sequence (no orphan).
    _assert_valid_anthropic_pairs(compacted_call["messages"])

    # The contract: the returned trajectory is the FULL history, not the
    # compacted view the model saw.
    assert len(result["messages"]) == 22
