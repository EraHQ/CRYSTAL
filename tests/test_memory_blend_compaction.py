"""Memory-blend Increment 1 — conversation compaction.

Covers `retrieval/compaction.py` (D-MB1, see docs/MEMORY_BLEND_PLAN.md).
Pure unit tests, no network: mem0 is disabled in the test environment, so
`compact_conversation` exercises the rule-based summary path. The
`_mem0_compact` monkeypatch makes that deterministic regardless of any
stray module state.
"""
from __future__ import annotations

import crystal_cache.retrieval.compaction as comp
from crystal_cache.retrieval.compaction import (
    COMPACT_THRESHOLD,
    KEEP_RECENT_TURNS,
    compact_conversation,
    should_compact,
    _rule_based_compact,
)


def _convo(user_turns: int) -> list[dict]:
    """Build a synthetic user/assistant conversation with `user_turns` pairs."""
    msgs: list[dict] = []
    for i in range(user_turns):
        msgs.append({"role": "user", "content": f"question {i}"})
        msgs.append(
            {"role": "assistant", "content": f"answer {i} line one\nmore detail"}
        )
    return msgs


def test_should_compact_threshold():
    """should_compact fires at exactly COMPACT_THRESHOLD user turns."""
    assert should_compact(_convo(COMPACT_THRESHOLD - 1)) is False
    assert should_compact(_convo(COMPACT_THRESHOLD)) is True


def test_compact_below_threshold_is_noop():
    """Below threshold, the message list is returned unchanged."""
    msgs = _convo(3)
    assert compact_conversation(msgs, customer_id="cus_test") == msgs


def test_compact_keeps_recent_and_summarizes(monkeypatch):
    """Above threshold: pre-existing system msg preserved, a summary system
    message is inserted, the last KEEP_RECENT_TURNS*2 turns are verbatim,
    and the total shrinks."""
    # Force the rule-based path (mem0 is off in tests anyway).
    monkeypatch.setattr(comp, "_mem0_compact", lambda *a, **k: None)

    sys_msg = {"role": "system", "content": "You are a helpful assistant."}
    msgs = [sys_msg] + _convo(12)

    out = compact_conversation(msgs, customer_id="cus_test", sequence_id="seq_1")

    # The caller's own system message survives.
    assert out[0] == sys_msg
    # A compaction summary system message is present.
    assert any(
        m.get("role") == "system"
        and "summary of the earlier part" in m.get("content", "")
        for m in out
    )
    # The most recent turns are preserved verbatim at the tail.
    keep = KEEP_RECENT_TURNS * 2
    assert out[-keep:] == msgs[-keep:]
    # Compaction actually reduced the message count.
    assert len(out) < len(msgs)


def test_rule_based_compact_produces_qa_summary():
    """The rule-based fallback emits Q/A lines for the old turns."""
    summary = _rule_based_compact(_convo(2))
    assert "Q: question 0" in summary
    assert "A:" in summary
