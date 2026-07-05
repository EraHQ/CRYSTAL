"""Pure transcript helpers for CRYS conversation persistence (P5 track a).

Deliberately dependency-free (only datetime + typing) so it is unit-testable
in isolation — importing cli.py drags in the whole agent stack (MCP client,
Anthropic, the encoder), which is why the rest of the CLI is validated live.
The part with real edge cases is `cap_transcript`: a resumed transcript must
be a VALID message history, so capping must never start mid tool_use/
tool_result pair. cli.py imports these; the I/O around them (recap printing,
the store upsert) stays in cli.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any


def is_user_turn_start(msg: Any) -> bool:
    """True if `msg` is a genuine user turn (a text/image message), not a
    tool_result carrier. The boundary `cap_transcript` snaps to.

    Anthropic puts tool_result blocks in a user-role message, so role alone is
    not enough — a real user turn has at least one non-tool_result block (or a
    plain string content).
    """
    if not isinstance(msg, dict) or msg.get("role") != "user":
        return False
    content = msg.get("content")
    if isinstance(content, str):
        return True
    if isinstance(content, list):
        return any(
            not (isinstance(b, dict) and b.get("type") == "tool_result")
            for b in content
        )
    # Unknown shape: treat as a turn start rather than risk slicing past it.
    return True


def cap_transcript(messages: list, max_messages: int) -> list:
    """Tail-cap the transcript WITHOUT splitting a tool_use/tool_result pair.

    A naive `messages[-N:]` can start on an assistant tool_use or a user
    tool_result, which the API rejects on resume. So take the last-N window
    but advance its start to the first genuine user turn; if the window has
    none (one huge turn of tool traffic), back off to the last user-turn start
    in the full history. Starting at a real user turn keeps every pair after it
    intact. Returns the input unchanged when it is already within the cap.
    """
    if len(messages) <= max_messages:
        return messages
    window = messages[-max_messages:]
    for i, m in enumerate(window):
        if is_user_turn_start(m):
            return window[i:]
    # No clean boundary in the window — back off to the last user-turn start
    # in the full history so we never split a pair.
    for i in range(len(messages) - 1, -1, -1):
        if is_user_turn_start(messages[i]):
            return messages[i:]
    return messages


def format_ago(then: Any) -> str:
    """Human 'time ago' for the launch recap, from a tz-aware datetime.
    Returns 'earlier' on any malformed input (never raises)."""
    try:
        secs = int((datetime.now(timezone.utc) - then).total_seconds())
    except Exception:
        return "earlier"
    if secs < 90:
        return "just now"
    mins = secs // 60
    if mins < 90:
        return f"~{mins}m ago"
    hours = mins // 60
    if hours < 36:
        return f"~{hours}h ago"
    return f"~{hours // 24}d ago"
