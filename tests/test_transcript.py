"""CRYS conversation persistence — pure transcript helpers (P5 track a, A2).

These cover the only A2 logic with real edge cases: capping a resumable
transcript WITHOUT splitting a tool_use/tool_result pair (an invalid history
the API rejects on /resume). The rest of the CLI wiring is validated live;
this is the part R14 says to pin.

crystal_code lives under CRYS/ (a segmented package, not on the
pytest path), and its __init__ is dependency-free, so we add CRYS to
sys.path here and import the light transcript module directly — without
dragging in cli.py's heavy agent stack.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_CODING_AGENT = Path(__file__).resolve().parents[1] / "CRYS"
if str(_CODING_AGENT) not in sys.path:
    sys.path.insert(0, str(_CODING_AGENT))

from crystal_code.transcript import (  # noqa: E402
    cap_transcript,
    format_ago,
    is_user_turn_start,
)

# A realistic trajectory: a tool-using turn (assistant tool_use → user
# tool_result) then a plain turn. Indices noted for the cap assertions.
_USER_Q1 = {"role": "user", "content": "q1"}                                    # 0
_ASSIST_TOOLUSE = {"role": "assistant", "content": [                            # 1
    {"type": "tool_use", "id": "t1", "name": "read", "input": {}},
]}
_USER_TOOLRESULT = {"role": "user", "content": [                                # 2
    {"type": "tool_result", "tool_use_id": "t1", "content": "ok"},
]}
_ASSIST_A1 = {"role": "assistant", "content": "a1"}                             # 3
_USER_Q2 = {"role": "user", "content": "q2"}                                    # 4
_ASSIST_A2 = {"role": "assistant", "content": "a2"}                            # 5

_CONVO = [_USER_Q1, _ASSIST_TOOLUSE, _USER_TOOLRESULT, _ASSIST_A1, _USER_Q2, _ASSIST_A2]


# --- is_user_turn_start ---

def test_is_user_turn_start_plain_text():
    assert is_user_turn_start(_USER_Q1) is True


def test_is_user_turn_start_list_with_text():
    assert is_user_turn_start(
        {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    ) is True


def test_tool_result_carrier_is_not_a_turn_start():
    # A user-role message that is ONLY a tool_result is not a real turn.
    assert is_user_turn_start(_USER_TOOLRESULT) is False


def test_assistant_is_not_a_turn_start():
    assert is_user_turn_start(_ASSIST_A1) is False
    assert is_user_turn_start("not a dict") is False


# --- cap_transcript ---

def test_cap_returns_input_when_within_limit():
    assert cap_transcript(_CONVO, 100) is _CONVO


def test_cap_snaps_to_clean_user_turn():
    # Window of the last 3 = [a1, q2, a2]; first real user turn is q2.
    assert cap_transcript(_CONVO, 3) == [_USER_Q2, _ASSIST_A2]


def test_cap_skips_a_tool_result_at_the_window_edge():
    # Window of the last 4 = [tool_result, a1, q2, a2]; must NOT start on the
    # tool_result carrier (that orphans t1's tool_use) — advances to q2.
    capped = cap_transcript(_CONVO, 4)
    assert capped[0] == _USER_Q2
    assert is_user_turn_start(capped[0]) is True


def test_cap_backs_off_when_window_has_no_clean_boundary():
    # One long tool-only burst: the last-2 window is [tool_use, tool_result],
    # no user turn — back off to the last real user turn in full history (q1),
    # never splitting a pair.
    burst = [
        _USER_Q1,
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "1"}]},
        {"role": "assistant", "content": [{"type": "tool_use", "id": "t2", "name": "x", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "2"}]},
    ]
    capped = cap_transcript(burst, 2)
    assert capped[0] == _USER_Q1  # backed off to the only real user turn
    assert is_user_turn_start(capped[0]) is True


# --- format_ago ---

def test_format_ago_buckets():
    now = datetime.now(timezone.utc)
    assert format_ago(now) == "just now"
    assert format_ago(now - timedelta(minutes=10)) == "~10m ago"
    assert format_ago(now - timedelta(hours=3)) == "~3h ago"
    assert format_ago(now - timedelta(days=2)) == "~2d ago"


def test_format_ago_is_safe_on_garbage():
    assert format_ago(None) == "earlier"
    assert format_ago("not a datetime") == "earlier"
