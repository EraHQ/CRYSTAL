"""Agent event stream — the vocabulary + fan-out for live tool activity.

Block 2 slice 1 (ratified 2026-07-21): Q1=A in-band SSE with an emitter
seam on the agent loop; Q3=C agent-native named events. This module is
the ONE home for the vocabulary (the tier_signal pattern — one rendering
point): event-type constants, bounded wire summaries, and the fan-out
multiplexer the endpoint subscribes its consumers to (the SSE writer,
the agent_events recorder).

Wire discipline mirrors C4: BOUNDED summaries on the live ticker, full
fidelity in the terminal `run_completed` payload — whose `result` is the
IDENTICAL dict the non-streaming path returns (one contract, two
deliveries). `EVT_TEXT_DELTA` is DEFINED here but never emitted in
slice 1; slice 2 (seam token streaming, Q2=C) makes it fire without a
wire-contract change.

The loop's `run_completed` / `error` events are NOT emitted by the Agent
itself — the endpoint's pipeline emits them after `finalize_agent_turn`,
so the terminal result carries `mcr` exactly like the non-streaming
response. The Agent emits activity (`run_started` .. `tool_result`) and
`notice` only.
"""
from __future__ import annotations

import json
import time
from typing import Any, Awaitable, Callable

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Vocabulary (Q3=C — agent-native named events)
# ---------------------------------------------------------------------------

EVT_RUN_STARTED = "run_started"
"""First event of a run: {model, max_iterations, message_count}."""

EVT_ITERATION_STARTED = "iteration_started"
"""Top of each loop pass, before the model call: {iteration}."""

EVT_TOOL_CALLS = "tool_calls"
"""The model issued tool calls this iteration:
{iteration, calls: [{tool_use_id, name, input_summary}]}."""

EVT_TOOL_RESULT = "tool_result"
"""One tool finished: {iteration, tool_use_id, name, duration_ms,
is_error, output_head}. `output_head` is BOUNDED (ticker copy); the
full output lives in the terminal result's `tool_calls` log."""

EVT_NOTICE = "notice"
"""Loop condition worth surfacing: {kind, ...}. Kinds emitted in
slice 1: compacted, tool_call_truncated (H2), deadline,
max_iterations, model_error."""

EVT_TEXT_DELTA = "text_delta"
"""RESERVED (Q2=C). Defined in slice 1, emitted only by slice 2 when
the provider seam streams model tokens: {iteration, text}."""

EVT_RUN_COMPLETED = "run_completed"
"""Terminal success: {result} — the full Agent-result dict verbatim,
mcr included (emitted by the endpoint pipeline after finalize)."""

EVT_ERROR = "error"
"""Terminal pipeline failure: {error, error_type}. Loop-internal model
errors are NOT this — they surface as notice(kind=model_error) and the
run still terminates with run_completed (stop_reason='error')."""

TERMINAL_EVENTS = frozenset({EVT_RUN_COMPLETED, EVT_ERROR})
"""Events that end an SSE stream. A viewer stops reading after one of
these; the detached run itself already finished by then (Q5=C)."""


# ---------------------------------------------------------------------------
# Bounded wire summaries (the C4 pattern: cap the live copy, never the
# record)
# ---------------------------------------------------------------------------

INPUT_SUMMARY_MAX_CHARS = 200
"""Cap for a tool call's input summary on the wire."""

OUTPUT_HEAD_MAX_CHARS = 400
"""Cap for a tool result's output head on the wire."""

LABEL_MAX_CHARS = 120
"""Cap for humanized labels — matches the session bookends' label cap."""


def summarize_tool_input(
    tool_input: Any, max_chars: int = INPUT_SUMMARY_MAX_CHARS,
) -> str:
    """One bounded line describing a tool call's input.

    JSON when serializable, str() otherwise — never raises (an event
    summary must not be able to break the loop)."""
    try:
        s = json.dumps(tool_input, default=str)
    except Exception:  # noqa: BLE001
        s = str(tool_input)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "…"


def bound_output_head(
    output: Any, max_chars: int = OUTPUT_HEAD_MAX_CHARS,
) -> str:
    """The head of a tool output for the ticker. Full output stays in
    the terminal result's tool_calls log (zero-truncation applies to
    the record, not the live summary — the bench's C4 discipline)."""
    if isinstance(output, str):
        s = output
    else:
        try:
            s = json.dumps(output, default=str)
        except Exception:  # noqa: BLE001
            s = str(output)
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + f"… [+{len(s) - max_chars} chars]"


def humanize_call(
    tool_name: str, tool_input: Any, max_chars: int = LABEL_MAX_CHARS,
) -> str:
    """Human-readable one-liner for a tool call — the Q4 recorder's
    `label`, playing the same role `style.humanize_call` plays for the
    coding-agent surfaces (P1c)."""
    summary = summarize_tool_input(tool_input, max_chars=max_chars)
    label = f"{tool_name} · {summary}" if summary not in ("{}", "null") \
        else tool_name
    return label[:max_chars]


# ---------------------------------------------------------------------------
# Fan-out multiplexer
# ---------------------------------------------------------------------------

Subscriber = Callable[[str, dict[str, Any]], Awaitable[None]]
"""An async callable receiving (event_type, payload)."""


class AgentEventMux:
    """Fan one event stream out to N subscribers, each individually
    fail-safe — a subscriber exception logs a warning and NEVER touches
    the run or the other subscribers (the bookends' best-effort
    posture, applied per-subscriber).

    `emit` is shaped to be passed directly as the Agent's `emit` seam.
    Every payload is stamped with `ts` (epoch ms) at emit time so
    viewers can compute durations without trusting their own clocks.
    """

    def __init__(self) -> None:
        self._subscribers: list[Subscriber] = []

    def subscribe(self, fn: Subscriber) -> None:
        self._subscribers.append(fn)

    async def emit(self, event_type: str, payload: dict[str, Any]) -> None:
        stamped = {"ts": int(time.time() * 1000), **payload}
        for fn in self._subscribers:
            try:
                await fn(event_type, stamped)
            except Exception as e:  # noqa: BLE001 — observability must
                # never break the run (per-subscriber isolation).
                logger.warning(
                    "agent.event_subscriber_failed",
                    event_type=event_type,
                    error=str(e),
                    error_type=type(e).__name__,
                )
