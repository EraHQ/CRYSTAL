"""Block 2 slice 1 — agent event stream + streaming delivery (2026-07-21).

Ratified map: Q1=A in-band SSE with an emitter seam on the loop; Q2=C
slice 1 activity-only with `text_delta` reserved; Q3=C agent-native
vocabulary (agent/events.py is its one home); Q4=A agent_events parity
(tool_called rows in the P1c shape for every HTTP agent turn); Q5=C the
run outlives the viewer (detached task; disconnect never cancels).

Covers: the pinned event sequence from the loop, emit=None byte-identity,
per-subscriber isolation, bounded-wire/full-record (C4 discipline),
notices (H2 truncation, max_iterations), the Q4 recorder's row shape
against the real store, SSE frame rendering, and the detached-run
survival contract. The full run_agent_messages happy path needs a live
upstream + app.state (see test_agent_api_session's note) — the endpoint
pipeline's pieces are pinned individually here instead; the loop emits
NO terminal event by contract (the endpoint pipeline emits run_completed
after finalize so the streamed result carries mcr, identical to the
non-streaming response).

R14 note: verified by pytest; describes expected behavior, not yet run
at authoring time.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from crystal_cache.agent import Agent
from crystal_cache.agent.events import (
    EVT_ITERATION_STARTED,
    EVT_NOTICE,
    EVT_RUN_STARTED,
    EVT_TEXT_DELTA,
    EVT_TOOL_CALLS,
    EVT_TOOL_RESULT,
    TERMINAL_EVENTS,
    AgentEventMux,
    bound_output_head,
    humanize_call,
    summarize_tool_input,
)
from crystal_cache.endpoints.agent import (
    _make_agent_events_recorder,
    _run_detached,
    _sse_frame,
)
from fakes import FakeAnthropic, FakeResponse, FakeToolUseBlock, FakeUsage


def _collector() -> tuple[list[tuple[str, dict[str, Any]]], Any]:
    events: list[tuple[str, dict[str, Any]]] = []

    async def _sub(event_type: str, payload: dict[str, Any]) -> None:
        events.append((event_type, payload))

    return events, _sub


# ---------------------------------------------------------------------------
# emit=None — byte-identical baseline (the seam's default-off contract)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_none_runs_unchanged(customer, tool_state, fake_anthropic):
    fake_anthropic.script_tool_use("knowledge_search", {"query": "q"}, "tu_1")
    fake_anthropic.script_text("done")
    agent = Agent(customer=customer, llm=fake_anthropic, tool_state=tool_state)
    result = await agent.run([{"role": "user", "content": "hi"}])
    assert result["final_text"] == "done"
    assert result["stop_reason"] == "end_turn"
    assert result["iterations"] == 2
    # duration_ms is the one additive key on tool_calls log entries.
    assert result["tool_calls"][0]["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# The pinned event sequence
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_event_sequence_tool_then_text(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_tool_use(
        "knowledge_search", {"query": "sparse keys"}, "tu_1",
    )
    fake_anthropic.script_text("here you go")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])

    kinds = [e for e, _ in events]
    assert kinds == [
        EVT_RUN_STARTED,
        EVT_ITERATION_STARTED,   # iteration 1 -> tool_use
        EVT_TOOL_CALLS,
        EVT_TOOL_RESULT,
        EVT_ITERATION_STARTED,   # iteration 2 -> final text
    ]
    # The loop emits NO terminal event (endpoint pipeline owns it) and
    # never text_delta in slice 1 (Q2=C reservation).
    assert not any(k in TERMINAL_EVENTS for k in kinds)
    assert EVT_TEXT_DELTA not in kinds

    rs = events[0][1]
    assert rs["max_iterations"] == agent.max_iterations
    assert rs["message_count"] == 1
    assert "ts" in rs
    tc = events[2][1]
    assert tc["iteration"] == 1
    assert tc["calls"][0]["name"] == "knowledge_search"
    assert tc["calls"][0]["tool_use_id"] == "tu_1"
    assert "sparse keys" in tc["calls"][0]["input_summary"]
    tr = events[3][1]
    assert tr["name"] == "knowledge_search"
    assert tr["tool_use_id"] == "tu_1"
    assert tr["duration_ms"] >= 0
    assert "output_head" in tr
    assert result["final_text"] == "here you go"


# ---------------------------------------------------------------------------
# Per-subscriber isolation (the mux's fail-safe posture)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_broken_subscriber_never_touches_run_or_peers(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_tool_use("knowledge_search", {"query": "x"}, "tu_1")
    fake_anthropic.script_text("fine")
    mux = AgentEventMux()

    async def _boom(event_type: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("subscriber down")

    events, sub = _collector()
    mux.subscribe(_boom)   # first, so its failure precedes the collector
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    assert result["final_text"] == "fine"
    assert len(events) == 5  # the peer saw everything


# ---------------------------------------------------------------------------
# Bounded wire, full record (the C4 discipline on the event stream)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_wire_head_bounded_log_full(
    customer, tool_state, fake_anthropic,
):
    fake_anthropic.script_tool_use(
        "knowledge_search", {"query": "anything"}, "tu_big",
    )
    fake_anthropic.script_text("ok")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])

    tr = [p for e, p in events if e == EVT_TOOL_RESULT][0]
    full = result["tool_calls"][0]["output"]
    # The wire head is bounded regardless of output size; the log entry
    # holds the tool's full output object untouched.
    assert len(tr["output_head"]) <= 450
    assert bound_output_head(full) == tr["output_head"]


# ---------------------------------------------------------------------------
# Notices
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_h2_truncation_notice(customer, tool_state, fake_anthropic):
    # A tool_use response cut at max_tokens — H2 refuses dispatch and
    # the loop announces it as a notice, then recovers.
    fake_anthropic._scripted.append(FakeResponse(
        content=[FakeToolUseBlock(
            id="tu_t", name="knowledge_search", input={},
        )],
        stop_reason="max_tokens",
        usage=FakeUsage(input_tokens=10, output_tokens=10),
    ))
    fake_anthropic.script_text("recovered")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    notices = [p for e, p in events if e == EVT_NOTICE]
    assert any(n["kind"] == "tool_call_truncated" for n in notices)
    assert result["final_text"] == "recovered"


@pytest.mark.asyncio
async def test_max_iterations_notice(customer, tool_state, fake_anthropic):
    for i in range(3):
        fake_anthropic.script_tool_use(
            "knowledge_search", {"query": str(i)}, f"tu_{i}",
        )
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        max_iterations=3, emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    notices = [p for e, p in events if e == EVT_NOTICE]
    assert any(n["kind"] == "max_iterations" for n in notices)
    assert result["stop_reason"] == "max_iterations"


# ---------------------------------------------------------------------------
# Q4=A — the agent_events recorder (P1c parity, against the real store)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_recorder_writes_tool_called_rows_in_p1c_shape(store, customer):
    recorder = _make_agent_events_recorder(
        store, session_id="crysapi_stream_1", team_id=customer.id,
    )
    # The recorder correlates the call's input summary (tool_calls) with
    # the result's duration + status (tool_result).
    await recorder(EVT_TOOL_CALLS, {
        "iteration": 1,
        "calls": [{
            "tool_use_id": "tu_1", "name": "knowledge_search",
            "input_summary": '{"query": "sparse keys"}',
        }],
    })
    await recorder(EVT_RUN_STARTED, {"model": "m"})  # ignored — no row
    await recorder(EVT_TOOL_RESULT, {
        "iteration": 1, "tool_use_id": "tu_1", "name": "knowledge_search",
        "duration_ms": 123, "is_error": False, "output_head": "…",
    })

    events = await store.list_events_for_session("crysapi_stream_1")
    assert len(events) == 1
    row = events[0]
    assert row["event_type"] == "tool_called"          # P1c shape
    assert row["phase"] == "tool"
    assert row["status"] == "ok"
    assert row["payload"] == {"tool": "knowledge_search"}
    assert row["label"].startswith("knowledge_search · ")
    assert "sparse keys" in row["label"]
    assert row["duration_ms"] == 123
    assert row["team_id"] == customer.id


@pytest.mark.asyncio
async def test_recorder_error_status_and_bare_label(store, customer):
    recorder = _make_agent_events_recorder(
        store, session_id="crysapi_stream_2", team_id=customer.id,
    )
    # A tool_result with no prior tool_calls entry (e.g. a subscriber
    # attached mid-run) still records, labeled by tool name alone.
    await recorder(EVT_TOOL_RESULT, {
        "iteration": 2, "tool_use_id": "tu_x", "name": "web_search",
        "duration_ms": 7, "is_error": True, "output_head": "boom",
    })
    events = await store.list_events_for_session("crysapi_stream_2")
    assert len(events) == 1
    assert events[0]["status"] == "error"
    assert events[0]["label"] == "web_search"


# ---------------------------------------------------------------------------
# SSE delivery mechanics
# ---------------------------------------------------------------------------

def test_sse_frame_shape():
    frame = _sse_frame("tool_result", {
        "name": "knowledge_search", "duration_ms": 42,
    })
    lines = frame.split("\n")
    assert lines[0] == "event: tool_result"
    assert lines[1].startswith("data: ")
    payload = json.loads(lines[1][len("data: "):])
    assert payload["name"] == "knowledge_search"
    assert frame.endswith("\n\n")


def test_sse_frame_never_raises_on_unserializable():
    class _W:
        pass
    frame = _sse_frame("notice", {"obj": _W()})  # json default=str
    assert frame.startswith("event: notice\n")


# ---------------------------------------------------------------------------
# Q5=C — the detached run outlives its viewer
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_detached_run_survives_viewer_cancellation():
    """Cancel the awaiting wrapper (the tab closed); the run completes
    and its side effect — standing in for finalize's query_log write,
    the S7 history another device picks up — lands anyway."""
    landed = asyncio.Event()

    async def _pipeline() -> dict[str, Any]:
        await asyncio.sleep(0.05)
        landed.set()
        return {"final_text": "done"}

    task = _run_detached(_pipeline())

    async def _viewer() -> dict[str, Any]:
        return await asyncio.shield(task)

    viewer = asyncio.create_task(_viewer())
    await asyncio.sleep(0.01)
    viewer.cancel()
    with pytest.raises(asyncio.CancelledError):
        await viewer

    result = await task
    assert result["final_text"] == "done"
    assert landed.is_set()


@pytest.mark.asyncio
async def test_detached_run_failure_reraises_to_a_live_awaiter():
    async def _boom() -> None:
        raise RuntimeError("pipeline down")

    task = _run_detached(_boom())
    with pytest.raises(RuntimeError):
        await task
    await asyncio.sleep(0)  # done-callback ran; no unretrieved-exc warn


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

def test_summary_and_label_helpers_bounded_and_safe():
    assert summarize_tool_input({"q": "x"}) == '{"q": "x"}'
    assert len(summarize_tool_input({"q": "y" * 500})) <= 201
    head = bound_output_head("k" * 1000)
    assert len(head) < 500 and "chars]" in head
    assert humanize_call("knowledge_search", {}) == "knowledge_search"
    assert humanize_call(
        "read_file", {"path": "a.py"},
    ).startswith("read_file · ")

    class _Weird:
        pass
    # An unserializable input must never raise out of an event summary.
    s = summarize_tool_input({"obj": _Weird()})
    assert isinstance(s, str) and s


# ---------------------------------------------------------------------------
# Slice 2 — token streaming (Q6=B: only where a viewer consumes it)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_stream_tokens_emits_ordered_text_deltas(
    customer, tool_state, fake_anthropic,
):
    """stream_tokens=True + emit wired: every model call streams; deltas
    carry {iteration, text}, concatenate exactly to each response's
    text, and land BEFORE the events emitted after the call returns
    (loop-FIFO ordering). The final result is unchanged in shape."""
    fake_anthropic.script_tool_use(
        "knowledge_search", {"query": "x"}, "tu_1",
        preamble_text="Let me check.",
    )
    fake_anthropic.script_text("final answer")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit, stream_tokens=True,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])

    deltas = [p for e, p in events if e == EVT_TEXT_DELTA]
    assert deltas, "streaming run must emit text deltas"
    it1 = "".join(d["text"] for d in deltas if d["iteration"] == 1)
    it2 = "".join(d["text"] for d in deltas if d["iteration"] == 2)
    assert it1 == "Let me check."
    assert it2 == "final answer"
    assert fake_anthropic.stream_calls == 2
    assert result["final_text"] == "final answer"
    kinds = [e for e, _ in events]
    assert kinds.index(EVT_TEXT_DELTA) < kinds.index(EVT_TOOL_CALLS)
    assert not any(k in TERMINAL_EVENTS for k in kinds)


@pytest.mark.asyncio
async def test_stream_tokens_default_off_never_streams(
    customer, tool_state, fake_anthropic,
):
    """The Q6=B pin: emit wired but stream_tokens at its default — the
    proven complete_messages path, zero deltas, zero stream calls."""
    fake_anthropic.script_text("plain")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        emit=mux.emit,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    assert fake_anthropic.stream_calls == 0
    assert not [e for e, _ in events if e == EVT_TEXT_DELTA]
    assert result["final_text"] == "plain"


@pytest.mark.asyncio
async def test_stream_tokens_without_emit_uses_plain_path(
    customer, tool_state, fake_anthropic,
):
    """stream_tokens=True but no emitter — no viewer exists, so the
    plain call path runs (deltas would have nowhere to go)."""
    fake_anthropic.script_text("plain")
    agent = Agent(
        customer=customer, llm=fake_anthropic, tool_state=tool_state,
        stream_tokens=True,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    assert fake_anthropic.stream_calls == 0
    assert result["final_text"] == "plain"


@pytest.mark.asyncio
async def test_seam_without_stream_messages_falls_back_cleanly(
    customer, tool_state, fake_anthropic,
):
    """A seam lacking stream_messages (older client, other provider
    wrapper) degrades honestly: complete_messages runs, no deltas, the
    run is otherwise identical."""
    class _NoStream:
        def __init__(self, inner):
            self._inner = inner

        @property
        def provider(self):
            return "anthropic"

        def complete_messages(self, **kw):
            return self._inner.complete_messages(**kw)

    fake_anthropic.script_text("fallback fine")
    mux = AgentEventMux()
    events, sub = _collector()
    mux.subscribe(sub)
    agent = Agent(
        customer=customer, llm=_NoStream(fake_anthropic),
        tool_state=tool_state, emit=mux.emit, stream_tokens=True,
    )
    result = await agent.run([{"role": "user", "content": "hi"}])
    assert not [e for e, _ in events if e == EVT_TEXT_DELTA]
    assert result["final_text"] == "fallback fine"
