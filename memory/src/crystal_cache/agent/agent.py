"""Agent — the conversation loop.

Per P0.25: receive a message list, send to the agent's LLM with the
current tool registry, parse tool calls, execute them against the
registry implementations, feed results back, recurse until the LLM
emits a final message. Configurable max-iterations cap (default 12).

Per P0.17: stateless. The full message history is passed in each
call; the Agent does not hold conversation state between requests.
Mem0 (`retrieval/mem0_session.py`) holds session memory separately,
addressed by sequence_id.

Per P0.15: the agent's controlling LLM is configurable via
settings (`CC_AGENT_MODEL`). Hosted deployment ships with
Claude Sonnet 4.5 (`claude-sonnet-4-5-20250929`); the Haiku
swap is `claude-haiku-4-5-20251001`.

Per P0.16: streaming uses Anthropic Messages API SSE shape.
Phase 7.5 ships the non-streaming path first; streaming wrapper is
opt-in and lands before Phase 8 if time. Both paths share the same
loop body; only the response emission differs.

CONTEXT INJECTION:
The agent injects shared state into the tool registry once at
construction time (via `set_tool_state(...)`). All tools read this
state lazily inside their implementations so import-time
registration is decoupled from request-time state. The state dict
carries: store, vector_store, fact_vector_store, encoder,
decomposer.

REQUEST/RESPONSE SHAPE (P0.27):
The endpoint `/v1/agent/messages` accepts the Anthropic Messages
API request body. The Agent class itself works with the same
shape: messages = [{role, content}], where content can be a string
or a list of content blocks (text + tool_use + tool_result).
"""
from __future__ import annotations

import json
import time
import uuid
from typing import TYPE_CHECKING, Any, Optional

import structlog

from .events import (
    EVT_ITERATION_STARTED,
    EVT_NOTICE,
    EVT_RUN_STARTED,
    EVT_TEXT_DELTA,
    EVT_TOOL_CALLS,
    EVT_TOOL_RESULT,
    bound_output_head,
    summarize_tool_input,
)
from .system_prompt import build_system_prompt
from .tool_registry import Tool, ToolRegistry, get_registry, import_all_tools
from .tools.retrievers import set_tool_state

if TYPE_CHECKING:
    from ..models import Customer

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_MAX_ITERATIONS = 24
"""Fallback max tool-call iterations when neither the constructor arg
nor settings provide one. Raised 12 → 24 (June 2026): the first
real-world CRYS session hit 12 mid-debug-loop — organic fix→test
cycles burn 2-3 iterations each. Deployment-tunable via
CC_AGENT_MAX_ITERATIONS (settings.agent_max_iterations)."""

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
"""Default agent model (P0.15). Overridable per-request via the
`model` field of the request body and per-deployment via the
`CC_AGENT_MODEL` env var (settings.agent_model)."""

DEFAULT_MAX_TOKENS = 8192
"""Fallback response max_tokens when neither the constructor arg nor
settings provide one. Raised 4096 -> 8192 (June 2026): 4096 was too
small for file writes — a large write_file truncated mid-tool-call, the
loop dispatched the partial call, and the agent spiralled (2026-06-13
MMORPG session). 8192 is Sonnet 4.5's standard output ceiling.
Deployment-tunable via CC_AGENT_MAX_TOKENS (settings.agent_max_tokens)."""


# ---------------------------------------------------------------------------
# C1 — prompt caching helpers (CC-D1 "both", CC-D2 5-min ephemeral)
# ---------------------------------------------------------------------------

# Ephemeral (5-min) cache breakpoint. The 5-min TTL needs no beta header, and
# a back-to-back agent loop keeps the prefix warm.
_EPHEMERAL_CACHE = {"type": "ephemeral"}


def _system_blocks(system_text: str) -> list[dict[str, Any]]:
    """The system prompt as a single cached text block (C1).

    The system prompt is identical across a customer's requests, so caching
    the tools+system prefix makes every iteration after the first a cache
    read. Returned in the Messages API's content-block form so `cache_control`
    can attach — a bare string can't carry a breakpoint.
    """
    return [{
        "type": "text",
        "text": system_text,
        "cache_control": _EPHEMERAL_CACHE,
    }]


def _messages_with_cache_breakpoint(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Copy of `messages` with an ephemeral breakpoint on the last message's
    final content block (C1, CC-D1 "both").

    This caches the conversation prefix incrementally: each iteration the
    breakpoint moves to the new last message, so the prior prefix is a cache
    read and only the newest turn is written. The marker is applied to a COPY
    so the agent's persisted `working` list never accumulates breakpoints —
    Anthropic caps a request at 4, and `working` is re-sent every iteration.

    A string-content message is converted to a one-element text block (the
    only shape that can carry `cache_control`); a block-list message gets the
    marker on its last block (text / tool_use / tool_result all accept it).
    """
    if not messages:
        return messages
    out = list(messages)
    last = dict(out[-1])
    content = last.get("content")
    if isinstance(content, str):
        last["content"] = [{
            "type": "text",
            "text": content,
            "cache_control": _EPHEMERAL_CACHE,
        }]
    elif isinstance(content, list) and content:
        blocks = list(content)
        marked = dict(blocks[-1])
        marked["cache_control"] = _EPHEMERAL_CACHE
        blocks[-1] = marked
        last["content"] = blocks
    else:
        return messages
    out[-1] = last
    return out


# ---------------------------------------------------------------------------
# C3 — agent-loop compaction (CC-D5/6/7)
# ---------------------------------------------------------------------------

_COMPACTION_HEADER = (
    "## Earlier progress in this task (summarized — recent turns follow "
    "verbatim below)"
)
"""Heading the folded compaction summary sits under in the system prompt.
The summary cannot ride in the `messages` array (Anthropic rejects
system-role turns there), so it folds into `system`; this header tells the
model the block is compressed history, not instructions."""


# ---------------------------------------------------------------------------
# C4 — tool-output trimming (CC-D8: cap, head+tail)
# ---------------------------------------------------------------------------


def _cap_tool_output(content: str, max_chars: int) -> str:
    """Bound a tool_result's content for the model-facing trajectory (C4).

    Keeps the head and tail — the parts that carry a result's structure and its
    most recent lines — around a truncation marker, dropping the middle. The
    full untrimmed output is preserved in `tool_calls_log`, so this only shapes
    what the model re-reads each iteration, not what's recorded. `max_chars`
    bounds the RETAINED content (head + tail); the marker is added on top.
    `max_chars <= 0` disables capping. Never returns a string longer than the
    input (a barely-over output where the marker would cost more than it saves
    is left as-is).
    """
    if max_chars <= 0 or len(content) <= max_chars:
        return content
    head_len = (max_chars * 3) // 5
    tail_len = max_chars - head_len
    dropped = len(content) - head_len - tail_len
    nl = chr(10)
    marker = f"{nl}{nl}...[{dropped} chars truncated]...{nl}{nl}"
    tail = content[-tail_len:] if tail_len > 0 else ""
    capped = content[:head_len] + marker + tail
    return capped if len(capped) < len(content) else content


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    """The agent's conversation loop.

    One Agent instance per request is the typical pattern; the
    instance carries no per-conversation state (P0.17). The shared
    state injected into the tool registry is process-wide, set
    once at app startup.

    Usage (typical):

        agent = Agent(
            customer=customer,
            llm=get_llm_client(),
            tool_state={
                "store": store,
                "vector_store": vector_store,
                "fact_vector_store": fact_vector_store,
                "encoder": encoder,
                "decomposer": decomposer,
            },
        )
        response = await agent.run(messages=[{"role": "user", "content": "..."}])
    """

    def __init__(
        self,
        customer: "Customer",
        llm: Any,
        tool_state: dict[str, Any],
        *,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_iterations: Optional[int] = None,
        registry: Optional[ToolRegistry] = None,
        sequence_id: Optional[str] = None,
        intercept: Optional[Any] = None,
        after_tool: Optional[Any] = None,
        emit: Optional[Any] = None,
        stream_tokens: bool = False,
    ) -> None:
        self.customer = customer
        self.llm = llm
        # Controlling model: explicit arg > CC_AGENT_MODEL (settings) >
        # provider default. The endpoint resolves the per-conversation sticky
        # model and passes it as `model`. Under the anthropic provider the
        # house DEFAULT_MODEL is the final fallback; under any other provider
        # the model stays None and the seam resolves its large tier
        # (CC_LLM_MODEL_LARGE), failing fast with a clear message when unset.
        # The settings read is guarded so a stripped-down test Settings
        # without the field can't break Agent construction.
        if model:
            self.model = model
        else:
            try:
                from ..config import get_settings
                house = getattr(get_settings(), "agent_model", "") or ""
            except Exception:  # noqa: BLE001
                house = ""
            if house:
                self.model = house
            elif getattr(llm, "provider", "anthropic") == "anthropic":
                self.model = DEFAULT_MODEL
            else:
                self.model = None
        # Output budget: explicit arg > CC_AGENT_MAX_TOKENS (settings) >
        # module default. 4096 was too small for file writes — a large
        # write truncated mid-tool-call and the model spiralled
        # (2026-06-13); 8192 is the Sonnet default ceiling and the
        # setting lets a deployment go higher.
        if max_tokens is not None:
            self.max_tokens = max_tokens
        else:
            try:
                from ..config import get_settings
                self.max_tokens = int(
                    getattr(get_settings(), "agent_max_tokens", DEFAULT_MAX_TOKENS)
                )
            except Exception:  # noqa: BLE001
                self.max_tokens = DEFAULT_MAX_TOKENS
        # Ceiling resolution: explicit arg > CC_AGENT_MAX_ITERATIONS
        # (settings) > module default. The settings read is guarded so
        # a stripped-down test Settings without the field can't break
        # Agent construction.
        if max_iterations is not None:
            self.max_iterations = max_iterations
        else:
            try:
                from ..config import get_settings
                self.max_iterations = int(
                    getattr(
                        get_settings(), "agent_max_iterations",
                        DEFAULT_MAX_ITERATIONS,
                    )
                )
            except Exception:  # noqa: BLE001
                self.max_iterations = DEFAULT_MAX_ITERATIONS
        # C4 — tool-output cap (CC-D8). Settings-only knob; 0 disables. Bounds
        # each tool_result's content in the model-facing trajectory; the full
        # output is still recorded in tool_calls_log.
        try:
            from ..config import get_settings
            self.tool_output_max_chars = int(
                getattr(get_settings(), "agent_tool_output_max_chars", 0)
            )
        except Exception:  # noqa: BLE001
            self.tool_output_max_chars = 0
        self.sequence_id = sequence_id
        # Optional tool interceptor (F0, coding-agent feature plan). An
        # async callable invoked before EVERY tool dispatch:
        #     await intercept(tool_name, tool_input) -> decision
        # The decision is a dict: {"action": "allow"} lets the call
        # through; {"action": "allow", "input": {...}} substitutes the
        # tool input; {"action": "deny", "reason": "..."} blocks the
        # call — the reason is returned to the model as an error
        # tool_result so the agent can adapt instead of crashing.
        # None (the default) = no interception; this is the only seam
        # the coding-agent segment needs in the library, and the web
        # app never sets it.
        self.intercept = intercept
        # F5 companion seam: an optional async observer invoked AFTER a
        # tool executes successfully:
        #     await after_tool(tool_name, tool_input) -> Optional[str]
        # A returned string is appended to the tool's output as a note,
        # so the model sees post-edit effects (e.g. a formatter hook
        # rewriting the file) immediately — otherwise its next edit
        # would target stale content. None (default) = off.
        self.after_tool = after_tool
        # Block 2 slice 1 (2026-07-21): optional event emitter — the
        # third default-off library seam (Q1=A ratified). An async
        # callable
        #     await emit(event_type, payload_dict)
        # fired at loop milestones: run start, iteration start, tool
        # calls issued, each tool result, notices (compaction / H2
        # truncation / deadline / max_iterations / model_error). The
        # endpoint passes AgentEventMux.emit (agent/events.py — the
        # vocabulary's one home); None (the default) = today's
        # behavior, byte-identical. The loop does NOT emit the
        # terminal run_completed/error — the endpoint pipeline does,
        # after finalize, so the terminal result carries mcr.
        self.emit = emit
        # Block 2 slice 2 (Q6=B, ratified 2026-07-21): token
        # streaming only where a viewer consumes it. When True AND
        # emit is wired AND the seam has stream_messages, model
        # calls stream and text deltas emit as EVT_TEXT_DELTA
        # {iteration, text}; every other configuration — including
        # every non-streaming turn — uses complete_messages, the
        # proven call path, unchanged. Default False.
        self.stream_tokens = stream_tokens

        # Ensure all tools are imported (triggers @register_tool side
        # effects). Idempotent — safe to call repeatedly.
        import_all_tools()

        self.registry = registry or get_registry()
        self.tools = self.registry.list_for_context("agent")
        self.tool_state = tool_state

        # Inject state into the module-level holder that tools read.
        # Last writer wins; OK because Agent instances run sequentially
        # within a single process by default (each request gets its own
        # Agent but they all share one event loop in FastAPI's worker).
        set_tool_state(tool_state)

    # -----------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, Any]],
        *,
        system: Optional[str] = None,
        extra_system_context: Optional[str] = None,
        deadline_seconds: Optional[float] = None,
        usage_sink: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Run the agent loop.

        Args:
            messages: Anthropic-shaped message list. Each entry has
                a `role` ('user' or 'assistant') and `content`
                (string or list of content blocks). The harness
                does NOT include a system message here — the system
                prompt is built separately via `build_system_prompt`
                and passed to the LLM as the `system` field of the
                Messages API call.
            system: Optional override for the system prompt. When
                omitted, the agent builds the standard prompt from
                its registered tools.
            extra_system_context: Optional retrieved-context block
                (C2 warm-start) appended AFTER the base/overridden
                system prompt, so the tool guidance survives. None =
                no warm-start.
            deadline_seconds: Optional SOFT deadline for the whole
                run, measured from run start (Q3A, ratified
                2026-07-15). When the elapsed wall clock reaches it at
                the top of an iteration, the loop stops issuing tool
                calls and makes ONE final model call instructing the
                model to compose its complete output from what it has
                gathered; tool_use blocks in that response are NOT
                dispatched. Returns with stop_reason='deadline'.
                None = no deadline (prior behavior, byte-identical).
            usage_sink: Optional mutable dict the loop updates with
                aggregate usage after every model call
                ({prompt_tokens, completion_tokens, cache_read_tokens,
                cache_creation_tokens, model, iterations}). Lets a
                caller that CANCELS the run (an outer belt timeout)
                still meter the spend — a cancelled coroutine's return
                value is unrecoverable, so without the sink that spend
                would go unbilled.

        Returns:
            A dict with keys:
                - `id`: response id (chatcmpl-agent-<uuid>)
                - `model`: model that produced the final response
                - `messages`: full message list (input + assistant
                  turns + tool results), so callers can persist the
                  trajectory
                - `final_text`: the assistant's final text output
                - `stop_reason`: 'end_turn' | 'max_iterations' |
                  'deadline' | 'error'
                - `iterations`: how many tool-call iterations ran
                - `prompt_tokens`: total input tokens across all
                  iterations
                - `completion_tokens`: total output tokens
                - `tool_calls`: list of {tool_name, input, output,
                  iteration} for telemetry
        """
        # Build system prompt — registry-derived per P0.24.
        base_system = system or build_system_prompt(
            self.customer, self.tools,
        )

        # C2 warm-start (P1 folded in): append the opening-turn pre-flight's
        # retrieved context after the base prompt. `system` overrides the
        # built prompt, but warm-start still layers on top of whichever won.
        if extra_system_context:
            base_system = f"{base_system}\n\n{extra_system_context}"

        # C3 — agent-loop compaction (CC-D5/6/7). Off unless CC_AGENT_COMPACTION.
        # When on, the running summary of dropped early turns folds into the
        # system prompt under _COMPACTION_HEADER (Anthropic rejects system-role
        # messages inside the array), so the trajectory the model sees stays
        # bounded. `compaction_summary` accumulates across events; each model
        # call uses `_effective_system()`, which layers it onto base_system.
        _compact = None
        try:
            from ..config import get_settings
            if getattr(get_settings(), "agent_compaction", False):
                from ..retrieval.compaction import (
                    compact_agent_trajectory as _compact,
                )
        except Exception:  # noqa: BLE001 — compaction is best-effort; a config
            # or import hiccup must never break the agent loop.
            _compact = None
        compaction_summary: Optional[str] = None

        def _effective_system() -> str:
            if compaction_summary:
                return (
                    f"{base_system}\n\n{_COMPACTION_HEADER}\n"
                    f"{compaction_summary}"
                )
            return base_system

        # Build the tool definitions in Anthropic format. The adapter
        # lives in agent/adapters/anthropic.py and renders each Tool's
        # parameters_schema into Anthropic's tool spec. Import here to
        # avoid a circular import at module load (adapters import the
        # registry).
        from .adapters.anthropic import render_tools_for_anthropic
        anthropic_tools = render_tools_for_anthropic(self.tools)

        # Track aggregate metadata.
        # 2026-07-09: total round-trip wall clock for the WHOLE turn
        # (all iterations, tool executions, upstream calls). Stamped
        # into query_logs.latency_ms by turn_finalize — the honest
        # speed number for pitches, not per-iteration slices.
        run_t0 = time.monotonic()
        iteration = 0
        stop_reason = "max_iterations"
        prompt_tokens_total = 0
        completion_tokens_total = 0
        cache_creation_total = 0
        cache_read_total = 0
        tool_calls_log: list[dict[str, Any]] = []

        # The working messages list — the model-facing view. We append
        # assistant + tool_result turns as the loop progresses; C3 may compact
        # it (replacing old turns with a system-folded summary) so it stays
        # bounded. `full_trajectory` mirrors every append but is NEVER
        # compacted, so the returned `messages` carries the complete history
        # for persistence / telemetry even when the model saw a compacted view.
        working: list[dict[str, Any]] = list(messages)
        full_trajectory: list[dict[str, Any]] = list(messages)

        final_text = ""
        current_text = ""  # last iteration's text blocks — read by the
        # for-else partial below; initialized so a zero-iteration config
        # can't NameError.

        def _update_sink() -> None:
            # Q3A companion (2026-07-15): progressive usage for callers
            # that may cancel the run — reads the enclosing totals at
            # call time, so the sink always mirrors what has actually
            # been spent so far.
            if usage_sink is not None:
                usage_sink.update({
                    "prompt_tokens": prompt_tokens_total,
                    "completion_tokens": completion_tokens_total,
                    "cache_read_tokens": cache_read_total,
                    "cache_creation_tokens": cache_creation_total,
                    "model": self.model,
                    "iterations": iteration,
                })

        await self._emit(
            EVT_RUN_STARTED,
            model=self.model,
            max_iterations=self.max_iterations,
            message_count=len(messages),
        )

        for iteration in range(1, self.max_iterations + 1):
            # Graceful deadline (Q3A, ratified 2026-07-15): at expiry,
            # stop issuing tool calls and force ONE final compose from
            # what's already gathered. Partial verified work lands —
            # text, tool trace, metering — instead of being discarded
            # by an outer cancellation (the render-salvage pattern,
            # one layer up). Tools stay in the request so the prompt-
            # cache prefix survives; any tool_use in the response is
            # simply not dispatched.
            if (
                deadline_seconds is not None
                and (time.monotonic() - run_t0) >= deadline_seconds
            ):
                await self._emit(
                    EVT_NOTICE, kind="deadline", iteration=iteration,
                )
                deadline_msg = {
                    "role": "user",
                    "content": (
                        "DEADLINE REACHED — stop researching now. "
                        "Compose your complete final output from what "
                        "you have already gathered. State explicitly "
                        "which items were verified and which were not "
                        "— never guess to fill a gap. Do not call any "
                        "more tools; respond with text only."
                    ),
                }
                working.append(deadline_msg)
                full_trajectory.append(deadline_msg)
                try:
                    response = await self._call_model(
                        system=_effective_system(),
                        messages=working,
                        tools=anthropic_tools,
                        iteration=iteration,
                    )
                except Exception as e:
                    logger.error(
                        "agent.deadline_compose_failed",
                        customer_id=self.customer.id,
                        iteration=iteration,
                        error=str(e),
                    )
                    stop_reason = "error"
                    final_text = (
                        f"I hit an error reaching the model: {e}. "
                        f"Please try again."
                    )
                    break
                if hasattr(response, "usage"):
                    prompt_tokens_total += getattr(
                        response.usage, "input_tokens", 0,
                    )
                    completion_tokens_total += getattr(
                        response.usage, "output_tokens", 0,
                    )
                    cache_creation_total += getattr(
                        response.usage, "cache_creation_input_tokens", 0,
                    ) or 0
                    cache_read_total += getattr(
                        response.usage, "cache_read_input_tokens", 0,
                    ) or 0
                _update_sink()
                assistant_content = self._content_to_dict_list(
                    response.content,
                )
                assistant_msg = {
                    "role": "assistant", "content": assistant_content,
                }
                working.append(assistant_msg)
                full_trajectory.append(assistant_msg)
                final_text = "\n".join(
                    b.get("text", "")
                    for b in assistant_content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
                stop_reason = "deadline"
                logger.warning(
                    "agent.deadline_compose",
                    customer_id=self.customer.id,
                    iterations=iteration,
                    elapsed_ms=int((time.monotonic() - run_t0) * 1000),
                    final_chars=len(final_text),
                )
                break
            # C3: compact the model-facing trajectory before the call. The
            # should_compact threshold self-gates the cadence — after a
            # compaction the context drops well below threshold and climbs
            # back over several tool rounds — so this is cheap when it no-ops.
            if _compact is not None:
                _res = _compact(
                    working, self.customer.id,
                    prior_summary=compaction_summary,
                )
                if _res is not None:
                    compaction_summary, working = _res
                    logger.info(
                        "agent.compacted",
                        customer_id=self.customer.id,
                        iteration=iteration,
                        working_messages=len(working),
                    )
                    await self._emit(
                        EVT_NOTICE, kind="compacted",
                        iteration=iteration,
                        working_messages=len(working),
                    )
            await self._emit(EVT_ITERATION_STARTED, iteration=iteration)
            t0 = time.monotonic()
            try:
                response = await self._call_model(
                    system=_effective_system(),
                    messages=working,
                    tools=anthropic_tools,
                    iteration=iteration,
                )
            except Exception as e:
                logger.error(
                    "agent.model_call_failed",
                    customer_id=self.customer.id,
                    iteration=iteration,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                await self._emit(
                    EVT_NOTICE, kind="model_error",
                    iteration=iteration, error=str(e),
                )
                stop_reason = "error"
                final_text = (
                    f"I hit an error reaching the model: {e}. "
                    f"Please try again."
                )
                break
            iteration_ms = int((time.monotonic() - t0) * 1000)

            # Anthropic usage object: input_tokens, output_tokens, plus the
            # cache_* fields when prompt caching is active (C1). input_tokens
            # is the NON-cached delta; cache_read is billed ~0.1x and
            # cache_creation ~1.25x — the cost ledger prices each separately.
            if hasattr(response, "usage"):
                prompt_tokens_total += getattr(
                    response.usage, "input_tokens", 0,
                )
                completion_tokens_total += getattr(
                    response.usage, "output_tokens", 0,
                )
                cache_creation_total += getattr(
                    response.usage, "cache_creation_input_tokens", 0,
                ) or 0
                cache_read_total += getattr(
                    response.usage, "cache_read_input_tokens", 0,
                ) or 0
            _update_sink()

            # Append the assistant turn to the working history
            # (Anthropic SDK serializes content blocks back into the
            # message list shape).
            assistant_content = self._content_to_dict_list(response.content)
            assistant_msg = {"role": "assistant", "content": assistant_content}
            working.append(assistant_msg)
            full_trajectory.append(assistant_msg)

            # Inspect for tool_use blocks. If none, the agent has
            # produced its final text — break out of the loop.
            tool_use_blocks = [
                b for b in assistant_content
                if isinstance(b, dict) and b.get("type") == "tool_use"
            ]

            text_blocks = [
                b for b in assistant_content
                if isinstance(b, dict) and b.get("type") == "text"
            ]
            current_text = "\n".join(
                b.get("text", "") for b in text_blocks
            )

            if not tool_use_blocks:
                # Final response — no more tool calls.
                stop_reason = getattr(response, "stop_reason", "end_turn") or "end_turn"
                final_text = current_text
                logger.info(
                    "agent.final_response",
                    customer_id=self.customer.id,
                    iterations=iteration,
                    final_chars=len(final_text),
                    iteration_ms=iteration_ms,
                )
                break

            # H2 (2026-06-13): a response cut off at the output token
            # limit can leave a tool_use block with incomplete/empty
            # input. Dispatching it writes an empty file and the model
            # spirals (the MMORPG write_file failure). Detect the
            # truncation, refuse to dispatch, and feed back an actionable
            # redirect; every tool_use still gets a matching tool_result
            # (protocol), then the loop continues so the model recovers.
            if getattr(response, "stop_reason", None) == "max_tokens":
                logger.warning(
                    "agent.tool_call_truncated",
                    customer_id=self.customer.id,
                    iteration=iteration,
                    tools_called=[b.get("name", "?") for b in tool_use_blocks],
                )
                await self._emit(
                    EVT_NOTICE, kind="tool_call_truncated",
                    iteration=iteration,
                    tools=[b.get("name", "?") for b in tool_use_blocks],
                )
                trunc_msg = {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": b.get("id", ""),
                            "content": (
                                "This tool call was cut off at the output "
                                "token limit before it finished, so it was "
                                "NOT executed — nothing was written or "
                                "changed. Do not assume it succeeded. If you "
                                "were writing a large file, write it in "
                                "sections: create the file with the first "
                                "portion, then append the rest with smaller "
                                "edit_file calls."
                            ),
                            "is_error": True,
                        }
                        for b in tool_use_blocks
                    ],
                }
                working.append(trunc_msg)
                full_trajectory.append(trunc_msg)
                continue

            # We have tool calls. Execute each one and append the
            # tool_result blocks as a single user turn (per
            # Anthropic's tool-use protocol).
            logger.info(
                "agent.tool_iteration",
                customer_id=self.customer.id,
                iteration=iteration,
                tools_called=[
                    b.get("name", "?") for b in tool_use_blocks
                ],
                iteration_ms=iteration_ms,
            )
            await self._emit(
                EVT_TOOL_CALLS,
                iteration=iteration,
                calls=[
                    {
                        "tool_use_id": b.get("id", ""),
                        "name": b.get("name", "?"),
                        "input_summary": summarize_tool_input(
                            b.get("input", {}) or {},
                        ),
                    }
                    for b in tool_use_blocks
                ],
            )
            tool_result_blocks: list[dict[str, Any]] = []
            for block in tool_use_blocks:
                tool_name = block.get("name", "")
                tool_input = block.get("input", {}) or {}
                tool_use_id = block.get("id", "")

                tool_t0 = time.monotonic()
                output, is_error = await self._dispatch_tool(
                    tool_name=tool_name,
                    tool_input=tool_input,
                )
                tool_ms = int((time.monotonic() - tool_t0) * 1000)
                # Persist for telemetry / response payload.
                tool_calls_log.append({
                    "iteration": iteration,
                    "tool_name": tool_name,
                    "tool_use_id": tool_use_id,
                    "input": tool_input,
                    "output": output,
                    "is_error": is_error,
                    "duration_ms": tool_ms,
                })
                await self._emit(
                    EVT_TOOL_RESULT,
                    iteration=iteration,
                    tool_use_id=tool_use_id,
                    name=tool_name,
                    duration_ms=tool_ms,
                    is_error=is_error,
                    output_head=bound_output_head(output),
                )

                # Anthropic tool_result content must be a string or a
                # list of content blocks. We serialize the dict output
                # as JSON string — agents read it back natively.
                content_str = (
                    output
                    if isinstance(output, str)
                    else json.dumps(output, default=str)
                )
                content_str = _cap_tool_output(
                    content_str, self.tool_output_max_chars,
                )
                tool_result_blocks.append({
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": content_str,
                    "is_error": is_error,
                })

            tr_msg = {
                "role": "user",
                "content": tool_result_blocks,
            }
            working.append(tr_msg)
            full_trajectory.append(tr_msg)

        else:
            # for-else: max_iterations exhausted without a final response.
            logger.warning(
                "agent.max_iterations_reached",
                customer_id=self.customer.id,
                iterations=self.max_iterations,
            )
            await self._emit(
                EVT_NOTICE, kind="max_iterations",
                iterations=self.max_iterations,
            )
            # Surface the last iteration's text — and when there is
            # none (the final iteration ended on pure tool_use blocks,
            # the live failure 2026-06-12: "Last partial response:"
            # printed blank), summarize the last actions instead so the
            # user sees WHERE it stopped.
            partial = current_text.strip()
            if not partial:
                last_tools = [
                    c["tool_name"] for c in tool_calls_log
                    if c.get("iteration") == iteration
                ]
                if last_tools:
                    partial = (
                        "(the last step produced no text — it ended "
                        "mid-work on: " + ", ".join(last_tools) + ")"
                    )
                else:
                    partial = "(the last step produced no text)"
            final_text = (
                f"I hit the {self.max_iterations}-iteration limit before "
                f"finishing. Work so far is preserved — say 'continue' "
                f"to resume from here.\n\nLast partial response:\n\n"
                + partial
            )

        return {
            "id": f"chatcmpl-agent-{uuid.uuid4().hex[:24]}",
            "model": self.model,
            "messages": full_trajectory,
            "final_text": final_text,
            "stop_reason": stop_reason,
            "iterations": iteration,
            "prompt_tokens": prompt_tokens_total,
            "completion_tokens": completion_tokens_total,
            "cache_creation_tokens": cache_creation_total,
            "cache_read_tokens": cache_read_total,
            "tool_calls": tool_calls_log,
            "duration_ms": int((time.monotonic() - run_t0) * 1000),
        }

    # -----------------------------------------------------------------
    # Internal: event emission (Block 2 slice 1)
    # -----------------------------------------------------------------

    async def _emit(self, event_type: str, **payload: Any) -> None:
        """Fire one loop event through the optional emitter seam.

        Belt on top of the mux's own per-subscriber guards, so a bare
        callable passed in tests stays safe too: an emitter failure
        logs and never touches the run."""
        if self.emit is None:
            return
        try:
            await self.emit(event_type, payload)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "agent.emit_failed", event_type=event_type, error=str(e),
            )

    # -----------------------------------------------------------------
    # Internal: model call
    # -----------------------------------------------------------------

    async def _call_model(
        self,
        *,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        iteration: int = 0,
    ) -> Any:
        """Call the agent's controlling LLM through the provider seam.

        PROVIDER BOUNDARY (docs/LOCAL_MODELS_PLAN.md): the loop's internal
        representation is the Anthropic Messages shape. Under the anthropic
        provider the seam passes everything through verbatim and returns the
        raw SDK response; the Anthropic-only prompt-cache decoration (C1) is
        applied here, on this path only. Under an OpenAI-compatible provider
        the seam translates the wire format both ways via
        agent/adapters/openai.py and returns a shim — either way the caller
        introspects `.content`, `.stop_reason`, and `.usage` identically.

        The call is synchronous either way (Anthropic SDK / httpx); we wrap
        in asyncio.to_thread so it doesn't block the event loop.

        Block 2 slice 2 (Q6=B): when token streaming is active
        (stream_tokens=True AND emit wired AND the seam has
        stream_messages), the call goes through the seam's streaming twin
        instead; text deltas hop from the worker thread onto the event
        loop via run_coroutine_threadsafe as EVT_TEXT_DELTA
        {iteration, text}. Loop FIFO keeps deltas ordered ahead of the
        events emitted after this call returns, and the final message is
        shape-identical, so everything downstream reads it unchanged. Any
        other configuration — including every non-streaming turn — uses
        complete_messages exactly as before.
        """
        import asyncio

        if getattr(self.llm, "provider", "anthropic") == "anthropic":
            sys_arg: Any = _system_blocks(system)
            msg_arg = _messages_with_cache_breakpoint(messages)
        else:
            sys_arg = system
            msg_arg = messages

        use_stream = (
            self.stream_tokens
            and self.emit is not None
            and hasattr(self.llm, "stream_messages")
        )

        if not use_stream:
            def _call() -> Any:
                return self.llm.complete_messages(
                    system=sys_arg,
                    messages=msg_arg,
                    tools=tools if tools else None,
                    max_tokens=self.max_tokens,
                    model=self.model,
                )

            return await asyncio.to_thread(_call)

        loop = asyncio.get_running_loop()

        def _on_text(chunk: str) -> None:
            # Worker thread -> event loop; fire-and-forget. call order =
            # scheduling order = delivery order, and _emit never raises.
            asyncio.run_coroutine_threadsafe(
                self._emit(
                    EVT_TEXT_DELTA, iteration=iteration, text=chunk,
                ),
                loop,
            )

        def _call_streaming() -> Any:
            return self.llm.stream_messages(
                system=sys_arg,
                messages=msg_arg,
                tools=tools if tools else None,
                max_tokens=self.max_tokens,
                model=self.model,
                on_text=_on_text,
            )

        return await asyncio.to_thread(_call_streaming)

    # -----------------------------------------------------------------
    # Internal: tool dispatch
    # -----------------------------------------------------------------

    async def _dispatch_tool(
        self,
        *,
        tool_name: str,
        tool_input: dict[str, Any],
    ) -> tuple[Any, bool]:
        """Dispatch a tool call by name.

        Returns:
            (output, is_error) — the output is whatever the tool
            returned (typically a dict per P0.23), or an error
            message string when is_error=True.
        """
        tool = self.registry.get(tool_name)
        if tool is None or "agent" not in tool.contexts:
            err = (
                f"Tool {tool_name!r} is not registered for the agent "
                f"context. Available tools: "
                f"{[t.name for t in self.tools]}."
            )
            logger.warning(
                "agent.tool_not_found",
                customer_id=self.customer.id,
                tool=tool_name,
            )
            return (err, True)

        # Customer id is injected automatically per P0.23 — the agent
        # cannot override it via tool input.
        sanitized_input = {
            k: v for k, v in tool_input.items() if k != "customer_id"
        }

        # F0: tool interceptor — gates, hooks, and plan mode stand here.
        if self.intercept is not None:
            try:
                decision = await self.intercept(tool_name, sanitized_input)
            except Exception as e:  # noqa: BLE001 — a broken policy must
                # fail CLOSED for safety, not silently allow.
                logger.error(
                    "agent.intercept_error",
                    customer_id=self.customer.id,
                    tool=tool_name,
                    error=str(e),
                )
                return (
                    f"Tool {tool_name!r} blocked: the safety interceptor "
                    f"raised {type(e).__name__}: {e}",
                    True,
                )
            if isinstance(decision, dict):
                if decision.get("action") == "deny":
                    reason = decision.get("reason") or "blocked by policy"
                    logger.info(
                        "agent.tool_denied",
                        customer_id=self.customer.id,
                        tool=tool_name,
                        reason=reason,
                    )
                    return (f"Tool {tool_name!r} not executed: {reason}", True)
                if "input" in decision and isinstance(decision["input"], dict):
                    sanitized_input = {
                        k: v for k, v in decision["input"].items()
                        if k != "customer_id"
                    }

        try:
            output = await tool.impl(
                customer_id=self.customer.id,
                **sanitized_input,
            )
            if self.after_tool is not None:
                try:
                    note = await self.after_tool(tool_name, sanitized_input)
                except Exception as e:  # noqa: BLE001 — an observer bug
                    # must never fail a successful tool call.
                    logger.error(
                        "agent.after_tool_error",
                        customer_id=self.customer.id,
                        tool=tool_name,
                        error=str(e),
                    )
                    note = None
                if note:
                    if isinstance(output, dict):
                        output = {**output, "post_hook_note": note}
                    else:
                        output = f"{output}\n\n[post-edit hook] {note}"
            return (output, False)
        except TypeError as e:
            # Wrong argument shape from the LLM — surface as
            # tool_result with is_error=True so the agent can correct.
            err = (
                f"Tool {tool_name!r} rejected the input: {e}. "
                f"Inspect the tool's parameter schema and retry."
            )
            logger.warning(
                "agent.tool_input_error",
                customer_id=self.customer.id,
                tool=tool_name,
                error=str(e),
            )
            return (err, True)
        except Exception as e:
            err = (
                f"Tool {tool_name!r} failed: "
                f"{type(e).__name__}: {e}"
            )
            logger.error(
                "agent.tool_runtime_error",
                customer_id=self.customer.id,
                tool=tool_name,
                error=str(e),
                error_type=type(e).__name__,
            )
            return (err, True)

    # -----------------------------------------------------------------
    # Internal: content normalization
    # -----------------------------------------------------------------

    @staticmethod
    def _content_to_dict_list(
        content: Any,
    ) -> list[dict[str, Any]]:
        """Normalize Anthropic content blocks to plain dicts.

        The Anthropic SDK returns content as a list of typed objects
        (TextBlock, ToolUseBlock). We convert to plain dicts so the
        working messages list is JSON-serializable end-to-end and
        callers can persist the trajectory.
        """
        if isinstance(content, str):
            return [{"type": "text", "text": content}]
        if not isinstance(content, list):
            return [{"type": "text", "text": str(content)}]

        out: list[dict[str, Any]] = []
        for block in content:
            # Plain dict — pass through.
            if isinstance(block, dict):
                out.append(block)
                continue
            # Anthropic SDK block — read its fields.
            block_type = getattr(block, "type", None)
            if block_type == "text":
                out.append({
                    "type": "text",
                    "text": getattr(block, "text", ""),
                })
            elif block_type == "tool_use":
                out.append({
                    "type": "tool_use",
                    "id": getattr(block, "id", ""),
                    "name": getattr(block, "name", ""),
                    "input": getattr(block, "input", {}) or {},
                })
            elif block_type == "thinking":
                # Extended-thinking block (2026-07-07 live fix). Anthropic
                # REQUIRES thinking blocks preceding tool_use to be replayed
                # VERBATIM incl. signature — the old unknown-branch stub
                # ({"type":"thinking","raw":...}) 400'd every multi-iteration
                # turn on thinking-capable models:
                #   messages.N.content.0.thinking.thinking: Field required
                out.append({
                    "type": "thinking",
                    "thinking": getattr(block, "thinking", "") or "",
                    "signature": getattr(block, "signature", "") or "",
                })
            elif block_type == "redacted_thinking":
                out.append({
                    "type": "redacted_thinking",
                    "data": getattr(block, "data", "") or "",
                })
            else:
                # Truly unknown block: DROP from the replay (a fabricated
                # stub guarantees an upstream 400; omission at worst loses
                # context) — but say so in the log.
                logger.warning(
                    "agent.unknown_content_block_dropped",
                    block_type=block_type,
                )
        return out
