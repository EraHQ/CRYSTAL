"""Agent endpoint — POST /v1/agent/messages.

The flagship endpoint under the agent reframe (D-A1 in
docs/AGENT_ARCHITECTURE.md). Phase 7.5 fills in the real
handler; v2 customers using the agent get the full tool-loop
experience here while existing v1 customers continue to be served
by /v1/chat/completions (the proxy adapter).

REQUEST/RESPONSE SHAPE (P0.27):
- Request body is Anthropic Messages API-shaped:
  {model?, max_tokens?, messages: [...], system?, ...}
- Response is the Agent.run() result dict, surfaced as JSON.
  Includes the final assistant text, the full message trajectory
  (so callers can persist), token usage, and tool-call telemetry.
- Phase 9A adds optional MCR identifiers to the response when
  trace + critique emission succeeds: `mcr.trace_id`,
  `mcr.critique_id`, `mcr.action_item_ids`. Absent or None-valued
  when emission failed (response is unaffected per P0.44).

STATELESS (P0.17):
- Full message history is sent each request. The agent does NOT
  hold conversation state between calls. Mem0 holds session
  memory separately, addressed by `metadata.sequence_id` (same
  resolution rules as chat_proxy uses).

CONTEXT INJECTION:
- The Agent class injects shared state into the tool registry on
  construction. The state dict is built here from
  request.app.state — same singletons the chat_proxy reads.
- The Anthropic client is constructed here per-request from
  settings.anthropic_api_key. Phase 11 can hoist a process-wide
  client into app.state for connection reuse.

STREAMING (P0.16):
- Phase 7.5 ships the non-streaming path. Streaming via SSE is
  opt-in; the endpoint currently returns the full response after
  the agent loop completes. Streaming wrapper added before
  Phase 8 if time, otherwise lands in Phase 7.5+1.

POST-TURN SIGNALS (Phase 9A + C0 + P3, now via the shared layer):
- After `agent.run(...)` returns, this endpoint calls
  `finalize_agent_turn(...)` (crystal_cache/agent/turn_finalize.py)
  to emit the universal post-turn signal set: the cost-ledger row
  (C0), citations + grounding + marketplace credit + the
  uncited-answer coverage gap (P3), and the MCR reasoning trace +
  Haiku self-critique + action items (Phase 9A). The same function
  is what the coding-agent surfaces call, so the signals can't
  drift between lenses (docs/SHARED_TURN_FINALIZE_DESIGN.md). Every
  step is fail-safe: a signal failure logs a warning and the
  agent's response to the caller is NOT blocked. The trace is built
  deterministically from the agent's tool_calls_log per P0.46.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Annotated, Any, Optional

import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ..agent import Agent
from ..agent.agent import DEFAULT_MODEL
from ..agent.turn_finalize import (  # noqa: F401 — re-exported for back-compat
    _AGENT_UNCITED_GAP_MIN_CHARS,
    _extract_last_user_query,
    finalize_agent_turn,
    ground_agent_citations,
    record_agent_llm_cost,
)
from ..config import settings
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import require_customer
from ..ingress.errors import InvalidRequestError
from ..llm import get_llm_client
from ..models import Customer

logger = structlog.get_logger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Request / response schemas (P0.27)
# ---------------------------------------------------------------------------

class AgentMessage(BaseModel):
    """One message in the agent conversation.

    Anthropic Messages API-shaped: role + content where content is
    a string OR a list of content blocks (text / tool_use /
    tool_result).
    """
    model_config = ConfigDict(extra="allow")

    role: str  # 'user' | 'assistant'
    content: Any  # str or list of content block dicts


class AgentRequest(BaseModel):
    """POST /v1/agent/messages body.

    Anthropic Messages API-shaped. The agent's controlling model
    is configurable via the `model` field; when omitted, the
    process-wide default (`CC_AGENT_MODEL` env / settings.agent_model)
    is used.
    """
    model_config = ConfigDict(extra="allow")

    messages: list[AgentMessage] = Field(min_length=1)
    model: Optional[str] = None
    max_tokens: Optional[int] = None
    system: Optional[str] = None  # Override the auto-built system prompt
    # Anthropic Messages API metadata; we adopt one well-known key:
    #   metadata.sequence_id — same role as the proxy uses
    metadata: Optional[dict[str, Any]] = None


# Per-turn signal helpers — _extract_last_user_query, record_agent_llm_cost,
# ground_agent_citations, _AGENT_UNCITED_GAP_MIN_CHARS — moved to
# crystal_cache/agent/turn_finalize.py so the coding agent can call them without
# importing FastAPI. They're re-exported at the top of this module for
# back-compat (tests + callers). See docs/SHARED_TURN_FINALIZE_DESIGN.md.


# ---------------------------------------------------------------------------
# C6 — per-conversation model selection
# ---------------------------------------------------------------------------

async def resolve_conversation_model(
    *,
    store: MetadataStore,
    customer_id: str,
    sequence_id: Optional[str],
    requested_model: Optional[str],
) -> Optional[str]:
    """Resolve the controlling model for an agent turn — per-conversation
    sticky model selection (C6).

    Precedence: the client's explicit `requested_model` wins AND is persisted
    as this conversation's sticky model (last-writer-wins), so a later turn
    from any device reuses it; a request with no model falls back to the
    conversation's saved model. Returns None when neither applies — the caller
    (the Agent) then fills the CC_AGENT_MODEL house default and finally the
    built-in DEFAULT_MODEL.

    Keyed on `sequence_id` (the conversation scope). A None sequence_id skips
    both the save and the lookup (an anonymous turn can't have a sticky model)
    and returns `requested_model` unchanged. Fail-safe (P0.44): a store error
    must never break the request — it is logged and the model falls through to
    the house default.
    """
    effective_model = requested_model
    if not sequence_id:
        return effective_model
    if effective_model:
        try:
            await store.set_conversation_model(
                customer_id,
                conversation_key=sequence_id,
                model=effective_model,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "agent.model_persist_failed",
                customer_id=customer_id, error=str(e),
            )
    else:
        try:
            effective_model = await store.get_conversation_model(
                customer_id, conversation_key=sequence_id,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "agent.model_lookup_failed",
                customer_id=customer_id, error=str(e),
            )
            effective_model = None
    return effective_model


# Agent citations (P3, CC-D11 = grounding-based implicit credit) moved to
# crystal_cache/agent/turn_finalize.py alongside the other per-turn signal
# helpers; re-exported above. See docs/SHARED_TURN_FINALIZE_DESIGN.md.


# ---------------------------------------------------------------------------
# C2 — retrieval pre-flight (cache-hit short-circuit + warm-start; folds P1)
# ---------------------------------------------------------------------------

@dataclass
class _PreflightResult:
    """Outcome of the opening-turn retrieval pre-flight (C2).

    `cache_hit_text` set → the caller short-circuits the loop with the cached
    answer. Otherwise `warm_start_context` (set or None) → the caller injects
    it into the system prompt. A None return from the helper itself means the
    pre-flight did not run at all.
    """
    cache_hit_text: Optional[str] = None
    cache_hit_crystal_id: Optional[str] = None
    warm_start_context: Optional[str] = None


async def agent_retrieval_preflight(
    *,
    messages: list[dict[str, Any]],
    customer: Customer,
    store: MetadataStore,
    vector_index: Any,
    encoder: Any,
) -> Optional[_PreflightResult]:
    """Opening-turn retrieval pre-flight (C2 — cost + parity; folds P1).

    Flag-gated on `settings.agent_retrieval_preflight` (default off) and run
    ONLY on a fresh / no-context turn — no assistant message in the history
    yet. CC-D4: once a conversation has context the model drives retrieval via
    tools, so the pre-flight is skipped.

    On an opening turn it calls the proxy's `retrieve_and_inject` (CC-D3 =
    reuse — the agent's own `knowledge_search` can't surface the cache-hit
    signal) and maps the outcome:

      - `cache_hit_response` set → `cache_hit_text` (PERFECT routing + top-1
        `source_kind == "model_reasoning"` + non-empty `answer_value`): the
        caller returns it without entering the loop — the CRYS analog of the
        proxy's `upstream_call_made=False` path.
      - else, when retrieval matched → `warm_start_context`: the raw retrieved
        text wrapped as an advisory block for the system prompt, so CRYS
        usually skips its first `knowledge_search`.

    Returns None when the pre-flight did not run (flag off / not an opening
    turn / it raised). Fail-safe: a pre-flight failure must never break the
    request — the caller proceeds with the normal loop.
    """
    if not settings.agent_retrieval_preflight:
        return None
    # CC-D4 gate: opening turn only. Any assistant turn means context exists.
    if any(m.get("role") == "assistant" for m in messages):
        return None
    try:
        from ..retrieval.pipeline import retrieve_and_inject
        outcome = await retrieve_and_inject(
            customer,
            messages,
            store,
            vector_index,
            encoder,
        )
    except Exception as e:
        logger.warning(
            "agent.preflight_failed", customer_id=customer.id, error=str(e),
        )
        return None

    if outcome.cache_hit_response:
        return _PreflightResult(
            cache_hit_text=outcome.cache_hit_response,
            cache_hit_crystal_id=outcome.cache_hit_crystal_id,
        )

    if outcome.injected_text and outcome.match_type != "none":
        block = (
            "## Retrieved context\n"
            "The following may be relevant to the request. Use it if it "
            "helps; call your retrieval tools for anything more.\n\n"
            f"{outcome.injected_text}"
        )
        return _PreflightResult(warm_start_context=block)

    return _PreflightResult()


def _build_cache_hit_result(
    *,
    messages: list[dict[str, Any]],
    model: str,
    cache_hit_text: str,
) -> dict[str, Any]:
    """Agent-result dict for a cache-hit short-circuit (C2).

    Same shape as `Agent.run()` so the endpoint + Inspector render it
    uniformly, but with no model call: `iterations=0`, all token counts 0, no
    tool calls, `stop_reason="cache_hit"`. The synthetic assistant turn carries
    the cached answer so the persisted trajectory stays complete.
    """
    return {
        "id": f"chatcmpl-agent-{uuid.uuid4().hex[:24]}",
        "model": model,
        "messages": list(messages) + [
            {"role": "assistant", "content": cache_hit_text}
        ],
        "final_text": cache_hit_text,
        "stop_reason": "cache_hit",
        "iterations": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cache_creation_tokens": 0,
        "cache_read_tokens": 0,
        "tool_calls": [],
    }


# ---------------------------------------------------------------------------
# D — agent-API session registration (HTTP surface in the Agents view)
# ---------------------------------------------------------------------------

async def _register_agent_api_session(
    store: MetadataStore,
    *,
    session_id: str,
    team_id: str,
    model: Optional[str],
    label: str,
) -> None:
    """Register an ephemeral per-request session for the HTTP agent surface so
    the turn shows in the Agents view (D — the unified-surfaces law: all CRYS
    activity visible, the agent endpoint included), and append its turn_started
    event.

    The agent endpoint is stateless (P0.17), so a request's natural session
    lifetime IS the request. `SessionHandle` lives in the coding-agent package,
    which the library cannot import, so this calls the store's session methods
    directly. Best-effort: a registry hiccup must never break the API response
    (the session-registry posture)."""
    try:
        await store.register_session(
            session_id, team_id,
            project_dir=None, model=model, status="running",
            current_action=(label[:160] or None),
        )
        await store.record_event(
            session_id, event_type="turn_started", team_id=team_id,
            phase="turn", turn_index=0, label=label[:120],
            payload={"surface": "agent_api"},
        )
    except Exception as e:  # noqa: BLE001 — observability is best-effort
        logger.debug("agent.api_session_register_failed", error=str(e))


async def _complete_agent_api_session(
    store: MetadataStore,
    *,
    session_id: str,
    team_id: str,
    result: dict[str, Any],
    cost_micro_usd: Optional[int],
    duration_ms: int,
) -> None:
    """Append the turn_completed event (tokens + cost + summary) and mark the
    ephemeral session exited, so the Agents view shows a finished turn rather
    than a session left to go stale → 'crashed'. Best-effort (see
    `_register_agent_api_session`)."""
    try:
        summary = (result.get("final_text") or "").strip()
        await store.record_event(
            session_id, event_type="turn_completed", team_id=team_id,
            phase="turn", turn_index=0, status="ok", label=summary[:120],
            payload={
                "summary": summary[:2000],
                "iterations": result.get("iterations"),
                "stop_reason": result.get("stop_reason"),
            },
            tokens_input=result.get("prompt_tokens"),
            tokens_output=result.get("completion_tokens"),
            cost_micro_usd=cost_micro_usd,
            duration_ms=duration_ms,
        )
        await store.heartbeat_session(session_id, status="exited")
    except Exception as e:  # noqa: BLE001
        logger.debug("agent.api_session_complete_failed", error=str(e))


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

async def run_agent_messages(
    *,
    body: AgentRequest,
    request: Request,
    customer: Customer,
    store: MetadataStore,
) -> JSONResponse:
    """Shared agent-run pipeline — everything after customer resolution.

    Used by the public Bearer-auth route (`agent_messages`) and the keyless
    admin inspector wrapper (`admin_customer_agent`); the caller resolves and
    authenticates the customer, this runs the agent loop.

    The pipeline:
      1. Receives an already-resolved customer (the caller handles auth).
      2. Builds the Anthropic client (uses settings.anthropic_api_key).
      3. Constructs an Agent with the customer + shared state from
         app.state + sequence_id resolved from request metadata/header.
      4. Calls agent.run(messages).
      5. Calls finalize_agent_turn(...) to emit the universal post-turn
         signal set (cost row + citations/credit/gap + MCR trace +
         self-critique + action items). Adds the resulting MCR ids to
         the response payload under `mcr` when emission succeeded.
      6. Returns the result dict as JSON.

    The agent's tool registry resolves at Agent construction time;
    `import_all_tools()` is idempotent so this is cheap to call per
    request.

    Errors:
      - Missing anthropic_api_key returns 400 (cannot run the agent
        without a controlling LLM).
      - Agent loop errors are surfaced inside the result dict's
        `stop_reason` field, not as HTTP errors. The agent layer
        is designed to degrade gracefully.
      - Post-turn signal errors do NOT propagate to the caller per
        P0.44; they log and the response continues with absent
        or None-valued `mcr.trace_id` / `mcr.critique_id` fields.
    """
    # E4 doors (2026-07-06, shared with the chat proxy — ratified: the
    # agent has EVERYTHING the proxy has, same commit). A managed tenant
    # at its monthly cap is refused before any model call.
    from ..control.admission import enforce_managed_budget, enforce_managed_model

    await enforce_managed_budget(store, customer)

    # Controlling LLM — routed through the provider seam (Slice 5 of the
    # provider-swap arc; docs/LOCAL_MODELS_PLAN.md). Fail fast when no
    # provider is configured.
    llm = get_llm_client()
    if not llm.is_ready():
        raise InvalidRequestError(
            "Agent mode requires a configured LLM provider. Set "
            "CC_LLM_API_KEY or ANTHROPIC_API_KEY for Anthropic, or "
            "CC_LLM_PROVIDER=openai with CC_LLM_BASE_URL and CC_LLM_API_KEY "
            "for an OpenAI-compatible endpoint.",
            param=None,
            code="agent_no_llm_provider",
        )

    # Resolve sequence_id from metadata or header. Same precedence as
    # chat_proxy: body.metadata.sequence_id → X-Sequence-Id header.
    # (The agent doesn't currently use server-side inference from
    # message hash; the agent endpoint is opt-in and callers using it
    # are expected to manage their own conversation ids.)
    sequence_id: Optional[str] = None
    if body.metadata is not None:
        candidate = body.metadata.get("sequence_id")
        if isinstance(candidate, str) and candidate.strip():
            sequence_id = candidate.strip()[:64]
    if not sequence_id:
        header_value = request.headers.get("x-sequence-id")
        if header_value and header_value.strip():
            sequence_id = header_value.strip()[:64]

    # Convert pydantic messages to plain dicts for the agent loop.
    messages_dicts = [m.model_dump(exclude_none=True) for m in body.messages]

    # Per-conversation model selection (C6). The client's explicit model wins
    # and is persisted as this conversation's sticky model (last-writer-wins);
    # a no-model request reuses the saved one. None flows to the Agent, which
    # fills the CC_AGENT_MODEL house default → built-in DEFAULT_MODEL. The
    # helper is fail-safe (a store hiccup never breaks the request).
    effective_model = await resolve_conversation_model(
        store=store,
        customer_id=customer.id,
        sequence_id=sequence_id,
        requested_model=body.model,
    )
    # E4 (2026-07-06): the customer's CONFIGURED model joins the chain —
    # request → conversation-sticky → customer's model_id → house
    # default. Without this, the model picked at onboarding/Settings
    # never governed agent turns (found live: hosted parity gap #4).
    if not effective_model:
        effective_model = (
            customer.model_routing_config.model_id or None
        )
    # Managed policy: whatever won must be a model the platform serves.
    enforce_managed_model(customer, effective_model)

    # C2 — retrieval pre-flight (cost + parity; folds P1 warm-start). Opening
    # turn only + flag-gated + fail-safe (see the helper). A cache hit returns
    # the cached answer without entering the loop (zero model calls); a miss
    # yields warm-start context for the system prompt.
    warm_start_context: Optional[str] = None
    preflight = await agent_retrieval_preflight(
        messages=messages_dicts,
        customer=customer,
        store=store,
        vector_index=request.app.state.vector_index,
        encoder=request.app.state.prompt_encoder,
    )
    if preflight is not None and preflight.cache_hit_text:
        logger.info(
            "agent.cache_hit",
            customer_id=customer.id,
            sequence_id=sequence_id,
            crystal_id=preflight.cache_hit_crystal_id,
        )
        hit = _build_cache_hit_result(
            messages=messages_dicts,
            # The response's model label when zero model calls ran: under
            # anthropic the built-in default is the honest answer; under any
            # other provider a Claude string would be a lie, so fall to the
            # configured house default or empty.
            model=effective_model or settings.agent_model or (
                DEFAULT_MODEL if llm.provider == "anthropic" else ""
            ),
            cache_hit_text=preflight.cache_hit_text,
        )
        hit["mcr"] = None  # no reasoning ran; MCR emission is skipped on a hit
        return JSONResponse(content=hit)
    if preflight is not None:
        warm_start_context = preflight.warm_start_context

    # Build the shared tool state from app.state. These are the same
    # singletons chat_proxy and the SDK endpoints read; the agent
    # joins the consumers.
    tool_state: dict[str, Any] = {
        "store": store,
        "vector_store": request.app.state.vector_store,
        "vector_index": getattr(request.app.state, "vector_index", None),
        "fact_vector_store": request.app.state.fact_vector_store,
        "encoder": request.app.state.prompt_encoder,
        "decomposer": getattr(request.app.state, "decomposer", None),
    }

    agent = Agent(
        customer=customer,
        llm=llm,
        tool_state=tool_state,
        model=effective_model,
        max_tokens=body.max_tokens or 4096,
        sequence_id=sequence_id,
    )

    logger.info(
        "agent.request",
        customer_id=customer.id,
        sequence_id=sequence_id,
        model=agent.model,
        message_count=len(messages_dicts),
    )

    # D — register an ephemeral session for this stateless HTTP turn so it shows
    # in the Agents view (the unified-surfaces law: all CRYS activity visible,
    # the agent endpoint included). A request's natural session lifetime IS the
    # request (P0.17): register → turn_started → run → turn_completed → exited,
    # all best-effort. The last user message labels the turn.
    user_query = _extract_last_user_query(messages_dicts)
    api_session_id = f"crysapi_{uuid.uuid4().hex[:16]}"
    api_turn_t0 = time.monotonic()
    await _register_agent_api_session(
        store, session_id=api_session_id, team_id=customer.id,
        model=agent.model, label=user_query,
    )

    result = await agent.run(
        messages=messages_dicts,
        system=body.system,
        extra_system_context=warm_start_context,
    )

    # Post-turn universal signal set — the shared layer both CRYS surfaces call
    # (docs/SHARED_TURN_FINALIZE_DESIGN.md). One function emits the cost row
    # (C0), grounds + credits citations + the uncited-answer gap (P3), and emits
    # the MCR trace + self-critique (Phase 9A), in that order. Every step is
    # individually fail-safe + flag-gated, so a signal failure never affects the
    # agent's response. origin="agent" attributes the cost row to the HTTP
    # surface; turn_index is None — the agent endpoint is stateless (P0.17) and
    # does not manage turn indexing.
    finalized = await finalize_agent_turn(
        store=store,
        encoder=request.app.state.prompt_encoder,
        customer=customer,
        result=result,
        user_query=user_query,
        sequence_id=sequence_id,
        origin="agent",
        turn_index=None,
        query_log_id=None,
    )

    # Surface MCR ids on the response. None values are valid (and documented)
    # when emission partially failed; absent dict keys would be surprising for
    # callers expecting the shape.
    result["mcr"] = finalized["mcr"]

    # D — close the ephemeral session: record turn_completed (reusing the cost
    # the ledger just computed) and mark it exited, so it reads as a finished
    # turn rather than going stale → 'crashed'.
    await _complete_agent_api_session(
        store, session_id=api_session_id, team_id=customer.id,
        result=result, cost_micro_usd=finalized["cost_micro_usd"],
        duration_ms=int((time.monotonic() - api_turn_t0) * 1000),
    )

    return JSONResponse(content=result)


@router.post("/v1/agent/messages")
async def agent_messages(
    body: AgentRequest,
    request: Request,
    customer: Annotated[Customer, Depends(require_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> JSONResponse:
    """Public agent endpoint (Bearer Key A). Authenticates the customer, then
    delegates to the shared `run_agent_messages` pipeline. Existing v1
    customers are still served by /v1/chat/completions (the proxy adapter);
    v2 agent callers get the full tool loop here.
    """
    return await run_agent_messages(
        body=body, request=request, customer=customer, store=store,
    )
