"""Chat completions proxy — POST /v1/chat/completions.

The OpenAI-compatible proxy adapter. Per the agent reframe
(D-A2 in docs/AGENT_ARCHITECTURE.md), this endpoint stays for
customers using the proxy deployment mode. v1 customers see no
behavioral change.

This is the largest endpoint in the v2 port (~830 lines).
The verbatim port from v1's `chat_completions` happened in Phase 7
Wave 7F (2026-05-25) after all of Wave 7B (retrieval pipeline),
Wave 7C (execution package), Wave 7D (learning package), and
Wave 7E (LearningService + ConsolidationService + new mixin) had
landed. The full pipeline:

  1. Auth via Bearer Key A (require_customer dependency)
  2. Sequence resolution + turn index lookup
  3. Crystal type resolution
  4. V2 retrieve_and_inject (V3 retrieval try-block DROPPED per
     P0.10 — see module docstring on V3 removal)
  5. Cache-hit short-circuit (both streaming + non-streaming)
  6. Extra params (tools, response_format, thinking, crystal tools)
  7. Shadow decision
  8. Streaming branch (delegates to _stream_chat_completion)
  9. Upstream call (with parallel shadow)
  10. Shadow delta
  11. Token accounting + cost estimation
  12. QueryLog write
  13. Crystal tool call processing (push signals + inline research)
  14. Optional 2nd upstream call when crystal_pull_research fires
  15. Tool call stripping from final response (with confirmations)
  16. Mem0 turn add (fire-and-forget)
  17. **Phase 9C: MCR trace + self-critique emission** (non-streaming
      paths only — streaming MCR deferred per P0.57)
  18. Return JSONResponse

Phase 9C (2026-05-27) MCR INTEGRATION
--------------------------------------
Per P0.55–P0.61, this module now emits MCR artifacts after the
upstream response is finalized. Three new code paths:

  - Cache-hit path: calls `emit_mcr_artifacts(...,
    skip_self_critique=True)` so a trace persists carrying the
    matched crystal id, but no Haiku self-critique runs (the cache
    hit didn't come from LLM reasoning — there's nothing to
    critique). Trace's `crystals_used = [cache_hit_crystal_id]`.

  - Upstream-served path: calls `emit_mcr_artifacts(...)` with full
    self-critique enabled (matches Phase 9A's agent endpoint
    pattern). The proxy-shaped `agent_result` is built by
    `_build_proxy_agent_result(...)` from the upstream response +
    crystal tool calls extracted from the LLM's response.
    `crystals_used` is pre-populated from `outcome.matched_crystal_ids`
    rather than extracted from retrieval-tool outputs — see P0.56
    rationale.

  - `handle_signals` call site: now passes `sequence_id`,
    `turn_index`, `agent_model=model`, `mcr_enabled=True` so
    Phase 9B's BD-3 + BD-11 writes fire in production. Existing
    fail-safe try/except wraps the call.

Streaming requests do NOT emit MCR per P0.57 — the streaming
generators continue writing QueryLog only. Phase 11+ may revisit
once streaming-shaped traces are designed (CU-21).

V3 RETRIEVAL REMOVAL (P0.10, Wave 7F decision)
----------------------------------------------
v1's chat_completions opened with a V3 retrieval try-block that
imported `retrieve_v3` from `retrieval.v3_pipeline`. That module
was DROPPED in Wave 7B per the agent reframe (P0.5, D-A1/D-A3).
v2 calls the V2 path directly.

The five module-level helpers below are direct ports of v1's
top-level helpers. They live in this module rather than a shared
helpers file because only `chat_completions` uses them. Phase 9C
adds a sixth helper (`_build_proxy_agent_result`) following the
same pattern.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Annotated, Any, AsyncIterator, Optional

import httpx
import structlog
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from ..agent import emit_mcr_artifacts
from ..llm import get_llm_client
from ..config import settings
from ..cost.emit import record_model_call
from ..execution.shadow_evaluator import ShadowEvaluator
from ..execution.upstream_client import StreamChunk, get_upstream_client
from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..ingress.auth import task_principal_customer, task_principal_operator
from ..ingress.errors import InvalidRequestError, UpstreamError
from ..ingress.schema import ChatCompletionRequest
from ..models import Customer, Operator, QueryLog
from ..retrieval import (
    CRYSTAL_TOOL_NAMES,
    RetrievalOutcome,
    add_conversation_turn,
    extract_crystal_tool_calls,
    ground_citations,
    handle_signals,
    inject_crystal_tools,
    map_citations,
    parse_citations,
    parse_tool_calls,
    render_sources_footer,
    retrieve_and_inject,
    rewrite_markers,
    run_inline_research,
)

logger = structlog.get_logger(__name__)

router = APIRouter()

# Growth G1c: minimum answer length (chars) before an uncited answer emits a
# knowledge-gap candidate — skips trivial "Done."/"Yes." turns from generating
# noise in the gaps queue.
_UNCITED_GAP_MIN_CHARS = 40


# ---------------------------------------------------------------------------
# Module-level helpers (verbatim from v1 app.py, lines 3427+)
# ---------------------------------------------------------------------------

def _resolve_sequence_id(
    *,
    request: Request,
    body: ChatCompletionRequest,
    customer_id: str,
    messages: list[dict[str, Any]],
) -> Optional[str]:
    """Resolve the sequence_id for this request.

    Resolution order (Stage 2a):
      1. body.metadata['sequence_id'] — OpenAI-style metadata field.
      2. X-Sequence-Id request header — alternate transport.
      3. Server-inferred from message-history hash.

    Server inference uses sha256(customer_id || first_user_message)
    truncated to 16 hex chars. Same conversation → same id.

    Returns None if the request has no user message at all.
    """
    if body.metadata is not None:
        candidate = body.metadata.get("sequence_id")
        if candidate and isinstance(candidate, str) and candidate.strip():
            return candidate.strip()[:64]

    header_value = request.headers.get("x-sequence-id")
    if header_value and header_value.strip():
        return header_value.strip()[:64]

    first_user_text: Optional[str] = None
    for m in messages:
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str) and content.strip():
                first_user_text = content
                break
            if isinstance(content, list):
                parts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = "".join(parts).strip()
                if joined:
                    first_user_text = joined
                    break

    if first_user_text is None:
        return None

    digest = hashlib.sha256(
        f"{customer_id}\x00{first_user_text}".encode("utf-8")
    ).hexdigest()
    return f"seq_{digest[:16]}"


def _extract_query_text(messages: list[dict[str, Any]]) -> str:
    """Best-effort extraction of the 'query' for logging.

    Uses the last user message's text content.
    """
    for m in reversed(messages):
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    part.get("text", "")
                    for part in content
                    if isinstance(part, dict) and part.get("type") == "text"
                )
    return ""


def _build_cache_hit_response(
    *,
    answer_text: str,
    model: str,
    crystal_id: Optional[str],
) -> dict[str, Any]:
    """Build an OpenAI-shaped chat completion response from a cached answer.

    Mirrors upstream_client clients so downstream consumers see the
    same structure whether the answer came from the cache or from
    upstream. `usage={prompt_tokens: 0, completion_tokens: 0,
    total_tokens: 0}` because no model call was made. Cache hits are
    distinguished from zero-token responses via the QueryLog
    (`upstream_call_made=False`, `injection_method="cache_hit"`).

    Phase 1.5.2: cache-hit responses NEVER carry `message.tool_calls`
    even when the request defined `tools`.
    """
    if crystal_id:
        response_id = f"chatcmpl-cache-{crystal_id}"
    else:
        response_id = f"chatcmpl-cache-{uuid.uuid4().hex[:24]}"
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": answer_text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


# ---------------------------------------------------------------------------
# Phase 9C helper: proxy-shaped agent_result construction (P0.56)
# ---------------------------------------------------------------------------

def _build_proxy_agent_result(
    *,
    upstream_assistant_text: str,
    upstream_openai_format: dict[str, Any],
    crystal_calls: list[dict[str, Any]],
    matched_crystal_ids: list[str],
) -> dict[str, Any]:
    """Translate the proxy's response state into a Phase 9A-compatible
    agent_result dict.

    Phase 9A's `emit_mcr_artifacts` expects an agent_result shape:
        {final_text, tool_calls, stop_reason, ...}
    where `tool_calls` is a list of per-call dicts the agent loop
    produced. The proxy doesn't have an agent loop — it has an
    upstream response + the crystal tool calls extracted from that
    response. We translate.

    Per P0.56:
      - `final_text` = upstream.assistant_text. What the customer
        actually saw (after any post-processing).
      - `tool_calls` = a list of dicts shaped like the agent's
        `tool_calls_log` entries, built from `crystal_calls` (the
        push_gap / push_correct / etc. tool calls the LLM made).
        Each crystal call becomes one entry with tool_name from the
        crystal call's function.name, input from the parsed
        arguments, output set to a placeholder string ("Processed."
        for non-research; the actual research result for research)
        — the trace's `tool_calls` column carries the audit trail,
        not the live results.
      - `stop_reason` = upstream's finish_reason from the OpenAI
        format.

    `crystals_used` is NOT extracted from the tool_calls log here
    (unlike Phase 9A's agent path). The proxy's retrieved crystals
    come from the V2 retrieval pipeline BEFORE the LLM is called —
    not from a tool the LLM invoked. The caller passes
    `matched_crystal_ids` and `emit_mcr_artifacts` will derive
    `crystals_used` from the tool_calls log; since none of the
    crystal tools are in `_RETRIEVAL_TOOL_NAMES`, the extracted
    list will be empty. **The caller must override
    `crystals_used` directly** by constructing the trace kwargs
    and calling `store.create_reasoning_trace(...)` directly — OR
    we synthesize a retrieval-tool entry in the tool_calls list.

    Phase 9C chooses the latter approach: synthesize ONE entry with
    tool_name="content_search" carrying `output.matched_fact_ids =
    matched_crystal_ids`. This satisfies Phase 9A's deterministic
    extraction without bypassing the helper. The synthetic entry's
    `iteration=0` (the retrieval ran BEFORE iteration 1's LLM call).
    Real proxy "iterations" start at 1 with the LLM response.

    Returns a dict suitable for splatting into `emit_mcr_artifacts(
    agent_result=...)`.
    """
    # Synthetic retrieval-tool entry — feeds Phase 9A's
    # `_extract_crystals_used` so `crystals_used` is populated
    # without bypassing the helper. content_search is in
    # `_RETRIEVAL_TOOL_NAMES` per Phase 9A; we use it as the
    # proxy-side label even though the V2 retrieval pipeline is a
    # different code path.
    tool_calls: list[dict[str, Any]] = []
    if matched_crystal_ids:
        tool_calls.append({
            "iteration": 0,
            "tool_name": "content_search",
            "tool_use_id": "v2_retrieve_synthetic",
            "input": {"source": "v2_retrieve_and_inject"},
            "output": {
                "matched_fact_ids": list(matched_crystal_ids),
                "injection_text": "(synthetic — see chat_proxy)",
            },
            "is_error": False,
        })

    # Translate each crystal tool call into an agent-style tool_call
    # entry. The output is a placeholder; the actual side effects
    # already happened in handle_signals.
    for idx, tc in enumerate(crystal_calls, start=1):
        func = tc.get("function", {})
        name = func.get("name", "")
        args_str = func.get("arguments", "{}")
        try:
            args = (
                json.loads(args_str)
                if isinstance(args_str, str)
                else args_str
            )
        except json.JSONDecodeError:
            args = {}
        tool_calls.append({
            "iteration": idx,
            "tool_name": name,
            "tool_use_id": tc.get("id", f"tc_{idx}"),
            "input": args,
            "output": "Processed.",
            "is_error": False,
        })

    # Extract stop_reason from the upstream OpenAI-shaped response.
    stop_reason = "stop"
    choices = upstream_openai_format.get("choices") or []
    if choices and isinstance(choices, list):
        finish = choices[0].get("finish_reason") if isinstance(choices[0], dict) else None
        if isinstance(finish, str) and finish:
            stop_reason = finish

    return {
        "final_text": upstream_assistant_text,
        "tool_calls": tool_calls,
        "stop_reason": stop_reason,
    }


# ---------------------------------------------------------------------------
# Streaming generators (Phase 1.5.1, May 2026)
# ---------------------------------------------------------------------------
#
# Two async generators feed sse_starlette.EventSourceResponse:
#
#   _stream_cache_hit       — emits the cached answer as one chunk
#                             plus [DONE]; writes QueryLog at close.
#   _stream_chat_completion — wraps client.stream(...), forwards
#                             chunks, accumulates response_text and
#                             token counts, writes QueryLog at close.
#
# Phase 9C (P0.57): streaming paths do NOT emit MCR artifacts.
# Streaming-shaped traces are deferred to a future phase (CU-21).

async def _stream_cache_hit(
    *,
    answer_text: str,
    model: str,
    crystal_id: Optional[str],
    log: QueryLog,
    store: MetadataStore,
    customer_id: str,
) -> AsyncIterator[str]:
    """Stream a cache-hit response as one OpenAI-shaped chunk + [DONE]."""
    if crystal_id:
        chunk_id = f"chatcmpl-cache-{crystal_id}"
    else:
        chunk_id = f"chatcmpl-cache-{uuid.uuid4().hex[:24]}"

    chunk_dict = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {"role": "assistant", "content": answer_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }

    try:
        yield json.dumps(chunk_dict)
        yield "[DONE]"
    finally:
        try:
            await store.write_query_log(log)
        except Exception as e:
            logger.error(
                "query_log.write_failed",
                customer_id=customer_id,
                error=str(e),
            )


async def _stream_chat_completion(
    *,
    client: Any,
    messages: list[dict[str, Any]],
    model: str,
    temperature: Optional[float],
    max_tokens: Optional[int],
    extra: dict[str, Any],
    customer_id: str,
    query_text: str,
    outcome: RetrievalOutcome,
    sequence_id: Optional[str],
    turn_index: Optional[int],
    retrieval_latency_ms: int,
    store: MetadataStore,
) -> AsyncIterator[str]:
    """Stream an upstream-served chat completion via SSE.

    Phase 9C (P0.57): does NOT emit MCR artifacts. Continues writing
    QueryLog only (existing behavior).
    """
    start = time.monotonic()
    response_text_parts: list[str] = []
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    stream_error: Optional[str] = None

    try:
        async for chunk in client.stream(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra,
        ):
            assert isinstance(chunk, StreamChunk)

            if chunk.delta_text:
                response_text_parts.append(chunk.delta_text)
            if chunk.prompt_tokens is not None:
                prompt_tokens = chunk.prompt_tokens
            if chunk.completion_tokens is not None:
                completion_tokens = chunk.completion_tokens

            yield json.dumps(chunk.raw_chunk)

        yield "[DONE]"
    except httpx.HTTPStatusError as e:
        stream_error = f"upstream_error: status {e.response.status_code}"
        logger.warning(
            "upstream.stream_http_error",
            customer_id=customer_id,
            status_code=e.response.status_code,
        )
        error_chunk = {
            "id": "chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
        }
        yield json.dumps(error_chunk)
        yield "[DONE]"
    except httpx.HTTPError as e:
        stream_error = f"upstream_transport_error: {e}"
        logger.error(
            "upstream.stream_transport_error",
            customer_id=customer_id,
            error=str(e),
        )
        error_chunk = {
            "id": "chatcmpl-error",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "error",
                }
            ],
        }
        yield json.dumps(error_chunk)
        yield "[DONE]"
    finally:
        upstream_latency_ms = int((time.monotonic() - start) * 1000)
        total_latency_ms = retrieval_latency_ms + upstream_latency_ms
        response_text = "".join(response_text_parts)

        log = QueryLog(
            id=f"qlog_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            query_text=query_text,
            query_vector=[],
            match_type=outcome.match_type,
            injection_method=outcome.injection_method,  # type: ignore[arg-type]
            confidence_gate_fires=0,
            matched_facts=outcome.matched_crystal_ids,
            response_text=response_text,
            response_confidence_at_commit=None,
            upstream_call_made=True,
            shadow_ran=False,
            shadow_delta=None,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            shadow_prompt_tokens=None,
            shadow_completion_tokens=None,
            prompt_token_overhead=None,
            concept_top_config=outcome.concept_top_config,
            concept_top_score=(
                outcome.concept_top_score
                if outcome.concept_path_ran
                else None
            ),
            concept_payload=outcome.concept_payload,
            sequence_id=sequence_id,
            turn_index=turn_index,
            routed_crystal_id=(
                outcome.matched_crystal_ids[0]
                if outcome.matched_crystal_ids
                else None
            ),
            top1_score=outcome.routing_top1,
            top2_score=outcome.routing_top2,
            latency_ms=total_latency_ms,
            timestamp=datetime.now(timezone.utc),
        )

        logger.info(
            "stream.completed",
            customer_id=customer_id,
            chars=len(response_text),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            upstream_latency_ms=upstream_latency_ms,
            error=stream_error,
        )

        try:
            await store.write_query_log(log)
        except Exception as e:
            logger.error(
                "query_log.write_failed",
                customer_id=customer_id,
                error=str(e),
            )


# ---------------------------------------------------------------------------
# Chat completions — THE critical path
# ---------------------------------------------------------------------------

async def run_chat_completion(
    *,
    body: ChatCompletionRequest,
    request: Request,
    customer: Customer,
    store: MetadataStore,
    operator: Optional[Operator] = None,
) -> Any:
    """OpenAI-compatible chat completions.

    Pipeline:
      1. Auth via Bearer Key A
      2. Sequence resolution + turn index lookup
      3. Crystal type resolution
      4. V2 retrieve_and_inject
      5. Cache-hit short-circuit (streaming + non-streaming)
      6. Extra params (tools, response_format, thinking, crystal tools)
      7. Shadow decision
      8. Streaming branch (delegates to _stream_chat_completion)
      9. Upstream call (with parallel shadow)
      10. Shadow delta
      11. Token accounting + cost estimation
      12. QueryLog write
      13. Crystal tool call processing (push signals + inline research)
      14. Optional 2nd upstream call when crystal_pull_research fires
      15. Tool call stripping from final response
      16. Mem0 turn add (fire-and-forget)
      17. **Phase 9C: MCR trace + self-critique emission**
          (non-streaming paths only — streaming MCR deferred per P0.57)
      18. Return JSONResponse
    """
    client = get_upstream_client(customer)
    model = body.model or customer.model_routing_config.model_id

    original_messages = [m.model_dump(exclude_none=True) for m in body.messages]
    query_text = _extract_query_text(original_messages)

    sequence_id = _resolve_sequence_id(
        request=request,
        body=body,
        customer_id=customer.id,
        messages=original_messages,
    )
    turn_index: Optional[int] = None
    if sequence_id is not None:
        try:
            turn_index = await store.next_turn_index(
                customer_id=customer.id,
                sequence_id=sequence_id,
            )
        except Exception as e:
            logger.warning(
                "sequence.turn_index_lookup_failed",
                customer_id=customer.id,
                sequence_id=sequence_id,
                error=str(e),
            )
            turn_index = None

    # ---- Conversation compaction (memory blend, Inc 1; D-MB1) -----------
    # Long conversations: compress older turns into a summary (Mem0
    # long-term when enabled, rule-based fallback otherwise) and keep only
    # the last few turns verbatim. Bounds upstream token cost and preserves
    # long-term context. Ported from v1's retrieve_v3 STEP 0, which v2
    # dropped with the v3 pipeline. See docs/MEMORY_BLEND_PLAN.md.
    #
    # Runs AFTER sequence-id resolution (so the conversation id stays
    # stable across turns) and BEFORE retrieval (so the compacted list
    # feeds both retrieval and the upstream call).
    request_messages = original_messages
    try:
        from ..retrieval.compaction import compact_conversation, should_compact
        if should_compact(original_messages):
            # compact_conversation is synchronous and its mem0 path does
            # embedding + network work — run it off the event loop so a
            # long compaction can't stall other in-flight requests
            # (Core Principle: no component starves another).
            request_messages = await asyncio.to_thread(
                compact_conversation,
                original_messages,
                customer_id=customer.id,
                sequence_id=sequence_id,
            )
    except Exception as e:
        logger.warning(
            "compaction.failed",
            customer_id=customer.id,
            error=str(e),
        )
        request_messages = original_messages

    # ---- Crystal type resolution (Phase 4.11) ---------------------------
    crystal_type: str = "customer:legacy"
    if body.metadata is not None:
        candidate_type = body.metadata.get("crystal_type")
        if candidate_type and isinstance(candidate_type, str) and candidate_type.strip():
            requested = candidate_type.strip()
            registered = await store.get_crystal_type(requested)
            if registered is None:
                raise InvalidRequestError(
                    f"crystal_type {requested!r} is not registered. "
                    f"Seed it via the admin API (PUT /admin/api/crystal_types/"
                    f"<id>) before requesting it.",
                    param="metadata.crystal_type",
                    code="unknown_crystal_type",
                )
            crystal_type = requested

    # ---- Retrieval + injection (skipped on follow-ups) ------------------
    retrieval_start = time.monotonic()

    # Follow-up gate (memory blend, Inc 2; D-MB2). If this turn references
    # conversation history the model already has, skip retrieval entirely:
    # a fresh vector search here is wasteful and often mismatched. The
    # crystal_pull_research tool is injected below, so the model can pull
    # context on demand if it actually needs it. Ported and generalized
    # from v1's retrieve_v3 follow-up detector. See docs/MEMORY_BLEND_PLAN.md.
    from ..retrieval.session_dispatch import (
        is_followup_no_retrieval_needed,
        session_subject_from_last_log,
    )

    # Session consumption (memory blend, Inc 3; D-MB3/D-MB4). Read the prior
    # turn's outcome from query_logs — the DB-backed replacement for v1's
    # module-global session dict — so a vague follow-up after an established
    # subject is recognized even when it isn't short. When Mem0 is enabled,
    # its session search can also supply the subject signal.
    _session_subject: Optional[str] = None
    if sequence_id is not None:
        try:
            _last_log = await store.get_last_query_log_for_sequence(
                customer_id=customer.id,
                sequence_id=sequence_id,
            )
            _session_subject = session_subject_from_last_log(_last_log)
        except Exception as e:
            logger.warning(
                "session_dispatch.last_log_failed",
                customer_id=customer.id,
                error=str(e),
            )
        if (
            _session_subject is None
            and getattr(request.app.state, "mem0", None) is not None
        ):
            try:
                from ..retrieval.mem0_session import search_session_context
                # Sync mem0 client (embedding + vector search) — keep it
                # off the event loop; this sits in the retrieval hot path.
                _hints = await asyncio.to_thread(
                    search_session_context,
                    query_text=query_text,
                    customer_id=customer.id,
                    sequence_id=sequence_id,
                )
                _session_subject = _hints.get("subject") if _hints else None
            except Exception as e:
                logger.warning(
                    "session_dispatch.mem0_read_failed",
                    customer_id=customer.id,
                    error=str(e),
                )

    _skip_retrieval = is_followup_no_retrieval_needed(
        request_messages,
        query_text,
        session_subject=_session_subject,
    )

    if _skip_retrieval:
        logger.info(
            "retrieval.followup_passthrough",
            customer_id=customer.id,
            query_preview=query_text[:60],
            note=(
                "Follow-up referencing history; skipping retrieval. The LLM "
                "can request context via crystal_pull_research."
            ),
        )
        outcome = RetrievalOutcome(
            messages=request_messages,
            match_type="none",
            injection_method="none",
            matched_crystal_ids=[],
            top_score=0.0,
        )
    else:
        # Identity routing (memory blend, Inc 4; D-MB6). "Where is X / what
        # file defines X" is a navigation question, not a resemblance one —
        # answer it from a precise sparse-key scan instead of vector recall.
        # Fully additive: any miss, ambiguity, or error returns None and we
        # fall through to the recall path below (which already surfaces a
        # provenance header), so this can add precision but never regress.
        nav_outcome = None
        try:
            from ..retrieval.navigation_dispatch import try_identity_injection
            nav_outcome = await try_identity_injection(
                query_text=query_text,
                customer_id=customer.id,
                store=store,
                messages=request_messages,
            )
        except Exception as e:
            logger.warning(
                "retrieval.identity_routing_failed",
                customer_id=customer.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            nav_outcome = None

        if nav_outcome is not None:
            outcome = nav_outcome
            logger.info(
                "retrieval.identity_routed",
                customer_id=customer.id,
                matched_crystal_ids=outcome.matched_crystal_ids,
            )
        else:
            try:
                outcome = await retrieve_and_inject(
                    customer=customer,
                    operator=operator,
                    messages=request_messages,
                    store=store,
                    vector_index=request.app.state.vector_index,
                    encoder=request.app.state.prompt_encoder,
                    decomposer=getattr(request.app.state, "decomposer", None),
                    config_store=getattr(request.app.state, "dsl_config_store", None),
                    decoder_loader=getattr(request.app.state, "decoder_loader", None),
                    crystal_type=crystal_type,
                    cite=settings.enable_citations,
                )
                logger.info(
                    "retrieval.path",
                    path="V2",
                    customer_id=customer.id,
                    match_type=outcome.match_type,
                )
            except Exception as e:
                logger.error(
                    "retrieval.failed",
                    customer_id=customer.id,
                    error=str(e),
                    error_type=type(e).__name__,
                )
                outcome = RetrievalOutcome(
                    messages=request_messages,
                    match_type="none",
                    injection_method="none",
                    matched_crystal_ids=[],
                    top_score=0.0,
                )
    retrieval_latency_ms = int((time.monotonic() - retrieval_start) * 1000)
    messages = outcome.messages

    # ---- Cache-hit short-circuit ----------------------------------------
    if outcome.cache_hit_response is not None:
        cache_response = _build_cache_hit_response(
            answer_text=outcome.cache_hit_response,
            model=model,
            crystal_id=outcome.cache_hit_crystal_id,
        )
        total_latency_ms = retrieval_latency_ms

        logger.info(
            "cache_hit.served",
            customer_id=customer.id,
            crystal_id=outcome.cache_hit_crystal_id,
            top_score=outcome.top_score,
            margin=outcome.routing_margin,
            answer_chars=len(outcome.cache_hit_response),
            retrieval_latency_ms=retrieval_latency_ms,
        )

        log = QueryLog(
            id=f"qlog_{uuid.uuid4().hex[:16]}",
            customer_id=customer.id,
            query_text=query_text,
            query_vector=[],
            match_type=outcome.match_type,
            injection_method="cache_hit",  # type: ignore[arg-type]
            confidence_gate_fires=0,
            matched_facts=outcome.matched_crystal_ids,
            response_text=outcome.cache_hit_response,
            response_confidence_at_commit=None,
            upstream_call_made=False,
            shadow_ran=False,
            shadow_delta=None,
            prompt_tokens=0,
            completion_tokens=0,
            shadow_prompt_tokens=None,
            shadow_completion_tokens=None,
            prompt_token_overhead=None,
            concept_top_config=outcome.concept_top_config,
            concept_top_score=(
                outcome.concept_top_score
                if outcome.concept_path_ran
                else None
            ),
            concept_payload=outcome.concept_payload,
            sequence_id=sequence_id,
            turn_index=turn_index,
            routed_crystal_id=outcome.cache_hit_crystal_id,
            top1_score=outcome.routing_top1,
            top2_score=outcome.routing_top2,
            latency_ms=total_latency_ms,
            timestamp=datetime.now(timezone.utc),
        )

        # Phase 1.5.1: streaming cache-hit. Phase 9C does NOT emit
        # MCR for streaming requests per P0.57.
        if body.stream:
            return EventSourceResponse(
                _stream_cache_hit(
                    answer_text=outcome.cache_hit_response,
                    model=model,
                    crystal_id=outcome.cache_hit_crystal_id,
                    log=log,
                    store=store,
                    customer_id=customer.id,
                ),
            )

        # Non-streaming cache hit.
        try:
            await store.write_query_log(log)
        except Exception as e:
            logger.error(
                "query_log.write_failed",
                customer_id=customer.id,
                error=str(e),
            )

        # ---- Phase 9C (P0.58): cache-hit MCR emission --------------
        # Trace YES, self-critique NO. The cache hit didn't come from
        # LLM reasoning — there's no reasoning process to critique.
        # The trace still records what the system did (matched crystal
        # id, query, served answer) so the metacognitive layer can
        # compute retrieval-quality alignments.
        try:
            cache_agent_result = {
                "final_text": outcome.cache_hit_response,
                "tool_calls": (
                    [
                        {
                            "iteration": 0,
                            "tool_name": "content_search",
                            "tool_use_id": "v2_retrieve_synthetic",
                            "input": {"source": "v2_retrieve_and_inject"},
                            "output": {
                                "matched_fact_ids": (
                                    [outcome.cache_hit_crystal_id]
                                    if outcome.cache_hit_crystal_id
                                    else []
                                ),
                                "injection_text": "(cache hit)",
                            },
                            "is_error": False,
                        }
                    ]
                    if outcome.cache_hit_crystal_id
                    else []
                ),
                "stop_reason": "stop",
            }
            await emit_mcr_artifacts(
                store=store,
                customer_id=customer.id,
                user_query=query_text,
                agent_result=cache_agent_result,
                anthropic_client=None,  # not used; skip_self_critique=True
                sequence_id=sequence_id,
                turn_index=turn_index,
                query_log_id=log.id,
                skip_self_critique=True,  # P0.58
            )
        except Exception as e:
            # P0.44 discipline: MCR persistence failures never break
            # the user's request. emit_mcr_artifacts already catches
            # internally, but defense-in-depth here too.
            logger.warning(
                "mcr.cache_hit_emit_failed",
                customer_id=customer.id,
                error=str(e),
            )

        return JSONResponse(content=cache_response)

    # ---- Extra params -----------------------------------------------------
    extra: dict[str, Any] = {}
    if body.top_p is not None:
        extra["top_p"] = body.top_p
    if body.frequency_penalty is not None:
        extra["frequency_penalty"] = body.frequency_penalty
    if body.presence_penalty is not None:
        extra["presence_penalty"] = body.presence_penalty
    if body.stop is not None:
        extra["stop"] = body.stop
    if body.n is not None:
        extra["n"] = body.n
    if body.tools is not None:
        extra["tools"] = body.tools
    if body.tool_choice is not None:
        extra["tool_choice"] = body.tool_choice
    if body.parallel_tool_calls is not None:
        extra["parallel_tool_calls"] = body.parallel_tool_calls

    # V3 Phase 9: Inject crystal push/pull tools for non-streaming requests.
    # CC_DISABLE_CRYSTAL_TOOLS=1 skips injection entirely — benchmark /
    # single-pass mode (the LongMemEval harness requires it).
    _crystal_tools_injected = False
    _fact_store = getattr(request.app.state, "fact_vector_store", None)
    if (
        not body.stream
        and _fact_store is not None
        and not settings.disable_crystal_tools
    ):
        extra["tools"] = inject_crystal_tools(extra.get("tools"))
        _crystal_tools_injected = True
        logger.info(
            "push_pull.tools_injected",
            customer_id=customer.id,
            total_tools=len(extra["tools"]),
            customer_tools=len(body.tools) if body.tools else 0,
        )
    if body.response_format is not None:
        extra["response_format"] = body.response_format

    _thinking = getattr(body, "thinking", None)
    if _thinking is not None:
        extra["thinking"] = _thinking
        logger.info(
            "extended_thinking.enabled",
            budget_tokens=_thinking.get("budget_tokens") if isinstance(_thinking, dict) else None,
        )

    # ---- Shadow decision ------------------------------------------------
    shadow_eval: Optional[ShadowEvaluator] = getattr(
        request.app.state, "shadow_evaluator", None
    )
    top_crystal_tier: Optional[str] = None
    if outcome.matched_crystal_ids and outcome.match_type in ("medium", "high"):
        try:
            top_c = await store.get_crystal(outcome.matched_crystal_ids[0])
            if top_c is not None:
                top_crystal_tier = top_c.quality_tier
        except Exception:
            top_crystal_tier = None

    will_shadow = bool(
        shadow_eval is not None
        and shadow_eval.should_shadow(
            customer=customer,
            match_type=outcome.match_type,
            crystal_quality_tier=top_crystal_tier,
        )
    )

    # Phase 1.5.1: streaming branch. Phase 9C (P0.57) does NOT emit
    # MCR for streaming requests.
    if body.stream:
        return EventSourceResponse(
            _stream_chat_completion(
                client=client,
                messages=messages,
                model=model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                extra=extra,
                customer_id=customer.id,
                query_text=query_text,
                outcome=outcome,
                sequence_id=sequence_id,
                turn_index=turn_index,
                retrieval_latency_ms=retrieval_latency_ms,
                store=store,
            ),
        )

    start = time.monotonic()
    try:
        if will_shadow:
            primary_coro = client.complete(
                messages=messages,
                model=model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                **extra,
            )
            shadow_coro = shadow_eval.run_shadow(
                client=client,
                original_messages=original_messages,
                model=model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                **extra,
            )
            upstream, shadow_response = await asyncio.gather(
                primary_coro, shadow_coro
            )
        else:
            upstream = await client.complete(
                messages=messages,
                model=model,
                temperature=body.temperature,
                max_tokens=body.max_tokens,
                **extra,
            )
            shadow_response = None
    except httpx.HTTPStatusError as e:
        logger.warning(
            "upstream.http_error",
            customer_id=customer.id,
            provider=customer.model_routing_config.provider,
            status_code=e.response.status_code,
        )
        raise UpstreamError(
            f"Upstream provider returned status {e.response.status_code}",
            upstream_status=e.response.status_code,
        )
    except httpx.HTTPError as e:
        logger.error(
            "upstream.transport_error",
            customer_id=customer.id,
            error=str(e),
        )
        raise UpstreamError(
            f"Upstream transport error: {e}",
        )

    total_latency_ms = int((time.monotonic() - start) * 1000) + retrieval_latency_ms

    # ---- Shadow delta ---------------------------------------------------
    shadow_delta: Optional[float] = None
    if will_shadow and shadow_response is not None and shadow_eval is not None:
        shadow_delta = shadow_eval.compute_delta(
            injected_response=upstream.assistant_text,
            baseline_response=shadow_response.assistant_text,
        )

    # ---- Token accounting ----------------------------------------------
    prompt_tokens: Optional[int] = upstream.prompt_tokens
    completion_tokens: Optional[int] = upstream.completion_tokens
    shadow_prompt_tokens: Optional[int] = None
    shadow_completion_tokens: Optional[int] = None
    prompt_token_overhead: Optional[int] = None
    if will_shadow and shadow_response is not None:
        shadow_prompt_tokens = shadow_response.prompt_tokens
        shadow_completion_tokens = shadow_response.completion_tokens
        if prompt_tokens is not None and shadow_prompt_tokens is not None:
            prompt_token_overhead = prompt_tokens - shadow_prompt_tokens

    # Per-turn cost ESTIMATE for the log line — presentation only; the
    # authoritative ledger rows come from record_model_call via the shared
    # verified price table (cost/pricing.py). None when the model is
    # unknown or usage is missing — never a fabricated zero.
    from ..cost.pricing import estimate_cost_usd, price_table_from_settings

    est_cost = estimate_cost_usd(
        model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        price_table=price_table_from_settings(
            settings.llm_price_table_overrides
        ),
    )

    logger.info(
        "tokens.request",
        customer_id=customer.id,
        model=model,
        match_type=outcome.match_type,
        injection_method=outcome.injection_method,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        shadow_prompt_tokens=shadow_prompt_tokens,
        shadow_completion_tokens=shadow_completion_tokens,
        prompt_token_overhead=prompt_token_overhead,
        est_cost_usd=est_cost,
    )

    log = QueryLog(
        id=f"qlog_{uuid.uuid4().hex[:16]}",
        customer_id=customer.id,
        query_text=query_text,
        query_vector=[],
        match_type=outcome.match_type,
        injection_method=outcome.injection_method,  # type: ignore[arg-type]
        confidence_gate_fires=0,
        matched_facts=outcome.matched_crystal_ids,
        response_text=upstream.assistant_text,
        response_confidence_at_commit=None,
        upstream_call_made=True,
        shadow_ran=will_shadow,
        shadow_delta=shadow_delta,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        shadow_prompt_tokens=shadow_prompt_tokens,
        shadow_completion_tokens=shadow_completion_tokens,
        prompt_token_overhead=prompt_token_overhead,
        concept_top_config=outcome.concept_top_config,
        concept_top_score=(
            outcome.concept_top_score
            if outcome.concept_path_ran
            else None
        ),
        concept_payload=outcome.concept_payload,
        sequence_id=sequence_id,
        turn_index=turn_index,
        routed_crystal_id=(
            outcome.matched_crystal_ids[0]
            if outcome.matched_crystal_ids
            else None
        ),
        top1_score=outcome.routing_top1,
        top2_score=outcome.routing_top2,
        latency_ms=total_latency_ms,
        timestamp=datetime.now(timezone.utc),
    )
    try:
        await store.write_query_log(log)
    except Exception as e:
        logger.error(
            "query_log.write_failed",
            customer_id=customer.id,
            error=str(e),
        )

    # ---- Growth G3 (cost accounting): emit one cost row -----------------
    #
    # Via the shared cost helper (cost/emit.py): flag-gated on
    # enable_cost_accounting + fail-safe, so when off this is a no-op and
    # behavior is byte-identical. Records the PRIMARY upstream call's tokens.
    # The upstream usage is OpenAI-shaped (no cache fields), so only
    # input/output are metered here; the agent path meters cache tokens too.
    # A plain proxy turn has no agent session (session_id None); the agent,
    # cognition, and depth paths attribute their own calls; the proxy's
    # research-loop 2nd call + shadow call remain unmetered for now.
    # Task-key attribution (Phase 3 G3): when the caller is a disposable
    # box, the cost row lands under session_id = task_id — the same sum the
    # budget check at the auth door and the remote-task monitor both read.
    # getattr twice: internal callers (admin chat, tests) delegate here
    # with duck-typed requests that may lack .state entirely.
    _task_id = getattr(getattr(request, "state", None), "task_key_task_id", None)
    await record_model_call(
        store=store,
        customer_id=customer.id,
        model=model,
        input_tokens=prompt_tokens or 0,
        output_tokens=completion_tokens or 0,
        origin="disposable_task" if _task_id else "interactive",
        session_id=_task_id,
        operator_id=operator.id if operator is not None else None,
    )

    # V3 Phase 9: Process crystal tool calls and complete the tool-use loop.
    #
    # Phase 9C (P0.59): the handle_signals call site now passes
    # sequence_id, turn_index, agent_model, and mcr_enabled=True so
    # Phase 9B's BD-3 + BD-11 writes fire in production.
    #
    # We capture crystal_calls as a module-scope variable here so
    # the Phase 9C MCR emission step below can build a proxy-shaped
    # agent_result from them.
    _crystal_calls_for_trace: list[dict[str, Any]] = []

    if _crystal_tools_injected:
        try:
            crystal_calls, _other_calls = extract_crystal_tool_calls(upstream.openai_format)
            _crystal_calls_for_trace = list(crystal_calls)

            if crystal_calls:
                signals = parse_tool_calls(crystal_calls)

                _conv_context_parts = []
                for m in messages[-6:]:
                    role = m.get("role", "")
                    content = m.get("content", "")
                    if isinstance(content, str) and content.strip():
                        _conv_context_parts.append(f"{role}: {content[:200]}")
                _conv_context = "\n".join(_conv_context_parts)

                # Phase 9C (P0.59): flip mcr_enabled=True; pass the
                # soft-join key + agent_model so Phase 9B's writes
                # have everything they need.
                try:
                    stats = await handle_signals(
                        signals,
                        customer_id=customer.id,
                        store=store,
                        encoder=request.app.state.prompt_encoder,
                        vector_store=request.app.state.vector_store,
                        vector_index=getattr(request.app.state, "vector_index", None),
                        conversation_context=_conv_context,
                        sequence_id=sequence_id,
                        turn_index=turn_index,
                        agent_model=model,
                        mcr_enabled=True,
                    )
                except Exception as e:
                    logger.warning("push_pull.handle_error", error=str(e))
                    stats = {"_immediate_research": []}

                immediate_research = stats.get("_immediate_research", [])

                if immediate_research:
                    tool_results_msgs = []
                    _vector_index_ref = getattr(request.app.state, "vector_index", None)
                    _encoder_ref = getattr(request.app.state, "prompt_encoder", None)

                    _inline_conv_parts = []
                    for m in messages[-6:]:
                        role = m.get("role", "")
                        content = m.get("content", "")
                        if isinstance(content, str) and content.strip():
                            _inline_conv_parts.append(f"{role}: {content[:300]}")
                    _inline_conv_context = "\n".join(_inline_conv_parts)

                    for tc in crystal_calls:
                        func = tc.get("function", {})
                        tc_name = func.get("name", "")
                        tc_id = tc.get("id", "")

                        if tc_name == "crystal_pull_research" and tc_id:
                            args_str = func.get("arguments", "{}")
                            try:
                                args = json.loads(args_str) if isinstance(args_str, str) else args_str
                            except Exception:
                                args = {}

                            research_result = await run_inline_research(
                                topic=args.get("topic", ""),
                                customer_id=customer.id,
                                store=store,
                                vector_index=_vector_index_ref,
                                encoder=_encoder_ref,
                                conversation_context=_inline_conv_context,
                            )
                            tool_results_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": research_result,
                            })
                        elif tc_id:
                            tool_results_msgs.append({
                                "role": "tool",
                                "tool_call_id": tc_id,
                                "content": "Processed.",
                            })

                    if tool_results_msgs:
                        loop_messages = list(messages)
                        asst_msg = upstream.openai_format["choices"][0]["message"]
                        loop_messages.append(asst_msg)
                        loop_messages.extend(tool_results_msgs)

                        logger.info(
                            "push_pull.tool_loop",
                            customer_id=customer.id,
                            research_topics=[r["topic"][:40] for r in immediate_research],
                            tool_results=len(tool_results_msgs),
                        )

                        try:
                            loop_upstream = await client.complete(
                                messages=loop_messages,
                                model=model,
                                temperature=body.temperature,
                                max_tokens=body.max_tokens,
                                **extra,
                            )
                            upstream = loop_upstream
                            logger.info(
                                "push_pull.tool_loop_complete",
                                customer_id=customer.id,
                                loop_tokens=loop_upstream.prompt_tokens,
                                loop_completion=loop_upstream.completion_tokens,
                            )
                        except Exception as e:
                            logger.warning("push_pull.tool_loop_failed", error=str(e))

                # Strip crystal tool calls from the final response
                for choice in upstream.openai_format.get("choices", []):
                    msg = choice.get("message", {})
                    tool_calls = msg.get("tool_calls", [])
                    if tool_calls:
                        cleaned = [
                            tc for tc in tool_calls
                            if tc.get("function", {}).get("name", "") not in CRYSTAL_TOOL_NAMES
                        ]
                        if cleaned:
                            msg["tool_calls"] = cleaned
                        else:
                            msg.pop("tool_calls", None)
                            if msg.get("content") is None or msg.get("content") == "":
                                confirmations = []
                                for tc in crystal_calls:
                                    fn = tc.get("function", {})
                                    name = fn.get("name", "")
                                    try:
                                        args = json.loads(fn.get("arguments", "{}"))
                                    except Exception:
                                        args = {}
                                    if name == "crystal_push_gap":
                                        confirmations.append(f"Knowledge gap recorded: {args.get('subject', args.get('missing', '')[:60])}")
                                    elif name == "crystal_push_store":
                                        confirmations.append(f"Knowledge stored: {args.get('key', '')[:60]}")
                                    elif name == "crystal_push_correct":
                                        confirmations.append(f"Correction flagged: {args.get('key', '')[:60]}")
                                    elif name == "crystal_pull_research":
                                        confirmations.append(f"Research requested: {args.get('topic', '')[:60]}")
                                msg["content"] = "\n".join(confirmations) if confirmations else (upstream.assistant_text or "Done.")

                logger.info(
                    "push_pull.round_trip",
                    customer_id=customer.id,
                    crystal_calls=len(crystal_calls),
                    customer_calls=len(_other_calls),
                    signals=signals.total_count if signals.has_signals else 0,
                    looped=bool(immediate_research),
                )
        except Exception as e:
            logger.warning("push_pull.round_trip_failed", error=str(e))

    # Feed conversation turn to Mem0 for session memory (Wave 7F).
    if getattr(request.app.state, "mem0", None) is not None:
        try:
            # Sync mem0 client write (embedding + network) — off the loop.
            await asyncio.to_thread(
                add_conversation_turn,
                query_text=query_text,
                response_text=upstream.assistant_text,
                customer_id=customer.id,
                sequence_id=sequence_id,
            )
        except Exception as e:
            logger.warning("mem0.post_response_failed", error=str(e))

    # ---- Growth G1 (citations): provenance + grounding + ledger --------
    #
    # Flag-gated (settings.enable_citations) + fail-safe: when off, this
    # block is skipped and the response is byte-identical to pre-G1. When on
    # (non-streaming only — streaming citations deferred like MCR per P0.57),
    # parse the [[cc:N]] markers the model emitted, map them to the injected
    # crystals via outcome.citation_manifest, grounding-check each against the
    # cited content (drop spurious), rewrite the markers to clean [N] refs +
    # append a Sources footer, and record the result for G4's metering rail.
    # v1 cites the primary injected crystal only, so the manifest holds one
    # source and every cited handle grounds against outcome.injected_text
    # (that source's raw content).
    if (
        settings.enable_citations
        and not body.stream
        and getattr(outcome, "citation_manifest", None)
    ):
        try:
            _choices = upstream.openai_format.get("choices") or []
            _msg = _choices[0].get("message", {}) if _choices else {}
            _answer = _msg.get("content") or ""
            _cited_sources = map_citations(
                parse_citations(_answer), outcome.citation_manifest
            )
            _grounded_count = 0
            if _cited_sources:
                _source_text = outcome.injected_text or ""
                _grounded_results = await ground_citations(
                    request.app.state.prompt_encoder,
                    _answer,
                    [(s, _source_text) for s in _cited_sources],
                )
                _kept = [
                    r["source"].handle
                    for r in _grounded_results
                    if r["grounded"]
                ]
                _grounded_count = len(_kept)
                # Rewrite markers ([[cc:N]] → [N] for grounded, strip
                # spurious) and append the provenance footer.
                _new_answer = rewrite_markers(_answer, _kept)
                _footer = render_sources_footer(
                    [r["source"] for r in _grounded_results if r["grounded"]]
                )
                if _footer:
                    _new_answer = f"{_new_answer}\n\n{_footer}"
                _msg["content"] = _new_answer

                # Record every parsed citation (grounded + spurious) for the
                # ledger rail + telemetry; grounded gates G4 credit.
                await store.record_citations(
                    customer.id,
                    query_log_id=log.id,
                    citations=[
                        {
                            "crystal_id": r["source"].crystal_id,
                            "version": r["source"].version,
                            "handle": r["source"].handle,
                            "claim_span": r["claim_span"],
                            "grounding_score": r["grounding_score"],
                            "grounded": r["grounded"],
                        }
                        for r in _grounded_results
                    ],
                )
                logger.info(
                    "citations.processed",
                    customer_id=customer.id,
                    cited=len(_cited_sources),
                    grounded=len(_kept),
                )

                # ---- Growth G4 (marketplace metering) -----------------
                # Flag-gated (settings.enable_marketplace_metering) +
                # fail-safe. A GROUNDED citation of a general/marketplace
                # crystal mints a shard credit for its owner — the closed
                # loop where G1's rail drives G4's economy. Idempotent on
                # (log.id, crystal): re-grounding the same turn never
                # double-credits. Self-traffic + non-marketplace crystals are
                # excluded inside record_citation_credit. Spurious (ungrounded)
                # citations never reach here. The bounded reward pool (D7) is
                # deferred — each grounded citation credits a fixed shard and
                # the fractional weight is preserved in the ledger.
                if settings.enable_marketplace_metering:
                    for _r in _grounded_results:
                        if not _r["grounded"]:
                            continue
                        try:
                            _cry = await store.get_crystal(
                                _r["source"].crystal_id
                            )
                            if _cry is None:
                                continue
                            await store.record_citation_credit(
                                crystal_id=_r["source"].crystal_id,
                                owner_operator_id=getattr(
                                    _cry, "owner_operator_id", None
                                ),
                                crystal_group_team_id=getattr(
                                    _cry, "group_team_id", None
                                ),
                                crystal_type=getattr(
                                    _cry, "crystal_type", None
                                ),
                                crystal_customer_id=getattr(
                                    _cry, "customer_id", None
                                ),
                                consuming_team_id=customer.id,
                                interaction_id=log.id,
                                raw_weight=1.0,
                            )
                        except Exception as e:
                            logger.warning(
                                "marketplace.credit_failed",
                                customer_id=customer.id,
                                error=str(e),
                            )

            # G1c — the dual: retrieval injected a crystal but NOTHING in the
            # answer grounded to it (the model cited nothing, or only
            # spuriously). That's a signal the bank may lack coverage for
            # this query → emit a knowledge_gaps CANDIDATE (reviewed
            # downstream via the gaps → cognition loop, never auto-acted).
            # Conservative: only for a substantive answer, so trivial
            # "Done."-style turns don't generate noise. A citation and a gap
            # are duals — a grounded citation says "the bank had it"; this
            # says "the bank was consulted but didn't."
            if (
                _grounded_count == 0
                and len((_answer or "").strip()) >= _UNCITED_GAP_MIN_CHARS
            ):
                try:
                    await store.create_knowledge_gap(
                        customer.id,
                        domain=None,
                        subject=(query_text[:256] or None),
                        missing=(
                            "An answer was produced with retrieved knowledge "
                            "injected, but no part of it grounded to a cited "
                            "crystal — the bank may lack coverage for this "
                            "query."
                        ),
                        source="uncited_answer",
                    )
                    logger.info(
                        "citations.uncited_gap", customer_id=customer.id
                    )
                except Exception as e:
                    logger.warning(
                        "citations.uncited_gap_failed",
                        customer_id=customer.id,
                        error=str(e),
                    )
        except Exception as e:
            # Fail-safe (mirrors the MCR emission discipline): citation
            # processing never breaks the customer's response.
            logger.warning(
                "citations.post_response_failed",
                customer_id=customer.id,
                error=str(e),
                error_type=type(e).__name__,
            )

    # ---- Phase 9C (P0.60): MCR trace + self-critique emission ----------
    #
    # Placement: AFTER the entire crystal-tool-loop processing block
    # (potentially including a 2nd upstream call), AFTER tool-call
    # stripping, AFTER the Mem0 turn-add — BEFORE return JSONResponse.
    # This ensures the trace's final_text reflects what the customer
    # actually receives and the tool_calls include the crystal tool
    # calls the LLM made.
    #
    # The self-critique runs through the provider-neutral LLM seam
    # (emit_mcr_artifacts -> run_self_critique -> get_llm_client). When
    # no provider is configured, is_ready() is False and we skip the
    # critique so the trace still persists without a failed step. The
    # proxy's customer-facing response is unaffected regardless.
    try:
        proxy_agent_result = _build_proxy_agent_result(
            upstream_assistant_text=upstream.assistant_text,
            upstream_openai_format=upstream.openai_format,
            crystal_calls=_crystal_calls_for_trace,
            matched_crystal_ids=outcome.matched_crystal_ids,
        )

        # emit_mcr_artifacts NEVER raises (P0.44). If self-critique
        # can't run (no client, no API key), it persists an empty
        # critique with the failure noted in summary_text. The trace
        # still lands.
        await emit_mcr_artifacts(
            store=store,
            customer_id=customer.id,
            user_query=query_text,
            agent_result=proxy_agent_result,
            anthropic_client=None,
            sequence_id=sequence_id,
            turn_index=turn_index,
            query_log_id=log.id,
            skip_self_critique=(not get_llm_client().is_ready()),
        )
    except Exception as e:
        # Defense-in-depth: P0.44 says emit_mcr_artifacts never
        # raises, but a hypothetical bug in the helper or the
        # _build_proxy_agent_result function must not break the
        # customer's response.
        logger.warning(
            "mcr.proxy_emit_failed",
            customer_id=customer.id,
            error=str(e),
            error_type=type(e).__name__,
        )

    return JSONResponse(content=upstream.openai_format)


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    request: Request,
    customer: Annotated[Customer, Depends(task_principal_customer)],
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    operator: Annotated[Optional[Operator], Depends(task_principal_operator)] = None,
) -> Any:
    """OpenAI-compatible chat completions (public proxy deployment mode).

    Auth resolves the bearer to a principal (resolve_principal): EITHER a
    team key (operator None -- today's unscoped, whole-team view) OR an
    operator key (operator set -- POSIX-scoped retrieval, F2). The two
    projection deps read one cached resolve_principal, so the bearer is
    resolved once. The full pipeline lives in run_chat_completion, which the
    trusted-internal admin chat route (/admin/api/customers/{id}/chat) also
    delegates to with operator=None.
    """
    return await run_chat_completion(
        body=body, request=request, customer=customer, operator=operator,
        store=store,
    )
