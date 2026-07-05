"""MCR emitter — Phase 9A (2026-05-27).

Phase 9 wires the agent built in Phase 7.5 to emit MCR reasoning
traces + self-critique on every response. The emitter lives here
(not inside `Agent.run`) so the agent class stays reusable for
non-MCR callers (tests, scripts, the Phase 9.5 shadow critic which
will share most of the read path).

This module is what `endpoints/agent.py` composes AFTER
`agent.run(...)` returns. It does three things:

  1. Build a `ReasoningTrace` deterministically from the agent's
     tool_calls_log + final state (per P0.46 — system record is
     authoritative; the agent does not free-form its trace for
     Phase 9). Persists via `store.create_reasoning_trace(...)`.

  2. Run a self-critique LLM call (Claude Haiku 4.5 by default via
     `settings.reflection_model`) with the user query + agent
     response + trace summary. Parses a JSON-shaped response into
     a `Critique` plus zero-or-more `ActionItem` rows.

  3. Persist the critique and action items via the store. Update
     the critique's `total_action_items` count to match the items
     actually written.

Failure modes (P0.44 — NEVER raises; `emit_mcr_artifacts` always
returns a dict, with None-valued ids where steps failed):
  - Trace persistence failure → log warning, return with
    trace_id=None. Critique/items not attempted.
  - Self-critique LLM call failure → log warning, persist a
    critique with empty observations and `summary_text` carrying
    the failure detail for forensic review. The agent's response
    to the user is NOT blocked by self-critique failures.
  - Self-critique parse failure → log warning, persist critique
    with `summary_text=raw_response_preview` and zero observations
    / action items.
  - Critique persistence failure → log warning, return with
    critique_id=None. Action items not attempted.
  - Action-item persistence failure (per-item) → log warning,
    skip that item, continue with the next.

Cost: doubles the LLM bill per agent request in Phase 9 (Sonnet
for the response + Haiku for the critique). Phase 9.5's sampling
policy may downgrade this to sampled per D-MCR-10; for Phase 9
self-critique is always-on per D-MCR-9.

Phase 9 SCOPE:
  - Agent endpoint emission (`endpoints/agent.py`). 9B and 9C
    extend this to the push/pull signal handler + chat_proxy
    respectively. The writer pattern is the same; only the
    invocation site changes.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings
from ..llm import get_llm_client

if TYPE_CHECKING:
    # Imports for type-checking only. The runtime code uses string
    # comparisons against the _OBSERVATION_TYPES / _ACTION_TYPES
    # tuples below (which mirror the Literal types in the Pydantic
    # models). Pulling in the Literal types themselves at runtime
    # would create no value and a small import-cost penalty.
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Critique, ReasoningTrace
    from ..models.action_item import ActionItem
    from ..models.action_item import ActionType  # noqa: F401
    from ..models.critique import ObservationType  # noqa: F401

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Allowed-vocabulary lists for the self-critique prompt (P0.45)
# ---------------------------------------------------------------------------

# These mirror the Phase 8.5 P0.40 wire-format strings exactly.
# Mirroring intentionally — if P0.40 ever drifts, the prompt drifts
# with it, but neither can drift silently because both lists feed
# the same Literal types in models/critique.py and models/action_item.py.

_OBSERVATION_TYPES: tuple[str, ...] = (
    "assumption_identified",
    "generalization_from_thin_evidence",
    "source_contradiction",
    "tool_output_questionable",
    "gap_papered_over",
    "border_crossing_unflagged",
    "reasoning_skip",
    "substrate_complaint",
)

_ACTION_TYPES: tuple[str, ...] = (
    "research_task",
    "verification_task",
    "evidence_gathering",
    "gap_declaration",
    "edit_proposal",
    "substrate_observation",
    "escalation",
)


# ---------------------------------------------------------------------------
# Trace extraction (P0.46 — deterministic from system record)
# ---------------------------------------------------------------------------

# Which tools' outputs carry crystals_used info. Phase 9 extracts
# crystals_used from these tools' return values deterministically;
# Phase 9.5+ may extend with agent-emitted structured trace events.
_RETRIEVAL_TOOL_NAMES: frozenset[str] = frozenset({
    "knowledge_search", "content_search", "navigation_search",
    "depth_search", "crystal_recall",
})

# Which tool name signals a gap the agent explicitly flagged.
_GAP_TOOL_NAME = "crystal_push_gap"


def _extract_crystals_used(
    tool_calls_log: list[dict[str, Any]],
) -> list[str]:
    """Pull the set of crystal_ids the agent's retrieval tools touched.

    Each entry in `tool_calls_log` looks like:
        {iteration, tool_name, tool_use_id, input, output, is_error}

    Retrieval tools return dicts with `matched_fact_ids` or similar.
    Phase 9 extracts everything that looks like a fact/crystal id:
        - knowledge_search → output.matched_fact_ids
        - content_search   → output.matched_fact_ids
        - navigation_search → output.matching_keys (no crystal_ids
          on this surface)
        - depth_search     → output.crystal_ids (when present)
        - crystal_recall   → output.matched_fact_ids

    For Phase 9, we conservatively collect fact_ids; the Phase 9.5
    metacognitive layer (when it lands) may diff this against
    QueryLogRow.matched_facts to surface divergence per D-MCR-11.
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for tc in tool_calls_log:
        if tc.get("is_error"):
            continue
        name = tc.get("tool_name", "")
        if name not in _RETRIEVAL_TOOL_NAMES:
            continue
        out = tc.get("output")
        if not isinstance(out, dict):
            continue
        # Try several common keys. Order matters — first hit wins,
        # since some tools may carry multiple lists for different
        # purposes.
        for key in ("matched_fact_ids", "crystal_ids", "fact_ids"):
            val = out.get(key)
            if not isinstance(val, list):
                continue
            for fid in val:
                if not isinstance(fid, str):
                    continue
                if fid in seen_set:
                    continue
                seen_set.add(fid)
                seen.append(fid)
            break  # found a list for this tool; don't double-count
    return seen


def _extract_gaps_felt(
    tool_calls_log: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pull `gaps_felt` entries from crystal_push_gap tool calls.

    Each gap entry is shaped per ReasoningTraceRow docstring:
        {want, why_needed}
    derived from the tool's input arguments.

    For Phase 9, we extract from `crystal_push_gap` tool calls
    only. If the agent uses a different mechanism in future phases
    (a structured "i_dont_know" content block, say), this function
    extends.
    """
    out: list[dict[str, Any]] = []
    for tc in tool_calls_log:
        if tc.get("is_error"):
            continue
        if tc.get("tool_name") != _GAP_TOOL_NAME:
            continue
        args = tc.get("input") or {}
        if not isinstance(args, dict):
            continue
        # crystal_push_gap fields per v3_push_pull.py: domain, subject,
        # missing. Map to {want, why_needed} for the trace.
        want = args.get("missing", "") or args.get("subject", "")
        why = args.get("subject", "") or args.get("domain", "")
        if not want:
            continue
        out.append({
            "want": want,
            "why_needed": why,
        })
    return out


def build_trace_from_agent_result(
    customer_id: str,
    agent_result: dict[str, Any],
    *,
    sequence_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    query_log_id: Optional[str] = None,
) -> dict[str, Any]:
    """Transform an Agent.run() result into kwargs for create_reasoning_trace.

    Returns a dict suitable for splatting into
    `store.create_reasoning_trace(**kwargs)`. Pure function — no
    DB writes; the caller persists.

    Per P0.46, this is the deterministic extraction. The agent does
    NOT free-form its trace events; we read them off the
    tool_calls_log. The five aggregate columns are:

      crystals_used   — fact_ids from retrieval tool outputs
      tool_calls      — the entire tool_calls_log (one entry per call)
      inferences      — empty for Phase 9 (no structured channel yet)
      borders_crossed — empty for Phase 9 (no structured channel yet)
      gaps_felt       — from crystal_push_gap tool calls

    The `events` JSON list mirrors `tool_calls` for now. Phase 9.5+
    may shift events to a more abstract event-stream representation
    while keeping tool_calls as the per-call detail.
    """
    tool_calls_log: list[dict[str, Any]] = list(
        agent_result.get("tool_calls") or []
    )

    crystals_used = _extract_crystals_used(tool_calls_log)
    gaps_felt = _extract_gaps_felt(tool_calls_log)

    # For Phase 9, `events` is a flat mirror of `tool_calls`. We
    # add a single trailing event for the final assistant text so
    # the Phase 10 metacognitive layer can locate the response in
    # the event stream without a separate column.
    events: list[dict[str, Any]] = list(tool_calls_log)
    final_text = agent_result.get("final_text", "")
    if final_text:
        events.append({
            "type": "final_text",
            "text": final_text,
            "stop_reason": agent_result.get("stop_reason"),
        })

    return {
        "customer_id": customer_id,
        "events": events,
        "sequence_id": sequence_id,
        "turn_index": turn_index,
        "query_log_id": query_log_id,
        "crystals_used": crystals_used,
        "tool_calls": tool_calls_log,
        "inferences": [],
        "borders_crossed": [],
        "gaps_felt": gaps_felt,
    }


# ---------------------------------------------------------------------------
# Self-critique prompt (P0.45)
# ---------------------------------------------------------------------------

_SELF_CRITIQUE_SYSTEM_PROMPT = """\
You are a self-critique reviewer for an AI assistant called Crystal Cache. \
Your job is to review the assistant's reasoning in a single response and \
produce a structured critique that surfaces where the assistant's reasoning \
was weak, where it crossed evidence borders without flagging, where it \
papered over gaps, or where it generalized from thin evidence.

You are NOT grading the answer's correctness. You are grading the reasoning \
process — was the assistant honest about what it knew? Did it appropriately \
call tools? Did it acknowledge uncertainty? Did the tool calls actually \
support the conclusions drawn?

Output STRICTLY valid JSON in this exact shape:
{
  "observations": [
    {
      "type": "<one of the allowed observation types>",
      "text": "<one-sentence description>",
      "confidence": <float 0.0-1.0>,
      "anchors": []
    },
    ...
  ],
  "action_items": [
    {
      "action_type": "<one of the allowed action types>",
      "content": {<free-form dict with action-specific keys>},
      "critic_confidence": <float 0.0-1.0>
    },
    ...
  ],
  "summary_text": "<one-paragraph overall assessment>"
}

ALLOWED observation types (use EXACTLY these strings):
  assumption_identified        — agent assumed something not in evidence
  generalization_from_thin_evidence — agent generalized from too little data
  source_contradiction         — agent's claim contradicts a source it consulted
  tool_output_questionable     — agent treated a tool output as reliable when it wasn't
  gap_papered_over             — agent talked around a gap rather than acknowledging it
  border_crossing_unflagged    — agent crossed from evidence to inference without flagging
  reasoning_skip               — agent skipped a reasoning step that was load-bearing
  substrate_complaint          — the system itself (retrieval, tools) made reasoning hard

ALLOWED action_item types (use EXACTLY these strings):
  research_task          — content: {topic, scope, why_needed}
  verification_task      — content: {crystal_id, claim_to_verify}
  evidence_gathering     — content: {topic, suggested_sources}
  gap_declaration        — content: {want, why_needed, domain, subject}
  edit_proposal          — content: {crystal_id, proposed_change, rationale}
  substrate_observation  — content: {subsystem, complaint, severity}
  escalation             — content: {issue, suggested_handler}

If the reasoning was clean, return an empty observations list and empty \
action_items list with a brief positive summary_text. \
NEVER invent observations to seem thorough. Empty critique is acceptable \
and signals honest review.

Output ONLY the JSON object. No prose before or after.\
"""


def _build_self_critique_user_message(
    user_query: str,
    agent_final_text: str,
    tool_calls_log: list[dict[str, Any]],
    crystals_used: list[str],
) -> str:
    """Render the user-message body for the self-critique LLM call.

    Compact JSON serialization keeps the prompt small. The reviewer
    LLM gets exactly the data needed to evaluate the reasoning — the
    user's last message, the agent's response, the system record of
    tool calls and crystals used.
    """
    # Trim long tool outputs so the prompt doesn't balloon. Outputs
    # over ~600 chars are summarized as "(truncated)". The full
    # output lives in the persisted trace; the critique LLM only
    # needs gist for reasoning review.
    def _trim(obj: Any, limit: int = 600) -> Any:
        if isinstance(obj, str) and len(obj) > limit:
            return obj[:limit] + "...(truncated)"
        if isinstance(obj, dict):
            return {k: _trim(v, limit) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_trim(x, limit) for x in obj[:5]]  # cap list len too
        return obj

    payload = {
        "user_query": user_query[:2000],
        "agent_response": agent_final_text[:2000],
        "tool_calls": [
            {
                "tool_name": tc.get("tool_name"),
                "input": _trim(tc.get("input")),
                "output_summary": _trim(tc.get("output")),
                "is_error": tc.get("is_error", False),
            }
            for tc in tool_calls_log
        ],
        "crystals_used": crystals_used,
    }
    return (
        "Review the reasoning in this Crystal Cache interaction:\n\n"
        + json.dumps(payload, default=str, indent=2)
    )


# ---------------------------------------------------------------------------
# Self-critique response parsing
# ---------------------------------------------------------------------------

def _parse_self_critique_response(
    raw_text: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Parse the JSON shape from the self-critique LLM into validated lists.

    Returns: (observations, action_items, summary_text).

    Failure modes (P0.45):
      - Not valid JSON → returns ([], [], raw_text_preview) so the
        critique persists with the raw response as summary for
        forensic review.
      - Unknown observation type → drops that observation, keeps
        others. (Logged at warning.)
      - Unknown action_type → drops that action, keeps others.
      - Missing required fields → drops the entry.

    Confidence values are coerced to floats and clamped to
    [0.0, 1.0]. Missing confidence defaults to 0.5 (neutral).
    """
    # Strip ```json fences if the LLM included them despite
    # instructions.
    text = raw_text.strip()
    if text.startswith("```"):
        # Remove first fence line + final fence line.
        lines = text.split("\n")
        if len(lines) >= 2:
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        logger.warning(
            "mcr_emitter.self_critique_parse_failed",
            raw_preview=raw_text[:200],
        )
        return ([], [], raw_text[:2000])

    if not isinstance(parsed, dict):
        logger.warning(
            "mcr_emitter.self_critique_not_object",
            top_level_type=type(parsed).__name__,
        )
        return ([], [], raw_text[:2000])

    # ---- observations ----
    raw_obs = parsed.get("observations") or []
    observations: list[dict[str, Any]] = []
    if isinstance(raw_obs, list):
        for o in raw_obs:
            if not isinstance(o, dict):
                continue
            obs_type = o.get("type")
            if obs_type not in _OBSERVATION_TYPES:
                logger.warning(
                    "mcr_emitter.unknown_observation_type",
                    received=obs_type,
                )
                continue
            text_field = o.get("text", "")
            if not isinstance(text_field, str) or not text_field.strip():
                continue
            confidence = _coerce_confidence(o.get("confidence"))
            anchors = o.get("anchors", [])
            if not isinstance(anchors, list):
                anchors = []
            observations.append({
                "type": obs_type,
                "text": text_field.strip(),
                "confidence": confidence,
                "anchors": anchors,
            })

    # ---- action_items ----
    raw_items = parsed.get("action_items") or []
    action_items: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for it in raw_items:
            if not isinstance(it, dict):
                continue
            act_type = it.get("action_type")
            if act_type not in _ACTION_TYPES:
                logger.warning(
                    "mcr_emitter.unknown_action_type",
                    received=act_type,
                )
                continue
            content = it.get("content")
            if not isinstance(content, dict):
                content = {}
            critic_conf = _coerce_confidence(it.get("critic_confidence"))
            action_items.append({
                "action_type": act_type,
                "content": content,
                "critic_confidence": critic_conf,
            })

    # ---- summary ----
    summary = parsed.get("summary_text", "")
    if not isinstance(summary, str):
        summary = ""

    return (observations, action_items, summary)


def _coerce_confidence(value: Any) -> float:
    """Coerce a value to a float in [0.0, 1.0]. Default 0.5 on failure."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.5
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


# ---------------------------------------------------------------------------
# Self-critique LLM call
# ---------------------------------------------------------------------------

async def run_self_critique(
    *,
    anthropic_client: Any = None,
    user_query: str,
    agent_final_text: str,
    tool_calls_log: list[dict[str, Any]],
    crystals_used: list[str],
    model: Optional[str] = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Run the self-critique LLM call.

    Returns: (observations, action_items, summary_text). Empty lists
    and empty summary_text are valid outputs and represent "the
    agent's reasoning was clean."

    On any failure (network, parse, etc.), returns empty observations
    + empty action_items and surfaces the failure in summary_text for
    forensic review. NEVER raises — the caller's response to the user
    must not be blocked by self-critique failures (P0.44).
    """
    import asyncio

    user_message = _build_self_critique_user_message(
        user_query=user_query,
        agent_final_text=agent_final_text,
        tool_calls_log=tool_calls_log,
        crystals_used=crystals_used,
    )

    def _call() -> str:
        client = anthropic_client if anthropic_client is not None else get_llm_client()
        return client.complete(
            system=_SELF_CRITIQUE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
            max_tokens=2048,
            temperature=0.0,
            tier="small",
            model=model,
        )

    try:
        raw_text = await asyncio.to_thread(_call)
    except Exception as e:
        logger.warning(
            "mcr_emitter.self_critique_call_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return (
            [],
            [],
            f"self-critique call failed: {type(e).__name__}: {e}",
        )

    if not raw_text.strip():
        logger.warning("mcr_emitter.self_critique_empty_response")
        return ([], [], "self-critique returned empty text")

    return _parse_self_critique_response(raw_text)


# ---------------------------------------------------------------------------
# Top-level emit function — called by endpoints/agent.py
# ---------------------------------------------------------------------------

async def emit_mcr_artifacts(
    *,
    store: "MetadataStore",
    customer_id: str,
    user_query: str,
    agent_result: dict[str, Any],
    anthropic_client: Any = None,
    sequence_id: Optional[str] = None,
    turn_index: Optional[int] = None,
    query_log_id: Optional[str] = None,
    self_critique_model: Optional[str] = None,
    skip_self_critique: bool = False,
) -> dict[str, Any]:
    """Persist MCR trace + critique + action_items for one agent turn.

    Called from endpoint handlers AFTER `agent.run(...)` returns. Per
    P0.44, this is a synchronous fan-out: trace → self-critique LLM
    call → critique → action items, in that order.

    Args:
        store: MetadataStore with the Phase 8.5 McrExtensionsMixin
            bound. Required.
        customer_id: the calling customer.
        user_query: the last user message text (passed to the
            self-critique LLM).
        agent_result: the dict returned by `Agent.run(...)`.
        anthropic_client: the Anthropic client (same one the agent
            used for its primary response; we re-use it for the
            self-critique call).
        sequence_id, turn_index, query_log_id: passed through to the
            trace row's columns.
        self_critique_model: override the model id for the
            self-critique call. Default
            `settings.reflection_model` (Haiku 4.5).
        skip_self_critique: when True (used by tests / dry runs),
            persists only the trace with no critique step.

    Returns: a dict with keys:
        trace_id: the persisted trace id (string), or None if the
            trace write failed.
        critique_id: the persisted critique id, or None if skipped
            or if persistence failed at the critique step.
        action_item_ids: list of persisted action_item ids (possibly
            empty).

    NEVER raises (P0.44). All failure modes log warnings and
    return with the partial state visible on the returned dict.
    """
    out: dict[str, Any] = {
        "trace_id": None,
        "critique_id": None,
        "action_item_ids": [],
    }

    # --- 1. Persist the trace. -------------------------------------
    trace_kwargs = build_trace_from_agent_result(
        customer_id=customer_id,
        agent_result=agent_result,
        sequence_id=sequence_id,
        turn_index=turn_index,
        query_log_id=query_log_id,
    )
    try:
        trace = await store.create_reasoning_trace(**trace_kwargs)
        out["trace_id"] = trace.id
    except Exception as e:
        # Trace persistence failure is meaningful — log and return.
        # The caller's response to the user is unaffected (the
        # response was already produced); we lose the trace this
        # turn. Phase 10's metacognitive layer can pick up the
        # next turn instead.
        logger.warning(
            "mcr_emitter.trace_persist_failed",
            customer_id=customer_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return out

    if skip_self_critique:
        return out

    # --- 2. Self-critique LLM call. --------------------------------
    final_text = agent_result.get("final_text", "")
    tool_calls_log = list(agent_result.get("tool_calls") or [])
    crystals_used = trace_kwargs["crystals_used"]

    observations, action_items, summary_text = await run_self_critique(
        anthropic_client=anthropic_client,
        user_query=user_query,
        agent_final_text=final_text,
        tool_calls_log=tool_calls_log,
        crystals_used=crystals_used,
        model=self_critique_model,
    )

    chosen_model = self_critique_model or settings.reflection_model

    # --- 3. Persist the critique. ----------------------------------
    try:
        critique = await store.create_critique(
            customer_id=customer_id,
            critic_role="agent_self",
            critic_model=chosen_model,
            trace_id=trace.id,
            sequence_id=sequence_id,
            turn_index=turn_index,
            observations=observations,
            summary_text=summary_text or None,
            total_action_items=len(action_items),
        )
        out["critique_id"] = critique.id
    except Exception as e:
        logger.warning(
            "mcr_emitter.critique_persist_failed",
            customer_id=customer_id,
            trace_id=trace.id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return out

    # --- 4. Persist action items. ----------------------------------
    persisted_ids: list[str] = []
    for item_data in action_items:
        try:
            item = await store.create_action_item(
                critique_id=critique.id,
                customer_id=customer_id,
                action_type=item_data["action_type"],
                content=item_data.get("content", {}),
                critic_confidence=item_data.get("critic_confidence"),
            )
            persisted_ids.append(item.id)
        except Exception as e:
            logger.warning(
                "mcr_emitter.action_item_persist_failed",
                customer_id=customer_id,
                critique_id=critique.id,
                action_type=item_data.get("action_type"),
                error=str(e),
                error_type=type(e).__name__,
            )
            continue

    out["action_item_ids"] = persisted_ids

    logger.info(
        "mcr_emitter.artifacts_persisted",
        customer_id=customer_id,
        trace_id=trace.id,
        critique_id=critique.id,
        observation_count=len(observations),
        action_item_count=len(persisted_ids),
    )

    return out
