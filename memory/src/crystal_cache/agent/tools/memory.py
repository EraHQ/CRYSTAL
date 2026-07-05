"""Memory tools — Mem0 session memory + crystal long-term memory.

Per D-A5: two memory tools, not one collapsed `memory(scope=...)`.
They have different lifetimes (session vs. long-term), different
storage backends (Mem0/Qdrant vs. crystal bank), and different write
shapes (Mem0 takes conversation turns; crystals take prompt+answer
pairs). One tool would hide the distinction the bonder needs.

CONTEXT ASSIGNMENTS (D-A10 + §6.5.2):
- `mem0_recall` and `crystal_recall` are read-side shared
  (agent ✅, cognition ✅).
- `mem0_write` and `crystal_write` are write-side agent-only
  (agent ✅, cognition ❌). Cognition writes via its commit gate
  after validator approval — see cognition/engine.py::_commit_and_finalize.

Wave 7F's `retrieval/mem0_session.py` exposes the four Mem0 surface
functions. These tools wrap them with the agent's customer_id-first
calling convention.

`crystal_recall` and `crystal_write` route through the store's
`add_pair_for_customer` (write) and a combination of route +
list_facts_for_crystal (read). The read-side intentionally reaches
for a higher-level helper than the bare V3 routers — the agent
already has `knowledge_search` for direct V3-router access; this
tool is meant to be the "lookup what we know about X" convenience
entry point.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from ...encoding.executor import encode_native_async
from .retrievers import _get_state

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# mem0_recall
# ---------------------------------------------------------------------------

@register_tool(
    name="mem0_recall",
    description=(
        "Look up session memory (recent conversation context, "
        "working state) for the current sequence. Returns a hint "
        "dict with extracted entities and locators that ground "
        "follow-up turns. Best for: 'remember what we were just "
        "talking about', 'recall the locator we just used', "
        "'check what we already established'. Empty dict when "
        "Mem0 is not configured (CC_MEM0_ENABLED not set)."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query for the Mem0 lookup (free text).",
            },
            "sequence_id": {
                "type": "string",
                "description": (
                    "Conversation sequence id to scope the lookup. "
                    "When omitted, the lookup is unscoped (returns "
                    "session memory across this customer)."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum memories to consider. Default 5.",
                "default": 5,
            },
        },
        "required": ["query"],
    },
    returns_description=(
        "{'hints': dict[str, str]}  # e.g. {'locator_prefix': 'Scene 5', 'subject': '...'}"
    ),
)
async def mem0_recall(
    customer_id: str,
    query: str,
    sequence_id: Optional[str] = None,
    limit: int = 5,
) -> dict[str, Any]:
    from ...retrieval.mem0_session import search_session_context

    hints = search_session_context(
        query_text=query,
        customer_id=customer_id,
        sequence_id=sequence_id,
        limit=limit,
    )
    return {"hints": hints}


# ---------------------------------------------------------------------------
# mem0_write
# ---------------------------------------------------------------------------

@register_tool(
    name="mem0_write",
    description=(
        "Persist a conversation turn to session memory. The agent "
        "calls this after producing a response so future turns in "
        "the same sequence have working context. Fire-and-forget: "
        "writes are best-effort and silently no-op when Mem0 is not "
        "configured. Only the agent calls this — cognition workers "
        "do not have session state to persist."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "User's turn text (the prompt or question).",
            },
            "response": {
                "type": "string",
                "description": (
                    "Assistant's response text. Truncated to 2000 "
                    "characters before storage to bound Mem0's "
                    "fact-extraction prompt budget."
                ),
            },
            "sequence_id": {
                "type": "string",
                "description": "Conversation sequence id to scope this turn.",
            },
        },
        "required": ["query", "response"],
    },
    returns_description="{'stored': bool}",
)
async def mem0_write(
    customer_id: str,
    query: str,
    response: str,
    sequence_id: Optional[str] = None,
) -> dict[str, Any]:
    from ...retrieval.mem0_session import add_conversation_turn, get_mem0

    enabled = get_mem0() is not None
    if enabled:
        add_conversation_turn(
            query_text=query,
            response_text=response,
            customer_id=customer_id,
            sequence_id=sequence_id,
        )
    return {"stored": enabled}


# ---------------------------------------------------------------------------
# crystal_recall
# ---------------------------------------------------------------------------

@register_tool(
    name="crystal_recall",
    description=(
        "General-purpose 'what do we know about X' lookup against "
        "the crystal bank. Returns the top matching crystals with "
        "their facts. Convenience entry point above the four V3 "
        "router tools — use this when the agent wants a quick "
        "lookup without choosing between content/knowledge/"
        "navigation/depth. For more targeted retrieval, prefer the "
        "specific V3 tool."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language query for the recall.",
            },
            "k": {
                "type": "integer",
                "description": "Maximum number of facts to consider. Default 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
    returns_description=(
        "{'crystals': [{'crystal_id': str, 'facts': [{'fact_id': str, "
        "'prompt_text': str, 'claim_text': str, 'pair_type': str, "
        "'score': float}]}], 'count': int}"
    ),
)
async def crystal_recall(
    customer_id: str,
    query: str,
    k: int = 10,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]
    fact_store = state["fact_vector_store"]
    encoder = state["encoder"]

    query_vector = await encode_native_async(encoder, query)
    search_results = await fact_store.search(
        customer_id=customer_id,
        query_vector=query_vector,
        # All pair_types — convenience entry point covers entity attrs,
        # Q&A, relationships, content chunks. The agent can filter
        # downstream if it cares.
        pair_types=[
            "entity_attribute",
            "question_answer",
            "entity_relationship",
            "content_chunk",
        ],
        k=k,
    )

    # Group facts by their crystal so the agent can see crystal-level
    # structure. Same pattern as cognition's _worker_crystal_search.
    crystals_by_id: dict[str, dict[str, Any]] = {}
    for fact_id, crystal_id, pair_type, score in search_results:
        crystal_entry = crystals_by_id.setdefault(
            crystal_id,
            {"crystal_id": crystal_id, "facts": []},
        )
        # Load the facts for this crystal once
        facts = await store.list_facts_for_crystal(crystal_id)
        for f in facts:
            if f.id == fact_id:
                crystal_entry["facts"].append({
                    "fact_id": f.id,
                    "prompt_text": f.prompt_text or "",
                    "claim_text": (f.claim_text or f.answer_value or "")[:1500],
                    "pair_type": pair_type,
                    "score": round(score, 4),
                })
                break

    crystals = list(crystals_by_id.values())
    return {"crystals": crystals, "count": len(crystals)}


# ---------------------------------------------------------------------------
# crystal_write
# ---------------------------------------------------------------------------

@register_tool(
    name="crystal_write",
    description=(
        "Persist a new (prompt, answer) pair to the customer's "
        "crystal bank. The bonder dispatches based on pair_type. "
        "Use this when the agent has produced or confirmed "
        "knowledge worth retaining for future conversations. "
        "Write-side: agent-only (cognition workers cannot write "
        "directly; they request writes via the commit gate after "
        "validator approval — D-A10)."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "The prompt/key for the pair (sparse-key form preferred).",
            },
            "value": {
                "type": "string",
                "description": "The answer/value to associate with the key.",
            },
            "pair_type": {
                "type": "string",
                "description": (
                    "One of: entity_attribute, question_answer, "
                    "entity_relationship, content_chunk, "
                    "failure_reflection, behavior_rule, "
                    "cached_solution. Default question_answer."
                ),
                "default": "question_answer",
            },
            "crystal_type": {
                "type": "string",
                "description": "Crystal type id (registry-defined). Default 'customer:legacy'.",
                "default": "customer:legacy",
            },
            "source_kind": {
                "type": "string",
                "description": (
                    "One of: model_reasoning, failed_reasoning, "
                    "user_provided, ingested_document. Default "
                    "model_reasoning."
                ),
                "default": "model_reasoning",
            },
            "answer_value": {
                "type": "string",
                "description": (
                    "Optional cached answer for cache-hit eligibility. "
                    "When set, future retrievals on this key may "
                    "short-circuit the upstream LLM."
                ),
            },
        },
        "required": ["key", "value"],
    },
    returns_description="{'crystal_id': str, 'fact_id': str, 'pair_type': str}",
)
async def crystal_write(
    customer_id: str,
    key: str,
    value: str,
    pair_type: str = "question_answer",
    crystal_type: str = "customer:legacy",
    source_kind: str = "model_reasoning",
    answer_value: Optional[str] = None,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]

    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer_id,
        prompt_text=key,
        answer_text=value,
        pair_type=pair_type,
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        vector_index=state.get("vector_index"),
        crystal_type=crystal_type,
        source_kind=source_kind,
        answer_value=answer_value,
    )
    return {
        "crystal_id": crystal.id,
        "fact_id": fact.id,
        "pair_type": fact.pair_type,
    }
