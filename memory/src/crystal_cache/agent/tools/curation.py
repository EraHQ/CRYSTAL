"""Curation tools — learn + self-curation reads, promoted into the registry.

WS C step 4. These three tools are exposed BOTH on the external MCP memory
surface (memory_learn / memory_conflicts / memory_gaps bridge to them) AND to
the agent loop and cognition, per the locked decision to promote the curation
tools so the agent can teach memory from outcomes and consult what its memory
contradicts / lacks. The implementations live HERE (single source of truth);
the MCP server bridges to them like every other registry tool.

Contexts:
  - crystal_learn         agent-only (write-side; cognition writes via its
                          commit gate, like crystal_write).
  - knowledge_conflicts   agent + cognition (read-only self-curation surface).
  - knowledge_gaps        agent + cognition (read-only self-curation surface).

Mode-agnostic: nothing here assumes the caller writes code. State (store,
encoder, vector_store) is injected the same way the retriever tools get it —
via set_tool_state / _get_state.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# crystal_learn — teach from an outcome (write-side)
# ---------------------------------------------------------------------------

@register_tool(
    name="crystal_learn",
    description=(
        "Teach memory from an outcome. outcome='success' caches a "
        "prompt -> solution pair for fast future recall; outcome='fail' records "
        "a correction (pass 'signal' describing what went wrong) so the system "
        "learns from the mistake. Use after you find out whether a past answer "
        "was right or wrong. Write-side: agent-only."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "prompt": {
                "type": "string",
                "description": "The task/question prompt the outcome is about.",
            },
            "response": {
                "type": "string",
                "description": "The answer/solution that was produced.",
            },
            "outcome": {
                "type": "string",
                "description": "'success' (cache the solution) or 'fail' (record a correction). Default 'success'.",
                "default": "success",
            },
            "signal": {
                "type": "string",
                "description": "On failure, a short description of what was wrong (optional).",
            },
            "crystal_type": {
                "type": "string",
                "description": "Crystal type id. Default 'customer:legacy'.",
                "default": "customer:legacy",
            },
        },
        "required": ["prompt", "response"],
    },
    returns_description=(
        "{'crystals_written': int, 'cached'?: bool, 'reflection'?: str, "
        "'knowledge'?: str, 'category'?: str, 'error'?: str}"
    ),
)
async def crystal_learn(
    customer_id: str,
    prompt: str,
    response: str,
    outcome: str = "success",
    signal: Optional[str] = None,
    crystal_type: str = "customer:legacy",
) -> dict[str, Any]:
    from ...learning import LearningService

    state = _get_state()
    svc = LearningService(
        store=state["store"],
        encoder=state["encoder"],
        vector_store=state["vector_store"],
        vector_index=state.get("vector_index"),
    )
    if outcome == "fail":
        result = await svc.learn_from_failure(
            customer_id=customer_id,
            prompt=prompt,
            response=response,
            failure_signal=signal or "User indicated this response was incorrect",
            crystal_type=crystal_type,
        )
        return {
            "crystals_written": result.crystals_written,
            "reflection": result.reflection,
            "knowledge": result.knowledge,
            "category": result.category,
            "error": result.error,
        }
    cached = await svc.cache_success(
        customer_id=customer_id,
        prompt=prompt,
        solution=response,
        crystal_type=crystal_type,
    )
    return {"crystals_written": 1 if cached else 0, "cached": cached}


# ---------------------------------------------------------------------------
# knowledge_conflicts — what the memory contradicts itself on (read)
# ---------------------------------------------------------------------------

@register_tool(
    name="knowledge_conflicts",
    description=(
        "List contradictions the system has detected in its own memory — pairs "
        "of stored facts that can't both be true. Returns each conflict's "
        "subject and the two conflicting claims. Use to check what the memory "
        "disagrees with itself on before trusting it. Read-only."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status: open / resolved / dismissed. Default 'open'.",
                "default": "open",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum conflicts to return. Default 50.",
                "default": 50,
            },
        },
    },
    returns_description=(
        "{'conflicts': [{'id','subject','claim_a','claim_b','status',"
        "'detector','created_at'}], 'count': int}"
    ),
)
async def knowledge_conflicts(
    customer_id: str,
    status: str = "open",
    limit: int = 50,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]
    conflicts = await store.list_knowledge_conflicts(
        customer_id, status=status or None, limit=limit,
    )
    return {
        "conflicts": [
            {
                "id": c.id,
                "subject": c.subject,
                "claim_a": c.claim_a,
                "claim_b": c.claim_b,
                "status": c.status,
                "detector": c.detector,
                "created_at": c.created_at.isoformat() if c.created_at else None,
            }
            for c in conflicts
        ],
        "count": len(conflicts),
    }


# ---------------------------------------------------------------------------
# knowledge_gaps — what the memory is missing (read)
# ---------------------------------------------------------------------------

@register_tool(
    name="knowledge_gaps",
    description=(
        "List gaps the system has identified in its own memory — things it was "
        "asked about or expected to know but doesn't. Returns each gap's subject "
        "and a description of what's missing. Use to see what the memory lacks "
        "(and might need taught). Read-only."
    ),
    contexts={"agent", "cognition"},
    parameters_schema={
        "type": "object",
        "properties": {
            "status": {
                "type": "string",
                "description": "Filter by status: open / filled / closed. Default 'open'.",
                "default": "open",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum gaps to return. Default 50.",
                "default": 50,
            },
        },
    },
    returns_description=(
        "{'gaps': [{'id','subject','domain','missing','priority','status',"
        "'source','created_at'}], 'count': int}"
    ),
)
async def knowledge_gaps(
    customer_id: str,
    status: str = "open",
    limit: int = 50,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]
    gaps = await store.list_knowledge_gaps(
        customer_id, status=status or None, limit=limit,
    )
    return {
        "gaps": [
            {
                "id": g.id,
                "subject": g.subject,
                "domain": g.domain,
                "missing": g.missing,
                "priority": g.priority,
                "status": g.status,
                "source": g.source,
                "created_at": g.created_at.isoformat() if g.created_at else None,
            }
            for g in gaps
        ],
        "count": len(gaps),
    }
