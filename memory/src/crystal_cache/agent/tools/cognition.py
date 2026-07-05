"""Cognition entry tool — `cognition_run`.

Per §4.3 + D-A6: the agent sees ONE tool for the orchestrator →
worker → validator loop. The internal multi-model loop is hidden
behind this tool. The agent passes a trigger context (the user's
request + relevant memory) and gets back either a synthesized
answer, a crystal write, or a validation-failed report.

CONTEXT (D-A10 + §6.5.3):
- cognition_run is agent-only. Cognition workers cannot recursively
  call cognition_run — that would create unbounded recursion and
  blow the cost model. One cognition workflow per user message is
  the budget.

WHEN THE AGENT CALLS THIS (P0.20):
- The agent's system prompt instructs it to call cognition_run
  when:
  (a) The task requires producing a deliverable the user will
      save/share (e.g. a report, an article, a structured
      analysis), OR
  (b) The task requires synthesis across 3+ retrieval results
      where the agent would otherwise have to manually compose.
- For one-shot questions answerable from a single retrieval call,
  the agent calls the retriever directly and uses llm_invoke to
  format the answer.

This is the configured heuristic for Phase 7.5. Phase 8 smoke
tests measure actual delegation rate; Phase 11 refines.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

from ..tool_registry import register_tool
from .retrievers import _get_state

logger = structlog.get_logger(__name__)


@register_tool(
    name="cognition_run",
    description=(
        "Run the multi-step cognition workflow (orchestrator → "
        "workers → validator) to produce a validated deliverable. "
        "Call this when the user's request requires synthesizing "
        "multiple sources, producing a saved/shared artifact, or "
        "any task that benefits from validator review. Returns the "
        "deliverable (or a validation-failed report if the workflow "
        "couldn't satisfy the goal after the configured retries)."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": (
                    "The goal statement. Should be specific enough "
                    "for the orchestrator to derive acceptance "
                    "criteria from."
                ),
            },
            "conversation_context": {
                "type": "string",
                "description": (
                    "Relevant context from the agent's conversation "
                    "so far. The orchestrator combines this with "
                    "the goal to plan execution. When omitted, the "
                    "goal alone is used as context."
                ),
            },
            "output_type": {
                "type": "string",
                "description": (
                    "What kind of deliverable: 'crystal' (write a "
                    "new crystal to the bank), 'report' (return the "
                    "synthesized text without writing), or 'file' "
                    "(not yet implemented). Default 'report' for "
                    "user-facing requests; 'crystal' for "
                    "knowledge-curation tasks."
                ),
                "default": "report",
            },
            "source_crystal_id": {
                "type": "string",
                "description": (
                    "Optional parent crystal id when the cognition "
                    "result should be linked back to an existing "
                    "crystal."
                ),
            },
            "max_attempts": {
                "type": "integer",
                "description": (
                    "Maximum orchestrate-workers-validate retries "
                    "before giving up. Default 3."
                ),
                "default": 3,
            },
        },
        "required": ["goal"],
    },
    returns_description=(
        "{'success': bool, 'text': str | None, 'crystal_id': str | "
        "None, 'confidence': float, 'reason': str | None, "
        "'tokens_used': int, 'cost_usd': float}"
    ),
)
async def cognition_run(
    customer_id: str,
    goal: str,
    conversation_context: str = "",
    output_type: str = "report",
    source_crystal_id: str = "",
    max_attempts: int = 3,
) -> dict[str, Any]:
    from ...cognition.engine import run_cognition_workflow
    from ...llm import get_llm_client

    state = _get_state()
    store = state["store"]
    fact_store = state["fact_vector_store"]
    encoder = state["encoder"]

    if not get_llm_client().is_ready():
        return {
            "success": False,
            "text": None,
            "crystal_id": None,
            "confidence": 0.0,
            "reason": (
                "cognition_run requires a configured LLM provider "
                "(set CC_LLM_API_KEY or ANTHROPIC_API_KEY)"
            ),
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

    try:
        result = await run_cognition_workflow(
            goal=goal,
            customer_id=customer_id,
            store=store,
            fact_store=fact_store,
            encoder=encoder,
            conversation_context=conversation_context,
            source_crystal_id=source_crystal_id,
            output_type=output_type,
            trigger_type="agent",
            trigger_id="",
            max_attempts=max_attempts,
        )
    except Exception as e:
        logger.error(
            "cognition_run.workflow_error",
            customer_id=customer_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "success": False,
            "text": None,
            "crystal_id": None,
            "confidence": 0.0,
            "reason": f"workflow error: {e}",
            "tokens_used": 0,
            "cost_usd": 0.0,
        }

    return {
        "success": result.success,
        "text": result.text,
        "crystal_id": result.crystal_id,
        "confidence": result.confidence,
        "reason": result.reason,
        "tokens_used": result.tokens_used,
        "cost_usd": result.cost_usd,
    }
