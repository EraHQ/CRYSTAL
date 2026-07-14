"""Cognition entry tools — `cognition_run` (enqueue) + `cognition_status`.

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
        "{'success': bool, 'task_id': str | None, 'status': "
        "'started' | None, 'reason': str | None}"
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
    """2026-07-13 (async cognition, ratified Q3A): ENQUEUE, don't run.

    The synchronous shape executed the whole orchestrator → workers →
    validator loop inside the chat request; a thorough run now takes
    5–15 minutes and Cloud Run's request timeout killed the reply at
    300s (Inspector showed `504: null`) while the run survived
    server-side. The tool now creates a cognition_task
    (priority='urgent' — claims ahead of background research) and
    returns the task id immediately. Results: the Cognition pane
    tracks the run live; the cognition_status tool answers "is it
    done?" conversationally; approved deliverables land in the review
    queue exactly as before.
    """
    from ...llm import get_llm_client

    state = _get_state()
    store = state["store"]

    if not get_llm_client().is_ready():
        return {
            "success": False,
            "task_id": None,
            "status": None,
            "reason": (
                "cognition_run requires a configured LLM provider "
                "(set CC_LLM_API_KEY or ANTHROPIC_API_KEY)"
            ),
        }

    try:
        task = await store.create_cognition_task(
            customer_id,
            task_type="agent_research",
            payload={
                "topic": goal,
                "conversation_context": conversation_context,
                "source_crystal_id": source_crystal_id,
                "output_type": output_type,
                "max_attempts": max_attempts,
            },
            priority="urgent",
        )
    except Exception as e:
        logger.error(
            "cognition_run.enqueue_error",
            customer_id=customer_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return {
            "success": False,
            "task_id": None,
            "status": None,
            "reason": f"could not start research: {e}",
        }

    logger.info(
        "cognition_run.enqueued",
        customer_id=customer_id,
        task_id=task.id,
        goal=goal[:80],
    )
    return {
        "success": True,
        "task_id": task.id,
        "status": "started",
        "reason": None,
    }


@register_tool(
    name="cognition_status",
    description=(
        "Check on a background research run started by cognition_run. "
        "Returns the run's status and, once complete, the deliverable "
        "text and confidence. Call this when the user asks whether "
        "their research is done or wants the result."
    ),
    contexts={"agent"},
    parameters_schema={
        "type": "object",
        "properties": {
            "task_id": {
                "type": "string",
                "description": (
                    "The task id returned by cognition_run."
                ),
            },
        },
        "required": ["task_id"],
    },
    returns_description=(
        "{'status': 'in_progress' | 'complete' | 'failed' | "
        "'not_found', 'text': str | None, 'crystal_id': str | None, "
        "'confidence': float | None, 'error': str | None}"
    ),
)
async def cognition_status(
    customer_id: str,
    task_id: str,
) -> dict[str, Any]:
    state = _get_state()
    store = state["store"]

    task = await store.get_cognition_task(task_id)
    # get_cognition_task is cross-tenant (the worker's read path);
    # the AGENT boundary enforces tenancy — a foreign task id is
    # indistinguishable from a missing one.
    if task is None or task.customer_id != customer_id:
        return {
            "status": "not_found",
            "text": None,
            "crystal_id": None,
            "confidence": None,
            "error": None,
        }

    if task.status in ("pending", "running"):
        return {
            "status": "in_progress",
            "text": None,
            "crystal_id": None,
            "confidence": None,
            "error": None,
        }

    if task.status == "failed":
        return {
            "status": "failed",
            "text": None,
            "crystal_id": None,
            "confidence": None,
            "error": task.error_message,
        }

    result = task.result or {}
    return {
        "status": "complete",
        "text": result.get("findings"),
        "crystal_id": result.get("crystal_id") or task.result_crystal_id,
        "confidence": result.get("confidence"),
        "error": (
            result.get("reason")
            if result.get("action") == "no_actionable_findings" else None
        ),
    }
