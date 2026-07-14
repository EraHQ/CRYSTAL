"""CognitionTask — queued cognition workflow.

A persistent record of a cognition workflow request: who asked, what
they asked for, the resulting plan and deliverable. Each row maps to
one CognitionEnvironment (the in-memory ephemeral workspace where the
orchestrator → worker → validator loop executes).

In the v2 agent architecture (per AGENT_ARCHITECTURE.md), the agent
calls `cognition_run` as one of its tools. That call writes a row
here with status='pending', the cognition worker picks it up, runs
the full loop, and updates status to 'complete' or 'failed' with the
deliverable in `result`.

Tasks can also originate from the push/pull protocol:
  - 'research'  — fill_gap follow-up; orchestrator plans a research
                  workflow against the gap's subject
  - 'verify'    — re-check a push that came in below high-confidence
                  but worth investigating
  - 'crystallize_doc' — chunk + extract + write for an uploaded doc

priority='background' tasks are deferred until cognition workers are
idle; 'urgent' tasks preempt the queue. The agent's synchronous
cognition_run call uses 'urgent' so the user-facing wait completes
in foreground.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Task types. The string is permissive because the cognition orchestrator
# is the one routing on it, not a strict registry. Today's known values:
#  'research'        — generic research workflow (the most common shape)
#  'fill_gap'        — research a specific knowledge_gap row
#  'verify'          — re-check a medium-confidence push
#  'crystallize_doc' — chunk + extract + write for an uploaded document
#  'reflect'         — meta-reflection over recent failures
TaskType = Literal[
    "research", "fill_gap", "verify", "crystallize_doc", "reflect",
    # 2026-07-13 (async cognition, ratified Q3A): the agent's
    # cognition_run tool ENQUEUES instead of running inline — the
    # synchronous shape died at Cloud Run's request timeout (504 in
    # the Inspector chat while the run survived server-side). Payload:
    # {topic, conversation_context, source_crystal_id, output_type,
    # max_attempts}. priority='urgent' so it claims ahead of
    # background research.
    "agent_research",
]

# 'urgent'     — agent's synchronous cognition_run call; foreground
# 'background' — async queue work; runs when workers idle
TaskPriority = Literal["urgent", "background"]

# Standard async-task lifecycle.
# 'pending'  → 'running' → 'complete'
# 'pending'  → 'running' → 'failed'
# 'pending'  → 'cancelled' (operator cancelled before start)
TaskStatus = Literal[
    "pending", "running", "complete", "failed", "cancelled"
]


class CognitionTask(BaseModel):
    id: str
    customer_id: str

    task_type: TaskType
    # Free-form payload — the orchestrator decides how to consume it
    # based on task_type. Schema is task-type-specific and not pinned
    # here, matching the v1 design where payload is dict-shaped.
    payload: Optional[dict[str, Any]] = None

    priority: TaskPriority = "background"
    status: TaskStatus = "pending"

    # Populated when status='complete'. The validator-approved
    # deliverable from the cognition workflow.
    result: Optional[dict[str, Any]] = None

    # Set when the deliverable was written as a crystal (e.g.,
    # research result crystallized as a new knowledge crystal).
    result_crystal_id: Optional[str] = None

    # Back-reference to the QueryLog that triggered this task, when
    # the task originated from an agent conversation.
    source_query_id: Optional[str] = None

    error_message: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
