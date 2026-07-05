"""Critique — the structured output of a single critic reviewing
a single reasoning trace.

MCR artifact 2 per `docs/MCR_FRAMEWORK.md` §4.2. Mirrors
CritiqueRow 1:1.

Critic identity is captured by TWO fields per P0.36:
  - `critic_role`: the kind of critic (agent_self | shadow |
                   specialist) — extensible per D-MCR-6.
  - `critic_model`: the model id (e.g. 'claude-sonnet-4-5-20250929')
                    — needed for §7 calibration which tracks
                    per-model track records.

Observations are inline JSON per P0.37, not a separate table.
Per-observation shape:
  {type: ObservationType, text: str, confidence: float, anchors: list[dict]}

The 8 observation types from MCR §4.2 are frozen for Phase 8.5
per P0.40 but stored as JSON `type` strings so adding a new
type later is a code-level change, not a schema migration.

Phase 8.5 lands the model + schema; writers ship in Phase 9
(agent self-critique) and Phase 9.5 (shadow critic).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Critic role (P0.36, P0.40). 'specialist' is reserved for future
# critic classes per D-MCR-6; Phase 9 introduces agent_self,
# Phase 9.5 introduces shadow.
CriticRole = Literal["agent_self", "shadow", "specialist"]

# Observation type taxonomy from MCR §4.2 (P0.40). Frozen for
# Phase 8.5. Stored as a string inside each observation dict;
# this Literal is the Phase 8.5 type-checking contract, NOT a
# DB constraint — the JSON value is unconstrained at the column
# level so new types can land in a future phase without an
# Alembic step.
ObservationType = Literal[
    "assumption_identified",
    "generalization_from_thin_evidence",
    "source_contradiction",
    "tool_output_questionable",
    "gap_papered_over",
    "border_crossing_unflagged",
    "reasoning_skip",
    "substrate_complaint",
]


class Critique(BaseModel):
    id: str
    customer_id: str

    # Soft pointer to reasoning_traces.id.
    trace_id: Optional[str] = None

    # Soft-join key (duplicated from trace for direct lookup).
    sequence_id: Optional[str] = None
    turn_index: Optional[int] = None

    # Critic identity (P0.36).
    critic_role: CriticRole
    critic_model: str

    # Observations from MCR §4.2 (P0.37). Each entry:
    #   {type: ObservationType, text: str, confidence: float,
    #    anchors: list[dict[str, Any]]}
    # The Pydantic model does not narrow the per-entry shape;
    # consumers validate at use time. Anchor structure is deferred
    # to Phase 9 alongside the trace event schema.
    observations: list[dict[str, Any]] = Field(default_factory=list)

    # Free-text narrative summary. Used by Phase 10's
    # human-surfacing path; not required for action-item generation.
    summary_text: Optional[str] = None

    # Denormalized count maintained by the writer.
    total_action_items: int = 0

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
