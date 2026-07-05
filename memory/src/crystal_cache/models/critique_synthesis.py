"""CritiqueSynthesis — MCR artifact 5 per `docs/MCR_FRAMEWORK.md` §4.5.

The output of the metacognitive layer's review of a trace's critiques.
One row per (trace, review-window) per P0.72 — a trace re-reviewed
later produces a NEW row, not an update, preserving the audit trail.

Phase 10A v1 algorithm (P0.74) walks the trace's action_items,
classifies each via the alignment row, and assigns to
promoted/deferred/dropped buckets per locked rules:
  - substrate_observation -> always defer (Principle 9)
  - same_action 2-critic -> promote
  - contradictory -> defer
  - similar -> defer
  - divergent agent_self -> promote (Phase 9.5 shadow is sampled)
  - divergent shadow -> defer (calibration question, not action)

Mirrors CritiqueSynthesisRow 1:1.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class CritiqueSynthesis(BaseModel):
    id: str
    customer_id: str

    # Soft pointer to reasoning_traces.id.
    trace_id: Optional[str] = None

    # Review window. For Phase 10A both fields are typically the
    # created_at moment; Phase 10B's scheduler may set wider windows.
    review_window_start: Optional[datetime] = None
    review_window_end: Optional[datetime] = None

    # The three decision buckets. Mutually exclusive per synthesis row.
    promoted_item_ids: list[str] = Field(default_factory=list)
    deferred_item_ids: list[str] = Field(default_factory=list)
    dropped_item_ids: list[str] = Field(default_factory=list)

    # Audit trail: action_item.id -> short rationale string.
    promotion_rationales: dict[str, str] = Field(default_factory=dict)

    # Placeholder for Phase 10B. Empty in Phase 10A.
    critic_calibration_updates: list[dict[str, Any]] = Field(
        default_factory=list
    )
    # Placeholder for future cross-trace pattern extraction. Empty in 10A.
    cross_trace_patterns: list[dict[str, Any]] = Field(default_factory=list)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
