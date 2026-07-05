"""ActionItem — a proposed next action emerging from a critique.

MCR artifact 3 per `docs/MCR_FRAMEWORK.md` §4.3. Mirrors
ActionItemRow 1:1.

Hard-linked to a Critique via `critique_id` per P0.35 — items are
1:1-owned by their critique with no re-attachment scenario, so a
SQL FK adds integrity.

Lifecycle statuses (P0.40) mirror MCR §4.5's metacognitive
decision vocabulary plus an explicit terminal 'acted' state:
  pending  → newly created, awaiting metacognitive review
  promoted → metacognitive layer decided to act
  deferred → set aside for re-review later
  dropped  → explicitly discarded (calibration signal)
  acted    → terminal; `acted_artifact_id` points to the
             produced row

Action types (P0.40) come from MCR §4.3 verbatim. Frozen for
Phase 8.5; new types are a P0-locked code change, not a schema
migration.

Phase 8.5 lands the model + schema; writers ship in Phase 9 and
Phase 9.5; the consuming reader (the metacognitive layer that
promotes/defers/drops/acts) ships in Phase 10.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


# Action type taxonomy from MCR §4.3 (P0.40). Frozen for Phase 8.5.
ActionType = Literal[
    "research_task",
    "verification_task",
    "evidence_gathering",
    "gap_declaration",
    "edit_proposal",
    "substrate_observation",
    "escalation",
]

# Lifecycle status vocabulary (P0.40). Mirrors §4.5's metacognitive
# decisions (promote/defer/drop) + a terminal 'acted' state for
# items whose action has been executed.
ActionItemStatus = Literal[
    "pending", "promoted", "deferred", "dropped", "acted"
]


class ActionItem(BaseModel):
    id: str

    # Hard FK to critiques.id per P0.35.
    critique_id: str
    # Denormalized customer_id for per-customer queries without
    # joining through critiques.
    customer_id: str

    action_type: ActionType

    # Type-specific payload. Per-type shape examples are in the
    # ActionItemRow docstring; the Pydantic model leaves the per-type
    # validation to consumers (Phase 9 / 9.5 / 10).
    content: dict[str, Any] = Field(default_factory=dict)

    # 0.0..1.0. Nullable because not every critic produces a
    # calibrated confidence per item.
    critic_confidence: Optional[float] = None

    status: ActionItemStatus = "pending"

    # Populated on the transition pending → {promoted, deferred,
    # dropped}.
    metacog_decision_at: Optional[datetime] = None

    # Populated on transition to status='acted'. Soft pointer to the
    # produced artifact (cognition_task / knowledge_gap /
    # crystal_edit / etc); the right target table is resolved by
    # inspecting `action_type` at read time.
    acted_artifact_id: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
