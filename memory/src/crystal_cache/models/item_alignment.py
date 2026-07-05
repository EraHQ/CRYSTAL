"""ItemAlignment — MCR artifact 4 per `docs/MCR_FRAMEWORK.md` §4.4.

Computed (not authored) by the metacognitive layer when it reviews a
trace's critiques. One row per (trace, focus_item) per P0.71: each
action item gets exactly one alignment record summarizing its position
in the cross-critic view.

Phase 10A v1 algorithm (P0.73) is a pure-function rule-based
classifier in `metacognition/alignment.py`. The Pydantic model
enforces the closed AlignmentClass vocabulary; the algorithm is
swappable without schema changes.

Mirrors ItemAlignmentRow 1:1.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# Alignment class vocabulary (P0.40-aligned extension; locked in P0.71).
# Stored as a String column with Pydantic Literal validation.
# Adding a fifth class later is a P0-lock + code change, not a schema
# migration.
AlignmentClass = Literal[
    "same_action",
    "similar_action",
    "divergent_action",
    "contradictory_action",
]


class ItemAlignment(BaseModel):
    id: str
    customer_id: str

    # Soft pointer to reasoning_traces.id. Mirrors CritiqueRow's pattern.
    trace_id: Optional[str] = None

    # Hard FK to action_items.id — the item this alignment is FOR.
    focus_item_id: str

    # The classification. See AlignmentClass for the closed vocabulary.
    alignment_class: AlignmentClass

    # action_items.id values from OTHER critics that matched/contradicted
    # the focus item. Empty list for solo `divergent_action`.
    paired_item_ids: list[str] = Field(default_factory=list)

    # Classification confidence. Phase 10A's rule-based algorithm
    # produces deterministic results so confidence is 1.0 or NULL;
    # Phase 10B may use this for a learned classifier.
    confidence: Optional[float] = None

    computed_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
