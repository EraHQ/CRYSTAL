"""CriticCalibration — MCR artifact 6 per `docs/MCR_FRAMEWORK.md` §7.

Running estimates per critic identity per customer. One row per
(customer_id, critic_role, critic_model). Written by Phase 10B's
`update_calibrations_from_synthesis` after each synthesis row is
persisted; mirrors `CriticCalibrationRow` 1:1.

Phase 10B does NOT use these counters in the promotion decision —
they accumulate for future use (Phase 11+ may add drop-on-low-trust
logic). Cold-start (§11 Q6) per P0.81 = "row doesn't exist": the
first synthesis touching a (customer, role, model) creates it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class CriticCalibration(BaseModel):
    id: str
    customer_id: str

    # Critic identity (P0.36 pattern from Phase 8.5 critiques).
    critic_role: str
    critic_model: str

    # Running counters. total_proposals == promoted + deferred + dropped
    # (eventually-consistent: Phase 10B v1 incrementing keeps the
    # invariant, but no DB check enforces it).
    total_proposals: int = 0
    promoted_count: int = 0
    deferred_count: int = 0
    dropped_count: int = 0

    last_synthesis_at: Optional[datetime] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
