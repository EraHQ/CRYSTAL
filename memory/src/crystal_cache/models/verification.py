"""Verification — §6 of BUILD_PROPOSAL.md."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


VerificationStatus = Literal["pending", "approved", "rejected", "edited"]


class VerificationTask(BaseModel):
    id: str
    customer_id: str

    candidate_claim: str
    candidate_vector: list[float] = Field(default_factory=list)
    source: Optional[str] = None  # which doc / query this came from

    # priority = freq_of_related_queries × customer_importance
    priority: float = 0.0

    status: VerificationStatus = "pending"
    assigned_to: Optional[str] = None  # employee id / email, nullable

    resolved_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
