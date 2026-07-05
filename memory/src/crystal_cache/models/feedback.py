"""Feedback \u2014 user thumbs signal on an assistant turn.

A `Feedback` row says "the user gave thumbs-{up|down} on customer C's
assistant turn at sequence S, position T." Joined back to the matching
QueryLog via (customer_id, sequence_id, turn_index) at read time.

Stage 2b of the GAIA fold-back (April 2026). The signal feeds the future
batch-distillation pipeline: thumbs-down rows enqueue for stronger-model
review, the resulting imperative rules get authored as failed_reasoning
crystals scoped to the customer.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'up' | 'down' for v0. The set may grow (e.g., 'flag', 'star') as we
# learn what signals are actually useful. The DB stores it as String(8)
# without a CHECK constraint; the Literal here is the application-side
# contract.
FeedbackSignal = Literal["up", "down"]


class Feedback(BaseModel):
    id: str
    customer_id: str

    # Identifies the assistant turn this feedback is about. The lookup
    # path is QueryLogRow WHERE customer_id = X AND sequence_id = Y AND
    # turn_index = Z \u2014 served by ix_query_logs_sequence.
    sequence_id: str
    turn_index: int

    signal: FeedbackSignal
    comment: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
