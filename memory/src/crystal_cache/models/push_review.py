"""PushReviewItem — medium-confidence LLM observations awaiting review.

V3 push/pull protocol: the LLM can emit `crystal_push` tool calls
during chat to surface knowledge it noticed but isn't certain about.
High-confidence pushes go directly to a crystal write; medium-
confidence pushes land here for human review before crystallization.
Low-confidence pushes are dropped.

The review queue is consumed by the inspector's review UI (admin
endpoints under /admin/push-queue). When an operator approves an
item, the system calls add_pair_for_customer with the (key, value)
and updates this row's status to 'approved' + crystal_id.

The confidence threshold for "medium" vs "high" lives in settings
(default 0.7 and 0.9 today); it's a deployment knob, not a per-row
value.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'pending'  — awaiting review
# 'approved' — operator approved; crystal_id set, ready to write
# 'rejected' — operator rejected; will not be crystallized
# 'auto_approved' — promoted from medium to high confidence by a
#                   later observation, written without review
# 'expired'  — sat in queue past retention window, dropped
PushReviewStatus = Literal[
    "pending", "approved", "rejected", "auto_approved", "expired"
]

# 'llm_observation' is the default for V3 push-tool-call origin.
# Other sources land here too:
#  'sync_extraction'  — drive sync extracted a candidate fact
#  'feedback_distill' — batch worker promoting thumbs-up rationale
#  'manual'           — operator authored via inspector
PushReviewSource = Literal[
    "llm_observation", "sync_extraction", "feedback_distill", "manual"
]


class PushReviewItem(BaseModel):
    id: str
    customer_id: str

    key: str    # The prompt-side of the candidate pair
    value: str  # The answer-side
    confidence: float  # 0.0..1.0; the LLM's self-reported confidence

    source: PushReviewSource = "llm_observation"
    status: PushReviewStatus = "pending"

    # Set when status='approved' or 'auto_approved'; the crystal the
    # write landed in.
    crystal_id: Optional[str] = None

    # Optional back-reference to the QueryLog that produced this push.
    source_query_id: Optional[str] = None

    reviewed_at: Optional[datetime] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
