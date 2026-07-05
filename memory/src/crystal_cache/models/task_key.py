"""Task-scoped key record (Phase 3 G3, 2026-07-03).

The resolved view of a disposable task's credential: tenant binding,
budget, and lifecycle timestamps. The raw key exists only in the mint
response — at rest there is only the hash.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class TaskKey(BaseModel):
    task_id: str
    customer_id: str
    budget_micro_usd: int
    expires_at: datetime
    revoked_at: Optional[datetime] = None
    created_at: datetime
