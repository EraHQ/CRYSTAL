"""SourceWatch — the general watch registration (Gate M, 2026-07-18).

The model mirror of SourceWatchRow. See the row's docstring for the
design; the short version: one shape for every watched source, scheme-
dispatched, git first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SourceWatch:
    id: str
    customer_id: str
    scheme: str
    source_name: str
    config: dict = field(default_factory=dict)
    cadence_minutes: int = 15
    last_state: Optional[dict] = None
    review_mode: str = "auto"
    encrypted_token: Optional[str] = None
    status: str = "active"
    last_checked_at: Optional[datetime] = None
    last_error: Optional[str] = None
    created_at: Optional[datetime] = None
