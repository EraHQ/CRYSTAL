"""SourceSchema — a JSON shape's reviewed mapping spec (Gate G, C5).

The registry entry behind "one human judgment per shape of data,
ever": first contact with a new JSON structure produces a proposed
mapping; approval makes every future record of that shape mechanical.
G-Q2=A: `status` is itself the review queue.

Mirrors `SourceSchemaRow` in `infrastructure/schema.py` 1:1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


class SourceSchema(BaseModel):
    """A (customer, schema_hash) mapping spec with review lifecycle."""

    id: str
    customer_id: str
    # sha256 over sorted key-paths + JSON types (schema_hash.py).
    schema_hash: str
    # The inference call's output: key-paths -> fact key/value roles,
    # locator/timestamp/speaker feeds, skips. Mechanically executable
    # per record; zero LLM after inference.
    mapping: dict[str, Any] = Field(default_factory=dict)
    # proposed -> approved | rejected (the review queue, G-Q2=A).
    status: str = "proposed"
    # Sample records for the proposal preview.
    sample: list[Any] = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
