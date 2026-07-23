"""Entity — a person or organization with a dedicated, curated crystal.

Entities layer (design gate 2026-07-22, Q1–Q5 in SESSION_HANDOFF 0c).
The one design rule: entities differ from regular crystals ONLY in
resolution — a mention must resolve DETERMINISTICALLY (registry name or
alias, never vector similarity) to its crystal, because referencing the
wrong person is a category error, not a ranking miss. Everything else
(vectors, tiers, scans, ledger, curation) is ordinary crystal machinery.

The operator's own entity row (operator_id set) is the primary case:
the agent's memory of WHO it's talking to. Mentioned coworkers, clients,
and orgs are the general case, lazily created by entity_memory_write.

Mirrors `EntityRow` in `infrastructure/schema.py` 1:1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class Entity(BaseModel):
    """A deterministically-resolvable person/org with a dedicated crystal."""

    id: str
    customer_id: str
    # String-backed kind per codebase convention (future kinds land
    # without breaking rows): 'person' | 'org' | ...
    kind: str = "person"
    display_name: str
    # Exact-match alternates ("Maria" for "Maria Lopez"). Word-boundary
    # case-insensitive matching ONLY — fuzzy merging is explicitly out
    # of scope (idle-scan board).
    aliases: list[str] = Field(default_factory=list)
    # The ONE pointer mechanism (Q5A superseding Q2): the entity's
    # dedicated crystal. NULL until first write/ensure — creation is
    # lazy so read paths stay side-effect free.
    crystal_id: Optional[str] = None
    # Link to F1's operator row when this entity IS an operator (the
    # "who am I talking to" case). NULL for mentioned third parties.
    operator_id: Optional[str] = None
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
