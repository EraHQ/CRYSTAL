"""ReasoningTrace — the agent's structured self-report of how it
produced a response.

MCR artifact 1 per `docs/MCR_FRAMEWORK.md` §4.1. Mirrors
ReasoningTraceRow 1:1.

Phase 8.5 lands the model + schema; the writer ships in Phase 9
(agent self-trace emission) and the reader in Phase 10
(metacognitive layer). For Phase 8.5 the model is exercised only
through the MetadataStore mixin's CRUD round-trip tests.

The event schema inside `events` is deliberately loose for
Phase 8.5 (Phase 9 open Q1 — "how much structure to impose on the
agent's reasoning"). Convenience aggregates (`crystals_used`,
`tool_calls`, `inferences`, `borders_crossed`, `gaps_felt`) are
denormalized from `events` for cheap metacognitive-layer filtering;
their per-entry structure is the contract that Phase 9's design
pass solidifies.

Soft-linked to QueryLog via (customer_id, sequence_id, turn_index)
per P0.35; an optional hard `query_log_id` FK is present for
callers that have the id in hand.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ReasoningTrace(BaseModel):
    id: str
    customer_id: str

    # Soft-join key.
    sequence_id: Optional[str] = None
    turn_index: Optional[int] = None

    # Optional hard link to query_logs.
    query_log_id: Optional[str] = None

    # Structured trace events. Schema deferred to Phase 9 (Q1).
    events: list[dict[str, Any]] = Field(default_factory=list)

    # Denormalized aggregates derived from `events` for cheap
    # filtering. Per-entry shapes:
    #   crystals_used   → list[str] of crystal_ids
    #   tool_calls      → list[{tool_name, input, output, role}]
    #   inferences      → list[{claim, basis, confidence}]
    #   borders_crossed → list[{claim, agent_confidence,
    #                           flagged_by_agent: bool}]
    #   gaps_felt       → list[{want, why_needed}]
    crystals_used: list[str] = Field(default_factory=list)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    inferences: list[dict[str, Any]] = Field(default_factory=list)
    borders_crossed: list[dict[str, Any]] = Field(default_factory=list)
    gaps_felt: list[dict[str, Any]] = Field(default_factory=list)

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
