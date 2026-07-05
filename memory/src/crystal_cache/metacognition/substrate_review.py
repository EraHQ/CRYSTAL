"""Substrate review surface — Phase 10.5 (D-MCR-13 V1).

Per MCR_FRAMEWORK.md §9 and §11 Q7: humans need a way to review
the agent's substrate observations — the agent's structured
complaints about the system surrounding it. Per Principle 9 and
D-MCR-15, these observations are NEVER auto-acted on. They get
recorded, deferred, and surfaced here.

What lives in this module:
  - `TraceSummary`        — slim Pydantic view of a reasoning
                            trace (no full event list)
  - `SubstrateObservationView` — composed view per substrate item
  - `list_substrate_observations(store, ...)` — the library entry
                            point that the CLI + HTTP endpoint both
                            call

The view object intentionally includes the full Critique and the
full ActionItem so operators can read the agent's observation
content + the surrounding critique summary. The trace is
deliberately slim — the full event list is noisy (hundreds of
entries possible); a future "drill into trace" surface can fetch
the full trace separately.

Phase 10.5 v1 scope was list-only. CU-30 (was PRD-6) closed 2026-07-02:
`group_substrate_observations` adds the grouping MCR §11 Q7 names — by
subsystem implicated, with frequency and a severity rollup — built ON TOP
of the lister (same composition + defensive semantics), so operators
scanning an accumulating list keep signal over noise.
"""
from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Optional

import structlog
from pydantic import BaseModel

from ..models.action_item import ActionItem
from ..models.critique import Critique

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


class TraceSummary(BaseModel):
    """Slim view of a ReasoningTrace for the substrate review surface.

    Includes the soft-join key (sequence_id, turn_index) + timing +
    customer ownership but NOT the full event list / crystals_used /
    tool_calls / inferences / borders_crossed / gaps_felt. Operators
    review the substrate review surface to see WHAT the agent
    flagged about the substrate; full trace replay belongs to a
    separate drill-down surface (Phase 10.5 v2+ or the future
    dashboard).
    """
    trace_id: str
    customer_id: str
    sequence_id: Optional[str] = None
    turn_index: Optional[int] = None
    created_at: datetime


class SubstrateObservationView(BaseModel):
    """A composed view per substrate observation, for human review.

    Each view brings together three artifacts:
      - action_item: the deferred substrate_observation item itself,
        with its content payload (the agent's structured complaint)
      - critique: the originating critique row, with its summary_text
        and observations list (gives the surrounding context: what
        was the critic looking at when this observation surfaced)
      - trace_summary: a slim view of the originating reasoning
        trace (for navigation; the full trace is heavy)

    Defensive on missing pieces: if the critique or trace can't be
    resolved (orphaned IDs, deletion, race), the corresponding
    field is None and the item still appears in the list with a
    structurally-valid view. The CLI / endpoint surface should
    display "(critique not found)" or similar when these are None.
    """
    action_item: ActionItem
    critique: Optional[Critique] = None
    trace_summary: Optional[TraceSummary] = None


async def list_substrate_observations(
    store: "MetadataStore",
    *,
    customer_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
) -> list[SubstrateObservationView]:
    """List deferred substrate observations with composed context.

    The library entry point for the substrate review surface.
    Called by:
      - The CLI tool at `cli/substrate_review.py` (Phase 10.5)
      - The HTTP endpoint at `metacognition/api.py` (Phase 10.5)
      - The future admin dashboard

    Workflow:
      1. Fetch deferred substrate_observation items via
         `store.list_substrate_action_items(...)` (P0.91).
      2. For each item, resolve the originating critique via
         `store.get_critique(item.critique_id)` (P0.90).
      3. For each critique with a trace_id, resolve the trace
         via `store.get_reasoning_trace(critique.trace_id)` and
         build a slim TraceSummary.
      4. Compose the SubstrateObservationView and append.

    Args:
        store: MetadataStore with all mixins bound.
        customer_id: scope to one customer; None = cross-tenant
            (operator wants system-wide view).
        since: only items with created_at >= since.
        limit: cap result count, default 50.

    Returns:
        list[SubstrateObservationView] ordered most-recent-first
        (inherits ordering from the mixin method).

    NEVER raises. Per-item resolution failures log warnings and
    leave the corresponding view field as None; the function
    returns whatever it could compose. This is intentional: an
    orphaned critique_id shouldn't hide all other observations.
    """
    items = await store.list_substrate_action_items(
        customer_id=customer_id,
        since=since,
        limit=limit,
    )

    views: list[SubstrateObservationView] = []
    for item in items:
        critique: Optional[Critique] = None
        trace_summary: Optional[TraceSummary] = None

        # Resolve critique.
        try:
            critique = await store.get_critique(item.critique_id)
        except Exception as e:
            logger.warning(
                "substrate_review.critique_lookup_failed",
                action_item_id=item.id,
                critique_id=item.critique_id,
                error=str(e),
            )
            critique = None

        # Resolve trace summary (only if critique resolved and has
        # a trace_id — Phase 9 trace_id is optional).
        if critique is not None and critique.trace_id is not None:
            try:
                trace = await store.get_reasoning_trace(critique.trace_id)
                if trace is not None:
                    trace_summary = TraceSummary(
                        trace_id=trace.id,
                        customer_id=trace.customer_id,
                        sequence_id=trace.sequence_id,
                        turn_index=trace.turn_index,
                        created_at=trace.created_at,
                    )
            except Exception as e:
                logger.warning(
                    "substrate_review.trace_lookup_failed",
                    action_item_id=item.id,
                    trace_id=critique.trace_id,
                    error=str(e),
                )
                trace_summary = None

        views.append(SubstrateObservationView(
            action_item=item,
            critique=critique,
            trace_summary=trace_summary,
        ))

    return views


_UNSPECIFIED = "(unspecified)"


class SubstrateGroup(BaseModel):
    """One subsystem's rollup of deferred substrate observations (CU-30).

    Frequency (`count`) + a severity histogram + the newest complaint
    text give an operator the scan-level signal; `item_ids` are the
    drill-down handles into the flat surface. Items whose payload omits
    subsystem or severity roll up under "(unspecified)" — defensive, per
    the surface's never-hide semantics.
    """

    subsystem: str
    count: int
    severities: dict[str, int]
    latest_at: datetime
    latest_complaint: str
    item_ids: list[str]


async def group_substrate_observations(
    store: "MetadataStore",
    *,
    customer_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 200,
) -> list[SubstrateGroup]:
    """Group deferred substrate observations by subsystem implicated (CU-30).

    Built ON TOP of `list_substrate_observations` — identical filtering,
    composition, and never-raise semantics; grouping happens in Python over
    the same bounded rows (no new SQL). Ordered most-frequent-first, then
    most-recent, so the loudest subsystem tops the list.
    """
    views = await list_substrate_observations(
        store, customer_id=customer_id, since=since, limit=limit,
    )

    groups: dict[str, dict] = {}
    for view in views:  # most-recent-first (inherited ordering)
        item = view.action_item
        content = item.content or {}
        subsystem = str(content.get("subsystem") or _UNSPECIFIED)
        severity = str(content.get("severity") or _UNSPECIFIED)
        complaint = str(content.get("complaint") or "")

        g = groups.get(subsystem)
        if g is None:
            g = {
                "subsystem": subsystem,
                "count": 0,
                "severities": {},
                # First-seen per subsystem IS the newest (ordering above).
                "latest_at": item.created_at,
                "latest_complaint": complaint,
                "item_ids": [],
            }
            groups[subsystem] = g
        g["count"] += 1
        g["severities"][severity] = g["severities"].get(severity, 0) + 1
        g["item_ids"].append(item.id)

    # Most-frequent first; ties broken newest-first.
    ordered = sorted(
        groups.values(),
        key=lambda g: (-g["count"], -g["latest_at"].timestamp()),
    )
    return [SubstrateGroup(**g) for g in ordered]
