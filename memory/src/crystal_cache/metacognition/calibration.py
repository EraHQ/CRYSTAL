"""Critic calibration helper — Phase 10B per P0.79.

After `engine.compute_alignment_and_synthesis_for_trace` persists a
synthesis row, this module's `update_calibrations_from_synthesis` is
called to increment per-critic-identity counters on the
`critic_calibrations` table.

Phase 10B does NOT use the calibration counters in the promotion
decision (P0.74's rules from Phase 10A are unchanged). The counters
accumulate for future use: Phase 11+ may add drop-on-low-trust-critic
logic that reads them. Cold-start (§11 Q6) per P0.81 = "row doesn't
exist": the first synthesis touching a (customer, role, model)
creates it via the mixin's upsert.

The function is async and NEVER raises (per the writer-pattern
discipline). Calibration failures log warnings and surface as
incomplete updates; they do not propagate to the engine.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ..models.action_item import ActionItem
from ..models.critique import Critique

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


async def update_calibrations_from_synthesis(
    store: "MetadataStore",
    *,
    customer_id: str,
    pending_items: list[ActionItem],
    critiques_by_id: dict[str, Critique],
    promoted_ids: list[str],
    deferred_ids: list[str],
    dropped_ids: list[str],
) -> dict[str, int]:
    """Increment calibration counters for each (critic_role, critic_model)
    based on the synthesis decisions.

    Groups the pending items by their critique's (critic_role,
    critic_model), counts how many of each critic's items landed in
    promoted / deferred / dropped, then calls
    `store.upsert_critic_calibration` once per critic identity with
    those deltas.

    Args:
        store: MetadataStore with MetacognitionExtensionsMixin bound.
        customer_id: the customer owning the trace.
        pending_items: the items the synthesis decided on this run.
        critiques_by_id: map of critique.id → Critique, for resolving
            each item's (critic_role, critic_model).
        promoted_ids / deferred_ids / dropped_ids: the synthesis's
            decision buckets.

    Returns: a dict with keys 'critics_updated', 'items_attributed',
        'items_skipped' for logging / test assertions.

    NEVER raises. Per-critic upsert failures log warnings and skip;
    the function returns whatever it managed to attribute.
    """
    out = {
        "critics_updated": 0,
        "items_attributed": 0,
        "items_skipped": 0,
    }

    promoted_set = set(promoted_ids)
    deferred_set = set(deferred_ids)
    dropped_set = set(dropped_ids)

    # Group items by critic identity, count buckets per identity.
    # Key: (critic_role, critic_model). Value: dict of bucket counts.
    counters: dict[tuple[str, str], dict[str, int]] = {}

    for item in pending_items:
        critique = critiques_by_id.get(item.critique_id)
        if critique is None:
            # Defensive: item's critique not in lookup. Skip.
            out["items_skipped"] += 1
            continue

        key = (critique.critic_role, critique.critic_model)
        bucket = counters.setdefault(
            key,
            {"promoted": 0, "deferred": 0, "dropped": 0},
        )

        if item.id in promoted_set:
            bucket["promoted"] += 1
            out["items_attributed"] += 1
        elif item.id in deferred_set:
            bucket["deferred"] += 1
            out["items_attributed"] += 1
        elif item.id in dropped_set:
            bucket["dropped"] += 1
            out["items_attributed"] += 1
        else:
            # Item not in any decision bucket — shouldn't happen if
            # caller wired things correctly, but defensive.
            out["items_skipped"] += 1

    # Apply per critic identity via the upsert primitive.
    for (role, model), bucket in counters.items():
        if bucket["promoted"] + bucket["deferred"] + bucket["dropped"] == 0:
            # No deltas — skip the upsert call.
            continue
        try:
            await store.upsert_critic_calibration(
                customer_id=customer_id,
                critic_role=role,
                critic_model=model,
                promoted_delta=bucket["promoted"],
                deferred_delta=bucket["deferred"],
                dropped_delta=bucket["dropped"],
            )
            out["critics_updated"] += 1
        except Exception as e:
            logger.warning(
                "calibration.upsert_failed",
                customer_id=customer_id,
                critic_role=role,
                critic_model=model,
                error=str(e),
                error_type=type(e).__name__,
            )

    return out
