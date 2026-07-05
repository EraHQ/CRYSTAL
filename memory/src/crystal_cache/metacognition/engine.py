"""Metacognitive layer engine — Phase 10A entry point per P0.70.

`compute_alignment_and_synthesis_for_trace(store, trace_id)` is the
per-trace entry point. The Phase 10C metacognition worker
(workers/metacognition.py, Pass 2) calls it on a cadence for settled
traces; tests and manual runs invoke it directly.

Steps (per P0.74 idempotency + the writer-pattern from mcr_emitter):
  1. Load the trace's critiques via store.list_critiques_for_trace.
  2. Load each critique's action items via
     store.list_action_items_for_critique.
  3. For each pending action item, compute its alignment_class
     against items from OTHER critics (cross-critic pairing).
  4. Persist one ItemAlignment row per pending action item via
     store.create_item_alignment.
  5. Apply the v1 synthesis policy to assign each pending item to
     promoted/deferred/dropped.
  6. Persist a CritiqueSynthesis row via
     store.create_critique_synthesis.
  7. Transition each pending item via store.update_action_item_status
     (pending → promoted | deferred).

NEVER raises — failures inside any single step log a warning and
return with partial state in the result dict. Inherits mcr_emitter's
P0.44 discipline because the metacognitive layer is an after-the-fact
review pass; if it blows up, the agent's response was already
delivered.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..models.action_item import ActionItem
from ..models.critic_calibration import CriticCalibration
from ..models.critique import Critique
from .alignment import classify_pair
from .calibration import update_calibrations_from_synthesis
from .synthesis import synthesize_for_trace

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


async def _load_critiques_and_items(
    store: "MetadataStore",
    trace_id: str,
) -> tuple[list[Critique], list[ActionItem]]:
    """Load critiques for the trace + all their action items.

    Returns (critiques, all_items). Both lists may be empty when the
    trace has no critiques or no items yet.
    """
    critiques = await store.list_critiques_for_trace(trace_id)
    all_items: list[ActionItem] = []
    for c in critiques:
        items = await store.list_action_items_for_critique(c.id)
        all_items.extend(items)
    return critiques, all_items


def _compute_alignment_for_item(
    focus_item: ActionItem,
    focus_critique_id: str,
    all_items: list[ActionItem],
    items_by_critique: dict[str, list[ActionItem]],
) -> tuple[str, list[str]]:
    """Compute (alignment_class, paired_item_ids) for one focus item.

    Walks every item from OTHER critiques (i.e. items whose critique_id
    differs from `focus_critique_id`) and classifies the pair. The
    aggregate decision rules:

      - If ANY pairing produced "contradictory_action" →
        contradictory_action (strongest signal).
      - Else if ANY pairing produced "same_action" → same_action.
      - Else if ANY pairing produced "similar_action" → similar_action.
      - Else → divergent_action.

    `paired_item_ids` collects items from other critiques that
    matched OR contradicted (i.e. produced same/similar/contradictory).
    Empty for solo divergent_action.

    This aggregation is part of Phase 10A v1; Phase 10B may use
    weighted-by-critic-confidence aggregation instead.
    """
    classes_seen: list[tuple[str, str]] = []  # (class, paired_item_id)

    for other_critique_id, other_items in items_by_critique.items():
        if other_critique_id == focus_critique_id:
            continue
        for other_item in other_items:
            cls = classify_pair(focus_item, other_item)
            classes_seen.append((cls, other_item.id))

    if not classes_seen:
        return "divergent_action", []

    # Aggregation: contradictory > same > similar > divergent.
    paired_ids: list[str] = []
    has_contradictory = False
    has_same = False
    has_similar = False

    for cls, other_id in classes_seen:
        if cls == "contradictory_action":
            has_contradictory = True
            if other_id not in paired_ids:
                paired_ids.append(other_id)
        elif cls == "same_action":
            has_same = True
            if other_id not in paired_ids:
                paired_ids.append(other_id)
        elif cls == "similar_action":
            has_similar = True
            if other_id not in paired_ids:
                paired_ids.append(other_id)
        # divergent_action contributes no paired_ids.

    if has_contradictory:
        return "contradictory_action", paired_ids
    if has_same:
        return "same_action", paired_ids
    if has_similar:
        return "similar_action", paired_ids
    return "divergent_action", []


async def compute_alignment_and_synthesis_for_trace(
    store: "MetadataStore",
    trace_id: str,
) -> dict[str, Any]:
    """Compute alignments + synthesis for one persisted trace.

    The Phase 10A manual/triggered entry point. Returns a result dict:

        {
            "trace_id": str,
            "alignment_ids": list[str],
            "synthesis_id": Optional[str],
            "promoted_count": int,
            "deferred_count": int,
            "dropped_count": int,
            "skipped_already_decided": int,
            "reason": str,
        }

    Idempotency (P0.74): action items whose status is already
    non-pending are SKIPPED. A second call against the same trace
    will produce a synthesis row with empty buckets (or only the new
    items added since the last synthesis) but no double-processing.

    NEVER raises. Per-step failures log warnings and surface in the
    `reason` field.
    """
    out: dict[str, Any] = {
        "trace_id": trace_id,
        "alignment_ids": [],
        "synthesis_id": None,
        "promoted_count": 0,
        "deferred_count": 0,
        "dropped_count": 0,
        "skipped_already_decided": 0,
        "reason": "",
    }

    # --- 1. Load trace metadata (need customer_id for writes). -----
    try:
        trace = await store.get_reasoning_trace(trace_id)
    except Exception as e:
        logger.warning(
            "metacognition.trace_load_failed",
            trace_id=trace_id,
            error=str(e),
        )
        out["reason"] = "trace_load_failed"
        return out

    if trace is None:
        out["reason"] = "trace_not_found"
        return out

    customer_id = trace.customer_id

    # --- 2. Load critiques + items. --------------------------------
    try:
        critiques, all_items = await _load_critiques_and_items(
            store, trace_id
        )
    except Exception as e:
        logger.warning(
            "metacognition.load_failed",
            trace_id=trace_id,
            error=str(e),
        )
        out["reason"] = "critique_load_failed"
        return out

    if not critiques or not all_items:
        out["reason"] = "no_critiques_or_items"
        return out

    # Group items by critique for the alignment pairing step.
    items_by_critique: dict[str, list[ActionItem]] = {}
    for item in all_items:
        items_by_critique.setdefault(item.critique_id, []).append(item)

    # Index critiques + alignments for the synthesis step.
    critiques_by_id: dict[str, Critique] = {c.id: c for c in critiques}

    # Idempotency split: only process pending items. Already-decided
    # items count toward skipped_already_decided.
    pending_items: list[ActionItem] = []
    for item in all_items:
        if item.status == "pending":
            pending_items.append(item)
        else:
            out["skipped_already_decided"] += 1

    if not pending_items:
        out["reason"] = "all_items_already_decided"
        # Still write a synthesis row so the audit trail shows the
        # re-review happened. Empty buckets, empty rationales.
        try:
            synth = await store.create_critique_synthesis(
                customer_id=customer_id,
                trace_id=trace_id,
                review_window_start=trace.created_at,
                review_window_end=None,
                promoted_item_ids=[],
                deferred_item_ids=[],
                dropped_item_ids=[],
                promotion_rationales={},
            )
            out["synthesis_id"] = synth.id
        except Exception as e:
            logger.warning(
                "metacognition.empty_synthesis_persist_failed",
                trace_id=trace_id,
                error=str(e),
            )
        return out

    # --- 3. Compute alignment for each pending focus item. ---------
    # Build alignments_by_focus_id as we write rows so the synthesis
    # step can read them in-memory rather than round-tripping the DB.
    alignments_by_focus_id = {}
    persisted_alignment_ids: list[str] = []

    for focus_item in pending_items:
        align_class, paired_ids = _compute_alignment_for_item(
            focus_item=focus_item,
            focus_critique_id=focus_item.critique_id,
            all_items=all_items,
            items_by_critique=items_by_critique,
        )
        try:
            alignment = await store.create_item_alignment(
                customer_id=customer_id,
                focus_item_id=focus_item.id,
                alignment_class=align_class,
                trace_id=trace_id,
                paired_item_ids=paired_ids,
                confidence=1.0,  # Phase 10A: deterministic rule-based
            )
            alignments_by_focus_id[focus_item.id] = alignment
            persisted_alignment_ids.append(alignment.id)
        except Exception as e:
            logger.warning(
                "metacognition.alignment_persist_failed",
                trace_id=trace_id,
                focus_item_id=focus_item.id,
                error=str(e),
            )
            continue

    out["alignment_ids"] = persisted_alignment_ids

    # --- 3.5. Read critic calibrations for the CU-28 drop rule. -----
    # (Phase 12 / P0.112.) The drop-on-low-trust decision needs each
    # proposing critic's cumulative track record. Read BEFORE synthesis
    # so the decision reflects history through trace N-1; step 5.5 then
    # folds in trace N's own decisions. Missing/cold-start calibrations
    # are simply absent from the map → synthesize_for_trace treats the
    # critic as not-low-trust (benefit of the doubt). Read failures log
    # and skip that critic — the synthesis still runs, just without the
    # drop signal for the unreadable identity.
    calibrations_by_critic: dict[tuple[str, str], CriticCalibration] = {}
    distinct_critics = {(c.critic_role, c.critic_model) for c in critiques}
    for role, model in distinct_critics:
        try:
            calib = await store.get_critic_calibration(
                customer_id, role, model
            )
        except Exception as e:
            logger.warning(
                "metacognition.calibration_read_failed",
                trace_id=trace_id,
                critic_role=role,
                critic_model=model,
                error=str(e),
            )
            continue
        if calib is not None:
            calibrations_by_critic[(role, model)] = calib

    # --- 4. Apply synthesis policy (with CU-28 low-trust drop). -----
    promoted_ids, deferred_ids, dropped_ids, rationales = synthesize_for_trace(
        pending_items=pending_items,
        critiques_by_id=critiques_by_id,
        alignments_by_focus_id=alignments_by_focus_id,
        calibrations_by_critic=calibrations_by_critic,
    )

    out["promoted_count"] = len(promoted_ids)
    out["deferred_count"] = len(deferred_ids)
    out["dropped_count"] = len(dropped_ids)

    # --- 5. Persist the synthesis row. -----------------------------
    try:
        synth = await store.create_critique_synthesis(
            customer_id=customer_id,
            trace_id=trace_id,
            review_window_start=trace.created_at,
            review_window_end=None,
            promoted_item_ids=promoted_ids,
            deferred_item_ids=deferred_ids,
            dropped_item_ids=dropped_ids,
            promotion_rationales=rationales,
        )
        out["synthesis_id"] = synth.id
    except Exception as e:
        logger.warning(
            "metacognition.synthesis_persist_failed",
            trace_id=trace_id,
            error=str(e),
        )
        out["reason"] = "synthesis_persist_failed"
        return out

    # --- 5.5. Update critic calibrations (Phase 10B — P0.79). -------
    # Forward-compatible bookkeeping: not used by promotion decision
    # in Phase 10B. Failures here log but do not block status
    # transitions — calibration is secondary to the action-item
    # state machine.
    try:
        calib_result = await update_calibrations_from_synthesis(
            store=store,
            customer_id=customer_id,
            pending_items=pending_items,
            critiques_by_id=critiques_by_id,
            promoted_ids=promoted_ids,
            deferred_ids=deferred_ids,
            dropped_ids=dropped_ids,
        )
        out["calibration_critics_updated"] = calib_result["critics_updated"]
    except Exception as e:
        logger.warning(
            "metacognition.calibration_update_failed",
            trace_id=trace_id,
            error=str(e),
        )
        out["calibration_critics_updated"] = 0

    # --- 6. Transition action item statuses. -----------------------
    # update_action_item_status defaults metacog_decision_at to now
    # when the status is not 'pending'.
    for item_id in promoted_ids:
        try:
            await store.update_action_item_status(item_id, "promoted")
        except Exception as e:
            logger.warning(
                "metacognition.status_promoted_failed",
                action_item_id=item_id,
                error=str(e),
            )

    for item_id in deferred_ids:
        try:
            await store.update_action_item_status(item_id, "deferred")
        except Exception as e:
            logger.warning(
                "metacognition.status_deferred_failed",
                action_item_id=item_id,
                error=str(e),
            )

    # Phase 12 (CU-28) can now produce dropped items when a proposing
    # critic is low-trust; the transition below was plumbed in Phase
    # 10A for forward compatibility and is now live.
    for item_id in dropped_ids:
        try:
            await store.update_action_item_status(item_id, "dropped")
        except Exception as e:
            logger.warning(
                "metacognition.status_dropped_failed",
                action_item_id=item_id,
                error=str(e),
            )

    out["reason"] = "synthesized"

    logger.info(
        "metacognition.synthesis_complete",
        trace_id=trace_id,
        customer_id=customer_id,
        alignment_count=len(persisted_alignment_ids),
        promoted=len(promoted_ids),
        deferred=len(deferred_ids),
        dropped=len(dropped_ids),
        skipped=out["skipped_already_decided"],
    )

    return out
