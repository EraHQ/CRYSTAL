"""Critique synthesis policy — Phase 10A v1 algorithm per P0.74.

Per `docs/MCR_FRAMEWORK.md` §4.5 and §6: the metacognitive layer's
review of a trace's critiques produces a synthesis recording which
action items got promoted, deferred, or dropped, and why.

Phase 10A v1 promotion rules (first-match wins):

  1. `action_type == "substrate_observation"` → ALWAYS defer
     (Principle 9 + §9 — never auto-act on substrate).
  2. `alignment_class == "same_action"` AND >= 2 critics proposed →
     promote, rationale "both critics agreed".
  3. `alignment_class == "contradictory_action"` → defer,
     rationale "critics contradicted".
  4. `alignment_class == "similar_action"` → defer,
     rationale "critics similar but not identical".
  5. `alignment_class == "divergent_action"`:
     - critic_role == "agent_self" → promote, rationale
       "agent_self solo proposal" (Phase 9.5 shadow is sampled,
       so divergent agent_self is common and shouldn't always
       defer).
     - critic_role == "shadow" → defer, rationale "shadow solo
       proposal — agent didn't self-flag".
  6. Default → defer (catch-all).

Phase 12 (CU-28 / P0.112) adds the drop-on-low-trust-critic layer
on top of these rules: any item that WOULD defer (rules 3, 4,
5-shadow, 5-unknown, 6) is instead DROPPED when its proposing
critic is low-trust (promotion rate <10% across 20+ samples).
Promotions (rules 2, 5-agent_self) are NOT affected — a strong
alignment signal wins regardless of critic track record.
Substrate observations (rule 1) are EXEMPT — they always defer
for human review and are never auto-dropped (Principle 9 + §9).

The drop layer is OPT-IN via the `calibrations_by_critic` argument:
when omitted (None) the function reproduces the pre-Phase-12 v1
behavior exactly (no drops). The engine passes calibrations read
from the `critic_calibrations` table (populated by Phase 10B);
pure-function callers and tests can omit it.

Feedback dynamic (honest disclosure): `total_proposals` includes
dropped items, so dropping lowers a critic's promotion rate
further, making recovery slow — a critic that climbs back above
10% must accumulate enough NEW promotions to outweigh its
cumulative drops. Promotions still happen via alignment rules
regardless of trust, so recovery is possible, just gradual. A
future windowed/decayed rate would recover faster; tracked as a
follow-on idea, not implemented in Phase 12.

Idempotency: action items already non-pending (promoted/deferred/
dropped/acted from a prior synthesis) are SKIPPED at the engine
level; this module's `synthesize_for_trace` only operates on the
unique pending action items passed in.

This module's `synthesize_for_trace` is a PURE FUNCTION returning
the decision tuples. The engine wraps it with persistence + status
transitions.
"""
from __future__ import annotations

from typing import Iterable, Optional

from ..models.action_item import ActionItem
from ..models.critic_calibration import CriticCalibration
from ..models.critique import Critique
from ..models.item_alignment import AlignmentClass, ItemAlignment


# Rationale strings — frozen so test assertions can match exactly.
RATIONALE_BOTH_CRITICS_AGREED = "both critics agreed"
RATIONALE_CRITICS_CONTRADICTED = "critics contradicted"
RATIONALE_SIMILAR_NOT_IDENTICAL = "critics similar but not identical"
RATIONALE_AGENT_SELF_SOLO = "agent_self solo proposal"
RATIONALE_SHADOW_SOLO = "shadow solo proposal — agent didn't self-flag"
RATIONALE_SUBSTRATE_DEFERRED = "substrate observation — never auto-promoted"
RATIONALE_DEFAULT_DEFER = "default defer"
RATIONALE_DROPPED_LOW_TRUST_CRITIC = "critic promotion rate <10% across 20+ samples"

# Low-trust thresholds (CU-28 / P0.112). A critic is low-trust when it
# has enough decided proposals to judge (sample size) AND its
# cumulative promotion rate is below the floor. Both must hold.
LOW_TRUST_MIN_SAMPLES = 20
LOW_TRUST_MAX_PROMOTION_RATE = 0.10


def _item_alignment_class(
    item_id: str,
    alignments_by_focus_id: dict[str, ItemAlignment],
) -> Optional[AlignmentClass]:
    """Look up an item's alignment_class. Returns None if not computed."""
    alignment = alignments_by_focus_id.get(item_id)
    if alignment is None:
        return None
    return alignment.alignment_class


def _is_low_trust_critic(
    critique: Optional[Critique],
    calibrations_by_critic: Optional[
        dict[tuple[str, str], CriticCalibration]
    ],
) -> bool:
    """Return True when the critique's critic identity is low-trust.

    Low-trust requires BOTH: enough decided proposals to judge
    (`total_proposals >= LOW_TRUST_MIN_SAMPLES`) AND a cumulative
    promotion rate below `LOW_TRUST_MAX_PROMOTION_RATE`.

    Returns False (benefit of the doubt → defer, not drop) when:
      - no calibrations dict was supplied (pure-function / pre-Phase-12
        callers),
      - the critique is None (unresolvable critic identity),
      - no calibration row exists for this critic yet (cold start,
        §11 Q6),
      - the critic has fewer than the minimum sample size.

    The `total_proposals >= LOW_TRUST_MIN_SAMPLES` guard (>= 20)
    also rules out any division-by-zero on the promotion-rate
    computation.
    """
    if not calibrations_by_critic:
        return False
    if critique is None:
        return False
    calib = calibrations_by_critic.get(
        (critique.critic_role, critique.critic_model)
    )
    if calib is None:
        return False
    if calib.total_proposals < LOW_TRUST_MIN_SAMPLES:
        return False
    promotion_rate = calib.promoted_count / calib.total_proposals
    return promotion_rate < LOW_TRUST_MAX_PROMOTION_RATE


def _route_defer(
    item: ActionItem,
    critique: Optional[Critique],
    defer_rationale: str,
    calibrations_by_critic: Optional[
        dict[tuple[str, str], CriticCalibration]
    ],
    deferred: list[str],
    dropped: list[str],
    rationales: dict[str, str],
) -> None:
    """Append an item to `dropped` if its critic is low-trust, else `deferred`.

    The single choke point for the CU-28 drop-on-low-trust rule:
    every would-be-defer decision (except substrate, which is
    handled inline and exempt) routes through here. Mutates the
    passed-in lists/dict in place.
    """
    if _is_low_trust_critic(critique, calibrations_by_critic):
        dropped.append(item.id)
        rationales[item.id] = RATIONALE_DROPPED_LOW_TRUST_CRITIC
    else:
        deferred.append(item.id)
        rationales[item.id] = defer_rationale


def _is_2plus_critic_same(
    item: ActionItem,
    alignments_by_focus_id: dict[str, ItemAlignment],
) -> bool:
    """For a same_action item, verify at least one paired item exists.

    A truly two-critic-agreed item has alignment_class == "same_action"
    AND at least one entry in `paired_item_ids` (the other critic's
    matching item). Without paired items the item is solo and falls
    through to the divergent rules.
    """
    alignment = alignments_by_focus_id.get(item.id)
    if alignment is None:
        return False
    if alignment.alignment_class != "same_action":
        return False
    return len(alignment.paired_item_ids) >= 1


def synthesize_for_trace(
    pending_items: Iterable[ActionItem],
    critiques_by_id: dict[str, Critique],
    alignments_by_focus_id: dict[str, ItemAlignment],
    calibrations_by_critic: Optional[
        dict[tuple[str, str], CriticCalibration]
    ] = None,
) -> tuple[list[str], list[str], list[str], dict[str, str]]:
    """Apply Phase 10A v1 promotion rules to a trace's pending items.

    Args:
        pending_items: action items WITH status='pending' to decide on.
            The caller (engine.py) filters out already-decided items.
        critiques_by_id: map of critique_id → Critique, for resolving
            each item's critic_role.
        alignments_by_focus_id: map of action_item.id → ItemAlignment,
            for the v1 alignment classification.
        calibrations_by_critic: optional map of (critic_role,
            critic_model) → CriticCalibration, enabling the Phase 12
            (CU-28) drop-on-low-trust rule. When None (the default),
            no item is dropped and behavior matches the pre-Phase-12
            v1 exactly.

    Returns:
        (promoted_ids, deferred_ids, dropped_ids, rationales)
        where rationales maps action_item.id → short string.
        dropped_ids is empty unless `calibrations_by_critic` flags a
        proposing critic as low-trust.
    """
    promoted: list[str] = []
    deferred: list[str] = []
    dropped: list[str] = []
    rationales: dict[str, str] = {}

    for item in pending_items:
        critique = critiques_by_id.get(item.critique_id)

        # Rule 1: substrate_observation always defers. EXEMPT from the
        # low-trust drop rule — substrate observations are never
        # auto-dropped; they always await human review (Principle 9 + §9).
        if item.action_type == "substrate_observation":
            deferred.append(item.id)
            rationales[item.id] = RATIONALE_SUBSTRATE_DEFERRED
            continue

        align_class = _item_alignment_class(item.id, alignments_by_focus_id)

        # Rule 2: same_action with >=2 critics → promote. (Unaffected by
        # CU-28: a two-critic agreement wins regardless of critic trust.)
        if align_class == "same_action" and _is_2plus_critic_same(
            item, alignments_by_focus_id
        ):
            promoted.append(item.id)
            rationales[item.id] = RATIONALE_BOTH_CRITICS_AGREED
            continue

        # Rule 3: contradictory → defer (or drop if low-trust critic).
        if align_class == "contradictory_action":
            _route_defer(
                item, critique, RATIONALE_CRITICS_CONTRADICTED,
                calibrations_by_critic, deferred, dropped, rationales,
            )
            continue

        # Rule 4: similar → defer (or drop if low-trust critic).
        if align_class == "similar_action":
            _route_defer(
                item, critique, RATIONALE_SIMILAR_NOT_IDENTICAL,
                calibrations_by_critic, deferred, dropped, rationales,
            )
            continue

        # Rule 5: divergent (or no alignment) → depends on critic_role.
        # `align_class is None` falls through here too (treated as solo).
        critic_role = critique.critic_role if critique is not None else None
        if align_class == "divergent_action" or align_class is None:
            if critic_role == "agent_self":
                # Promote (unaffected by CU-28): the agent's own
                # solo reasoning is not dropped on aggregate trust.
                promoted.append(item.id)
                rationales[item.id] = RATIONALE_AGENT_SELF_SOLO
                continue
            if critic_role == "shadow":
                _route_defer(
                    item, critique, RATIONALE_SHADOW_SOLO,
                    calibrations_by_critic, deferred, dropped, rationales,
                )
                continue
            # Unknown role: defer (or drop) with catch-all rationale.
            _route_defer(
                item, critique, RATIONALE_DEFAULT_DEFER,
                calibrations_by_critic, deferred, dropped, rationales,
            )
            continue

        # Rule 6: anything else (shouldn't be reachable given the
        # AlignmentClass Literal, but defensive) → defer (or drop).
        _route_defer(
            item, critique, RATIONALE_DEFAULT_DEFER,
            calibrations_by_critic, deferred, dropped, rationales,
        )

    return promoted, deferred, dropped, rationales
