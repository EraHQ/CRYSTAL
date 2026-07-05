"""The 'promotion' rule type — clears the recall gate on gated memory.

Ships now as the first system-rules type. It answers: "when may a
recall-gated crystal (e.g. background-worker output) become usable without
a human clicking approve?" The user writes a rule; absent any matching
rule, gated memory stays gated until a human approves (safe default).

Shape (STRICT — unknown keys reject):

  selector:   which gated crystals this rule considers
    origin              str   (e.g. "background_worker")  [required]
    recall_gated        bool  (must be true to select gated ones) [optional]

  conditions: ALL must hold for the rule to fire on a candidate
    outbound_scan_passed  bool  (the high-tier review verdict) [optional]
    no_open_conflicts     bool  (reuse the conflict signal)     [optional]
    min_grounded_citations int (reuse the citation signal)      [optional]

  action:
    clear_recall_gate   bool  (must be true — the whole point)  [required]
    set_tier            str   ("quarantine" default | "neutral")[optional]

Safety: a promotion rule can ONLY loosen toward usable, and only clears the
gate; it never sets a gate, never touches blacklist, and the outbound-scan
verdict must be supplied by the sandbox review (stored on the crystal), not
inferred here. Default set_tier is 'quarantine' (stays flagged unvetted;
the tier machinery earns it up over real usage) — user-overridable to
'neutral' if they trust their scan.
"""
from __future__ import annotations

from typing import Any

from .model import (
    RuleValidationError,
    SystemRule,
    require_keys,
    require_type,
)
from .registry import register_rule_type


_SELECTOR_KEYS = {"origin", "recall_gated"}
_CONDITION_KEYS = {
    "outbound_scan_passed", "no_open_conflicts", "min_grounded_citations",
}
_ACTION_KEYS = {"clear_recall_gate", "set_tier"}
_ALLOWED_TIERS = {"quarantine", "neutral"}


def validate(rule: SystemRule) -> None:
    sel, cond, act = rule.selector, rule.conditions, rule.action

    # --- selector (strict) ---
    require_keys(sel, _SELECTOR_KEYS, where="promotion.selector")
    if "origin" not in sel:
        raise RuleValidationError(
            "promotion.selector: 'origin' is required (which memory this "
            "rule promotes, e.g. 'background_worker')"
        )
    require_type(sel["origin"], str, key="origin", where="promotion.selector")
    if "recall_gated" in sel:
        require_type(
            sel["recall_gated"], bool, key="recall_gated",
            where="promotion.selector",
        )

    # --- conditions (strict) ---
    require_keys(cond, _CONDITION_KEYS, where="promotion.conditions")
    if "outbound_scan_passed" in cond:
        require_type(
            cond["outbound_scan_passed"], bool,
            key="outbound_scan_passed", where="promotion.conditions",
        )
    if "no_open_conflicts" in cond:
        require_type(
            cond["no_open_conflicts"], bool,
            key="no_open_conflicts", where="promotion.conditions",
        )
    if "min_grounded_citations" in cond:
        require_type(
            cond["min_grounded_citations"], int,
            key="min_grounded_citations", where="promotion.conditions",
        )
        if cond["min_grounded_citations"] < 0:
            raise RuleValidationError(
                "promotion.conditions: min_grounded_citations must be >= 0"
            )

    # --- action (strict) ---
    require_keys(act, _ACTION_KEYS, where="promotion.action")
    if act.get("clear_recall_gate") is not True:
        raise RuleValidationError(
            "promotion.action: clear_recall_gate must be true (a promotion "
            "rule exists to make gated memory usable)"
        )
    if "set_tier" in act:
        require_type(act["set_tier"], str, key="set_tier",
                     where="promotion.action")
        if act["set_tier"] not in _ALLOWED_TIERS:
            raise RuleValidationError(
                f"promotion.action: set_tier must be one of "
                f"{sorted(_ALLOWED_TIERS)} (never blacklist/whitelist by "
                f"rule); got {act['set_tier']!r}"
            )


def _selects(rule: SystemRule, candidate) -> bool:
    """Does this candidate match the selector? candidate is a Crystal."""
    sel = rule.selector
    if getattr(candidate, "origin", "direct") != sel["origin"]:
        return False
    if "recall_gated" in sel:
        if bool(getattr(candidate, "recall_gated", False)) != sel["recall_gated"]:
            return False
    return True


async def _conditions_hold(rule: SystemRule, *, store, candidate) -> bool:
    """Evaluate the conditions against store signals. ALL must hold."""
    cond = rule.conditions

    # outbound_scan_passed: the sandbox review verdict, recorded as the
    # tag "outbound_scan_passed" in the crystal's diagnostic_tags (a
    # list[str]) by the background-worker path. Absent that tag, a rule
    # that requires the verdict does NOT fire (fail-safe: no verdict means
    # not-yet-reviewed, so the gate stays).
    if cond.get("outbound_scan_passed") is True:
        tags = getattr(candidate, "diagnostic_tags", None) or []
        if "outbound_scan_passed" not in tags:
            return False

    if cond.get("no_open_conflicts") is True:
        open_conflicts = await store.count_open_conflicts_for_crystal(
            candidate.customer_id, candidate.id,
        )
        if open_conflicts > 0:
            return False

    min_cites = cond.get("min_grounded_citations")
    if isinstance(min_cites, int) and min_cites > 0:
        n = await store.count_grounded_citations_for_crystal(
            candidate.customer_id, candidate.id,
        )
        if n < min_cites:
            return False

    return True


async def apply(rule: SystemRule, *, store, candidate) -> bool:
    """Apply the promotion rule to one candidate. Returns True if it acted
    (cleared the gate). Idempotent: a candidate that doesn't match or whose
    conditions don't hold is left untouched."""
    if not _selects(rule, candidate):
        return False
    if not await _conditions_hold(rule, store=store, candidate=candidate):
        return False

    # Act: clear the gate, optionally set the tier.
    cleared = await store.set_crystal_recall_gate(
        candidate.id, candidate.customer_id, gated=False,
    )
    if not cleared:
        return False
    set_tier = rule.action.get("set_tier", "quarantine")
    # Only move the tier if it would change (avoid a needless write).
    if getattr(candidate, "quality_tier", None) != set_tier:
        await store.set_crystal_quality_tier(
            candidate.id, candidate.customer_id, set_tier,
        )
    return True


register_rule_type("promotion", validate, apply)
