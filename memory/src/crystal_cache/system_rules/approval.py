"""The 'approval' rule type — approves an assumption once the bank agrees
with it (Q99 C + the first approval rule, 2026-10-23).

Anthony: "the core idea was to be able to control how automated you
wanted the system to be and create approval rules. The first rule
option can be a fact needs to have enough consistent_with edges to auto
approve." A consequence (an assumption of kind "consequence") carries
one typed edge per fact it was judged against; this rule counts the
consistent_with edges and approves when enough of them agree and
nothing contradicts.

Shape (STRICT — unknown keys reject):

  selector:
    kind                 str   "consequence" (today) | "assumption" | "any"  [required]

  conditions: ALL must hold
    min_consistent_edges   int  >= 1                                 [required]
    no_contradicted_edges  bool (default true: one contradicted_by edge
                                 or an open conflict blocks approval)  [optional]

  action:
    approve              bool  (must be true)                        [required]

Safety: an approval rule only ever approves (clears the gate and, for a
consequence, takes the parent's standing); it never invalidates, never
touches blacklist, and every fire is audited on the rule and on the
curation feed ("Approved by rule <name>"). Absent any rule nothing is
approved by the system; a person approves.
"""
from __future__ import annotations

import structlog

from .model import (
    RuleValidationError,
    SystemRule,
    require_keys,
    require_type,
)
from .registry import register_rule_type

logger = structlog.get_logger(__name__)

RULE_TYPE = "approval"
_SELECTOR_KEYS = {"kind"}
_CONDITION_KEYS = {"min_consistent_edges", "no_contradicted_edges"}
_ACTION_KEYS = {"approve"}
_KINDS = {"consequence", "assumption", "any"}


def validate(rule: SystemRule) -> None:
    sel, cond, act = rule.selector, rule.conditions, rule.action

    require_keys(sel, _SELECTOR_KEYS, where="approval.selector")
    if "kind" not in sel:
        raise RuleValidationError(
            "approval.selector: 'kind' is required (consequence | assumption | any)"
        )
    require_type(sel["kind"], str, key="kind", where="approval.selector")
    if sel["kind"] not in _KINDS:
        raise RuleValidationError(
            f"approval.selector: kind must be one of {sorted(_KINDS)}; got {sel['kind']!r}"
        )

    require_keys(cond, _CONDITION_KEYS, where="approval.conditions")
    if "min_consistent_edges" not in cond:
        raise RuleValidationError(
            "approval.conditions: 'min_consistent_edges' is required (how many "
            "facts must agree before the system approves)"
        )
    require_type(cond["min_consistent_edges"], int, key="min_consistent_edges",
                 where="approval.conditions")
    if cond["min_consistent_edges"] < 1:
        raise RuleValidationError(
            "approval.conditions: min_consistent_edges must be >= 1 (zero would "
            "approve an unchecked assumption)"
        )
    if "no_contradicted_edges" in cond:
        require_type(cond["no_contradicted_edges"], bool, key="no_contradicted_edges",
                     where="approval.conditions")

    require_keys(act, _ACTION_KEYS, where="approval.action")
    if act.get("approve") is not True:
        raise RuleValidationError(
            "approval.action: approve must be true (an approval rule exists to approve)"
        )


def _selects(rule: SystemRule, candidate: dict) -> bool:
    """candidate is a list_assumption_crystals row (kind, state, tags)."""
    kind = rule.selector["kind"]
    if kind == "any":
        return True
    return candidate.get("kind") == kind


async def _conditions_hold(rule: SystemRule, *, store, candidate: dict) -> bool:
    cond = rule.conditions
    counts = await store.count_consequence_edges(candidate["id"])
    if counts.get("consistent_with", 0) < int(cond["min_consistent_edges"]):
        return False
    if cond.get("no_contradicted_edges", True):
        if counts.get("contradicted_by", 0) > 0:
            return False
        customer_id = candidate.get("customer_id") or rule.customer_id
        open_conflicts = await store.count_open_contradictions_for_crystal(
            customer_id, candidate["id"],
        )
        if open_conflicts > 0:
            return False
    return True


async def apply(rule: SystemRule, *, store, candidate: dict) -> bool:
    """Approve one unapproved assumption when the rule selects it and its
    conditions hold. Returns True if it acted."""
    if not _selects(rule, candidate):
        return False
    if not await _conditions_hold(rule, store=store, candidate=candidate):
        return False
    customer_id = candidate.get("customer_id") or rule.customer_id
    try:
        await store.approve_assumption_standing(
            customer_id, candidate["id"], approved_by=f"rule:{rule.id}",
        )
    except ValueError:
        return False
    subject = (candidate.get("statement") or "").strip()
    try:
        await store.record_curation_event(
            customer_id,
            event_type="assumption_approved",
            subject_id=candidate["id"],
            label=(f"Approved by rule {rule.name} - {subject}"[:256] if subject
                   else f"Approved by rule {rule.name}"),
            payload={"rule_id": rule.id, "rule": rule.name, "kind": candidate.get("kind")},
        )
    except Exception:  # noqa: BLE001 — witness never breaks the approval
        logger.debug("curation_event.emit_failed", exc_info=True)
    return True


register_rule_type(RULE_TYPE, validate, apply)
