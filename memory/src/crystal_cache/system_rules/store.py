"""Persistence + evaluation for system rules.

Kept out of metadata_store.py proper: these helpers operate ON a store
(passed in) rather than being mixed into it, so the rule layer stays a
cohesive unit. SQL still lives in metadata_store via the generic row ops
this calls — this module composes those, it doesn't hand-write SQL for
crystals. The system_rules table CRUD is here and uses the session only for
the rules table itself (R9: SQL for a table lives with that table's
owner; system_rules is owned here).
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select, update

from ..infrastructure.schema import SystemRuleRow
from .model import SystemRule
from .registry import get_rule_type, validate_rule

logger = structlog.get_logger(__name__)


def _rule_from_row(row: SystemRuleRow) -> SystemRule:
    return SystemRule(
        id=row.id,
        customer_id=row.customer_id,
        rule_type=row.rule_type,
        name=row.name,
        selector=dict(row.selector or {}),
        conditions=dict(row.conditions or {}),
        action=dict(row.action or {}),
        enabled=bool(row.enabled),
        priority=row.priority,
        last_fired_at=row.last_fired_at,
        fire_count=row.fire_count,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


async def create_rule(
    store,
    customer_id: str,
    rule_type: str,
    name: str,
    *,
    selector: dict,
    conditions: dict,
    action: dict,
    enabled: bool = True,
    priority: Optional[int] = None,
) -> SystemRule:
    """Create + persist a rule after STRICT validation. Raises
    RuleValidationError before touching the DB if the rule is malformed.
    priority None (Q101) appends the rule after the tenant's last one:
    the list is ordered and a new line goes at the bottom."""
    rule = SystemRule(
        id=f"rule_{uuid.uuid4().hex[:16]}",
        customer_id=customer_id,
        rule_type=rule_type,
        name=name,
        selector=selector,
        conditions=conditions,
        action=action,
        enabled=enabled,
        priority=priority if priority is not None else 100,
    )
    validate_rule(rule)  # strict; raises before any write

    if priority is None:
        existing = await list_rules(store, customer_id)
        rule.priority = (max((r.priority for r in existing), default=0) + 10) if existing else 10
        priority = rule.priority

    async with store.session() as session:
        session.add(SystemRuleRow(
            id=rule.id,
            customer_id=customer_id,
            rule_type=rule_type,
            enabled=enabled,
            name=name,
            selector=selector,
            conditions=conditions,
            action=action,
            priority=priority,
        ))
    return rule


async def list_rules(
    store, customer_id: str, *, rule_type: Optional[str] = None,
    enabled_only: bool = False,
) -> list[SystemRule]:
    async with store.session() as session:
        stmt = select(SystemRuleRow).where(
            SystemRuleRow.customer_id == customer_id
        )
        if rule_type is not None:
            stmt = stmt.where(SystemRuleRow.rule_type == rule_type)
        if enabled_only:
            stmt = stmt.where(SystemRuleRow.enabled.is_(True))
        stmt = stmt.order_by(SystemRuleRow.priority, SystemRuleRow.created_at, SystemRuleRow.id)
        result = await session.execute(stmt)
        return [_rule_from_row(r) for r in result.scalars().all()]


async def move_rule(
    store, customer_id: str, rule_id: str, direction: str,
) -> Optional[list[SystemRule]]:
    """Q101: move one rule up or down one line. The whole list is
    renumbered 10, 20, 30... so order is always total and a new rule can
    append. Returns the reordered list, or None when the rule is not this
    tenant's. Moving the top rule up (or the bottom one down) is a no-op."""
    if direction not in ("up", "down"):
        raise ValueError("direction must be 'up' or 'down'")
    rules = await list_rules(store, customer_id)
    idx = next((i for i, r in enumerate(rules) if r.id == rule_id), None)
    if idx is None:
        return None
    target = idx - 1 if direction == "up" else idx + 1
    if 0 <= target < len(rules):
        rules[idx], rules[target] = rules[target], rules[idx]
    async with store.session() as session:
        for n, r in enumerate(rules, start=1):
            r.priority = n * 10
            await session.execute(
                update(SystemRuleRow)
                .where(SystemRuleRow.id == r.id, SystemRuleRow.customer_id == customer_id)
                .values(priority=r.priority)
            )
    return rules


async def convert_legacy_rules(store, customer_id: str) -> int:
    """Q101: fold a tenant's 'promotion' and 'approval' rules into
    'curation' rules in place (same id, name, order, switch and audit).
    Idempotent; returns how many rows changed. Runs before any list the
    console or the engine reads, so a rule written under the old shape
    keeps working as the same sentence."""
    from .curation import from_legacy

    changed = 0
    async with store.session() as session:
        rows = (await session.execute(
            select(SystemRuleRow)
            .where(SystemRuleRow.customer_id == customer_id)
            .where(SystemRuleRow.rule_type.in_(["promotion", "approval"]))
        )).scalars().all()
        for row in rows:
            folded = from_legacy(_rule_from_row(row))
            if folded is None:
                continue
            validate_rule(folded)
            row.rule_type = folded.rule_type
            row.selector = folded.selector
            row.conditions = folded.conditions
            row.action = folded.action
            row.updated_at = datetime.now(timezone.utc)
            changed += 1
    if changed:
        logger.info("system_rules.legacy_converted", customer_id=customer_id, converted=changed)
    return changed


async def delete_rule(store, customer_id: str, rule_id: str) -> bool:
    from sqlalchemy import delete as sa_delete
    async with store.session() as session:
        stmt = sa_delete(SystemRuleRow).where(
            SystemRuleRow.id == rule_id,
            SystemRuleRow.customer_id == customer_id,
        )
        result = await session.execute(stmt)
        return bool(result.rowcount)


async def _record_fire(store, rule_id: str, n: int) -> None:
    """Audit: bump fire_count + last_fired_at after a rule acts."""
    async with store.session() as session:
        stmt = (
            update(SystemRuleRow)
            .where(SystemRuleRow.id == rule_id)
            .values(
                last_fired_at=datetime.now(timezone.utc),
                fire_count=SystemRuleRow.fire_count + n,
            )
        )
        await session.execute(stmt)


async def update_rule(
    store, customer_id: str, rule_id: str, *, enabled: Optional[bool] = None,
    name: Optional[str] = None, conditions: Optional[dict] = None,
    selector: Optional[dict] = None, action: Optional[dict] = None,
) -> Optional[SystemRule]:
    """Change a rule's enabled flag, name, selector, conditions or action.
    The changed rule is re-validated STRICTLY before the write; returns
    None when the rule is not this tenant's."""
    async with store.session() as session:
        row = await session.get(SystemRuleRow, rule_id)
        if row is None or row.customer_id != customer_id:
            return None
        rule = _rule_from_row(row)
        if enabled is not None:
            rule.enabled = bool(enabled)
        if name is not None:
            rule.name = name
        if selector is not None:
            rule.selector = dict(selector)
        if conditions is not None:
            rule.conditions = dict(conditions)
        if action is not None:
            rule.action = dict(action)
        validate_rule(rule)  # strict; raises before any write
        row.enabled = rule.enabled
        row.name = rule.name
        row.selector = rule.selector
        row.conditions = rule.conditions
        row.action = rule.action
        row.updated_at = datetime.now(timezone.utc)
        return rule


async def run_curation_rules(
    store, customer_id: str,
) -> dict[str, int]:
    """Q101: one pass of the ordered allow/deny list over everything
    waiting for a decision. Legacy rules are folded first. For each
    candidate the first matching rule decides: allow acts (approve, or let
    into recall), deny holds it for a person, no match leaves it for a
    person. Allow fires are audited on the rule; a deny is counted in the
    result only (it would match again every pass). Returns
    {allowed, held, rules_fired}. Absent any rule nothing is touched."""
    from .curation import RULE_TYPE, _attach_keys, allow, decide, list_candidates

    await convert_legacy_rules(store, customer_id)
    rules = await list_rules(store, customer_id, rule_type=RULE_TYPE, enabled_only=True)
    if not rules:
        return {"allowed": 0, "held": 0, "rules_fired": 0}
    candidates = await list_candidates(store, customer_id)
    if not candidates:
        return {"allowed": 0, "held": 0, "rules_fired": 0}
    if any("key_prefix" in r.selector for r in rules):
        await _attach_keys(store, candidates)

    allowed = held = 0
    fired: dict[str, int] = {}
    for candidate in candidates:
        try:
            rule, decision = await decide(rules, candidate, store=store)
            if rule is None:
                continue
            if decision == "deny":
                held += 1
                continue
            if await allow(rule, store=store, candidate=candidate):
                allowed += 1
                fired[rule.id] = fired.get(rule.id, 0) + 1
        except Exception:  # noqa: BLE001 - one bad candidate never aborts the pass
            logger.warning("system_rules.curation_apply_failed", customer_id=customer_id,
                           crystal_id=candidate.id)
            continue
    for rule_id, n in fired.items():
        await _record_fire(store, rule_id, n)
    _log = logger.info if allowed else logger.debug
    _log("system_rules.curation_run", customer_id=customer_id, allowed=allowed, held=held,
         rules_fired=len(fired))
    return {"allowed": allowed, "held": held, "rules_fired": len(fired)}


async def run_approval_rules(
    store, customer_id: str,
) -> dict[str, int]:
    """Evaluate a customer's enabled 'approval' rules against their
    unapproved assumptions (Q99 C: consequences today, any kind the
    selector names). Returns {approved: N, rules_fired: M}. Absent any
    rule nothing is approved by the system."""
    rules = await list_rules(
        store, customer_id, rule_type="approval", enabled_only=True,
    )
    if not rules:
        return {"approved": 0, "rules_fired": 0}
    candidates = await store.list_unapproved_assumptions(customer_id)
    if not candidates:
        return {"approved": 0, "rules_fired": 0}

    approved = 0
    rules_fired = 0
    done: set[str] = set()
    for rule in rules:  # priority-ordered
        impl = get_rule_type(rule.rule_type)
        fired_this_rule = 0
        for candidate in candidates:
            if candidate["id"] in done:
                continue
            candidate = {**candidate, "customer_id": customer_id}
            try:
                acted = await impl.apply(rule, store=store, candidate=candidate)
            except Exception:  # noqa: BLE001 — one bad candidate never aborts
                logger.warning(
                    "system_rules.approval_apply_failed",
                    rule_id=rule.id, crystal_id=candidate.get("id"),
                )
                continue
            if acted:
                approved += 1
                fired_this_rule += 1
                done.add(candidate["id"])
        if fired_this_rule:
            rules_fired += 1
            await _record_fire(store, rule.id, fired_this_rule)
    _log = logger.info if approved else logger.debug
    _log("system_rules.approval_run", customer_id=customer_id,
         approved=approved, rules_fired=rules_fired)
    return {"approved": approved, "rules_fired": rules_fired}


async def run_promotion_rules(
    store, customer_id: str,
) -> dict[str, int]:
    """Evaluate a customer's enabled 'promotion' rules against their
    recall-gated crystals, clearing gates where a rule's conditions hold.

    Safe by construction: only gated crystals are considered; absent any
    matching rule nothing is promoted (human approval remains the default);
    rules only loosen toward usable; every fire is audited. Returns
    {promoted: N, rules_fired: M}.
    """
    rules = await list_rules(
        store, customer_id, rule_type="promotion", enabled_only=True,
    )
    if not rules:
        return {"promoted": 0, "rules_fired": 0}

    gated = await store.list_recall_gated_crystals(customer_id)
    if not gated:
        return {"promoted": 0, "rules_fired": 0}

    promoted = 0
    rules_fired = 0
    impl_cache = {r.id: get_rule_type(r.rule_type) for r in rules}

    for rule in rules:  # already priority-ordered
        impl = impl_cache[rule.id]
        fired_this_rule = 0
        for candidate in gated:
            # A candidate already promoted by an earlier rule this pass is
            # no longer gated in the DB; skip to avoid a redundant write.
            try:
                acted = await impl.apply(rule, store=store, candidate=candidate)
            except Exception:  # noqa: BLE001 — one bad candidate never aborts
                logger.warning(
                    "system_rules.promotion_apply_failed",
                    rule_id=rule.id, crystal_id=getattr(candidate, "id", None),
                )
                continue
            if acted:
                promoted += 1
                fired_this_rule += 1
        if fired_this_rule:
            rules_fired += 1
            await _record_fire(store, rule.id, fired_this_rule)

    logger.info(
        "system_rules.promotion_run",
        customer_id=customer_id, promoted=promoted, rules_fired=rules_fired,
    )
    return {"promoted": promoted, "rules_fired": rules_fired}
