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
    priority: int = 100,
) -> SystemRule:
    """Create + persist a rule after STRICT validation. Raises
    RuleValidationError before touching the DB if the rule is malformed."""
    rule = SystemRule(
        id=f"rule_{uuid.uuid4().hex[:16]}",
        customer_id=customer_id,
        rule_type=rule_type,
        name=name,
        selector=selector,
        conditions=conditions,
        action=action,
        enabled=enabled,
        priority=priority,
    )
    validate_rule(rule)  # strict; raises before any write

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
        stmt = stmt.order_by(SystemRuleRow.priority)
        result = await session.execute(stmt)
        return [_rule_from_row(r) for r in result.scalars().all()]


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
