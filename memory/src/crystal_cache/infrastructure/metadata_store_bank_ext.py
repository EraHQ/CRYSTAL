"""Bank surface store primitives — fact ledger (Q1A + Q6B, 2026-07-15).

The Crystal Bank redesign's immutability layer. Design ratified
2026-07-15: history lives in ONE immutable home — the `fact_ledger`
table — which carries the FULL before/after claim text of every
bank-surface mutation (supersede, retire). The fact row itself is then
removed through the existing, battle-tested `delete_fact` machinery
(vector rebuild from survivors, tenancy checks, index parity), so the
retrieval surface needed zero changes.

APPEND-ONLY BY CONSTRUCTION: this mixin exposes create and read
methods for the ledger and nothing else. Never add an update or a
delete. That absence IS the immutability guarantee (same pattern as
"write tools don't exist in the worker's universe").

Orchestration (ledger + add replacement + delete original) lives in
the endpoint layer, composing these primitives with the Phase 3 store
methods — SQL stays here per R9.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import select

from .schema import FactLedgerRow

logger = structlog.get_logger(__name__)


def _ledger_to_dict(row: FactLedgerRow) -> dict:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "crystal_id": row.crystal_id,
        "fact_id": row.fact_id,
        "op": row.op,
        "actor": row.actor,
        "before_prompt": row.before_prompt,
        "before_text": row.before_text,
        "after_text": row.after_text,
        "successor_fact_id": row.successor_fact_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


class BankExtensionsMixin:
    """Mixed into MetadataStore via infrastructure/__init__.py."""

    async def append_fact_ledger(
        self,
        customer_id: str,
        crystal_id: str,
        fact_id: str,
        *,
        op: str,
        actor: str = "operator",
        before_prompt: Optional[str] = None,
        before_text: Optional[str] = None,
        after_text: Optional[str] = None,
        successor_fact_id: Optional[str] = None,
    ) -> dict:
        """Append one immutable history row. The ONLY write this table
        ever receives."""
        row = FactLedgerRow(
            id=f"fl_{uuid.uuid4().hex[:16]}",
            customer_id=customer_id,
            crystal_id=crystal_id,
            fact_id=fact_id,
            op=op,
            actor=(actor or "operator")[:128],
            before_prompt=before_prompt,
            before_text=before_text,
            after_text=after_text,
            successor_fact_id=successor_fact_id,
            created_at=datetime.now(timezone.utc),
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(row)
        return _ledger_to_dict(row)

    async def list_fact_ledger_for_crystal(
        self, crystal_id: str, limit: int = 200
    ) -> list[dict]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(FactLedgerRow)
                .where(FactLedgerRow.crystal_id == crystal_id)
                .order_by(FactLedgerRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [_ledger_to_dict(r) for r in rows]

    async def list_fact_ledger_for_customer(
        self, customer_id: str, limit: int = 200
    ) -> list[dict]:
        """The bank-level Activity view (Q5A)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(FactLedgerRow)
                .where(FactLedgerRow.customer_id == customer_id)
                .order_by(FactLedgerRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [_ledger_to_dict(r) for r in rows]
