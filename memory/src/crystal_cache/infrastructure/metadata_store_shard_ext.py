"""Shard-ledger + vetting primitives — the Growth G4 store surface.

The append-only shard economy. Metering is CITATIONS (the G1 rail): a *cited*
crystal — not merely an injected one — accrues a shard, because citation means
the model found it load-bearing, which is what kills the key-stuffing attack.
The ledger is **never mutated**: corrections are compensating entries
(`clawback`), and `append_shard_event` is **idempotent** on
(interaction_id, crystal_id, event_type), so a replayed interaction can never
double-credit a crystal. **Shard units are integers**; balance = sum of
shards_credited; spends are negative debits in the same ledger (closed-loop —
subscription only, no cash-out).

Eligibility + weighting policy (self-traffic exclusion, marketplace-scope
gate, weight→shards) lives in the pure `marketplace/crediting.py` so it's
unit-testable without a DB; R9 keeps the SQL here. Reward-pool apportionment
is **D7 — deferred**; v1 credits a fixed integer per grounded citation as the
placeholder, and convertibility (shards offsetting subscription) stays OFF
until metering survives adversarial traffic.

`expert_authorizations` is the minimal vetting substrate: an operator
authorized to author general crystals in a `general:<domain>`. The
team→general promotion gate (F3's top rung) checks it.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func, select

from ..marketplace.crediting import (
    is_marketplace_crystal,
    is_self_traffic,
    shards_from_weight,
)
from .schema import ExpertAuthorizationRow, ShardEventRow

logger = structlog.get_logger(__name__)

# Event types.
CREDIT = "credit"
DEBIT = "debit"
CLAWBACK = "clawback"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _event_to_dict(row: ShardEventRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "owner_operator_id": row.owner_operator_id,
        "crystal_id": row.crystal_id,
        "consuming_team_id": row.consuming_team_id,
        "interaction_id": row.interaction_id,
        "event_type": row.event_type,
        "signal_type": row.signal_type,
        "raw_weight": row.raw_weight,
        "shards_credited": row.shards_credited,
        "created_at": row.created_at,
    }


def _expert_to_dict(row: ExpertAuthorizationRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "operator_id": row.operator_id,
        "team_id": row.team_id,
        "domain": row.domain,
        "status": row.status,
        "created_at": row.created_at,
    }


class ShardExtensionsMixin:
    """shard_events + expert_authorizations CRUD bound onto MetadataStore."""

    # ------------------------------------------------------------------
    # Ledger
    # ------------------------------------------------------------------

    async def append_shard_event(
        self,
        *,
        event_type: str,
        signal_type: str = "citation",
        owner_operator_id: Optional[str] = None,
        crystal_id: Optional[str] = None,
        consuming_team_id: Optional[str] = None,
        interaction_id: Optional[str] = None,
        raw_weight: float = 0.0,
        shards_credited: int = 0,
    ) -> dict[str, Any]:
        """Append one ledger event — idempotent on (interaction_id, crystal_id,
        event_type) when all three are present.

        If a matching credit/clawback already exists for that triple, the
        existing row is returned unchanged (no double-credit) — the replay
        guard. Events with NULL interaction_id or crystal_id (e.g. spends) are
        never deduped and always insert. SQLite is single-writer, so the
        select-then-insert is atomic in practice; the UNIQUE index is the
        backstop that turns a concurrent double-insert into an error rather
        than a double-credit.
        """
        # Idempotency pre-check (only when the dedup key is fully present).
        if interaction_id is not None and crystal_id is not None:
            async with self.session() as session:  # type: ignore[attr-defined]
                existing = (await session.execute(
                    select(ShardEventRow).where(
                        ShardEventRow.interaction_id == interaction_id,
                        ShardEventRow.crystal_id == crystal_id,
                        ShardEventRow.event_type == event_type,
                    ).limit(1)
                )).scalar_one_or_none()
                if existing is not None:
                    logger.info(
                        "shard_ledger.idempotent_skip",
                        interaction_id=interaction_id,
                        crystal_id=crystal_id,
                        event_type=event_type,
                    )
                    return _event_to_dict(existing)

        event_id = f"shard_{uuid.uuid4().hex[:16]}"
        async with self.session() as session:  # type: ignore[attr-defined]
            row = ShardEventRow(
                id=event_id,
                owner_operator_id=owner_operator_id,
                crystal_id=crystal_id,
                consuming_team_id=consuming_team_id,
                interaction_id=interaction_id,
                event_type=event_type,
                signal_type=signal_type,
                raw_weight=raw_weight,
                shards_credited=shards_credited,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _event_to_dict(row)

    async def shard_balance(self, owner_operator_id: str) -> int:
        """An expert's shard balance = sum of shards_credited (credits minus
        debits/clawbacks, since debits are stored negative). Integer shards."""
        async with self.session() as session:  # type: ignore[attr-defined]
            total = (await session.execute(
                select(
                    func.coalesce(func.sum(ShardEventRow.shards_credited), 0)
                ).where(ShardEventRow.owner_operator_id == owner_operator_id)
            )).scalar_one()
            return int(total)

    async def record_citation_credit(
        self,
        *,
        crystal_id: str,
        owner_operator_id: Optional[str],
        crystal_group_team_id: Optional[str],
        crystal_type: Optional[str],
        crystal_customer_id: Optional[str],
        consuming_team_id: str,
        interaction_id: str,
        raw_weight: float = 1.0,
    ) -> Optional[dict[str, Any]]:
        """Mint a shard credit for a grounded crystal citation — the closed
        loop where G1's metering rail drives G4's economy.

        Eligibility (pure policy in marketplace/crediting.py):
          - the crystal must be a marketplace (general-scoped) crystal —
            private/team citations don't earn;
          - self-traffic is excluded — a team citing its own crystal earns
            nothing (the seeder-decoupling instinct).
        When eligible, appends an idempotent `credit` event; the (interaction,
        crystal) dedup key means re-grounding the same answer never
        double-credits. Returns the event dict, or None when not eligible.

        `raw_weight` is the pre-pool usefulness weight (e.g. split across
        co-cited crystals by the caller); shards_from_weight maps it to an
        integer shard count — a fixed placeholder pending D7 (the bounded
        reward pool).
        """
        if not is_marketplace_crystal(crystal_type, crystal_customer_id):
            return None
        if is_self_traffic(crystal_group_team_id, consuming_team_id):
            return None
        shards = shards_from_weight(raw_weight)
        if shards <= 0:
            return None
        return await self.append_shard_event(
            event_type=CREDIT,
            signal_type="citation",
            owner_operator_id=owner_operator_id,
            crystal_id=crystal_id,
            consuming_team_id=consuming_team_id,
            interaction_id=interaction_id,
            raw_weight=raw_weight,
            shards_credited=shards,
        )

    async def clawback_citation(
        self,
        *,
        crystal_id: str,
        interaction_id: str,
        owner_operator_id: Optional[str],
        shards: int,
    ) -> dict[str, Any]:
        """Compensating entry that reverses a credit on a later-disproven fact.

        A negative `clawback` event; it coexists with the original `credit`
        (the idempotency key includes event_type), so the ledger stays
        append-only and the balance nets out. `shards` is the positive
        magnitude to claw back; it is stored negative.
        """
        return await self.append_shard_event(
            event_type=CLAWBACK,
            signal_type="correction",
            owner_operator_id=owner_operator_id,
            crystal_id=crystal_id,
            interaction_id=interaction_id,
            shards_credited=-abs(int(shards)),
        )

    async def spend_shards(
        self,
        owner_operator_id: str,
        shards: int,
        *,
        note: Optional[str] = None,
    ) -> dict[str, Any]:
        """Spend shards as a closed-loop debit (subscription credit only — no
        cash-out). A negative `debit` event with NULL interaction/crystal keys
        (so it is never deduped). Convertibility is gated OFF at launch; this
        is the substrate the billing integration will call once metering has
        survived adversarial traffic."""
        return await self.append_shard_event(
            event_type=DEBIT,
            signal_type="spend",
            owner_operator_id=owner_operator_id,
            shards_credited=-abs(int(shards)),
        )

    async def list_shard_events(
        self, owner_operator_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """An expert's ledger entries, newest first (the Inspector ledger
        view — role-gated financial at the HTTP boundary)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(ShardEventRow)
                .where(ShardEventRow.owner_operator_id == owner_operator_id)
                .order_by(ShardEventRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [_event_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    # Vetting (expert authorizations)
    # ------------------------------------------------------------------

    async def authorize_expert(
        self, operator_id: str, team_id: str, domain: str
    ) -> dict[str, Any]:
        """Authorize (or re-activate) an operator to author general crystals
        in `domain` (general:<domain>). Idempotent on (operator_id, domain) —
        a re-authorize flips a revoked row back to active."""
        async with self.session() as session:  # type: ignore[attr-defined]
            existing = (await session.execute(
                select(ExpertAuthorizationRow).where(
                    ExpertAuthorizationRow.operator_id == operator_id,
                    ExpertAuthorizationRow.domain == domain,
                ).limit(1)
            )).scalar_one_or_none()
            if existing is not None:
                existing.status = "active"
                existing.team_id = team_id
                await session.commit()
                await session.refresh(existing)
                return _expert_to_dict(existing)
            row = ExpertAuthorizationRow(
                id=f"xauth_{uuid.uuid4().hex[:16]}",
                operator_id=operator_id,
                team_id=team_id,
                domain=domain,
                status="active",
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _expert_to_dict(row)

    async def revoke_expert(self, operator_id: str, domain: str) -> bool:
        """Revoke an operator's authorization in a domain. Returns False if no
        such authorization."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(ExpertAuthorizationRow).where(
                    ExpertAuthorizationRow.operator_id == operator_id,
                    ExpertAuthorizationRow.domain == domain,
                ).limit(1)
            )).scalar_one_or_none()
            if row is None:
                return False
            row.status = "revoked"
            await session.commit()
            return True

    async def is_expert_authorized(
        self, operator_id: str, domain: str
    ) -> bool:
        """True iff the operator holds an active authorization in `domain`."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = (await session.execute(
                select(ExpertAuthorizationRow).where(
                    ExpertAuthorizationRow.operator_id == operator_id,
                    ExpertAuthorizationRow.domain == domain,
                    ExpertAuthorizationRow.status == "active",
                ).limit(1)
            )).scalar_one_or_none()
            return row is not None

    async def list_expert_authorizations(
        self, team_id: str, *, active_only: bool = True
    ) -> list[dict[str, Any]]:
        """A team's expert authorizations, newest first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(ExpertAuthorizationRow).where(
                ExpertAuthorizationRow.team_id == team_id
            )
            if active_only:
                stmt = stmt.where(ExpertAuthorizationRow.status == "active")
            stmt = stmt.order_by(ExpertAuthorizationRow.created_at.desc())
            rows = (await session.execute(stmt)).scalars().all()
            return [_expert_to_dict(r) for r in rows]
