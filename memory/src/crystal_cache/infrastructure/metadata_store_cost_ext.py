"""Cost-accounting primitives — the Growth G3 store surface.

The single `record_llm_call()` choke point every model invocation flows
through, plus the GROUP BY aggregations the Inspector reads. The quality of
cost data equals how centralized the model calls are, so this is the one
place a cost row is born; callers (the chat proxy now; cognition / agent loop
/ depth / metacognition / inline research as they're wired) hand it tokens +
attribution and it computes + persists.

**Money is INTEGER micro-USD (1e-6 USD), never a float.** Cost is computed
from a per-model price table kept in config (prices move → externalized); the
pure math lives in `cost/pricing.py` so it's unit-testable without a DB. R9
keeps the SQL here.

Average is **per-agent** (D6): `average_cost_per_agent` divides total spend by
the count of distinct sessions. Daily / weekly buckets are computed in Python
over a time-windowed fetch (robust across SQLite / Postgres — the same
fetch-then-bucket choice the staleness sweep makes — rather than
backend-specific date functions). Per-session / per-operator rollups are
SQL GROUP BYs (portable).
"""
from __future__ import annotations

import uuid
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func, select

from ..cost.pricing import DEFAULT_PRICE_TABLE, compute_cost_micro_usd
from ..models.spend_budget import SpendBudget
from .schema import LlmCallRow, SpendBudgetRow

logger = structlog.get_logger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _aware(dt: Optional[datetime]) -> datetime:
    if dt is None:
        return _utcnow()
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _call_to_dict(row: LlmCallRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "session_id": row.session_id,
        "parent_session_id": row.parent_session_id,
        "operator_id": row.operator_id,
        "origin": row.origin,
        "model": row.model,
        "input_tokens": row.input_tokens,
        "output_tokens": row.output_tokens,
        "cache_creation_tokens": row.cache_creation_tokens,
        "cache_read_tokens": row.cache_read_tokens,
        "computed_cost_micro_usd": row.computed_cost_micro_usd,
        "created_at": row.created_at,
    }


class CostExtensionsMixin:
    """llm_calls record + aggregation bound onto MetadataStore."""

    async def record_llm_call(
        self,
        customer_id: str,
        *,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        session_id: Optional[str] = None,
        parent_session_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        origin: str = "interactive",
        billing: Optional[str] = None,
        price_table: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """Persist one model invocation's cost + attribution (G3 choke point).

        Cost is computed from `price_table` (or the built-in DEFAULT_PRICE_TABLE
        when the caller passes None — the config-driven table is threaded in by
        the endpoints/proxy) and stored as INTEGER micro-USD. An unknown model
        costs 0 micro-USD and is logged, rather than raising — observability
        must never break the call path.
        """
        table = price_table if price_table is not None else DEFAULT_PRICE_TABLE
        cost_micro = compute_cost_micro_usd(
            model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_tokens=cache_creation_tokens,
            cache_read_tokens=cache_read_tokens,
            price_table=table,
        )
        if cost_micro == 0 and model not in table:
            logger.warning("cost.unknown_model", model=model, origin=origin)

        call_id = f"llm_{uuid.uuid4().hex[:16]}"
        async with self.session() as session:  # type: ignore[attr-defined]
            row = LlmCallRow(
                id=call_id,
                customer_id=customer_id,
                billing=billing,
                session_id=session_id,
                parent_session_id=parent_session_id,
                operator_id=operator_id,
                origin=origin,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
                computed_cost_micro_usd=cost_micro,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _call_to_dict(row)

    # ------------------------------------------------------------------
    # Aggregations (the Inspector cost views)
    # ------------------------------------------------------------------

    async def managed_spend_micro_usd_this_month(self, customer_id: str) -> int:
        """Month-to-date MANAGED ledger spend for a tenant (integer
        micro-USD) — the CostReader for the E4 monthly cap at the proxy
        door (Accounts Phase B, 2026-07-06). Counts ONLY rows stamped
        billing='managed' (per-call truth; mid-month inference_mode flips
        never distort it). Month = UTC calendar month.
        """
        now = datetime.now(timezone.utc)
        month_start = now.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(
                func.coalesce(func.sum(LlmCallRow.computed_cost_micro_usd), 0)
            ).where(
                LlmCallRow.customer_id == customer_id,
                LlmCallRow.billing == "managed",
                LlmCallRow.created_at >= month_start,
            )
            return int((await session.execute(stmt)).scalar_one())

    # ------------------------------------------------------------------
    # Spend budgets — the S4 substrate (2026-07-08). One row = one cap
    # for one spend function, optionally per-operator. The ledger's
    # origin stamps are the meter; no second counter to drift.
    # ------------------------------------------------------------------

    async def upsert_spend_budget(
        self,
        customer_id: str,
        *,
        function: str,
        cap_micro_usd: int,
        period: str = "monthly",
        operator_id: Optional[str] = None,
    ) -> SpendBudget:
        """Create or update the (customer, function, operator) budget row."""
        import uuid

        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(SpendBudgetRow).where(
                SpendBudgetRow.customer_id == customer_id,
                SpendBudgetRow.function == function,
                SpendBudgetRow.operator_id.is_(None)
                if operator_id is None
                else SpendBudgetRow.operator_id == operator_id,
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                row = SpendBudgetRow(
                    id=f"bud_{uuid.uuid4().hex[:16]}",
                    customer_id=customer_id,
                    function=function,
                    operator_id=operator_id,
                    period=period,
                    cap_micro_usd=int(cap_micro_usd),
                    # Explicit: the mapper reads pre-flush; ORM defaults
                    # fire at flush.
                    created_at=datetime.now(timezone.utc),
                )
                session.add(row)
            else:
                row.period = period
                row.cap_micro_usd = int(cap_micro_usd)
            return _spend_budget_from_row(row)

    async def get_spend_budget(
        self,
        customer_id: str,
        *,
        function: str,
        operator_id: Optional[str] = None,
    ) -> Optional[SpendBudget]:
        """Resolution order (redesign doc): operator row if present, else
        the tenant-wide row, else None."""
        async with self.session() as session:  # type: ignore[attr-defined]
            if operator_id is not None:
                stmt = select(SpendBudgetRow).where(
                    SpendBudgetRow.customer_id == customer_id,
                    SpendBudgetRow.function == function,
                    SpendBudgetRow.operator_id == operator_id,
                )
                row = (await session.execute(stmt)).scalar_one_or_none()
                if row is not None:
                    return _spend_budget_from_row(row)
            stmt = select(SpendBudgetRow).where(
                SpendBudgetRow.customer_id == customer_id,
                SpendBudgetRow.function == function,
                SpendBudgetRow.operator_id.is_(None),
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            return _spend_budget_from_row(row) if row is not None else None

    async def list_spend_budgets(self, customer_id: str) -> list[SpendBudget]:
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(SpendBudgetRow)
                .where(SpendBudgetRow.customer_id == customer_id)
                .order_by(SpendBudgetRow.function.asc())
            )
            rows = (await session.execute(stmt)).scalars().all()
            return [_spend_budget_from_row(r) for r in rows]

    async def origin_spend_micro_usd_this_period(
        self, customer_id: str, *, origin: str, period: str = "monthly"
    ) -> int:
        """Ledger spend for one origin in the current UTC period — the
        meter behind function_budget_allows. Mirrors
        managed_spend_micro_usd_this_month (the E4 CostReader)."""
        now = datetime.now(timezone.utc)
        if period == "daily":
            start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            start = now.replace(
                day=1, hour=0, minute=0, second=0, microsecond=0
            )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(
                func.coalesce(func.sum(LlmCallRow.computed_cost_micro_usd), 0)
            ).where(
                LlmCallRow.customer_id == customer_id,
                LlmCallRow.origin == origin,
                LlmCallRow.created_at >= start,
            )
            return int((await session.execute(stmt)).scalar_one())

    async def cost_totals_for_team(
        self, customer_id: str, *, since: Optional[datetime] = None
    ) -> dict[str, Any]:
        """All-time (or since `since`) totals for a team: summed cost, summed
        tokens, and the call count."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = select(
                func.coalesce(func.sum(LlmCallRow.computed_cost_micro_usd), 0),
                func.coalesce(func.sum(LlmCallRow.input_tokens), 0),
                func.coalesce(func.sum(LlmCallRow.output_tokens), 0),
                func.count(LlmCallRow.id),
            ).where(LlmCallRow.customer_id == customer_id)
            if since is not None:
                stmt = stmt.where(LlmCallRow.created_at >= since)
            cost, in_tok, out_tok, n = (await session.execute(stmt)).one()
            return {
                "customer_id": customer_id,
                "cost_micro_usd": int(cost),
                "input_tokens": int(in_tok),
                "output_tokens": int(out_tok),
                "call_count": int(n),
            }

    async def cost_by_session(
        self, customer_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Per-session cost rollup for a team, costliest first (the sortable
        all-time-per-agent view; the session is the agent unit)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(
                    LlmCallRow.session_id,
                    func.coalesce(
                        func.sum(LlmCallRow.computed_cost_micro_usd), 0
                    ).label("cost"),
                    func.count(LlmCallRow.id).label("calls"),
                )
                .where(LlmCallRow.customer_id == customer_id)
                .group_by(LlmCallRow.session_id)
                .order_by(func.sum(LlmCallRow.computed_cost_micro_usd).desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "session_id": r[0],
                    "cost_micro_usd": int(r[1]),
                    "call_count": int(r[2]),
                }
                for r in rows
            ]

    async def cost_by_operator(
        self, customer_id: str, *, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Per-operator cost rollup for a team, costliest first."""
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(
                    LlmCallRow.operator_id,
                    func.coalesce(
                        func.sum(LlmCallRow.computed_cost_micro_usd), 0
                    ).label("cost"),
                    func.count(LlmCallRow.id).label("calls"),
                )
                .where(LlmCallRow.customer_id == customer_id)
                .group_by(LlmCallRow.operator_id)
                .order_by(func.sum(LlmCallRow.computed_cost_micro_usd).desc())
                .limit(limit)
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "operator_id": r[0],
                    "cost_micro_usd": int(r[1]),
                    "call_count": int(r[2]),
                }
                for r in rows
            ]

    async def average_cost_per_agent(self, customer_id: str) -> int:
        """Average spend per agent (D6 — the session is the unit). Total team
        cost / count of distinct sessions, in micro-USD. Zero when no
        sessions have any attributed cost."""
        async with self.session() as session:  # type: ignore[attr-defined]
            total = (await session.execute(
                select(
                    func.coalesce(
                        func.sum(LlmCallRow.computed_cost_micro_usd), 0
                    )
                ).where(LlmCallRow.customer_id == customer_id)
            )).scalar_one()
            n_sessions = (await session.execute(
                select(func.count(func.distinct(LlmCallRow.session_id)))
                .where(
                    LlmCallRow.customer_id == customer_id,
                    LlmCallRow.session_id.isnot(None),
                )
            )).scalar_one()
            if not n_sessions:
                return 0
            return int(int(total) // int(n_sessions))

    async def cost_timeseries(
        self,
        customer_id: str,
        *,
        bucket: str = "day",
        days: int = 30,
    ) -> list[dict[str, Any]]:
        """Cost over time, bucketed daily or weekly, oldest bucket first.

        Buckets in Python over a window of the last `days` days (portable
        across SQLite / Postgres; row counts in dev are small — a
        daily-rollup table is the documented future optimization). bucket ∈
        'day' | 'week'. Each entry: {bucket, cost_micro_usd, call_count}.
        """
        from datetime import timedelta

        since = _utcnow() - timedelta(days=days)
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(
                    LlmCallRow.created_at,
                    LlmCallRow.computed_cost_micro_usd,
                )
                .where(
                    LlmCallRow.customer_id == customer_id,
                    LlmCallRow.created_at >= since,
                )
            )).all()

        agg: dict[str, dict[str, int]] = defaultdict(
            lambda: {"cost_micro_usd": 0, "call_count": 0}
        )
        for created_at, cost in rows:
            d = _aware(created_at)
            if bucket == "week":
                iso = d.isocalendar()
                key = f"{iso[0]}-W{iso[1]:02d}"
            else:
                key = d.date().isoformat()
            agg[key]["cost_micro_usd"] += int(cost or 0)
            agg[key]["call_count"] += 1
        return [
            {"bucket": k, **v} for k, v in sorted(agg.items())
        ]

    # ------------------------------------------------------------------
    # Budget reads (the G2 auto-pause tie-back consumes these)
    # ------------------------------------------------------------------

    async def session_cost_micro_usd(self, session_id: str) -> int:
        """Total cost attributed to one session, micro-USD. The per-session
        budget cap compares against this."""
        async with self.session() as session:  # type: ignore[attr-defined]
            total = (await session.execute(
                select(
                    func.coalesce(
                        func.sum(LlmCallRow.computed_cost_micro_usd), 0
                    )
                ).where(LlmCallRow.session_id == session_id)
            )).scalar_one()
            return int(total)

    async def team_cost_since(
        self, customer_id: str, since: datetime
    ) -> int:
        """Total team cost since `since`, micro-USD. The daily-cap budget
        compares against this (since = start of day)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            total = (await session.execute(
                select(
                    func.coalesce(
                        func.sum(LlmCallRow.computed_cost_micro_usd), 0
                    )
                ).where(
                    LlmCallRow.customer_id == customer_id,
                    LlmCallRow.created_at >= since,
                )
            )).scalar_one()
            return int(total)


def _spend_budget_from_row(row: "SpendBudgetRow") -> SpendBudget:
    return SpendBudget(
        id=row.id,
        customer_id=row.customer_id,
        function=row.function,
        operator_id=row.operator_id,
        period=row.period,  # type: ignore[arg-type]
        cap_micro_usd=int(row.cap_micro_usd or 0),
        created_at=row.created_at,
        updated_at=getattr(row, "updated_at", None),
    )
