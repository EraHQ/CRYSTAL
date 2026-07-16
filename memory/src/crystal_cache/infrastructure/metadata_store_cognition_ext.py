"""Cognition worker primitives — Phase 6 Wave C.

Phase 6 Wave C ports v1's `cognition/` package. The worker
`_worker_crystal_key_scan` in v1 had inline SQLAlchemy
(`sa_select(FactRow).join(CrystalRow,...)`) to perform a prefix
scan on facts' prompt_text within a tenant. v2's "no SQL outside
the store" rule (R9) requires this to land as a store method.

The method does not fit naturally into AuditTablesMixin (which is
scoped to the eight audit tables) and editing the Phase 3 verbatim
port of metadata_store.py is the explicit non-goal of the mixin
pattern (D12). So we follow the same shape Phase 6.5 P4.1
introduced for customer extensions: a separate file with its own
mixin, bound by `infrastructure/__init__.py` alongside the others.

If future cognition workers need additional store primitives
(rerank-by-recency, cross-crystal aggregate scans, etc.), they
belong here next to `list_facts_by_key_prefix`.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import structlog
from sqlalchemy import and_, or_, select

from ..models import Fact
from .schema import CognitionRunRow, CrystalRow, FactRow, RunCritiqueRow

logger = structlog.get_logger(__name__)


def _fact_from_row_minimal(row: FactRow) -> Fact:
    """Local converter — mirrors `metadata_store._fact_from_row`.

    Duplicated here rather than imported because the Phase 3 file's
    `_fact_from_row` is a module-private helper not exposed at the
    package boundary. The duplication is cheap (one function); the
    alternative is exposing internal converters or importing from
    a sibling private module, both of which add coupling.
    """
    return Fact(
        id=row.id,
        crystal_id=row.crystal_id,
        claim_text=row.claim_text,
        pair_type=row.pair_type,
        source_kind=row.source_kind,  # type: ignore[arg-type]
        answer_value=row.answer_value,
        prompt_text=row.prompt_text or "",
        vector=row.vector or [],
        source_doc_id=row.source_doc_id,
        extracted_by=row.extracted_by,
        verified_by=row.verified_by,
        grating_strength=row.grating_strength,
        hit_count=row.hit_count,
        last_hit_at=row.last_hit_at,
        created_at=row.created_at,
    )


class CognitionExtensionsMixin:
    """Cognition-worker store primitives, bound onto MetadataStore.

    Same binding pattern as AuditTablesMixin and
    CustomerExtensionsMixin: `infrastructure/__init__.py` iterates
    public methods at import time and `setattr`s them onto
    MetadataStore via `_bind_mixin_methods`. The mixin is NOT in
    MetadataStore's MRO — the binding is attribute-level.
    """

    async def upsert_cognition_run(
        self,
        run_id: str,
        customer_id: str,
        *,
        status: str,
        trigger_type: str = "",
        trigger_id: Optional[str] = None,
        goal_title: str = "",
        summary: Optional[dict] = None,
        detail: Optional[dict] = None,
        terminal: bool = False,
    ) -> None:
        """Persist one environment snapshot (S9). Called by the engine
        at every lifecycle transition; last write wins."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionRunRow, run_id)
            now = datetime.now(timezone.utc)
            if row is None:
                row = CognitionRunRow(
                    id=run_id,
                    customer_id=customer_id,
                    status=status,
                    trigger_type=trigger_type or None,
                    trigger_id=trigger_id or None,
                    goal_title=(goal_title or None),
                    summary=summary,
                    detail=detail,
                    created_at=now,
                )
                session.add(row)
            else:
                row.status = status
                if trigger_type:
                    row.trigger_type = trigger_type
                if trigger_id:
                    row.trigger_id = trigger_id
                if goal_title:
                    row.goal_title = goal_title
                if summary is not None:
                    row.summary = summary
                if detail is not None:
                    row.detail = detail
            if terminal and row.completed_at is None:
                row.completed_at = now

    async def count_cognition_runs_by_triggers(
        self,
        customer_id: str,
        trigger_ids: list[str],
    ) -> dict[str, dict]:
        """Gaps surface (2026-07-16, Gate 2): per-trigger loop status
        in one query — terminal run count (the cycles-used number)
        plus the newest run overall (active or terminal) for
        click-through. Returns
        {trigger_id: {run_count, last_run_id, last_run_status}}."""
        if not trigger_ids:
            return {}
        active = (
            "created", "orchestrating", "working", "validating",
            "rejected",
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CognitionRunRow)
                .where(
                    CognitionRunRow.customer_id == customer_id,
                    CognitionRunRow.trigger_id.in_(trigger_ids),
                )
                .order_by(CognitionRunRow.created_at.desc())
            )
            rows = (await session.execute(stmt)).scalars().all()
        out: dict[str, dict] = {}
        for row in rows:
            slot = out.setdefault(row.trigger_id, {
                "run_count": 0,
                "last_run_id": row.id,
                "last_run_status": row.status,
            })
            if row.status not in active:
                slot["run_count"] += 1
        return out

    async def list_run_verdicts_for_trigger(
        self,
        customer_id: str,
        *,
        trigger_id: str,
        exclude_run_id: Optional[str] = None,
        limit: int = 3,
    ) -> dict:
        """Cognition cycles (2026-07-16, Q1B): the cross-run read. All
        TERMINAL runs on this trigger for this customer, newest first.

        Returns {run_count, verdicts, hint_findings}:
        - run_count: total terminal runs on the trigger (cycle number
          for a new run = run_count + 1; the cap check is run_count
          vs cognition_cycle_cap);
        - verdicts: the newest `limit` runs' final verdicts, compact —
          run_id, cycle, score, reasoning, issues, suggestions,
          attempts, status, created_at. Sourced from the persisted
          detail snapshot (validation, else last rejection_log entry);
        - hint_findings: the NEWEST prior run's harvested findings
          (carried_findings, else the last archived attempt's), capped
          at 15 — passed onward as UNVERIFIED HINTS per Q1B.
        """
        active = (
            "created", "orchestrating", "working", "validating",
            "rejected",
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(CognitionRunRow)
                .where(
                    CognitionRunRow.customer_id == customer_id,
                    CognitionRunRow.trigger_id == trigger_id,
                    CognitionRunRow.status.notin_(active),
                )
                .order_by(CognitionRunRow.created_at.desc())
            )
            if exclude_run_id:
                stmt = stmt.where(CognitionRunRow.id != exclude_run_id)
            rows = (await session.execute(stmt)).scalars().all()

        verdicts: list[dict] = []
        hint_findings: list[dict] = []
        for i, row in enumerate(rows[: max(0, limit)]):
            d = dict(row.detail or {})
            v = d.get("validation") or {}
            if not v:
                rej = d.get("rejection_log") or []
                v = rej[-1] if rej else {}
            verdicts.append({
                "run_id": row.id,
                "cycle": d.get("cycle", 1),
                "score": float(v.get("score") or 0.0),
                "reasoning": (v.get("reasoning") or ""),
                "issues": list(v.get("issues") or []),
                "suggestions": list(v.get("suggestions") or []),
                "attempts": d.get("attempts", 0),
                "status": row.status,
                "created_at": (
                    row.created_at.isoformat() if row.created_at else ""
                ),
            })
            if i == 0:
                found = list(d.get("carried_findings") or [])
                if not found:
                    hist = d.get("attempt_history") or []
                    if hist:
                        found = list(
                            (hist[-1] or {}).get("findings") or []
                        )
                hint_findings = [
                    f for f in found[:15] if isinstance(f, dict)
                ]
        return {
            "run_count": len(rows),
            "verdicts": verdicts,
            "hint_findings": hint_findings,
        }

    async def list_cognition_runs(
        self,
        customer_id: str = "",
        *,
        completed_limit: int = 10,
    ) -> list[dict]:
        """Active runs (all) + the most recent terminal runs (capped).
        Returns stored summary dicts, active first, newest first."""
        active_statuses = (
            "created", "orchestrating", "working", "validating", "rejected",
        )
        async with self.session() as session:  # type: ignore[attr-defined]
            base = select(CognitionRunRow)
            if customer_id:
                base = base.where(CognitionRunRow.customer_id == customer_id)
            active = (await session.execute(
                base.where(CognitionRunRow.status.in_(active_statuses))
                .order_by(CognitionRunRow.updated_at.desc())
            )).scalars().all()
            done = (await session.execute(
                base.where(CognitionRunRow.status.not_in(active_statuses))
                .order_by(CognitionRunRow.updated_at.desc())
                .limit(completed_limit)
            )).scalars().all()
            out = []
            for row in list(active) + list(done):
                d = dict(row.summary or {})
                d.setdefault("id", row.id)
                d.setdefault("customer_id", row.customer_id)
                d["status"] = row.status
                d["completed_at"] = (
                    row.completed_at.isoformat() if row.completed_at else None
                )
                out.append(d)
            return out

    async def get_cognition_run(
        self, run_id: str
    ) -> Optional[dict]:
        """One run's stored detail snapshot (the env.to_dict() shape)."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(CognitionRunRow, run_id)
            if row is None:
                return None
            d = dict(row.detail or {})
            d.setdefault("id", row.id)
            d.setdefault("customer_id", row.customer_id)
            d["status"] = row.status
            d["completed_at"] = (
                row.completed_at.isoformat() if row.completed_at else None
            )
            return d

    async def list_facts_by_key_prefix(
        self,
        customer_id: str,
        *,
        key_prefix: str,
        subject_contains: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """Prefix scan on facts' `prompt_text` (sparse key) within a tenant.

        Returns all matching facts ordered by prompt_text ascending so
        cognition's `crystal_key_scan` worker can produce a stable,
        enumerable listing.

        Distinct from semantic vector search (FactVectorStore.search):
        this is a literal-string prefix scan on the indexed
        `prompt_text` column. Use it for enumeration questions
        ("how many scenes does the script have", "list every chapter")
        where vector top-k would return a truncated similarity-ranked
        sample instead of the full set.

        Args:
            customer_id: tenant scope. Joined through crystals.customer_id.
                Since the general-crystals merge (2026-06-12), scope is
                the tenant's crystals PLUS the general bank (customer_id
                NULL) for every type the tenant subscribes to — resolved
                here via get_customer_general_types so all key-scan
                consumers (agent key_scan tool, cognition
                crystal_key_scan) inherit general knowledge with zero
                call-site changes. Empty subscription = tenant-only,
                exactly the old behavior.
            key_prefix: literal prefix the fact's prompt_text must start
                with (e.g. `Script|Scene `). Caller is responsible for
                escaping any SQL LIKE metacharacters; current usage
                doesn't need it because v1's sparse keys are
                pipe-separated ASCII without `_` or `%`.
            subject_contains: optional secondary filter — prompt_text
                must additionally contain this substring. Used by
                cognition to narrow `Script|Scene |*` to a specific
                document subject.
            limit: optional row cap. None = no cap; callers that need
                cross-corpus enumeration pass None deliberately. The
                in-tenant set is bounded by document count so unbounded
                is generally fine at inspector scale.

        Returns:
            list of Fact in ascending prompt_text order.
        """
        subscribed = await self.get_customer_general_types(customer_id)  # type: ignore[attr-defined]
        scope = CrystalRow.customer_id == customer_id
        if subscribed:
            scope = or_(
                scope,
                and_(
                    CrystalRow.customer_id.is_(None),
                    CrystalRow.crystal_type.in_(subscribed),
                ),
            )
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(
                    and_(
                        scope,
                        FactRow.prompt_text.like(f"{key_prefix}%"),
                    )
                )
                .order_by(FactRow.prompt_text)
            )
            if subject_contains:
                stmt = stmt.where(FactRow.prompt_text.contains(subject_contains))
            if limit is not None:
                stmt = stmt.limit(limit)

            result = await session.execute(stmt)
            return [
                _fact_from_row_minimal(r)
                for r in result.scalars().all()
            ]

    async def list_recent_facts_for_customer(
        self,
        customer_id: str,
        *,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """A customer's OWN facts, newest first — candidate source for the
        contradiction scan (docs/NEVER_IDLE_CONVERGENCE.md).

        Deliberately distinct from `list_facts_by_key_prefix`:
          * OWN FACTS ONLY (D8). Joined through crystals.customer_id ==
            customer_id, with NO general-subscription merge. A customer's
            contradictions in v1 are within its own bank; general-vs-customer
            conflicts are a fast-follow. (list_facts_by_key_prefix folds in
            the general bank for subscribed types — wrong scope here.)
          * RECENCY ORDER (D2). Ordered by FactRow.created_at DESC so the
            generator's candidate cap keeps the freshest facts and the scan
            stays bounded; ties broken by id for determinism.
          * NO prefix filter. The generator buckets by crystal + sparse-key
            Subject in Python over the returned set; this method just hands
            it the (bounded) recent own-fact population.

        Args:
            customer_id: tenant scope (own crystals only).
            limit: optional cap on facts returned (the generator passes a
                cap so a huge bank doesn't enumerate unbounded). None = all.

        Returns:
            list of Fact, newest first.
        """
        async with self.session() as session:  # type: ignore[attr-defined]
            stmt = (
                select(FactRow)
                .join(CrystalRow, FactRow.crystal_id == CrystalRow.id)
                .where(CrystalRow.customer_id == customer_id)
                .order_by(FactRow.created_at.desc(), FactRow.id.desc())
            )
            if limit is not None:
                stmt = stmt.limit(limit)
            result = await session.execute(stmt)
            return [
                _fact_from_row_minimal(r)
                for r in result.scalars().all()
            ]

    # =================================================================
    # run_critiques — operator critiques (Q2B, 2026-07-15)
    # =================================================================

    async def create_run_critique(
        self,
        run_id: str,
        customer_id: str,
        *,
        target_path: str,
        text: str,
        author: str = "operator",
        trigger_id: Optional[str] = None,
    ) -> dict:
        """Pin a critique to part of a run's anatomy. trigger_id is
        denormalized so the ratchet feed (engine) can pull open
        critiques across runs of the same query-class."""
        import uuid
        cid = f"crit_{uuid.uuid4().hex[:16]}"
        now = datetime.now(timezone.utc)
        async with self.session() as session:  # type: ignore[attr-defined]
            session.add(RunCritiqueRow(
                id=cid, run_id=run_id, customer_id=customer_id,
                trigger_id=trigger_id or None,
                target_path=(target_path or "run")[:256],
                author=(author or "operator")[:128],
                text=text, status="open", created_at=now,
            ))
        return {"id": cid, "run_id": run_id, "customer_id": customer_id,
                "trigger_id": trigger_id, "target_path": target_path,
                "author": author, "text": text, "status": "open",
                "created_at": now.isoformat(), "resolved_at": None}

    async def list_run_critiques(self, run_id: str) -> list[dict]:
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(RunCritiqueRow)
                .where(RunCritiqueRow.run_id == run_id)
                .order_by(RunCritiqueRow.created_at.asc())
            )).scalars().all()
            return [_critique_to_dict(r) for r in rows]

    async def get_run_critique(self, critique_id: str) -> Optional[dict]:
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(RunCritiqueRow, critique_id)
            return _critique_to_dict(row) if row else None

    async def set_run_critique_status(
        self, critique_id: str, status: str
    ) -> bool:
        """open|resolved. Returns False when the critique is unknown."""
        async with self.session() as session:  # type: ignore[attr-defined]
            row = await session.get(RunCritiqueRow, critique_id)
            if row is None:
                return False
            row.status = status
            row.resolved_at = (
                datetime.now(timezone.utc) if status == "resolved" else None
            )
            return True

    async def count_open_critiques_by_run(
        self, run_ids: list[str]
    ) -> dict[str, int]:
        """Bulk open-critique counts for the run list's badges."""
        if not run_ids:
            return {}
        from sqlalchemy import func
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(RunCritiqueRow.run_id, func.count())
                .where(RunCritiqueRow.run_id.in_(run_ids),
                       RunCritiqueRow.status == "open")
                .group_by(RunCritiqueRow.run_id)
            )).all()
            return {r[0]: int(r[1]) for r in rows}

    async def list_open_critiques_for_trigger(
        self,
        customer_id: str,
        *,
        trigger_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 10,
    ) -> list[dict]:
        """The ratchet feed's read (Q2B): open critiques on THIS run
        (retry case) plus open critiques from other runs sharing the
        same trigger (same gap/task = same query-class). Newest first,
        capped — the orchestrator gets signal, not a backlog dump."""
        conds = []
        if run_id:
            conds.append(RunCritiqueRow.run_id == run_id)
        if trigger_id:
            conds.append(RunCritiqueRow.trigger_id == trigger_id)
        if not conds:
            return []
        async with self.session() as session:  # type: ignore[attr-defined]
            rows = (await session.execute(
                select(RunCritiqueRow)
                .where(RunCritiqueRow.customer_id == customer_id,
                       RunCritiqueRow.status == "open",
                       or_(*conds))
                .order_by(RunCritiqueRow.created_at.desc())
                .limit(limit)
            )).scalars().all()
            return [_critique_to_dict(r) for r in rows]


def _critique_to_dict(row: RunCritiqueRow) -> dict:
    return {
        "id": row.id, "run_id": row.run_id,
        "customer_id": row.customer_id, "trigger_id": row.trigger_id,
        "target_path": row.target_path, "author": row.author,
        "text": row.text, "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }
