"""Agent-task queue primitives — the CRYS daemon's store surface (2026-06-11).

The daemon (coding-agent/crystal_code/daemon.py) is to coding tasks
what the cognition worker is to cognition tasks: a poll loop over a
table, because "database tables ARE the message queues." R9 puts the
SQL here, in the store, as the seventh mixin (same binding pattern as
the other six — see infrastructure/__init__.py).

Methods return plain dicts rather than ORM rows: rows detach from the
session, and the daemon/CLI consumers only format and branch on these
values — no domain model is warranted for a work-queue row.

Claiming SELECTs with `FOR UPDATE SKIP LOCKED`, so multiple daemons
against one Postgres store claim disjoint rows (each locks its row and
skips rows another daemon already holds). On SQLite the hint is a
no-op (the dialect omits it) and the single-writer, local-first
posture makes the claim safe anyway.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Optional

import structlog
from sqlalchemy import func
from sqlalchemy import inspect as sa_inspect
from sqlalchemy import select, update

from .schema import AgentTaskRow, Base, KnowledgeGapRow

logger = structlog.get_logger(__name__)

# Gap-retry branches are named agent/gap-<gap_id> — deterministic, so the
# daemon recovers the gap id from the branch with no extra columns.
# Module-level (not a class attribute): the mixin binding copies public
# CALLABLES only, so class attributes never reach MetadataStore.
GAP_RETRY_BRANCH_PREFIX = "agent/gap-"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _task_to_dict(row: AgentTaskRow) -> dict[str, Any]:
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "project_dir": row.project_dir,
        "task": row.task,
        "branch": row.branch,
        "status": row.status,
        "source": row.source,
        "run_at": row.run_at,
        "recur_seconds": row.recur_seconds,
        "parent_task_id": row.parent_task_id,
        "series_failures": row.series_failures,
        "report": row.report,
        "error": row.error,
        "log_path": row.log_path,
        "created_at": row.created_at,
        "started_at": row.started_at,
        "finished_at": row.finished_at,
    }


def _gap_to_dict(row: KnowledgeGapRow) -> dict[str, Any]:
    # Module-level like _task_to_dict above — underscore-prefixed names
    # are skipped by the mixin binding, so a private METHOD here would
    # vanish at runtime (it did once: self._gap_to_dict crashed the
    # daemon while the public methods around it worked fine).
    return {
        "id": row.id,
        "customer_id": row.customer_id,
        "subject": row.subject,
        "missing": row.missing,
        "status": row.status,
        "created_at": row.created_at,
        "resolved_at": row.resolved_at,
        "filled_by_crystal_id": row.filled_by_crystal_id,
    }


def _parse_gap_missing(missing: str) -> dict[str, str]:
    """Recover the structured sections from a gap's `missing` body — the
    inverse of create_agent_gap's formatting. FAILURE is everything after
    its header line."""
    out = {"task": "", "task_id": "", "branch": "", "project": "", "failure": ""}
    head, sep, tail = missing.partition("FAILURE:\n")
    if sep:
        out["failure"] = tail.strip()
    for line in head.splitlines():
        for field, prefix in (("task", "TASK: "), ("task_id", "TASK_ID: "),
                              ("branch", "BRANCH: "), ("project", "PROJECT: ")):
            if line.startswith(prefix):
                out[field] = line[len(prefix):].strip()
    return out


class AgentTasksMixin:
    """Work-queue CRUD for `agent_tasks`, bound onto MetadataStore."""

    async def create_agent_task(
        self,
        customer_id: str,
        *,
        project_dir: str,
        task: str,
        branch: Optional[str] = None,
        source: str = "cli",
        run_at: Optional[datetime] = None,
        recur_seconds: Optional[int] = None,
        parent_task_id: Optional[str] = None,
        series_failures: int = 0,
    ) -> dict[str, Any]:
        """Enqueue a task (status='queued'); returns the row as a dict.

        run_at NULL = ASAP; UTC (callers convert from local — see
        crystal_code/schedule.py boundary rule). recur_seconds NULL =
        one-shot. parent_task_id/series_failures thread recurrence
        lineage — the daemon passes them when enqueueing the next
        occurrence of a series.
        """
        row = AgentTaskRow(
            id=f"atask_{uuid.uuid4().hex}",
            customer_id=customer_id,
            project_dir=project_dir,
            task=task,
            branch=branch or None,
            status="queued",
            source=source,
            run_at=run_at,
            recur_seconds=recur_seconds,
            parent_task_id=parent_task_id,
            series_failures=series_failures,
        )
        async with self.session() as session:
            session.add(row)
            await session.commit()
            await session.refresh(row)
            result = _task_to_dict(row)
        logger.info(
            "agent_tasks.enqueued",
            task_id=result["id"], source=source, project_dir=project_dir,
        )
        return result

    async def count_agent_tasks_by_status(
        self, customer_id: str, statuses: tuple[str, ...],
    ) -> int:
        """How many of this tenant's tasks are in the given statuses —
        the admission check's queue-depth read (Phase 3 G6)."""
        async with self.session() as session:
            stmt = select(func.count()).select_from(AgentTaskRow).where(
                AgentTaskRow.customer_id == customer_id,
                AgentTaskRow.status.in_(statuses),
            )
            return int((await session.execute(stmt)).scalar_one())

    async def claim_next_agent_task(self) -> Optional[dict[str, Any]]:
        """Oldest DUE queued task -> 'running' (started_at stamped), or None.

        Due = run_at IS NULL (ASAP) or run_at <= now. Ordering is by
        effective time — COALESCE(run_at, created_at) — so a task
        scheduled for 08:00 takes its fair place among ASAP tasks
        queued around then, and an overdue scheduled task doesn't jump
        a queue it was never ahead of. Concurrent daemons claim
        disjoint rows via FOR UPDATE SKIP LOCKED (see the module
        docstring).
        """
        async with self.session() as session:
            now = _utcnow()
            stmt = (
                select(AgentTaskRow)
                .where(
                    AgentTaskRow.status == "queued",
                    (AgentTaskRow.run_at.is_(None)) | (AgentTaskRow.run_at <= now),
                )
                .order_by(
                    func.coalesce(AgentTaskRow.run_at, AgentTaskRow.created_at).asc(),
                    AgentTaskRow.created_at.asc(),
                )
                .limit(1)
                .with_for_update(skip_locked=True)
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            row.status = "running"
            row.started_at = _utcnow()
            await session.commit()
            await session.refresh(row)
            result = _task_to_dict(row)
        logger.info("agent_tasks.claimed", task_id=result["id"])
        return result

    async def finish_agent_task(
        self,
        task_id: str,
        *,
        status: str,
        report: Optional[str] = None,
        error: Optional[str] = None,
        log_path: Optional[str] = None,
    ) -> bool:
        """Mark a task 'done' or 'failed' with its outcome fields."""
        if status not in ("done", "failed"):
            raise ValueError(f"finish_agent_task: status must be done|failed, got {status!r}")
        async with self.session() as session:
            stmt = (
                update(AgentTaskRow)
                .where(AgentTaskRow.id == task_id)
                .values(
                    status=status,
                    report=report,
                    error=error,
                    log_path=log_path,
                    finished_at=_utcnow(),
                )
            )
            result = await session.execute(stmt)
            await session.commit()
            updated = result.rowcount > 0
        logger.info("agent_tasks.finished", task_id=task_id, status=status, found=updated)
        return updated

    async def get_agent_task(self, task_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:
            row = (
                await session.execute(
                    select(AgentTaskRow).where(AgentTaskRow.id == task_id)
                )
            ).scalar_one_or_none()
            return _task_to_dict(row) if row is not None else None

    async def list_agent_tasks(
        self,
        customer_id: Optional[str] = None,
        *,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Newest first; optionally scoped to a customer."""
        async with self.session() as session:
            stmt = select(AgentTaskRow).order_by(AgentTaskRow.created_at.desc()).limit(limit)
            if customer_id is not None:
                stmt = stmt.where(AgentTaskRow.customer_id == customer_id)
            rows = (await session.execute(stmt)).scalars().all()
            return [_task_to_dict(r) for r in rows]

    async def fail_stale_running_tasks(self, note: str) -> int:
        """Daemon-startup hygiene: any row still 'running' belonged to a
        daemon that died mid-task. Mark it failed with the note rather
        than re-running — the previous run may have left a branch and
        partial commits, and silent re-execution against that state is
        exactly the surprise this table exists to prevent."""
        async with self.session() as session:
            stmt = (
                update(AgentTaskRow)
                .where(AgentTaskRow.status == "running")
                .values(status="failed", error=note, finished_at=_utcnow())
            )
            result = await session.execute(stmt)
            await session.commit()
            count = result.rowcount or 0
        if count:
            logger.warning("agent_tasks.failed_stale", count=count)
        return count

    async def cancel_agent_task(self, task_id: str) -> dict[str, Any]:
        """Cancel a queued task and/or stop a recurring series.

        - queued  -> status 'cancelled' (terminal). The claim query filters
                     status='queued', so the daemon never picks it up; and
                     recur_seconds is nulled, which stops a recurring series
                     because the successor is only enqueued when an occurrence
                     FINISHES (a cancelled one never does).
        - running -> the in-flight run can't be stopped from here, but
                     recur_seconds is nulled so no successor is scheduled when
                     it finishes (the daemon re-reads recur at recurrence
                     time). Status is left for finish_agent_task to set.
        - done | failed | cancelled -> no-op (already terminal).

        Returns {found, outcome, was_recurring, status (the PRIOR status),
        task}. outcome is one of 'cancelled', 'recurrence_stopped' (a running
        recurring occurrence), 'running_uncancelable' (a running one-shot),
        'already_terminal', 'not_found'.
        """
        async with self.session() as session:
            row = (await session.execute(
                select(AgentTaskRow).where(AgentTaskRow.id == task_id)
            )).scalar_one_or_none()
            if row is None:
                return {
                    "found": False, "outcome": "not_found",
                    "was_recurring": False, "status": None, "task": None,
                }
            prior = row.status
            was_recurring = row.recur_seconds is not None
            if prior == "queued":
                row.status = "cancelled"
                row.recur_seconds = None
                row.finished_at = _utcnow()
                row.error = "cancelled by operator"
                outcome = "cancelled"
            elif prior == "running":
                # Can't stop the in-flight process; stop the series instead.
                row.recur_seconds = None
                outcome = "recurrence_stopped" if was_recurring else "running_uncancelable"
            else:  # done | failed | cancelled — terminal, nothing to do
                outcome = "already_terminal"
            await session.commit()
            await session.refresh(row)
            result = {
                "found": True, "outcome": outcome,
                "was_recurring": was_recurring, "status": prior,
                "task": _task_to_dict(row),
            }
        logger.info("agent_tasks.cancel", task_id=task_id, prior=prior, outcome=outcome)
        return result

    async def check_schema_compatibility(self) -> dict[str, Any]:
        """Diff the ORM schema against what the connected database
        actually has. Returns {"mismatches": ["table.column", ...],
        "database": <path-or-url, credentials hidden>}.

        WHY THIS EXISTS (backlog 2026-06-11, found in the F9 live
        smoke): `init()` creates MISSING TABLES but never ALTERs
        existing ones, and only the Alembic-managed dev DB migrates.
        A local default store created before a schema change (e.g. the
        VS-D2 source-versioning columns) fails at QUERY time with a
        cryptic `no such column` deep in some feature. This check
        turns that into a startup-time message naming the columns and
        the fix. The coding agent's runtime and the daemon call it
        right after init().

        What counts as a mismatch: a column the ORM declares that an
        EXISTING table lacks. Tables absent from the DB are fine
        (init() creates them); extra DB columns are fine (forward
        compatible). Placed in this mixin because its consumers are
        the CRYS runtime and daemon — the server's Alembic-managed DB
        should never trip it.
        """
        def _diff(sync_conn) -> list[str]:
            insp = sa_inspect(sync_conn)
            existing = set(insp.get_table_names())
            out: list[str] = []
            for table in Base.metadata.sorted_tables:
                if table.name not in existing:
                    continue  # init() will create it
                db_cols = {c["name"] for c in insp.get_columns(table.name)}
                for col in table.columns:
                    if col.name not in db_cols:
                        out.append(f"{table.name}.{col.name}")
            return out

        async with self.engine.connect() as conn:
            mismatches = await conn.run_sync(_diff)
        url = self.engine.url
        if url.get_backend_name().startswith("sqlite"):
            database = url.database or ":memory:"
        else:
            database = url.render_as_string(hide_password=True)
        if mismatches:
            logger.warning(
                "store.schema_mismatch", database=database, mismatches=mismatches,
            )
        return {"mismatches": mismatches, "database": database}

    # ------------------------------------------------------------------
    # Agent knowledge gaps (Phase C — terminal-failure runs become gaps
    # an idle daemon can retry; defeated retries escalate to the
    # operator). Reuses the shared knowledge_gaps table with
    # source='agent_run'; no schema change. Lineage that the table has
    # no columns for travels two ways: structured sections inside
    # `missing` (TASK_ID/BRANCH/PROJECT lines — readable in any viewer)
    # and the retry's deterministic GAP_RETRY_BRANCH_PREFIX branch name,
    # from which the gap id is recoverable for resolution.
    # ------------------------------------------------------------------

    def gap_retry_branch_prefix(self) -> str:
        """The deterministic branch prefix for gap retries. A method
        (not a class attribute) so the mixin binding carries it."""
        return GAP_RETRY_BRANCH_PREFIX

    def parse_agent_gap_missing(self, missing: str) -> dict[str, str]:
        """Public, instance-shaped wrapper over _parse_gap_missing — a
        bound @staticmethod would unwrap to a plain function and eat the
        instance as its first argument."""
        return _parse_gap_missing(missing)

    async def create_agent_gap(
        self,
        customer_id: str,
        *,
        task: str,
        task_id: str,
        branch: str,
        failing_tail: str,
        project_dir: str,
    ) -> dict[str, Any]:
        """Record a terminally-failed background run as an open gap."""
        gap_id = f"gap_{uuid.uuid4().hex}"
        missing = (
            f"Background run could not complete this task.\n"
            f"TASK: {' '.join(task.split())}\n"
            f"TASK_ID: {task_id}\n"
            f"BRANCH: {branch}\n"
            f"PROJECT: {project_dir}\n"
            f"FAILURE:\n{failing_tail.strip() or '(no verify output captured)'}"
        )
        async with self.session() as session:
            row = KnowledgeGapRow(
                id=gap_id,
                customer_id=customer_id,
                domain="coding_agent",
                subject=" ".join(task.split())[:250],
                missing=missing,
                priority="medium",
                status="open",
                source="agent_run",
            )
            session.add(row)
            await session.commit()
        logger.info("agent_gap.created", gap_id=gap_id, customer_id=customer_id)
        return _gap_to_dict(row)

    async def claim_next_open_agent_gap(self) -> Optional[dict[str, Any]]:
        """Claim the oldest open agent gap (open → retrying), or None.

        Claim-by-update mirrors claim_next_agent_task: the UPDATE's
        WHERE status='open' makes a double-claim a no-op even if two
        daemons race (rowcount tells the truth).
        """
        async with self.session() as session:
            row = (await session.execute(
                select(KnowledgeGapRow)
                .where(
                    KnowledgeGapRow.source == "agent_run",
                    KnowledgeGapRow.status == "open",
                )
                .order_by(KnowledgeGapRow.created_at.asc())
                .limit(1)
            )).scalar_one_or_none()
            if row is None:
                return None
            result = await session.execute(
                update(KnowledgeGapRow)
                .where(KnowledgeGapRow.id == row.id, KnowledgeGapRow.status == "open")
                .values(status="retrying")
            )
            await session.commit()
            if result.rowcount == 0:
                return None  # lost the race; the other claimant has it
            row.status = "retrying"
            return _gap_to_dict(row)

    async def resolve_agent_gap(
        self,
        gap_id: str,
        *,
        status: str,
        filled_by_crystal_id: Optional[str] = None,
    ) -> bool:
        """Move a gap to its terminal state: 'filled' or 'needs_operator'."""
        values: dict[str, Any] = {
            "status": status,
            "resolved_at": datetime.now(timezone.utc),
        }
        if filled_by_crystal_id:
            values["filled_by_crystal_id"] = filled_by_crystal_id
        async with self.session() as session:
            result = await session.execute(
                update(KnowledgeGapRow)
                .where(KnowledgeGapRow.id == gap_id)
                .values(**values)
            )
            await session.commit()
            return result.rowcount > 0

    async def get_agent_gap(self, gap_id: str) -> Optional[dict[str, Any]]:
        async with self.session() as session:
            row = await session.get(KnowledgeGapRow, gap_id)
            return _gap_to_dict(row) if row is not None else None

    async def list_agent_gaps(
        self, *, statuses: Optional[list[str]] = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """Agent gaps for the --tasks surface, newest first."""
        stmt = (
            select(KnowledgeGapRow)
            .where(KnowledgeGapRow.source == "agent_run")
            .order_by(KnowledgeGapRow.created_at.desc())
            .limit(limit)
        )
        if statuses:
            stmt = stmt.where(KnowledgeGapRow.status.in_(statuses))
        async with self.session() as session:
            rows = (await session.execute(stmt)).scalars().all()
            return [_gap_to_dict(r) for r in rows]

    async def reopen_stale_retrying_gaps(self) -> int:
        """Daemon-startup recovery, mirroring fail_stale_running_tasks: a
        gap stuck 'retrying' whose retry task no longer exists in the
        queue (the daemon died between claiming the gap and finishing
        the retry) goes back to 'open' so it gets its retry after all.
        A 'retrying' gap WITH a live queued/running retry task is left
        alone — normal resolution will close it."""
        async with self.session() as session:
            rows = (await session.execute(
                select(KnowledgeGapRow).where(
                    KnowledgeGapRow.source == "agent_run",
                    KnowledgeGapRow.status == "retrying",
                )
            )).scalars().all()
            reopened = 0
            for row in rows:
                live = (await session.execute(
                    select(AgentTaskRow.id).where(
                        AgentTaskRow.branch == f"{GAP_RETRY_BRANCH_PREFIX}{row.id}",
                        AgentTaskRow.status.in_(["queued", "running"]),
                    ).limit(1)
                )).scalar_one_or_none()
                if live is None:
                    row.status = "open"
                    reopened += 1
            if reopened:
                await session.commit()
            return reopened

