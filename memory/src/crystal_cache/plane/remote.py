"""Phase 3 remote execution: TaskSpec + the remote-task runner (G2/G4).

Canonical home moved from CRYS/crystal_code (2026-07-05): the hosted
plane executes these and its image excludes the agent tree. The agent
re-exports this module as crystal_code.remote_env, unchanged surface.

The hosted plane cannot bind-mount a workspace into a Cloud Run Job and
Jobs have no exec API, so the worker program runs INSIDE the job and the
plane's role shrinks to: pack + upload the seed, launch, monitor limits,
harvest, tear down. This module is that plane-side shape, written against
two small protocols so it is fully testable with fakes today and wired to
the real GCS + Cloud Run Jobs backends when the project stands (design
doc §9 item 5):

  TaskStore  — seed/harvest storage (GCS gs://crys-tasks/{tenant}/{task}/
               in production; in-memory in tests).
  JobBackend — launch/poll/cancel one job execution (Cloud Run Jobs in
               production; a scripted fake in tests).

Reuses Phase 1's vocabulary wholesale: SeedMode + seed_workspace for
seeding semantics (they must never drift between local and remote),
DisposableLimits / TaskOutcome / DisposableResult for the runner
contract, and the CostReader indirection so budget enforcement reads
the G3 ledger without this module depending on the store.

Key lifecycle (ratified G3): the runner REVOKES the task key on every
exit path — completion, limit trip, failure, or crash — via the injected
revoke callable. Revocation is the kill switch's twin: after teardown a
leaked key is worthless.

Seed format: tar.gz (stdlib). zstd is the design-doc target; it lands
when a zstandard dep is justified — the format is an internal detail of
seed_ref/harvest_ref, so swapping costs nothing.
"""
from __future__ import annotations

import asyncio
import io
import tarfile
import tempfile
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional, Protocol

from .tasks import (
    CostReader,
    DisposableLimits,
    DisposableResult,
    SeedMode,
    TaskOutcome,
    seed_workspace,
)

__all__ = [
    "JobBackend",
    "JobState",
    "TaskSpec",
    "TaskStore",
    "pack_seed",
    "run_remote_task",
    "unpack_harvest",
]


class JobState(str, Enum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass
class TaskSpec:
    """Everything a remote job needs (ratified G2 shape). seed_ref and
    harvest_ref are STORAGE references (per-task signed URLs in
    production); task_key is the box's only credential (G3), injected
    into the job's environment by the backend."""
    task_id: str
    image: str
    command: list[str]
    seed_ref: str
    harvest_ref: str
    task_key: str
    limits: DisposableLimits = dataclass_field(default_factory=DisposableLimits)
    gpu: bool = False


class TaskStore(Protocol):
    """Per-task seed/harvest storage (GCS in production)."""

    async def put_seed(self, task_id: str, data: bytes) -> str:
        """Store the seed tarball; returns the seed_ref for the spec."""
        ...

    async def harvest_ref(self, task_id: str) -> str:
        """The (write-only, in production) reference the job uploads to."""
        ...

    async def get_harvest(self, task_id: str) -> Optional[bytes]:
        """The harvest tarball the job uploaded, or None if it made none."""
        ...

    async def delete_task_prefix(self, task_id: str) -> None:
        """Teardown: remove everything under the task's prefix. Idempotent."""
        ...


class JobBackend(Protocol):
    """One job execution's lifecycle (Cloud Run Jobs in production)."""

    async def launch(self, spec: TaskSpec) -> str:
        """Start the job; returns an execution id."""
        ...

    async def poll(self, job_id: str) -> JobState: ...

    async def cancel(self, job_id: str) -> None:
        """Stop a running execution. Idempotent; safe after completion."""
        ...


def pack_seed(
    seed_mode: SeedMode,
    working_dir: Optional[Path] = None,
    seed_path: Optional[Path] = None,
) -> bytes:
    """Materialize the seed exactly as a local box would see it (via the
    SHARED seed_workspace — semantics must never drift between backends)
    and pack it as a tar.gz. EMPTY packs an empty archive; a missing
    source fails safe to empty, same as Phase 1."""
    with tempfile.TemporaryDirectory(prefix="crys-seed-") as tmp:
        root = Path(tmp)
        seed_workspace(root, seed_mode, working_dir, seed_path)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for item in sorted(root.rglob("*")):
                tar.add(item, arcname=str(item.relative_to(root)))
        return buf.getvalue()


def unpack_harvest(data: bytes, dest: Path) -> None:
    """Extract a harvest tarball into dest. `filter="data"` strips
    absolute paths, parent traversal, and device nodes — the harvest
    crossed the trust boundary and gets no benefit of the doubt."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tar:
        tar.extractall(dest, filter="data")


async def run_remote_task(
    spec: TaskSpec,
    backend: JobBackend,
    store: TaskStore,
    *,
    cost_reader: Optional[CostReader] = None,
    revoke_key: Optional[Callable[[], Awaitable[None]]] = None,
    poll_seconds: float = 0.05,
) -> DisposableResult:
    """Launch one remote task and own its limits, mirroring
    run_disposable_task's contract on the remote shape: the monitor polls
    job state + cost, cancels the execution the moment a limit trips, and
    teardown (key revocation + prefix deletion) runs UNCONDITIONALLY.

    cost_reader is the task's ledger sum (store.task_spend_micro_usd
    bound to this task_id in production) — the same number the proxy's
    auth door enforces, read here to stop the box, not just its calls.
    """
    limits = spec.limits or DisposableLimits()
    started = time.monotonic()
    outcome = TaskOutcome.FAILED
    harvest: Optional[dict] = None
    error: Optional[str] = None
    cost = 0

    async def _read_cost() -> int:
        if cost_reader is None:
            return 0
        try:
            return await cost_reader()
        except Exception:  # noqa: BLE001 — a flaky reader never aborts
            return cost

    def _limit_hit(now_cost: int) -> Optional[TaskOutcome]:
        if (limits.deadline_seconds is not None
                and time.monotonic() - started >= limits.deadline_seconds):
            return TaskOutcome.DEADLINE_EXCEEDED
        if (limits.budget_micro_usd is not None
                and now_cost >= limits.budget_micro_usd):
            return TaskOutcome.BUDGET_EXCEEDED
        return None

    job_id: Optional[str] = None
    try:
        job_id = await backend.launch(spec)
        while True:
            state = await backend.poll(job_id)
            cost = await _read_cost()
            hit = _limit_hit(cost)
            if hit is not None:
                outcome = hit
                await backend.cancel(job_id)
                break
            if state == JobState.SUCCEEDED:
                outcome = TaskOutcome.COMPLETED
                break
            if state == JobState.FAILED:
                outcome = TaskOutcome.FAILED
                error = "job execution failed"
                break
            await asyncio.sleep(poll_seconds)

        if outcome == TaskOutcome.COMPLETED:
            data = await store.get_harvest(spec.task_id)
            if data is not None:
                harvest = {"tar_gz": data}
    except Exception as e:  # noqa: BLE001 — result carries the error
        outcome = TaskOutcome.FAILED
        error = str(e)
        if job_id is not None:
            try:
                await backend.cancel(job_id)
            except Exception:  # noqa: BLE001 — teardown still runs
                pass
    finally:
        # Teardown, unconditional and idempotent: the key dies and the
        # task's storage prefix goes away no matter how we got here.
        if revoke_key is not None:
            try:
                await revoke_key()
            except Exception:  # noqa: BLE001
                pass
        try:
            await store.delete_task_prefix(spec.task_id)
        except Exception:  # noqa: BLE001
            pass

    return DisposableResult(
        outcome=outcome,
        duration_seconds=time.monotonic() - started,
        cost_micro_usd=cost,
        harvest=harvest,
        error=error,
    )
