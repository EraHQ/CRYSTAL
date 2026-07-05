"""Disposable execution environment for autonomous background workers.

Step 5 part 2, Phase 1 (2026-07-03). Autonomous workers (gap-fill, idle
research, metacognition, "build me an app") need their OWN space: seeded by
default with a COPY of the working dir, networked (research needs the
internet), and TORN DOWN after use so that even if a worker wrecks it,
nothing real is touched. See docs/DISPOSABLE_ENVIRONMENT_DESIGN.md.

This module is the lifecycle contract + the LOCAL workspace implementation
(temp-dir copy, execution through the E1c chokepoint, unconditional
teardown). The container backend (cpu_untrusted/gpu) and the hosted
multi-tenant plane are later phases that implement the SAME
DisposableEnvironment protocol without touching callers.

Design properties (all enforced here):
  * seed: copy the working dir by default; empty or a specified path also.
  * limits: a wall-clock DEADLINE and a COST budget are first-class. The
    runner owns them and can stop + tear down. Values come from the caller
    (subscription tier when hosted, operator config when self-host) — this
    module does not hard-code them.
  * teardown: UNCONDITIONAL and idempotent — happy path, limit-kill, or
    crash. A wrecked box is meaningless by design.
  * network: the disposable box's posture is the OPPOSITE of the foreground
    agent's (which denies net to auto commands) BECAUSE it is disposable and
    its output is reviewed on the way out. Phase 1 (bwrap `cpu`) inherits
    E1c's interactive net posture via allow_shell_features.
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
import time
from dataclasses import dataclass, field as dataclass_field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional, Protocol


class SeedMode(str, Enum):
    """How the disposable workspace is seeded."""
    COPY = "copy"    # default: a copy of the working dir
    EMPTY = "empty"  # scratch build (e.g. "build me an app from nothing")
    PATH = "path"    # copy from an explicitly named path


class TaskOutcome(str, Enum):
    COMPLETED = "completed"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    BUDGET_EXCEEDED = "budget_exceeded"
    FAILED = "failed"


@dataclass
class DisposableLimits:
    """Time + cost caps. The runner OWNS these and stops the box when either
    is exceeded. deadline_seconds is wall-clock across the whole task;
    budget_micro_usd is checked against a caller-supplied cost reader so this
    module never depends on the store's ledger directly (keeps it testable +
    self-host portable). Either may be None to leave that dimension uncapped
    (an operator choice; hosted always sets both from the tier)."""
    deadline_seconds: Optional[float] = None
    budget_micro_usd: Optional[int] = None


@dataclass
class DisposableResult:
    outcome: TaskOutcome
    duration_seconds: float
    cost_micro_usd: int = 0
    # The harvested deliverable, if the worker produced one. Phase 1 returns
    # the workspace path pre-teardown to the harvest callback; this carries
    # whatever that callback chose to keep (a diff, a report, a ref).
    harvest: Optional[dict] = None
    error: Optional[str] = None


# A worker is any async callable given the live workspace dir + a "should I
# stop" check it is expected to honor between steps of long work. It returns
# an optional harvest dict (what to keep). The runner ALSO enforces the
# limits out-of-band, so a worker that ignores the stop-check is still
# bounded — the check just lets a cooperative worker exit cleanly.
Worker = Callable[["WorkspaceHandle", "StopCheck"], Awaitable[Optional[dict]]]
StopCheck = Callable[[], Optional[TaskOutcome]]
# Reads cumulative cost so far in micro-USD (e.g. the G3 ledger total for
# this task). Called by the runner to enforce the budget. Defaults to a
# zero reader (no cost tracking) so the environment works without a ledger.
CostReader = Callable[[], Awaitable[int]]


@dataclass
class WorkspaceHandle:
    """A live disposable workspace. `root` is the seeded copy the worker
    operates in; it is destroyed at teardown. `meta` carries backend
    specifics (e.g. the container name for the Phase 2 container env)."""
    root: Path
    seed_mode: SeedMode
    profile: str  # the SandboxProfile value the box runs commands under
    meta: dict = dataclass_field(default_factory=dict)


class DisposableEnvironment(Protocol):
    """The lifecycle contract. Phase 1 ships LocalWorkspaceEnv; the container
    and hosted phases implement this same protocol."""

    async def provision(self) -> WorkspaceHandle: ...
    async def teardown(self, handle: WorkspaceHandle) -> None: ...


def seed_workspace(
    root: Path,
    seed_mode: SeedMode,
    working_dir: Optional[Path],
    seed_path: Optional[Path],
) -> None:
    """Seed a freshly-created workspace root. Shared by every backend
    (local temp dir, container volume) so seeding semantics never drift:
    EMPTY leaves the root bare; COPY merges the working dir's tree in;
    PATH merges a named path. A missing/invalid source fails SAFE by
    leaving the root empty rather than raising, so teardown always runs
    and the caller sees a clean (if empty) box."""
    if seed_mode == SeedMode.EMPTY:
        return
    src = seed_path if seed_mode == SeedMode.PATH else working_dir
    if src is None or not Path(src).is_dir():
        return
    for item in Path(src).iterdir():
        dst = root / item.name
        if item.is_dir():
            shutil.copytree(item, dst, symlinks=True)
        else:
            shutil.copy2(item, dst)


# ---------------------------------------------------------------------------
# Phase 1: the local temp-dir workspace (self-host, fully testable now).
# ---------------------------------------------------------------------------

class LocalWorkspaceEnv:
    """A disposable workspace backed by a temp directory on the local host.

    Seeds a copy of the working dir (default), runs the worker through the
    E1c chokepoint under the `cpu` profile (bubblewrap when available), and
    tears the temp dir down unconditionally. This is the whole feature for
    self-host on local hardware; the container/hosted phases swap only the
    provision/teardown/exec substrate behind the same run_disposable_task().
    """

    def __init__(
        self,
        working_dir: Path,
        *,
        seed_mode: SeedMode = SeedMode.COPY,
        seed_path: Optional[Path] = None,
        profile: str = "cpu",
    ) -> None:
        self._working_dir = Path(working_dir)
        self._seed_mode = seed_mode
        self._seed_path = Path(seed_path) if seed_path else None
        self._profile = profile

    async def provision(self) -> WorkspaceHandle:
        root = Path(tempfile.mkdtemp(prefix="crys-disposable-"))
        seed_workspace(root, self._seed_mode, self._working_dir, self._seed_path)
        return WorkspaceHandle(
            root=root, seed_mode=self._seed_mode, profile=self._profile,
        )

    async def teardown(self, handle: WorkspaceHandle) -> None:
        # Unconditional + idempotent: ignore_errors so a partially-wrecked
        # box never blocks teardown, and a double-teardown is a no-op.
        shutil.rmtree(handle.root, ignore_errors=True)


async def _zero_cost() -> int:
    return 0


async def run_disposable_task(
    env: DisposableEnvironment,
    worker: Worker,
    *,
    limits: Optional[DisposableLimits] = None,
    cost_reader: Optional[CostReader] = None,
) -> DisposableResult:
    """Run one autonomous worker in a disposable environment, enforcing the
    time + cost limits, and tearing the environment down UNCONDITIONALLY.

    The runner owns the limits: it provides the worker a stop-check derived
    from the deadline and the cost budget, and — because a worker may ignore
    the check — also treats a returned outcome and its own post-run limit
    evaluation as authoritative. Teardown always runs, even on exception.
    """
    limits = limits or DisposableLimits()
    cost_reader = cost_reader or _zero_cost
    started = time.monotonic()

    def _elapsed() -> float:
        return time.monotonic() - started

    # Live limit state. `_last_cost` is refreshed by the monitor loop DURING
    # the run (not only after), so both a cooperative worker polling _stop()
    # and the runner's own monitor see the current cost.
    _last_cost = {"v": 0}
    _tripped: dict = {"v": None}

    def _limit_hit() -> Optional[TaskOutcome]:
        if (limits.deadline_seconds is not None
                and _elapsed() >= limits.deadline_seconds):
            return TaskOutcome.DEADLINE_EXCEEDED
        if (limits.budget_micro_usd is not None
                and _last_cost["v"] >= limits.budget_micro_usd):
            return TaskOutcome.BUDGET_EXCEEDED
        return None

    def _stop() -> Optional[TaskOutcome]:
        # The cooperative signal the worker polls between steps. A trip
        # detected HERE is recorded into _tripped too — otherwise a
        # cooperative worker that sees the limit and exits cleanly would be
        # scored COMPLETED because only the monitor recorded trips (a race
        # the Windows scheduler exposed: the worker's poll can win against
        # the monitor's).
        if _tripped["v"] is not None:
            return _tripped["v"]
        hit = _limit_hit()
        if hit is not None:
            _tripped["v"] = hit
        return hit

    async def _monitor(worker_task: "asyncio.Task") -> None:
        """Poll limits live; cancel the worker the moment one trips. This
        makes limits AUTHORITATIVE even for a worker that never polls
        _stop() — the cooperative check is a nicety, not the bound."""
        poll = 0.01
        while not worker_task.done():
            try:
                _last_cost["v"] = await cost_reader()
            except Exception:  # noqa: BLE001 — a flaky reader never aborts
                pass
            hit = _limit_hit()
            if hit is not None:
                _tripped["v"] = hit
                worker_task.cancel()
                return
            await asyncio.sleep(poll)

    handle = await env.provision()
    outcome = TaskOutcome.COMPLETED
    harvest: Optional[dict] = None
    error: Optional[str] = None
    try:
        # Prime cost so an already-over-budget task stops before doing work.
        try:
            _last_cost["v"] = await cost_reader()
        except Exception:  # noqa: BLE001
            _last_cost["v"] = 0
        pre = _limit_hit()
        if pre is not None:
            outcome = pre
        else:
            worker_task = asyncio.ensure_future(worker(handle, _stop))
            monitor_task = asyncio.ensure_future(_monitor(worker_task))
            try:
                harvest = await worker_task
            except asyncio.CancelledError:
                # Cancelled by the monitor because a limit tripped.
                outcome = _tripped["v"] or TaskOutcome.FAILED
                harvest = None
            finally:
                monitor_task.cancel()
                try:
                    await monitor_task
                except asyncio.CancelledError:
                    pass
            # A limit that tripped exactly as the worker finished still wins.
            if outcome == TaskOutcome.COMPLETED and _tripped["v"] is not None:
                outcome = _tripped["v"]
                harvest = None
    except Exception as e:  # noqa: BLE001 — any worker failure is contained
        outcome = TaskOutcome.FAILED
        error = str(e)
    finally:
        await env.teardown(handle)

    return DisposableResult(
        outcome=outcome,
        duration_seconds=_elapsed(),
        cost_micro_usd=_last_cost["v"],
        harvest=harvest if outcome == TaskOutcome.COMPLETED else None,
        error=error,
    )
