"""Shared task vocabulary for disposable execution (Phase 3 relocation, 2026-07-05).

Moved here from CRYS/crystal_code/disposable.py because the HOSTED plane
executes remote tasks (ratified G2) and the hosted image deliberately
excludes the agent tree — so the vocabulary both sides share must live in
the package both sides have. The agent's disposable module re-exports
everything from here; agent-side code and tests are unchanged.

Contents are verbatim from the agent module: SeedMode + seed_workspace
(seeding semantics that must never drift between local boxes and remote
jobs), DisposableLimits / TaskOutcome / DisposableResult (the runner
contract), and the CostReader indirection (budget reads the G3 ledger
without this module depending on the store).
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Awaitable, Callable, Optional

__all__ = [
    "CostReader",
    "DisposableLimits",
    "DisposableResult",
    "SeedMode",
    "StopCheck",
    "TaskOutcome",
    "Worker",
    "seed_workspace",
]


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
