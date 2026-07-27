"""Background workers — long-running async tasks that v1 had inline
in app.py's lifespan.

v1 had three workers wired directly into the lifespan generator:
  - _crystallization_worker  → workers/crystallization.py
  - _drive_sync_worker       → RETIRED 2026-07-24 (DRIVE-Q1=B): Drive
    is a source-watch scheme now; DriveSourceHandler syncs gdrive
    watches under workers/source_sync.py, the one sync loop.
  - _cognition_worker        → workers/cognition.py

v2 extracts each into its own module with the same shape: a coroutine
that polls + processes until a shutdown event is set. The lifespan
constructs the shared shutdown event, spawns each worker as an
asyncio.Task, and waits on shutdown.

All workers consume the v2 MetadataStore methods (Phase 5) for table
access; they do NOT use inline SQLAlchemy queries (the hard rule from
the ledger).

Phase 6 of the v2 port (May 2026).

Phase 10B addition (2026-05-27): `run_metacognition_worker` lands in
`workers/metacognition.py`. It automates the metacognitive layer
(Phase 10A's `compute_alignment_and_synthesis_for_trace`) and the
shadow-critic scheduling (Phase 9.5's `shadow_review_trace`). Per
P0.82, Phase 10B does NOT auto-wire it into lifespan — operators
invoke it manually or via a Phase 10C+ wiring decision.
"""
from .crystallization import run_crystallization_worker
from .source_sync import run_source_sync_worker
from .cognition import run_cognition_worker
from .metacognition import run_metacognition_worker


def worker_roles() -> set[str]:
    """CC_WORKER_ROLES (ratified 2026-07-27): which workers THIS
    process runs; "all" (default) preserves the original shape. Lives
    HERE because production boots workers from the app lifespan while
    the standalone entry (__main__) exists for the split-process
    shape — the Gate M wiring-site lesson, now encoded as one shared
    parser both sites import instead of a rule to remember."""
    import os
    return {
        r.strip()
        for r in os.environ.get("CC_WORKER_ROLES", "all").split(",")
        if r.strip()
    }


def role_enabled(name: str, roles: "set[str] | None" = None) -> bool:
    rs = worker_roles() if roles is None else roles
    return "all" in rs or name in rs


__all__ = [
    "run_crystallization_worker",
    "run_source_sync_worker",
    "run_cognition_worker",
    "run_metacognition_worker",
    "worker_roles",
    "role_enabled",
]
