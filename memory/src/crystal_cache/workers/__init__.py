"""Background workers — long-running async tasks that v1 had inline
in app.py's lifespan.

v1 had three workers wired directly into the lifespan generator:
  - _crystallization_worker  → workers/crystallization.py
  - _drive_sync_worker       → workers/drive_sync.py
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
from .drive_sync import run_drive_sync_worker
from .source_sync import run_source_sync_worker
from .cognition import run_cognition_worker
from .metacognition import run_metacognition_worker

__all__ = [
    "run_crystallization_worker",
    "run_drive_sync_worker",
    "run_cognition_worker",
    "run_metacognition_worker",
]
