"""Metacognitive layer — Phase 10A (2026-05-27).

The metacognitive layer is the NEW component MCR introduces per
`docs/MCR_FRAMEWORK.md` §5.3 and D-MCR-12. It is NOT the cognition
loop (which is the research department that fills tasks); the
metacognitive layer is the editorial board that decides which tasks
are worth filling.

Phase 10A scope (P0.62 + P0.70):
  - `alignment.py` — pure-function classifier `classify_pair` (P0.73)
    + helpers for canonical-content-key extraction per action_type.
  - `synthesis.py` — v1 promotion policy `synthesize_for_trace` (P0.74)
    walking action_items + alignments to produce (promoted, deferred,
    dropped) buckets + per-item rationale strings.
  - `engine.py` — top-level entry point
    `compute_alignment_and_synthesis_for_trace(store, trace_id)` that
    threads alignment computation + synthesis + persistence + status
    transitions.

What this package does NOT do in Phase 10A:
  - Automatic background scheduling (Phase 10B per §11 Q5).
  - Critic calibration tracking (Phase 10B per §7).
  - Cross-trace pattern extraction (future).
  - Modifying the harness (D-MCR-15 / Principle 9).

The package consumes McrExtensionsMixin reads
(`list_critiques_for_trace`, `list_action_items_for_critique`,
`update_action_item_status`) and writes through
MetacognitionExtensionsMixin (`create_item_alignment`,
`create_critique_synthesis`).

Public surface:
  classify_pair (alignment.py)            — pure-function classifier
  synthesize_for_trace (synthesis.py)     — synthesis decision policy
  compute_alignment_and_synthesis_for_trace (engine.py) — entry point
"""
from .alignment import classify_pair
from .calibration import update_calibrations_from_synthesis
from .engine import compute_alignment_and_synthesis_for_trace
from .substrate_review import (
    SubstrateGroup,
    SubstrateObservationView,
    TraceSummary,
    group_substrate_observations,
    list_substrate_observations,
)
from .synthesis import synthesize_for_trace

__all__ = [
    "classify_pair",
    "compute_alignment_and_synthesis_for_trace",
    "synthesize_for_trace",
    "update_calibrations_from_synthesis",
    "list_substrate_observations",
    "group_substrate_observations",
    "SubstrateGroup",
    "SubstrateObservationView",
    "TraceSummary",
]
