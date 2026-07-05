"""KnowledgeConflict — two stored facts that contradict each other.

The contradiction-scan generator (the convergence half of the
accommodation thesis — see `docs/NEVER_IDLE_CONVERGENCE.md`) compares
*subject-adjacent* facts and, when its discriminator returns
CONTRADICTS, writes one of these rows. A KnowledgeConflict is the
first-class peer of KnowledgeGap: a gap is "we lack knowledge about X";
a conflict is "we hold two facts about X that can't both be true."

Mirrors KnowledgeGap's shape + lifecycle deliberately. The differences
are intrinsic to a conflict being about a PAIR rather than an absence:
two fact ids, two claim snapshots (so the row reads without joins and
survives REPLACE deleting the underlying facts), two provenance strings
(the human-facing "contract vs catalog"), and a `pair_key` idempotence
hash so re-scanning an unchanged bank surfaces nothing new.

v1 (surfacing-only, locked): the scan WRITES `open` rows and nothing
else. The resolution verbs below name what a *later* curation gate may
do — qualify (add an UNLESS exception, the v1 consolidation idea),
supersede (prefer the fresher/source-of-truth side), blacklist (drop the
wrong side), or dismiss (not actually a conflict) — but the scan never
applies them. Detection and surfacing are automatic; anything
destructive waits for an explicit operator action.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'open'      — surfaced, not yet adjudicated
# 'resolved'  — an operator/curation gate applied a resolution (see below)
# 'dismissed' — judged not a real conflict; terminal, never re-surfaced
ConflictStatus = Literal["open", "resolved", "dismissed"]

# How a resolved conflict was settled. NULL while open.
#   'qualified'   — both kept; one gains an UNLESS exception clause
#   'superseded'  — one side preferred (fresher / source-of-truth)
#   'blacklisted' — the wrong side dropped
#   'dismissed'   — closed as not-a-conflict (paired with status='dismissed')
ConflictResolution = Literal[
    "qualified", "superseded", "blacklisted", "dismissed"
]

# Which generator found it. String-backed in the DB so a future detector
# ('staleness_scan') lands without a migration. 'dedup_scan' is the
# duplicate-detection generator (scan/dedup.py) — a duplicate is surfaced as a
# conflict (resolved by keeping one), so it reuses this table + the gate.
ConflictDetector = Literal["contradiction_scan", "dedup_scan"]


class KnowledgeConflict(BaseModel):
    id: str
    customer_id: str

    # The two conflicting facts. Soft pointers (no FK) — REPLACE
    # semantics delete facts/crystals, and a dangling FK is worse than
    # a dangling id the reader tolerates. Mirrors the citations /
    # shard_events soft-pointer precedent.
    fact_a_id: str
    fact_b_id: str
    crystal_a_id: Optional[str] = None
    crystal_b_id: Optional[str] = None

    # The sparse-key Subject / region where the two facts collide.
    subject: Optional[str] = None

    # Claim snapshots captured at detection time, so the row is
    # readable and auditable even after the underlying facts change or
    # are deleted.
    claim_a: str
    claim_b: str

    # Per-side provenance, human-facing ("source_kind @ source_path").
    provenance_a: Optional[str] = None
    provenance_b: Optional[str] = None

    detector: ConflictDetector = "contradiction_scan"
    status: ConflictStatus = "open"
    resolution: Optional[ConflictResolution] = None

    # Idempotence key (D4): stable hash of the sorted fact-id pair PLUS
    # a hash of both claim texts. A re-scan skips any pair whose
    # pair_key already has a row; a fact whose claim text CHANGED
    # produces a different pair_key, so the conflict is re-evaluated.
    pair_key: str

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
