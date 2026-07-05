"""Diagnostic entities — §6 of BUILD_PROPOSAL.md.

These are the CORE new entities that make this a learning cache rather than
a static one. Per-crystal observations roll up into proposed edits, which
humans approve, which the rebuild worker executes.

The telemetry loop is:
    shadow eval -> QueryLog -> CrystalDiagnostic -> CrystalEdit
                                    (engine)          (proposer)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


class CrystalDiagnostic(BaseModel):
    """One per crystal, rewritten periodically from telemetry rollups."""

    id: str
    crystal_id: str
    observed_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # Failure mode distribution — classifier tags from baseline_forensics
    # Example: {"arithmetic_error": 0.6, "format_fail": 0.2, "wrong_setup": 0.2}
    failure_mode_distribution: dict[str, float] = Field(default_factory=dict)

    # Human-readable exemplar queries where this crystal helped/hurt
    top_help_query_exemplars: list[str] = Field(default_factory=list)
    top_hurt_query_exemplars: list[str] = Field(default_factory=list)

    # Compression ratio distribution (research §2.3)
    compression_ratio_p25: Optional[float] = None
    compression_ratio_p50: Optional[float] = None
    compression_ratio_p75: Optional[float] = None

    # Drift: how different is the live query distribution from the
    # distribution this crystal was built on? Expressed as KL divergence
    # of keyword distributions, or cosine distance of query centroids.
    query_distribution_drift: Optional[float] = None

    # List of CrystalEdit ids the engine proposed based on this diagnostic
    proposed_edit_ids: list[str] = Field(default_factory=list)


EditType = Literal["split", "merge", "remove_pairs", "reroute", "rebuild"]
EditStatus = Literal["pending", "approved", "rejected", "executed"]
ProposedBy = Literal["diagnostic_engine", "human"]


class CrystalEdit(BaseModel):
    """A proposed modification to a crystal. Humans approve; workers execute."""

    id: str
    crystal_id: str

    edit_type: EditType
    proposed_by: ProposedBy = "diagnostic_engine"

    # Human-readable explanation of why this edit is being proposed
    rationale: str = ""

    # IDs of facts affected by the edit.
    # For split: which pairs belong to which sub-cluster.
    # For remove_pairs: which pairs to drop.
    # For merge: (encode other crystal_id in edit_type-specific metadata).
    affected_facts: list[str] = Field(default_factory=list)

    # Engine's estimate of impact — validate vs actual outcome after execute
    expected_impact: Optional[str] = None
    # e.g. "expected help_rate delta +0.15"

    status: EditStatus = "pending"
    executed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # After execution: what was the actual outcome? (measured by diagnostic
    # engine after rebuild)
    actual_impact: Optional[str] = None
