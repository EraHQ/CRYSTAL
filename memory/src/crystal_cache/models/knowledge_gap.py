"""KnowledgeGap — missing knowledge identified by the LLM or routers.

When the LLM emits a `crystal_pull` tool call without finding a
match, or when the navigation router returns no results for a query
the LLM expected to find facts under, a KnowledgeGap row is written.
The gap describes what the system was looking for and didn't find.

Gaps feed three downstream consumers:
  1. The cognition orchestrator's "fill_gap" task type — a background
     worker that researches the gap (web search, document review) and
     proposes a crystal write.
  2. The inspector's gap dashboard — operators see "we lack
     information about X subject in Y domain" and can route data in.
  3. The customer's own UI (future) — surfacing "we don't know this
     yet, can you teach us?" as a conversational nudge.

When a gap is filled (a new crystal lands that answers it), the
filling code sets filled_by_crystal_id and status='filled'. Gaps
older than retention window go to 'closed' without filling.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# 'low'    — informational; fill when convenient
# 'medium' — default; queue for next research pass
# 'high'   — actively blocking; surface in real-time UI
GapPriority = Literal["low", "medium", "high"]

# 'open'   — not yet addressed
# 'filled' — a crystal now answers this gap; filled_by_crystal_id set
# 'closed' — dropped without filling (past retention, or operator dismissed)
GapStatus = Literal["open", "filled", "closed"]

# 'llm_observation' — LLM noticed it couldn't answer something
# 'navigation_miss' — navigation router returned 0 results
# 'manual'          — operator flagged a gap via inspector
# 'gap_discovery'   — the idle gap-discovery scan named an important
#                     unanswered question for a subject (scan/gap_discovery.py)
GapSource = Literal[
    "llm_observation", "navigation_miss", "manual", "gap_discovery",
    # The citation-dual (proxy + agent, S3-legitimized 2026-07-08): an
    # answer produced WITH retrieval but ZERO grounded citations — the
    # bank was consulted and didn't carry it. (Was being written without
    # a literal entry — rows persisted, model validation failed silently.)
    "uncited_answer",
    # Topic seeding (2026-07-02, scan/topic_seeding.py) — store-signal
    # seeds, no model calls. 'thin_crystal_seed' was RETIRED 2026-07-08
    # (Gap Engine redesign S1: gaps are demand-driven, never inventory
    # audits) — the literal stays only so pre-existing rows still parse;
    # nothing creates it anymore.
    "thin_crystal_seed", "topic_spec",
]


class KnowledgeGap(BaseModel):
    id: str
    customer_id: str

    # Optional taxonomic hints. The LLM-observation path fills these
    # when the model has enough context to label the gap; the
    # navigation-miss path leaves them None.
    domain: Optional[str] = None    # e.g., "medical_records"
    subject: Optional[str] = None   # e.g., "patient onboarding date"

    missing: str  # Free text describing what's missing

    # S3 provenance (2026-07-08, Gap Engine redesign P5): a gap carries
    # its full sparse key (never a bare Subject — "what else about
    # Overview?" is a question about nothing) and, when demand-driven,
    # the QUERY that missed. Both optional: operator topics have
    # neither; scan-born gaps have a key but no query.
    full_key: Optional[str] = None
    triggering_query: Optional[str] = None

    priority: GapPriority = "medium"
    status: GapStatus = "open"
    source: GapSource = "llm_observation"

    # Set when status='filled'; the crystal that resolved this gap.
    filled_by_crystal_id: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    resolved_at: Optional[datetime] = None
