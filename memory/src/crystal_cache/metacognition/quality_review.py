"""Response-quality critique stream — S11 (2026-07-09).

The Critiques console surface has, until now, shown only the
substrate_complaint channel (the system making good work hard).
Every OTHER observation the critics record — assumptions, thin
generalizations, source contradictions, questionable tool outputs,
papered-over gaps, unflagged evidence→inference crossings, skipped
reasoning steps — is the RESPONSE-QUALITY stream: what the shadow
and self critics think of the agent's actual work. Recorded on
every critiqued turn since Phase 9.5; surfaced nowhere. This module
is the surfacing path.

Mirrors substrate_review.py deliberately: same never-raise
composition, same list/group pair, same consumers (HTTP endpoint,
future CLI). Per Principle 9 / D-MCR-15 this is a READ surface —
the harness is never modified based on what gets read here, and v1
is deliberately dismissal-free: observations live inside the
critique row's JSON, carry no per-item status, and exist to be
READ, not triaged away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import structlog

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

logger = structlog.get_logger(__name__)

# The one observation type that belongs to the OTHER surface.
_SUBSTRATE_TYPE = "substrate_complaint"


@dataclass
class QualityObservationView:
    """One quality observation composed with its critique context."""

    observation_type: str
    detail: dict[str, Any]
    critique_id: str
    customer_id: str
    critic_role: str
    critic_model: str
    summary_text: Optional[str]
    sequence_id: Optional[str]
    turn_index: Optional[int]
    trace_id: Optional[str]
    created_at: Optional[datetime]

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_type": self.observation_type,
            "detail": self.detail,
            "critique_id": self.critique_id,
            "customer_id": self.customer_id,
            "critic_role": self.critic_role,
            "critic_model": self.critic_model,
            "summary_text": self.summary_text,
            "sequence_id": self.sequence_id,
            "turn_index": self.turn_index,
            "trace_id": self.trace_id,
            "created_at": (
                self.created_at.isoformat() if self.created_at else None
            ),
        }


@dataclass
class QualityGroup:
    """Observations grouped by type — loudest failure mode first."""

    observation_type: str
    count: int
    latest: list[QualityObservationView] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_type": self.observation_type,
            "count": self.count,
            "latest": [v.to_dict() for v in self.latest],
        }


async def list_quality_observations(
    store: "MetadataStore",
    *,
    customer_id: Optional[str] = None,
    critic_role: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 50,
) -> list[QualityObservationView]:
    """Flatten recent critiques into quality observations.

    Reads bounded recent critiques (list_recent_critiques), flattens
    each row's observations JSON, EXCLUDES substrate_complaint (that
    channel has its own surface), and composes each survivor with its
    critique context. Ordered most-recent-critique-first, preserving
    in-critique observation order.

    NEVER raises: malformed observation entries are logged and
    skipped; one bad critique row never hides the rest.
    """
    critiques = await store.list_recent_critiques(
        customer_id=customer_id,
        critic_role=critic_role,
        since=since,
        # Over-fetch critiques relative to the observation limit: many
        # critiques carry zero quality observations (a clean shadow
        # pass is a valid outcome).
        limit=max(limit, 50) * 4,
    )

    views: list[QualityObservationView] = []
    for c in critiques:
        if len(views) >= limit:
            break
        for obs in c.observations or []:
            if len(views) >= limit:
                break
            try:
                obs_type = str(
                    obs.get("observation_type")
                    or obs.get("type")
                    or ""
                ).strip()
                if not obs_type or obs_type == _SUBSTRATE_TYPE:
                    continue
                detail = {
                    k: v for k, v in obs.items()
                    if k not in ("observation_type", "type")
                }
                views.append(QualityObservationView(
                    observation_type=obs_type,
                    detail=detail,
                    critique_id=c.id,
                    customer_id=c.customer_id,
                    critic_role=c.critic_role,
                    critic_model=c.critic_model,
                    summary_text=c.summary_text,
                    sequence_id=c.sequence_id,
                    turn_index=c.turn_index,
                    trace_id=c.trace_id,
                    created_at=c.created_at,
                ))
            except Exception as e:
                logger.warning(
                    "quality_review.observation_skipped",
                    critique_id=getattr(c, "id", None),
                    error=str(e),
                )
    return views


async def group_quality_observations(
    store: "MetadataStore",
    *,
    customer_id: Optional[str] = None,
    since: Optional[datetime] = None,
    limit: int = 200,
    examples_per_group: int = 3,
) -> list[QualityGroup]:
    """Group quality observations by type, most-frequent-first.

    Built ON TOP of list_quality_observations — identical filtering
    and composition; grouping happens in Python over the same bounded
    rows. The loudest failure mode tops the list; ties break toward
    the type seen most recently.
    """
    views = await list_quality_observations(
        store, customer_id=customer_id, since=since, limit=limit,
    )
    by_type: dict[str, list[QualityObservationView]] = {}
    for v in views:
        by_type.setdefault(v.observation_type, []).append(v)

    groups = [
        QualityGroup(
            observation_type=t,
            count=len(vs),
            latest=vs[:examples_per_group],
        )
        for t, vs in by_type.items()
    ]
    groups.sort(key=lambda g: g.count, reverse=True)
    return groups
