"""Topic seeding — research seeds without model calls (BACKLOG §3 remainder,
2026-07-02).

One store-signal generator writes `knowledge_gaps` rows that the existing
Phase-2 fill sweep consumes (so every model call this causes stays inside
the fill sweep's existing per-cycle budget):

  * OPERATOR TOPICS — CC_RESEARCH_TOPICS (comma-separated) names things the
    operator wants the bank to know; each uncovered topic seeds one gap
    (source='topic_spec'). Empty list = the pass is inert.

THIN-CRYSTAL seeding was DELETED 2026-07-08 (Gap Engine redesign P2,
docs/GAP_ENGINE_AND_LEARN_REDESIGN.md S1): gaps are demand-driven — a
query missed — never an inventory audit of young crystals. Every crystal
born from a single learn call has one fact at birth; seeding on that is
the system nagging about its own metabolism. Operator topics survive
because they ARE demand: the operator explicitly named them.

Idempotence mirrors gap_discovery: one read of the open gaps builds the
skip-sets — a topic with an open gap is never re-seeded. A flood guard
skips the whole pass when the customer already has >= the open-gap cap,
so seeds can never pile up faster than the fill sweep drains them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

logger = structlog.get_logger(__name__)


@dataclass
class TopicSeedingResult:
    """Outcome of one topic-seeding pass for one customer."""

    customer_id: str
    seeded_topics: int
    skipped_existing: int
    flood_guarded: bool


def parse_topics(raw: str) -> list[str]:
    """CC_RESEARCH_TOPICS is comma-separated; strip + drop empties/dupes
    (order-preserving)."""
    seen: set[str] = set()
    out: list[str] = []
    for part in (raw or "").split(","):
        topic = part.strip()
        if topic and topic.lower() not in seen:
            seen.add(topic.lower())
            out.append(topic)
    return out


async def run_topic_seeding(
    *,
    store: "MetadataStore",
    customer_id: str,
    topics: Optional[list[str]] = None,
    max_seeds: Optional[int] = None,
    open_gap_cap: Optional[int] = None,
    log: Any = None,
) -> TopicSeedingResult:
    """One seeding pass. Knobs default from settings; no model calls."""
    log = log or logger
    if topics is None:
        topics = parse_topics(settings.research_topics)
    if max_seeds is None:
        max_seeds = settings.topic_seed_max_per_cycle
    if open_gap_cap is None:
        open_gap_cap = settings.topic_seed_open_gap_cap

    open_count = await store.count_knowledge_gaps(customer_id, status="open")
    if open_count >= open_gap_cap:
        return TopicSeedingResult(customer_id, 0, 0, flood_guarded=True)

    open_gaps = await store.list_knowledge_gaps(
        customer_id, status="open", limit=1000
    )
    open_subjects = {g.subject for g in open_gaps if g.subject}
    open_topics = {
        (g.subject or "").lower()
        for g in open_gaps
        if g.source == "topic_spec" and g.subject
    }

    seeded_topics = 0
    skipped_existing = 0
    budget = max(0, min(max_seeds, open_gap_cap - open_count))

    # --- Operator topics ---
    for topic in topics:
        if seeded_topics >= budget:
            break
        if topic.lower() in open_topics or topic in open_subjects:
            skipped_existing += 1
            continue
        await store.create_knowledge_gap(
            customer_id,
            domain=None,
            subject=topic,
            missing=(
                f"Operator research topic: {topic}. Gather foundational "
                f"knowledge on this subject."
            ),
            priority="medium",
            source="topic_spec",
        )
        open_topics.add(topic.lower())
        seeded_topics += 1

    result = TopicSeedingResult(
        customer_id=customer_id,
        seeded_topics=seeded_topics,
        skipped_existing=skipped_existing,
        flood_guarded=False,
    )
    if seeded_topics:
        log.info(
            "topic_seeding.pass_complete",
            customer_id=customer_id,
            seeded_topics=seeded_topics,
        )
    return result
