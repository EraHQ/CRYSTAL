"""Topic seeding — research seeds without model calls (BACKLOG §3 remainder,
2026-07-02).

Two store-signal generators write `knowledge_gaps` rows that the existing
Phase-2 fill sweep consumes (so every model call this causes stays inside
the fill sweep's existing per-cycle budget):

  * THIN CRYSTALS — a crystal with only 1..N facts is thin coverage; seed
    "what else is important about {subject}?" (source='thin_crystal_seed').
    Blacklisted crystals never seed. Zero-fact crystals are skipped (no
    subject to anchor research on).
  * OPERATOR TOPICS — CC_RESEARCH_TOPICS (comma-separated) names things the
    operator wants the bank to know; each uncovered topic seeds one gap
    (source='topic_spec'). Empty list = this half is inert.

Idempotence mirrors gap_discovery: one read of the open gaps builds the
skip-sets — a subject with ANY open gap is never re-seeded, a topic with an
open topic_spec gap is never re-seeded. A flood guard skips the whole pass
when the customer already has >= the open-gap cap, so seeds can never pile
up faster than the fill sweep drains them.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import settings
# Reuse the canonical sparse-key parsers (fed via a prompt_text shim).
from .contradiction import _subject_of
from .gap_discovery import _domain_of

if TYPE_CHECKING:
    from ..infrastructure import MetadataStore

logger = structlog.get_logger(__name__)


@dataclass
class TopicSeedingResult:
    """Outcome of one topic-seeding pass for one customer."""

    customer_id: str
    seeded_thin: int
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
    thin_max_facts: Optional[int] = None,
    max_seeds: Optional[int] = None,
    open_gap_cap: Optional[int] = None,
    log: Any = None,
) -> TopicSeedingResult:
    """One seeding pass. Knobs default from settings; no model calls."""
    log = log or logger
    if topics is None:
        topics = parse_topics(settings.research_topics)
    if thin_max_facts is None:
        thin_max_facts = settings.thin_crystal_max_facts
    if max_seeds is None:
        max_seeds = settings.topic_seed_max_per_cycle
    if open_gap_cap is None:
        open_gap_cap = settings.topic_seed_open_gap_cap

    open_count = await store.count_knowledge_gaps(customer_id, status="open")
    if open_count >= open_gap_cap:
        return TopicSeedingResult(customer_id, 0, 0, 0, flood_guarded=True)

    open_gaps = await store.list_knowledge_gaps(
        customer_id, status="open", limit=1000
    )
    open_subjects = {g.subject for g in open_gaps if g.subject}
    open_topics = {
        (g.subject or "").lower()
        for g in open_gaps
        if g.source == "topic_spec" and g.subject
    }

    seeded_thin = 0
    seeded_topics = 0
    skipped_existing = 0
    budget = max(0, min(max_seeds, open_gap_cap - open_count))

    # --- Thin crystals ---
    thin = await store.list_thin_crystals_for_customer(
        customer_id, max_facts=thin_max_facts, limit=50,
    )
    for row in thin:
        if seeded_thin + seeded_topics >= budget:
            break
        shim = SimpleNamespace(prompt_text=row["sample_key"])
        subject = _subject_of(shim)  # type: ignore[arg-type]
        if not subject:
            continue
        if subject in open_subjects:
            skipped_existing += 1
            continue
        await store.create_knowledge_gap(
            customer_id,
            domain=_domain_of(shim),  # type: ignore[arg-type]
            subject=subject,
            missing=(
                f"Coverage for {subject} is thin "
                f"({row['fact_count']} fact(s)). What else is important to "
                f"know about {subject}?"
            ),
            priority="low",
            source="thin_crystal_seed",
        )
        open_subjects.add(subject)
        seeded_thin += 1

    # --- Operator topics ---
    for topic in topics:
        if seeded_thin + seeded_topics >= budget:
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
        seeded_thin=seeded_thin,
        seeded_topics=seeded_topics,
        skipped_existing=skipped_existing,
        flood_guarded=False,
    )
    if seeded_thin or seeded_topics:
        log.info(
            "topic_seeding.pass_complete",
            customer_id=customer_id,
            seeded_thin=seeded_thin,
            seeded_topics=seeded_topics,
        )
    return result
