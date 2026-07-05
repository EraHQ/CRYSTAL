"""V3 Navigation Router — Phase 6 of Cognitive Routing Architecture.

The Navigation Router uses the Index of Indexes (sparse key registry)
to answer queries like "What do you know about X?" without vector search.

Instead of searching for the closest fact, it scans ALL sparse keys for
a customer, filters by subject/domain, groups by the wide (broadest)
segment, and produces a structured summary of what knowledge exists.

This is the system's awareness of its own knowledge — not retrieval,
but foresight. It also supports gap detection: "We have Scenes 1-4 and
6-68 but no Scene 5."

Usage:
    nav_router = NavigationRouter(fact_store=fact_store, metadata_store=store)
    result = await nav_router.search(
        customer_id="cus_xxx",
        hints={"subject": "corporate mistletoe"},
    )

Unified sparse key (see docs/UNIFIED_SPARSE_KEY.md): a key is an ordered
path of segments running WIDE -> SPECIFIC, variable length. Navigation
filters subject hints against ANY segment (enter-anywhere) and domain
hints against the wide (left) end, groups by the wide segment, and
detects numeric gaps within each parent path.

v2 port (Phase 7 Wave 7A): survives as the `navigation_search` agent
tool (D-A3). Reads from `store.list_all_facts_for_customer` and operates
purely on the returned Pydantic Fact list. No SQL violations.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import structlog

from .sparse_key import parse_key, detect_gaps, SparseKey

if TYPE_CHECKING:
    from ..infrastructure.fact_vector_store import FactVectorStore
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)


@dataclass
class NavigationResult:
    """Result from the Navigation Router."""
    router_name: str = "navigation"
    injection_text: Optional[str] = None
    matched_fact_ids: list[str] = field(default_factory=list)
    matched_crystal_ids: list[str] = field(default_factory=list)
    top_score: float = 0.0
    fact_count: int = 0
    voicing: str = "informational"

    # Navigation-specific fields
    total_keys: int = 0
    matching_keys: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    gaps: list[str] = field(default_factory=list)
    subjects: list[str] = field(default_factory=list)
    domains: list[str] = field(default_factory=list)


class NavigationRouter:
    """Scans the sparse key registry to answer "what do you know?" queries.

    No vector search needed. Operates on the structured keys directly.
    """

    def __init__(
        self,
        fact_store: "FactVectorStore",
        metadata_store: "MetadataStore",
    ) -> None:
        self._fact_store = fact_store
        self._store = metadata_store

    async def search(
        self,
        customer_id: str,
        *,
        hints: Optional[dict[str, str]] = None,
        query_text: str = "",
    ) -> NavigationResult:
        """Scan the key registry and produce a knowledge overview.

        Args:
            customer_id: which customer's knowledge to scan
            hints: from QueryClassifier (subject, domain filters)
            query_text: the original query text (for context)

        Returns:
            NavigationResult with a structured summary of what
            knowledge exists for the given subject/domain.
        """
        all_facts = await self._store.list_all_facts_for_customer(customer_id)

        if not all_facts:
            return NavigationResult(
                injection_text="No knowledge has been stored for this customer yet.",
                top_score=0.5,
            )

        all_keys = [f.prompt_text for f in all_facts if f.prompt_text]

        # Structured keys are paths of >= 2 segments (wide -> specific).
        # Depth-1 keys (e.g. general benchmark questions with no path
        # structure) are counted as "other" and not summarized here —
        # navigation reports on the customer's structured, document-
        # ingested knowledge.
        parsed: list[SparseKey] = []
        unstructured_count = 0
        for key in all_keys:
            sk = parse_key(key)
            if sk.depth >= 2:
                parsed.append(sk)
            else:
                unstructured_count += 1

        logger.info(
            "navigation_router.scanning",
            customer_id=customer_id,
            total_facts=len(all_facts),
            structured_keys=len(parsed),
            unstructured_keys=unstructured_count,
        )

        # Filters from hints.
        subject_filter = hints.get("subject", "").lower() if hints else ""
        domain_filter = hints.get("domain", "").lower() if hints else ""

        # If no subject hint, try to extract one from the query text.
        if not subject_filter and query_text:
            m = re.search(
                r'(?:about|on|regarding)\s+(?:the\s+)?(.+?)(?:\?|$)',
                query_text, re.IGNORECASE
            )
            if m:
                subject_filter = m.group(1).strip().lower()

        # Subject hint matches ANY segment (enter-anywhere); domain hint
        # matches the wide (left) end.
        if subject_filter:
            matching = [
                sk for sk in parsed
                if any(subject_filter in s.lower() for s in sk.segments)
            ]
        elif domain_filter:
            matching = [sk for sk in parsed if domain_filter in sk.wide.lower()]
        else:
            matching = parsed

        if not matching and not unstructured_count:
            return NavigationResult(
                injection_text=f"No knowledge found matching '{subject_filter or domain_filter or 'any topic'}'.",
                total_keys=len(all_keys),
                matching_keys=0,
                top_score=0.3,
            )

        # Group by the wide (broadest) segment — the category.
        by_category: dict[str, list[SparseKey]] = defaultdict(list)
        for sk in matching:
            by_category[sk.wide].append(sk)

        # The segment just inside the wide end acts as the "subject"
        # dimension for the overview.
        subjects: set[str] = set()
        for sk in matching:
            if sk.depth >= 2:
                subjects.add(sk.segments[1])

        # Build the summary text.
        lines: list[str] = []
        if subject_filter:
            lines.append(f"Knowledge overview for '{subject_filter}':")
        else:
            lines.append("Knowledge overview:")

        lines.append(
            f"Total items: {len(matching)} structured facts"
            + (f" + {unstructured_count} other facts" if unstructured_count else "")
        )
        if subjects:
            lines.append(f"Subjects: {', '.join(sorted(subjects))}")
        if by_category:
            lines.append(f"Categories: {', '.join(sorted(by_category))}")
        lines.append("")

        # Per-category breakdown: list the sub-path after the category.
        for category, keys in sorted(by_category.items()):
            lines.append(f"{category}: {len(keys)} items")
            subpaths = sorted({
                " | ".join(sk.segments[1:]) for sk in keys if sk.depth >= 2
            })
            if len(subpaths) <= 10:
                for sp in subpaths:
                    lines.append(f"  - {sp}")
            elif subpaths:
                lines.append(
                    f"  - {subpaths[0]} through {subpaths[-1]} "
                    f"({len(subpaths)} total)"
                )

        # Gap detection: scope to each parent path so unrelated numeric
        # sequences (Scenes under a Script vs. Sections under a Policy)
        # don't contaminate each other. Need a few points before
        # claiming a gap.
        by_parent: dict[tuple[str, ...], list[SparseKey]] = defaultdict(list)
        for sk in matching:
            if sk.depth >= 2:
                by_parent[tuple(sk.segments[:-1])].append(sk)

        gap_lines: list[str] = []
        for parent, keys in sorted(by_parent.items()):
            if len(keys) < 3:
                continue
            gaps = detect_gaps([str(sk) for sk in keys], prefix=list(parent))
            if gaps:
                gap_lines.append(f"Gaps under {' | '.join(parent)}:")
                for gap in gaps[:10]:
                    gap_lines.append(f"  - Missing: {gap}")

        if gap_lines:
            lines.append("")
            lines.extend(gap_lines)

        # Pair type breakdown — only show user-relevant types.
        pair_type_counts: dict[str, int] = defaultdict(int)
        for f in all_facts:
            pair_type_counts[f.pair_type] = pair_type_counts.get(f.pair_type, 0) + 1

        type_display = {
            "content_chunk": "Document content chunks",
            "entity_attribute": "Entity facts",
            "question_answer": "Q&A pairs",
            "entity_relationship": "Relationships",
            "usage_question_to_example": "Usage examples",
            "cached_solution": "Cached solutions",
            "promoted_knowledge_iter1": "Promoted knowledge",
            "promoted_knowledge_iter2": "Promoted knowledge",
        }

        if pair_type_counts:
            lines.append("")
            lines.append("Knowledge types:")
            shown: dict[str, int] = {}
            for pt, count in pair_type_counts.items():
                display = type_display.get(pt)
                if display:
                    shown[display] = shown.get(display, 0) + count
            for display, count in sorted(shown.items(), key=lambda x: -x[1]):
                lines.append(f"  - {display}: {count}")

        injection_text = "\n".join(lines)

        logger.info(
            "navigation_router.complete",
            customer_id=customer_id,
            matching_keys=len(matching),
            categories=dict(sorted({k: len(v) for k, v in by_category.items()}.items())),
            subject_filter=subject_filter,
            injection_chars=len(injection_text),
        )

        return NavigationResult(
            injection_text=injection_text,
            total_keys=len(all_keys),
            matching_keys=len(matching),
            sources={k: len(v) for k, v in by_category.items()},
            gaps=[],
            subjects=list(subjects),
            domains=sorted(by_category),
            top_score=0.8 if matching else 0.3,
            fact_count=len(matching),
            voicing="informational",
        )
