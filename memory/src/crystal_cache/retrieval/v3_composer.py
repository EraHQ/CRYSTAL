"""V3 Composer — Phase 5 of Cognitive Routing Architecture.

Merges results from multiple routers into a single injection context.
Each router's results get their own section with appropriate framing.

Content chunks are never truncated (their value IS the full text).
Knowledge items are capped per section.
The composer manages total token budget across sections.

Usage:
    from crystal_cache.retrieval.v3_composer import Composer

    composer = Composer()
    injection = composer.compose([content_result, knowledge_result])
    # → "Document content:\n---\n[scene text]\n---\n\nRelevant knowledge:\n- Director: Crishna Murray\n- Producer: Big Bam Productions"

v2 port (Phase 7 Wave 7A): verbatim from v1. Per the agent reframe
(D-A2, D-A4), the composer survives as a proxy-mode helper —
chat_proxy uses it to merge router results before injection.
In agent mode, the agent's `llm_invoke` tool does composition
instead, so this module is not on the agent's tool list. Pure
function logic; no SQL, no I/O.
"""
from __future__ import annotations

import structlog
from typing import Optional

from .v3_routers import RouterResult

logger = structlog.get_logger(__name__)


class Composer:
    """Merges results from multiple routers into injection text.

    Rules:
    - Content chunks get their own section, never truncated
    - Knowledge items get their own section, capped at max_knowledge_chars
    - Empty router results are skipped
    - Sections are ordered: content first, then knowledge
    - Total injection is logged for token budget awareness
    """

    def __init__(
        self,
        *,
        max_knowledge_chars: int = 800,
        max_total_chars: Optional[int] = None,
    ) -> None:
        self.max_knowledge_chars = max_knowledge_chars
        self.max_total_chars = max_total_chars  # None = no cap

    def compose(
        self,
        results: list[RouterResult],
        *,
        primary_router: Optional[str] = None,
    ) -> Optional[str]:
        """Compose injection text from multiple router results.

        Args:
            results: list of RouterResult from each router that ran
            primary_router: name of the primary router (gets priority
                           in ordering and budget allocation)

        Returns:
            Composed injection text, or None if all routers returned
            nothing.
        """
        # Filter to results that have injection text
        active = [r for r in results if r.injection_text and r.injection_text.strip()]

        if not active:
            return None

        # Sort: primary router first, then by result quality (score)
        def sort_key(r: RouterResult) -> tuple[int, float]:
            is_primary = 0 if r.router_name == primary_router else 1
            return (is_primary, -r.top_score)

        active.sort(key=sort_key)

        sections: list[str] = []

        for result in active:
            section = self._format_section(result)
            if section:
                sections.append(section)

        if not sections:
            return None

        composed = "\n\n".join(sections)

        # Apply total cap if set (but never truncate content chunks)
        if self.max_total_chars and len(composed) > self.max_total_chars:
            logger.info(
                "composer.total_cap_exceeded",
                total_chars=len(composed),
                max_chars=self.max_total_chars,
                sections=len(sections),
            )
            # Don't truncate — just warn. Content chunks shouldn't be cut.

        logger.info(
            "composer.composed",
            sections=len(sections),
            total_chars=len(composed),
            routers=[r.router_name for r in active],
        )

        return composed

    def _format_section(self, result: RouterResult) -> Optional[str]:
        """Format a single router's result into a section."""
        text = result.injection_text
        if not text or not text.strip():
            return None

        if result.router_name == "content":
            # Content: verbatim text with document framing
            return (
                f"Document content:\n"
                f"---\n"
                f"{text.strip()}\n"
                f"---"
            )

        elif result.router_name == "knowledge":
            # Knowledge: structured facts with advisory framing
            # Cap at max_knowledge_chars
            if len(text) > self.max_knowledge_chars:
                text = text[:self.max_knowledge_chars].rstrip()
                # Don't cut mid-line
                last_newline = text.rfind("\n")
                if last_newline > 0:
                    text = text[:last_newline]

            return f"Relevant knowledge:\n{text.strip()}"

        elif result.router_name == "faq":
            return f"Related examples:\n{text.strip()}"

        elif result.router_name == "code":
            return f"Reference solution:\n{text.strip()}"

        elif result.router_name == "navigation":
            return f"Knowledge overview:\n{text.strip()}"

        elif result.router_name == "depth":
            return f"Analysis:\n{text.strip()}"

        else:
            # Unknown router — include with generic framing
            return f"Additional context:\n{text.strip()}"


def determine_injection_method(results: list[RouterResult]) -> str:
    """Determine the injection_method string for telemetry.

    Maps active router combinations to injection method values.
    """
    active = [r.router_name for r in results if r.injection_text]

    if not active:
        return "none"

    if active == ["content"]:
        return "text"
    elif active == ["knowledge"]:
        return "text"
    elif "content" in active and "knowledge" in active:
        return "text+text"
    elif "content" in active:
        return "text"
    elif "knowledge" in active:
        return "text"
    else:
        return "text"


def determine_match_type(
    results: list[RouterResult],
    *,
    high_threshold: float = 0.7,
    medium_threshold: float = 0.5,
) -> str:
    """Determine match_type from router results.

    Uses the best score across all routers.
    """
    if not results:
        return "none"

    best_score = max((r.top_score for r in results if r.injection_text), default=0.0)

    if best_score >= high_threshold:
        return "high"
    elif best_score >= medium_threshold:
        return "medium"
    elif best_score > 0.0:
        return "low"
    else:
        return "none"
