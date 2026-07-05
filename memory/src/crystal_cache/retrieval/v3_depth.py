"""V3 Depth Router — Phase 8 of Cognitive Routing Architecture.

Handles analytical queries that require cross-crystal understanding.

The Depth Router:
  1. Searches relationship and entity facts about subjects
  2. Finds content chunks for scene references
  3. Organizes results chronologically using sparse key locators
  4. If SLM is available: pre-digests the raw context into an analytical
     summary before handing it to the user's LLM
  5. Builds structured analytical context the LLM can reason over

v2 port (Phase 7 Wave 7A): verbatim from v1. Per the agent reframe
(D-A4 — Retriever synthesis exception), this is the one retriever
that synthesizes internally rather than handing raw results back to
the agent. Survives as the `depth_search` agent tool. Reads only
through FactVectorStore + MetadataStore (Phase 3 verbatim). No SQL
violations.
"""
from __future__ import annotations

import asyncio
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import structlog

from ..cost.emit import record_model_call
from ..llm import get_llm_client
from .sparse_key import parse_key

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_index import VectorIndex

logger = structlog.get_logger(__name__)


@dataclass
class DepthResult:
    """Result from the Depth Router."""
    router_name: str = "depth"
    injection_text: Optional[str] = None
    matched_fact_ids: list[str] = field(default_factory=list)
    matched_crystal_ids: list[str] = field(default_factory=list)
    top_score: float = 0.0
    fact_count: int = 0
    voicing: str = "informational"


class DepthRouter:
    """Assembles structured analytical context for depth queries.

    When an SLM client is provided, the router pre-digests the raw
    facts into an analytical summary. The user's LLM gets pre-processed
    analysis instead of raw data points. This is the system doing the
    thinking so the LLM can focus on communication.
    """

    MAX_INJECTION_CHARS = 3000

    def __init__(
        self,
        vector_index: "VectorIndex",
        metadata_store: "MetadataStore",
    ) -> None:
        self._index = vector_index
        self._store = metadata_store

    async def search(
        self,
        customer_id: str,
        query_vector: np.ndarray,
        *,
        k: int = 20,
        hints: Optional[dict[str, str]] = None,
        query_text: str = "",
    ) -> DepthResult:
        logger.info("depth_router.ENTRY", customer_id=customer_id, k=k, hints=hints)

        all_fact_ids: list[str] = []
        all_crystal_ids: list[str] = []
        top_score = 0.0

        # --- Section 1: Relationship and entity facts ---
        relationship_results = await self._index.search_facts(
            customer_id=customer_id,
            query_vector=query_vector,
            pair_types=["entity_relationship", "entity_attribute"],
            k=k,
        )

        relationship_lines: list[str] = []
        seen_facts: set[str] = set()

        for fact_id, crystal_id, pair_type, score in relationship_results:
            if fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)
            if score > top_score:
                top_score = score
            facts = await self._store.list_facts_for_crystal(crystal_id)
            for f in facts:
                if f.id == fact_id:
                    all_fact_ids.append(f.id)
                    all_crystal_ids.append(crystal_id)
                    val = f.claim_text or f.answer_value or ""
                    if val:
                        relationship_lines.append(val)
                    break

        # --- Section 2: Scene references (content chunk previews) ---
        content_results = await self._index.search_facts(
            customer_id=customer_id,
            query_vector=query_vector,
            pair_types=["content_chunk"],
            k=k,
        )

        scene_entries: list[dict[str, Any]] = []

        for fact_id, crystal_id, pair_type, score in content_results:
            if fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)
            if score > top_score:
                top_score = score
            facts = await self._store.list_facts_for_crystal(crystal_id)
            for f in facts:
                if f.id == fact_id:
                    all_fact_ids.append(f.id)
                    all_crystal_ids.append(crystal_id)
                    scene_num = 999
                    locator = ""
                    if f.prompt_text:
                        sk = parse_key(f.prompt_text)
                        locator = sk.specific
                        m = re.search(r'(\d+)', locator)
                        if m:
                            scene_num = int(m.group(1))
                    content = f.claim_text or f.answer_value or ""
                    lines = [l.strip() for l in content.split("\n") if l.strip()]
                    preview_lines = []
                    for line in lines[1:6]:
                        if line and len(line) > 5:
                            preview_lines.append(line)
                    preview = " | ".join(preview_lines[:3])
                    if len(preview) > 200:
                        preview = preview[:200] + "..."
                    scene_entries.append({
                        "scene_num": scene_num,
                        "locator": locator or f"Scene {scene_num}",
                        "preview": preview,
                        "score": score,
                    })
                    break

        scene_entries.sort(key=lambda e: e["scene_num"])

        # --- Section 3: Q&A facts ---
        qa_results = await self._index.search_facts(
            customer_id=customer_id,
            query_vector=query_vector,
            pair_types=["question_answer"],
            k=10,
        )

        qa_lines: list[str] = []
        for fact_id, crystal_id, pair_type, score in qa_results:
            if fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)
            facts = await self._store.list_facts_for_crystal(crystal_id)
            for f in facts:
                if f.id == fact_id:
                    all_fact_ids.append(f.id)
                    all_crystal_ids.append(crystal_id)
                    val = f.claim_text or f.answer_value or ""
                    if val and len(val) > 10:
                        qa_lines.append(val)
                    break

        # --- Assemble raw sections ---
        sections: list[str] = []
        total_chars = 0

        if relationship_lines:
            header = "Character and entity relationships:"
            body = "\n".join(f"- {line}" for line in relationship_lines[:10])
            section = f"{header}\n{body}"
            sections.append(section)
            total_chars += len(section)

        if scene_entries:
            header = "Relevant scenes (chronological):"
            scene_lines = []
            for entry in scene_entries:
                if total_chars > self.MAX_INJECTION_CHARS:
                    break
                line = f"- {entry['locator']}: {entry['preview']}"
                scene_lines.append(line)
                total_chars += len(line)
            if scene_lines:
                section = f"{header}\n" + "\n".join(scene_lines)
                sections.append(section)

        if qa_lines and total_chars < self.MAX_INJECTION_CHARS:
            header = "Additional context:"
            remaining = self.MAX_INJECTION_CHARS - total_chars
            qa_body_lines = []
            for line in qa_lines[:5]:
                if len(line) < remaining:
                    qa_body_lines.append(f"- {line}")
                    remaining -= len(line)
            if qa_body_lines:
                section = f"{header}\n" + "\n".join(qa_body_lines)
                sections.append(section)

        if not sections:
            return DepthResult()

        raw_context = "\n\n".join(sections)

        # --- SLM synthesis: only for deep/compound analytical queries ---
        # Simple depth queries ("how does the conflict evolve?") get organized
        # facts. Compound queries ("what's the arc AND how does it serve the
        # theme?") get SLM pre-processing. This saves money on simple ones.
        depth_mode = (hints or {}).get("depth_mode", "shallow")
        injection_text = raw_context
        if depth_mode == "deep" and get_llm_client().is_ready():
            synthesized = await self._slm_synthesize(
                raw_context, query_text, customer_id=customer_id,
            )
            if synthesized:
                # SLM synthesis leads, raw context supports
                synthesis_section = (
                    f"Analytical summary (pre-processed from source material):\n"
                    f"{synthesized}"
                )
                remaining = self.MAX_INJECTION_CHARS - len(synthesis_section) - 40
                if remaining > 200:
                    trimmed = raw_context[:remaining].rstrip()
                    last_nl = trimmed.rfind("\n")
                    if last_nl > remaining // 2:
                        trimmed = trimmed[:last_nl]
                    injection_text = f"{synthesis_section}\n\nSupporting detail:\n{trimmed}"
                else:
                    injection_text = synthesis_section

        # Final cap
        if len(injection_text) > self.MAX_INJECTION_CHARS:
            injection_text = injection_text[:self.MAX_INJECTION_CHARS].rstrip()
            last_nl = injection_text.rfind("\n")
            if last_nl > self.MAX_INJECTION_CHARS // 2:
                injection_text = injection_text[:last_nl]

        logger.info(
            "depth_router.complete",
            customer_id=customer_id,
            relationship_facts=len(relationship_lines),
            scene_refs=len(scene_entries),
            qa_facts=len(qa_lines),
            slm_used=get_llm_client().is_ready(),
            injection_chars=len(injection_text),
            top_score=round(top_score, 4),
        )

        return DepthResult(
            injection_text=injection_text,
            matched_fact_ids=all_fact_ids,
            matched_crystal_ids=list(set(all_crystal_ids)),
            top_score=top_score,
            fact_count=len(all_fact_ids),
            voicing="informational",
        )

    async def _slm_synthesize(
        self,
        raw_context: str,
        query_text: str,
        *,
        customer_id: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Optional[str]:
        """Have the SLM pre-digest raw facts into analytical context.

        The SLM reads the assembled relationships, scene references,
        and Q&A facts, then produces a structured analytical summary
        that the user's LLM can reason over more effectively.
        """
        prompt = (
            "You are an analytical pre-processor for a knowledge retrieval system. "
            "A user asked a depth/analysis question. Below are the raw facts retrieved "
            "from the knowledge base. Your job is to organize these facts into a clear "
            "analytical summary that answers the user's question.\n\n"
            "Rules:\n"
            "- Be specific and factual. Only use information from the facts below.\n"
            "- Organize chronologically when scene numbers are present.\n"
            "- Identify patterns, arcs, and turning points.\n"
            "- Keep it to 3-5 sentences.\n"
            "- Do NOT add information that isn't in the facts.\n\n"
            f"User's question: {query_text}\n\n"
            f"Retrieved facts:\n{raw_context[:3000]}\n\n"
            "Analytical summary:"
        )

        try:
            result = await asyncio.to_thread(
                lambda: get_llm_client().complete_detailed(
                    system=None,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.0,
                    tier="small",
                )
            )
        except Exception as e:
            logger.warning("depth_router.slm_synthesis_failed", error=str(e))
            return None

        # Meter the synthesis call (flag-gated + fail-safe) with the resolved
        # model and normalized token usage the seam reports, only when there
        # is a customer to attribute the row to.
        if customer_id:
            await record_model_call(
                customer_id=customer_id,
                model=result.model,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_creation_tokens=result.cache_creation_tokens,
                cache_read_tokens=result.cache_read_tokens,
                origin="depth",
                session_id=session_id,
            )
        return result.text or None
