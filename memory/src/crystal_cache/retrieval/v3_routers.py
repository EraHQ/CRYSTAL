"""V3 Routers — Phase 4 of Cognitive Routing Architecture.

Content Router and Knowledge Router. Each router searches the
FactVectorStore filtered by pair_type and produces injection text.

v2 port (Phase 7 Wave 7A): verbatim from v1. Per the agent reframe
(D-A3), these survive as standalone tool implementations for the
agent surface (`content_search`, `knowledge_search`). They are also
consumed by v3_composer (Wave 7A) and will be consumed by chat_proxy
in Wave 7F. Pure consumers of FactVectorStore + MetadataStore; no
SQL violations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

import structlog

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_index import VectorIndex

logger = structlog.get_logger(__name__)


@dataclass
class RouterResult:
    """Result from a single router's search."""
    router_name: str
    injection_text: Optional[str] = None
    matched_fact_ids: list[str] = field(default_factory=list)
    matched_crystal_ids: list[str] = field(default_factory=list)
    top_score: float = 0.0
    fact_count: int = 0
    voicing: str = "advisory"


def _calibrate_by_subtype(
    candidates: list[tuple],
) -> list[tuple]:
    """Re-rank content candidates so code isn't buried under prose.

    gtr-t5-base scores English prose systematically higher than verbatim
    code for conceptual queries (same-language bias), so on a mixed
    content_chunk bank the prose / ledger crystals dominate the top and
    code crystals — the actual implementation — fall below the cutoff.
    This cancels that skew WITHOUT a code-aware encoder: bucket the
    candidates by source (sparse key starting "Code|" = code, everything
    else = prose), z-normalize cosine WITHIN each bucket, and re-sort by
    the calibrated score so each candidate competes against its own
    modality's distribution. A strongly-matching code crystal (high z in
    the code bucket) then out-ranks a weakly-matching prose crystal (low
    z in the prose bucket), while the overall-strongest match still leads.

    Input rows are 5-tuples (fact_id, crystal_id, pair_type, cosine,
    sparse_key) as returned by FactVectorStore.search(with_keys=True).
    Output rows are the historical 4-tuples (fact_id, crystal_id,
    pair_type, cosine) in calibrated order — the ORIGINAL cosine is
    preserved in slot 3 so downstream thresholds and telemetry are
    unchanged; only the ordering reflects the calibration.

    A bucket with fewer than 2 members (or zero variance) can't yield a
    meaningful z-score, so its members get a neutral 0.0 (treated as
    "average") and fall mid-pack rather than being spuriously promoted or
    buried. With a single modality present (all prose, or all code)
    z-normalization is monotonic in cosine, so ordering is unchanged —
    the calibration only bites when both modalities compete.
    """
    if not candidates:
        return []

    code_idx: list[int] = []
    prose_idx: list[int] = []
    for i, row in enumerate(candidates):
        key = (row[4] if len(row) > 4 else "") or ""
        (code_idx if key.startswith("Code|") else prose_idx).append(i)

    calibrated = [0.0] * len(candidates)
    for bucket in (code_idx, prose_idx):
        if len(bucket) < 2:
            continue  # neutral 0.0 — can't estimate a distribution
        scores = [candidates[i][3] for i in bucket]
        mean = sum(scores) / len(scores)
        std = (sum((s - mean) ** 2 for s in scores) / len(scores)) ** 0.5
        if std <= 1e-9:
            continue  # zero variance — leave neutral
        for i in bucket:
            calibrated[i] = (candidates[i][3] - mean) / std

    order = sorted(
        range(len(candidates)),
        key=lambda i: (calibrated[i], candidates[i][3]),
        reverse=True,
    )
    return [tuple(candidates[i][:4]) for i in order]


class ContentRouter:
    """Routes content retrieval queries to content_chunk facts.

    Returns verbatim document text (scenes, sections, passages).
    No character cap — the text IS the answer.
    """

    PAIR_TYPES = ["content_chunk"]

    def __init__(self, vector_index: "VectorIndex", metadata_store: "MetadataStore") -> None:
        self._index = vector_index
        self._store = metadata_store

    async def search(
        self, customer_id: str, query_vector: np.ndarray,
        *, k: int = 5, hints: Optional[dict[str, str]] = None,
    ) -> RouterResult:
        # Prepend a provenance header (Source: Locator) to content so
        # identity queries ("where is X defined?") can name the source
        # file — the address lives in the fact's sparse key (prompt_text).
        from .reader import _provenance_header

        # When hints are present, search wider — the hint matching
        # is more precise than vector scores and may find the right
        # fact below the top-20 cutoff.
        search_k = max(k, 80) if hints and hints.get("locator_prefix") else max(k, 20)

        logger.info("content_router.ENTRY", customer_id=customer_id, k=search_k, hints=hints)

        from ..config import settings as _settings

        if _settings.enable_hybrid_rank:
            # Hybrid rank (stage 1a): widen the candidate pool, then
            # calibrate code-vs-prose so verbatim code isn't buried under
            # prose by gtr's same-language cosine bias. with_keys=True
            # returns each candidate's sparse key; _calibrate_by_subtype
            # reorders by within-bucket z-score and strips back to
            # 4-tuples (original cosine preserved). Truncate to search_k
            # so the returned size matches the non-hybrid path.
            pool_k = max(search_k, _settings.hybrid_rank_pool_size)
            pooled = await self._index.search_facts(
                customer_id=customer_id, query_vector=query_vector,
                pair_types=self.PAIR_TYPES, k=pool_k, with_keys=True,
            )
            results = _calibrate_by_subtype(pooled)[:search_k]
            code_n = sum(1 for r in pooled if (r[4] or "").startswith("Code|"))
            logger.info(
                "content_router.calibrated",
                customer_id=customer_id, pool=len(pooled),
                code=code_n, prose=len(pooled) - code_n, returned=len(results),
            )
        else:
            results = await self._index.search_facts(
                customer_id=customer_id, query_vector=query_vector,
                pair_types=self.PAIR_TYPES, k=search_k,
            )

        if not results:
            logger.info("content_router.NO_RESULTS")
            return RouterResult(router_name="content")

        # Log top 5 results
        top5_info = []
        for fid, cid, pt, score in results[:5]:
            fl = await self._store.list_facts_for_crystal(cid)
            prompt = fl[0].prompt_text[:60] if fl else "?"
            top5_info.append({"score": round(score, 4), "prompt": prompt})
        logger.info("content_router.search_results", count=len(results), top_5=top5_info)

        # Hint matching: find the fact whose prompt_text contains the locator prefix
        if hints and hints.get("locator_prefix"):
            prefix = hints["locator_prefix"].lower()
            logger.info("content_router.hint_scanning", prefix=prefix, checking=len(results))
            for fact_id, crystal_id, pair_type, score in results:
                f_list = await self._store.list_facts_for_crystal(crystal_id)
                for f in f_list:
                    if not f.prompt_text:
                        continue
                    pt_lower = f.prompt_text.lower()
                    # Match prefix as a whole locator, not as a substring.
                    # "scene 5" should match "Script|Scene 5|..." but NOT "Script|Scene 54|..."
                    # Check that the character after the prefix is a delimiter, end of string, or non-digit.
                    idx = pt_lower.find(prefix)
                    if idx >= 0:
                        end_pos = idx + len(prefix)
                        if end_pos >= len(pt_lower) or pt_lower[end_pos] in '|\n' or not pt_lower[end_pos].isdigit():
                            body = (f.claim_text or f.answer_value or "").strip()
                            header = _provenance_header(f.prompt_text)
                            injection_text = (
                                f"{header}\n{body}" if header and body else (body or None)
                            )
                            logger.info(
                                "content_router.hint_matched",
                                fact_id=f.id, crystal_id=crystal_id, score=score,
                                hint=prefix, prompt_text=f.prompt_text[:60],
                            )
                            return RouterResult(
                                router_name="content",
                                injection_text=injection_text,
                                matched_fact_ids=[f.id],
                                matched_crystal_ids=[crystal_id],
                                top_score=score, fact_count=1, voicing="informational",
                            )

        # No hint or hint didn't match — use top vector result
        top_fact_id, top_crystal_id, _, top_score = results[0]
        facts = await self._store.list_facts_for_crystal(top_crystal_id)

        if not facts:
            return RouterResult(
                router_name="content", top_score=top_score,
                matched_fact_ids=[top_fact_id], matched_crystal_ids=[top_crystal_id],
            )

        # Gate D (VS-D1): under file-grain a crystal holds MANY chunk
        # facts — inject the fact the vector search actually MATCHED,
        # not the file's first chunk. (Grain-agnostic: identical result
        # on legacy single-fact crystals.)
        fact = next((f for f in facts if f.id == top_fact_id), facts[0])
        body = (fact.claim_text or fact.answer_value or "").strip()
        header = _provenance_header(fact.prompt_text)
        injection_text = f"{header}\n{body}" if header and body else (body or None)

        logger.info(
            "content_router.matched",
            fact_id=top_fact_id, crystal_id=top_crystal_id,
            score=top_score, text_chars=len(injection_text or ""),
        )

        return RouterResult(
            router_name="content",
            injection_text=injection_text,
            matched_fact_ids=[r[0] for r in results],
            matched_crystal_ids=list(set(r[1] for r in results)),
            top_score=top_score, fact_count=len(results), voicing="informational",
        )


class KnowledgeRouter:
    """Routes knowledge lookup queries to entity/qa/relationship facts."""

    PAIR_TYPES = ["entity_attribute", "question_answer", "entity_relationship"]
    MAX_INJECTION_CHARS = 800

    def __init__(self, vector_index: "VectorIndex", metadata_store: "MetadataStore") -> None:
        self._index = vector_index
        self._store = metadata_store

    async def search(
        self, customer_id: str, query_vector: np.ndarray,
        *, k: int = 10, hints: Optional[dict[str, str]] = None,
    ) -> RouterResult:
        results = await self._index.search_facts(
            customer_id=customer_id, query_vector=query_vector,
            pair_types=self.PAIR_TYPES, k=k,
        )

        if not results:
            return RouterResult(router_name="knowledge")

        top_fact_id, top_crystal_id, _, top_score = results[0]

        lines: list[str] = []
        seen_facts: set[str] = set()
        total_chars = 0

        for fact_id, crystal_id, pair_type, score in results:
            if fact_id in seen_facts:
                continue
            seen_facts.add(fact_id)

            crystal_facts = await self._store.list_facts_for_crystal(crystal_id)
            for f in crystal_facts:
                if f.id == fact_id:
                    key = f.prompt_text or ""
                    val = f.claim_text or f.answer_value or ""
                    if key and val:
                        line = f"{key}: {val}"
                    elif val:
                        line = val
                    else:
                        continue

                    if total_chars + len(line) > self.MAX_INJECTION_CHARS:
                        break
                    lines.append(line)
                    total_chars += len(line)
                    break

            if total_chars >= self.MAX_INJECTION_CHARS:
                break

        injection_text = "\n".join(lines) if lines else None

        logger.info(
            "knowledge_router.matched",
            top_fact_id=top_fact_id, top_score=top_score,
            facts_assembled=len(lines), injection_chars=total_chars,
        )

        return RouterResult(
            router_name="knowledge",
            injection_text=injection_text,
            matched_fact_ids=[r[0] for r in results],
            matched_crystal_ids=list(set(r[1] for r in results)),
            top_score=top_score, fact_count=len(lines), voicing="advisory",
        )
