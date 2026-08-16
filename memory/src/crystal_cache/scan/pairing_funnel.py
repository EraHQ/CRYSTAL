"""Pairing funnel — the crystal_edges writer (Assumptions funnel F1).

Ratified 2026-08-05/06: Q4=B strict priority tiers with structural
backfill, Q6=A scores persist as crystal_edges rows — this module is
that empty table's FIRST WRITER since v3. Anthony's design principle,
verbatim: "use the data we are saving... let's not waste it."

Every tier emits edges from data the bank already records; NOTHING
here calls a model. The verdict spender (run_assumptions_scan, F2)
reads the graph in tier order and spends its bounded small-tier
budget on the top of it.

Tiers, strongest first (the edge_type vocabulary):

  co_cited      — two grounded citations in ONE answer turn
                  (citations GROUP BY query_log_id). The model already
                  combined these crystals in a real answer.
  co_routed     — two crystals a single conversation touched
                  (query_logs GROUP BY sequence_id over
                  routed_crystal_id + matched_facts->crystal).
  chained       — an authored chain edge joins them (existing input).
  gap_subject   — an open gap's sparse-key Subject spans them
                  (existing input).
  key_adjacent  — structural: two crystals share a RARE sparse-key
                  segment (C3 Q1=A, 2026-08-11 — any position, any key
                  shape, namespace-scale segments excluded via the same
                  fraction discipline as the sibling scans; supersedes
                  the retired positional "Source = parts[0]" parse,
                  which under unified wide-leftmost keys grouped at
                  domain scale). Locator numeric adjacency stays
                  deferred.
  vector_similar— structural: routing_vector cosine >= 0.5 (stored
                  vectors, numpy — no encoder in the worker). Crystals
                  without a routing_vector (pre-6.3) skip this tier.

Weight semantics: demand tiers ACCUMULATE event counts across passes
(watermarks prevent double-counting); chained/gap/key edges carry
existence weight 1.0 per source observation; vector_similar carries
the cosine itself. last_reinforced_at bumps on every re-observation.

DEFERRED TIER on record (2026-08-06): per-tool-use pairing from
agent_events — tool-event payloads carry truncated input summaries /
output heads for the live ticker, not structured crystal ids;
citations already capture the agent lane's GROUNDED usage, which is
the same signal at higher quality. Revisit if the terminal
tool_calls log grows a structured retrieval record.

Assumption-typed crystals are excluded from every tier — speculation
never becomes pairing input (the slice-1 rule, funnel-wide).

Watermarks + the structural rotation offset are PROCESS-LOCAL,
owned by the worker (the rotation-offset posture on record): a
restart costs one redundant rescore pass, which the accumulate-with-
watermark upserts absorb as a bounded overcount of zero (nothing is
re-read below the watermark) plus one structural re-slice.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import get_settings
from .segments import exclusion_threshold, has_segment, segments_of

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# Bounded fetches per pass per customer (module consts, not knobs —
# they bound memory, not behavior).
_CITATIONS_FETCH_LIMIT = 2000
_QUERY_LOGS_FETCH_LIMIT = 2000
_FACTS_FETCH_LIMIT = 2000
_CRYSTALS_FETCH_LIMIT = 300
_GAPS_FETCH_LIMIT = 100

# vector_similar emission floor. A const, not a knob (R6): no evidence
# yet says tenants need to tune it.
_VECTOR_SIM_THRESHOLD = 0.5

# The funnel's edge_type vocabulary, strongest first — F2's read order.
EDGE_TIER_ORDER: tuple[str, ...] = (
    "co_cited",
    "co_routed",
    "chained",
    "gap_subject",
    "key_adjacent",
    "vector_similar",
)


@dataclass
class FunnelState:
    """Process-local per-customer funnel memory (worker-owned)."""

    # Per-signal high-water marks (created_at of the newest row seen).
    citations_since: Optional[datetime] = None
    query_logs_since: Optional[datetime] = None
    # Structural rotation: index into the canonical pair enumeration.
    structural_offset: int = 0


@dataclass
class FunnelResult:
    customer_id: str
    co_cited_edges: int = 0
    co_routed_edges: int = 0
    chained_edges: int = 0
    gap_subject_edges: int = 0
    key_adjacent_edges: int = 0
    vector_similar_edges: int = 0
    structural_pairs_examined: int = 0

    @property
    def total_edges(self) -> int:
        return (
            self.co_cited_edges + self.co_routed_edges
            + self.chained_edges + self.gap_subject_edges
            + self.key_adjacent_edges + self.vector_similar_edges
        )


def _canonical(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a <= b else (b, a)


def _pairs_from_group(crystal_ids: set[str]) -> set[tuple[str, str]]:
    ids = sorted(crystal_ids)
    return {
        (ids[i], ids[j])
        for i in range(len(ids))
        for j in range(i + 1, len(ids))
    }


def _cosine(u: list[float], v: list[float]) -> float:
    nu = math.sqrt(sum(x * x for x in u))
    nv = math.sqrt(sum(x * x for x in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    dot = sum(x * y for x, y in zip(u, v))
    return dot / (nu * nv)


async def run_pairing_funnel(
    *,
    store: "MetadataStore",
    customer_id: str,
    state: FunnelState,
    structural_pairs_limit: Optional[int] = None,
    log: Any = None,
) -> FunnelResult:
    """One funnel pass for one customer: aggregate every usage signal
    past its watermark, emit/reinforce crystal_edges, advance the
    structural rotation. Store reads + edge writes only — no model
    calls, so callers may run it outside the spend gates.
    """
    log = log or logger
    settings = get_settings()
    structural_pairs_limit = (
        settings.assumptions_structural_pairs_per_cycle
        if structural_pairs_limit is None else structural_pairs_limit
    )
    result = FunnelResult(customer_id=customer_id)

    # ---- shared substrate: crystals (ids, types, vectors) + facts ----
    crystals = await store.list_crystal_pairing_info(
        customer_id, limit=_CRYSTALS_FETCH_LIMIT,
    )
    non_assumption_ids = {
        c["id"] for c in crystals if c["crystal_type"] != "assumption"
    }
    if len(non_assumption_ids) < 2:
        return result

    facts = await store.list_recent_facts_for_customer(
        customer_id, limit=_FACTS_FETCH_LIMIT,
    )
    fact_to_crystal = {
        f.id: f.crystal_id
        for f in facts
        if f.crystal_id in non_assumption_ids
    }

    edges: dict[tuple[str, str, str], float] = {}

    def emit(a: str, b: str, edge_type: str, weight: float) -> bool:
        """True only when the edge was actually recorded — tier
        counters key off this so a filtered pair (assumption endpoint,
        self-pair) never inflates a count."""
        if a == b:
            return False
        if a not in non_assumption_ids or b not in non_assumption_ids:
            return False
        ca, cb = _canonical(a, b)
        key = (ca, cb, edge_type)
        edges[key] = edges.get(key, 0.0) + weight
        return True

    # ---- Tier 1: co_cited ------------------------------------------------
    citations = await store.list_grounded_citations_since(
        customer_id,
        since=state.citations_since,
        limit=_CITATIONS_FETCH_LIMIT,
    )
    by_turn: dict[str, set[str]] = {}
    for c in citations:
        if c["query_log_id"]:
            by_turn.setdefault(c["query_log_id"], set()).add(
                c["crystal_id"]
            )
        if state.citations_since is None or (
            c["created_at"] > state.citations_since
        ):
            state.citations_since = c["created_at"]
    for crystal_ids in by_turn.values():
        for a, b in _pairs_from_group(crystal_ids):
            if emit(a, b, "co_cited", 1.0):
                result.co_cited_edges += 1

    # ---- Tier 2: co_routed -----------------------------------------------
    routings = await store.list_query_routings_since(
        customer_id,
        since=state.query_logs_since,
        limit=_QUERY_LOGS_FETCH_LIMIT,
    )
    by_sequence: dict[str, set[str]] = {}
    for r in routings:
        seq = r["sequence_id"]
        if not seq:
            continue
        touched = by_sequence.setdefault(seq, set())
        if r["routed_crystal_id"]:
            touched.add(r["routed_crystal_id"])
        for fact_id in (r["matched_facts"] or []):
            crystal_id = fact_to_crystal.get(fact_id)
            if crystal_id:
                touched.add(crystal_id)
        if state.query_logs_since is None or (
            r["timestamp"] > state.query_logs_since
        ):
            state.query_logs_since = r["timestamp"]
    for crystal_ids in by_sequence.values():
        for a, b in _pairs_from_group(crystal_ids):
            if emit(a, b, "co_routed", 1.0):
                result.co_routed_edges += 1

    # ---- Tier 3: chained (existing input, now an edge source) -----------
    chained = await store.list_chained_crystal_pairs(
        customer_id, limit=_CRYSTALS_FETCH_LIMIT,
    )
    for a, b in chained:
        if emit(a, b, "chained", 1.0):
            result.chained_edges += 1

    # ---- Tier 4: gap_subject (existing grouping, now an edge source) ----
    gaps = await store.list_knowledge_gaps(
        customer_id, status="open", limit=_GAPS_FETCH_LIMIT,
    )
    subjects = {
        (g.subject or "").strip()
        for g in gaps
        if (g.subject or "").strip()
    }
    if subjects:
        # C3 Q1=A: a gap subject matches any-position segments.
        by_subject: dict[str, set[str]] = {}
        for f in facts:
            if f.crystal_id not in non_assumption_ids:
                continue
            for subject in subjects:
                if has_segment(f.prompt_text, subject):
                    by_subject.setdefault(subject, set()).add(f.crystal_id)
        for crystal_ids in by_subject.values():
            for a, b in _pairs_from_group(crystal_ids):
                if emit(a, b, "gap_subject", 1.0):
                    result.gap_subject_edges += 1

    # ---- Tier 5: structural backfill (rotating bounded slice) -----------
    # Canonical pair enumeration over id-sorted crystals; the offset
    # walks the full pair space across cycles (Q4=B backfill — coverage
    # of EVERYTHING, eventually, when the tenant's explore toggle is on;
    # F3 wires the toggle, F2 enforces it at spend time — emitting the
    # structural edges is free and toggle-independent).
    # C3 Q1=A: crystal-level RARE-segment sets. A segment carried by
    # more than the exclusion threshold of crystals is namespace-scale
    # noise, not adjacency (same fraction knob as the sibling scans,
    # counted over crystals here).
    by_crystal_source: dict[str, set[str]] = {}
    _seg_crystal_count: dict[str, int] = {}
    for f in facts:
        if f.crystal_id not in non_assumption_ids:
            continue
        for seg in segments_of(f.prompt_text):
            low = seg.lower()
            crystal_segs = by_crystal_source.setdefault(f.crystal_id, set())
            if low not in crystal_segs:
                crystal_segs.add(low)
                _seg_crystal_count[low] = _seg_crystal_count.get(low, 0) + 1
    _seg_cap = exclusion_threshold(
        len(non_assumption_ids),
        get_settings().scan_segment_max_group_fraction,
    )
    _too_common = {s for s, n in _seg_crystal_count.items() if n > _seg_cap}
    if _too_common:
        for _segs in by_crystal_source.values():
            _segs -= _too_common
    vectors = {
        c["id"]: c["routing_vector"]
        for c in crystals
        if c["crystal_type"] != "assumption" and c["routing_vector"]
    }

    ids = sorted(non_assumption_ids)
    n = len(ids)
    total_pairs = n * (n - 1) // 2
    if total_pairs and structural_pairs_limit > 0:
        offset = state.structural_offset % total_pairs
        examined = 0
        idx = offset
        while examined < min(structural_pairs_limit, total_pairs):
            # Unrank pair index -> (i, j), i < j (row-major over the
            # upper triangle).
            i, remaining = 0, idx
            row = n - 1
            while remaining >= row:
                remaining -= row
                i += 1
                row -= 1
            j = i + 1 + remaining
            a, b = ids[i], ids[j]

            shared_sources = (
                by_crystal_source.get(a, set())
                & by_crystal_source.get(b, set())
            )
            if shared_sources:
                if emit(a, b, "key_adjacent", 1.0):
                    result.key_adjacent_edges += 1

            va, vb = vectors.get(a), vectors.get(b)
            if va and vb:
                sim = _cosine(va, vb)
                if sim >= _VECTOR_SIM_THRESHOLD:
                    if emit(a, b, "vector_similar", sim):
                        result.vector_similar_edges += 1

            examined += 1
            idx = (idx + 1) % total_pairs
        state.structural_offset = idx
        result.structural_pairs_examined = examined

    # ---- persist ---------------------------------------------------------
    if edges:
        await store.upsert_crystal_edges([
            (a, b, edge_type, weight)
            for (a, b, edge_type), weight in edges.items()
        ])

    _log_fn = log.info if result.total_edges else log.debug
    _log_fn(
        "pairing_funnel.completed",
        customer_id=customer_id,
        co_cited=result.co_cited_edges,
        co_routed=result.co_routed_edges,
        chained=result.chained_edges,
        gap_subject=result.gap_subject_edges,
        key_adjacent=result.key_adjacent_edges,
        vector_similar=result.vector_similar_edges,
        structural_examined=result.structural_pairs_examined,
    )
    return result
