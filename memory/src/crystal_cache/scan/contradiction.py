"""Contradiction scan — the first Never-Idle convergence generator.

Surfaces pairs of stored facts that contradict each other, writing one
`open` knowledge_conflict per genuine contradiction. SURFACING-ONLY (D5):
it never blacklists, supersedes, qualifies, or deletes — that is a later,
curation-gated step. This is the convergence half of the accommodation
thesis: the bank doesn't just accumulate, it continuously notices when two
things it holds can't both be true.

Pipeline (per customer):
  1. Pull the customer's OWN recent facts (store.list_recent_facts_for_customer
     — D8 own-facts-only, newest first for the recency cap).
  2. Enumerate CANDIDATE pairs, bounded (D2): facts WITHIN the same crystal
     (bonding already clustered same-topic facts) UNION facts ACROSS crystals
     that share the same sparse-key Subject segment — capped at
     `max_candidate_pairs`. No extra vector search in v1.
  3. Cheap pre-check (D4): skip any candidate whose `pair_key` already has a
     conflict row in ANY status — no LLM call. This is what makes a re-scan
     over an unchanged bank a no-op and keeps dismissed conflicts from
     re-surfacing.
  4. Discriminate the survivors with the ported v1 CROSS_VALIDATE prompt
     (D3) — fast model, ~16 output tokens, batched under a semaphore. Bounded
     by `max_discriminator_calls` (the budget gate).
  5. Only a CONTRADICTS verdict writes an `open` conflict (the worth gate).

The discriminator prompt is a faithful port of v1's `CROSS_VALIDATE_SYSTEM`
(crystal-cache-v1/scripts/run_level_f.py), generalized from "old rule vs new
knowledge" to "two stored facts about the same subject," with an explicit
guard against over-flagging unrelated facts.

R9: this module contains NO SQL. Candidate enumeration runs in Python over
Fact objects returned by the store; all reads/writes go through store methods.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ._seam import metered_small_call
from ..llm import get_llm_client

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

# Concurrency for the batched discriminator calls (v1 used 10 for the same
# cross-validation pass; 8 here leaves headroom under the live API).
_DISCRIMINATOR_CONCURRENCY = 8

# Faithful port of v1's CROSS_VALIDATE_SYSTEM, generalized to a symmetric
# fact-vs-fact judgment. Same three-way contract + few-shot shape; the added
# line guards against the over-flagging failure mode (different things read as
# CONTRADICTS) that a symmetric framing invites.
CROSS_VALIDATE_SYSTEM = (
    "You are checking whether two stored facts about the same subject "
    "contradict each other.\n\n"
    "You are given CLAIM A and CLAIM B, two facts retrieved from a knowledge "
    "base.\n\n"
    "Do the two claims CONTRADICT each other? Examples:\n"
    "- A: 'The contract rate is $120/hr' vs B: 'The contract rate is $95/hr' "
    "-> CONTRADICTS\n"
    "- A: 'The office opens at 9am' vs B: 'Staff arrive by nine in the "
    "morning' -> CONSISTENT\n"
    "- A: 'Validate input types' vs B: 'The API accepts a data_url "
    "parameter' -> UNRELATED\n\n"
    "A genuine contradiction means both claims cannot be true at once for the "
    "same entity and time. Facts about different things are UNRELATED, not "
    "CONTRADICTS.\n\n"
    "Respond with ONLY one word: CONTRADICTS or CONSISTENT or UNRELATED"
)


@dataclass
class ScanResult:
    """Outcome of one contradiction-scan run for one customer."""

    customer_id: str
    facts_scanned: int
    candidate_pairs: int
    pairs_evaluated: int      # discriminator calls actually spent this run
    skipped_existing: int     # candidates skipped (pair_key already recorded)
    conflicts_found: int      # CONTRADICTS verdicts → open rows written
    budget_exhausted: bool    # True if the call budget capped the run early


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _subject_of(fact: "Fact") -> Optional[str]:
    """The sparse-key Subject segment of a fact, or None.

    Sparse key format is `Source | Locator | Subject | Domain`
    (docs/UNIFIED_SPARSE_KEY.md). Facts whose prompt_text isn't a
    pipe-delimited key (free-text prompts) have no Subject and only
    participate in within-crystal candidate pairs.
    """
    key = (fact.prompt_text or "").strip()
    if "|" not in key:
        return None
    parts = [p.strip() for p in key.split("|")]
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    return None


def _provenance(fact: "Fact") -> str:
    """Human-facing provenance string for one side of a conflict."""
    source_kind = (fact.source_kind or "").strip() or "unknown"
    if fact.source_doc_id:
        return f"{source_kind} @ {fact.source_doc_id}"
    return source_kind


def _pair_key(a: "Fact", b: "Fact") -> str:
    """Idempotence key (D4): a sha256 over the SORTED (fact-id, claim) pairs.

    Sorting by fact id makes the key order-independent (a,b == b,a). Folding
    in each claim's text means a fact whose claim CHANGED yields a different
    key — so the changed pair is re-evaluated rather than silently skipped,
    while an unchanged pair stays stable and is skipped on re-scan. 64 hex
    chars, well under the column's 128.
    """
    items = sorted(
        [(a.id, a.claim_text or ""), (b.id, b.claim_text or "")],
        key=lambda t: t[0],
    )
    raw = "||".join(f"{fid}\u241f{claim}" for fid, claim in items)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _enumerate_candidate_pairs(
    facts: list["Fact"], max_pairs: int
) -> list[tuple["Fact", "Fact"]]:
    """Bounded candidate pairs (D2): within-crystal ∪ same-Subject-cross-crystal.

    `facts` arrives newest-first, so iterating in order prioritizes recent
    facts. Pairs are de-duplicated (a within-crystal pair is never re-counted
    as a cross-crystal one) and capped at `max_pairs`.

    v1 limitation (documented): the cap is a simple total. A single very large
    crystal could fill the cap with its own within-crystal pairs before
    cross-crystal candidates are reached. Round-robin/interleaved selection is
    a P5 refinement; the cap is the bound that matters for cost in v1.
    """
    usable = [f for f in facts if (f.claim_text or "").strip()]

    by_crystal: dict[str, list["Fact"]] = {}
    by_subject: dict[str, list["Fact"]] = {}
    for f in usable:
        by_crystal.setdefault(f.crystal_id, []).append(f)
        subject = _subject_of(f)
        if subject:
            by_subject.setdefault(subject, []).append(f)

    seen: set[frozenset[str]] = set()
    pairs: list[tuple["Fact", "Fact"]] = []

    def _add(a: "Fact", b: "Fact") -> bool:
        """Add a pair if new; return True when the cap has been reached."""
        if a.id == b.id:
            return False
        key = frozenset((a.id, b.id))
        if key in seen:
            return False
        seen.add(key)
        pairs.append((a, b))
        return len(pairs) >= max_pairs

    # (a) within-crystal pairs
    capped = False
    for group in by_crystal.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if _add(group[i], group[j]):
                    capped = True
                    break
            if capped:
                break
        if capped:
            break

    # (b) same-Subject pairs across DIFFERENT crystals
    if not capped:
        for group in by_subject.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if group[i].crystal_id == group[j].crystal_id:
                        continue  # already covered by within-crystal
                    if _add(group[i], group[j]):
                        capped = True
                        break
                if capped:
                    break
            if capped:
                break

    return pairs


# ---------------------------------------------------------------------------
# Discriminator (one fast-model call per pair)
# ---------------------------------------------------------------------------

async def _discriminate(
    client: Any, claim_a: str, claim_b: str, *, customer_id: str,
    store: Any = None, log: Any,
) -> str:
    """Return CONTRADICTS | CONSISTENT | UNRELATED | UNKNOWN | ERROR.

    Runs the seam call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row;
    Core Principle #1 preserved). Fail-safe: any exception returns
    "ERROR" (treated as no-contradiction — never writes a row).
    """
    user = f"CLAIM A: {claim_a}\n\nCLAIM B: {claim_b}"
    try:
        raw_text = await metered_small_call(
            client,
            customer_id=customer_id,
            origin="scan_contradiction",
            system=CROSS_VALIDATE_SYSTEM,
            user=user,
            max_tokens=16,
            store=store,
        )
        raw = (raw_text or "").strip().upper()
        if raw.startswith("CONTRADICTS"):
            return "CONTRADICTS"
        if raw.startswith("CONSISTENT"):
            return "CONSISTENT"
        if raw.startswith("UNRELATED"):
            return "UNRELATED"
        return "UNKNOWN"
    except Exception as e:  # fail-safe — a discriminator error never writes
        log.warning(
            "contradiction_scan.discriminator_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return "ERROR"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def scan_for_contradictions(
    *,
    store: "MetadataStore",
    slm_client: Any = None,
    customer_id: str,
    max_candidate_pairs: int = 200,
    max_discriminator_calls: int = 50,
    fact_fetch_limit: Optional[int] = None,
    log: Any = None,
) -> ScanResult:
    """Scan one customer's own facts for contradictions, surfacing-only.

    Args:
        store: MetadataStore (provides the enumeration read + conflict CRUD).
        slm_client: optional client override exposing `complete` (tests).
            None → the provider-neutral seam; a no-op ScanResult when neither
            an override nor a ready seam exists.
        customer_id: the tenant to scan (own facts only, D8).
        max_candidate_pairs: D2 cap on enumerated candidate pairs.
        max_discriminator_calls: the budget gate — at most this many NEW
            (not-already-recorded) pairs are sent to the discriminator.
        fact_fetch_limit: optional cap on facts pulled from the store (bounds
            enumeration on a huge bank). None = all own facts.
        log: structlog-style logger; defaults to the module logger.

    Returns:
        ScanResult with the run's counters.
    """
    log = log or logger
    if slm_client is None and not get_llm_client().is_ready():
        return ScanResult(customer_id, 0, 0, 0, 0, 0, False)
    client = slm_client if slm_client is not None else get_llm_client()

    facts = await store.list_recent_facts_for_customer(
        customer_id, limit=fact_fetch_limit
    )
    candidates = _enumerate_candidate_pairs(facts, max_candidate_pairs)

    # Cheap pre-filter (D4): drop candidates already recorded (any status),
    # no LLM call. Stop selecting once the call budget is reached.
    to_check: list[tuple["Fact", "Fact", str]] = []
    skipped_existing = 0
    budget_exhausted = False
    for (a, b) in candidates:
        if len(to_check) >= max_discriminator_calls:
            budget_exhausted = True
            break
        pair_key = _pair_key(a, b)
        if await store.knowledge_conflict_exists(customer_id, pair_key=pair_key):
            skipped_existing += 1
            continue
        to_check.append((a, b, pair_key))

    # Batched discrimination (worth gate = CONTRADICTS).
    sem = asyncio.Semaphore(_DISCRIMINATOR_CONCURRENCY)

    async def _judge(
        a: "Fact", b: "Fact", pair_key: str
    ) -> tuple["Fact", "Fact", str, str]:
        async with sem:
            verdict = await _discriminate(
                client, a.claim_text or "", b.claim_text or "",
                customer_id=customer_id, store=store,
                log=log,
            )
        return (a, b, pair_key, verdict)

    judged = (
        await asyncio.gather(*[_judge(a, b, pk) for (a, b, pk) in to_check])
        if to_check
        else []
    )

    conflicts_found = 0
    for (a, b, pair_key, verdict) in judged:
        if verdict != "CONTRADICTS":
            continue
        subject = _subject_of(a) or _subject_of(b)
        await store.create_knowledge_conflict(
            customer_id,
            fact_a_id=a.id,
            fact_b_id=b.id,
            claim_a=a.claim_text or "",
            claim_b=b.claim_text or "",
            pair_key=pair_key,
            crystal_a_id=a.crystal_id,
            crystal_b_id=b.crystal_id,
            subject=subject,
            provenance_a=_provenance(a),
            provenance_b=_provenance(b),
        )
        conflicts_found += 1
        log.info(
            "contradiction_scan.conflict_found",
            customer_id=customer_id,
            fact_a=a.id,
            fact_b=b.id,
            subject=subject,
        )

    result = ScanResult(
        customer_id=customer_id,
        facts_scanned=len(facts),
        candidate_pairs=len(candidates),
        pairs_evaluated=len(to_check),
        skipped_existing=skipped_existing,
        conflicts_found=conflicts_found,
        budget_exhausted=budget_exhausted,
    )
    log.info("contradiction_scan.completed", **dataclasses.asdict(result))
    return result
