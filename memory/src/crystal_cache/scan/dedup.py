"""Dedup scan — the second Never-Idle convergence generator.

Surfaces pairs of stored facts that say the SAME THING (one is redundant
with the other), writing one `open` knowledge_conflict per genuine duplicate
with `detector="dedup_scan"`. SURFACING-ONLY (D5): it never deactivates,
merges, or deletes — that is the curation-gated step (the operator resolves a
dedup row through the same gate a contradiction uses: "Outdated" supersedes
the redundant fact, keeping one).

Why a duplicate is a `knowledge_conflict` and not a new table: a conflict row
is "two facts about the same subject that need reconciliation," and a
duplicate is exactly that — reconciled by keeping one. The B2 resolution gate
(superseded / blacklisted / qualified / dismissed) already fits, and the
admin Conflicts panel already renders the Claim A | Claim B shape. The
`detector` field is what tells an operator "these say the same thing" vs.
"these conflict."

Pipeline (per customer) — deliberately identical to the contradiction scan,
so the two share one candidate set and one idempotence keyspace:
  1. Pull the customer's OWN recent facts (list_recent_facts_for_customer,
     newest-first, D8 own-facts-only).
  2. Enumerate CANDIDATE pairs via contradiction._enumerate_candidate_pairs
     (within-crystal ∪ same-Subject-cross-crystal, capped) — REUSED so both
     generators consider the same pairs.
  3. Cheap pre-check (D4): skip any candidate whose pair_key already has a
     conflict row in ANY status — REUSED pair_key, so a pair the contradiction
     scan already wrote (CONTRADICTS) is skipped here, and a duplicate written
     here is skipped by a later contradiction scan. The two scans are thereby
     mutually exclusive on a given pair: it gets ONE conflict row, by whichever
     scan first finds a relationship, with the `detector` recording which.
  4. Discriminate the survivors with the DEDUP prompt — fast model, ~16 output
     tokens, batched under a semaphore, bounded by max_discriminator_calls.
  5. Only a DUPLICATE verdict writes an `open` conflict (the worth gate).

The DEDUP prompt explicitly carves out the contradiction case: two claims that
differ in any value are DISTINCT (a conflict, not a duplicate), so the dedup
scan does not double-flag contradictions even before the shared pre-check.

R9: this module contains NO SQL. Candidate enumeration runs in Python over
Fact objects; all reads/writes go through store methods.
"""
from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ._seam import metered_small_call
from ..llm import get_llm_client

# Reuse the contradiction scan's candidate enumeration + idempotence key +
# provenance/subject helpers VERBATIM. Sharing _pair_key is what makes the
# two scans share one keyspace (see the module docstring); sharing the
# enumeration means they consider the same candidate pairs.
from .contradiction import (
    _enumerate_candidate_pairs,
    _pair_key,
    _provenance,
    _subject_of,
)

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

# Same concurrency as the contradiction scan's batched discriminator calls.
_DISCRIMINATOR_CONCURRENCY = 8

# Symmetric same-meaning judgment. The DISTINCT example carves out the
# contradiction case (different values → DISTINCT, not DUPLICATE) so dedup
# never claims a contradiction is a duplicate; UNRELATED guards over-flagging.
DEDUP_SYSTEM = (
    "You are checking whether two stored facts about the same subject are "
    "DUPLICATES — they assert the same thing, so keeping both adds nothing.\n\n"
    "You are given CLAIM A and CLAIM B, two facts retrieved from a knowledge "
    "base.\n\n"
    "Are the two claims DUPLICATES? Examples:\n"
    "- A: 'The office opens at 9am' vs B: 'We open at nine in the morning' "
    "-> DUPLICATE\n"
    "- A: 'The contract rate is $120/hr' vs B: 'The contract rate is $95/hr' "
    "-> DISTINCT (different values — that is a conflict, not a duplicate)\n"
    "- A: 'Validate input types' vs B: 'The API accepts a data_url "
    "parameter' -> UNRELATED\n\n"
    "Two claims are DUPLICATE only when they state the SAME fact with the "
    "same meaning (wording may differ). Claims that differ in any value, "
    "scope, or detail are DISTINCT. Claims about different things are "
    "UNRELATED.\n\n"
    "Respond with ONLY one word: DUPLICATE or DISTINCT or UNRELATED"
)


@dataclass
class DedupScanResult:
    """Outcome of one dedup-scan run for one customer.

    Mirrors contradiction.ScanResult; `duplicates_found` is the dedup-specific
    name for "open rows written" (rows carry detector='dedup_scan')."""

    customer_id: str
    facts_scanned: int
    candidate_pairs: int
    pairs_evaluated: int      # discriminator calls actually spent this run
    skipped_existing: int     # candidates skipped (pair_key already recorded)
    duplicates_found: int     # DUPLICATE verdicts → open rows written
    budget_exhausted: bool    # True if the call budget capped the run early


# ---------------------------------------------------------------------------
# Discriminator (one fast-model call per pair)
# ---------------------------------------------------------------------------

async def _discriminate_dup(
    client: Any, claim_a: str, claim_b: str, *, customer_id: str,
    store: Any = None, log: Any,
) -> str:
    """Return DUPLICATE | DISTINCT | UNRELATED | UNKNOWN | ERROR.

    Runs the seam call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row).
    Fail-safe: any exception returns "ERROR"
    (treated as no-duplicate — never writes a row)."""
    user = f"CLAIM A: {claim_a}\n\nCLAIM B: {claim_b}"
    try:
        raw_text = await metered_small_call(
            client,
            customer_id=customer_id,
            origin="scan_dedup",
            system=DEDUP_SYSTEM,
            user=user,
            max_tokens=16,
            store=store,
        )
        raw = (raw_text or "").strip().upper()
        if raw.startswith("DUPLICATE"):
            return "DUPLICATE"
        if raw.startswith("DISTINCT"):
            return "DISTINCT"
        if raw.startswith("UNRELATED"):
            return "UNRELATED"
        return "UNKNOWN"
    except Exception as e:  # fail-safe — a discriminator error never writes
        log.warning(
            "dedup_scan.discriminator_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return "ERROR"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def scan_for_duplicates(
    *,
    store: "MetadataStore",
    slm_client: Any = None,
    customer_id: str,
    max_candidate_pairs: int = 200,
    max_discriminator_calls: int = 50,
    fact_fetch_limit: Optional[int] = None,
    log: Any = None,
) -> DedupScanResult:
    """Scan one customer's own facts for duplicates, surfacing-only.

    Args mirror scan_for_contradictions. slm_client is an optional client
    override exposing `complete` (tests); None → the provider-neutral seam,
    and a no-op when neither an override nor a ready seam exists. DUPLICATE
    verdicts write `open` conflicts with detector='dedup_scan'.
    """
    log = log or logger
    if slm_client is None and not get_llm_client().is_ready():
        return DedupScanResult(customer_id, 0, 0, 0, 0, 0, False)
    client = slm_client if slm_client is not None else get_llm_client()

    facts = await store.list_recent_facts_for_customer(
        customer_id, limit=fact_fetch_limit
    )
    candidates = _enumerate_candidate_pairs(facts, max_candidate_pairs)

    # Cheap pre-filter (D4): drop candidates already recorded (any status,
    # any detector — shared keyspace), no LLM call. Stop selecting once the
    # call budget is reached.
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

    # Batched discrimination (worth gate = DUPLICATE).
    sem = asyncio.Semaphore(_DISCRIMINATOR_CONCURRENCY)

    async def _judge(
        a: "Fact", b: "Fact", pair_key: str
    ) -> tuple["Fact", "Fact", str, str]:
        async with sem:
            verdict = await _discriminate_dup(
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

    duplicates_found = 0
    for (a, b, pair_key, verdict) in judged:
        if verdict != "DUPLICATE":
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
            detector="dedup_scan",
        )
        duplicates_found += 1
        log.info(
            "dedup_scan.duplicate_found",
            customer_id=customer_id,
            fact_a=a.id,
            fact_b=b.id,
            subject=subject,
        )

    result = DedupScanResult(
        customer_id=customer_id,
        facts_scanned=len(facts),
        candidate_pairs=len(candidates),
        pairs_evaluated=len(to_check),
        skipped_existing=skipped_existing,
        duplicates_found=duplicates_found,
        budget_exhausted=budget_exhausted,
    )
    # Log-noise fix (2026-07-08): found-nothing cycles are debug; info is
    # reserved for cycles that found something or hit the scan budget.
    _log_fn = (
        log.info
        if (result.duplicates_found or result.budget_exhausted)
        else log.debug
    )
    _log_fn("dedup_scan.completed", **dataclasses.asdict(result))
    return result
