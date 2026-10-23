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
    _learned,
    _memoized_keys,
    _pair_key,
    _provenance,
    _remember_verdicts,
    claims_block,
    explain_contradiction,
    fact_details_for,
    judge_context,
)
from .segments import most_specific_shared_segment, widest_segment_of

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

# Same concurrency as the contradiction scan's batched discriminator calls.
_DISCRIMINATOR_CONCURRENCY = 8

DETECTOR = "dedup_scan"
# Verdicts remembered in scan_verdicts: judged, not a duplicate.
_MEMO_VERDICTS = frozenset({"DISTINCT", "UNRELATED"})

# 2026-10-21 (Anthony: "just like in assumptions I think duplicate
# options need some explanation"): a found duplicate gets the same
# second call a contradiction gets, with the duplicate's own meaning of
# the four verbs. Shares contradiction.EXPLAIN_SCHEMA / parse.
EXPLAIN_DUP_SYSTEM = (
    "Two stored facts were judged to say the same thing. Explain to the "
    "person who owns this memory, in one or two plain sentences, what "
    "both facts state and what, if anything, one of them carries that "
    "the other lacks, then recommend what to do.\n\n"
    "Resolutions:\n"
    "- superseded: one copy is redundant; keep the other (name the loser: "
    "the copy that carries less, or the older one when they are equal).\n"
    "- qualified: each carries a detail the other lacks; keep both.\n"
    "- dismissed: they are not the same fact after all.\n"
    "- blacklisted: one copy is wrong, not merely redundant; drop it "
    "(name the loser).\n\n"
    "Prefer superseded when one copy says everything the other says; "
    "prefer qualified when each adds something. Name the losing side as "
    "'a' or 'b', or 'none' for qualified and dismissed. Keep the reason "
    "to one sentence."
)

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
    "Each claim may come with where it is filed (its key path, general to "
    "specific), the source it was learned from, and when. Two claims are "
    "the same fact only in the same scope: when the key paths or sources "
    "show different products, companies, policies, people, places or "
    "periods, answer DISTINCT even if the words look alike.\n\n"
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
    memoized: int = 0         # 2026-10-21: DISTINCT/UNRELATED verdicts remembered in scan_verdicts
    explained: int = 0        # 2026-10-21: duplicates that also got reasoning + a recommendation


# ---------------------------------------------------------------------------
# Discriminator (one fast-model call per pair)
# ---------------------------------------------------------------------------

async def _discriminate_dup(
    client: Any, claim_a: str, claim_b: str, *, customer_id: str,
    store: Any = None, log: Any, context_a: str = "", context_b: str = "",
) -> str:
    """Return DUPLICATE | DISTINCT | UNRELATED | UNKNOWN | ERROR.

    Runs the seam call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row).
    Fail-safe: any exception returns "ERROR"
    (treated as no-duplicate — never writes a row). Contexts as in
    contradiction._discriminate (key path, source, date per claim)."""
    user = claims_block(claim_a, claim_b, context_a, context_b)
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
    # CF-Q1=A (2026-07-23): content chunks exit the dedup scan
    # entirely. A chunk duplicates its own extractions BY DESIGN
    # (that's what extraction is), and cross-file duplicate ingestion
    # is already caught by content-hash at the door — dedup governs
    # knowledge claims only. Without this, one ingested table fills
    # the candidate budget with fact-vs-source pairs.
    facts = [f for f in facts if (f.pair_type or "") != "content_chunk"]
    candidates = _enumerate_candidate_pairs(facts, max_candidate_pairs)

    # Cheap pre-filter (D4): drop candidates already recorded (a conflict
    # row in any status, any detector — shared keyspace — or a
    # scan_verdicts memo for THIS detector, 2026-10-21), no LLM call. Stop
    # selecting once the call budget is reached.
    keyed = [(a, b, _pair_key(a, b)) for (a, b) in candidates]
    memo = await _memoized_keys(
        store, customer_id, detector=DETECTOR, pair_keys=[k for (_, _, k) in keyed],
    )
    to_check: list[tuple["Fact", "Fact", str]] = []
    skipped_existing = 0
    budget_exhausted = False
    for (a, b, pair_key) in keyed:
        if len(to_check) >= max_discriminator_calls:
            budget_exhausted = True
            break
        if pair_key in memo:
            skipped_existing += 1
            continue
        if await store.knowledge_conflict_exists(customer_id, pair_key=pair_key):
            skipped_existing += 1
            continue
        to_check.append((a, b, pair_key))

    # Batched discrimination (worth gate = DUPLICATE), each side with its
    # key path, source and date.
    sem = asyncio.Semaphore(_DISCRIMINATOR_CONCURRENCY)
    details = await fact_details_for(
        store, [f for pair in to_check for f in pair[:2]], log=log,
    )

    async def _judge(
        a: "Fact", b: "Fact", pair_key: str
    ) -> tuple["Fact", "Fact", str, str]:
        async with sem:
            verdict = await _discriminate_dup(
                client, a.claim_text or "", b.claim_text or "",
                customer_id=customer_id, store=store,
                log=log,
                context_a=judge_context(a, details),
                context_b=judge_context(b, details),
            )
        return (a, b, pair_key, verdict)

    judged = (
        await asyncio.gather(*[_judge(a, b, pk) for (a, b, pk) in to_check])
        if to_check
        else []
    )

    memoized = await _remember_verdicts(
        store, customer_id, detector=DETECTOR,
        verdicts=[(pk, v) for (_, _, pk, v) in judged if v in _MEMO_VERDICTS],
        log=log,
    )

    duplicates_found = 0
    explained = 0
    for (a, b, pair_key, verdict) in judged:
        if verdict != "DUPLICATE":
            continue
        subject = (
            most_specific_shared_segment(a.prompt_text, b.prompt_text)
            or widest_segment_of(a.prompt_text)
            or widest_segment_of(b.prompt_text)
        )
        explanation = await explain_contradiction(
            client,
            customer_id=customer_id,
            claim_a=a.claim_text or "",
            claim_b=b.claim_text or "",
            learned_a=_learned(a),
            learned_b=_learned(b),
            source_a=_provenance(a),
            source_b=_provenance(b),
            store=store,
            log=log,
            system=EXPLAIN_DUP_SYSTEM,
            origin="scan_dedup",
        )
        if explanation is not None:
            explained += 1
        conflict = await store.create_knowledge_conflict(
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
            reasoning=explanation["why"] if explanation else None,
            recommendation=(
                {k: explanation[k] for k in ("resolution", "loser", "reason")}
                if explanation else None
            ),
        )
        duplicates_found += 1
        # C3 Q2=A (2026-08-11): every conflict the scans open gets a
        # witness in the curation activity feed — dedup rows are
        # knowledge_conflicts too. Best-effort by contract.
        try:
            await store.record_curation_event(
                customer_id,
                event_type="conflict_found",
                subject_id=conflict.id,
                label=(
                    f"Duplicate found - {subject}"[:256]
                    if subject else "Duplicate found"
                ),
                payload={
                    "detector": "dedup_scan",
                    "fact_a": a.id, "fact_b": b.id,
                },
            )
        except Exception:  # noqa: BLE001 — witness never breaks the scan
            log.debug("curation_event.emit_failed", exc_info=True)
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
        memoized=memoized,
        explained=explained,
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
