"""Gap-discovery scan — the third Never-Idle convergence generator.

Surfaces PROACTIVE knowledge gaps: where the reactive gap path records "we
lacked X" when an answer fell short during a chat, this scans the bank when
idle and asks, per subject, "what important question do these facts NOT
answer?" A named gap writes a `knowledge_gap` with `source="gap_discovery"`,
which the existing Phase-2 `_fill_open_gaps` sweep can then try to fill (or an
operator can act on from the Cognition surface). SURFACING-ONLY (D5): it only
ever creates an `open` gap.

Pipeline (per customer):
  1. Pull the customer's OWN recent facts (list_recent_facts_for_customer,
     newest-first, D8 own-facts-only).
  2. Group by sparse-key Subject (contradiction._subject_of, REUSED). Only
     subjects with >= min_facts_per_subject facts are candidates — a subject
     the model can actually reason about. Free-text facts (no Subject) don't
     participate. Subjects are visited newest-first (recency bias, like the
     pairwise scans).
  3. Idempotence: skip any subject that already has an `open` gap with
     source='gap_discovery' (fetch the open gaps once, build the skip-set) —
     so a re-scan over an unchanged bank doesn't pile duplicate discovered
     gaps onto the same subject.
  4. Ask the fast model, per surviving subject (batched under a semaphore,
     bounded by max_subjects — the budget gate), for ONE important unanswered
     question, with a conservative NONE escape.
  5. Only a non-NONE answer writes an `open` gap (the worth gate).

Noise control: the worth gate is (a) subjects with enough facts to reason
over, (b) the NONE escape + a conservative prompt, (c) the per-cycle budget,
(d) per-subject idempotence, and (e) discovered gaps are written at priority
'low' so they rank BELOW reactive gaps and conflicts in the unified backlog.
The autonomous pass is OFF by default (settings.enable_gap_discovery).

R9: NO SQL here. Grouping runs in Python; reads/writes go through the store.
"""
from __future__ import annotations

import asyncio
import dataclasses
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ._seam import metered_small_call
from ..llm import get_llm_client

# Reuse the contradiction scan's sparse-key Subject parser verbatim.
from .contradiction import _subject_of

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

# Same concurrency as the pairwise scans' batched model calls.
_GENERATOR_CONCURRENCY = 8

# Bound the per-subject prompt: at most this many claims, each truncated.
_MAX_CLAIMS_IN_PROMPT = 25
_CLAIM_TRUNC = 200
# Tidy cap on the stored gap text (the `missing` column is Text/unbounded).
_MISSING_TRUNC = 400

GAP_DISCOVERY_SYSTEM = (
    "You are reviewing what a knowledge base knows about ONE subject, to spot "
    "an important MISSING piece.\n\n"
    "You are given the SUBJECT and the FACTS the knowledge base currently "
    "holds about it.\n\n"
    "Name ONE important question about this subject that these facts do NOT "
    "answer and that a user would plausibly ask. State it as a single concise "
    "question.\n\n"
    "Be conservative: only name a gap that is clearly important AND clearly "
    "absent from the facts. If the facts reasonably cover the subject, or you "
    "are unsure, respond with exactly: NONE\n\n"
    "Respond with EITHER one concise question OR the single word NONE — "
    "nothing else."
)


@dataclass
class GapScanResult:
    """Outcome of one gap-discovery run for one customer."""

    customer_id: str
    facts_scanned: int
    subjects_seen: int        # subjects meeting the min-facts threshold
    subjects_evaluated: int   # model calls actually spent this run
    skipped_existing: int     # subjects skipped (already have a discovered gap)
    gaps_found: int           # non-NONE answers → open gaps written
    budget_exhausted: bool    # True if the call budget capped the run early


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _domain_of(fact: "Fact") -> Optional[str]:
    """The sparse-key Domain segment (4th field) of a fact, or None.

    Sparse key format is `Source | Locator | Subject | Domain`. Used to set
    the discovered gap's `domain` so the Cognition surface can group it."""
    key = (fact.prompt_text or "").strip()
    if "|" not in key:
        return None
    parts = [p.strip() for p in key.split("|")]
    if len(parts) >= 4 and parts[3]:
        return parts[3]
    return None


def _group_by_subject(
    facts: list["Fact"], min_facts: int
) -> "OrderedDict[str, list[Fact]]":
    """Bucket facts by sparse-key Subject, newest-first insertion order,
    keeping only subjects with >= min_facts usable (non-blank-claim) facts.

    `facts` arrives newest-first, so OrderedDict insertion order visits the
    most recently touched subjects first (recency bias, matching the pairwise
    scans)."""
    grouped: "OrderedDict[str, list[Fact]]" = OrderedDict()
    for f in facts:
        if not (f.claim_text or "").strip():
            continue
        subject = _subject_of(f)
        if not subject:
            continue
        grouped.setdefault(subject, []).append(f)
    return OrderedDict(
        (s, fs) for s, fs in grouped.items() if len(fs) >= min_facts
    )


def _build_prompt(subject: str, facts: list["Fact"]) -> str:
    lines = [
        f"- {(f.claim_text or '').strip()[:_CLAIM_TRUNC]}"
        for f in facts[:_MAX_CLAIMS_IN_PROMPT]
    ]
    return f"SUBJECT: {subject}\n\nFACTS:\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator (one fast-model call per subject)
# ---------------------------------------------------------------------------

async def _discover_for_subject(
    client: Any, subject: str, facts: list["Fact"], *, customer_id: str,
    store: Any = None, log: Any,
) -> Optional[str]:
    """Return the missing-question text for a subject, or None.

    None means "no gap" — the model answered NONE, returned nothing, or the
    call failed (fail-safe: an exception never writes a gap). Runs the seam
    call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row)."""
    user = _build_prompt(subject, facts)
    try:
        raw_text = await metered_small_call(
            client,
            customer_id=customer_id,
            origin="scan_gap_discovery",
            system=GAP_DISCOVERY_SYSTEM,
            user=user,
            max_tokens=64,
            store=store,
        )
        raw = (raw_text or "").strip()
        if not raw or raw.upper().startswith("NONE"):
            return None
        return raw[:_MISSING_TRUNC]
    except Exception as e:  # fail-safe — a generator error never writes
        log.warning(
            "gap_discovery.generator_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def discover_gaps(
    *,
    store: "MetadataStore",
    slm_client: Any = None,
    customer_id: str,
    max_subjects: int = 20,
    min_facts_per_subject: int = 2,
    fact_fetch_limit: Optional[int] = None,
    log: Any = None,
) -> GapScanResult:
    """Scan one customer's own facts for proactive gaps, surfacing-only.

    slm_client is an optional client override exposing `complete` (tests);
    None → the provider-neutral seam, and a no-op when neither an override
    nor a ready seam exists. Non-NONE answers write `open` gaps with
    source='gap_discovery', priority='low'.
    """
    log = log or logger
    if slm_client is None and not get_llm_client().is_ready():
        return GapScanResult(customer_id, 0, 0, 0, 0, 0, False)
    client = slm_client if slm_client is not None else get_llm_client()

    facts = await store.list_recent_facts_for_customer(
        customer_id, limit=fact_fetch_limit
    )
    grouped = _group_by_subject(facts, min_facts_per_subject)
    subjects_seen = len(grouped)

    # Idempotence (D4): skip subjects that already carry an open discovered
    # gap. One read of the open gaps builds the skip-set.
    open_gaps = await store.list_knowledge_gaps(
        customer_id, status="open", limit=1000
    )
    already_discovered = {
        g.subject
        for g in open_gaps
        if g.source == "gap_discovery" and g.subject
    }

    # Select candidates up to the budget, newest-subject first.
    to_eval: list[tuple[str, list["Fact"]]] = []
    skipped_existing = 0
    budget_exhausted = False
    for subject, subject_facts in grouped.items():
        if subject in already_discovered:
            skipped_existing += 1
            continue
        if len(to_eval) >= max_subjects:
            budget_exhausted = True
            break
        to_eval.append((subject, subject_facts))

    # Batched generation (worth gate = a non-NONE answer).
    sem = asyncio.Semaphore(_GENERATOR_CONCURRENCY)

    async def _ask(
        subject: str, subject_facts: list["Fact"]
    ) -> tuple[str, list["Fact"], Optional[str]]:
        async with sem:
            missing = await _discover_for_subject(
                client, subject, subject_facts,
                customer_id=customer_id, store=store, log=log
            )
        return (subject, subject_facts, missing)

    answered = (
        await asyncio.gather(*[_ask(s, fs) for (s, fs) in to_eval])
        if to_eval
        else []
    )

    gaps_found = 0
    for (subject, subject_facts, missing) in answered:
        if not missing:
            continue
        await store.create_knowledge_gap(
            customer_id,
            domain=_domain_of(subject_facts[0]),
            subject=subject,
            missing=missing,
            priority="low",
            source="gap_discovery",
        )
        gaps_found += 1
        log.info(
            "gap_discovery.gap_found",
            customer_id=customer_id,
            subject=subject,
        )

    result = GapScanResult(
        customer_id=customer_id,
        facts_scanned=len(facts),
        subjects_seen=subjects_seen,
        subjects_evaluated=len(to_eval),
        skipped_existing=skipped_existing,
        gaps_found=gaps_found,
        budget_exhausted=budget_exhausted,
    )
    log.info("gap_discovery.completed", **dataclasses.asdict(result))
    return result
