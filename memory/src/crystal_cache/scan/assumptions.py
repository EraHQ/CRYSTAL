"""Assumptions scan — bridging inferences over paired crystals.

Slice 1 of the Assumptions worker (design ratified 2026-07-20, refresh
2026-07-31). The inference CORE shared by both ratified entry points:
the idle scan (slice 2's `run_assumptions_worker`, a new 'assumptions'
role on crystal-worker per RQ1=B) and the agent `assume` tool (slice 3,
ephemeral-by-default). Nothing here loops or registers tools — callers
own cadence and gating.

Per pair of crystals, ONE small-tier structured-verdict call asks
whether a bridging assumption connects them — something neither crystal
states but both together imply. A passing verdict (assumption_exists
AND confidence >= min_confidence) writes a born-quarantine,
recall-gated assumption crystal via
MetadataStore.create_assumption_crystal (all SQL there, R9).

Pairing inputs (Q1=C):
  Phase 1 — CHAINED pairs: crystals an author explicitly connected
    (store.list_chained_crystal_pairs; assumptions themselves excluded
    so speculation never compounds).
  Phase 2 — GAP-SEEDED pairs: for each open knowledge gap, the two
    distinct crystals holding the most recent facts under the gap's
    sparse-key Subject. A subject whose facts live in ONE crystal has
    no pair and is skipped — a bridging assumption needs two banks of
    evidence to bridge.

Every model call rides scan/_seam.metered_call with
origin="assumptions" (RQ3=B — one ledger kwarg from birth; the budget
gate stays origin-blind until the budget-origin gate ships) and a
json_schema carrying "additionalProperties": false (the v75/v76
Anthropic 400 lesson).

Fail-safe like the sibling scans: a model/parse/write failure on one
pair costs that pair, never the cycle.
"""
from __future__ import annotations

import dataclasses
import json
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..config import get_settings
from ._seam import metered_call

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact, KnowledgeGap

logger = structlog.get_logger(__name__)

# Bounded prompt: at most this many claims per parent, each truncated.
_MAX_CLAIMS_PER_PARENT = 12
_CLAIM_TRUNC = 200
_SUMMARY_TRUNC = 300
# Verdict fields are capped on write so a runaway completion can't
# balloon a crystal row.
_STATEMENT_TRUNC = 500
_SUBJECT_TRUNC = 120

# Structured verdict. "additionalProperties": false is REQUIRED on
# every object level — Anthropic 400s without it (live incident,
# v75/v76).
ASSUMPTION_VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assumption_exists": {
            "type": "boolean",
            "description": (
                "true only when the two crystals TOGETHER imply "
                "something neither states alone"
            ),
        },
        "statement": {
            "type": "string",
            "description": (
                "the bridging assumption as one declarative sentence; "
                "empty string when assumption_exists is false"
            ),
        },
        "subject": {
            "type": "string",
            "description": (
                "2-6 word topic label for the assumption; empty string "
                "when assumption_exists is false"
            ),
        },
        "confidence": {
            "type": "number",
            "description": (
                "0.0-1.0: how strongly the evidence supports the "
                "bridge; 0 when assumption_exists is false"
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "one sentence naming the evidence from each side",
        },
    },
    "required": [
        "assumption_exists", "statement", "subject", "confidence",
        "reasoning",
    ],
    "additionalProperties": False,
}

ASSUMPTION_SYSTEM = (
    "You are reviewing TWO knowledge crystals from the same knowledge "
    "base to decide whether they jointly support a BRIDGING "
    "ASSUMPTION: a plausible inference that NEITHER crystal states on "
    "its own but that follows from holding both together.\n\n"
    "Be conservative. A restatement of either side is NOT an "
    "assumption. A generic truism is NOT an assumption. Only report a "
    "bridge that is specific, grounded in claims from BOTH sides, and "
    "would be useful if true. When in doubt, report "
    "assumption_exists=false.\n\n"
    "Confidence is how strongly the given evidence supports the "
    "bridge, not how confident you feel in general.\n\n"
    "Respond ONLY with the JSON object matching the provided schema."
)


@dataclass
class AssumptionVerdict:
    """Parsed structured verdict for one pair."""

    assumption_exists: bool
    statement: str
    subject: str
    confidence: float
    reasoning: str


@dataclass
class AssumptionScanResult:
    """Outcome of one assumptions run for one customer."""

    customer_id: str
    chained_pairs_seen: int
    gap_pairs_seen: int
    pairs_evaluated: int      # model calls actually spent this run
    skipped_existing: int     # pairs skipped (assumption already exists)
    assumptions_written: int  # passing verdicts persisted
    below_threshold: int      # verdicts that existed but under min_confidence


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _subject_of_key(prompt_text: str) -> Optional[str]:
    """Sparse-key Subject (3rd `|` segment) of a fact key, or None.

    Local copy of the parsing rule (`Source | Locator | Subject |
    Domain`) rather than importing scan.contradiction's private helper
    across modules — the format is the stable contract, the helper is
    not.
    """
    key = (prompt_text or "").strip()
    if "|" not in key:
        return None
    parts = [p.strip() for p in key.split("|")]
    if len(parts) >= 3 and parts[2]:
        return parts[2]
    return None


def _render_side(
    label: str, summary: Optional[str], facts: "list[Fact]"
) -> str:
    lines = [f"{label}:"]
    if summary:
        lines.append(f"  ABOUT: {summary.strip()[:_SUMMARY_TRUNC]}")
    for f in facts[:_MAX_CLAIMS_PER_PARENT]:
        claim = (f.claim_text or "").strip()
        if claim:
            lines.append(f"  - {claim[:_CLAIM_TRUNC]}")
    return "\n".join(lines)


def _build_prompt(
    a_summary: Optional[str], a_facts: "list[Fact]",
    b_summary: Optional[str], b_facts: "list[Fact]",
    gap: "Optional[KnowledgeGap]" = None,
) -> str:
    parts = [
        _render_side("CRYSTAL A", a_summary, a_facts),
        "",
        _render_side("CRYSTAL B", b_summary, b_facts),
    ]
    if gap is not None:
        parts += [
            "",
            "CONTEXT — an open knowledge gap touches these crystals: "
            f"{(gap.missing or '').strip()[:_CLAIM_TRUNC]}",
            "Only report a bridge the crystals actually support; the "
            "gap is context, not a target to satisfy.",
        ]
    return "\n".join(parts)


def _parse_verdict(raw: Optional[str]) -> Optional[AssumptionVerdict]:
    """Parse the structured completion; None on any malformation.

    The json_schema output_config makes the text a JSON document, but
    the parse still defends itself — a provider fallback or a legacy
    test client can hand back anything.
    """
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
        return AssumptionVerdict(
            assumption_exists=bool(data["assumption_exists"]),
            statement=str(data.get("statement") or "")[:_STATEMENT_TRUNC],
            subject=str(data.get("subject") or "")[:_SUBJECT_TRUNC],
            confidence=max(0.0, min(1.0, float(data["confidence"]))),
            reasoning=str(data.get("reasoning") or ""),
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Inference (one metered small-tier call per pair)
# ---------------------------------------------------------------------------

async def infer_bridging_assumption(
    client: Any,
    store: "MetadataStore",
    *,
    customer_id: str,
    crystal_a_id: str,
    crystal_b_id: str,
    gap: "Optional[KnowledgeGap]" = None,
    log: Any = None,
) -> Optional[AssumptionVerdict]:
    """One pair -> one structured verdict, or None.

    None means "no usable verdict" — hydration found a missing crystal,
    the call failed, or the completion didn't parse. Fail-safe: an
    error never writes anything. The call is metered on the seam with
    origin='assumptions' (RQ3=B).
    """
    log = log or logger
    try:
        crystal_a = await store.get_crystal(crystal_a_id)
        crystal_b = await store.get_crystal(crystal_b_id)
        if crystal_a is None or crystal_b is None:
            return None
        a_facts = await store.list_facts_for_crystal(crystal_a_id)
        b_facts = await store.list_facts_for_crystal(crystal_b_id)
        user = _build_prompt(
            crystal_a.summary_text, a_facts,
            crystal_b.summary_text, b_facts,
            gap=gap,
        )
        raw = await metered_call(
            client,
            customer_id=customer_id,
            origin="assumptions",
            system=ASSUMPTION_SYSTEM,
            user=user,
            max_tokens=400,
            tier="small",
            store=store,
            json_schema=ASSUMPTION_VERDICT_SCHEMA,
        )
        verdict = _parse_verdict(raw)
        if verdict is None:
            log.warning(
                "assumptions.verdict_unparseable",
                customer_id=customer_id,
                crystal_a=crystal_a_id,
                crystal_b=crystal_b_id,
            )
        return verdict
    except Exception as e:  # fail-safe — one pair, never the cycle
        log.warning(
            "assumptions.inference_failed",
            customer_id=customer_id,
            crystal_a=crystal_a_id,
            crystal_b=crystal_b_id,
            error=str(e),
            error_type=type(e).__name__,
        )
        return None


# ---------------------------------------------------------------------------
# Gap-seeded pairing (store reads only — no encoder, no vector search)
# ---------------------------------------------------------------------------

async def _gap_seeded_pairs(
    store: "MetadataStore",
    customer_id: str,
    *,
    gaps_limit: int,
    log: Any,
) -> list[tuple[str, str, "KnowledgeGap"]]:
    """(crystal_a_id, crystal_b_id, gap) triples for open gaps.

    For each open gap with a Subject: group the customer's recent facts
    by sparse-key Subject and take the two distinct crystals holding
    the newest facts under it. One-crystal subjects are skipped — no
    pair, nothing to bridge.
    """
    if gaps_limit <= 0:
        return []
    gaps = await store.list_knowledge_gaps(
        customer_id, status="open", limit=100
    )
    gaps = [g for g in gaps if (g.subject or "").strip()][:gaps_limit * 3]
    if not gaps:
        return []

    facts = await store.list_recent_facts_for_customer(customer_id)
    by_subject: "OrderedDict[str, list[str]]" = OrderedDict()
    for f in facts:  # newest-first: first two distinct crystals win
        subject = _subject_of_key(f.prompt_text)
        if not subject:
            continue
        crystal_ids = by_subject.setdefault(subject, [])
        if f.crystal_id not in crystal_ids:
            crystal_ids.append(f.crystal_id)

    triples: list[tuple[str, str, "KnowledgeGap"]] = []
    for gap in gaps:
        crystal_ids = by_subject.get((gap.subject or "").strip(), [])
        if len(crystal_ids) < 2:
            continue
        a, b = sorted(crystal_ids[:2])
        triples.append((a, b, gap))
        if len(triples) >= gaps_limit:
            break
    log.debug(
        "assumptions.gap_pairs",
        customer_id=customer_id,
        open_gaps=len(gaps),
        paired=len(triples),
    )
    return triples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def run_assumptions_scan(
    *,
    store: "MetadataStore",
    slm_client: Any = None,
    customer_id: str,
    encoder: Any,
    pairs_limit: Optional[int] = None,
    gaps_limit: Optional[int] = None,
    min_confidence: Optional[float] = None,
    log: Any = None,
) -> AssumptionScanResult:
    """One assumptions run for one customer: chained pairs, then
    gap-seeded pairs, sequentially (the pair count is small by design —
    this run's budget is the settings knobs, not a semaphore).

    slm_client is the test override exposing `complete`/
    `complete_detailed`; None -> the provider-neutral seam, and a
    no-op result when the seam isn't ready. `encoder` is required for
    the write path (assumption vectors ride the serialized encoder
    lane). Limits/threshold default from settings
    (CC_ASSUMPTIONS_PAIRS_PER_CYCLE / _GAPS_PER_CYCLE /
    _MIN_CONFIDENCE) so slice 2's worker calls this bare.
    """
    log = log or logger
    settings = get_settings()
    pairs_limit = (
        settings.assumptions_pairs_per_cycle
        if pairs_limit is None else pairs_limit
    )
    gaps_limit = (
        settings.assumptions_gaps_per_cycle
        if gaps_limit is None else gaps_limit
    )
    min_confidence = (
        settings.assumptions_min_confidence
        if min_confidence is None else min_confidence
    )

    if slm_client is None:
        from ..llm import get_llm_client
        if not get_llm_client().is_ready():
            return AssumptionScanResult(customer_id, 0, 0, 0, 0, 0, 0)
        client = get_llm_client()
    else:
        client = slm_client

    chained = await store.list_chained_crystal_pairs(
        customer_id, limit=max(pairs_limit * 3, pairs_limit),
    )
    gap_triples = await _gap_seeded_pairs(
        store, customer_id, gaps_limit=gaps_limit, log=log,
    )

    # One work queue: chained pairs first (highest-signal input), then
    # gap-seeded. Gap triples carry their gap for prompt context + the
    # provenance tag.
    work: list[tuple[str, str, "Optional[KnowledgeGap]"]] = [
        (a, b, None) for (a, b) in chained
    ]
    chained_budget = pairs_limit
    work = work[:chained_budget] + [
        (a, b, g) for (a, b, g) in gap_triples
    ]

    evaluated = 0
    skipped_existing = 0
    written = 0
    below_threshold = 0
    seen_this_run: set[tuple[str, str]] = set()

    for a, b, gap in work:
        canonical = (a, b) if a <= b else (b, a)
        if canonical in seen_this_run:
            continue
        seen_this_run.add(canonical)

        existing = await store.find_assumption_for_parents(
            customer_id, a, b
        )
        if existing is not None:
            skipped_existing += 1
            continue

        verdict = await infer_bridging_assumption(
            client, store,
            customer_id=customer_id,
            crystal_a_id=a,
            crystal_b_id=b,
            gap=gap,
            log=log,
        )
        evaluated += 1
        if verdict is None or not verdict.assumption_exists:
            continue
        if not verdict.statement.strip() or not verdict.subject.strip():
            continue
        if verdict.confidence < min_confidence:
            below_threshold += 1
            continue

        try:
            await store.create_assumption_crystal(
                customer_id,
                statement=verdict.statement.strip(),
                subject=verdict.subject.strip(),
                parent_a_id=a,
                parent_b_id=b,
                confidence=verdict.confidence,
                encoder=encoder,
                gap_id=gap.id if gap is not None else None,
            )
            written += 1
        except Exception as e:  # fail-safe: one write, never the cycle
            log.warning(
                "assumptions.write_failed",
                customer_id=customer_id,
                crystal_a=a,
                crystal_b=b,
                error=str(e),
                error_type=type(e).__name__,
            )

    result = AssumptionScanResult(
        customer_id=customer_id,
        chained_pairs_seen=len(chained),
        gap_pairs_seen=len(gap_triples),
        pairs_evaluated=evaluated,
        skipped_existing=skipped_existing,
        assumptions_written=written,
        below_threshold=below_threshold,
    )
    # Sibling-scan log discipline: found-nothing cycles are debug.
    _log_fn = log.info if result.assumptions_written else log.debug
    _log_fn("assumptions.completed", **dataclasses.asdict(result))
    return result
