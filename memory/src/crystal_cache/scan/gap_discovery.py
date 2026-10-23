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

# C3 Q1=A (2026-08-11): the shared segment layer replaced the retired
# positional parsers across every scan.
from .segments import group_by_shared_segment, widest_segment_of

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
    "Never ask (Q102): what code does with empty, invalid, null, missing or "
    "oversized input, or what a function, field or type is for when the "
    "facts describe it; the exact time of day or wording of something the "
    "facts already date; the 'specific reason' behind a number or a plain "
    "fact; how something works that the facts say is not decided, not "
    "priced or planned for later; advice on how to do something. Ask only "
    "for a missing piece of knowledge about the person's work, plans, "
    "people, customers, product or decisions.\n\n"
    "You may also be given ALREADY RECORDED questions: open gaps this "
    "knowledge base already knows about. If the question you would name is "
    "one of them, or a rewording of one, or only narrows one, respond with "
    "exactly: NONE\n\n"
    "Respond with EITHER one concise question OR the single word NONE — "
    "nothing else."
)

# How many open gaps the prompt lists, and how long each may be.
_MAX_KNOWN_IN_PROMPT = 30
_KNOWN_TRUNC = 160

_STOPWORDS = frozenset(
    "a an the and or of to for in on at by with from is are was were be been "
    "being what which who whom whose when where why how does do did has have "
    "had it its this that these those there their they them he she his her "
    "as if then than so such any some all each every current currently yet "
    "still also into about over under between".split()
)


def _question_tokens(text: str) -> frozenset[str]:
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def same_question(a: str, b: str) -> bool:
    """Two gap questions that ask the same thing in other words (2026-10-21:
    six subjects of one bank each asked "what is the plugin review's
    status?"). Content words only; the same when one question's words all
    sit inside the other's or half the union is shared. Deliberately
    conservative: a missed duplicate is a visible extra row, a false merge
    silently loses a gap. The prompt's ALREADY RECORDED list is the main
    guard; this is the backstop behind it."""
    ta, tb = _question_tokens(a), _question_tokens(b)
    if not ta or not tb:
        return (a or "").strip().lower() == (b or "").strip().lower()
    if ta <= tb or tb <= ta:
        return True
    return len(ta & tb) / len(ta | tb) >= 0.5


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
    deduped: int = 0          # 2026-10-21: answers dropped as a rewording of an open gap
    already_answered: int = 0 # Q102: questions the bank answers elsewhere, never opened
    answer_checks: int = 0    # Q102: model calls spent on that check
    code_subjects_skipped: int = 0  # Q102: subjects whose facts are all code descriptions


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _group_by_subject(
    facts: list["Fact"], min_facts: int
) -> "OrderedDict[str, list[Fact]]":
    """Bucket facts by sparse-key Subject, newest-first insertion order,
    keeping only subjects with >= min_facts usable (non-blank-claim) facts.

    `facts` arrives newest-first, so OrderedDict insertion order visits the
    most recently touched subjects first (recency bias, matching the pairwise
    scans)."""
    # C3 Q1=A (2026-08-11): segment-set grouping — a "subject" is any
    # shared segment at any position, rarest first, namespace-scale
    # segments excluded (scan/segments.py). Supersedes the retired
    # positional parts[2] parse.
    from ..config import get_settings
    return group_by_shared_segment(
        facts,
        max_group_fraction=get_settings().scan_segment_max_group_fraction,
        min_group=min_facts,
    )


def _build_prompt(
    subject: str, facts: list["Fact"], known: Optional[list[str]] = None,
) -> str:
    lines = [
        f"- {(f.claim_text or '').strip()[:_CLAIM_TRUNC]}"
        for f in facts[:_MAX_CLAIMS_IN_PROMPT]
    ]
    text = f"SUBJECT: {subject}\n\nFACTS:\n" + "\n".join(lines)
    recorded = [
        f"- {q.strip()[:_KNOWN_TRUNC]}"
        for q in (known or [])[:_MAX_KNOWN_IN_PROMPT] if (q or "").strip()
    ]
    if recorded:
        text += "\n\nALREADY RECORDED (answer NONE if yours is one of these):\n" + "\n".join(recorded)
    return text


# ---------------------------------------------------------------------------
# Generator (one fast-model call per subject)
# ---------------------------------------------------------------------------

async def _discover_for_subject(
    client: Any, subject: str, facts: list["Fact"], *, customer_id: str,
    store: Any = None, log: Any, known: Optional[list[str]] = None,
) -> Optional[str]:
    """Return the missing-question text for a subject, or None.

    None means "no gap" — the model answered NONE, returned nothing, or the
    call failed (fail-safe: an exception never writes a gap). Runs the seam
    call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row)."""
    user = _build_prompt(subject, facts, known)
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
# Q102 (2026-10-23): is it really missing?
# ---------------------------------------------------------------------------
# Anthony's demo bank opened 76 gaps; 21 asked what other facts already
# said (the user's own name, the reason the launch moved, the total of a
# cost line) and 27 asked what a function does with empty input. A subject
# only sees its own facts, so before a gap is opened the question is
# searched across the whole bank and a light call reads the best hits.

ANSWERED_SYSTEM = (
    "You decide whether a knowledge base already answers a question.\n\n"
    "You are given a QUESTION and the FACTS a search for it found. If the "
    "facts answer it (fully, or as well as anyone could from what was "
    "recorded, including by saying it is not decided yet), reply ANSWERED. "
    "If the answer is truly not there, reply OPEN.\n\n"
    "Reply with exactly one word: ANSWERED or OPEN."
)
_ANSWER_HITS = 6
_ANSWER_MIN_SCORE = 0.35
_CODE_HEAD = "code"


def _is_code_subject(subject_facts: list["Fact"]) -> bool:
    """Every fact under the subject is a code description (Code|...). Its
    'gaps' are edge cases the code itself answers."""
    heads = [((getattr(f, "prompt_text", "") or "").split("|")[0].strip().lower()) for f in subject_facts]
    return bool(heads) and all(h == _CODE_HEAD for h in heads)


async def _bank_answers(
    client: Any, question: str, *, index: Any, encoder: Any, customer_id: str,
    store: Any, log: Any,
) -> Optional[bool]:
    """True when the bank answers the question, False when it doesn't,
    None when the check could not run (the gap is then opened as before)."""
    from ..encoding.executor import Priority, encode_native_async
    try:
        vec = await encode_native_async(encoder, question, priority=Priority.BULK)
    except Exception as e:  # noqa: BLE001
        log.warning("gap_discovery.answer_encode_failed", error=str(e)[:120])
        return None
    hits = index.search(vec, k=_ANSWER_HITS, exclude_id="", exclude_doc=None, min_score=_ANSWER_MIN_SCORE)
    if not hits:
        return False
    facts = "\n".join(f"- {(f.claim_text or '').strip()[:_CLAIM_TRUNC * 2]}" for f, _s in hits)
    try:
        raw = await metered_small_call(
            client, customer_id=customer_id, origin="scan_gap_discovery",
            system=ANSWERED_SYSTEM, user=f"QUESTION: {question}\n\nFACTS:\n{facts}",
            max_tokens=8, store=store,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("gap_discovery.answer_check_failed", error=str(e)[:120])
        return None
    return (raw or "").strip().upper().startswith("ANSWERED")


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
    encoder: Any = None,
) -> GapScanResult:
    """Scan one customer's own facts for proactive gaps, surfacing-only.

    slm_client is an optional client override exposing `complete` (tests);
    None → the provider-neutral seam, and a no-op when neither an override
    nor a ready seam exists. Non-NONE answers write `open` gaps with
    source='gap_discovery', priority='low'. With an encoder (Q102), a
    question the bank already answers elsewhere is not opened; subjects
    made only of code descriptions are never asked.
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
    # 2026-10-21: every open gap's question, whatever its source, goes to
    # the model as ALREADY RECORDED and is the dedupe set below — the
    # same question must not be asked once per subject that touches it.
    known_questions = [
        (g.missing or "").strip() for g in open_gaps if (g.missing or "").strip()
    ]

    # Select candidates up to the budget, newest-subject first.
    to_eval: list[tuple[str, list["Fact"]]] = []
    skipped_existing = 0
    budget_exhausted = False
    code_subjects_skipped = 0
    for subject, subject_facts in grouped.items():
        if subject in already_discovered:
            skipped_existing += 1
            continue
        if _is_code_subject(subject_facts):
            code_subjects_skipped += 1
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
                customer_id=customer_id, store=store, log=log,
                known=known_questions,
            )
        return (subject, subject_facts, missing)

    answered = (
        await asyncio.gather(*[_ask(s, fs) for (s, fs) in to_eval])
        if to_eval
        else []
    )

    gaps_found = 0
    deduped = 0
    already_answered = 0
    answer_checks = 0
    index = None
    if encoder is not None and any(m for (_s, _f, m) in answered):
        from .implication import _Index
        index = _Index([f for f in facts if getattr(f, "vector", None) and (f.claim_text or "").strip()])
    recorded: list[str] = list(known_questions)
    for (subject, subject_facts, missing) in answered:
        if not missing:
            continue
        if any(same_question(missing, q) for q in recorded):
            deduped += 1
            continue
        if index is not None:
            verdict = await _bank_answers(client, missing, index=index, encoder=encoder,
                                          customer_id=customer_id, store=store, log=log)
            if verdict is not None:
                answer_checks += 1
            if verdict:
                already_answered += 1
                recorded.append(missing)  # the same question from another subject is not checked twice
                log.info("gap_discovery.already_answered", customer_id=customer_id, subject=subject)
                continue
        recorded.append(missing)
        await store.create_knowledge_gap(
            customer_id,
            # C3 Q3=A: domain = the widest (leftmost) segment — the
            # broadest judgment the write-time model made, read back.
            domain=widest_segment_of(
                getattr(subject_facts[0], "prompt_text", None)
            ),
            subject=subject,
            missing=missing,
            priority="low",
            source="gap_discovery",
            # S3: the representative fact's sparse key anchors the gap.
            full_key=(getattr(subject_facts[0], "prompt_text", None) or None),
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
        deduped=deduped,
        already_answered=already_answered,
        answer_checks=answer_checks,
        code_subjects_skipped=code_subjects_skipped,
    )
    # Log-noise fix (2026-07-08): found-nothing cycles are debug; info is
    # reserved for cycles that found something or hit the scan budget.
    _log_fn = (
        log.info
        if (result.gaps_found or result.budget_exhausted)
        else log.debug
    )
    _log_fn("gap_discovery.completed", **dataclasses.asdict(result))
    return result
