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
  2. Enumerate CANDIDATE pairs, bounded (D2), from three sources taken in
     turns so the cap is shared: each fact's nearest facts in OTHER
     crystals by vector (2026-10-21: cosine over the vectors the scan
     already loads, k per fact above a floor — the only source that pairs
     "Product launch | November 5" with "Launch on Product Hunt |
     November 12" when no key segment is shared); facts ACROSS crystals
     that share ANY sparse-key segment (C3 Q1=A, 2026-08-11 — segment-set
     grouping via scan/segments.py, rarest segments first, namespace-scale
     segments excluded); facts WITHIN the same crystal. Same-document
     pairs are refused at enumeration, before the cap (2026-10-21; they
     used to be filtered after it and ate the whole cap on a document-
     heavy bank). Document passages (content_chunk) do not enter: a
     passage against a sentence of another document is a restatement
     judged as a contradiction by a small model (the demo tenant's first
     ten rows), and extraction carries the claims.
  3. Cheap pre-check (D4): skip any candidate whose `pair_key` already has a
     conflict row in ANY status, or a scan_verdicts memo for this detector
     (2026-10-21) — no LLM call. A pair is judged once per version of its
     claim texts; before the memo, every CONSISTENT pair was re-judged
     every idle cycle and the budget never reached a new pair.
  4. Discriminate the survivors: fast model, one sentence of reasoning
     and then the verdict through a strict JSON schema (2026-10-21; the
     one-word form let the small tier flag restatements), batched under a
     semaphore. Bounded by `max_discriminator_calls` (the budget gate).
  5. Only a CONTRADICTS verdict writes an `open` conflict (the worth gate);
     CONSISTENT and UNRELATED are remembered in scan_verdicts.

The discriminator prompt descends from v1's `CROSS_VALIDATE_SYSTEM`
(crystal-cache-v1/scripts/run_level_f.py), generalized from "old rule vs new
knowledge" to "two stored facts about the same subject," with explicit
guards against the two over-flagging modes a symmetric framing invites:
restatements read as contradictions, and different facts read as
contradictions.

R9: this module contains NO SQL. Candidate enumeration runs in Python over
Fact objects returned by the store; all reads/writes go through store methods.
"""
from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Optional

import numpy as np
import structlog

from ..config import get_settings
from ._seam import metered_small_call
from ..llm import get_llm_client
from .segments import (
    group_by_shared_segment, most_specific_shared_segment,
    widest_segment_of,
)

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

# Concurrency for the batched discriminator calls (v1 used 10 for the same
# cross-validation pass; 8 here leaves headroom under the live API).
_DISCRIMINATOR_CONCURRENCY = 8

# Descends from v1's CROSS_VALIDATE_SYSTEM, generalized to a symmetric
# fact-vs-fact judgment. Same three-way contract + few-shot shape. The
# 2026-10-21 revision asks for one sentence of reasoning BEFORE the verdict
# (a small model that must commit in one word over-flags) and names the
# two over-flagging modes seen on a real bank: a restatement, or a longer
# claim that contains the shorter one, is CONSISTENT; two different facts
# that merely share a topic are UNRELATED.
CROSS_VALIDATE_SYSTEM = (
    "You are checking whether two stored facts about the same subject "
    "contradict each other.\n\n"
    "You are given CLAIM A and CLAIM B, two facts retrieved from a knowledge "
    "base.\n\n"
    "Verdicts:\n"
    "- CONTRADICTS: both claims name the same attribute of the same thing "
    "for the same time, and give values that cannot both hold (two prices, "
    "two dates, two counts, two owners, opposite statuses).\n"
    "- CONSISTENT: the claims can both be true. This includes a claim that "
    "restates the other in different words, a longer claim that contains "
    "the shorter one, two details of the same thing, and a claim that a "
    "value changed from X to Y next to a claim of X (before) or Y (after).\n"
    "- UNRELATED: the claims are about different things, people, products, "
    "features, rules or events, even when they share a topic.\n\n"
    "Examples:\n"
    "- A: 'The contract rate is $120/hr' vs B: 'The contract rate is $95/hr' "
    "-> CONTRADICTS\n"
    "- A: 'The office opens at 9am' vs B: 'Staff arrive by nine in the "
    "morning' -> CONSISTENT\n"
    "- A: 'Deploys never happen on Friday afternoon' vs B: 'Friday is the "
    "day unfinished work gets cut' -> UNRELATED (two different rules)\n"
    "- A: 'Validate input types' vs B: 'The API accepts a data_url "
    "parameter' -> UNRELATED\n\n"
    "Facts about different things are UNRELATED, never CONTRADICTS. When "
    "in doubt between CONTRADICTS and CONSISTENT, answer CONSISTENT.\n\n"
    "Each claim may come with where it is filed (its key path, general to "
    "specific), the source it was learned from, and when. Two claims "
    "CONTRADICT only when they are about the same thing in the same scope: "
    "when the key paths or sources show different products, companies, "
    "policies, people, places or periods, answer UNRELATED even if the "
    "words look alike.\n\n"
    "Reply with JSON only: {\"reason\": one sentence written before you "
    "decide, \"verdict\": \"CONTRADICTS\" | \"CONSISTENT\" | "
    "\"UNRELATED\"}"
)

VERDICT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reason": {
            "type": "string",
            "description": "one sentence on how the two claims relate, written before the verdict",
        },
        "verdict": {
            "type": "string",
            "enum": ["CONTRADICTS", "CONSISTENT", "UNRELATED"],
        },
    },
    "required": ["reason", "verdict"],
    "additionalProperties": False,
}

_VERDICTS = ("CONTRADICTS", "CONSISTENT", "UNRELATED")
# Verdicts worth remembering in scan_verdicts: the pair was judged and is
# not a conflict. UNKNOWN and ERROR are not remembered, so they retry.
_MEMO_VERDICTS = frozenset({"CONSISTENT", "UNRELATED"})


def parse_verdict(raw: Optional[str], verdicts: tuple[str, ...] = _VERDICTS) -> str:
    """The verdict in a judge reply, or UNKNOWN.

    Accepts the JSON object the schema asks for ({"reason", "verdict"})
    and the bare one-word form (legacy fakes, a provider that ignored the
    schema). A word is matched at the start of the reply, the way the
    one-word contract always was."""
    text = (raw or "").strip()
    if not text:
        return "UNKNOWN"
    if text.startswith("{"):
        try:
            data = json.loads(text)
            word = str((data or {}).get("verdict") or "").strip().upper()
            if word in verdicts:
                return word
        except Exception:  # noqa: BLE001 — fall through to the word form
            pass
    head = text.upper()
    for v in verdicts:
        if head.startswith(v):
            return v
    return "UNKNOWN"


# ---------------------------------------------------------------------------
# Console Refine PR 7b (2026-10-16): the explanation call
# ---------------------------------------------------------------------------
# Once a pair is judged CONTRADICTS, a SECOND small call explains it and
# recommends one of the four console verbs. Kept apart from the verdict
# call on purpose: CROSS_VALIDATE_SYSTEM is calibrated against
# over-flagging and runs once per candidate pair at 16 tokens; this runs
# once per FOUND conflict, carries what a decision needs (learned dates,
# sources), and the same function backs the on-demand
# POST /admin/api/conflicts/{id}/explain. Any failure leaves the conflict
# unexplained, never unwritten.

EXPLAIN_SYSTEM = (
    "Two stored facts about the same subject were judged to contradict "
    "each other. Explain to the person who owns this memory, in one or two "
    "plain sentences, why the two claims cannot both be true, and recommend "
    "what to do.\n\n"
    "Resolutions:\n"
    "- superseded: one claim is outdated; keep the other (name the loser).\n"
    "- blacklisted: one claim is wrong, not merely old; drop it (name the loser).\n"
    "- qualified: both are true under different conditions; keep both.\n"
    "- dismissed: not a real contradiction.\n\n"
    "Prefer superseded when the claims differ by time and one is newer; "
    "prefer qualified when they can be reconciled; use blacklisted only "
    "when a claim is clearly false. Name the losing side as 'a' or 'b', "
    "or 'none' for qualified and dismissed. Keep the reason to one sentence."
)

EXPLAIN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "why": {
            "type": "string",
            "description": "one or two plain sentences on why the two claims cannot both be true",
        },
        "resolution": {
            "type": "string",
            "enum": ["superseded", "blacklisted", "qualified", "dismissed"],
            "description": "the recommended resolution",
        },
        "loser": {
            "type": "string",
            "enum": ["a", "b", "none"],
            "description": "the side to drop for superseded or blacklisted; 'none' otherwise",
        },
        "reason": {
            "type": "string",
            "description": "one sentence on why this resolution",
        },
    },
    "required": ["why", "resolution", "loser", "reason"],
    "additionalProperties": False,
}

_EXPLAIN_RESOLUTIONS = frozenset({"superseded", "blacklisted", "qualified", "dismissed"})
_EXPLAIN_TRUNC = 1200


def parse_explanation(raw: Optional[str]) -> Optional[dict[str, Any]]:
    """{'why', 'resolution', 'loser', 'reason'} or None on anything
    malformed. Defensive like assumptions._parse_verdict: a provider
    fallback or a legacy test client can hand back anything. A loser is
    required for superseded / blacklisted and dropped for the other two,
    so the console can apply the suggestion exactly as stored."""
    import json

    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
        why = str(data.get("why") or "").strip()[:_EXPLAIN_TRUNC]
        resolution = str(data.get("resolution") or "").strip()
        loser = data.get("loser")
        reason = str(data.get("reason") or "").strip()[:_EXPLAIN_TRUNC]
    except Exception:  # noqa: BLE001
        return None
    if not why or resolution not in _EXPLAIN_RESOLUTIONS:
        return None
    if resolution in ("superseded", "blacklisted"):
        if loser not in ("a", "b"):
            return None
    else:
        loser = None
    return {"why": why, "resolution": resolution, "loser": loser, "reason": reason}


async def explain_contradiction(
    client: Any,
    *,
    customer_id: str,
    claim_a: str,
    claim_b: str,
    learned_a: Optional[str],
    learned_b: Optional[str],
    source_a: Optional[str],
    source_b: Optional[str],
    store: Any = None,
    log: Any = logger,
    system: str = EXPLAIN_SYSTEM,
    origin: str = "scan_contradiction",
) -> Optional[dict[str, Any]]:
    """The explanation call (metered, origin scan_contradiction). None on
    any failure, fail-safe like _discriminate: the conflict is still
    written, just unexplained, and the console offers Explain.

    2026-10-21: the dedup scan explains its rows through the same call
    with its own `system` (dedup.EXPLAIN_DUP_SYSTEM) and origin; the
    schema and the parse are shared, so the console renders both."""
    user = (
        f"CLAIM A: {claim_a}\n  learned: {learned_a or 'unknown'}\n  source: {source_a or 'unknown'}\n\n"
        f"CLAIM B: {claim_b}\n  learned: {learned_b or 'unknown'}\n  source: {source_b or 'unknown'}"
    )
    try:
        raw = await metered_small_call(
            client,
            customer_id=customer_id,
            origin=origin,
            system=system,
            user=user,
            max_tokens=400,
            store=store,
            json_schema=EXPLAIN_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001
        log.warning(
            "contradiction_scan.explain_failed",
            customer_id=customer_id, error=str(e), error_type=type(e).__name__,
        )
        return None
    parsed = parse_explanation(raw)
    if parsed is None:
        log.warning("contradiction_scan.explain_unparseable", customer_id=customer_id)
    return parsed


def _learned(fact: "Fact") -> Optional[str]:
    """YYYY-MM-DD of the fact's created_at, for the explanation prompt."""
    created = getattr(fact, "created_at", None)
    try:
        return created.date().isoformat() if created else None
    except AttributeError:
        return None


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
    explained: int = 0        # PR 7b: conflicts that also got reasoning + a recommendation
    memoized: int = 0         # 2026-10-21: non-conflict verdicts remembered in scan_verdicts


# ---------------------------------------------------------------------------
# Pure helpers (no I/O)
# ---------------------------------------------------------------------------

def _provenance(fact: "Fact") -> str:
    """Human-facing provenance string for one side of a conflict."""
    source_kind = (fact.source_kind or "").strip() or "unknown"
    if fact.source_doc_id:
        return f"{source_kind} @ {fact.source_doc_id}"
    return source_kind


def judge_context(fact: "Fact", details: Optional[dict] = None) -> str:
    """What the judges are told about one claim besides its text
    (2026-10-22, Anthony: "how does the model know the context? what if
    they were from two different policies?"): where it is filed (the key
    path the extractor wrote, general to specific), the source it came
    from (the document's label when the store gave it, else kind @ doc id)
    and the day it was learned. Empty when nothing is known."""
    parts: list[str] = []
    key = (getattr(fact, "prompt_text", "") or "").strip()
    segs = [s.strip() for s in key.split("|") if s.strip()]
    if segs:
        parts.append("filed under " + " > ".join(segs))
    d = (details or {}).get(getattr(fact, "id", None)) or {}
    src = (d.get("source_path") or "").strip() if isinstance(d, dict) else ""
    if not src:
        src = _provenance(fact)
        if src in ("", "unknown"):
            src = ""
    if src:
        parts.append(f'from "{src}"')
    learned = d.get("learned_at") if isinstance(d, dict) else None
    learned = str(learned)[:10] if learned else _learned(fact)
    if learned:
        parts.append(f"learned {learned}")
    return "; ".join(parts)


async def fact_details_for(store: Any, facts: list["Fact"], *, log: Any = None) -> dict:
    """{fact id: {learned_at, source_kind, source_path, ...}} from the store's
    conflict_fact_details, one query for the whole batch; empty when the
    store has no such read (a legacy fake) or it fails. Best-effort: the
    judges run without it."""
    ids = [f.id for f in facts if getattr(f, "id", None)]
    reader = getattr(store, "conflict_fact_details", None)
    if not ids or reader is None:
        return {}
    try:
        return await reader(ids) or {}
    except Exception as e:  # noqa: BLE001
        if log is not None:
            log.warning("scan.fact_details_failed", error=str(e)[:160])
        return {}


def claims_block(claim_a: str, claim_b: str, context_a: str = "", context_b: str = "") -> str:
    """The user message both judges read: each claim with its context in
    parentheses when there is one."""
    a = f"CLAIM A ({context_a}): {claim_a}" if context_a else f"CLAIM A: {claim_a}"
    b = f"CLAIM B ({context_b}): {claim_b}" if context_b else f"CLAIM B: {claim_b}"
    return f"{a}\n\n{b}"


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


# Vector-neighbor candidates are computed over at most this many of the
# newest facts: an N x N similarity over the 2000-fact fetch limit is
# 16 MB of float32, and the cap keeps a bank fetched without a limit
# bounded.
_NEIGHBOR_MAX_FACTS = 3000


def _neighbor_pairs(
    usable: list["Fact"], *, k: int, min_score: float,
) -> list[tuple["Fact", "Fact"]]:
    """Cross-crystal pairs by cosine over the facts' own vectors, strongest
    first: for each fact, its k nearest facts in OTHER crystals whose score
    clears `min_score`. Pure numpy over what the scan already loaded — no
    index call, no encoder. Facts without a vector, or with a vector of
    another dimension than the first, form no neighbor pairs."""
    if k <= 0:
        return []
    with_vec = [f for f in usable if getattr(f, "vector", None)]
    if len(with_vec) < 2:
        return []
    d = len(with_vec[0].vector)
    rows = [f for f in with_vec if len(f.vector) == d][:_NEIGHBOR_MAX_FACTS]
    n = len(rows)
    if n < 2:
        return []
    m = np.asarray([f.vector for f in rows], dtype=np.float32)
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    m = m / norms
    sims = m @ m.T
    crystals = np.asarray([f.crystal_id or "" for f in rows])
    # Same-crystal pairs belong to the within-crystal source; the
    # diagonal is a fact against itself.
    sims[crystals[:, None] == crystals[None, :]] = -1.0
    kk = min(k, n - 1)
    scored: dict[frozenset[int], float] = {}
    for i in range(n):
        top = np.argpartition(-sims[i], kk - 1)[:kk]
        for j in top:
            j = int(j)
            score = float(sims[i, j])
            if j == i or score < min_score:
                continue
            key = frozenset((i, j))
            if scored.get(key, -1.0) < score:
                scored[key] = score
    pairs: list[tuple["Fact", "Fact"]] = []
    for key, _score in sorted(scored.items(), key=lambda kv: -kv[1]):
        i, j = sorted(key)
        pairs.append((rows[i], rows[j]))
    return pairs


def _enumerate_candidate_pairs(
    facts: list["Fact"],
    max_pairs: int,
    *,
    skip_same_source: bool = False,
    neighbor_k: Optional[int] = None,
    neighbor_min_score: Optional[float] = None,
) -> list[tuple["Fact", "Fact"]]:
    """Bounded candidate pairs (D2) from three sources, taken in turns.

    The sources, in the order each round visits them: vector neighbors
    across crystals (strongest first), shared-segment pairs across
    crystals (rarest segment first, scan/segments.py), pairs within one
    crystal. One new pair per source per round until the cap, so no
    source can fill the cap alone — before 2026-10-21 the within-crystal
    pairs of one document-heavy bank did exactly that, and the
    cross-document pairs the scan exists for were never reached.

    `facts` arrives newest-first, so within every source recent facts
    come first. Pairs are de-duplicated across sources, and with
    `skip_same_source` two facts of the same document are refused here,
    before the cap, rather than filtered after it (CF-Q1=A: a document
    cannot contradict itself through its own extraction).

    `neighbor_k` / `neighbor_min_score` default to the settings
    (scan_neighbor_k, scan_neighbor_min_score); k=0 turns the neighbor
    source off.
    """
    settings = get_settings()
    k = settings.scan_neighbor_k if neighbor_k is None else neighbor_k
    floor = (
        settings.scan_neighbor_min_score
        if neighbor_min_score is None else neighbor_min_score
    )
    usable = [f for f in facts if (f.claim_text or "").strip()]

    by_crystal: dict[str, list["Fact"]] = {}
    for f in usable:
        by_crystal.setdefault(f.crystal_id, []).append(f)
    by_segment = group_by_shared_segment(
        usable,
        max_group_fraction=settings.scan_segment_max_group_fraction,
    )

    def _within() -> Iterator[tuple["Fact", "Fact"]]:
        for group in by_crystal.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    yield group[i], group[j]

    def _segment() -> Iterator[tuple["Fact", "Fact"]]:
        for group in by_segment.values():
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    if group[i].crystal_id == group[j].crystal_id:
                        continue  # the within-crystal source's
                    yield group[i], group[j]

    def _neighbors() -> Iterator[tuple["Fact", "Fact"]]:
        yield from _neighbor_pairs(usable, k=k, min_score=floor)

    seen: set[frozenset[str]] = set()
    pairs: list[tuple["Fact", "Fact"]] = []

    def _add(a: "Fact", b: "Fact") -> bool:
        """Add a pair if new; return True once the cap is reached."""
        if a.id == b.id:
            return False
        if skip_same_source:
            doc_a = getattr(a, "source_doc_id", None)
            if doc_a and doc_a == getattr(b, "source_doc_id", None):
                return False
        key = frozenset((a.id, b.id))
        if key in seen:
            return False
        seen.add(key)
        pairs.append((a, b))
        return len(pairs) >= max_pairs

    if max_pairs <= 0:
        return pairs
    sources: list[Iterator[tuple["Fact", "Fact"]]] = [
        _neighbors(), _segment(), _within(),
    ]
    while sources:
        still_open: list[Iterator[tuple["Fact", "Fact"]]] = []
        for src in sources:
            for a, b in src:
                before = len(pairs)
                if _add(a, b):
                    return pairs
                if len(pairs) > before:
                    still_open.append(src)
                    break
            # A source that yielded nothing new is exhausted and dropped.
        sources = still_open
    return pairs


# ---------------------------------------------------------------------------
# Discriminator (one fast-model call per pair)
# ---------------------------------------------------------------------------

async def _discriminate(
    client: Any, claim_a: str, claim_b: str, *, customer_id: str,
    store: Any = None, log: Any, context_a: str = "", context_b: str = "",
) -> str:
    """Return CONTRADICTS | CONSISTENT | UNRELATED | UNKNOWN | ERROR.

    Runs the seam call off the event loop via the shared metered helper
    (_seam.metered_small_call — emits an origin-tagged llm_calls cost row;
    Core Principle #1 preserved). Fail-safe: any exception returns
    "ERROR" (treated as no-contradiction — never writes a row).
    `context_a` / `context_b` are judge_context() strings (key path,
    source, date), so a look-alike pair from two scopes is UNRELATED.
    """
    user = claims_block(claim_a, claim_b, context_a, context_b)
    try:
        raw_text = await metered_small_call(
            client,
            customer_id=customer_id,
            origin="scan_contradiction",
            system=CROSS_VALIDATE_SYSTEM,
            user=user,
            max_tokens=200,
            store=store,
            json_schema=VERDICT_SCHEMA,
        )
        return parse_verdict(raw_text)
    except Exception as e:  # fail-safe — a discriminator error never writes
        log.warning(
            "contradiction_scan.discriminator_failed",
            error=str(e),
            error_type=type(e).__name__,
        )
        return "ERROR"


DETECTOR = "contradiction_scan"


async def _memoized_keys(
    store: Any, customer_id: str, *, detector: str, pair_keys: list[str],
) -> set[str]:
    """Pair keys this detector already judged (scan_verdicts). A store
    without the memo (a legacy fake) means nothing is remembered."""
    reader = getattr(store, "recorded_scan_verdict_keys", None)
    if reader is None or not pair_keys:
        return set()
    return await reader(customer_id, detector=detector, pair_keys=pair_keys)


async def _remember_verdicts(
    store: Any, customer_id: str, *, detector: str,
    verdicts: list[tuple[str, str]], log: Any,
) -> int:
    """Best-effort memo write: a failure costs a re-judgment next cycle,
    never the scan."""
    writer = getattr(store, "record_scan_verdicts", None)
    if writer is None or not verdicts:
        return 0
    try:
        return int(await writer(customer_id, detector=detector, verdicts=verdicts))
    except Exception as e:  # noqa: BLE001
        log.warning(
            "scan.verdict_memo_failed", detector=detector,
            customer_id=customer_id, error=str(e)[:200],
        )
        return 0


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
    # 2026-10-21: document passages leave the contradiction scan the way
    # they left the dedup scan (CF-Q1=A kept "chunk-vs-fact reach" here;
    # on the first real bank that reach produced seven of ten rows, each
    # a passage against a sentence it contained, judged by a small
    # model). Extraction carries the claims; a passage is provenance.
    facts = [f for f in facts if (getattr(f, "pair_type", "") or "") != "content_chunk"]
    # CF-Q1=A (2026-07-23): a document cannot contradict itself via its
    # own extraction — same-source pairs are provenance, not conflict.
    # Refused at enumeration (before the cap) since 2026-10-21.
    candidates = _enumerate_candidate_pairs(
        facts, max_candidate_pairs, skip_same_source=True,
    )

    # Cheap pre-filter (D4): drop candidates already recorded (a conflict
    # row in any status, or a scan_verdicts memo for this detector), no
    # LLM call. Stop selecting once the call budget is reached.
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

    # Batched discrimination (worth gate = CONTRADICTS). Each side goes to
    # the judge with its key path, source and date (judge_context).
    sem = asyncio.Semaphore(_DISCRIMINATOR_CONCURRENCY)
    details = await fact_details_for(
        store, [f for pair in to_check for f in pair[:2]], log=log,
    )

    async def _judge(
        a: "Fact", b: "Fact", pair_key: str
    ) -> tuple["Fact", "Fact", str, str]:
        async with sem:
            verdict = await _discriminate(
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

    conflicts_found = 0
    explained = 0
    for (a, b, pair_key, verdict) in judged:
        if verdict != "CONTRADICTS":
            continue
        subject = (
            most_specific_shared_segment(a.prompt_text, b.prompt_text)
            or widest_segment_of(a.prompt_text)
            or widest_segment_of(b.prompt_text)
        )
        # PR 7b: the second call, only for found conflicts. None leaves
        # the row unexplained; the console offers Explain for those.
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
            reasoning=explanation["why"] if explanation else None,
            recommendation=(
                {k: explanation[k] for k in ("resolution", "loser", "reason")}
                if explanation else None
            ),
        )
        conflicts_found += 1
        # C3 Q2=A (2026-08-11): every conflict the scan opens gets a
        # witness in the curation activity feed — the most important
        # thing the system notices about its own knowledge must not be
        # the one thing it notices silently. Best-effort by contract.
        try:
            await store.record_curation_event(
                customer_id,
                event_type="conflict_found",
                subject_id=conflict.id,
                label=(
                    f"Contradiction found - {subject}"[:256]
                    if subject else "Contradiction found"
                ),
                payload={
                    "detector": "contradiction_scan",
                    "fact_a": a.id, "fact_b": b.id,
                },
            )
        except Exception:  # noqa: BLE001 — witness never breaks the scan
            log.debug("curation_event.emit_failed", exc_info=True)
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
        explained=explained,
        memoized=memoized,
    )
    # Log-noise fix (2026-07-08): found-nothing cycles are debug; info is
    # reserved for cycles that found something or hit the scan budget.
    _log_fn = (
        log.info
        if (result.conflicts_found or result.budget_exhausted)
        else log.debug
    )
    _log_fn("contradiction_scan.completed", **dataclasses.asdict(result))
    return result
