"""Implication probing: contradictions reachable only by reasoning.

Q98 B (Anthony, 2026-10-22: "contradictions could have much looser ties
than just similarity and wording. 'Oxygen only exists as a gas' and
'Water is made of two hydrogen and one oxygen' is a clear contradiction
but our system may not pick it up"). The pairwise scans only ever judge
pairs that similarity put in front of them (vector neighbours, a shared
key segment, the same cluster); two facts that contradict through one
step of reasoning share no meaning in embedding space and never meet.

This scan adds a fourth way in that runs on consequence, not wording:

  1. A fact is probed ONCE, when it is new to the scan (memo per fact in
     scan_verdicts, detector "implication_probe"). A light model is asked
     what else must be true if the fact is (up to 3 CONSEQUENCES) and
     what a knowledge base would say if it disagreed (up to 3 search
     QUESTIONS). Reasoning off; JSON.
  2. Every consequence and question is encoded with the bank's own
     encoder and searched against the tenant's facts by cosine over the
     facts' stored vectors (in process, like the neighbour source). A
     question is what a memory is built to answer, which is why this
     works where sentence similarity does not: "which compounds contain
     oxygen?" lands on the water fact.
  2b. The checker (Anthony, 2026-10-22: "how do we limit the search so a
     fake implication isn't made?"). Before anything is searched, one
     more light call grades every consequence: FOLLOWS (from A plus
     definitions anyone would grant: water is a liquid), SPECULATES
     (adds a specific about this person's world that A does not contain:
     a product, a price, a person, a date, a policy), DOESNT. Only
     FOLLOWS consequences are searched; the rest are logged with their
     grade. General knowledge is allowed, invention about the user's
     world is not; a search question is not graded because a search
     alone can never write a row.
  3. Each hit (not the fact itself, not its own document, best consequence
     per hit) goes to a judge WITH THE BRIDGE written out: FACT A, the
     CONSEQUENCE that follows from A, FACT B found by searching for it.
     The judge reasons one step across the bridge. CONTRADICTS writes a
     knowledge_conflict under detector "implication_scan", with the
     bridge and the judge's sentence as its reasoning, so the console can
     say "A implies ...; B says ...". CONSISTENT and UNRELATED are
     memoized per pair like the other scans.

Every pair gets at least one chance: the later fact is always probed
against a bank that already holds the earlier one. One bridge, not two:
a three-fact chain is the regional sweep's job (Q98 A, boarded).

  4. Q99 C (Anthony, 2026-10-23: "an implication and an assumption are
     the same thing... knowing something doesn't contradict is knowledge
     in itself"). A consequence the checker kept is written as an
     assumption crystal of kind "consequence" with ONE parent (the fact),
     and every judged hit ties it to that fact with an edge typed by the
     verdict: consistent_with / unrelated_to / contradicted_by (the
     conflict row still exists for CONTRADICTS). It is born gated; the
     first consistent_with edge lets it into recall (labelled "derived
     from" the parent) once the parent itself is recallable; approval
     (a person, a rule, or the parent) gives it the parent's standing.
     Consequences are never probed as A and never sit in the index, so
     edges always run consequence -> real fact; a chain of consequences
     is Q98 A, still boarded.

Budget: one derive call and one check call per fact plus one judge call
per hit, counted
against the shared daily ceiling (convergence_max_calls_per_day) and
capped per cycle by implication_max_facts_per_cycle. Nothing here
changes a fact; it adds consequences beside them. R9: no SQL; reads and
writes go through the store. The encoder runs on the BULK lane.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import structlog

from ._seam import metered_small_call
from ..encoding.executor import Priority, encode_native_async
from ..llm import get_llm_client
from .contradiction import (
    VERDICT_SCHEMA,
    _learned,
    _memoized_keys,
    _pair_key,
    _provenance,
    _remember_verdicts,
    fact_details_for,
    judge_context,
    parse_verdict,
)
from .segments import most_specific_shared_segment, widest_segment_of

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..models import Fact

logger = structlog.get_logger(__name__)

DETECTOR = "implication_scan"
PROBE_DETECTOR = "implication_probe"
_MEMO_VERDICTS = {"CONSISTENT", "UNRELATED"}
_CONCURRENCY = 4
_MAX_FACTS_IN_INDEX = 3000
_DERIVE_MAX_TOKENS = 400
_JUDGE_MAX_TOKENS = 220
_CLAIM_TRUNC = 600
# Derived namespaces are never probed and never searched (Q99 C).
_DERIVED_NAMESPACES = {"assumptions", "assumptionresponse", "consequences"}
_CHECK_VERDICTS = {"CONSISTENT", "UNRELATED", "CONTRADICTS"}

DERIVE_SYSTEM = (
    "You read ONE stored fact from a personal knowledge base and think one "
    "step ahead of it, so that other stored facts which quietly disagree "
    "with it can be found.\n\n"
    "Return JSON with two lists:\n"
    "- \"consequences\": up to 3 short statements that MUST be true if the "
    "fact is true. Each one a plain, general sentence that another fact "
    "could directly contradict. Prefer consequences about categories, "
    "limits and absolutes (what cannot exist, what always holds, what "
    "something is made of, how much, by when, who owns it).\n"
    "- \"questions\": up to 3 short search questions whose answers, if they "
    "exist in the knowledge base, would show the fact to be wrong.\n\n"
    "Example. Fact: 'Oxygen only exists as a gas.' consequences: ['No "
    "liquid or solid substance contains oxygen', 'Oxygen is never part of "
    "a compound that is liquid at room temperature']; questions: ['Which "
    "compounds contain oxygen?', 'Is any substance containing oxygen a "
    "liquid?'].\n\n"
    "Stay strictly with what the fact says; never add knowledge of your "
    "own as a consequence. Short, plain sentences. Empty lists are fine "
    "for a fact with no consequences (a date, a name, a preference)."
)

DERIVE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        # No maxItems: the structured-output subset rejects it (the 2026-10-02
        # 502 lesson); _derive slices to 3 and the prompt says "up to 3".
        "consequences": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["consequences", "questions"],
    "additionalProperties": False,
}

CHECK_SYSTEM = (
    "You grade whether each proposed CONSEQUENCE really follows from one "
    "stored FACT.\n\n"
    "Grades:\n"
    "- FOLLOWS: it follows from the fact plus definitions and common "
    "knowledge anyone would grant (water is a liquid; a month has about "
    "thirty days; a price in dollars per month is a recurring charge). It "
    "names nothing about the fact's owner, company, product, people, "
    "prices, dates or policies that the fact itself does not name.\n"
    "- SPECULATES: it introduces a specific about this person's or "
    "company's world that the fact does not contain (another product or "
    "plan, a price, a person, a date, a policy, a customer, a decision), "
    "however plausible. Plausible is not the same as following.\n"
    "- DOESNT: it does not follow from the fact at all.\n\n"
    "Example. FACT: 'Oxygen only exists as a gas.' 'No liquid substance "
    "contains oxygen' -> FOLLOWS. 'Oxygen is sold by the tank' -> DOESNT. "
    "FACT: 'Pro costs $18 a month.' 'Pro is a recurring monthly charge' -> "
    "FOLLOWS. 'The Team plan costs more than Pro' -> SPECULATES (the fact "
    "names no Team plan).\n\n"
    "Return JSON: {\"grades\": [one entry per consequence, in order, each "
    "{\"consequence\": the text as given, \"grade\": \"FOLLOWS\" | "
    "\"SPECULATES\" | \"DOESNT\"}]}"
)

CHECK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grades": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "consequence": {"type": "string"},
                    "grade": {"type": "string", "enum": ["FOLLOWS", "SPECULATES", "DOESNT"]},
                },
                "required": ["consequence", "grade"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["grades"],
    "additionalProperties": False,
}
_CHECK_MAX_TOKENS = 300

BRIDGE_SYSTEM = (
    "You check whether one stored fact contradicts a CONSEQUENCE of "
    "another stored fact.\n\n"
    "You are given FACT A, a CONSEQUENCE that follows from A in one step "
    "of reasoning, and FACT B, which was found by searching the knowledge "
    "base for that consequence.\n\n"
    "Verdicts:\n"
    "- CONTRADICTS: B cannot be true if A is true, because B goes against "
    "the consequence (B names a case the consequence rules out, a value "
    "it forbids, a state it says cannot exist).\n"
    "- CONSISTENT: B can be true alongside A and its consequence.\n"
    "- UNRELATED: B does not bear on the consequence at all.\n\n"
    "Example. A: 'Oxygen only exists as a gas.' Consequence: 'No liquid "
    "substance contains oxygen.' B: 'Water is made of two hydrogen and one "
    "oxygen.' -> CONTRADICTS (water is a liquid that contains oxygen).\n\n"
    "Each fact may come with where it is filed, its source and when it was "
    "learned. Facts from different products, companies, policies, people, "
    "places or periods are UNRELATED even when the words look alike. When "
    "in doubt between CONTRADICTS and CONSISTENT, answer CONSISTENT.\n\n"
    "Reply with JSON only: {\"reason\": one sentence naming what B says and "
    "why it does or does not go against the consequence, \"verdict\": "
    "\"CONTRADICTS\" | \"CONSISTENT\" | \"UNRELATED\"}"
)


@dataclass
class ImplicationScanResult:
    customer_id: str
    facts_considered: int
    facts_probed: int          # derive calls spent
    checks: int                # checker calls spent (one per fact with consequences)
    dropped: int               # consequences the checker refused (SPECULATES / DOESNT)
    searches: int              # consequences + questions encoded and searched
    hits: int                  # judge calls spent
    conflicts_found: int
    memoized: int
    skipped_probed: int
    budget_exhausted: bool
    consequences_written: int = 0   # Q99 C: consequence crystals created this run
    edges_written: int = 0          # typed check edges written this run
    entered_recall: int = 0         # consequences whose gate opened this run

    @property
    def calls(self) -> int:
        return self.facts_probed + self.checks + self.hits


def _subject_of(a: "Fact", b: "Fact") -> Optional[str]:
    return (
        most_specific_shared_segment(a.prompt_text, b.prompt_text)
        or widest_segment_of(a.prompt_text)
        or widest_segment_of(b.prompt_text)
    )


def _derived(f: "Fact") -> bool:
    """An assumption's or a consequence's own fact (Q99 C): knowledge the
    bank derived, never probed for further consequences and never a
    search hit, so every check edge runs consequence -> stated fact."""
    head = (getattr(f, "prompt_text", "") or "").split("|")[0].strip().lower()
    return head in _DERIVED_NAMESPACES


def _usable(facts: list["Fact"]) -> list["Fact"]:
    """Claims with a vector, no content chunks (they are passages, not
    facts), nothing the bank derived itself."""
    out = []
    for f in facts:
        if (getattr(f, "pair_type", "") or "") == "content_chunk":
            continue
        if not (f.claim_text or "").strip():
            continue
        if not getattr(f, "vector", None):
            continue
        if _derived(f):
            continue
        out.append(f)
    return out


class _Index:
    """Cosine over the facts' own vectors, built once per scan."""

    def __init__(self, facts: list["Fact"]):
        d = len(facts[0].vector) if facts else 0
        rows = [f for f in facts if len(f.vector) == d][:_MAX_FACTS_IN_INDEX]
        self.facts = rows
        m = np.asarray([f.vector for f in rows], dtype=np.float32) if rows else np.zeros((0, 0), np.float32)
        if rows:
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            m = m / norms
        self.m = m
        self.d = d

    def search(self, q: np.ndarray, *, k: int, exclude_id: str, exclude_doc: Optional[str], min_score: float) -> list[tuple["Fact", float]]:
        if not self.facts or q.shape[0] != self.d:
            return []
        q = np.asarray(q, dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n > 0:
            q = q / n
        sims = self.m @ q
        order = np.argsort(-sims)
        out: list[tuple["Fact", float]] = []
        for i in order:
            s = float(sims[i])
            if s < min_score:
                break
            f = self.facts[i]
            if f.id == exclude_id:
                continue
            if exclude_doc and getattr(f, "source_doc_id", None) == exclude_doc:
                continue
            out.append((f, s))
            if len(out) >= k:
                break
        return out


async def _derive(client: Any, fact: "Fact", *, customer_id: str, store: Any, log: Any) -> Optional[tuple[list[str], list[str]]]:
    """(consequences, questions), or None when the call failed (the fact is
    not memoized as probed, so the next cycle tries again)."""
    user = f"FACT: {(fact.claim_text or '')[:_CLAIM_TRUNC]}"
    key = (fact.prompt_text or "").strip()
    if key:
        user += f"\n(filed under {' > '.join(s.strip() for s in key.split('|') if s.strip())})"
    try:
        raw = await metered_small_call(
            client, customer_id=customer_id, origin="scan_implication",
            system=DERIVE_SYSTEM, user=user, max_tokens=_DERIVE_MAX_TOKENS,
            store=store, json_schema=DERIVE_SCHEMA,
        )
        data = json.loads((raw or "").strip())
    except Exception as e:  # noqa: BLE001 — a failed derive probes nothing; the fact is retried next cycle
        log.warning("implication_scan.derive_failed", fact_id=fact.id, error=str(e)[:160])
        return None
    cons = [str(x).strip() for x in (data.get("consequences") or []) if str(x).strip()][:3]
    qs = [str(x).strip() for x in (data.get("questions") or []) if str(x).strip()][:3]
    return cons, qs


async def _check(client: Any, fact: "Fact", consequences: list[str], *, customer_id: str,
                 store: Any, log: Any) -> tuple[list[str], list[tuple[str, str]]]:
    """(kept consequences, [(dropped consequence, grade)]). A failed check
    keeps nothing: an ungraded consequence is never searched."""
    if not consequences:
        return [], []
    user = f"FACT: {(fact.claim_text or '')[:_CLAIM_TRUNC]}\n\nCONSEQUENCES:\n" + "\n".join(
        f"{i + 1}. {c}" for i, c in enumerate(consequences)
    )
    try:
        raw = await metered_small_call(
            client, customer_id=customer_id, origin="scan_implication",
            system=CHECK_SYSTEM, user=user, max_tokens=_CHECK_MAX_TOKENS,
            store=store, json_schema=CHECK_SCHEMA,
        )
        grades = (json.loads((raw or "").strip()) or {}).get("grades") or []
    except Exception as e:  # noqa: BLE001
        log.warning("implication_scan.check_failed", fact_id=fact.id, error=str(e)[:160])
        return [], [(c, "UNCHECKED") for c in consequences]
    by_text = {str(g.get("consequence", "")).strip().lower(): str(g.get("grade", "")).upper()
               for g in grades if isinstance(g, dict)}
    kept: list[str] = []
    dropped: list[tuple[str, str]] = []
    for i, c in enumerate(consequences):
        grade = by_text.get(c.strip().lower())
        if grade is None and i < len(grades) and isinstance(grades[i], dict):
            grade = str(grades[i].get("grade", "")).upper()  # by position when the text was rephrased
        if grade == "FOLLOWS":
            kept.append(c)
        else:
            dropped.append((c, grade or "UNGRADED"))
    return kept, dropped


def _bridge_block(a: "Fact", consequence: str, b: "Fact", details: dict) -> str:
    ca, cb = judge_context(a, details), judge_context(b, details)
    fa = f"FACT A ({ca}): " if ca else "FACT A: "
    fb = f"FACT B ({cb}): " if cb else "FACT B: "
    return (
        f"{fa}{(a.claim_text or '')[:_CLAIM_TRUNC]}\n\n"
        f"CONSEQUENCE OF A: {consequence}\n\n"
        f"{fb}{(b.claim_text or '')[:_CLAIM_TRUNC]}"
    )


async def _judge(client: Any, a: "Fact", consequence: str, b: "Fact", *, details: dict,
                 customer_id: str, store: Any, log: Any) -> tuple[str, str]:
    """(verdict, reason)."""
    try:
        raw = await metered_small_call(
            client, customer_id=customer_id, origin="scan_implication",
            system=BRIDGE_SYSTEM, user=_bridge_block(a, consequence, b, details),
            max_tokens=_JUDGE_MAX_TOKENS, store=store, json_schema=VERDICT_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001
        log.warning("implication_scan.judge_failed", error=str(e)[:160])
        return "ERROR", ""
    verdict = parse_verdict(raw)
    reason = ""
    try:
        reason = str((json.loads((raw or "").strip()) or {}).get("reason") or "").strip()
    except Exception:  # noqa: BLE001 — a bare-word reply has no reason
        reason = ""
    return verdict, reason


async def scan_by_implication(
    *,
    store: "MetadataStore",
    encoder: Any,
    customer_id: str,
    llm_client: Any = None,
    max_facts: int = 10,
    hits_per_fact: int = 3,
    min_score: float = 0.45,
    fact_fetch_limit: Optional[int] = None,
    log: Any = None,
) -> ImplicationScanResult:
    """Probe up to `max_facts` not-yet-probed facts of one tenant (newest
    first) and judge what their consequences find. Surfacing-only."""
    log = log or logger
    client = llm_client or get_llm_client()
    facts = await store.list_recent_facts_for_customer(customer_id, limit=fact_fetch_limit)
    usable = _usable(list(facts))
    result = ImplicationScanResult(
        customer_id=customer_id, facts_considered=len(usable), facts_probed=0,
        checks=0, dropped=0, searches=0, hits=0, conflicts_found=0, memoized=0,
        skipped_probed=0, budget_exhausted=False,
    )
    if not usable or max_facts <= 0 or not getattr(client, "is_ready", lambda: True)():
        return result
    probed = await _memoized_keys(store, customer_id, detector=PROBE_DETECTOR,
                                  pair_keys=[f.id for f in usable])
    todo = [f for f in usable if f.id not in probed]
    result.skipped_probed = len(usable) - len(todo)
    if not todo:
        return result
    if len(todo) > max_facts:
        result.budget_exhausted = True
        todo = todo[:max_facts]
    index = _Index(usable)
    details = await fact_details_for(store, usable, log=log)
    sem = asyncio.Semaphore(_CONCURRENCY)
    probed_ok: list[str] = []
    judged_pairs: list[tuple[str, str]] = []

    async def _probe(a: "Fact") -> None:
        async with sem:
            derived = await _derive(client, a, customer_id=customer_id, store=store, log=log)
        result.facts_probed += 1  # the call was spent either way
        if derived is None:
            return
        cons, qs = derived
        probed_ok.append(a.id)
        if cons:
            async with sem:
                cons, dropped = await _check(client, a, cons, customer_id=customer_id, store=store, log=log)
            result.checks += 1
            result.dropped += len(dropped)
            for text, grade in dropped:
                log.info("implication_scan.consequence_dropped", fact_id=a.id, grade=grade, consequence=text[:200])
        # Q99 C: what survived the checker is knowledge in its own right.
        cons_ids: dict[str, str] = {}
        if cons:
            subject = widest_segment_of(a.prompt_text) or "General"
            for text in cons:
                try:
                    cid = await store.find_consequence_crystal(customer_id, parent_fact_id=a.id, statement=text)
                    if cid is None:
                        cid = await store.create_consequence_crystal(
                            customer_id, statement=text, subject=subject,
                            parent_fact_id=a.id, parent_crystal_id=a.crystal_id, encoder=encoder,
                        )
                        result.consequences_written += 1
                    cons_ids[text] = cid
                except Exception as e:  # noqa: BLE001 — one write never ends the probe
                    log.warning("implication_scan.consequence_write_failed", fact_id=a.id, error=str(e)[:160])
        # Best consequence per hit, across every consequence and question.
        best: dict[str, tuple["Fact", str, float]] = {}
        for text, is_question in [(c, False) for c in cons] + [(q, True) for q in qs]:
            try:
                vec = await encode_native_async(encoder, text, priority=Priority.BULK)
            except Exception as e:  # noqa: BLE001
                log.warning("implication_scan.encode_failed", error=str(e)[:120])
                continue
            result.searches += 1
            for b, score in index.search(vec, k=hits_per_fact, exclude_id=a.id,
                                         exclude_doc=getattr(a, "source_doc_id", None),
                                         min_score=min_score):
                # A question finds a fact; the bridge the judge reads is the
                # consequence closest in spirit, so a question carries the
                # first consequence (or itself, phrased as what A rules out).
                bridge = text if not is_question else (cons[0] if cons else f"(A rules out the answer to: {text})")
                cur = best.get(b.id)
                if cur is None or score > cur[2]:
                    best[b.id] = (b, bridge, score)
        if not best:
            return
        pairs = [(b, bridge, _pair_key(a, b)) for (b, bridge, _s) in sorted(best.values(), key=lambda t: -t[2])]
        keys = [pk for (_b, _br, pk) in pairs]
        memo = await _memoized_keys(store, customer_id, detector=DETECTOR, pair_keys=keys)
        result.memoized += sum(1 for pk in keys if pk in memo)
        for b, bridge, pk in pairs[:hits_per_fact]:
            if pk in memo:
                continue
            if await store.knowledge_conflict_exists(customer_id, pair_key=pk):
                continue
            async with sem:
                verdict, reason = await _judge(client, a, bridge, b, details=details,
                                               customer_id=customer_id, store=store, log=log)
            result.hits += 1
            # Q99 C: the verdict ties the consequence to B, whatever it was.
            cid = cons_ids.get(bridge)
            if cid and verdict in _CHECK_VERDICTS:
                try:
                    tie = await store.add_consequence_check(
                        customer_id, cid, fact_id=b.id, fact_crystal_id=b.crystal_id, verdict=verdict,
                    )
                    result.edges_written += 1
                    if tie.get("opened"):
                        result.entered_recall += 1
                except Exception as e:  # noqa: BLE001
                    log.warning("implication_scan.check_edge_failed", consequence=cid, error=str(e)[:160])
            if verdict in _MEMO_VERDICTS:
                judged_pairs.append((pk, verdict))
                continue
            if verdict != "CONTRADICTS":
                continue
            subject = _subject_of(a, b)
            reasoning = f"A implies: {bridge}" + (f" {reason}" if reason else "")
            conflict = await store.create_knowledge_conflict(
                customer_id,
                fact_a_id=a.id, fact_b_id=b.id,
                claim_a=a.claim_text or "", claim_b=b.claim_text or "",
                pair_key=pk, crystal_a_id=a.crystal_id, crystal_b_id=b.crystal_id,
                subject=subject, provenance_a=_provenance(a), provenance_b=_provenance(b),
                detector=DETECTOR, reasoning=reasoning[:2000], recommendation=None,
            )
            result.conflicts_found += 1
            try:
                await store.record_curation_event(
                    customer_id, event_type="conflict_found", subject_id=conflict.id,
                    label=(f"Found by reasoning - {subject}"[:256] if subject else "Found by reasoning"),
                )
            except Exception as e:  # noqa: BLE001 — the feed is a witness, never a gate
                log.warning("implication_scan.event_failed", error=str(e)[:120])

    await asyncio.gather(*[_probe(a) for a in todo])
    await _remember_verdicts(store, customer_id, detector=PROBE_DETECTOR,
                             verdicts=[(fid, "PROBED") for fid in probed_ok], log=log)
    await _remember_verdicts(store, customer_id, detector=DETECTOR, verdicts=judged_pairs, log=log)
    log.info("implication_scan.completed", customer_id=customer_id,
             facts_considered=result.facts_considered, facts_probed=result.facts_probed,
             checks=result.checks, dropped=result.dropped,
             searches=result.searches, hits=result.hits, conflicts_found=result.conflicts_found,
             memoized=result.memoized, skipped_probed=result.skipped_probed,
             budget_exhausted=result.budget_exhausted,
             consequences_written=result.consequences_written, edges_written=result.edges_written,
             entered_recall=result.entered_recall)
    return result
