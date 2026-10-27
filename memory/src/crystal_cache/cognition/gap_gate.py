"""The gap gate (Q134 A, 2026-10-26): the only door a knowledge gap goes through.

Seven writers opened gaps. One (the gap-discovery scan) asked whether the
bank already answered the question; every check compared only against OPEN
gaps, so a question answered or dismissed yesterday came back today; and the
answer check searched with the question itself, which sits far from the
statement that answers it ("What is the name of the product?" found facts
about "the product", not the fact naming Kerning; the subject's key found it).

open_gap() decides in this order, cheapest first:

  1. Same question (no model): the question against EVERY gap of the
     customer, any status, by content words and, with an encoder, by
     meaning (cosine >= SAME_GAP_COSINE against each gap's stored vector).
     A match is not written: an open match is returned as it is; a filled
     or dismissed one stays settled (a person can reopen it).
  2. Already answered (vectors, then the large model): the bank is searched
     with three probes, the question, its content words and the gap's
     key/subject, merged by fact. No fact above ANSWER_FLOOR opens the gap
     with no model call. Otherwise the large model reads the candidates;
     when one answers, the gap is written born filled, with
     answered_by_fact_id (the edge from question to answer).
  3. Otherwise the gap opens, carrying its vector for step 1 next time.

Writers on a response path pass check_answer=False: step 1 runs inline,
step 2 is left to recheck_open_gaps(), which the cognition worker runs
before each discovery pass. The same sweep fills an open gap when a newer
fact answers it, so the edge is written once and later questions are cheap.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

SAME_GAP_COSINE = 0.85      # Q103's measured threshold for "the same question"
ANSWER_FLOOR = 0.45         # below this no fact is a plausible answer: open without a model call
ANSWER_HITS_PER_PROBE = 8
ANSWER_CANDIDATES = 12
JUDGE_TIER = "large"
JUDGE_ORIGIN = "scan_gap_discovery"
_CLAIM_TRUNC = 400
HOT_PATH_GAPS = 300

# Process wiring: runtime.build_core_runtime registers the shared encoder and
# vector index so every writer, however deep, reaches the same search.
_RUNTIME: dict[str, Any] = {"encoder": None, "vector_index": None}


def set_gap_runtime(*, encoder: Any = None, vector_index: Any = None) -> None:
    _RUNTIME["encoder"] = encoder
    _RUNTIME["vector_index"] = vector_index


def _runtime(encoder: Any = None, vector_index: Any = None) -> tuple[Any, Any]:
    return encoder or _RUNTIME["encoder"], vector_index or _RUNTIME["vector_index"]


@dataclass
class GateResult:
    opened: bool                    # a new open gap was written
    verdict: str                    # opened | duplicate | answered | invalid
    gap: Any = None                 # the KnowledgeGap written (open or born filled), if any
    existing_gap_id: Optional[str] = None
    fact_id: Optional[str] = None   # the answering fact
    detail: dict = field(default_factory=dict)

    @property
    def id(self) -> Optional[str]:
        return getattr(self.gap, "id", None) or self.existing_gap_id


# Sources whose `missing` is a template: the question is elsewhere.
_DEMAND_SOURCES = frozenset({"uncited_answer", "navigation_miss"})
_TOPIC_SOURCES = frozenset({"topic_spec"})


def question_of(missing: str, triggering_query: Optional[str] = None, source: Optional[str] = None,
                subject: Optional[str] = None) -> str:
    """What the gap asks. Response-path writers record boilerplate in
    `missing` ("the bank may lack coverage for this query") and the demand
    in triggering_query; operator topics record a template around the topic.
    Deduping on a template would merge every such gap into one, so those
    compare on the demand or the topic; everything else on `missing`."""
    if source in _DEMAND_SOURCES and (triggering_query or "").strip():
        return triggering_query.strip()
    if source in _TOPIC_SOURCES and (subject or "").strip():
        return subject.strip()
    return (missing or "").strip()


def content_phrase(text: str) -> str:
    """The question without its question words: closer in meaning to the
    statement that answers it than the question is."""
    from ..scan.gap_discovery import _question_tokens
    words = "".join(ch.lower() if ch.isalnum() else " " for ch in (text or "")).split()
    keep = _question_tokens(text)
    return " ".join(w for w in words if w in keep)


def key_phrase(full_key: Optional[str], subject: Optional[str]) -> str:
    parts = [p.strip() for p in (full_key or "").split("|") if p.strip()]
    if subject and subject.strip() not in parts:
        parts.append(subject.strip())
    return " ".join(parts)


async def _encode(encoder: Any, text: str) -> Optional[list[float]]:
    if encoder is None or not (text or "").strip():
        return None
    from ..encoding.executor import Priority, encode_native_async
    try:
        vec = await encode_native_async(encoder, text, priority=Priority.BULK)
    except Exception as e:  # noqa: BLE001 - the gate degrades to word matching
        logger.warning("gap_gate.encode_failed", error=str(e)[:160])
        return None
    return [float(x) for x in (vec.tolist() if hasattr(vec, "tolist") else vec)]


def _same(question: str, qvec: Optional[list[float]], row: dict) -> Optional[float]:
    """1.0 for the same words, the cosine for the same meaning, else None."""
    from ..scan.gap_discovery import same_question
    from ..scan.implication import _cosine
    theirs = question_of(row.get("missing") or "", row.get("triggering_query"), row.get("source"), row.get("subject"))
    if theirs and same_question(question, theirs):
        return 1.0
    if qvec is not None and row.get("vector"):
        c = _cosine(qvec, row["vector"])
        if c >= SAME_GAP_COSINE:
            return c
    return None


JUDGE_SYSTEM = (
    "You decide whether a knowledge base already answers a question.\n\n"
    "You get a QUESTION and numbered FACTS found by searching the knowledge "
    "base. A fact answers the question when a person asking it would be "
    "satisfied by that fact, including a fact saying it is not decided or not "
    "known yet. A fact that is only about the same topic does not answer it. "
    "Return the number of the fact that answers it best, or -1 when none does."
)
JUDGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "integer"}, "why": {"type": "string"}},
    "required": ["answer", "why"],
    "additionalProperties": False,
}


async def find_answer(store: Any, customer_id: str, *, question: str, full_key: Optional[str] = None,
                      subject: Optional[str] = None, qvec: Optional[list[float]] = None,
                      encoder: Any = None, vector_index: Any = None, llm: Any = None,
                      judged: Optional[list[str]] = None) -> dict[str, Any]:
    """Step 2. {checked, verdict: answered|open|unchecked, fact_id,
    crystal_id, score, candidates, judged}. `judged`: fact ids a model
    already read for this question; when the candidates hold nothing new,
    no model is called (a recheck only pays when a new fact arrived).
    Never raises."""
    encoder, vector_index = _runtime(encoder, vector_index)
    if encoder is None or vector_index is None:
        return {"checked": False, "verdict": "unchecked", "reason": "no_vectors"}
    probes = [question, content_phrase(question), key_phrase(full_key, subject)]
    best: dict[str, tuple[float, str]] = {}
    for i, probe in enumerate(p for p in probes if p.strip()):
        vec = qvec if (i == 0 and qvec is not None) else await _encode(encoder, probe)
        if vec is None:
            continue
        try:
            import numpy as np
            hits = await vector_index.search_facts(customer_id=customer_id, query_vector=np.asarray(vec, dtype=np.float32),
                                                   k=ANSWER_HITS_PER_PROBE, with_keys=True)
        except Exception as e:  # noqa: BLE001
            logger.warning("gap_gate.search_failed", customer_id=customer_id, error=str(e)[:160])
            continue
        for h in hits:
            fact_id, crystal_id, score = h[0], h[1], float(h[3])
            if fact_id not in best or score > best[fact_id][0]:
                best[fact_id] = (score, crystal_id)
    ranked = sorted(best.items(), key=lambda kv: -kv[1][0])[:ANSWER_CANDIDATES]
    top = ranked[0][1][0] if ranked else 0.0
    if not ranked or top < ANSWER_FLOOR:
        return {"checked": True, "verdict": "open", "score": round(top, 3), "candidates": 0}
    seen = set(judged or [])
    if seen and all(fid in seen for fid, _ in ranked if best[fid][0] >= ANSWER_FLOOR):
        return {"checked": True, "verdict": "open", "score": round(top, 3), "candidates": 0,
                "judged": list(seen)[:60], "reason": "no_new_facts"}
    claims = await store.list_fact_claims(customer_id, [fid for fid, _ in ranked])
    numbered = [(fid, claims[fid]) for fid, _ in ranked if fid in claims and (claims[fid].get("claim") or "").strip()]
    if not numbered:
        return {"checked": True, "verdict": "open", "score": round(top, 3), "candidates": 0}
    facts_text = "\n".join(f"{i}. {c['claim'][:_CLAIM_TRUNC]}" + (f"  [{c['key']}]" if c.get("key") else "")
                           for i, (_fid, c) in enumerate(numbered))
    from ..llm import get_llm_client
    from ..scan._seam import metered_call
    try:
        raw = await metered_call(llm or get_llm_client(), customer_id=customer_id, origin=JUDGE_ORIGIN,
                                 system=JUDGE_SYSTEM, user=f"QUESTION: {question}\n\nFACTS:\n{facts_text}",
                                 max_tokens=300, tier=JUDGE_TIER, store=store, json_schema=JUDGE_SCHEMA)
        out = json.loads((raw or "").strip())
        pick = int(out.get("answer", -1))
    except Exception as e:  # noqa: BLE001 - an unread question stays open
        logger.warning("gap_gate.judge_failed", customer_id=customer_id, error=str(e)[:160])
        return {"checked": False, "verdict": "unchecked", "reason": "judge_failed", "score": round(top, 3),
                "candidates": len(numbered)}
    read = list(dict.fromkeys([*(judged or []), *(fid for fid, _ in numbered)]))[-60:]
    if 0 <= pick < len(numbered):
        fid, claim = numbered[pick]
        return {"checked": True, "verdict": "answered", "fact_id": fid, "crystal_id": claim.get("crystal_id"),
                "score": round(best[fid][0], 3), "candidates": len(numbered), "judged": read,
                "why": str(out.get("why") or "")[:300]}
    return {"checked": True, "verdict": "open", "score": round(top, 3), "candidates": len(numbered), "judged": read}


async def open_gap(store: Any, customer_id: str, *, missing: str, domain: Optional[str] = None,
                   subject: Optional[str] = None, priority: str = "medium", source: str = "llm_observation",
                   full_key: Optional[str] = None, triggering_query: Optional[str] = None,
                   disposition: Optional[str] = None, check_answer: bool = True,
                   encoder: Any = None, vector_index: Any = None, llm: Any = None,
                   known: Optional[list[dict]] = None) -> GateResult:
    """Write a gap only when it is new and not already answered. `known`:
    the customer's gaps from store.list_gate_gaps (a writer opening several
    passes it once and the gate appends to it)."""
    question = question_of(missing, triggering_query, source, subject)
    if not question:
        return GateResult(False, "invalid")
    encoder, vector_index = _runtime(encoder, vector_index)
    # A response-path writer compares against the newest gaps only (each
    # row carries a vector to decode); the background sweep sees them all.
    rows = known if known is not None else await store.list_gate_gaps(
        customer_id, limit=(2000 if check_answer else HOT_PATH_GAPS))
    qvec = await _encode(encoder, question)
    for row in rows:
        score = _same(question, qvec, row)
        if score is not None:
            logger.info("gap_gate.duplicate", customer_id=customer_id, source=source, of=row["id"],
                        status=row.get("status"), score=round(score, 3))
            return GateResult(False, "duplicate", existing_gap_id=row["id"],
                              detail={"status": row.get("status"), "score": round(score, 3)})
    now = datetime.now(timezone.utc).isoformat()
    found: dict[str, Any] = {"checked": False, "verdict": "unchecked", "reason": "deferred"}
    if check_answer:
        found = await find_answer(store, customer_id, question=question, full_key=full_key, subject=subject,
                                  qvec=qvec, encoder=encoder, vector_index=vector_index, llm=llm)
    gate = {**found, "at": now}
    answered = found.get("verdict") == "answered"
    gap = await store.create_knowledge_gap(
        customer_id, domain=domain, subject=subject, missing=missing, priority=priority, source=source,
        full_key=full_key, triggering_query=triggering_query, disposition=disposition, vector=qvec, gate=gate,
        status="filled" if answered else "open",
        answered_by_fact_id=found.get("fact_id") if answered else None,
        filled_by_crystal_id=found.get("crystal_id") if answered else None)
    rows.insert(0, {"id": gap.id, "status": gap.status, "missing": missing, "triggering_query": triggering_query,
                    "subject": subject, "source": source, "full_key": full_key, "vector": qvec, "gate": gate})
    if answered:
        logger.info("gap_gate.answered", customer_id=customer_id, source=source, gap_id=gap.id,
                    fact_id=found.get("fact_id"), score=found.get("score"))
        return GateResult(False, "answered", gap=gap, fact_id=found.get("fact_id"), detail=found)
    return GateResult(True, "opened", gap=gap, detail=found)


async def recheck_open_gaps(store: Any, customer_id: str, *, limit: int = 20, encoder: Any = None,
                            vector_index: Any = None, llm: Any = None) -> dict[str, int]:
    """The deferred half of the gate and the fill-on-new-fact sweep. Each
    pass takes up to `limit` open gaps: ones never checked (response-path
    gaps, rows from before the gate) first, then the longest unchecked
    since. A fact that answers one fills it with the edge. Returns counts;
    `calls` is the model calls spent (for the cognition budget)."""
    encoder, vector_index = _runtime(encoder, vector_index)
    out = {"checked": 0, "filled": 0, "calls": 0, "vectors": 0}
    if encoder is None or vector_index is None:
        return out
    rows = [r for r in await store.list_gate_gaps(customer_id) if r.get("status") == "open"]

    def order(r: dict) -> tuple:
        g = r.get("gate") or {}
        return (bool(g.get("checked")), str(g.get("at") or ""))
    for row in sorted(rows, key=order)[:max(0, limit)]:
        question = question_of(row.get("missing") or "", row.get("triggering_query"), row.get("source"),
                               row.get("subject"))
        qvec = row.get("vector")
        if not qvec:
            qvec = await _encode(encoder, question)
            if qvec is not None:
                out["vectors"] += 1
        found = await find_answer(store, customer_id, question=question, full_key=row.get("full_key"),
                                  subject=row.get("subject"), qvec=qvec, encoder=encoder,
                                  vector_index=vector_index, llm=llm, judged=(row.get("gate") or {}).get("judged"))
        if found.get("candidates"):
            out["calls"] += 1
        gate = {**found, "at": datetime.now(timezone.utc).isoformat()}
        if found.get("verdict") == "answered":
            await store.update_gap_gate(row["id"], customer_id, vector=qvec, gate=gate, status="filled",
                                        answered_by_fact_id=found.get("fact_id"),
                                        filled_by_crystal_id=found.get("crystal_id"))
            out["filled"] += 1
            logger.info("gap_gate.filled_by_existing_fact", customer_id=customer_id, gap_id=row["id"],
                        fact_id=found.get("fact_id"))
        else:
            await store.update_gap_gate(row["id"], customer_id, vector=qvec, gate=gate)
        if found.get("checked"):
            out["checked"] += 1
    return out
