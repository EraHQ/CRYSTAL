"""The Mind tab's one read (2026-10-21).

GET /admin/api/customers/{customer_id}/mind

Everything the scene draws, in one call, laid out server-side so the
browser only paints: regions and every fact's position (mind/layout.py,
from the facts' own vectors), the sources that fed each region, the
bonds between clusters (crystal_edges), the system's own thoughts
(contradictions and duplicates, assumptions with their parents, gaps),
what it did in the window (runs, recalls, curation events) and how hard
each process worked per hour (the model ledger, tokens only; dollars
stay on the platform admin's spend view).

Tenant-pathed like the Activity routes: the guard scopes
/admin/api/customers/{cid}/* to the tenant's own id. The layout is
cached per bank generation so a reload is instant until the bank
changes.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from typing import Annotated, Any, Optional

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import JSONResponse

from ..infrastructure import MetadataStore
from ..infrastructure.metadata_store import get_metadata_store
from ..infrastructure.metadata_store_assumption_ext import parse_assumption_tags
from ..ingress.limits import ID64
from ..mind.layout import layout_facts

router = APIRouter()

_MAX_FACTS = 4000
_TIER = {"whitelist": "trusted", "neutral": "reviewed", "quarantine": "new", "blacklist": "wrong"}
_LANE_OF_ORIGIN = {
    "ingest": "Reading", "document_extraction": "Extracting", "extraction": "Extracting",
    "code_description": "Extracting", "bonding": "Clustering", "crystallization": "Clustering",
    "scan_contradiction": "Contradictions", "scan_dedup": "Duplicates", "scan_gap_discovery": "Gaps",
    "scan_implication": "Reasoning",
    "assumptions": "Assumptions", "cognition": "Research", "research": "Research",
    "scheduled": "Research", "agent": "Recalls", "chat": "Recalls", "recall": "Recalls",
}
_LANES = ["Reading", "Extracting", "Clustering", "Contradictions", "Duplicates", "Reasoning", "Gaps", "Assumptions", "Research", "Recalls", "Other"]

# (customer_id, bank_generation, fact_count) -> layout payload pieces
_layout_cache: dict[tuple[str, int, int], dict[str, Any]] = {}


def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=timezone.utc)


def _iso(dt: Optional[datetime]) -> Optional[str]:
    dt = _aware(dt)
    return dt.isoformat() if dt else None


def _head(text: Optional[str], n: int = 160) -> str:
    t = (text or "").strip().replace("\n", " ")
    return t if len(t) <= n else t[: n - 1] + "…"


def _lane(origin: Optional[str]) -> str:
    o = (origin or "").lower()
    for key, lane in _LANE_OF_ORIGIN.items():
        if key in o:
            return lane
    return "Other"


async def _layout(store: MetadataStore, customer_id: str, facts: list[Any]) -> dict[str, Any]:
    try:
        gen = int(await store.bank_generation(customer_id))
    except Exception:  # noqa: BLE001 — the cache is a courtesy
        gen = -1
    key = (customer_id, gen, len(facts))
    hit = _layout_cache.get(key) if gen >= 0 else None
    if hit is not None:
        return hit
    lay = layout_facts([list(f.vector or []) for f in facts], [f.prompt_text for f in facts])
    out = {
        "positions": lay.positions.tolist(),
        "region_of": lay.region_of.tolist(),
        "regions": lay.regions,
    }
    if gen >= 0:
        if len(_layout_cache) > 64:
            _layout_cache.clear()
        _layout_cache[key] = out
    return out


@router.get("/admin/api/customers/{customer_id}/mind")
async def mind(
    request: Request,
    customer_id: ID64,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
    hours: Annotated[int, Query(ge=1, le=24 * 14)] = 24,
) -> JSONResponse:
    now = datetime.now(timezone.utc)
    since = now - timedelta(hours=hours)

    # --- the bank: facts (claims only), their clusters, their documents ---
    crystals = {c.id: c for c in await store.list_crystals_for_customer(customer_id)}
    all_facts = await store.list_recent_facts_for_customer(customer_id, limit=_MAX_FACTS)
    facts = [
        f for f in all_facts
        if (f.pair_type or "") != "content_chunk"
        and (f.claim_text or "").strip()
        and getattr(crystals.get(f.crystal_id), "crystal_type", "") != "assumption"
        and (getattr(f, "grating_strength", 1.0) or 0.0) > 0
    ]
    lay = await _layout(store, customer_id, facts)
    region_of = lay["region_of"]
    positions = lay["positions"]
    fact_index = {f.id: i for i, f in enumerate(facts)}

    docs = {d.id: d for d in await store.list_document_uploads(customer_id, limit=500)}
    doc_regions: dict[str, Counter] = defaultdict(Counter)
    crystal_region: dict[str, Counter] = defaultdict(Counter)
    crystal_pos: dict[str, list[list[float]]] = defaultdict(list)
    fact_rows: list[dict[str, Any]] = []
    for i, f in enumerate(facts):
        c = crystals.get(f.crystal_id)
        tier = _TIER.get(getattr(c, "quality_tier", "quarantine") or "quarantine", "new")
        r = int(region_of[i])
        fact_rows.append({
            "id": f.id,
            "crystal_id": f.crystal_id,
            "region": r,
            "pos": [round(x, 2) for x in positions[i]],
            "tier": tier,
            "hits": int(getattr(f, "hit_count", 0) or 0),
            "key": f.prompt_text or "",
            "claim": _head(f.claim_text, 220),
            "source_kind": f.source_kind,
            "doc_id": f.source_doc_id,
            "learned_at": _iso(getattr(f, "created_at", None)),
        })
        crystal_region[f.crystal_id][r] += 1
        crystal_pos[f.crystal_id].append(positions[i])
        if f.source_doc_id:
            doc_regions[f.source_doc_id][r] += 1

    regions = []
    for reg in lay["regions"]:
        idx = reg["index"]
        members = [fr for fr in fact_rows if fr["region"] == idx]
        regions.append({
            "id": idx,
            "name": reg["name"],
            "center": [round(x, 2) for x in reg["center"]],
            "facts": len(members),
            "clusters": len({m["crystal_id"] for m in members}),
            "trusted": sum(1 for m in members if m["tier"] == "trusted"),
            "hits": sum(m["hits"] for m in members),
        })
    region_center = {r["id"]: r["center"] for r in regions}

    def _crystal_center(cid: str) -> Optional[list[float]]:
        ps = crystal_pos.get(cid)
        if not ps:
            return None
        n = len(ps)
        return [round(sum(p[k] for p in ps) / n, 2) for k in range(3)]

    def _crystal_region(cid: str) -> Optional[int]:
        cr = crystal_region.get(cid)
        return cr.most_common(1)[0][0] if cr else None

    sources = []
    for doc_id, counter in doc_regions.items():
        d = docs.get(doc_id)
        r = counter.most_common(1)[0][0]
        sources.append({
            "id": doc_id,
            "label": (getattr(d, "label", None) or doc_id)[:120],
            "region": r,
            "facts": int(sum(counter.values())),
            "learned_at": _iso(getattr(d, "created_at", None)) if d else None,
        })

    # --- bonds: the funnel graph between clusters ---
    bonds = []
    try:
        for e in await store.list_candidate_edges(customer_id, limit=300):
            a, b = e["crystal_a_id"], e["crystal_b_id"]
            pa, pb = _crystal_center(a), _crystal_center(b)
            if pa is None or pb is None:
                continue
            bonds.append({
                "a": a, "b": b, "a_pos": pa, "b_pos": pb,
                "region_a": _crystal_region(a), "region_b": _crystal_region(b),
                "type": e.get("edge_type"), "weight": round(float(e.get("weight") or 0.0), 3),
            })
    except Exception:  # noqa: BLE001 — a bank without a funnel graph still draws
        bonds = []

    # --- the system's own thoughts ---
    conflicts = []
    rows = list(await store.list_knowledge_conflicts(customer_id, status="open", limit=200))
    rows += [c for c in await store.list_knowledge_conflicts(customer_id, limit=200)
             if c.status != "open" and _aware(c.resolved_at or c.created_at) >= since]
    for c in rows:
        ia, ib = fact_index.get(c.fact_a_id), fact_index.get(c.fact_b_id)
        conflicts.append({
            "id": c.id,
            "kind": "duplicate" if c.detector == "dedup_scan" else "contradiction",
            "detector": c.detector,
            "subject": c.subject,
            "status": c.status,
            "resolution": c.resolution,
            "a": {"fact_id": c.fact_a_id, "claim": _head(c.claim_a), "pos": fact_rows[ia]["pos"] if ia is not None else None,
                  "region": fact_rows[ia]["region"] if ia is not None else _crystal_region(c.crystal_a_id or "")},
            "b": {"fact_id": c.fact_b_id, "claim": _head(c.claim_b), "pos": fact_rows[ib]["pos"] if ib is not None else None,
                  "region": fact_rows[ib]["region"] if ib is not None else _crystal_region(c.crystal_b_id or "")},
            "reasoning": c.reasoning,
            "recommendation": c.recommendation,
            "found_at": _iso(c.created_at),
        })

    assumptions = []
    for a in await store.list_assumption_crystals(customer_id, limit=200):
        tags = parse_assumption_tags(list(a.get("diagnostic_tags") or []))
        parents = []
        for chain in await store.list_chains_from_source(a["id"]):
            pid = chain.target_crystal_id
            if pid in crystals:
                parents.append({
                    "crystal_id": pid,
                    "label": (crystals[pid].summary_text or crystals[pid].source_path or pid)[:120],
                    "pos": _crystal_center(pid),
                    "region": _crystal_region(pid),
                })
        state = "invalidated" if a.get("quality_tier") == "blacklist" else "approved" if not a.get("recall_gated") else "waiting"
        assumptions.append({
            "id": a["id"],
            "statement": a.get("statement") or "",
            "confidence": tags.get("confidence"),
            "rationale": tags.get("rationale"),
            "state": state,
            "parents": parents,
            "formed_at": a.get("created_at"),
        })

    gaps = []
    region_by_name = {r["name"].lower(): r["id"] for r in regions}
    for g in await store.list_knowledge_gaps(customer_id, status="open", limit=200):
        widest = ((g.full_key or g.subject or g.domain or "").split("|")[0] or "").strip().lower()
        gaps.append({
            "id": g.id,
            "missing": _head(g.missing, 240),
            "subject": g.subject,
            "priority": g.priority,
            "source": g.source,
            "disposition": g.disposition,
            "region": region_by_name.get(widest),
            "opened_at": _iso(g.created_at),
        })

    # --- what it did: runs, recalls, curation events in the window ---
    runs = []
    for t in await store.list_cognition_tasks(customer_id, limit=60):
        when = _aware(t.completed_at or t.started_at or t.created_at)
        if when is None or when < since:
            continue
        payload = t.payload or {}
        title = payload.get("title") or payload.get("topic") or payload.get("question") or payload.get("goal") or t.task_type
        result = t.result or {}
        runs.append({
            "id": t.id,
            "type": t.task_type,
            "origin": payload.get("origin") or payload.get("trigger") or None,
            "title": _head(str(title), 140),
            "status": t.status,
            "outcome": result.get("outcome") if isinstance(result, dict) else None,
            "summary": _head(str(result.get("summary") or result.get("deliverable") or ""), 240) if isinstance(result, dict) else "",
            "started_at": _iso(t.started_at or t.created_at),
            "finished_at": _iso(t.completed_at),
            "result_crystal_id": t.result_crystal_id,
            "region": _crystal_region(t.result_crystal_id) if t.result_crystal_id else None,
            "landing": _crystal_center(t.result_crystal_id) if t.result_crystal_id else None,
        })

    recalls = []
    _total, logs = await store.list_query_logs_for_customer(customer_id, limit=150)
    for q in logs:
        ts = _aware(q.timestamp)
        if ts is None or ts < since:
            continue
        touched = [fact_index[fid] for fid in (q.matched_facts or []) if fid in fact_index][:12]
        counter = Counter(fact_rows[i]["region"] for i in touched)
        recalls.append({
            "id": q.id,
            "question": _head(q.query_text, 140),
            "at": _iso(q.timestamp),
            "client": getattr(q, "client", None),
            "model": getattr(q, "model", None),
            "match": q.match_type,
            "latency_ms": q.latency_ms,
            "region": counter.most_common(1)[0][0] if counter else None,
            "touched": [fact_rows[i]["id"] for i in touched],
        })

    events = []
    for ev in await store.list_curation_events(customer_id, limit=300):
        ts = _aware(ev.get("created_at"))
        if ts is None or ts < since:
            continue
        events.append({
            "id": ev.get("id"), "type": ev.get("event_type"), "label": ev.get("label"),
            "subject_id": ev.get("subject_id"), "at": ts.isoformat(),
        })

    # --- how hard each process worked, per hour (tokens, never dollars) ---
    buckets: dict[str, dict[int, dict[str, int]]] = {lane: defaultdict(lambda: {"calls": 0, "tokens": 0}) for lane in _LANES}
    for call in await store.list_llm_calls(customer_id, since=since, limit=4000):
        at = call.get("created_at")
        try:
            ts = datetime.fromisoformat(at) if isinstance(at, str) else _aware(at)
        except ValueError:
            continue
        if ts is None:
            continue
        hour = int((ts - since).total_seconds() // 3600)
        if hour < 0 or hour >= hours:
            continue
        b = buckets[_lane(call.get("origin"))][hour]
        b["calls"] += 1
        b["tokens"] += int(call.get("input_tokens") or 0) + int(call.get("output_tokens") or 0)
    lanes = [
        {"name": lane, "hours": [dict(buckets[lane].get(h) or {"calls": 0, "tokens": 0}) for h in range(hours)]}
        for lane in _LANES if any(buckets[lane].values()) or lane != "Other"
    ]

    return JSONResponse(content={
        "generated_at": now.isoformat(),
        "window": {"since": since.isoformat(), "until": now.isoformat(), "hours": hours},
        "totals": {
            "facts": len(fact_rows), "clusters": len(crystal_pos), "regions": len(regions), "sources": len(sources),
            "contradictions": sum(1 for c in conflicts if c["kind"] == "contradiction" and c["status"] == "open"),
            "duplicates": sum(1 for c in conflicts if c["kind"] == "duplicate" and c["status"] == "open"),
            "assumptions_waiting": sum(1 for a in assumptions if a["state"] == "waiting"),
            "gaps": len(gaps), "runs": len(runs), "recalls": len(recalls),
        },
        "regions": regions,
        "facts": fact_rows,
        "sources": sources,
        "bonds": bonds,
        "conflicts": conflicts,
        "assumptions": assumptions,
        "gaps": gaps,
        "runs": runs,
        "recalls": recalls,
        "events": events,
        "lanes": lanes,
    })
