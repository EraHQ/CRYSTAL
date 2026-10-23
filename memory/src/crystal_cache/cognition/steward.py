"""The Steward (Q100 A, ratified 2026-10-22, built 2026-10-23).

Anthony: "We need to let a model have the hands. Basically an agent that
can fully control a crystal account. That way users just guide the one
agent and the agent keeps the mind going with a simple task of keep
improving my crystal bank until my cost runs out. Then ask for a
refill." And: "it's time to start loosing [control] and make a statement
that this is how we think this system should work."

One standing session per enabled tenant per idle cycle: the agent loop
(agent/agent.py) with a FRESH registry holding the curation toolset and
nothing else, the goal "improve this bank", a tool-call budget and a
wall clock, run on the tenant's background lane. Every write tool lands
a row in steward_actions with the inverse that undoes it (24 hours, from
the Activity page) and a curation event, so nothing the Steward does is
silent or final. Its final message is the session's brief: what it did,
why, what it left alone.

What it may do (and undo):
  decide_conflict     settle an open conflict (superseded / qualified /
                      dismissed)                         -> reopen
  approve_assumption  approve an assumption or consequence (never one
                      a fact contradicts)                 -> re-gate, restore tier
  reject_assumption   invalidate one                      -> restore
  research_gap        queue background research for a gap -> cancel the task
  dismiss_gap         close a gap not worth researching    -> reopen
  reprobe_fact        let the implication scan probe a fact again (no undo)
What it cannot do: delete, blacklist a fact directly, write facts, spend
beyond the plan (the tenant door and the daily session cap stop it; at
the wall it asks for a refill once a day through a steward_paused event).

Pieces: build_steward_registry (tools), STEWARD_CHARTER (the system
prompt), run_steward_session (one session, metered under origin
"steward"), undo_steward_action (the inverse), and the worker pass in
workers/cognition.py. Approval rules (system_rules) keep running beside
it; the Steward is a second pair of hands, not a bypass.
"""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from typing import Any, Optional

import structlog

from ..config import get_settings
from ..cost.emit import record_model_call
from ..infrastructure.metadata_store_assumption_ext import assumption_state, parse_assumption_tags

logger = structlog.get_logger(__name__)

STEWARD_ORIGIN = "steward"
_GRACE_SECONDS = 90.0
_MODEL_CEILING = 150.0
_TOOL_WALL = 60.0
_REASON_MIN = 12
_RECENT_DAYS = 7

CONFLICT_RESOLUTIONS = ("superseded", "qualified", "dismissed")


def steward_charter(budget_calls: int, budget_actions: int) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return f"""You are the Steward of one person's memory bank (Crystal Cache). TODAY IS {today} (UTC). Your standing task: improve this bank. Nobody is watching this session; everything you do is recorded, visible on their Activity page, and undoable for 24 hours.

How to work:
1. Call bank_status first, then recent_actions (never repeat an action a person undid).
2. Look at what waits: open_conflicts, waiting_assumptions, open_gaps. Read what the bank already says (crystal_search, crystal_key_scan) before judging anything.
3. Act only where the bank itself decides the matter:
   - decide_conflict when one claim is plainly outdated (a dated, more recent, more specific claim supersedes an older one: superseded, with the loser), when both are true under different conditions (qualified), or when the pair was never a contradiction (dismissed). Leave open any conflict that needs the person's own knowledge, and say so in the brief.
   - approve_assumption when the stated facts around it agree with it and nothing contradicts it. reject_assumption when the bank shows it is wrong. Leave the rest waiting.
   - research_gap for a gap marked researchable that the web can answer; dismiss_gap for one not worth researching (trivia, already answered, out of scope). Never research the same gap twice.
   - reprobe_fact when a fact's consequences deserve another look (new facts arrived around it).
4. Every write needs a reason a person will read later; write it as you would explain it to them.
5. Budget: {budget_calls} tool calls and at most {budget_actions} actions this session. Do not spend calls re-reading what you already have. Stop early when nothing more is safe to do.

End with the session's brief in three short sections, plain prose, no markdown headings:
Did: what you changed and why (one line each).
Left alone: what waits for the person and what it needs from them.
Next: what you would do next session, if anything.
If you did nothing, say what you looked at and why nothing was safe to change."""


class StewardSession:
    """Everything the tools close over for one session."""

    def __init__(self, *, store: Any, customer: Any, run_id: str, encoder: Any, fact_store: Any,
                 max_actions: int, log: Any = None) -> None:
        self.store = store
        self.customer = customer
        self.customer_id = customer.id
        self.run_id = run_id
        self.encoder = encoder
        self.fact_store = fact_store
        self.max_actions = max_actions
        self.actions: list[dict[str, Any]] = []
        self.log = log or logger

    def _cap(self) -> Optional[dict[str, Any]]:
        if len(self.actions) >= self.max_actions:
            return {"ok": False, "code": "session_action_cap",
                    "error": f"This session already took {self.max_actions} actions; write the brief and stop."}
        return None

    @staticmethod
    def _reason_ok(reason: str) -> Optional[dict[str, Any]]:
        if len((reason or "").strip()) < _REASON_MIN:
            return {"ok": False, "code": "reason_required",
                    "error": "Give a reason a person will read later (one or two sentences)."}
        return None

    async def _record(self, *, action: str, subject_kind: str, subject_id: str, label: str,
                      reason: str, inverse: Optional[dict[str, Any]], undoable: bool = True) -> dict[str, Any]:
        row = await self.store.record_steward_action(
            self.customer_id, run_id=self.run_id, action=action, subject_kind=subject_kind,
            subject_id=subject_id, label=label, reason=reason, inverse=inverse, undoable=undoable,
        )
        self.actions.append({"id": row["id"], "action": action, "subject_kind": subject_kind,
                             "subject_id": subject_id, "label": label, "reason": reason})
        try:
            await self.store.record_curation_event(
                self.customer_id, event_type="steward_action", subject_id=subject_id,
                label=label[:256], payload={"action": action, "action_id": row["id"], "run_id": self.run_id, "reason": reason[:500]},
            )
        except Exception:  # noqa: BLE001 — the feed is a witness, never a gate
            self.log.debug("steward.event_failed", exc_info=True)
        return row


def _head(s: Optional[str], n: int = 240) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def build_steward_registry(session: StewardSession):
    """A FRESH registry with exactly the Steward's tools (structural
    enforcement, like the research worker's: no delete, no fact writes,
    no cognition_run, no scheduling of itself)."""
    from ..agent.tool_registry import Tool, ToolRegistry
    from .agentic import _WORKER_TOOLS
    from .retrieval_adapter import dispatch_cognition_retrieval

    store, cid = session.store, session.customer_id
    registry = ToolRegistry()

    def add(name: str, description: str, schema: dict, impl, returns: str = "") -> None:
        registry.register(Tool(name=name, description=description, contexts=frozenset({"agent"}),
                               parameters_schema=schema, impl=impl, returns_description=returns))

    # ---- reads ----
    async def bank_status(customer_id: str) -> dict:
        from ..control.admission import daily_capacity
        open_conf = await store.count_knowledge_conflicts(customer_id, status="open")
        rows = await store.list_assumption_crystals(customer_id, limit=300)
        waiting = [r for r in rows if assumption_state(r) == "waiting"]
        gaps = await store.list_knowledge_gaps(customer_id, status="open", limit=200)
        out = {
            "open_conflicts": int(open_conf),
            "waiting_assumptions": sum(1 for r in waiting if r.get("kind") != "consequence"),
            "waiting_consequences": sum(1 for r in waiting if r.get("kind") == "consequence"),
            "open_gaps": len(gaps),
            "researchable_gaps": sum(1 for g in gaps if (g.disposition or "") == "researchable"),
            "actions_this_session": len(session.actions),
            "actions_left_this_session": max(0, session.max_actions - len(session.actions)),
        }
        try:
            cap = await daily_capacity(store, session.customer)
            if cap.get("allowance"):
                out["capacity_today"] = {"state": cap["state"], "pct_used": cap["pct"]}
        except Exception:  # noqa: BLE001 — a courtesy
            pass
        return out
    add("bank_status", "Where this bank stands right now: what waits (conflicts, assumptions, consequences, gaps) and how much of this session's budget is left. Call it first.",
        {"type": "object", "properties": {}, "additionalProperties": False}, bank_status)

    async def recent_actions(customer_id: str, limit: int = 20) -> dict:
        since = datetime.now(timezone.utc) - timedelta(days=_RECENT_DAYS)
        rows = await store.list_steward_actions(customer_id, limit=max(1, min(int(limit or 20), 50)), since=since)
        return {"actions": [
            {"action": r["action"], "subject_kind": r["subject_kind"], "subject_id": r["subject_id"],
             "label": r["label"], "undone": bool(r["undone_at"]), "at": r["created_at"]}
            for r in rows
        ]}
    add("recent_actions", f"What the Steward did in the last {_RECENT_DAYS} days, newest first, with whether a person undid it. Never repeat an undone action and never research the same gap twice.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}, "additionalProperties": False}, recent_actions)

    async def open_conflicts(customer_id: str, limit: int = 10) -> dict:
        from ..scan.contradiction import fact_details_for, judge_context
        rows = await store.list_knowledge_conflicts(customer_id, status="open", limit=max(1, min(int(limit or 10), 30)))
        facts = []
        for c in rows:
            for fid in (c.fact_a_id, c.fact_b_id):
                f = await store.get_fact(fid)
                if f is not None:
                    facts.append(f)
        details = await fact_details_for(store, facts, log=session.log) if facts else {}
        by_id = {f.id: f for f in facts}
        out = []
        for c in rows:
            fa, fb = by_id.get(c.fact_a_id), by_id.get(c.fact_b_id)
            out.append({
                "conflict_id": c.id,
                "kind": "duplicate" if c.detector == "dedup_scan" else "contradiction",
                "found_by": c.detector,
                "subject": c.subject,
                "a": {"fact_id": c.fact_a_id, "claim": _head(c.claim_a, 400), "scope": judge_context(fa, details) if fa else ""},
                "b": {"fact_id": c.fact_b_id, "claim": _head(c.claim_b, 400), "scope": judge_context(fb, details) if fb else ""},
                "why": _head(c.reasoning, 500),
                "suggested": c.recommendation,
                "found_at": c.created_at.isoformat() if c.created_at else None,
            })
        return {"conflicts": out}
    add("open_conflicts", "Open conflicts, newest first: both claims with their scope (key path, source, date), why they were flagged, and any suggested resolution.",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "additionalProperties": False}, open_conflicts)

    async def waiting_assumptions(customer_id: str, kind: str = "any", limit: int = 10) -> dict:
        rows = await store.list_assumption_crystals(customer_id, limit=300)
        want = (kind or "any").lower()
        picked = []
        for r in rows:
            if assumption_state(r) != "waiting":
                continue
            if want != "any" and r.get("kind") != want:
                continue
            picked.append(r)
            if len(picked) >= max(1, min(int(limit or 10), 30)):
                break
        fact_ids = []
        for r in picked:
            t = parse_assumption_tags(r["diagnostic_tags"])
            if t.get("parent_fact_id"):
                fact_ids.append(t["parent_fact_id"])
            fact_ids += [c["fact_id"] for c in t.get("checks", [])]
        claims = await store.list_fact_claims(customer_id, fact_ids) if fact_ids else {}
        out = []
        for r in picked:
            t = parse_assumption_tags(r["diagnostic_tags"])
            item: dict[str, Any] = {
                "crystal_id": r["id"], "kind": r.get("kind"), "statement": _head(r.get("statement"), 400),
                "recallable": not r.get("recall_gated"), "confidence": t.get("confidence"), "why": _head(t.get("rationale"), 400),
            }
            if r.get("kind") == "consequence":
                pf = t.get("parent_fact_id")
                item["derived_from"] = _head(claims.get(pf, {}).get("claim"), 300) if pf else None
                item["checks"] = [{"verdict": c["verdict"], "claim": _head(claims.get(c["fact_id"], {}).get("claim"), 200)} for c in t.get("checks", [])]
                item["contradicted"] = t.get("contradicted_edges", 0) > 0
            else:
                parents = []
                for chain in await store.list_chains_from_source(r["id"]):
                    p = await store.get_crystal(chain.target_crystal_id)
                    if p is not None and p.customer_id == customer_id:
                        parents.append(_head(p.summary_text or p.source_path or p.id, 200))
                item["built_from"] = parents
            out.append(item)
        return {"assumptions": out}
    add("waiting_assumptions", "Assumptions and consequences waiting for review: the statement, what it was built from or derived from, how the bank's facts judged it (consequences), and whether anything contradicts it. kind: any | assumption | consequence.",
        {"type": "object", "properties": {"kind": {"type": "string", "enum": ["any", "assumption", "consequence"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "additionalProperties": False}, waiting_assumptions)

    async def open_gaps(customer_id: str, limit: int = 10) -> dict:
        gaps = await store.list_knowledge_gaps(customer_id, status="open", limit=max(1, min(int(limit or 10), 30)))
        return {"gaps": [
            {"gap_id": g.id, "missing": _head(g.missing, 300), "subject": g.subject, "key": g.full_key,
             "priority": g.priority, "disposition": g.disposition, "source": g.source,
             "asked": _head(g.triggering_query, 200), "opened_at": g.created_at.isoformat() if g.created_at else None}
            for g in gaps
        ]}
    add("open_gaps", "Open knowledge gaps: what is missing, under which key, how it was noticed, and whether background research can take it (disposition researchable).",
        {"type": "object", "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 30}}, "additionalProperties": False}, open_gaps)

    for spec in _WORKER_TOOLS:
        if spec["name"] not in ("crystal_search", "crystal_key_scan"):
            continue

        def _make(action_value: str):
            async def _impl(customer_id: str, **kwargs: Any) -> dict:
                return await dispatch_cognition_retrieval(
                    action_value=action_value, step_input=kwargs, customer_id=customer_id,
                    store=store, fact_store=session.fact_store, encoder=session.encoder,
                )
            return _impl
        add(spec["name"], spec["description"], spec["schema"], _make(spec["action"]))

    # ---- writes ----
    async def decide_conflict(customer_id: str, conflict_id: str, resolution: str, reason: str, loser: Optional[str] = None) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        if resolution not in CONFLICT_RESOLUTIONS:
            return {"ok": False, "code": "bad_resolution", "error": f"resolution must be one of {', '.join(CONFLICT_RESOLUTIONS)}"}
        if resolution == "superseded" and loser not in ("a", "b"):
            return {"ok": False, "code": "loser_required", "error": "superseded needs loser='a' or 'b' (the outdated claim)"}
        c = await store.get_knowledge_conflict(conflict_id)
        if c is None or c.customer_id != customer_id:
            return {"ok": False, "code": "not_found", "error": "No such open conflict."}
        if c.status != "open":
            return {"ok": False, "code": "not_open", "error": "That conflict is already settled."}
        try:
            updated = await store.apply_conflict_resolution(
                conflict_id, resolution=resolution, resolved_at=datetime.now(timezone.utc),
                loser=loser if resolution == "superseded" else None, customer_id=customer_id,
            )
        except ValueError as e:
            # detail-ok: apply_conflict_resolution's ValueError text is ours.
            return {"ok": False, "code": "refused", "error": str(e)}
        if updated is None:
            return {"ok": False, "code": "not_found", "error": "No such open conflict."}
        subject = c.subject or _head(c.claim_a, 60)
        word = {"superseded": "retired the outdated claim", "qualified": "kept both claims", "dismissed": "dismissed the pairing"}[resolution]
        row = await session._record(
            action="decide_conflict", subject_kind="conflict", subject_id=conflict_id,
            label=f"Steward settled a conflict - {subject}: {word}", reason=reason,
            inverse={"reopen_conflict": conflict_id, "resolution": resolution},
        )
        return {"ok": True, "action_id": row["id"], "conflict_id": conflict_id, "resolution": resolution,
                "retired_claim": (c.claim_a if loser == "a" else c.claim_b) if resolution == "superseded" else None}
    add("decide_conflict", "Settle an open conflict. superseded: one claim is outdated (pass loser 'a' or 'b'; that fact leaves recall, reversibly); qualified: both are true under different conditions; dismissed: never a real contradiction. Needs a reason. Leave a conflict open when only the person can know.",
        {"type": "object", "properties": {
            "conflict_id": {"type": "string", "maxLength": 64}, "resolution": {"type": "string", "enum": list(CONFLICT_RESOLUTIONS)},
            "loser": {"type": "string", "enum": ["a", "b"]}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["conflict_id", "resolution", "reason"], "additionalProperties": False}, decide_conflict)

    async def approve_assumption(customer_id: str, crystal_id: str, reason: str) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        c = await store.get_crystal(crystal_id)
        if c is None or c.customer_id != customer_id or c.crystal_type != "assumption":
            return {"ok": False, "code": "not_found", "error": "No such assumption."}
        if c.quality_tier == "blacklist":
            return {"ok": False, "code": "invalidated", "error": "That one was invalidated; it cannot be approved."}
        tags = parse_assumption_tags(list(c.diagnostic_tags or []))
        if tags.get("approved_by") or (c.build_method != "consequence" and not c.recall_gated):
            return {"ok": False, "code": "already_approved", "error": "Already approved."}
        edges = await store.count_consequence_edges(crystal_id)
        open_conf = await store.count_open_conflicts_for_crystal(customer_id, crystal_id)
        if edges.get("contradicted_by", 0) or open_conf:
            return {"ok": False, "code": "contradicted", "error": "A stated fact contradicts it (or a conflict is open on it); a person decides this one."}
        before = {"quality_tier": c.quality_tier, "recall_gated": bool(c.recall_gated)}
        standing = await store.approve_assumption_standing(customer_id, crystal_id, approved_by="steward")
        gap_filled = None
        try:
            gap_filled = await store.close_gap_for_approved_assumption(crystal_id, customer_id)
        except Exception:  # noqa: BLE001 — the gap close is a courtesy
            gap_filled = None
        label = f"Steward approved {'a consequence' if standing['kind'] == 'consequence' else 'an assumption'} - {_head(c.summary_text, 120)}"
        row = await session._record(
            action="approve_assumption", subject_kind="assumption", subject_id=crystal_id, label=label, reason=reason,
            inverse={**before, "gap_filled": gap_filled},
        )
        try:
            await store.record_curation_event(customer_id, event_type="assumption_approved", subject_id=crystal_id,
                                              label=f"Assumption approved by the Steward - {_head(c.summary_text, 180)}"[:256],
                                              payload={"by": "steward", "action_id": row["id"]})
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "quality_tier": standing["quality_tier"], "kind": standing["kind"]}
    add("approve_assumption", "Approve a waiting assumption or consequence: it enters recall and takes its parent's standing. Refused when a stated fact contradicts it or a conflict is open on it. Needs a reason.",
        {"type": "object", "properties": {"crystal_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["crystal_id", "reason"], "additionalProperties": False}, approve_assumption)

    async def reject_assumption(customer_id: str, crystal_id: str, reason: str) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        c = await store.get_crystal(crystal_id)
        if c is None or c.customer_id != customer_id or c.crystal_type != "assumption":
            return {"ok": False, "code": "not_found", "error": "No such assumption."}
        if c.quality_tier == "blacklist":
            return {"ok": False, "code": "already_invalidated", "error": "Already invalidated."}
        before = {"quality_tier": c.quality_tier, "recall_gated": bool(c.recall_gated)}
        await store.set_crystal_recall_gate(crystal_id, customer_id, True)
        await store.set_crystal_quality_tier(crystal_id, customer_id, "blacklist")
        kind = "a consequence" if c.build_method == "consequence" else "an assumption"
        row = await session._record(
            action="reject_assumption", subject_kind="assumption", subject_id=crystal_id,
            label=f"Steward rejected {kind} - {_head(c.summary_text, 120)}", reason=reason, inverse=before,
        )
        try:
            await store.record_curation_event(customer_id, event_type="assumption_invalidated", subject_id=crystal_id,
                                              label=f"Assumption rejected by the Steward - {_head(c.summary_text, 180)}"[:256],
                                              payload={"by": "steward", "action_id": row["id"]})
        except Exception:  # noqa: BLE001
            pass
        return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id}
    add("reject_assumption", "Invalidate a waiting assumption or consequence the bank's own facts show to be wrong. It leaves recall; a person can restore it for 24 hours. Needs a reason.",
        {"type": "object", "properties": {"crystal_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["crystal_id", "reason"], "additionalProperties": False}, reject_assumption)

    async def research_gap(customer_id: str, gap_id: str, reason: str) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        g = await store.get_knowledge_gap(gap_id)
        if g is None or g.customer_id != customer_id or g.status != "open":
            return {"ok": False, "code": "not_found", "error": "No such open gap."}
        if (g.disposition or "") == "needs_document":
            return {"ok": False, "code": "needs_document", "error": "Research already concluded the bank needs a document for this one; a person must add it."}
        since = datetime.now(timezone.utc) - timedelta(days=_RECENT_DAYS)
        for prior in await store.list_steward_actions(customer_id, limit=100, since=since):
            if prior["action"] == "research_gap" and prior["subject_id"] == gap_id and not prior["undone_at"]:
                return {"ok": False, "code": "already_researching", "error": "The Steward already queued research for this gap this week."}
        task = await store.create_cognition_task(
            customer_id, task_type="research",
            payload={"gap_id": g.id, "topic": g.missing, "full_key": g.full_key, "triggering_query": g.triggering_query, "steward": True},
            priority="background",
        )
        row = await session._record(
            action="research_gap", subject_kind="gap", subject_id=gap_id,
            label=f"Steward queued research - {_head(g.missing, 140)}", reason=reason,
            inverse={"cancel_task": task.id},
        )
        return {"ok": True, "action_id": row["id"], "gap_id": gap_id, "task_id": task.id}
    add("research_gap", "Queue background research for an open gap the web can answer (disposition researchable). The run spends the plan's daily capacity. Needs a reason.",
        {"type": "object", "properties": {"gap_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["gap_id", "reason"], "additionalProperties": False}, research_gap)

    async def dismiss_gap(customer_id: str, gap_id: str, reason: str) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        g = await store.get_knowledge_gap(gap_id)
        if g is None or g.customer_id != customer_id or g.status != "open":
            return {"ok": False, "code": "not_found", "error": "No such open gap."}
        if not await store.set_knowledge_gap_status(gap_id, customer_id, "closed"):
            return {"ok": False, "code": "not_found", "error": "No such open gap."}
        row = await session._record(
            action="dismiss_gap", subject_kind="gap", subject_id=gap_id,
            label=f"Steward dismissed a gap - {_head(g.missing, 140)}", reason=reason,
            inverse={"reopen_gap": gap_id},
        )
        return {"ok": True, "action_id": row["id"], "gap_id": gap_id}
    add("dismiss_gap", "Close an open gap that is not worth researching (trivia, already answered elsewhere in the bank, out of scope). Reversible. Needs a reason.",
        {"type": "object", "properties": {"gap_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["gap_id", "reason"], "additionalProperties": False}, dismiss_gap)

    async def reprobe_fact(customer_id: str, fact_id: str, reason: str) -> dict:
        if (cap := session._cap()) or (bad := session._reason_ok(reason)):
            return cap or bad
        f = await store.get_fact(fact_id)
        if f is None:
            return {"ok": False, "code": "not_found", "error": "No such fact."}
        owner = await store.get_crystal(f.crystal_id)
        if owner is None or owner.customer_id != customer_id:
            return {"ok": False, "code": "not_found", "error": "No such fact."}
        n = await store.forget_scan_verdicts(customer_id, detector="implication_probe", pair_keys=[fact_id])
        row = await session._record(
            action="reprobe_fact", subject_kind="fact", subject_id=fact_id,
            label=f"Steward asked for a re-probe - {_head(f.claim_text, 140)}", reason=reason,
            inverse=None, undoable=False,
        )
        return {"ok": True, "action_id": row["id"], "fact_id": fact_id, "was_probed": bool(n)}
    add("reprobe_fact", "Let the implication scan probe a fact again on the next idle cycle (its consequences get re-derived and re-checked against the bank). Not undoable; harmless. Needs a reason.",
        {"type": "object", "properties": {"fact_id": {"type": "string", "maxLength": 64}, "reason": {"type": "string", "maxLength": 1000}},
         "required": ["fact_id", "reason"], "additionalProperties": False}, reprobe_fact)

    return registry


async def run_steward_session(
    *,
    store: Any,
    customer: Any,
    encoder: Any,
    fact_store: Any,
    llm: Any = None,
    session_number: int = 1,
    log: Any = None,
) -> dict[str, Any]:
    """One Steward session for one tenant: open the run, let the agent
    work the bank with the Steward's tools under the tenant's background
    lane, meter the spend under origin "steward", close the run with the
    brief. Never raises for the worker: a failure closes the run as
    failed and is returned as {"ok": False, ...}."""
    from ..agent.agent import Agent
    from ..llm import get_llm_client
    from ..llm.lanes import tenant_background

    log = log or logger
    settings = get_settings()
    budget_calls = int(getattr(settings, "steward_max_tool_calls", 14))
    budget_actions = int(getattr(settings, "steward_max_actions_per_session", 6))
    wall = float(getattr(settings, "steward_wall_seconds", 240.0))
    max_tokens = int(getattr(settings, "steward_max_tokens", 2500))

    run_id = await store.open_steward_run(customer.id, payload={
        "topic": "Steward: improve this bank", "goal": "improve this bank",
        "session": session_number, "budget": {"tool_calls": budget_calls, "actions": budget_actions},
    })
    session = StewardSession(store=store, customer=customer, run_id=run_id, encoder=encoder,
                             fact_store=fact_store, max_actions=budget_actions, log=log)
    registry = build_steward_registry(session)
    usage_sink: dict[str, Any] = {}
    trajectory: list[dict[str, Any]] = []
    started = datetime.now(timezone.utc)
    result: dict[str, Any] = {}
    model = None
    try:
        async with tenant_background(customer, store):
            client = llm or get_llm_client()
            try:
                model = client._resolve_model("large", None) if hasattr(client, "_resolve_model") else None
            except Exception:  # noqa: BLE001 — a provider with no large pick falls back to the agent default
                model = None
            agent = Agent(
                customer=SimpleNamespace(id=customer.id),
                llm=client,
                tool_state={"store": store, "fact_vector_store": fact_store, "encoder": encoder},
                model=model,
                max_tokens=max_tokens,
                max_iterations=budget_calls,
                registry=registry,
            )
            prompt = (
                f"Improve this bank. This is session {session_number} today. "
                f"Budget: {budget_calls} tool calls, at most {budget_actions} actions. "
                "Start with bank_status and recent_actions."
            )
            run = await asyncio.wait_for(
                agent.run(
                    messages=[{"role": "user", "content": prompt}],
                    system=steward_charter(budget_calls, budget_actions),
                    deadline_seconds=wall,
                    usage_sink=usage_sink,
                    trajectory_sink=trajectory,
                    default_tool_timeout=_TOOL_WALL,
                    model_timeout=_MODEL_CEILING,
                ),
                timeout=wall + _GRACE_SECONDS,
            )
        brief = (run.get("final_text") or "").strip()
        tool_calls = run.get("tool_calls") or []
        result = {
            "output_type": "report",
            "deliverable": brief,
            "brief": brief,
            "actions": list(session.actions),
            "tool_calls": [{"tool": t.get("tool_name"), "input": t.get("input")} for t in tool_calls][:40],
            "calls": len(tool_calls),
            "stop_reason": run.get("stop_reason"),
            "model": run.get("model") or model,
            "session": session_number,
        }
        await store.mark_cognition_task_complete(run_id, result=result, completed_at=datetime.now(timezone.utc))
        ok = True
    except asyncio.TimeoutError:
        result = {"output_type": "report", "deliverable": "", "brief": "", "actions": list(session.actions),
                  "stop_reason": "belt", "session": session_number}
        await store.mark_cognition_task_failed(run_id, error_message=f"Steward session exceeded {wall + _GRACE_SECONDS:.0f}s",
                                               completed_at=datetime.now(timezone.utc))
        ok = False
    except Exception as e:  # noqa: BLE001 — one tenant's failure never ends the pass
        log.warning("steward.session_failed", customer_id=customer.id, run_id=run_id, error=str(e)[:300], error_type=type(e).__name__)
        result = {"output_type": "report", "deliverable": "", "brief": "", "actions": list(session.actions),
                  "stop_reason": "error", "session": session_number}
        try:
            await store.mark_cognition_task_failed(run_id, error_message=f"{type(e).__name__}: {str(e)[:300]}",
                                                   completed_at=datetime.now(timezone.utc))
        except Exception:  # noqa: BLE001
            pass
        ok = False

    # Meter the session: one ledger row, origin steward. The sink survives
    # a belt cancellation; an empty sink means no model call completed.
    prompt_tokens = int(usage_sink.get("prompt_tokens") or 0)
    completion_tokens = int(usage_sink.get("completion_tokens") or 0)
    if prompt_tokens or completion_tokens:
        try:
            await record_model_call(
                customer_id=customer.id, model=usage_sink.get("model") or model or "unknown",
                input_tokens=prompt_tokens, output_tokens=completion_tokens,
                cache_read_tokens=usage_sink.get("cache_read_tokens"),
                cache_creation_tokens=usage_sink.get("cache_creation_tokens"),
                origin=STEWARD_ORIGIN, session_id=run_id, store=store,
            )
        except Exception as e:  # noqa: BLE001 — metering must not fail the session
            log.warning("steward.metering_failed", run_id=run_id, error=str(e)[:200])

    n = len(session.actions)
    try:
        await store.record_curation_event(
            customer.id, event_type="steward_session", subject_id=run_id,
            label=(f"Steward session - {n} action{'s' if n != 1 else ''}" if ok else "Steward session failed"),
            payload={"run_id": run_id, "actions": n, "ok": ok, "session": session_number,
                     "calls": result.get("calls"), "seconds": int((datetime.now(timezone.utc) - started).total_seconds())},
        )
    except Exception:  # noqa: BLE001
        log.debug("steward.event_failed", exc_info=True)
    log.info("steward.session_completed", customer_id=customer.id, run_id=run_id, ok=ok, actions=n,
             calls=result.get("calls"), stop_reason=result.get("stop_reason"), prompt_tokens=prompt_tokens,
             completion_tokens=completion_tokens)
    return {"ok": ok, "run_id": run_id, "actions": n, "brief": result.get("brief", ""), "calls": result.get("calls") or 0,
            "stop_reason": result.get("stop_reason")}


async def undo_steward_action(store: Any, customer_id: str, action_id: str) -> dict[str, Any]:
    """Apply the inverse of one Steward action within the 24-hour
    window. Returns {"undone": bool, "code"?: str, "action"?: dict}."""
    a = await store.get_steward_action(action_id, customer_id)
    if a is None:
        return {"undone": False, "code": "not_found"}
    if a["undone_at"]:
        return {"undone": False, "code": "already_undone", "action": a}
    if not a["undoable"]:
        return {"undone": False, "code": "not_undoable" if a["inverse"] is None or a["inverse"] == {} else "undo_expired", "action": a}
    inv = a["inverse"] or {}
    kind = a["action"]
    now = datetime.now(timezone.utc)
    try:
        if kind == "decide_conflict":
            out = await store.reopen_conflict(inv.get("reopen_conflict") or a["subject_id"], customer_id=customer_id)
            if out is None:
                return {"undone": False, "code": "already_open", "action": a}
            try:
                await store.record_curation_event(customer_id, event_type="conflict_reopened", subject_id=a["subject_id"],
                                                  label="Conflict reopened - a Steward decision was undone", payload={"action_id": action_id})
            except Exception:  # noqa: BLE001
                pass
        elif kind == "approve_assumption":
            await store.set_crystal_recall_gate(a["subject_id"], customer_id, bool(inv.get("recall_gated", True)))
            if inv.get("quality_tier"):
                await store.set_crystal_quality_tier(a["subject_id"], customer_id, inv["quality_tier"])
            await store.strip_crystal_diagnostic_tags(a["subject_id"], customer_id, ("assumption_approved_by:",))
            if inv.get("gap_filled"):
                await store.set_knowledge_gap_status(inv["gap_filled"], customer_id, "open")
        elif kind == "reject_assumption":
            if inv.get("quality_tier"):
                await store.set_crystal_quality_tier(a["subject_id"], customer_id, inv["quality_tier"])
            await store.set_crystal_recall_gate(a["subject_id"], customer_id, bool(inv.get("recall_gated", True)))
        elif kind == "research_gap":
            task_id = inv.get("cancel_task")
            task = await store.get_cognition_task(task_id) if task_id else None
            if task is not None and task.status in ("pending", "running"):
                await store.mark_cognition_task_cancelled(task_id, completed_at=now, reason="A person undid the Steward's research request")
        elif kind == "dismiss_gap":
            await store.set_knowledge_gap_status(inv.get("reopen_gap") or a["subject_id"], customer_id, "open")
        else:
            return {"undone": False, "code": "not_undoable", "action": a}
    except Exception as e:  # noqa: BLE001 — report, never half-mark
        logger.warning("steward.undo_failed", action_id=action_id, error=str(e)[:300])
        return {"undone": False, "code": "undo_failed", "error": "The undo did not go through; nothing was changed.", "action": a}
    await store.mark_steward_action_undone(action_id, customer_id)
    try:
        await store.record_curation_event(customer_id, event_type="steward_undone", subject_id=a["subject_id"],
                                          label=f"Undone - {a['label']}"[:256], payload={"action_id": action_id, "action": kind})
    except Exception:  # noqa: BLE001
        pass
    refreshed = await store.get_steward_action(action_id, customer_id)
    return {"undone": True, "action": refreshed or a}
