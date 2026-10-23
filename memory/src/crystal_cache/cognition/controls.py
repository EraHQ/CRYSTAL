"""The controls (2026-10-23): one implementation of every curation act,
two pairs of hands.

Anthony: "I want ALL THE CONTROLS. Give me the mcp route." The in-app
Steward (cognition/steward.py) and a person or their AI client acting
through the MCP controls (agent/mcp_server.py) call the same functions
here. Every act lands in steward_actions with who did it (actor), why
(reason), and the inverse that undoes it within 24 hours (undo_action);
every act is on the curation feed. Nothing here bypasses the walls the
console pays: the fact cap for writes that add facts, the tenant door for
research. R9: no SQL; everything goes through the store.

Acts and their inverses:
  decide_conflict      apply a resolution              -> reopen
  reopen_conflict      undo a resolution               -> re-apply it
  approve_assumption   gate open + parent's standing   -> re-gate, restore tier
  reject_assumption    blacklist + gate                -> restore
  verify_assumption    queue verification research     -> cancel the task
  research_gap         queue research                  -> cancel the task
  dismiss_gap          close                           -> reopen
  reopen_gap           open                            -> close
  answer_gap           write the answer as a fact      -> forget that crystal, reopen
  edit_fact            supersede a fact's text / key   -> supersede back
  set_tier             move a crystal's tier           -> previous tier
  retire_fact          delete one fact                 (no undo; the ledger keeps the text)
  reprobe_fact         forget the probe memo           (no undo; harmless)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Optional

import structlog

from ..infrastructure.metadata_store_assumption_ext import parse_assumption_tags

logger = structlog.get_logger(__name__)

CONFLICT_RESOLUTIONS = ("superseded", "qualified", "dismissed", "blacklisted")
TIERS = ("whitelist", "neutral", "quarantine", "blacklist")
REASON_MIN = 12
RECENT_DAYS = 7


def head(s: Optional[str], n: int = 240) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def refusal(code: str, error: str) -> dict[str, Any]:
    return {"ok": False, "code": code, "error": error}


def reason_check(reason: str) -> Optional[dict[str, Any]]:
    if len((reason or "").strip()) < REASON_MIN:
        return refusal("reason_required", "Give a reason a person will read later (one or two sentences).")
    return None


def who(actor: str) -> str:
    """The actor's name in a label: 'Steward' or 'Chat'."""
    return "Steward" if actor == "steward" else "Chat"


async def record(
    store: Any, customer_id: str, *, actor: str, run_id: Optional[str], action: str, subject_kind: str,
    subject_id: str, label: str, reason: str, inverse: Optional[dict[str, Any]], undoable: bool = True,
) -> dict[str, Any]:
    """The ledger row and the witness event for one act."""
    row = await store.record_steward_action(
        customer_id, run_id=run_id, action=action, subject_kind=subject_kind, subject_id=subject_id,
        label=label, reason=reason, inverse=inverse, undoable=undoable, actor=actor,
    )
    try:
        await store.record_curation_event(
            customer_id, event_type="steward_action" if actor == "steward" else "control_action",
            subject_id=subject_id, label=label[:256],
            payload={"action": action, "action_id": row["id"], "run_id": run_id, "actor": actor, "reason": reason[:500]},
        )
    except Exception:  # noqa: BLE001 — the feed is a witness, never a gate
        logger.debug("controls.event_failed", exc_info=True)
    return row


# ---------------------------------------------------------------- conflicts

async def decide_conflict(store: Any, customer_id: str, *, conflict_id: str, resolution: str, reason: str,
                          loser: Optional[str] = None, actor: str, run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    if resolution not in CONFLICT_RESOLUTIONS:
        return refusal("bad_resolution", f"resolution must be one of {', '.join(CONFLICT_RESOLUTIONS)}")
    if resolution in ("superseded", "blacklisted") and loser not in ("a", "b"):
        return refusal("loser_required", f"{resolution} needs loser='a' or 'b' (the claim that loses)")
    c = await store.get_knowledge_conflict(conflict_id)
    if c is None or c.customer_id != customer_id:
        return refusal("not_found", "No such open conflict.")
    if c.status != "open":
        return refusal("not_open", "That conflict is already settled.")
    try:
        updated = await store.apply_conflict_resolution(
            conflict_id, resolution=resolution, resolved_at=datetime.now(timezone.utc),
            loser=loser if resolution in ("superseded", "blacklisted") else None, customer_id=customer_id,
        )
    except ValueError as e:
        # detail-ok: apply_conflict_resolution's ValueError text is ours.
        return refusal("refused", str(e))
    if updated is None:
        return refusal("not_found", "No such open conflict.")
    try:
        await store.bank_changed(customer_id)
    except Exception:  # noqa: BLE001 — the index reload is best-effort here
        pass
    subject = c.subject or head(c.claim_a, 60)
    word = {"superseded": "retired the outdated claim", "qualified": "kept both claims",
            "dismissed": "dismissed the pairing", "blacklisted": "blacklisted the losing claim"}[resolution]
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="decide_conflict", subject_kind="conflict",
                       subject_id=conflict_id, label=f"{who(actor)} settled a conflict - {subject}: {word}", reason=reason,
                       inverse={"reopen_conflict": conflict_id, "resolution": resolution, "loser": loser})
    return {"ok": True, "action_id": row["id"], "conflict_id": conflict_id, "resolution": resolution,
            "retired_claim": (c.claim_a if loser == "a" else c.claim_b) if resolution in ("superseded", "blacklisted") else None}


async def reopen_conflict(store: Any, customer_id: str, *, conflict_id: str, reason: str, actor: str,
                          run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    out = await store.reopen_conflict(conflict_id, customer_id=customer_id)
    if out is None:
        return refusal("not_found", "No such settled conflict.")
    conflict, previous = out
    try:
        await store.bank_changed(customer_id)
    except Exception:  # noqa: BLE001
        pass
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="reopen_conflict", subject_kind="conflict",
                       subject_id=conflict_id, label=f"{who(actor)} reopened a conflict - {conflict.subject or head(conflict.claim_a, 60)}",
                       reason=reason, inverse={"reapply": previous})
    return {"ok": True, "action_id": row["id"], "conflict_id": conflict_id, "was": previous}


# -------------------------------------------------------------- assumptions

async def approve_assumption(store: Any, customer_id: str, *, crystal_id: str, reason: str, actor: str,
                             run_id: Optional[str] = None, force: bool = False) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    c = await store.get_crystal(crystal_id)
    if c is None or c.customer_id != customer_id or c.crystal_type != "assumption":
        return refusal("not_found", "No such assumption.")
    if c.quality_tier == "blacklist":
        return refusal("invalidated", "That one was invalidated; it cannot be approved.")
    tags = parse_assumption_tags(list(c.diagnostic_tags or []))
    if tags.get("approved_by") or (c.build_method != "consequence" and not c.recall_gated):
        return refusal("already_approved", "Already approved.")
    edges = await store.count_consequence_edges(crystal_id)
    open_conf = await store.count_open_conflicts_for_crystal(customer_id, crystal_id)
    if (edges.get("contradicted_by", 0) or open_conf) and not force:
        return refusal("contradicted", "A stated fact contradicts it (or a conflict is open on it); a person decides this one.")
    before = {"quality_tier": c.quality_tier, "recall_gated": bool(c.recall_gated)}
    standing = await store.approve_assumption_standing(customer_id, crystal_id, approved_by=actor)
    gap_filled = None
    try:
        gap_filled = await store.close_gap_for_approved_assumption(crystal_id, customer_id)
    except Exception:  # noqa: BLE001 — the gap close is a courtesy
        gap_filled = None
    kind = "a consequence" if standing["kind"] == "consequence" else "an assumption"
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="approve_assumption", subject_kind="assumption",
                       subject_id=crystal_id, label=f"{who(actor)} approved {kind} - {head(c.summary_text, 120)}", reason=reason,
                       inverse={**before, "gap_filled": gap_filled})
    try:
        await store.record_curation_event(customer_id, event_type="assumption_approved", subject_id=crystal_id,
                                          label=f"Assumption approved by {who(actor).lower()} - {head(c.summary_text, 180)}"[:256],
                                          payload={"by": actor, "action_id": row["id"]})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "quality_tier": standing["quality_tier"], "kind": standing["kind"]}


async def reject_assumption(store: Any, customer_id: str, *, crystal_id: str, reason: str, actor: str,
                            run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    c = await store.get_crystal(crystal_id)
    if c is None or c.customer_id != customer_id or c.crystal_type != "assumption":
        return refusal("not_found", "No such assumption.")
    if c.quality_tier == "blacklist":
        return refusal("already_invalidated", "Already invalidated.")
    before = {"quality_tier": c.quality_tier, "recall_gated": bool(c.recall_gated)}
    await store.set_crystal_recall_gate(crystal_id, customer_id, True)
    await store.set_crystal_quality_tier(crystal_id, customer_id, "blacklist")
    kind = "a consequence" if c.build_method == "consequence" else "an assumption"
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="reject_assumption", subject_kind="assumption",
                       subject_id=crystal_id, label=f"{who(actor)} rejected {kind} - {head(c.summary_text, 120)}", reason=reason, inverse=before)
    try:
        await store.record_curation_event(customer_id, event_type="assumption_invalidated", subject_id=crystal_id,
                                          label=f"Assumption rejected by {who(actor).lower()} - {head(c.summary_text, 180)}"[:256],
                                          payload={"by": actor, "action_id": row["id"]})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id}


async def verify_assumption(store: Any, customer_id: str, *, crystal_id: str, reason: str, actor: str,
                            run_id: Optional[str] = None) -> dict[str, Any]:
    from ..scan.verification import VERIFICATION_TASK_TYPE, verification_goal

    if (bad := reason_check(reason)):
        return bad
    c = await store.get_crystal(crystal_id)
    if c is None or c.customer_id != customer_id or c.crystal_type != "assumption":
        return refusal("not_found", "No such assumption.")
    if c.quality_tier == "blacklist":
        return refusal("invalidated", "That one was invalidated; there is nothing to verify.")
    if any(str(t).startswith("verification_task:") for t in (c.diagnostic_tags or [])):
        return refusal("already_verifying", "A verification run is already queued for it.")
    statement = (c.summary_text or "").strip()
    task = await store.create_cognition_task(
        customer_id, task_type=VERIFICATION_TASK_TYPE,
        payload={"topic": verification_goal(statement), "assumption_crystal_id": crystal_id, "statement": statement, "actor": actor},
        priority="background",
    )
    await store.tag_assumption_verification(customer_id, crystal_id, task.id)
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="verify_assumption", subject_kind="assumption",
                       subject_id=crystal_id, label=f"{who(actor)} queued verification - {head(statement, 120)}", reason=reason,
                       inverse={"cancel_task": task.id})
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "task_id": task.id}


# --------------------------------------------------------------------- gaps

async def research_gap(store: Any, customer_id: str, *, gap_id: str, reason: str, actor: str,
                       run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    g = await store.get_knowledge_gap(gap_id)
    if g is None or g.customer_id != customer_id or g.status != "open":
        return refusal("not_found", "No such open gap.")
    if (g.disposition or "") == "needs_document":
        return refusal("needs_document", "Research already concluded the bank needs a document for this one; a person must add it.")
    since = datetime.now(timezone.utc) - timedelta(days=RECENT_DAYS)
    for prior in await store.list_steward_actions(customer_id, limit=100, since=since):
        if prior["action"] == "research_gap" and prior["subject_id"] == gap_id and not prior["undone_at"]:
            return refusal("already_researching", "Research for this gap was queued this week.")
    task = await store.create_cognition_task(
        customer_id, task_type="research",
        payload={"gap_id": g.id, "topic": g.missing, "full_key": g.full_key, "triggering_query": g.triggering_query, "actor": actor},
        priority="background",
    )
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="research_gap", subject_kind="gap",
                       subject_id=gap_id, label=f"{who(actor)} queued research - {head(g.missing, 140)}", reason=reason,
                       inverse={"cancel_task": task.id})
    return {"ok": True, "action_id": row["id"], "gap_id": gap_id, "task_id": task.id}


async def dismiss_gap(store: Any, customer_id: str, *, gap_id: str, reason: str, actor: str,
                      run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    g = await store.get_knowledge_gap(gap_id)
    if g is None or g.customer_id != customer_id or g.status != "open":
        return refusal("not_found", "No such open gap.")
    if not await store.set_knowledge_gap_status(gap_id, customer_id, "closed"):
        return refusal("not_found", "No such open gap.")
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="dismiss_gap", subject_kind="gap",
                       subject_id=gap_id, label=f"{who(actor)} dismissed a gap - {head(g.missing, 140)}", reason=reason,
                       inverse={"reopen_gap": gap_id})
    return {"ok": True, "action_id": row["id"], "gap_id": gap_id}


async def reopen_gap(store: Any, customer_id: str, *, gap_id: str, reason: str, actor: str,
                     run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    g = await store.get_knowledge_gap(gap_id)
    if g is None or g.customer_id != customer_id or g.status != "closed":
        return refusal("not_found", "No such dismissed gap.")
    await store.set_knowledge_gap_status(gap_id, customer_id, "open")
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="reopen_gap", subject_kind="gap",
                       subject_id=gap_id, label=f"{who(actor)} reopened a gap - {head(g.missing, 140)}", reason=reason,
                       inverse={"close_gap": gap_id})
    return {"ok": True, "action_id": row["id"], "gap_id": gap_id}


async def answer_gap(store: Any, state: dict, customer_id: str, *, gap_id: str, text: str, key: Optional[str],
                     reason: str, actor: str, run_id: Optional[str] = None) -> dict[str, Any]:
    """Write the answer as a stated fact under the gap's key and fill the gap
    with it. The caller pays the fact-cap door."""
    if (bad := reason_check(reason)):
        return bad
    text = (text or "").strip()
    if not text:
        return refusal("text_required", "Give the answer text.")
    g = await store.get_knowledge_gap(gap_id)
    if g is None or g.customer_id != customer_id or g.status != "open":
        return refusal("not_found", "No such open gap.")
    prompt = (key or "").strip() or (g.full_key or "").strip() or "|".join(x for x in (g.domain, g.subject) if x) or g.missing[:200]
    crystal, fact = await store.add_pair_for_customer(
        customer_id=customer_id, prompt_text=prompt, answer_text=text, pair_type="question_answer",
        encoder=state["encoder"], vector_store=state["vector_store"], vector_index=state.get("vector_index"),
        crystal_type="customer:legacy", source_kind="operator_stated",
    )
    await store.mark_knowledge_gap_filled(gap_id, filled_by_crystal_id=crystal.id, resolved_at=datetime.now(timezone.utc))
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="answer_gap", subject_kind="gap",
                       subject_id=gap_id, label=f"{who(actor)} answered a gap - {head(g.missing, 120)}", reason=reason,
                       inverse={"forget_crystal": crystal.id, "reopen_gap": gap_id})
    try:
        await store.record_curation_event(customer_id, event_type="gap_filled", subject_id=gap_id,
                                          label=f"Gap filled by {who(actor).lower()} - {head(g.missing, 160)}"[:256],
                                          payload={"crystal_id": crystal.id, "action_id": row["id"]})
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "action_id": row["id"], "gap_id": gap_id, "crystal_id": crystal.id, "fact_id": fact.id, "key": prompt}


# -------------------------------------------------------------------- facts

async def edit_fact(store: Any, state: dict, customer_id: str, *, crystal_id: str, fact_id: str, text: str,
                    key: Optional[str], reason: str, actor: str, run_id: Optional[str] = None) -> dict[str, Any]:
    """Supersede a fact's text (and key) in place: the old text survives in
    the fact ledger and in the inverse, so the edit can be put back."""
    if (bad := reason_check(reason)):
        return bad
    text = (text or "").strip()
    if not text:
        return refusal("text_required", "Give the new text.")
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer_id:
        return refusal("not_found", "No such crystal.")
    fact = await store.get_fact(fact_id)
    if fact is None or fact.crystal_id != crystal_id:
        return refusal("not_found", "No such fact in that crystal.")
    new_prompt = (key or "").strip() or fact.prompt_text
    successor = await _supersede(store, state, customer_id, crystal_id, fact, new_prompt, text, actor)
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="edit_fact", subject_kind="fact",
                       subject_id=successor.id, label=f"{who(actor)} edited a fact - {head(text, 120)}", reason=reason,
                       inverse={"crystal_id": crystal_id, "fact_id": successor.id, "text": fact.claim_text, "key": fact.prompt_text})
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "superseded": fact_id, "fact_id": successor.id}


async def _supersede(store: Any, state: dict, customer_id: str, crystal_id: str, fact: Any, new_prompt: str, new_text: str, actor: str):
    new_fact = await store.add_pair_to_crystal(
        crystal_id, new_prompt, new_text, pair_type=fact.pair_type, encoder=state["encoder"],
        source_kind=fact.source_kind, citation=fact.citation, chunk_index=fact.chunk_index,
    )
    await store.append_fact_ledger(
        customer_id, crystal_id, fact.id, op="supersede", actor=actor,
        before_prompt=fact.prompt_text, before_text=fact.claim_text, after_text=new_text, successor_fact_id=new_fact.id,
    )
    await store.delete_fact(fact.id, customer_id, encoder=state["encoder"], vector_store=state.get("vector_store"),
                            fact_vector_store=state.get("fact_vector_store"))
    return new_fact


async def retire_fact(store: Any, state: dict, customer_id: str, *, crystal_id: str, fact_id: str, reason: str,
                      actor: str, run_id: Optional[str] = None) -> dict[str, Any]:
    """Delete one fact. Permanent: the ledger keeps its text, nothing restores it."""
    if (bad := reason_check(reason)):
        return bad
    crystal = await store.get_crystal(crystal_id)
    if crystal is None or crystal.customer_id != customer_id:
        return refusal("not_found", "No such crystal.")
    fact = await store.get_fact(fact_id)
    if fact is None or fact.crystal_id != crystal_id:
        return refusal("not_found", "No such fact in that crystal.")
    await store.append_fact_ledger(customer_id, crystal_id, fact_id, op="retire", actor=actor,
                                   before_prompt=fact.prompt_text, before_text=fact.claim_text)
    await store.delete_fact(fact_id, customer_id, encoder=state["encoder"], vector_store=state.get("vector_store"),
                            fact_vector_store=state.get("fact_vector_store"))
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="retire_fact", subject_kind="fact",
                       subject_id=fact_id, label=f"{who(actor)} retired a fact - {head(fact.claim_text, 120)}", reason=reason,
                       inverse=None, undoable=False)
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "retired": fact_id, "permanent": True}


async def set_tier(store: Any, customer_id: str, *, crystal_id: str, tier: str, reason: str, actor: str,
                   run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    if tier not in TIERS:
        return refusal("bad_tier", f"tier must be one of {', '.join(TIERS)}")
    c = await store.get_crystal(crystal_id)
    if c is None or c.customer_id != customer_id:
        return refusal("not_found", "No such crystal.")
    if c.quality_tier == tier:
        return refusal("unchanged", f"Already {tier}.")
    before = c.quality_tier
    await store.set_crystal_quality_tier(crystal_id, customer_id, tier)
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="set_tier", subject_kind="crystal",
                       subject_id=crystal_id, label=f"{who(actor)} moved a crystal to {tier} - {head(c.summary_text or c.source_path or crystal_id, 100)}",
                       reason=reason, inverse={"quality_tier": before})
    return {"ok": True, "action_id": row["id"], "crystal_id": crystal_id, "quality_tier": tier, "was": before}


async def reprobe_fact(store: Any, customer_id: str, *, fact_id: str, reason: str, actor: str,
                       run_id: Optional[str] = None) -> dict[str, Any]:
    if (bad := reason_check(reason)):
        return bad
    f = await store.get_fact(fact_id)
    if f is None:
        return refusal("not_found", "No such fact.")
    owner = await store.get_crystal(f.crystal_id)
    if owner is None or owner.customer_id != customer_id:
        return refusal("not_found", "No such fact.")
    n = await store.forget_scan_verdicts(customer_id, detector="implication_probe", pair_keys=[fact_id])
    row = await record(store, customer_id, actor=actor, run_id=run_id, action="reprobe_fact", subject_kind="fact",
                       subject_id=fact_id, label=f"{who(actor)} asked for a re-probe - {head(f.claim_text, 140)}", reason=reason,
                       inverse=None, undoable=False)
    return {"ok": True, "action_id": row["id"], "fact_id": fact_id, "was_probed": bool(n)}


# --------------------------------------------------------------------- undo

async def undo_action(store: Any, customer_id: str, action_id: str, *, state: Optional[dict] = None,
                      actor: str = "chat") -> dict[str, Any]:
    """Apply the inverse of one act within the 24-hour window."""
    a = await store.get_steward_action(action_id, customer_id)
    if a is None:
        return {"undone": False, "code": "not_found"}
    if a["undone_at"]:
        return {"undone": False, "code": "already_undone", "action": a}
    if not a["undoable"]:
        return {"undone": False, "code": "not_undoable" if not a["inverse"] else "undo_expired", "action": a}
    inv = a["inverse"] or {}
    kind = a["action"]
    now = datetime.now(timezone.utc)
    try:
        if kind == "decide_conflict":
            out = await store.reopen_conflict(inv.get("reopen_conflict") or a["subject_id"], customer_id=customer_id)
            if out is None:
                return {"undone": False, "code": "already_open", "action": a}
            await store.bank_changed(customer_id)
            try:
                await store.record_curation_event(customer_id, event_type="conflict_reopened", subject_id=a["subject_id"],
                                                  label="Conflict reopened - a decision was undone", payload={"action_id": action_id})
            except Exception:  # noqa: BLE001
                pass
        elif kind == "reopen_conflict":
            previous = inv.get("reapply")
            if previous:
                c = await store.get_knowledge_conflict(a["subject_id"])
                loser = None
                if previous in ("superseded", "blacklisted") and c is not None:
                    # The side that lost is the one whose fact is inactive now; the
                    # reopen restored it, so re-apply against the recommendation's
                    # loser when there is one, else refuse rather than guess.
                    loser = ((c.recommendation or {}).get("loser") if c.recommendation else None)
                    if loser not in ("a", "b"):
                        return {"undone": False, "code": "cannot_reapply", "action": a}
                await store.apply_conflict_resolution(a["subject_id"], resolution=previous, resolved_at=now, loser=loser, customer_id=customer_id)
                await store.bank_changed(customer_id)
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
        elif kind in ("research_gap", "verify_assumption"):
            task_id = inv.get("cancel_task")
            task = await store.get_cognition_task(task_id) if task_id else None
            if task is not None and task.status in ("pending", "running"):
                await store.mark_cognition_task_cancelled(task_id, completed_at=now, reason="A person undid the request")
            if kind == "verify_assumption":
                await store.strip_crystal_diagnostic_tags(a["subject_id"], customer_id, ("verification_task:",))
        elif kind == "dismiss_gap":
            await store.set_knowledge_gap_status(inv.get("reopen_gap") or a["subject_id"], customer_id, "open")
        elif kind == "reopen_gap":
            await store.set_knowledge_gap_status(inv.get("close_gap") or a["subject_id"], customer_id, "closed")
        elif kind == "answer_gap":
            if state is None:
                return {"undone": False, "code": "needs_state", "action": a}
            if inv.get("forget_crystal"):
                await store.delete_crystal(inv["forget_crystal"], customer_id, vector_store=state.get("vector_store"),
                                           fact_vector_store=state.get("fact_vector_store"))
            await store.set_knowledge_gap_status(inv.get("reopen_gap") or a["subject_id"], customer_id, "open")
        elif kind == "edit_fact":
            if state is None:
                return {"undone": False, "code": "needs_state", "action": a}
            fact = await store.get_fact(inv.get("fact_id") or a["subject_id"])
            if fact is None:
                return {"undone": False, "code": "fact_gone", "action": a}
            await _supersede(store, state, customer_id, inv["crystal_id"], fact, inv.get("key") or fact.prompt_text, inv["text"], actor)
        elif kind == "set_tier":
            if inv.get("quality_tier"):
                await store.set_crystal_quality_tier(a["subject_id"], customer_id, inv["quality_tier"])
        else:
            return {"undone": False, "code": "not_undoable", "action": a}
    except Exception as e:  # noqa: BLE001 — report, never half-mark
        logger.warning("controls.undo_failed", action_id=action_id, error=str(e)[:300])
        return {"undone": False, "code": "undo_failed", "error": "The undo did not go through; nothing was changed.", "action": a}
    await store.mark_steward_action_undone(action_id, customer_id)
    try:
        await store.record_curation_event(customer_id, event_type="steward_undone", subject_id=a["subject_id"],
                                          label=f"Undone - {a['label']}"[:256], payload={"action_id": action_id, "action": kind, "by": actor})
    except Exception:  # noqa: BLE001
        pass
    refreshed = await store.get_steward_action(action_id, customer_id)
    return {"undone": True, "action": refreshed or a}


# -------------------------------------------------------------------- reads

async def list_assumptions_view(store: Any, customer_id: str, *, state: str = "waiting", kind: str = "any",
                                limit: int = 30) -> list[dict[str, Any]]:
    """Assumptions and consequences as a reviewer reads them: the statement,
    what it was built from or derived from, how the facts judged it, and
    whether anything contradicts it. state: waiting | approved |
    invalidated | any; kind: any | assumption | consequence."""
    from ..infrastructure.metadata_store_assumption_ext import assumption_state

    rows = await store.list_assumption_crystals(customer_id, limit=300)
    want_state = (state or "waiting").lower()
    want_kind = (kind or "any").lower()
    picked = []
    for r in rows:
        if want_state != "any" and assumption_state(r) != want_state:
            continue
        if want_kind != "any" and r.get("kind") != want_kind:
            continue
        picked.append(r)
        if len(picked) >= max(1, min(int(limit or 30), 100)):
            break
    fact_ids: list[str] = []
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
            "crystal_id": r["id"], "kind": r.get("kind"), "state": assumption_state(r),
            "statement": head(r.get("statement"), 400), "recallable": not r.get("recall_gated"),
            "quality_tier": r.get("quality_tier"), "confidence": t.get("confidence"), "why": head(t.get("rationale"), 400),
            "approved_by": t.get("approved_by"), "created_at": r.get("created_at"),
        }
        if r.get("kind") == "consequence":
            pf = t.get("parent_fact_id")
            item["derived_from"] = head(claims.get(pf, {}).get("claim"), 300) if pf else None
            item["checks"] = [{"fact_id": c["fact_id"], "verdict": c["verdict"], "claim": head(claims.get(c["fact_id"], {}).get("claim"), 200)}
                              for c in t.get("checks", [])]
            item["contradicted"] = t.get("contradicted_edges", 0) > 0
        else:
            parents = []
            for chain in await store.list_chains_from_source(r["id"]):
                p = await store.get_crystal(chain.target_crystal_id)
                if p is not None and p.customer_id == customer_id:
                    parents.append({"crystal_id": p.id, "label": head(p.summary_text or p.source_path or p.id, 200)})
            item["built_from"] = parents
        out.append(item)
    return out
