"""Cognition workflow engine.

Orchestrates the full lifecycle:
  create environment -> orchestrator plans -> workers execute ->
  validator approves/rejects -> commit or retry -> destroy environment

v2 port (Phase 6 Wave C): one targeted refactor against the v1
verbatim — `_commit_and_finalize` now calls
`store.create_document_upload(...)` (Phase 5 AuditTablesMixin)
instead of inline `DocumentUploadRow` insert. Per R9 (no SQL outside
the store) and consistent with Phase 6.5 P0.1 (v1 status names; the
upload lands in `status='pending'` by default and the crystallization
worker picks it up).
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import structlog

from .models import (
    CognitionEnvironment, CognitionResult, OutputType,
    StepStatus, ValidationResult, WorkflowStatus,
)
from .roles import run_orchestrator, run_validator, run_worker

if TYPE_CHECKING:
    from ..infrastructure.fact_vector_store import FactVectorStore
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# Active environments for status tracking via API.
# Process-local dict; not persisted across restarts. The cognition_tasks
# table is the persistent record; this dict is just for live inspector
# polling against in-flight workflows.
_active_environments: dict[str, CognitionEnvironment] = {}


def get_active_environments(customer_id: str = "") -> list[CognitionEnvironment]:
    """Return active environments, optionally filtered by customer."""
    envs = list(_active_environments.values())
    if customer_id:
        envs = [e for e in envs if e.customer_id == customer_id]
    return envs


def get_environment(env_id: str) -> Optional[CognitionEnvironment]:
    """Get a specific environment by ID."""
    return _active_environments.get(env_id)


def env_summary(env) -> dict:
    """Compact summary for the list view."""
    step_statuses = {}
    for sid, step in env.step_outputs.items():
        step_statuses[str(sid)] = {
            "action": step.action,
            "status": step.status.value,
            "duration_ms": step.duration_ms,
        }

    return {
        "id": env.id,
        "customer_id": env.customer_id,
        "status": env.status.value,
        "trigger_type": env.trigger_type,
        "goal_title": env.goal.title if env.goal else "",
        "output_type": env.output_type.value,
        "attempts": env.attempts,
        "max_attempts": env.max_attempts,
        "step_count": len(env.plan.steps) if env.plan else 0,
        "steps_complete": sum(
            1 for s in env.step_outputs.values()
            if s.status.value == "complete"
        ),
        "steps": step_statuses,
        "validation": {
            "approved": env.validation.approved,
            "score": env.validation.score,
        } if env.validation else None,
        "tokens_used": env.tokens_used,
        "cost_usd": round(env.total_cost_usd, 6),
        "created_at": env.created_at.isoformat(),
    }


async def _persist_snapshot(store, env, *, terminal: bool = False) -> None:
    """S9 (2026-07-08): write the environment's state to cognition_runs.
    The in-memory registry is process-local and the UI polls a different
    process — this table is the surface. Stores the EXACT wire shapes
    (summary + detail) so the tracker needs no changes. Never raises."""
    try:
        await store.upsert_cognition_run(
            env.id,
            env.customer_id,
            status=env.status.value,
            trigger_type=env.trigger_type,
            goal_title=(env.goal.title if env.goal else ""),
            summary=env_summary(env),
            detail=env.to_dict(),
            terminal=terminal,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "cognition.snapshot_failed", env_id=env.id, error=str(e)
        )


# C2 answerability gate. Action-value strings (the wire format StepAction
# serializes to) for retrieval vs. composition steps.
_RETRIEVAL_ACTIONS = frozenset({"crystal_search", "crystal_key_scan", "web_search", "web_fetch", "research", "source_lookup"})
_COMPOSITION_ACTIONS = frozenset({"analyze", "synthesize", "format"})


def _retrieval_grounding_count(output: Any) -> int:
    """How many grounding items a retrieval step produced.

    Counts crystal_search / crystal_key_scan findings AND source_lookup
    results (search matches, list entries, or a non-empty file read), so a
    source read is recognized as grounding by the answerability gate
    rather than treated as zero (which would wrongly park the workflow).
    """
    if not isinstance(output, dict):
        return 0
    n = len(output.get("findings", []) or [])
    n += len(output.get("matches", []) or [])
    n += len(output.get("entries", []) or [])
    if output.get("content"):
        n += 1
    return n


def _should_park_unanswerable(plan, executed: set, env: CognitionEnvironment) -> bool:
    """C2 answerability probe (evidence-based continue-gate).

    Returns True only when every retrieval step has executed, at least one
    composition step has NOT yet executed, and the retrieval steps produced
    zero findings. In that state nothing retrieved grounds an answer (bank
    empty on the topic; any web/source steps also came back empty or
    errored), so the composition + validation cycle could only fabricate
    (which C3 would then flag). Returns False whenever
    retrieval found anything, when the plan has no retrieval steps, or when
    no composition step remains — so an answerable task is never parked. The
    retrieved evidence decides, not the wording of the task.
    """
    # Orchestrator-sourced bank findings (2026-07-11) are grounding: the
    # plan carries curated bank material, so composition is not
    # fabricating even if its own retrieval steps come back empty.
    if getattr(plan, "bank_findings", None):
        return False
    retrieval_ids = [s.id for s in plan.steps if s.action.value in _RETRIEVAL_ACTIONS]
    if not retrieval_ids:
        return False
    if not all(rid in executed for rid in retrieval_ids):
        return False
    composition_remaining = any(
        s.id not in executed
        for s in plan.steps
        if s.action.value in _COMPOSITION_ACTIONS
    )
    if not composition_remaining:
        return False
    total_findings = 0
    for rid in retrieval_ids:
        out = env.step_outputs.get(rid)
        if out is not None:
            total_findings += _retrieval_grounding_count(out.output)
    return total_findings == 0


# Actions whose completed outputs are research already paid for — what the
# revision-aware retry carries across attempts (Q1A). Mirrors
# _RETRIEVAL_ACTIONS but includes source_lookup: any evidence-gathering
# step's findings are carryover.
_CARRYOVER_ACTIONS = frozenset(
    {"crystal_search", "crystal_key_scan", "web_search", "web_fetch",
     "research", "source_lookup"}
)


def _harvest_findings(env: CognitionEnvironment) -> list[dict[str, Any]]:
    """Render the completed retrieval steps' outputs into carryover text.

    Revision-aware retry (2026-07-10). Runs at rejection time, BEFORE the
    retry hygiene clears step_outputs: each completed evidence-gathering
    step becomes one {attempt, step_id, action, description, text} entry
    the next attempt's composition steps read as source material — the
    research is already paid for; a retry should never re-buy it. Text is
    rendered through the same content/findings extraction the composition
    prompt uses (roles._render_step_output_text), bounded per entry.
    Prior carryover (attempt N-2's findings) is retained and deduped by
    (attempt, step_id) so a gap_fill route accumulates rather than
    replaces.
    """
    from .roles import _render_step_output_text

    plan_desc = {
        s.id: s.description for s in (env.plan.steps if env.plan else [])
    }
    harvested: list[dict[str, Any]] = list(env.carried_findings)
    seen = {(f.get("attempt"), f.get("step_id")) for f in harvested}
    for step_id, out in sorted(env.step_outputs.items()):
        if out.action not in _CARRYOVER_ACTIONS:
            continue
        if out.status != StepStatus.COMPLETE:
            continue
        key = (env.attempts, step_id)
        if key in seen:
            continue
        text = _render_step_output_text(out)[:6000]
        if not text.strip():
            continue
        harvested.append({
            "attempt": env.attempts,
            "step_id": step_id,
            "action": out.action,
            "description": plan_desc.get(step_id, ""),
            "text": text,
        })
    return harvested


def _fail_fast_step(
    plan: Any,
    executed: set,
    remaining: list,
    env: CognitionEnvironment,
) -> tuple:
    """Q1A (ratified 2026-07-15): detect a FAILED step whose
    un-executed transitive dependents reach a deliverable-producing
    sink.

    When one exists, the attempt cannot produce a complete
    deliverable: its composition steps would honestly report the holes
    (correct, unchanged behavior when they DO run) and the validator
    would rightly reject — a verdict already known before paying for
    it. The engine fails the attempt fast instead: composition +
    validator are skipped and a synthetic rejection rides the SAME
    rails as a real one (rejection_log, attempt archive, findings
    harvest, revision routes, critique feed).

    Deliberately NOT fail-fast:
      - a failed SINK (e.g. format itself): no pending dependents —
        the existing deliverable-salvage + validator path owns it;
      - a failed step nothing depends on: its findings were optional.

    Returns (failed_step, affected_remaining_ids) or (None, set()).
    """
    if plan is None:
        return None, set()
    steps = list(plan.steps or [])
    by_id = {s.id: s for s in steps}
    dependents: dict = {s.id: set() for s in steps}
    for s in steps:
        for dep in (s.depends_on or []):
            if dep in dependents:
                dependents[dep].add(s.id)
    sinks = {sid for sid, kids in dependents.items() if not kids}
    remaining_ids = {s.id for s in remaining}
    for sid in sorted(executed):
        out = env.step_outputs.get(sid)
        if out is None or out.status != StepStatus.FAILED:
            continue
        closure: set = set()
        frontier = [sid]
        while frontier:
            for nxt in dependents.get(frontier.pop(), ()):
                if nxt not in closure:
                    closure.add(nxt)
                    frontier.append(nxt)
        if (closure & remaining_ids) and (closure & sinks):
            return by_id.get(sid), closure & remaining_ids
    return None, set()


async def run_cognition_workflow(
    goal: str,
    customer_id: str,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any,
    *,
    conversation_context: str = "",
    source_crystal_id: str = "",
    output_type: str = "crystal",
    trigger_type: str = "research",
    trigger_id: str = "",
    max_attempts: int = 3,
) -> CognitionResult:
    """Execute the full cognition workflow.

    Contract verified against Wave A's worker call site (G6A-2):
    `workers/cognition.py::_process_pending_tasks` and `_fill_open_gaps`
    both call this function with exactly the keyword args this
    signature accepts, and read the return value's fields
    (`success`, `text`, `crystal_id`, `confidence`, `reason`,
    `tokens_used`, `cost_usd`) — all of which CognitionResult carries.

    Model routing runs through the provider-neutral seam (roles map the
    persisted wire-keys haiku/sonnet onto the small/large tiers); callers
    no longer pass a client.
    """
    env = CognitionEnvironment(
        customer_id=customer_id,
        trigger_type=trigger_type,
        trigger_id=trigger_id,
        task_goal=goal,
        conversation_context=conversation_context,
        source_crystal_id=source_crystal_id,
        output_type=OutputType(output_type),
        max_attempts=max_attempts,
    )
    _active_environments[env.id] = env
    await _persist_snapshot(store, env)

    logger.info(
        "cognition.env_created",
        env_id=env.id,
        customer_id=customer_id,
        trigger=trigger_type,
        output_type=output_type,
    )

    try:
        for attempt in range(max_attempts):
            env.attempts = attempt + 1

            # --- Phase 1: Orchestrator creates goal + plan ---
            env.status = WorkflowStatus.ORCHESTRATING
            # Q2B (2026-07-15): the ratchet feed. Open operator
            # critiques on this run (mid-run/retry) and on prior runs
            # of the same trigger enter the orchestrator's context —
            # operator judgment shapes the next contract and plan
            # instead of sitting as a sticky note. Sourced by code,
            # per-attempt (a critique written during attempt 1 lands
            # on attempt 2). Fetch failures never block a run.
            try:
                env.operator_critiques = (
                    await store.list_open_critiques_for_trigger(
                        env.customer_id,
                        trigger_id=env.trigger_id or None,
                        run_id=env.id,
                        limit=10,
                    )
                )
                if env.operator_critiques:
                    env.record_event(
                        "critiques_applied",
                        count=len(env.operator_critiques),
                        attempt=attempt + 1,
                    )
            except Exception as e:  # noqa: BLE001
                logger.warning("cognition.critique_fetch_failed",
                               env_id=env.id, error=str(e)[:200])
                env.operator_critiques = []
            await _persist_snapshot(store, env)
            try:
                goal_doc, plan = await run_orchestrator(
                    env=env,
                    store=store,
                    fact_store=fact_store,
                    encoder=encoder,
                )
                env.goal = goal_doc
                env.plan = plan
            except Exception as e:
                logger.error("cognition.orchestrator_failed", env_id=env.id, error=str(e))
                env.status = WorkflowStatus.FAILED
                return _finalize(env, success=False, reason=f"Orchestrator failed: {e}")

            # Revision routes (2026-07-10, ratified Q2A/Q5A). Honored on
            # retries only — attempt 1 has no verdict to classify, and the
            # C2 answerability park already handles genuinely unanswerable
            # tasks without burning composition/validation.
            if attempt > 0 and plan.retry_route == "give_up":
                env.status = WorkflowStatus.NEEDS_REVIEW
                explanation = (
                    plan.reasoning
                    or "Orchestrator judged the goal unachievable with "
                       "available tools."
                )
                logger.info(
                    "cognition.gave_up", env_id=env.id,
                    attempt=attempt + 1, explanation=explanation[:200],
                )
                return _finalize(
                    env, success=False,
                    reason=f"orchestrator_gave_up: {explanation}",
                )
            if attempt > 0 and plan.retry_route == "amend_contract":
                # Contract amendment (2026-07-14, ratified Q2A): the
                # appeal seat. Applied ONLY to criteria the LAST verdict
                # flagged possibly_infeasible; every application lands in
                # the goal's permanent audit trail and the validator sees
                # it. Unflagged proposals are ignored and logged — the
                # planner cannot relax criteria the judge didn't question.
                _last_val = {}
                if env.attempt_history:
                    _last_val = (
                        (env.attempt_history[-1] or {}).get("validation")
                        or {}
                    )
                _flagged_idx = {
                    i for i, c in enumerate(
                        _last_val.get("criteria_evaluation") or []
                    )
                    if isinstance(c, dict) and c.get("possibly_infeasible")
                }
                for a in plan.contract_amendments:
                    idx = a.get("criterion_index")
                    if (
                        idx not in _flagged_idx
                        or not env.goal
                        or idx >= len(env.goal.acceptance_criteria)
                    ):
                        logger.warning(
                            "cognition.amendment_rejected", env_id=env.id,
                            criterion_index=idx,
                            reason="not flagged possibly_infeasible",
                        )
                        env.record_event("amendment_rejected",
                                         criterion_index=idx)
                        continue
                    original = env.goal.acceptance_criteria[idx]
                    env.goal.acceptance_criteria[idx] = a["amended"]
                    env.goal.amendments.append({
                        "attempt": attempt + 1,
                        "index": idx,
                        "original": original,
                        "amended": a["amended"],
                        "evidence": a.get("evidence", ""),
                    })
                    logger.info(
                        "cognition.contract_amended", env_id=env.id,
                        attempt=attempt + 1, criterion_index=idx,
                        amended=a["amended"][:120],
                    )
                    env.record_event("contract_amended",
                                     criterion_index=idx,
                                     amended=a["amended"][:120])

            if attempt > 0 and plan.retry_route == "replan":
                # The anchoring hedge: the orchestrator judged the prior
                # attempt incoherent — a cold restart beats revising into
                # the same hole. Drop the carryover so workers see nothing
                # from the failed attempt.
                env.carried_findings = []
                env.prior_deliverable = ""
                logger.info(
                    "cognition.replan_cold", env_id=env.id,
                    attempt=attempt + 1,
                )

            # --- Phase 2: Workers execute steps ---
            env.status = WorkflowStatus.WORKING
            await _persist_snapshot(store, env)

            executed = set()
            remaining = list(plan.steps)
            parked_unanswerable = False
            failed_fast = None
            failed_fast_affected: set = set()

            while remaining:
                ready = [s for s in remaining if all(d in executed for d in s.depends_on)]

                if not ready:
                    logger.error("cognition.deadlock", env_id=env.id,
                                 remaining=[s.id for s in remaining])
                    break

                # Group by parallel_group
                parallel_groups: dict[Optional[str], list] = {}
                sequential = []
                for step in ready:
                    if step.parallel_group:
                        parallel_groups.setdefault(step.parallel_group, []).append(step)
                    else:
                        sequential.append(step)

                # Execute parallel groups concurrently
                for group_name, group_steps in parallel_groups.items():
                    results = await asyncio.gather(*[
                        run_worker(env, step, store, fact_store, encoder)
                        for step in group_steps
                    ])
                    for step, result in zip(group_steps, results):
                        env.step_outputs[step.id] = result
                        executed.add(step.id)
                        remaining.remove(step)

                # Execute sequential steps one at a time
                for step in sequential:
                    result = await run_worker(
                        env, step, store, fact_store, encoder,
                    )
                    env.step_outputs[step.id] = result
                    executed.add(step.id)
                    remaining.remove(step)

                    if result.output.get("is_deliverable") and result.status == StepStatus.COMPLETE:
                        env.deliverables["main"] = result.output.get("content", "")

                    if result.status == StepStatus.FAILED:
                        break

                # Q1A fail-fast (ratified 2026-07-15): a FAILED step
                # with un-executed dependents on the deliverable chain
                # dooms the attempt — stop here instead of spending
                # composition + validator money proving a rejection
                # already known.
                ff_step, ff_affected = _fail_fast_step(
                    plan, executed, remaining, env,
                )
                if ff_step is not None:
                    failed_fast = ff_step
                    failed_fast_affected = ff_affected
                    break

                # C2 answerability gate (evidence-based continue-gate). Once
                # every retrieval step has run and composition steps still
                # remain, park if retrieval produced zero grounding: nothing
                # (bank OR web/source steps) grounded the topic, so the
                # composition steps could only fabricate. Parking skips the
                # expensive analyze/synthesize/format + validator + retry burn.
                # If retrieval found anything we proceed normally.
                if _should_park_unanswerable(plan, executed, env):
                    parked_unanswerable = True
                    break

            if parked_unanswerable:
                env.status = WorkflowStatus.NEEDS_REVIEW
                logger.info(
                    "cognition.parked_unanswerable",
                    env_id=env.id,
                    attempt=attempt + 1,
                    note=(
                        "Retrieval returned no grounding and no external tool "
                        "can supply it; skipped composition/validation."
                    ),
                )
                return _finalize(
                    env, success=False,
                    reason=("needs_capability: retrieval found no grounding "
                            "in the bank; parked before composition"),
                )

            # If no deliverable flagged, use the last successful step
            # (skipped under fail-fast: promoting a surviving step's
            # content to "deliverable" would misrepresent the attempt).
            if failed_fast is None and not env.deliverables:
                for step_id in sorted(env.step_outputs.keys(), reverse=True):
                    out = env.step_outputs[step_id]
                    if out.status == StepStatus.COMPLETE:
                        content = out.output.get("content", "")
                        if content and len(content) > 50:
                            env.deliverables["main"] = content
                            break

            # --- Phase 3: Validator reviews ---
            await _persist_snapshot(store, env)  # workers done — steps visible
            if failed_fast is not None:
                # Q1A (ratified 2026-07-15): the attempt cannot produce
                # a complete deliverable — skip the validator and
                # synthesize the rejection on the SAME rails a real
                # verdict rides, so the revision routes, archive,
                # harvest, and critique feed all engage unchanged.
                ff_out = env.step_outputs.get(failed_fast.id)
                ff_error = (
                    ff_out.error if ff_out is not None and ff_out.error
                    else "no error recorded"
                )
                validation = ValidationResult(
                    approved=False,
                    score=0.0,
                    reasoning=(
                        f"FAIL-FAST: step {failed_fast.id} "
                        f"({failed_fast.action.value}) failed before its "
                        f"dependent steps ran — {ff_error}. Composition "
                        f"and validation were skipped; no deliverable "
                        f"was produced for this attempt."
                    ),
                    issues=[
                        f"step {failed_fast.id} "
                        f"({failed_fast.action.value}) failed: {ff_error}"
                    ],
                    suggestions=[
                        "The failed step's output is missing entirely — "
                        "re-acquire it before recomposing: narrower "
                        "research steps (at most 3 targets each) or "
                        "targeted web_fetch of known URLs."
                    ],
                    model_used="engine-fail-fast",
                )
                env.validation = validation
                env.record_event(
                    "fail_fast", step_id=failed_fast.id,
                    attempt=attempt + 1,
                    skipped_steps=sorted(failed_fast_affected),
                )
                logger.info(
                    "cognition.fail_fast", env_id=env.id,
                    step_id=failed_fast.id, attempt=attempt + 1,
                    skipped_steps=sorted(failed_fast_affected),
                    error=str(ff_error)[:200],
                )
            else:
                env.status = WorkflowStatus.VALIDATING
                await _persist_snapshot(store, env)
                try:
                    validation = await run_validator(env=env)
                    env.validation = validation
                except Exception as e:
                    logger.error("cognition.validator_failed", env_id=env.id, error=str(e))
                    if env.deliverables:
                        env.status = WorkflowStatus.COMPLETE
                        return await _commit_and_finalize(env, store, encoder, fact_store)
                    else:
                        env.status = WorkflowStatus.FAILED
                        return _finalize(env, success=False, reason=f"Validator failed: {e}")

            if validation.approved:
                env.status = WorkflowStatus.COMPLETE
                logger.info("cognition.approved",
                            env_id=env.id, score=validation.score,
                            attempt=attempt + 1)
                return await _commit_and_finalize(env, store, encoder, fact_store)
            else:
                env.status = WorkflowStatus.REJECTED
                env.rejection_log.append({
                    "attempt": attempt + 1,
                    "reasoning": validation.reasoning,
                    "issues": validation.issues,
                    "suggestions": validation.suggestions,
                    "score": validation.score,
                })
                # Archive the FULL attempt before the retry hygiene wipes
                # it: plan (the orchestrator re-plans each attempt, so
                # plans differ), every step's output record, the
                # deliverable (bounded), and the verdict. This is what
                # the tracker renders as per-attempt flow.
                env.attempt_history.append({
                    "attempt": attempt + 1,
                    "plan": env.plan.to_dict() if env.plan else None,
                    "steps": {
                        str(k): v.to_dict()
                        for k, v in env.step_outputs.items()
                    },
                    "deliverable": (
                        env.deliverables.get("main", "")[:12000]
                    ),
                    "validation": validation.to_dict(),
                })
                # Revision-aware retry (2026-07-10, ratified Q1A): before
                # the hygiene clear, harvest what the NEXT attempt revises
                # from — the retrieval findings already paid for and the
                # rejected deliverable. Prior to this, the archive above
                # served only the observer (AttemptFlow); attempts were
                # independent cold re-rolls, and under identical inputs a
                # re-roll is as likely worse as better (the rematch-#3
                # regression). The verdict itself rides in rejection_log.
                env.carried_findings = _harvest_findings(env)
                env.prior_deliverable = env.deliverables.get("main", "")
                env.step_outputs.clear()
                env.deliverables.clear()
                env.validation = None
                logger.info("cognition.rejected",
                            env_id=env.id, score=validation.score,
                            attempt=attempt + 1,
                            issues=validation.issues)

        env.status = WorkflowStatus.NEEDS_REVIEW
        # 2026-07-10 (filed by the shadow critic itself): the bare
        # "Failed after N attempts" gave the calling agent zero
        # diagnostics — which invited it to CONFABULATE a cause to the
        # user ("token-budget"). Structured per-attempt reasons let the
        # agent relay the truth, the critic see it, and the operator
        # read it without opening the pane.
        attempt_summaries = "; ".join(
            f"attempt {r.get('attempt', '?')}: "
            f"score {r.get('score', 0.0):.0%} — "
            + str(
                (r.get("issues") or [r.get("reasoning", "no detail")])[0]
            )[:140]
            for r in env.rejection_log
        )
        reason = f"Failed after {max_attempts} attempts"
        if attempt_summaries:
            reason = f"{reason}. {attempt_summaries}"
        return _finalize(env, success=False, reason=reason)

    except Exception as e:
        env.status = WorkflowStatus.FAILED
        logger.error("cognition.workflow_error", env_id=env.id, error=str(e))
        return _finalize(env, success=False, reason=str(e))
    finally:
        # S9: terminal snapshot — whatever status the env exited with,
        # the row records it and stamps completed_at. Guaranteed on
        # every exit path (success, rejection, exception).
        await _persist_snapshot(store, env, terminal=True)


async def _commit_deliverable_to_scratchpad(
    env: CognitionEnvironment,
    store: "MetadataStore",
    deliverable: Optional[str],
) -> Optional[str]:
    """Commit an approved deliverable to the crystallization scratchpad
    (document_uploads, inferred_knowledge lane, review-gated). Shared by
    the CRYSTAL and (Q4A) REPORT output paths. Returns the upload id, or
    None (trivial deliverable, or commit failure — logged, never raised:
    commit is best-effort and must not fail the run)."""
    if not deliverable or len(deliverable) <= 20:
        return None
    try:
        suggested_key = env.plan.suggested_key if env.plan else ""

        # C3: deterministic groundedness gate. The validator is
        # barriered from step outputs and cannot tell retrieved facts
        # from reconstructed ones; the engine can. We do not block
        # (cognition output goes to the review queue, not live
        # knowledge) — we stamp the verdict on the document label so
        # the human reviewer sees it, and log it for telemetry.
        from .groundedness import assess_groundedness
        grounding = assess_groundedness(deliverable, env.step_outputs)

        base_label = f"{suggested_key or env.goal.title} (cognition)"
        if grounding["verdict"] == "ungrounded":
            label = f"{base_label} [grounding: ungrounded]"
            logger.warning(
                "cognition.commit_ungrounded",
                env_id=env.id,
                ungrounded_paths=grounding["ungrounded_paths"],
                cert_phrases=grounding["cert_phrases"],
                had_retrieval=grounding["had_retrieval"],
                note=(
                    "Deliverable asserts paths/verification absent "
                    "from retrieved facts; committing to review flagged."
                ),
            )
        else:
            label = base_label

        doc = await store.create_document_upload(
            customer_id=env.customer_id,
            label=label,
            text=deliverable,
            detected_type="inferred_knowledge",
        )
        logger.info("cognition.committed_deliverable",
                    env_id=env.id, upload_id=doc.id,
                    output_type=env.output_type.value,
                    key=suggested_key, chars=len(deliverable),
                    grounding=grounding["verdict"])
        return doc.id
    except Exception as e:  # noqa: BLE001
        logger.error("cognition.commit_failed", env_id=env.id, error=str(e))
        return None


async def _commit_and_finalize(
    env: CognitionEnvironment,
    store: "MetadataStore",
    encoder: Any,
    fact_store: Any,
) -> CognitionResult:
    """Commit approved output to persistent storage.

    v2 port (Phase 6 Wave C): the OutputType.CRYSTAL branch now calls
    `store.create_document_upload(...)` instead of inline
    `DocumentUploadRow` construction + session.add. The Phase 5 store
    method accepts `detected_type` as a kwarg, returns a
    `DocumentUpload` Pydantic, and the upload lands in v1's
    `status='pending'` (the default). The crystallization worker
    picks it up from there and runs the chunk/extract pipeline.
    """
    deliverable = env.get_final_deliverable()

    if env.output_type == OutputType.REPORT:
        # Q4A (2026-07-11, ratified): approved reports ALSO commit
        # through the crystallization scratchpad. Before this, only
        # output_type=crystal fed the bank; the agent's cognition_run
        # uses report, so validated research evaporated — three rematch
        # runs paid for the same video-ecosystem research and the next
        # run's orchestrator sourcing would still have found nothing.
        # The commit is review-gated (same inferred_knowledge lane as
        # agent uploads: nothing enters recall without approval) and
        # BEST-EFFORT: a commit failure never fails the report — the
        # caller still gets its text.
        upload_id = await _commit_deliverable_to_scratchpad(
            env, store, deliverable,
        )
        return _finalize(env, success=True, text=deliverable,
                         crystal_id=upload_id)

    elif env.output_type == OutputType.CRYSTAL:
        if deliverable and len(deliverable) > 20:
            upload_id = await _commit_deliverable_to_scratchpad(
                env, store, deliverable,
            )
            if upload_id:
                return _finalize(env, success=True, text=deliverable,
                                 crystal_id=upload_id)
            return _finalize(env, success=True, text=deliverable,
                             reason="Commit failed, returning as report")
        else:
            return _finalize(env, success=False,
                             reason="No deliverable content to commit")

    elif env.output_type == OutputType.FILE:
        return _finalize(env, success=True, text=deliverable,
                         reason="File output not yet implemented, returning as report")

    return _finalize(env, success=False, reason="Unknown output type")


def _finalize(
    env: CognitionEnvironment,
    *,
    success: bool,
    text: Optional[str] = None,
    crystal_id: Optional[str] = None,
    file_path: Optional[str] = None,
    reason: Optional[str] = None,
) -> CognitionResult:
    """Create result, clean up environment."""
    result = CognitionResult(
        success=success,
        text=text,
        crystal_id=crystal_id,
        file_path=file_path,
        confidence=env.validation.score if env.validation else 0.0,
        workflow_summary=env.to_dict(),
        reason=reason,
        tokens_used=env.tokens_used,
        cost_usd=env.total_cost_usd,
    )

    # Keep completed environments for UI polling (TTL cleanup in production).
    # Don't destroy immediately so the frontend can show final state.

    return result
