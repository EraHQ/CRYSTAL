"""Residual gaps (ratified 2026-07-16) + verdict-aware commit.

An approved verdict names precisely what remains unverified — the
system's clearest statement of its own knowledge gaps. The commit path
now converts the validator's residual_gaps into open researchable
knowledge gaps (side by side with the scratchpad commit) and stamps
the validator's grade on the upload label so the reviewer sees the
epistemic grade without opening the run. Approved-only by placement
(Q2A); fill_gap triggers never spawn (Q3A — no gap-begets-gap chains).

R14 note: verified by pytest; describes expected behavior.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.cognition import engine as engine_mod
from crystal_cache.cognition.engine import run_cognition_workflow
from crystal_cache.cognition.models import (
    CognitionEnvironment,
    CriterionEval,
    GoalDocument,
    Plan,
    PlanStep,
    StepAction,
    StepOutput,
    StepStatus,
    ValidationResult,
)


def _goal(title: str = "Video toolchain report") -> GoalDocument:
    return GoalDocument(title=title, description="d",
                        acceptance_criteria=["c"])


def _plan() -> Plan:
    return Plan(steps=[
        PlanStep(id=1, action=StepAction.FORMAT, description="f"),
    ], reasoning="scripted")


def _verdict(residuals: list[dict], partials: int = 1) -> ValidationResult:
    crits = [CriterionEval(criterion="a", status="MET", evidence="e")]
    crits += [
        CriterionEval(criterion=f"p{i}", status="PARTIALLY_MET",
                      evidence="weak")
        for i in range(partials)
    ]
    return ValidationResult(
        approved=True, score=0.78, reasoning="ok with residuals",
        criteria_evaluation=crits, residual_gaps=residuals,
    )


def _scripted(monkeypatch, validator_result):
    async def fake_orchestrator(*, env, store, fact_store, encoder=None):
        return (_goal(), _plan())

    async def fake_worker(env, step, _store, _fact_store, _encoder):
        out = StepOutput(step_id=step.id, action=step.action.value,
                         status=StepStatus.COMPLETE)
        out.output = {
            "content": ("deliverable text comfortably longer than "
                        "fifty characters for the salvage gate"),
            "is_deliverable": True,
        }
        return out

    async def fake_validator(*, env):
        return validator_result

    monkeypatch.setattr(engine_mod, "run_orchestrator", fake_orchestrator)
    monkeypatch.setattr(engine_mod, "run_worker", fake_worker)
    monkeypatch.setattr(engine_mod, "run_validator", fake_validator)


async def _cust(store) -> str:
    return (await store.create_customer(
        provider="anthropic", model_id="m", api_key_ref="ref")).id


# ---------------------------------------------------------------------------
# Full-run behavior
# ---------------------------------------------------------------------------

async def test_approved_run_spawns_residual_gaps_and_stamps_label(
        monkeypatch, store):
    cust = await _cust(store)
    _scripted(monkeypatch, _verdict([
        {"subject": "LTX Desktop launch date",
         "missing": "Exact launch date of LTX Desktop from the repo "
                    "or an official announcement"},
        {"subject": "LTX Desktop license",
         "missing": "LTX Desktop's open-source license, confirmed "
                    "from its repository"},
    ]))
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-res", max_attempts=1,
    )
    assert result.success is True

    gaps = await store.list_knowledge_gaps(cust, status="open", limit=10)
    assert len(gaps) == 2
    for g in gaps:
        assert g.source == "run_residual"
        assert g.disposition == "researchable"
        assert g.triggering_query == "Video toolchain report"
    subjects = {g.subject for g in gaps}
    assert "LTX Desktop launch date" in subjects

    # Verdict-aware commit: the reviewer sees the grade on the label.
    uploads = await store.list_document_uploads(cust)
    assert uploads, "approved report must commit to the scratchpad"
    assert "[verdict: 78%, 1 partial]" in uploads[0].label


async def test_fill_gap_trigger_never_spawns(monkeypatch, store):
    """Q3A: no gap-begets-gap chains — a fill run's residuals die."""
    cust = await _cust(store)
    _scripted(monkeypatch, _verdict([
        {"subject": "s", "missing": "A concrete fact left unverified"},
    ]))
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="crystal", trigger_type="fill_gap",
        trigger_id="gap-x", max_attempts=1,
    )
    assert result.success is True
    gaps = await store.list_knowledge_gaps(cust, status="open", limit=10)
    assert gaps == []


async def test_rejected_run_spawns_nothing(monkeypatch, store):
    """Q2A: the commit path is the spawn site, so rejection means no
    gaps even if the validator (incorrectly) listed residuals."""
    cust = await _cust(store)
    verdict = ValidationResult(
        approved=False, score=0.2, reasoning="no",
        issues=["bad"],
        residual_gaps=[{"subject": "s", "missing": "Should not spawn"}],
    )
    _scripted(monkeypatch, verdict)
    result = await run_cognition_workflow(
        "goal", cust, store, None, None,
        output_type="report", trigger_type="research",
        trigger_id="trig-rej", max_attempts=1,
    )
    assert result.success is False
    gaps = await store.list_knowledge_gaps(cust, status="open", limit=10)
    assert gaps == []


# ---------------------------------------------------------------------------
# Helper-level: dedup, cap, malformed candidates
# ---------------------------------------------------------------------------

async def test_spawn_dedups_against_open_gaps_and_caps_at_three(store):
    cust = await _cust(store)
    await store.create_knowledge_gap(
        cust, domain=None, subject="dup",
        missing="Already open fact", source="manual",
    )
    env = CognitionEnvironment(customer_id=cust, trigger_type="research")
    env.goal = _goal()
    env.validation = _verdict([
        {"subject": "dup", "missing": "already open fact"},  # dedup (ci)
        {"subject": "a", "missing": "New fact A"},
        {"subject": "b", "missing": "New fact B"},
        {"subject": "c", "missing": "New fact C — beyond the cap"},
        {"subject": "bad", "missing": "   "},                # malformed
    ])
    spawned = await engine_mod._spawn_residual_gaps(env, store)
    # Candidates cap at 3 BEFORE dedup: [dup, A, B] -> dup skipped.
    assert spawned == 2
    gaps = await store.list_knowledge_gaps(cust, status="open", limit=10)
    missing = {g.missing for g in gaps}
    assert missing == {"Already open fact", "New fact A", "New fact B"}
    assert any(e.get("kind") == "residual_gaps_spawned"
               for e in env.events)


async def test_spawn_requires_validation_and_store(store):
    cust = await _cust(store)
    env = CognitionEnvironment(customer_id=cust, trigger_type="research")
    env.goal = _goal()
    env.validation = None
    assert await engine_mod._spawn_residual_gaps(env, store) == 0
    env.validation = _verdict([])
    assert await engine_mod._spawn_residual_gaps(env, store) == 0
    assert await engine_mod._spawn_residual_gaps(env, None) == 0


# ---------------------------------------------------------------------------
# Seam pins
# ---------------------------------------------------------------------------

async def test_validator_schema_and_engine_seams():
    import inspect
    import crystal_cache.cognition.roles as roles_mod
    v_src = inspect.getsource(roles_mod.run_validator)
    assert "residual_gaps" in v_src           # prompt schema + parse
    assert "ONLY when approving" in v_src     # the concrete-facts guard
    c_src = inspect.getsource(engine_mod._commit_and_finalize)
    assert "_spawn_residual_gaps" in c_src    # side-by-side placement
    h_src = inspect.getsource(engine_mod._spawn_residual_gaps)
    assert "fill_gap" in h_src                # Q3A chain guard
    assert ValidationResult().residual_gaps == []
