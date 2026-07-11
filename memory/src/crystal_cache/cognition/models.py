"""Cognition data models.

Defines the shared state structures for the multi-agent cognition environment.
All state lives in-memory during execution and gets serialized to JSON for
persistence in cognition_tasks.result.

Verbatim port from v1 cognition/models.py — no SQL, pure dataclasses.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class WorkflowStatus(str, Enum):
    CREATED = "created"
    ORCHESTRATING = "orchestrating"
    WORKING = "working"
    VALIDATING = "validating"
    COMPLETE = "complete"
    REJECTED = "rejected"
    FAILED = "failed"
    NEEDS_REVIEW = "needs_human_review"
    DESTROYED = "destroyed"


class StepAction(str, Enum):
    CRYSTAL_SEARCH = "crystal_search"
    CRYSTAL_KEY_SCAN = "crystal_key_scan"
    WEB_SEARCH = "web_search"
    SOURCE_LOOKUP = "source_lookup"
    ANALYZE = "analyze"
    SYNTHESIZE = "synthesize"
    FORMAT = "format"


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    SKIPPED = "skipped"


class OutputType(str, Enum):
    CRYSTAL = "crystal"
    FILE = "file"
    REPORT = "report"


# ---------------------------------------------------------------------------
# Goal Document (contract between orchestrator and validator ONLY)
# ---------------------------------------------------------------------------

@dataclass
class GoalDocument:
    """The contract. Orchestrator writes it, validator evaluates against it.
    Workers never see this."""
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:12]}")
    title: str = ""
    description: str = ""
    acceptance_criteria: list[str] = field(default_factory=list)
    output_type: OutputType = OutputType.CRYSTAL
    output_metadata: dict[str, Any] = field(default_factory=dict)
    source_context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "acceptance_criteria": self.acceptance_criteria,
            "output_type": self.output_type.value,
            "output_metadata": self.output_metadata,
            "source_context": self.source_context,
        }


# ---------------------------------------------------------------------------
# Plan (orchestrator writes, workers read)
# ---------------------------------------------------------------------------

@dataclass
class PlanStep:
    """A single step in the execution plan."""
    id: int
    action: StepAction
    description: str
    input: dict[str, Any] = field(default_factory=dict)
    depends_on: list[int] = field(default_factory=list)
    parallel_group: Optional[str] = None
    model: str = "haiku"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "action": self.action.value,
            "description": self.description,
            "input": self.input,
            "depends_on": self.depends_on,
            "parallel_group": self.parallel_group,
            "model": self.model,
        }


@dataclass
class Plan:
    """Execution plan. Workers see this but not the goal."""
    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:12]}")
    reasoning: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    expected_output: str = ""
    suggested_key: str = ""
    parent_crystal_id: str = ""
    # Revision-aware retry (2026-07-10, ratified Q2A/Q5A): on attempt>1
    # the orchestrator classifies the validator's verdict and picks a
    # route — "compose_only" (findings were fine; revise the deliverable
    # without new research), "gap_fill" (targeted research for the named
    # gaps, then revise), "replan" (prior attempt incoherent; cold
    # restart — carryover is dropped), or "give_up" (the goal is not
    # achievable with available tools; stop burning budget and explain).
    # Empty on attempt 1.
    retry_route: str = ""
    # Q4A: the orchestrator PROPOSES the per-call output-token budget for
    # this plan's composition steps, honestly sized to the deliverable it
    # expects; a platform ceiling in roles.py clamps it. 0 = the default
    # composition ceiling. Re-proposed fresh each attempt — budgets reset,
    # they never shrink across retries (ratified: shrinking budgets make
    # later attempts structurally worse).
    max_output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "reasoning": self.reasoning,
            "steps": [s.to_dict() for s in self.steps],
            "expected_output": self.expected_output,
            "suggested_key": self.suggested_key,
            "parent_crystal_id": self.parent_crystal_id,
            "retry_route": self.retry_route,
            "max_output_tokens": self.max_output_tokens,
        }


# ---------------------------------------------------------------------------
# Step Output (workers write, subsequent workers read)
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """Result from a single worker step."""
    step_id: int
    action: str
    status: StepStatus = StepStatus.PENDING
    output: dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    tokens_in: int = 0
    tokens_out: int = 0
    model_used: str = ""
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    duration_ms: int = 0

    def to_dict(self) -> dict:
        return {
            "step_id": self.step_id,
            "action": self.action,
            "status": self.status.value,
            "output": self.output,
            "error": self.error,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model_used": self.model_used,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": self.duration_ms,
        }


# ---------------------------------------------------------------------------
# Validation (validator writes)
# ---------------------------------------------------------------------------

@dataclass
class CriterionEval:
    """Evaluation of a single acceptance criterion."""
    criterion: str
    status: str  # "MET", "PARTIALLY_MET", "NOT_MET"
    evidence: str = ""

    def to_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "status": self.status,
            "evidence": self.evidence,
        }


@dataclass
class ValidationResult:
    """Validator's verdict."""
    approved: bool = False
    score: float = 0.0
    reasoning: str = ""
    criteria_evaluation: list[CriterionEval] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    model_used: str = ""

    def to_dict(self) -> dict:
        return {
            "approved": self.approved,
            "score": self.score,
            "reasoning": self.reasoning,
            "criteria_evaluation": [c.to_dict() for c in self.criteria_evaluation],
            "issues": self.issues,
            "suggestions": self.suggestions,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# Environment (ephemeral, per-task)
# ---------------------------------------------------------------------------

@dataclass
class CognitionEnvironment:
    """Ephemeral environment for a single cognition task.

    Created when a task starts, destroyed when it completes.
    Only validated outputs survive into persistent storage.
    """
    id: str = field(default_factory=lambda: f"env_{uuid.uuid4().hex[:12]}")
    customer_id: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    max_wall_time: float = 300.0
    token_budget: int = 50_000
    tokens_used: int = 0

    # Trigger info
    trigger_type: str = ""
    trigger_id: str = ""
    conversation_context: str = ""
    # 2026-07-09 (video-infra run): the caller's goal text, first-class.
    # Previously the engine folded it into conversation_context ("context
    # or goal") and the orchestrator prompt read ONLY the context — so a
    # caller passing BOTH silently lost the goal. The orchestrator now
    # reads task_goal as the TASK and context as supporting color.
    task_goal: str = ""
    source_crystal_id: str = ""

    # Scratch storage (information barriers enforced by the engine, not the model)
    goal: Optional[GoalDocument] = None
    plan: Optional[Plan] = None
    step_outputs: dict[int, StepOutput] = field(default_factory=dict)
    deliverables: dict[str, str] = field(default_factory=dict)
    validation: Optional[ValidationResult] = None
    rejection_log: list[dict[str, Any]] = field(default_factory=list)
    # 2026-07-09: full per-attempt archive. The engine CLEARS step_outputs
    # and deliverables on rejection (information hygiene for the retry),
    # which destroyed the evidence the tracker exists to show — a failed
    # run's trace held only the LAST attempt's steps beside N validation
    # stubs. Each entry: {attempt, plan, steps, deliverable, validation}.
    attempt_history: list[dict[str, Any]] = field(default_factory=list)
    # Revision-aware retry (2026-07-10, ratified Q1A): attempts are
    # REVISIONS, not independent samples. What carries across a
    # rejection: the retrieval findings already paid for
    # (carried_findings — rendered text per completed retrieval step)
    # and the rejected deliverable (prior_deliverable, full text; trimmed
    # at injection). What resets: the plan and step statuses (the
    # orchestrator replans every attempt). The verdict rides in
    # rejection_log as before. The "replan" route drops both fields —
    # the anchoring hedge for an incoherent prior attempt.
    carried_findings: list[dict[str, Any]] = field(default_factory=list)
    prior_deliverable: str = ""

    # Lifecycle
    status: WorkflowStatus = WorkflowStatus.CREATED
    output_type: OutputType = OutputType.CRYSTAL
    attempts: int = 0
    max_attempts: int = 3

    # Cost tracking
    total_cost_usd: float = 0.0

    def record_tokens(self, tokens_in: int, tokens_out: int, model: str):
        """Track token usage and estimate cost.

        The env-level figure is a UI ESTIMATE keyed by the plan's wire-key
        (haiku/sonnet); the authoritative per-call ledger rows come from
        record_model_call via cost/pricing.py. Rates mirror the verified
        table there (2026-07-02: haiku 1/5, sonnet 3/15 USD per Mtok) —
        update BOTH when rates move.
        """
        self.tokens_used += tokens_in + tokens_out
        costs = {
            "haiku": (1.00, 5.00),
            "sonnet": (3.00, 15.00),
        }
        rate_in, rate_out = costs.get(model, (1.00, 5.00))
        self.total_cost_usd += (tokens_in * rate_in + tokens_out * rate_out) / 1_000_000

    def get_final_deliverable(self) -> Optional[str]:
        """Return the main deliverable text, or None."""
        if self.deliverables:
            return next(iter(self.deliverables.values()))
        return None

    def to_dict(self) -> dict:
        """Full serialization for persistence and API responses."""
        return {
            "id": self.id,
            "customer_id": self.customer_id,
            "created_at": self.created_at.isoformat(),
            "status": self.status.value,
            "trigger_type": self.trigger_type,
            "trigger_id": self.trigger_id,
            "output_type": self.output_type.value,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "tokens_used": self.tokens_used,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "goal": self.goal.to_dict() if self.goal else None,
            "plan": self.plan.to_dict() if self.plan else None,
            "steps": {
                str(k): v.to_dict() for k, v in self.step_outputs.items()
            },
            "deliverables": {k: v[:500] + "..." if len(v) > 500 else v
                            for k, v in self.deliverables.items()},
            "validation": self.validation.to_dict() if self.validation else None,
            "rejection_log": self.rejection_log,
            "attempt_history": self.attempt_history,
            "carried_findings": [
                {**f, "text": (f.get("text", "") or "")[:500]}
                for f in self.carried_findings
            ],
            "prior_deliverable": (
                self.prior_deliverable[:500] + "..."
                if len(self.prior_deliverable) > 500
                else self.prior_deliverable
            ),
        }

    def destroy(self):
        """Release all state. Call only after outputs are committed."""
        self.goal = None
        self.plan = None
        self.step_outputs.clear()
        self.deliverables.clear()
        self.validation = None
        self.carried_findings.clear()
        self.prior_deliverable = ""
        self.status = WorkflowStatus.DESTROYED


# ---------------------------------------------------------------------------
# Final result
# ---------------------------------------------------------------------------

@dataclass
class CognitionResult:
    """What gets returned to the caller after the environment is destroyed."""
    success: bool
    text: Optional[str] = None
    crystal_id: Optional[str] = None
    file_path: Optional[str] = None
    confidence: float = 0.0
    workflow_summary: Optional[dict] = None
    reason: Optional[str] = None
    tokens_used: int = 0
    cost_usd: float = 0.0
