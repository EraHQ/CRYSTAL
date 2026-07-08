"""SpendBudget — one cap for one spend function (S4, 2026-07-08).

The tenant-owned budget substrate ratified in the Gap Engine redesign
(docs/GAP_ENGINE_AND_LEARN_REDESIGN.md): budgets are rows, not
per-feature knobs. `function` names the spend function ('auto_research'
first); `operator_id` narrows to one team member (F1 follow-on; None =
whole tenant); `cap_micro_usd` of 0 means the function is OFF for auto
paths — the manual-by-default posture. Enforcement reads the llm_calls
ledger by origin (control/admission.py: function_budget_allows).
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field

BudgetPeriod = Literal["daily", "monthly"]


class SpendBudget(BaseModel):
    id: str
    customer_id: str
    function: str
    operator_id: Optional[str] = None
    period: BudgetPeriod = "monthly"
    cap_micro_usd: int = 0
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    updated_at: Optional[datetime] = None
