"""QueryLog — every request/response captured for telemetry. §6 of BUILD_PROPOSAL.md.

Research-grounded additions (§4 telemetry loop):
  - injection_method           — what path was used
  - confidence_gate_fires      — per-token uncertainty triggers
  - response_confidence_at_commit — lp_at_answer
  - shadow_ran, shadow_delta   — ground truth when available
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


MatchType = Literal["high", "medium", "low", "none"]
InjectionMethod = Literal[
    "text",
    "text+text",  # SPREAD routing: two references injected with hedged framing
    "text+concept",
    "text+synthesis",  # SPREAD-decision path injects bind-v1 decoded joint statement
    "text+synthesis+concept",  # SPREAD + concept-path hint together
    "hidden_state",
    "cache_hit",  # April 2026: cached answer served without upstream call (GAIA fold-back)
    # C2 (2026-07-08): the AGENT surface — knowledge arrives via retrieval
    # TOOLS in the loop, not prompt injection. Agent turns log with this.
    "agent_tools",
    "none",
]


class QueryLog(BaseModel):
    id: str
    customer_id: str

    query_text: str
    query_vector: list[float] = Field(default_factory=list)  # 10k-dim

    match_type: MatchType
    injection_method: InjectionMethod = "none"
    # Reserved: per-token gate fires for the parked hidden-state research
    # line (docs/RESEARCH_DIRECTIONS.md). Always 0 on the prompt-injection
    # product path; the column is kept so re-opening the research needs no
    # migration.
    confidence_gate_fires: int = 0

    # List of fact_ids that were retrieved and (maybe) used
    matched_facts: list[str] = Field(default_factory=list)

    # Response
    response_text: Optional[str] = None
    response_confidence_at_commit: Optional[float] = None  # lp_at_answer

    # Whether we called the customer's upstream model (for token savings calc)
    upstream_call_made: bool = True

    # Shadow evaluator output
    shadow_ran: bool = False
    shadow_delta: Optional[float] = None
    # TODO: spec the delta metric. String match? Semantic similarity?
    #       Length diff? We'll learn what's useful from §4 replay.

    # v0.4 token accounting. Source of truth is the upstream response's
    # `usage` block; we persist what came back. None means "upstream did
    # not report it" — either the call failed, the upstream didn't include
    # usage (some self-hosted endpoints don't), or the shadow didn't run.
    #
    # prompt_token_overhead is redundant (prompt_tokens - shadow_prompt_tokens)
    # but persisted so dashboard queries aggregate a single column instead
    # of subtracting two nullable columns per row.
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    shadow_prompt_tokens: Optional[int] = None
    shadow_completion_tokens: Optional[int] = None
    prompt_token_overhead: Optional[int] = None

    # v0.3+ concept-path observations. Recorded whether or not the
    # concept path influenced routing. Used by offline analysis to
    # correlate concept-match quality with shadow outcomes.
    concept_top_config: Optional[str] = None
    concept_top_score: Optional[float] = None
    concept_payload: Optional[dict] = None

    # Stage 2a (April 2026): sequence anchoring.
    #
    # Both nullable. Pre-migration rows have NULL; new rows always
    # populate them when there's a user message to anchor on. None
    # means "unsequenced" — the row exists but is not part of any
    # conversation we can attach feedback to.
    sequence_id: Optional[str] = None
    turn_index: Optional[int] = None

    # Phase 1.2 (April 2026): routing-decision telemetry.
    #
    # routed_crystal_id is the crystal id the four-way classifier
    # picked as top-1. None when retrieval found no candidates or
    # when retrieval failed entirely.
    #
    # top1_score / top2_score are the cosine similarities of the
    # top two candidates. They feed offline computation of margin
    # signal (top1 - top2) for the Bricken et al. 2023 eviction
    # heuristic.
    #
    # Independent nullability: routed_crystal_id can be None while
    # top1_score is populated (match below cleanup threshold), and
    # vice versa. Read sites should not assume both populate together.
    routed_crystal_id: Optional[str] = None
    top1_score: Optional[float] = None
    top2_score: Optional[float] = None

    # V2: sparse key used for this query's retrieval.
    sparse_key: Optional[str] = None

    latency_ms: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
