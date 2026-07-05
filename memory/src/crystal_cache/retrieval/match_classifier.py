"""Match classifier — similarity score → discrete routing decision.

Two classifier paths share this module:

  Legacy three-way (high/medium/low) — single top-1 cosine score. Drives
  the existing pipeline.classify_for_customer() flow that picks
  inject-with-context vs pass-through.

      top_score ≥ thresholds.high      → "high"
      top_score ≥ thresholds.medium    → "medium"
      else                              → "low"

  Four-way routing decision (April 2026) — top-K margins + noise floor.
  Drives the pipeline that decides whether to invoke bind-v1 synthesis
  for spread matches. Defined in CLAUDE.md's "When to use which decoder"
  section.

      top1 < noise_floor                          → NoMatch
      top1 − top2 ≥ perfect_margin AND top1 large → Perfect
      top1 − top2 ≥ spread_margin                 → Spread
      otherwise                                    → LowConfidence

The two classifiers are intentionally NOT a single function. They take
different inputs (one float vs a list of top-K scores), they answer
different questions ("how confident is the top match?" vs "what should
the pipeline do next?"), and the legacy classifier's contract is
exercised by tests that should keep working unchanged. The four-way
classifier is added alongside, not in place of, the legacy one.

Research-grounded addition (§4): confidence-based re-routing is a
SECONDARY signal that runs AFTER the upstream model commits, not here.
This classifier is purely retrieval-similarity. The confidence gate
lives in the execution layer and is used only on the hidden-state
injection path.

Per-customer thresholds come from Customer.retrieval_thresholds; both
classifiers are pure functions of (scores, thresholds) so we can
unit-test them cleanly.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal, Sequence

from ..models import Customer
from ..models.customer import RetrievalThresholds


# -----------------------------------------------------------------------------
# Legacy three-way classifier — UNCHANGED to preserve existing tests/pipeline.
# -----------------------------------------------------------------------------

MatchType = Literal["high", "medium", "low", "none"]


class MatchClassifier:
    """Stateless. Pure function of (top_score, thresholds).

    Provides BOTH the legacy three-way classifier (`classify`) and the
    new four-way routing classifier (`classify_routing`). They live on
    the same class so callers can switch between them without rewiring
    construction sites.
    """

    # -------------------------------------------------------------------
    # Legacy three-way path
    # -------------------------------------------------------------------

    def classify(
        self,
        top_score: float,
        thresholds: RetrievalThresholds,
    ) -> MatchType:
        """Classify the top similarity score using the customer's thresholds.

        Note on the "none" value: this classifier never returns "none".
        "none" in the QueryLog schema means "no retrieval was attempted"
        (e.g. pass-through before Group D lands). The classifier always
        returns one of high/medium/low because by definition it's only
        called AFTER retrieval ran.
        """
        if top_score >= thresholds.high:
            return "high"
        if top_score >= thresholds.medium:
            return "medium"
        return "low"

    def classify_for_customer(
        self, top_score: float, customer: Customer
    ) -> MatchType:
        """Ergonomic helper when you have a Customer in hand."""
        return self.classify(top_score, customer.retrieval_thresholds)

    # -------------------------------------------------------------------
    # New four-way routing path (April 2026)
    # -------------------------------------------------------------------

    def classify_routing(
        self,
        scores: Sequence[float],
        thresholds: RetrievalThresholds,
    ) -> "RoutingResult":
        """Classify a top-K cosine-score list into a routing decision.

        Implements the table from CLAUDE.md's "When to use which decoder"
        section. The decision branches:

          1. Empty scores → NoMatch.
          2. top1 below noise_floor → NoMatch (random-orientation regime).
          3. Only one score (no top-2) → Perfect if above noise_floor;
             we have nothing to compare against, so we trust the top-1.
          4. (top1 − top2) ≥ perfect_margin → Perfect.
          5. (top1 − top2) ≥ spread_margin AND top2 above noise_floor →
             Spread (synthesis candidate). The top-2 condition prevents
             "spread" decisions where top-1 is high but top-2 is noise —
             that's just a perfect match shaped weirdly by the bank.
          6. Otherwise → LowConfidence.

        The returned RoutingResult carries the diagnostic numbers
        (top1, top2, margin) so callers can log them without re-deriving.

        Args:
            scores: Cosine scores in DESCENDING order. Caller is
                responsible for the ordering — VectorStore.search()
                already sorts descending.
            thresholds: Customer.retrieval_thresholds.

        Returns:
            RoutingResult with .decision and the diagnostic fields.
        """
        if not scores:
            return RoutingResult(
                decision=RoutingDecision.NO_MATCH,
                top1=0.0,
                top2=None,
                margin=None,
            )

        top1 = float(scores[0])
        top2: float | None = float(scores[1]) if len(scores) > 1 else None
        margin: float | None = (top1 - top2) if top2 is not None else None

        # Below noise floor → no signal at all.
        if top1 < thresholds.noise_floor:
            return RoutingResult(
                decision=RoutingDecision.NO_MATCH,
                top1=top1,
                top2=top2,
                margin=margin,
            )

        # Single-element bank: nothing to compare against, treat as perfect
        # so long as top-1 cleared the noise floor.
        if top2 is None:
            return RoutingResult(
                decision=RoutingDecision.PERFECT,
                top1=top1,
                top2=top2,
                margin=margin,
            )

        assert margin is not None

        if margin >= thresholds.perfect_margin:
            return RoutingResult(
                decision=RoutingDecision.PERFECT,
                top1=top1,
                top2=top2,
                margin=margin,
            )

        if margin >= thresholds.spread_margin and top2 >= thresholds.noise_floor:
            return RoutingResult(
                decision=RoutingDecision.SPREAD,
                top1=top1,
                top2=top2,
                margin=margin,
            )

        return RoutingResult(
            decision=RoutingDecision.LOW_CONFIDENCE,
            top1=top1,
            top2=top2,
            margin=margin,
        )

    def classify_routing_for_customer(
        self, scores: Sequence[float], customer: Customer
    ) -> "RoutingResult":
        """Ergonomic helper when you have a Customer in hand."""
        return self.classify_routing(scores, customer.retrieval_thresholds)


# -----------------------------------------------------------------------------
# Four-way RoutingDecision and the structured result type
# -----------------------------------------------------------------------------


class RoutingDecision(str, Enum):
    """Four-way routing decision per CLAUDE.md's decision table.

    String enum so it serializes naturally into QueryLog rows and
    structlog events. The string values are stable contract; don't
    rename them without a migration on the QueryLog schema.
    """

    PERFECT = "perfect"
    """Top-1 owns the query. Serve top-1's stored answer text directly.
    No decoder needed."""

    SPREAD = "spread"
    """Two FAQs both plausibly relevant. Synthesize via bind-v1 on
    bind(top1, top2) and inject the joint statement alongside both
    raw FAQ texts."""

    LOW_CONFIDENCE = "low_confidence"
    """Top-1 above noise but no clear winner. Pass to upstream LLM
    with a hint that we don't have a confident match. Do NOT synthesize
    \u2014 decoder would hallucinate from noise."""

    NO_MATCH = "no_match"
    """Top-1 at or below noise floor, or no candidates at all. Pass
    query through unmodified."""


@dataclass(frozen=True)
class RoutingResult:
    """Four-way classifier output plus diagnostic numbers.

    Diagnostic fields are exposed so callers (pipeline, telemetry)
    can log or branch on them without re-deriving. They're never
    None when `decision` is anything other than NO_MATCH on an
    empty score list.
    """

    decision: RoutingDecision
    top1: float
    top2: float | None
    margin: float | None
