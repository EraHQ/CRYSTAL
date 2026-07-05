"""Diagnostic engine — §4 of BUILD_PROPOSAL.md.

Takes per-crystal telemetry events and produces a CrystalDiagnostic record
that captures WHY a crystal is performing the way it is. This is the
learning loop's analysis stage.

Input shape (`CrystalEvent`):
  crystal_id, case_idx, outcome (helped/hurt/both_ok/both_fail),
  compression_ratio, response_text, baseline_failure_mode (optional),
  query_text (optional)

Output (`CrystalDiagnostic`, see models/diagnostic.py):
  failure_mode_distribution
  top_help/hurt_query_exemplars
  compression_ratio_p25/p50/p75
  query_distribution_drift
  proposed_edit_ids (populated later by CrystalEditProposer)

Production wiring: the engine reads from QueryLog via MetadataStore.
For offline replay (Path M), we pass events directly via analyze_from_events().
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from ..models import Crystal, CrystalDiagnostic, QueryLog

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore


@dataclass
class CrystalEvent:
    """A single crystal-use event for diagnostic purposes.

    Production events come from QueryLog + shadow evaluator. Offline replay
    events come from rich-sweep-v1 reconstruction. Same shape either way.
    """
    crystal_id: str
    outcome: str                           # "helped" | "hurt" | "both_ok" | "both_fail"
    case_idx: Optional[int] = None
    query_text: str = ""
    response_text: str = ""
    compression_ratio: float = 1.0
    baseline_failure_mode: Optional[str] = None  # if hurt, what kind?
    chose_score: float = 0.0
    bias_norm: float = 0.0

    # Phase 1.2 (April 2026): routing-decision telemetry on the source
    # QueryLog row. Both fields were added together in migration 0010
    # and are populated by _query_log_to_event when the routing fields
    # are present.
    #
    # is_top1: whether THIS crystal was the routing top-1 for THIS query.
    # When True, this event contributes to the crystal's coverage and
    # margin signals. When False, the event still exists (matched_facts
    # included the crystal) but the routing data was about a different
    # crystal.
    #
    # margin: top1_score - top2_score for this query, BUT only attributed
    # to the top-1 crystal. None when the crystal was not top-1, or when
    # the source QueryLog row pre-dates the migration. Bricken et al.
    # 2023 (App. B.3): margin-based eviction outperforms raw-activity
    # eviction in SDM-shaped systems by a meaningful gap. We capture it
    # here so BankStatistics can aggregate it.
    is_top1: bool = False
    margin: Optional[float] = None


class DiagnosticEngine:
    """Analyzes per-crystal events and emits CrystalDiagnostic records.

    The engine is DETERMINISTIC for a given event stream — two replays with
    the same input produce the same diagnostic. This matters because
    diagnostics drive CrystalEdit proposals that modify the bank.
    """

    # How many exemplars to keep per direction
    EXEMPLAR_COUNT = 5

    # Events below this count → skip; not enough signal
    MIN_EVENTS_FOR_DIAGNOSTIC = 3

    def __init__(self, store: Optional["MetadataStore"] = None) -> None:
        """Optionally takes a MetadataStore for the production async path.
        The offline/test path uses analyze_from_events() which doesn't need it.
        """
        self._store = store

    # -----------------------------------------------------------------
    # Main entry point for production path (reads from store)
    # -----------------------------------------------------------------

    async def analyze(
        self, crystal: Crystal, window_hours: int = 168
    ) -> CrystalDiagnostic:
        """Production path: pull QueryLog rows from store, convert to events,
        delegate to analyze_from_events(). Does NOT persist the result — that
        is the caller's responsibility (see scripts/run_diagnostic_loop.py).

        Returns a diagnostic with no exemplars/failure-mode data if the crystal
        has no touched QueryLogs in the window. Callers should still persist
        the (mostly empty) diagnostic so we can track "no activity" over time.
        """
        if self._store is None:
            raise RuntimeError(
                "DiagnosticEngine was constructed without a MetadataStore. "
                "Call analyze_from_events() directly for offline use, or "
                "pass store=... when constructing for production."
            )

        logs = await self._store.list_query_logs_for_crystal(
            crystal_id=crystal.id,
            window_hours=window_hours,
        )
        events = [_query_log_to_event(log, crystal.id) for log in logs]
        return self.analyze_from_events(crystal, events)

    # -----------------------------------------------------------------
    # Offline / test entry point — takes events directly
    # -----------------------------------------------------------------

    def analyze_from_events(
        self,
        crystal: Crystal,
        events: list[CrystalEvent],
    ) -> CrystalDiagnostic:
        """Build a CrystalDiagnostic from a list of events.

        Pure function. Safe to call from scripts, tests, or the async
        production path once it's wired.
        """
        diag_id = f"diag_{crystal.id}_{uuid.uuid4().hex[:8]}"

        if len(events) < self.MIN_EVENTS_FOR_DIAGNOSTIC:
            # Not enough data — emit minimal diagnostic with a flag
            return CrystalDiagnostic(
                id=diag_id,
                crystal_id=crystal.id,
                observed_at=datetime.now(timezone.utc),
                failure_mode_distribution={},
                top_help_query_exemplars=[],
                top_hurt_query_exemplars=[],
                compression_ratio_p25=None,
                compression_ratio_p50=None,
                compression_ratio_p75=None,
                query_distribution_drift=None,
                proposed_edit_ids=[],
            )

        # Partition events by outcome
        helped = [e for e in events if e.outcome == "helped"]
        hurt = [e for e in events if e.outcome == "hurt"]

        # ---- Failure mode distribution -----------------------------------
        # Over HURT events only. If baseline_failure_mode missing, classify as "unclassified".
        failure_dist = self._failure_mode_distribution(hurt)

        # ---- Exemplars ---------------------------------------------------
        # Rank by chose_score descending — the crystal "was most confident" on these
        help_examples = sorted(helped, key=lambda e: e.chose_score, reverse=True)
        hurt_examples = sorted(hurt, key=lambda e: e.chose_score, reverse=True)

        help_exemplars = [self._format_exemplar(e) for e in help_examples[: self.EXEMPLAR_COUNT]]
        hurt_exemplars = [self._format_exemplar(e) for e in hurt_examples[: self.EXEMPLAR_COUNT]]

        # ---- Compression ratio distribution ------------------------------
        p25, p50, p75 = self._compression_quantiles(events)

        # ---- Query distribution drift (placeholder — see TODO below) ----
        drift = self._query_distribution_drift(crystal, events)

        return CrystalDiagnostic(
            id=diag_id,
            crystal_id=crystal.id,
            observed_at=datetime.now(timezone.utc),
            failure_mode_distribution=failure_dist,
            top_help_query_exemplars=help_exemplars,
            top_hurt_query_exemplars=hurt_exemplars,
            compression_ratio_p25=p25,
            compression_ratio_p50=p50,
            compression_ratio_p75=p75,
            query_distribution_drift=drift,
            proposed_edit_ids=[],  # populated later by CrystalEditProposer
        )

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _failure_mode_distribution(hurt: list[CrystalEvent]) -> dict[str, float]:
        """Normalized distribution of failure modes across hurt events.

        If no hurt events, returns empty dict. If modes are all None,
        returns {"unclassified": 1.0}.
        """
        if not hurt:
            return {}
        counts: dict[str, int] = {}
        for e in hurt:
            mode = e.baseline_failure_mode or "unclassified"
            counts[mode] = counts.get(mode, 0) + 1
        total = sum(counts.values())
        return {mode: n / total for mode, n in counts.items()}

    @staticmethod
    def _compression_quantiles(
        events: list[CrystalEvent],
    ) -> tuple[Optional[float], Optional[float], Optional[float]]:
        """p25, p50, p75 of compression ratios. None if <4 events."""
        ratios = sorted(e.compression_ratio for e in events)
        if len(ratios) < 4:
            return None, None, None
        n = len(ratios)
        p25 = ratios[n // 4]
        p50 = ratios[n // 2]
        p75 = ratios[(3 * n) // 4]
        return p25, p50, p75

    @staticmethod
    def _query_distribution_drift(
        crystal: Crystal,
        events: list[CrystalEvent],
    ) -> Optional[float]:
        """How far has the live query distribution drifted from training?

        Measure: average chose_score. When the crystal was well-matched
        to training, chose_score is high. Low average score → queries
        routed here by default, not because they fit.

        TODO: replace with proper KL divergence of keyword distributions
        once we have more than one window of telemetry. This is a proxy.
        """
        scores = [e.chose_score for e in events if e.chose_score > 0]
        if len(scores) < 3:
            return None
        # Drift = 1 - mean similarity. Higher = worse match.
        return 1.0 - statistics.mean(scores)

    @staticmethod
    def _format_exemplar(event: CrystalEvent) -> str:
        """Short human-readable string for a diagnostic exemplar."""
        # Prefer query_text for readability, fall back to truncated response
        text = event.query_text or event.response_text or f"case {event.case_idx}"
        if len(text) > 200:
            text = text[:200] + "..."
        # Strip newlines so the exemplar fits on one line in reports
        return text.replace("\n", " ").strip()


# ---------------------------------------------------------------------------
# QueryLog → CrystalEvent conversion (production path)
# ---------------------------------------------------------------------------

def _query_log_to_event(log: QueryLog, crystal_id: str) -> CrystalEvent:
    """Convert a stored QueryLog row into the event shape the engine uses.

    Outcome labeling uses the shadow evaluation if it ran:
      - shadow_delta > 0.01  → "helped"  (injected differed enough to matter
                                            in the positive direction per
                                            the configured metric)
      - shadow_delta < -0.01 → "hurt"    (injected underperformed baseline)
      - |shadow_delta| <= 0.01 → "both_ok"  (indistinguishable)
      - shadow_ran False OR  → "both_ok"  (no signal — conservative default;
        shadow_delta None                  most traffic lands here because
                                            sampling is typically 5-10%)

    The 0.01 dead-band is chosen deliberately: tiny deltas (e.g. the
    response differs in trailing whitespace) are noise, not signal.
    Widen this if the configured metric produces a lot of near-zero
    numeric wobble.

    chose_score sourcing (Phase 1.2, April 2026):
        Prefer the actual cosine score that drove the routing decision.
        Three cases:
          - This crystal WAS top-1 (log.routed_crystal_id == crystal_id):
            the score is log.top1_score.
          - This crystal was likely top-2 in the same window: we can't
            confirm without an explicit second-place crystal id, so we
            fall back to log.top2_score as a best-effort signal. False
            positives here only affect the "distribution drift" proxy,
            which is itself documented as a placeholder.
          - Pre-Phase-1.2 rows have NULL routing fields: chose_score
            falls through to 0.0, matching legacy behavior. Diagnostics
            on those rows produce no drift signal, same as before.

    Limitations: baseline_failure_mode isn't in QueryLog yet; hurt-mode
    classification requires comparing actual texts, which we could add
    later by re-reading the baseline from an audit store. Today,
    diagnostics that depend on failure-mode distribution will show
    everything as "unclassified" for live hurt events.
    """
    if log.shadow_ran and log.shadow_delta is not None:
        if log.shadow_delta > 0.01:
            outcome = "helped"
        elif log.shadow_delta < -0.01:
            outcome = "hurt"
        else:
            outcome = "both_ok"
    else:
        outcome = "both_ok"

    # Pick the right score for this crystal in this query. See docstring.
    is_top1 = (
        log.routed_crystal_id is not None
        and log.routed_crystal_id == crystal_id
    )
    if is_top1 and log.top1_score is not None:
        chose_score = float(log.top1_score)
    elif log.top2_score is not None:
        # Best-effort: this crystal probably wasn't top-1 (else we'd have
        # taken the branch above) but appeared in matched_facts, which
        # makes top-2 a reasonable approximation. Pre-1.2 rows fall
        # through to 0.0 below.
        chose_score = float(log.top2_score)
    else:
        chose_score = 0.0

    # Margin attribution: only meaningful for the top-1 crystal of this
    # query. Other crystals in matched_facts have no claim to the margin.
    margin: Optional[float] = None
    if (
        is_top1
        and log.top1_score is not None
        and log.top2_score is not None
    ):
        margin = float(log.top1_score) - float(log.top2_score)

    return CrystalEvent(
        crystal_id=crystal_id,
        outcome=outcome,
        query_text=log.query_text,
        response_text=log.response_text or "",
        compression_ratio=1.0,  # not measured in live path v0
        baseline_failure_mode=None,
        chose_score=chose_score,
        bias_norm=0.0,
        is_top1=is_top1,
        margin=margin,
    )
