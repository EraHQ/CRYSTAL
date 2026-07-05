"""Crystal edit proposer — §4 of BUILD_PROPOSAL.md.

Generates CrystalEdit proposals from a diagnostic + bank-wide statistics.

DESIGN PRINCIPLE: data-driven, dual-gated.

The proposer computes per-crystal scores using signals derived from the
bank's own distribution. Crystals get proposals only when they fail BOTH:

  1. A MAD-based z-score gate (direction: "is this unusual?")
  2. A rank-based percentile gate (magnitude: "is this really extreme?")

Both gates matter. MAD alone produces false positives on tight distributions
(a compression of 0.97 vs median 1.00 with MAD 0.01 registers as 1.7 MADs —
statistically "unusual" but practically meaningless). Rank alone loses the
directional information and can't distinguish "tail by luck" from "tail by
pathology". Combined, they catch crystals that are BOTH clearly in the bad
direction AND clearly at the distribution extreme.

Four-plus-two signals drive the scores:

  1. hurt_score    — hurt_rate relative to bank median (z-score)
  2. size_score    — how far above the bank's median fact_count
  3. generic_score — how low the crystal's keyword DISTINCTIVENESS is
                     (words unique to this crystal vs shared with peers)
  4. compression_score — how severely the crystal's p50 ratio is below median
  5. coverage_score — (Phase 1.2.1, SDM-paper) inverse z-score of
                     top-1 hit rate. High coverage_score means the crystal
                     RARELY won routing in the analysis window — dead
                     capacity. Off-manifold neuron analog from
                     Bricken et al. 2023 (App. A.4).
  6. margin_score  — (Phase 1.2.1, SDM-paper) inverse z-score of
                     mean (top1 - top2) margin when this crystal was
                     top-1. High margin_score means the crystal is
                     OFTEN tied with neighbors — redundant. Bricken
                     et al. 2023 (App. B.3): margin-based eviction
                     outperforms raw-activity eviction in SDM systems.

A split proposal needs: hurt_score > 0 AND (big OR generic OR low_coverage
OR low_margin) via dual gate. A rebuild proposal needs: hurt_score > 0
AND compression via dual gate.

No hardcoded absolute thresholds (fact_count >= 200, compression < 0.5).
Everything emerges from the bank's distribution.
"""
from __future__ import annotations

import statistics
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Sequence

from ..models import Crystal, CrystalDiagnostic, CrystalEdit


# -----------------------------------------------------------------------------
# Bank-wide statistics
# -----------------------------------------------------------------------------

@dataclass
class CrystalStats:
    """Per-crystal telemetry + structural summary."""
    crystal_id: str
    n_events: int
    n_helped: int
    n_hurt: int
    compression_p50: Optional[float]
    fact_count: int
    # Distinctiveness: 1 - (mean document-frequency of fingerprint tokens)
    # High means the crystal's keywords are unique to it; low means
    # shared with many peers.
    keyword_distinctiveness: float

    # Phase 1.2.1: top-1 routing telemetry, populated when CrystalEvent
    # rows carry routing data (post-migration 0010 traffic). Pre-1.2
    # rows leave both at zero/None.
    #
    # n_top1_events: how many times this crystal was the routing top-1
    # in the analysis window. Used for coverage_score.
    #
    # margin_mean: average (top1 - top2) margin across the events where
    # THIS crystal was top-1. None if n_top1_events == 0 — a crystal
    # that never won routing has no margin to attribute. Used for
    # margin_score.
    n_top1_events: int = 0
    margin_mean: Optional[float] = None

    @property
    def outcome_rate_denominator(self) -> int:
        return self.n_helped + self.n_hurt

    @property
    def hurt_rate(self) -> Optional[float]:
        d = self.outcome_rate_denominator
        return self.n_hurt / d if d > 0 else None

    @property
    def coverage_rate(self) -> Optional[float]:
        """Fraction of analyzed events where this crystal was top-1.

        None when there are no events at all (no signal to compute
        from). 0.0 is a meaningful value — the crystal had events
        but was never top-1 — and is treated separately from None.
        """
        if self.n_events <= 0:
            return None
        return self.n_top1_events / self.n_events


@dataclass
class BankStatistics:
    """Bank-wide distributions used to derive scores.

    All signals are RELATIVE to this bank. The same crystal, analyzed
    against a different bank, would get different scores.
    """
    per_crystal: dict[str, CrystalStats] = field(default_factory=dict)

    fact_count_median: float = 0.0
    fact_count_mad: float = 0.0
    fact_count_p90: float = 0.0       # "big" requires crystal in top decile

    compression_median: float = 1.0
    compression_mad: float = 0.0
    compression_p10: float = 1.0      # "compressing" requires bottom decile

    hurt_rate_median: float = 0.0
    hurt_rate_mad: float = 0.0
    hurt_rate_p75: float = 0.0

    distinctiveness_median: float = 0.0
    distinctiveness_mad: float = 0.0
    distinctiveness_p10: float = 0.0  # "generic" requires bottom decile

    # Phase 1.2.1 distributions for the new signals. coverage_rate is
    # the fraction of a crystal's events where it was top-1; we want
    # to flag crystals in the bottom decile (rarely won routing).
    # margin_mean is the mean top1-top2 gap when the crystal was
    # top-1; we flag the bottom decile (often-tied crystals).
    coverage_median: float = 0.0
    coverage_mad: float = 0.0
    coverage_p10: float = 0.0

    margin_median: float = 0.0
    margin_mad: float = 0.0
    margin_p10: float = 0.0

    # Sample size floor for reliable rate estimation
    MIN_EVENTS_FOR_RATE = 10

    @classmethod
    def compute(
        cls,
        crystals: Sequence[Crystal],
        diagnostics: dict[str, CrystalDiagnostic],
        events_by_crystal: dict[str, list],  # list[CrystalEvent]
    ) -> "BankStatistics":
        """Build a BankStatistics from the analyzed bank.

        Phase 1.2.1: events are expected to carry `is_top1` and `margin`
        attributes (see diagnostic_engine.CrystalEvent). When a CrystalEvent
        was built from a pre-1.2 QueryLog row, `is_top1` is False and
        `margin` is None; this function silently treats those as "no
        signal" — the corresponding crystal accumulates 0 top-1 events
        and a None margin_mean, which is exactly the legacy behavior.
        """
        stats = cls()

        # Build token document-frequency across the bank
        token_doc_count: dict[str, int] = {}
        for c in crystals:
            for tok in set(t.lower() for t in c.keyword_fingerprint):
                token_doc_count[tok] = token_doc_count.get(tok, 0) + 1
        n_crystals = max(1, len(crystals))

        fact_counts: list[int] = []
        compressions: list[float] = []
        hurt_rates: list[float] = []
        distinctivenesses: list[float] = []
        coverage_rates: list[float] = []
        margin_means: list[float] = []

        for c in crystals:
            events = events_by_crystal.get(c.id, [])
            helped = sum(1 for e in events if e.outcome == "helped")
            hurt = sum(1 for e in events if e.outcome == "hurt")

            # Phase 1.2.1 aggregations: top-1 count and per-crystal
            # margin mean. The `getattr(e, ..., default)` calls are
            # belt-and-suspenders for offline-replay events that may
            # have been constructed without the new fields — same
            # defensive shape as compression_ratio handling above.
            n_top1 = sum(1 for e in events if getattr(e, "is_top1", False))
            margins = [
                e.margin
                for e in events
                if getattr(e, "is_top1", False)
                and getattr(e, "margin", None) is not None
            ]
            margin_mean: Optional[float]
            if margins:
                margin_mean = sum(margins) / len(margins)
            else:
                margin_mean = None

            fp = [t.lower() for t in c.keyword_fingerprint]
            if fp:
                avg_inverse_uniqueness = sum(
                    token_doc_count.get(t, 0) / n_crystals for t in fp
                ) / len(fp)
                distinctiveness = 1.0 - avg_inverse_uniqueness
            else:
                distinctiveness = 0.0

            diag = diagnostics.get(c.id)
            cp50 = diag.compression_ratio_p50 if diag else None

            cs = CrystalStats(
                crystal_id=c.id,
                n_events=len(events),
                n_helped=helped,
                n_hurt=hurt,
                compression_p50=cp50,
                fact_count=c.fact_count,
                keyword_distinctiveness=distinctiveness,
                n_top1_events=n_top1,
                margin_mean=margin_mean,
            )
            stats.per_crystal[c.id] = cs

            fact_counts.append(c.fact_count)
            distinctivenesses.append(distinctiveness)
            if cp50 is not None:
                compressions.append(cp50)
            if cs.outcome_rate_denominator >= cls.MIN_EVENTS_FOR_RATE:
                hurt_rates.append(cs.hurt_rate or 0.0)
            # Coverage rate is meaningful only when the crystal had
            # enough events to estimate a top-1 fraction. Same
            # MIN_EVENTS_FOR_RATE floor as hurt_rates — with too few
            # events, a single hit-or-miss swings the rate wildly.
            cov = cs.coverage_rate
            if cov is not None and cs.n_events >= cls.MIN_EVENTS_FOR_RATE:
                coverage_rates.append(cov)
            # Margin mean exists only for crystals that won at least one
            # routing decision; no min-events floor here because the
            # margin itself is a per-decision measurement, not a rate.
            if margin_mean is not None:
                margin_means.append(margin_mean)

        stats.fact_count_median = _median(fact_counts)
        stats.fact_count_mad = _mad(fact_counts, stats.fact_count_median)
        stats.fact_count_p90 = _percentile(fact_counts, 0.90)

        if compressions:
            stats.compression_median = _median(compressions)
            stats.compression_mad = _mad(compressions, stats.compression_median)
            stats.compression_p10 = _percentile(compressions, 0.10)

        if hurt_rates:
            stats.hurt_rate_median = _median(hurt_rates)
            stats.hurt_rate_mad = _mad(hurt_rates, stats.hurt_rate_median)
            stats.hurt_rate_p75 = _percentile(hurt_rates, 0.75)

        stats.distinctiveness_median = _median(distinctivenesses)
        stats.distinctiveness_mad = _mad(distinctivenesses, stats.distinctiveness_median)
        stats.distinctiveness_p10 = _percentile(distinctivenesses, 0.10)

        if coverage_rates:
            stats.coverage_median = _median(coverage_rates)
            stats.coverage_mad = _mad(coverage_rates, stats.coverage_median)
            stats.coverage_p10 = _percentile(coverage_rates, 0.10)

        if margin_means:
            stats.margin_median = _median(margin_means)
            stats.margin_mad = _mad(margin_means, stats.margin_median)
            stats.margin_p10 = _percentile(margin_means, 0.10)

        return stats


def _median(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _mad(values: Sequence[float], median: float) -> float:
    """Median Absolute Deviation — robust scale estimate."""
    if not values:
        return 0.0
    deviations = [abs(v - median) for v in values]
    return float(statistics.median(deviations))


def _percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation percentile. Empty → 0.0."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return float(s[0])
    idx = q * (len(s) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return float(s[lo] + (s[hi] - s[lo]) * frac)


def _robust_z(value: float, median: float, mad: float) -> float:
    """Robust z-score using MAD.

    Returns how many median-absolute-deviations `value` is from `median`.
    The 1.4826 factor normalizes MAD to match standard deviation under normal
    distributions.

    Edge case: when MAD is zero (constant distribution), any deviation is
    maximally unusual — return a large sentinel value rather than 0.
    Returning 0 would hide real outliers in tight distributions.
    """
    if mad <= 1e-9:
        if abs(value - median) <= 1e-9:
            return 0.0
        return 10.0 if value > median else -10.0
    return (value - median) / (1.4826 * mad)


# -----------------------------------------------------------------------------
# Proposer
# -----------------------------------------------------------------------------

class CrystalEditProposer:
    """Generates CrystalEdit proposals using dual-gated bank-relative scoring.

    Gate 1 (direction): MAD-based z-score above threshold → "unusual"
    Gate 2 (magnitude): rank-based percentile → "extreme"

    Both gates must pass. Protects against tight-distribution false positives
    and against rank-tail noise.
    """

    MIN_EVENTS_FOR_PROPOSAL = 10

    # Thresholds — these ARE bank-relative since MAD and percentiles are
    # computed from the bank. The "1.0 MAD" number is the usual threshold
    # for "clearly unusual" in robust-statistics literature.
    STRUCTURAL_MAD_BAR = 1.0
    COMPRESSION_MAD_BAR = 1.0

    def __init__(self, bank_stats: Optional[BankStatistics] = None) -> None:
        self.bank_stats = bank_stats

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def propose(
        self,
        crystal: Crystal,
        diagnostic: CrystalDiagnostic,
    ) -> list[CrystalEdit]:
        return self.propose_sync(crystal, diagnostic)

    def propose_sync(
        self,
        crystal: Crystal,
        diagnostic: CrystalDiagnostic,
    ) -> list[CrystalEdit]:
        if self.bank_stats is None:
            return []

        stats = self.bank_stats.per_crystal.get(crystal.id)
        if stats is None or stats.outcome_rate_denominator < self.MIN_EVENTS_FOR_PROPOSAL:
            return []

        scores = self._score(stats)

        if scores["hurt_score"] <= 0:
            return []

        proposals: list[CrystalEdit] = []

        split = self._propose_split(crystal, stats, scores)
        if split is not None:
            proposals.append(split)

        rebuild = self._propose_rebuild(crystal, stats, scores)
        if rebuild is not None:
            proposals.append(rebuild)

        return proposals

    # -----------------------------------------------------------------
    # Scoring
    # -----------------------------------------------------------------

    def _score(self, stats: CrystalStats) -> dict[str, float]:
        bs = self.bank_stats
        assert bs is not None

        hurt = _robust_z(stats.hurt_rate or 0.0, bs.hurt_rate_median, bs.hurt_rate_mad)
        size = _robust_z(float(stats.fact_count), bs.fact_count_median, bs.fact_count_mad)
        generic = -_robust_z(
            stats.keyword_distinctiveness, bs.distinctiveness_median, bs.distinctiveness_mad
        )
        compression = -_robust_z(
            stats.compression_p50 if stats.compression_p50 is not None else bs.compression_median,
            bs.compression_median,
            bs.compression_mad,
        )

        # Phase 1.2.1 signals. Both inverted (negative z-score) so that
        # "low coverage" and "low margin" produce HIGH scores — the same
        # convention as generic and compression. A crystal that's never
        # top-1 (low coverage) or always tied (low margin) ranks high
        # in the redundancy direction, which is what we want to surface
        # for split proposals.
        #
        # When the bank-wide distribution is empty (no crystals had
        # enough events for the coverage/margin sample), both medians
        # and MADs are 0 and _robust_z's MAD-zero branch hits. The
        # branch returns 0 when the value matches the (zero) median —
        # which it will, when stats.coverage_rate / margin_mean are
        # also zero/None. So pre-1.2 traffic produces zero scores here
        # and the gates below silently never fire. Same behavior as
        # legacy.
        coverage = -_robust_z(
            stats.coverage_rate if stats.coverage_rate is not None else bs.coverage_median,
            bs.coverage_median,
            bs.coverage_mad,
        )
        margin = -_robust_z(
            stats.margin_mean if stats.margin_mean is not None else bs.margin_median,
            bs.margin_median,
            bs.margin_mad,
        )

        return {
            "hurt_score": hurt,
            "size_score": size,
            "generic_score": generic,
            "compression_score": compression,
            "coverage_score": coverage,
            "margin_score": margin,
        }

    # -----------------------------------------------------------------
    # SPLIT rule — dual gate (MAD + percentile)
    # -----------------------------------------------------------------

    def _propose_split(
        self,
        crystal: Crystal,
        stats: CrystalStats,
        scores: dict[str, float],
    ) -> Optional[CrystalEdit]:
        bs = self.bank_stats
        assert bs is not None

        is_big = (
            scores["size_score"] >= self.STRUCTURAL_MAD_BAR
            and crystal.fact_count >= bs.fact_count_p90
        )
        is_generic = (
            scores["generic_score"] >= self.STRUCTURAL_MAD_BAR
            and stats.keyword_distinctiveness <= bs.distinctiveness_p10
        )
        # Phase 1.2.1: low-coverage and low-margin gates. Same dual-gate
        # discipline as is_big / is_generic — MAD direction signal AND
        # percentile magnitude signal must both clear. The percentile
        # check uses the bank's bottom-decile threshold so that pathological
        # bank-wide low-coverage (e.g. an under-trafficked customer)
        # doesn't fire on every crystal at once.
        is_low_coverage = (
            scores["coverage_score"] >= self.STRUCTURAL_MAD_BAR
            and stats.coverage_rate is not None
            and stats.coverage_rate <= bs.coverage_p10
        )
        is_low_margin = (
            scores["margin_score"] >= self.STRUCTURAL_MAD_BAR
            and stats.margin_mean is not None
            and stats.margin_mean <= bs.margin_p10
        )

        if not (is_big or is_generic or is_low_coverage or is_low_margin):
            return None

        reasons = []
        if is_big:
            reasons.append(
                f"fact_count={crystal.fact_count} is {scores['size_score']:.1f} "
                f"MADs above bank median ({bs.fact_count_median:.0f}) "
                f"and in top decile (p90={bs.fact_count_p90:.0f})"
            )
        if is_generic:
            reasons.append(
                f"keyword_distinctiveness={stats.keyword_distinctiveness:.2f} "
                f"is {scores['generic_score']:.1f} MADs below bank median "
                f"({bs.distinctiveness_median:.2f}) and in bottom decile "
                f"(p10={bs.distinctiveness_p10:.2f}) — its fingerprint is "
                f"dominated by tokens shared with peers"
            )
        if is_low_coverage:
            assert stats.coverage_rate is not None  # for the type checker
            reasons.append(
                f"coverage_rate={stats.coverage_rate:.0%} (top-1 in "
                f"{stats.n_top1_events}/{stats.n_events} routed events) is "
                f"{scores['coverage_score']:.1f} MADs below bank median "
                f"({bs.coverage_median:.0%}) and in bottom decile "
                f"(p10={bs.coverage_p10:.0%}) — the crystal rarely wins "
                f"routing in this analysis window, suggesting dead capacity "
                f"or off-manifold positioning (Bricken et al. 2023, App. A.4)"
            )
        if is_low_margin:
            assert stats.margin_mean is not None  # for the type checker
            reasons.append(
                f"margin_mean={stats.margin_mean:.3f} "
                f"(top1−top2 average over {stats.n_top1_events} top-1 events) "
                f"is {scores['margin_score']:.1f} MADs below bank median "
                f"({bs.margin_median:.3f}) and in bottom decile "
                f"(p10={bs.margin_p10:.3f}) — the crystal is often tied "
                f"with a neighbor at routing time, suggesting redundancy "
                f"(Bricken et al. 2023, App. B.3)"
            )
        reasons.append(
            f"hurt_rate={stats.hurt_rate:.0%} is {scores['hurt_score']:.1f} MADs "
            f"above bank median ({bs.hurt_rate_median:.0%}); "
            f"helped={stats.n_helped}/hurt={stats.n_hurt}"
        )

        k = max(2, min(5, stats.n_hurt // 5 + 2))

        rationale = (
            "Crystal is hurting live traffic above peer baseline AND is "
            "structurally in the problem tail (MAD-unusual and at distribution extreme). "
            "Specifics: "
            + "; ".join(reasons)
            + f". Proposed: split into ~{k} sub-clusters."
        )

        return CrystalEdit(
            id=f"edit_{uuid.uuid4().hex[:12]}",
            crystal_id=crystal.id,
            edit_type="split",
            proposed_by="diagnostic_engine",
            rationale=rationale,
            affected_facts=[],
            expected_impact=(
                f"Split into {k} sub-clusters. Expect the helpful subset "
                f"to surface with lower hurt_rate."
            ),
            status="pending",
            created_at=datetime.now(timezone.utc),
        )

    # -----------------------------------------------------------------
    # REBUILD rule — dual gate (MAD + percentile)
    # -----------------------------------------------------------------

    def _propose_rebuild(
        self,
        crystal: Crystal,
        stats: CrystalStats,
        scores: dict[str, float],
    ) -> Optional[CrystalEdit]:
        if stats.compression_p50 is None:
            return None
        if scores["compression_score"] < self.COMPRESSION_MAD_BAR:
            return None

        bs = self.bank_stats
        assert bs is not None
        if stats.compression_p50 > bs.compression_p10:
            return None

        rationale = (
            f"Crystal output is severely compressed vs peers. "
            f"compression_p50={stats.compression_p50:.2f} is "
            f"{scores['compression_score']:.1f} MADs below bank median "
            f"({bs.compression_median:.2f}) and in bottom decile "
            f"(p10={bs.compression_p10:.2f}). Combined with "
            f"hurt_rate={stats.hurt_rate:.0%} (bank median "
            f"{bs.hurt_rate_median:.0%}), the crystal is truncating outputs "
            f"and harming accuracy. Research §2.3: this pattern arises from "
            f"terminal '#### N' in training answers baked into the bias. "
            f"Recommended fix: rebuild with terminal pattern stripped."
        )

        return CrystalEdit(
            id=f"edit_{uuid.uuid4().hex[:12]}",
            crystal_id=crystal.id,
            edit_type="rebuild",
            proposed_by="diagnostic_engine",
            rationale=rationale,
            affected_facts=[],
            expected_impact=(
                f"Rebuild with answer-text-stripped training. Expect "
                f"compression_p50 to rise from {stats.compression_p50:.2f} "
                f"toward the bank median ({bs.compression_median:.2f})."
            ),
            status="pending",
            created_at=datetime.now(timezone.utc),
        )
