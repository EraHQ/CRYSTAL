"""scan/ — proactive bank-scanning generators (Never-Idle Convergence).

The convergence half of the accommodation thesis: generators that DISCOVER
work by scanning the bank, as opposed to the cognition engine which CONSUMES
already-identified work (gaps/tasks). Each is surfacing-only and budget-
bounded. Generators:
  * contradiction.scan_for_contradictions — two facts that can't both be true
    → knowledge_conflict (detector='contradiction_scan').
  * dedup.scan_for_duplicates — two facts that say the same thing →
    knowledge_conflict (detector='dedup_scan'); reuses the contradiction
    candidate enumeration + pair_key, so the two share one keyspace.
  * gap_discovery.discover_gaps — a Subject's facts leave an important
    question unanswered → knowledge_gap (source='gap_discovery').
  * tier_promotion.run_tier_promotion_scan — the cheapest member (no model
    calls): quality tiers that MOVE, promoted on grounded citations + age +
    zero open conflicts, demoted on an open conflict.
  * topic_seeding.run_topic_seeding — research seeds without model calls:
    thin crystals + the operator topic list write knowledge_gaps the
    Phase-2 fill sweep consumes.

Staleness is intentionally NOT here yet: it needs a definition of "stale"
(a real "the source changed under us" signal — the source-watcher backlog
item), not an age heuristic. See docs/NEVER_IDLE_CONVERGENCE.md.
"""
from __future__ import annotations

from .contradiction import ScanResult, scan_for_contradictions
from .dedup import DedupScanResult, scan_for_duplicates
from .gap_discovery import GapScanResult, discover_gaps
from .tier_promotion import TierPromotionResult, run_tier_promotion_scan
from .topic_seeding import TopicSeedingResult, run_topic_seeding

__all__ = [
    "ScanResult",
    "scan_for_contradictions",
    "DedupScanResult",
    "scan_for_duplicates",
    "GapScanResult",
    "discover_gaps",
    "TierPromotionResult",
    "run_tier_promotion_scan",
    "TopicSeedingResult",
    "run_topic_seeding",
]
