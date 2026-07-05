"""Maintenance layer — §7 of BUILD_PROPOSAL.md.

Background services that keep the bank healthy:
  - ConsolidationService: merge/tighten passes over a customer's bank.
  - promotion_service (imported directly by its consumers): F3's
    scope-promotion engine — the operator→team→general rung, curation
    gated. Distinct from QUALITY-TIER promotion, which lives in
    scan/tier_promotion.py alongside the other idle scans.

History note (launch-prep purge, 2026-07-02): v1's scaffolded
DecayProcessor, GraphUpdater, CrystalSpawner, CrystalQuarantine, and
CrystalRebuildWorker were NotImplementedError stubs and were removed.
Decay policy and co-query edge population are tracked in
docs/BACKLOG.md §13 (the crystal_edges schema is kept as the landing
zone); spawn/quarantine lifecycle was superseded by the crystallizer's
neutral-tier births + the tier-promotion scan.
"""
from .consolidation_service import ConsolidationService, ConsolidationResult

__all__ = [
    "ConsolidationService",
    "ConsolidationResult",
]
