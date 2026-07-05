"""Diagnostic orchestration loop.

Single entry point: `run_once(store, customer_id, window_hours)`. Does the
full learning-loop pass:

    1. Fetch crystals from the store
    2. For each, pull recent QueryLog rows and build a CrystalDiagnostic
    3. Persist diagnostics
    4. Compute bank-wide statistics
    5. Run the proposer to generate CrystalEdits
    6. Persist proposed edits

Meant to run on a schedule (cron every N hours) or on-demand. Lives in
the package rather than in scripts/ so it's importable by tests, and so
scripts/run_diagnostic_loop.py stays a thin CLI wrapper.
"""
from __future__ import annotations

from typing import Optional

from ..infrastructure.metadata_store import MetadataStore
from ..models import Crystal
from .diagnostic_engine import CrystalEvent, DiagnosticEngine, _query_log_to_event
from .edit_proposer import BankStatistics, CrystalEditProposer


async def run_once(
    store: MetadataStore,
    customer_id: Optional[str] = None,
    window_hours: int = 168,
) -> dict[str, int]:
    """Run the full diagnostic + proposer loop once.

    Returns counts for logging: crystals analyzed, diagnostics written,
    proposals emitted.
    """
    # Fetch crystals — by customer if scoped, else everything
    if customer_id:
        crystals: list[Crystal] = await store.list_crystals_for_customer(customer_id)
    else:
        crystals = await store.list_all_crystals()

    if not crystals:
        return {"crystals": 0, "diagnostics": 0, "proposals": 0}

    engine = DiagnosticEngine(store=store)

    # Step 1: analyze each crystal.
    # We also keep the events alongside so bank statistics can reuse them
    # without re-reading from the store.
    diagnostics: dict[str, "object"] = {}
    events_by_id: dict[str, list[CrystalEvent]] = {}

    for crystal in crystals:
        logs = await store.list_query_logs_for_crystal(
            crystal_id=crystal.id,
            window_hours=window_hours,
        )
        events = [_query_log_to_event(log, crystal.id) for log in logs]
        events_by_id[crystal.id] = events
        # analyze_from_events is a pure function — we use it directly so we
        # don't double-read the DB.
        diagnostics[crystal.id] = engine.analyze_from_events(crystal, events)

    # Step 2: persist diagnostics
    n_diag = 0
    for diag in diagnostics.values():
        await store.write_diagnostic(diag)
        n_diag += 1

    # Step 3: compute bank stats and run the proposer
    bank_stats = BankStatistics.compute(crystals, diagnostics, events_by_id)
    proposer = CrystalEditProposer(bank_stats=bank_stats)

    n_proposals = 0
    for crystal in crystals:
        for edit in proposer.propose_sync(crystal, diagnostics[crystal.id]):
            await store.write_edit(edit)
            n_proposals += 1

    return {
        "crystals": len(crystals),
        "diagnostics": n_diag,
        "proposals": n_proposals,
    }
