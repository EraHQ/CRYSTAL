"""Gap disposition — who can close this gap (S4, 2026-07-08).

Redesign P3 (docs/GAP_ENGINE_AND_LEARN_REDESIGN.md): cheapest capable
actor first, human LAST. Capability-aware via the tool registry's
availability predicates (2026-07-07): if the agent can reach the web,
a demand-driven gap defaults to researchable; only when it can't does
it fall to needs_document. 'workable' (agent closes it by DOING —
running software, trying the scenario) is accepted when the caller —
typically the push_gap model — names it explicitly; v1 has no
auto-detector for action-shaped gaps.
"""
from __future__ import annotations

from typing import Optional

VALID_DISPOSITIONS = ("researchable", "workable", "needs_document")


def classify_gap_disposition(explicit: Optional[str] = None) -> str:
    """Classify at creation time. `explicit` (e.g. from the push_gap
    tool) wins when valid; otherwise capability decides."""
    if explicit in VALID_DISPOSITIONS:
        return explicit
    try:
        from ..agent.tool_registry import get_registry, import_all_tools

        import_all_tools()
        names = {t.name for t in get_registry().list_for_context("agent")}
        if "web_search" in names or "web_fetch" in names:
            return "researchable"
    except Exception:  # noqa: BLE001 — classification must never break creation
        pass
    return "needs_document"
