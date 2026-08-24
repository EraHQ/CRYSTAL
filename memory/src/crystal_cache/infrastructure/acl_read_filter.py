"""ACL read-filter for enumeration paths — Phase 1.4 gate 2.

The vector backends filter SEARCH results per-candidate inside their own
`search_facts`/`search_routing` implementations (fact_vector_store.py,
vector_store.py, qdrant_vector_index.py, sqlite_vec_index.py — duplicated
there deliberately, per the parity-bar note). The ENUMERATION paths have
no vector search to hook: `NavigationRouter.search` walks
`list_all_facts_for_customer` and `key_scan` walks
`list_facts_by_key_prefix`, both returning raw Fact rows whose keys and
content previews would otherwise leak a teammate's 0o600 crystals into
the overview/enumeration output (read-path rows #13/#14 in the sharing
model map — the widest reads in the system).

`readable_facts` is the one shared filter for those paths. Same shape as
the backends' filter: group memberships fetched ONCE so P3 group grants
resolve instead of fail-closing, then a per-crystal verdict cache over
`can_read`. `operator=None` returns the input unchanged — the ratified
filter-never-replaces contract (Q2=A): the system lane (scans, workers,
cognition) stays unfiltered, and no extra queries run.

A store-level SQL predicate (WHERE-clause equivalent) remains the
sharing-map's b1 follow-up; this filter is in-memory over rows a
tenancy-scoped query already fetched, so it narrows visibility without
changing pagination or query shape.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

from .permissions import can_read

if TYPE_CHECKING:
    from ..models import Fact, Operator
    from .metadata_store import MetadataStore


async def readable_facts(
    store: "MetadataStore",
    operator: "Optional[Operator]",
    facts: "Sequence[Fact]",
) -> "list[Fact]":
    """Keep only the facts whose crystal `operator` may read.

    `operator=None` (the system lane) returns the input as a list,
    unchanged and query-free — Q2=A. Otherwise: one
    `list_group_ids_for_operator` fetch, then lazy per-crystal
    crystal+ACL fetches with a verdict cache, exactly mirroring the
    vector backends' filter. A missing crystal row fails closed.
    """
    if operator is None:
        return list(facts)

    group_ids = await store.list_group_ids_for_operator(operator.id)
    verdicts: dict[str, bool] = {}
    allowed: "list[Fact]" = []
    for fact in facts:
        crystal_id = fact.crystal_id
        verdict = verdicts.get(crystal_id)
        if verdict is None:
            crystal = await store.get_crystal(crystal_id)
            if crystal is None:
                verdict = False  # fail closed on a dangling reference
            else:
                acls = await store.list_acls_for_crystal(crystal_id)
                verdict = can_read(crystal, operator, acls, group_ids)
            verdicts[crystal_id] = verdict
        if verdict:
            allowed.append(fact)
    return allowed
