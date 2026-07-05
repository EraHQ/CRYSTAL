"""ChainResolver — extends a crystal's cleanup codebook with chained crystals' Facts.

Phase 3 (April 2026, BIND_STORAGE_REBUILD.md §3.3) lands the chaining
primitive on top of the bind-storage read path. Single-crystal recall
(Phase 2's `recall_from_crystal`) walks one crystal's codebook to find
the matched pair. Chain-extended recall walks the source crystal's
codebook PLUS the codebooks of any chained crystals that grant
read_codebook access to the source's customer.

The resolver answers ONE question:

    "Given a source crystal and the customer making the query, which
     additional Fact rows should be unioned into the cleanup codebook?"

It does NOT do the unbind, the cosine search, or the threshold gate.
Those live in `recall_from_crystal`. The resolver just produces the
extra Facts to splice in.

ACL CHECK
---------
A chain edge from source to target produces extra Facts only if:
  1. A chain row exists with source_crystal_id == source.id and
     target_crystal_id == target.id.
  2. The target's ACLs grant `read_codebook` to the source's customer
     (by default rule, the owning customer of a customer-scope crystal
     has implicit read on its own crystals; see _principal_can_read_codebook
     for the resolution order).

ACL violations are SILENT — the chained Facts are absent from the
cleanup codebook, no error is raised. The user gets a recall result
based only on the source's own codebook (or None if cleanup doesn't
clear threshold). Per spec §3.4 "ACL violations are silent" — leaks
about which other crystals exist would defeat the isolation guarantee.

DIRECTIONALITY
--------------
A chain row carries `direction = 'source_uses_target'` only. The
bidirectional case is represented as TWO rows — (A→B) and (B→A) —
rather than one row with `direction='bidirectional'`. This lets the
resolver forward-walk only and keeps the read path simple. The
authoring API still accepts the `bidirectional` keyword and writes
both rows transparently; see `MetadataStore.add_chain` Phase 3 audit
fix #7 (April 2026).

Why two rows instead of one + reverse-walk: a single bidirectional
row forces every recall to query both "chains FROM source" AND
"chains TO source where direction=bidirectional" and union them.
Two rows means the resolver only ever asks one question ("chains
FROM source"), which is one DB roundtrip instead of two and one set
of ACL checks instead of two.

Legacy single-row bidirectional rows (if any exist from pre-fix #7
DBs) are normalized when the edge is next written via `add_chain`.
The resolver treats any row that's neither 'source_uses_target' nor
recognized as legacy bidirectional as a forward-walked edge for
robustness; see `_walk_forward` for the exact handling.

SELF-LOOPS
----------
Chain rows with source == target are rejected at write time
(`MetadataStore.add_chain` raises ValueError) and double-checked here.
A crystal already includes its own facts in cleanup; chaining to
itself adds nothing and would just waste a DB lookup.

DEPTH
-----
The resolver walks ONE hop only. If A chains to B and B chains to C,
recall on A pulls B's facts but NOT C's. Multi-hop chain transitivity
is out of scope for Phase 3 — it would require either an explicit
graph traversal with cycle detection or DSL-level "transitive chain"
declarations. Not in scope.

FAN-OUT (open scale issue, not a Phase 3 concern)
-------------------------------------------------
There is NO cap on how many chained targets a single recall walks, and
no cap on how many Facts each target contributes. A crystal chained to
N targets, each with ~50 facts, expands the cleanup codebook to ~50N
entries. Cleanup is one cosine pass over the union, so cost is linear
in the codebook size.

We deliberately do not cap fan-out in Phase 3. Two reasons:
  1. Spec §3.3 didn't ask for one. Adding a cap without empirical
     justification preempts authoring patterns we haven't validated.
  2. Phase 6.3's per-crystal `cleanup_threshold` calibrator is the
     right tool for managing extension noise. If a chained target's
     facts contribute mostly crosstalk, the threshold rises and they
     stop crowding the matched-pair cosine, regardless of how many
     targets are in the chain.

When this becomes a real concern (signal: cleanup latency on
chain-extended recalls drifts above the ~10ms research budget at
production bank scale), the right fix is at Phase 6.3's calibrator,
not a hardcoded cap here.

DEDUPLICATION
-------------
The resolver returns Facts deduplicated by `Fact.id`. If two chain
edges land at the same target, the target's facts appear once in the
extension list. The source's own facts are NOT in the resolver's
output — caller (recall_from_crystal) loads those separately and
unions.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ..models import CrystalAcl, Fact

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore


logger = structlog.get_logger(__name__)


class ChainResolver:
    """Resolve chained Facts for cleanup-codebook extension.

    Stateless except for the MetadataStore reference. Constructed once
    per pipeline (or per request, doesn't matter — the only state is
    the store handle which is process-shared).

    Usage from `recall_from_crystal`:

        resolver = ChainResolver(store=store)
        extra_facts = await resolver.resolve_extra_facts(
            source_crystal_id=top1_crystal.id,
            requesting_customer_id=customer.id,
        )
        codebook_facts = own_facts + extra_facts

    The resolver does NOT enforce that source_crystal's customer ==
    requesting_customer_id. The caller (pipeline.py) already enforces
    that the requesting customer can route into source_crystal in the
    first place via the four-way classifier's per-customer scoping
    (VectorStore.search filters by customer_id). Defense-in-depth
    duplicate checks here would be redundant.
    """

    def __init__(self, store: "MetadataStore") -> None:
        self._store = store

    async def resolve_extra_facts(
        self,
        *,
        source_crystal_id: str,
        requesting_customer_id: str,
    ) -> list[Fact]:
        """Return Facts to splice into source_crystal's cleanup codebook.

        Args:
            source_crystal_id: the crystal whose recall is in progress.
                Outgoing chains from this crystal are walked; the
                bidirectional case is naturally covered because
                bidirectional edges are stored as two rows (one per
                direction) per Phase 3 audit fix #7.
            requesting_customer_id: the customer whose query is in
                flight. Used to check `read_codebook` ACL grants on
                each chain target.

        Returns:
            A flat list of Fact rows from chained crystals that grant
            read_codebook to `requesting_customer_id`. Deduplicated by
            Fact.id (one chain target's facts appear once even if
            multiple chains lead there). Empty list if no chains
            exist, no chains are ACL-permitted, or all chained
            crystals have empty codebooks.

            The order is target-by-target as discovered (not sorted) —
            cleanup at the call site does an argmax over cosines so
            order doesn't matter.
        """
        # Step 1: collect candidate target crystal ids via forward walk.
        #
        # Under the two-row representation (Phase 3 audit fix #7), a
        # bidirectional A↔B is stored as (A→B) AND (B→A). When recall
        # runs on A, list_chains_from_source(A) returns the (A→B) row
        # so B is a forward target. When recall runs on B, the same
        # query returns the (B→A) row so A is a forward target. The
        # bidirectionality emerges from the data, not from a separate
        # reverse-walk.
        #
        # Legacy bidirectional rows (single-row, pre-fix-#7) are
        # tolerated: a row with direction='bidirectional' on the
        # forward side still produces the forward target, so one
        # direction works. The reverse direction won't fire until the
        # row is rewritten via add_chain. Acceptable for Phase 3 —
        # the registry was just seeded; no production rows in this
        # state.
        forward_targets: set[str] = set()
        forward_chains = await self._store.list_chains_from_source(
            source_crystal_id
        )
        for chain in forward_chains:
            if chain.target_crystal_id == source_crystal_id:
                # Self-loop. Should be rejected at write time (add_chain
                # raises ValueError on self-loops); double-check here as
                # a defensive guard against direct DB writes that
                # bypass the validation layer.
                logger.warning(
                    "chain_resolver.skip_self_loop",
                    source_crystal_id=source_crystal_id,
                    note=(
                        "Chain row with source == target found in DB. "
                        "Should have been rejected at add_chain time. "
                        "Skipping; chaining to self adds nothing."
                    ),
                )
                continue
            forward_targets.add(chain.target_crystal_id)

        if not forward_targets:
            return []

        # Step 2: ACL gate. For each candidate target, check whether
        # the requesting customer has read_codebook on it. The default
        # rule is that customer-scope crystals are readable by their
        # owning customer (no explicit ACL row needed); explicit
        # crystal_acls rows extend access to additional principals.
        #
        # We collect the targets that pass the gate; targets that fail
        # are silently dropped (per spec §3.4).
        permitted_targets: list[str] = []
        for target_id in forward_targets:
            target_crystal = await self._store.get_crystal(target_id)
            if target_crystal is None:
                # Dangling chain row. Should be rare — chain rows have
                # FKs to crystals.id — but possible during eventual-
                # consistency windows. Log and skip.
                logger.warning(
                    "chain_resolver.target_not_found",
                    source_crystal_id=source_crystal_id,
                    target_crystal_id=target_id,
                )
                continue

            permitted = await self._principal_can_read_codebook(
                target_crystal_id=target_id,
                target_customer_id=target_crystal.customer_id,
                requesting_customer_id=requesting_customer_id,
            )
            if permitted:
                permitted_targets.append(target_id)
            else:
                logger.debug(
                    "chain_resolver.acl_denied",
                    source_crystal_id=source_crystal_id,
                    target_crystal_id=target_id,
                    requesting_customer_id=requesting_customer_id,
                )

        if not permitted_targets:
            return []

        # Step 3: load Facts from permitted targets, dedupe by Fact.id.
        # Order is by target-discovery order; within a target, by
        # Fact.created_at ascending (matches list_facts_for_crystal's
        # default).
        seen_fact_ids: set[str] = set()
        extra: list[Fact] = []
        for target_id in permitted_targets:
            target_facts = await self._store.list_facts_for_crystal(
                target_id
            )
            for fact in target_facts:
                if fact.id in seen_fact_ids:
                    continue
                seen_fact_ids.add(fact.id)
                extra.append(fact)

        logger.debug(
            "chain_resolver.resolved",
            source_crystal_id=source_crystal_id,
            requesting_customer_id=requesting_customer_id,
            forward_target_count=len(forward_targets),
            permitted_target_count=len(permitted_targets),
            extra_fact_count=len(extra),
        )
        return extra

    async def _principal_can_read_codebook(
        self,
        *,
        target_crystal_id: str,
        target_customer_id: str,
        requesting_customer_id: str,
    ) -> bool:
        """Resolution order for the 'can principal read this codebook?' check.

        1. If the requesting customer IS the target's owning customer:
           always permitted (a customer can always extend their own
           cleanup with their own crystals). No ACL row needed.

        2. Otherwise, check `crystal_acls` for an explicit grant of
           'read_codebook' OR 'read' to one of:
             - (customer, requesting_customer_id) — tenant-scoped grant
             - (global, 'world')                  — public grant
           A 'read' grant implies 'read_codebook' (read is the strict
           superset; if you can route in and consume facts, you can
           certainly extend a codebook with them).

        3. Otherwise, denied.

        Returns True iff permitted.
        """
        if target_customer_id == requesting_customer_id:
            return True

        acls = await self._store.list_acls_for_crystal(target_crystal_id)
        for acl in acls:
            if self._acl_grants_codebook(acl, requesting_customer_id):
                return True
        return False

    @staticmethod
    def _acl_grants_codebook(
        acl: CrystalAcl, requesting_customer_id: str
    ) -> bool:
        """True iff this ACL row grants codebook access to the requester.

        - principal_type=customer + principal_id=requesting_customer_id:
          tenant-scoped grant for this specific customer.
        - principal_type=global + principal_id='world': public grant.
        - principal_type=crystal_chain: NOT a customer grant. These
          rows govern chain-target access from a specific source
          crystal, not from a customer's queries directly. The
          chain-target's customer would have authored such a row
          when setting up a private inter-crystal pipeline; the
          customer-scope check happens via the customer-typed grants.
          Returning False here is correct: a crystal_chain principal
          isn't a customer.

        Both 'read' and 'read_codebook' grants permit codebook access.
        'read' is the stricter grant (route + consume); if you have
        it, you trivially have read_codebook too.
        """
        if acl.grant not in ("read", "read_codebook"):
            return False
        if acl.principal_type == "customer" and acl.principal_id == requesting_customer_id:
            return True
        if acl.principal_type == "global" and acl.principal_id == "world":
            return True
        return False
