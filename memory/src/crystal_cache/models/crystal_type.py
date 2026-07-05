"""CrystalType + CrystalAcl + CrystalChain — Phase 3 of bind-storage rebuild.

Three models supporting the typed-crystal architecture
(BIND_STORAGE_REBUILD.md §3):

  CrystalType  — registry entry. Every Crystal points at one of these
                 by string id. The type carries scope, capacity,
                 autosplit policy, and per-type threshold overrides.

  CrystalAcl   — per-crystal access grant. Two grant levels: 'read'
                 (route in + consume facts) and 'read_codebook'
                 (chain-extend cleanup codebook only). Multiple grants
                 per crystal coexist as separate rows.

  CrystalChain — directed edge between crystals. Cleanup at recall
                 time walks outgoing chains and unions chained
                 crystals' Facts into the codebook (subject to ACL).

These three pieces compose: a customer's query routes into one of
their crystals (filtered by crystal_type, ACL'd by 'read' grant);
recall extends the cleanup codebook with chained crystals' Facts
(ACL'd by 'read_codebook' on the chain target). General-tier
crystals carry a (global, world, read) grant by default; customer
crystals carry a (customer, customer_id, read) grant.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enum types (Literal, validated at the Pydantic layer)
# ---------------------------------------------------------------------------

# Crystal scope. Matches the spec's four-tier proposal:
#   - 'general'  : world-readable, authored centrally (math, poetry, domain).
#   - 'customer' : owned by one tenant, indexed under their bank.
#   - 'document' : per-document crystal, owned by the uploading customer.
#                  Phase 5 lands document ingestion that produces these.
#   - 'personal' : per-person within a customer (Phase 5+ idea, not used
#                  by Phase 3 directly but reserved in the enum so the
#                  registry can hold it from day one).
CrystalScope = Literal["general", "customer", "document", "personal"]

# What happens when add_pair_to_crystal hits the type's capacity ceiling.
#   - 'split'  : auto-spawn a sibling crystal (Phase 1.2 auto-split logic).
#   - 'refuse' : raise CrystalCapacityError; operator partitions explicitly.
AutosplitPolicy = Literal["split", "refuse"]

# Who an ACL grant applies to.
#   - 'customer'      + customer_id : tenant-scoped grant.
#   - 'global'        + 'world'     : public grant (general tier).
#   - 'crystal_chain' + crystal_id  : another crystal can borrow this
#                                     one's facts via chain. Granted at
#                                     the chain target's ACL.
PrincipalType = Literal[
    "customer", "global", "crystal_chain",
    # P3 (ratified 2026-07-02): named grants to one operator or one
    # group — sub-team / individual sharing without changing the
    # crystal's POSIX mode.
    "operator", "group",
]

# What the principal can do with this crystal.
#   - 'read'           : route INTO + consume facts.
#   - 'read_codebook'  : chain-extend cleanup codebook only. Cannot route.
AclGrant = Literal["read", "read_codebook"]

# Chain edge directionality.
#   - 'source_uses_target' : one-way; only source's recall pulls target's
#                            facts in. Default.
#   - 'bidirectional'      : both directions. Single row, not two; the
#                            resolver consults the direction column when
#                            traversing target -> sources.
ChainDirection = Literal["source_uses_target", "bidirectional"]


# ---------------------------------------------------------------------------
# CrystalType
# ---------------------------------------------------------------------------

class CrystalType(BaseModel):
    """Registry entry for a kind of crystal.

    The id namespace IS the scope marker by convention:
    'general:math', 'customer:medical_records', 'document:contracts',
    'personal:notes'. The model validates `scope` against the prefix
    of `id` for catch errors at the Pydantic boundary.

    `routing_threshold` and `cleanup_threshold` are nullable per-type
    overrides of the global settings. NULL means "use whatever the
    global is at write/recall time"; a populated value pins the
    threshold for this type. Phase 6.3's calibrator will populate
    these per-type as bank-scale validation tells us what works.
    """

    id: str
    display_name: str
    scope: CrystalScope

    # Default 50 matches the global CRYSTAL_CAPACITY_HARD_CEILING in
    # metadata_store.py. Customer-tier types can raise it; document-tier
    # tends to keep it at 50 with autosplit on.
    capacity_default: int = 50

    autosplit_policy: AutosplitPolicy = "split"

    # Both nullable — see module docstring for the snapshot vs. live-
    # global rationale.
    routing_threshold: Optional[float] = None
    cleanup_threshold: Optional[float] = None

    # Phase 4 hook. Empty string today; the Phase 4 DSL parser/compiler
    # consumes this column when authored crystal types arrive.
    pair_schema_dsl: str = ""

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# CrystalAcl
# ---------------------------------------------------------------------------

class CrystalAcl(BaseModel):
    """Per-crystal access grant.

    A single crystal can carry multiple grants:
      (crys_X, 'customer', 'cus_A', 'read')
      (crys_X, 'crystal_chain', 'crys_Y', 'read_codebook')

    Each row is independent; there's no UPDATE-on-conflict semantics.
    DELETE one to revoke that specific grant.

    Grants are ADDITIVE on top of scope defaults. A customer-scope
    crystal with no ACL rows is still readable by its owning customer
    (the resolver falls back to scope defaults). ACL rows extend
    access; the absence of rows doesn't restrict it below scope
    default. To restrict, the system would need explicit deny-grants
    — out of scope for Phase 3.
    """

    crystal_id: str
    principal_type: PrincipalType
    principal_id: str
    grant: AclGrant

    granted_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


# ---------------------------------------------------------------------------
# CrystalChain
# ---------------------------------------------------------------------------

class CrystalChain(BaseModel):
    """Directed edge between two crystals for cleanup-codebook extension.

    When recall_from_crystal runs cleanup on `source_crystal_id`, it
    walks outgoing chains (rows where source_crystal_id matches) and
    unions `target_crystal_id`'s Facts into the cleanup codebook —
    subject to the target's `read_codebook` ACL grant being present
    for the source's customer.

    direction='source_uses_target' is one-way (default).
    direction='bidirectional' means the same row also extends in the
    reverse direction: when recall runs on `target_crystal_id`, it
    pulls `source_crystal_id`'s Facts in too. Implementation detail:
    bidirectional chains are stored as ONE row, and the resolver
    consults `direction` when traversing target -> sources via the
    `ix_crystal_chains_target` index. We don't store two rows for
    bidirectional because (a) the source/target labels are still
    semantically distinct (the row's "owner" is the source) and
    (b) it'd require keeping two rows in sync on direction changes.

    Self-loops (source == target) are model-allowed but resolver-
    skipped — a crystal already includes its own facts in cleanup,
    chaining to itself adds nothing and would just waste a DB lookup.
    The MetadataStore.add_chain method rejects self-loops at write
    time as a defensive guard.
    """

    source_crystal_id: str
    target_crystal_id: str
    direction: ChainDirection = "source_uses_target"

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
