"""Infrastructure layer — §7 of BUILD_PROPOSAL.md.

Storage + observability primitives used by all other layers:
  - VectorStore: physical storage for fact vectors (10k-dim each)
  - MetadataStore: Postgres or similar for Crystal, Fact, QueryLog, etc
  - schema: SQLAlchemy ORM table definitions

Everything else talks to these via their interfaces, never directly.

NOTE (v2 port, May 2026): v1's `Telemetry` and `TenancyEnforcer` stubs
were not ported. Both raised NotImplementedError everywhere — they were
placeholder classes from an early scaffolding pass and no production
code paths referenced them. Cleanup queue items CU-2 (telemetry.py)
and CU-3 (tenancy.py) document this; if either capability lands later
it'll be a deliberate new design, not a port of the empty stubs.

Mixin pattern (D12) — bind extension methods onto MetadataStore at
module import time rather than editing the 117 KB Phase 3 verbatim
port. SIX mixins live alongside the core today (Phase 10A 2026-05-27):

  - AuditTablesMixin (Phase 5, `metadata_store_audit.py`).
  - CustomerExtensionsMixin (Phase 6.5 P4.1,
    `metadata_store_customer_ext.py`).
  - CognitionExtensionsMixin (Phase 6 Wave C,
    `metadata_store_cognition_ext.py`).
  - LearningExtensionsMixin (Phase 7 Wave 7E,
    `metadata_store_learning_ext.py`).
  - McrExtensionsMixin (Phase 8.5, `metadata_store_mcr_ext.py`) —
    MCR artifact CRUD for `reasoning_traces`, `critiques`,
    `action_items`.
  - MetacognitionExtensionsMixin (Phase 10A,
    `metadata_store_metacog_ext.py`) — metacognitive artifact
    CRUD for `item_alignments`, `critique_syntheses`.
  - **AgentTasksMixin (CRYS daemon, 2026-06-11,
    `metadata_store_agent_ext.py`)** — work-queue CRUD for
    `agent_tasks` (the coding-agent daemon's poll target).

SEVEN mixins as of 2026-06-11; R9 itself is count-agnostic.
"""
from .vector_store import VectorStore
from .metadata_store import MetadataStore
from .metadata_store_audit import AuditTablesMixin
from .metadata_store_customer_ext import CustomerExtensionsMixin
from .metadata_store_cognition_ext import CognitionExtensionsMixin
from .metadata_store_learning_ext import LearningExtensionsMixin
from .metadata_store_mcr_ext import McrExtensionsMixin
from .metadata_store_metacog_ext import MetacognitionExtensionsMixin
from .metadata_store_agent_ext import AgentTasksMixin
from .metadata_store_promotion_ext import PromotionExtensionsMixin
from .metadata_store_session_ext import SessionRegistryMixin
from .metadata_store_citation_ext import CitationExtensionsMixin
from .metadata_store_control_ext import ControlExtensionsMixin
from .metadata_store_cost_ext import CostExtensionsMixin
from .metadata_store_shard_ext import ShardExtensionsMixin
from .metadata_store_event_ext import EventLogMixin
from .metadata_store_conflict_ext import ConflictExtensionsMixin
from .metadata_store_websearch_ext import WebSearchExtensionsMixin
from .metadata_store_gap_ext import GapExtensionsMixin
from .metadata_store_backlog_ext import BacklogExtensionsMixin
from .metadata_store_conversation_ext import ConversationExtensionsMixin
from . import schema


def _bind_mixin_methods(target_class, mixin_class) -> None:
    """Bind all public callable attributes from a mixin onto a target class.

    Iterates the mixin's public method names (no underscore prefix) and
    setattr's each one onto the target. The mixin is NOT in the target's
    MRO — the binding is attribute-level, not inheritance-level. Inside
    a bound method, `self.session()` resolves to MetadataStore.session
    via normal attribute lookup on the bound callable.

    Module-level converter functions in the mixin files stay at module
    level — they're not callables on the mixin class itself.
    """
    for attr_name in dir(mixin_class):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mixin_class, attr_name)
        if callable(attr):
            setattr(target_class, attr_name, attr)


# Bind all SIX mixins onto MetadataStore at import time.
# Why not class-level inheritance? Two reasons:
#   1. Avoids touching the existing 117 KB metadata_store.py file
#      (lower risk, easier diff review)
#   2. The mixins are conceptually adjunct to the existing methods,
#      not parent classes with their own identity. The binding here
#      makes that relationship explicit at the package boundary.
_bind_mixin_methods(MetadataStore, AuditTablesMixin)
_bind_mixin_methods(MetadataStore, CustomerExtensionsMixin)
_bind_mixin_methods(MetadataStore, CognitionExtensionsMixin)
_bind_mixin_methods(MetadataStore, LearningExtensionsMixin)
_bind_mixin_methods(MetadataStore, McrExtensionsMixin)
_bind_mixin_methods(MetadataStore, MetacognitionExtensionsMixin)
_bind_mixin_methods(MetadataStore, AgentTasksMixin)
# Foundation F3 (promotion engine): crystal_contributions provenance CRUD.
_bind_mixin_methods(MetadataStore, PromotionExtensionsMixin)
# Foundation F4 (surface consolidation): agent_sessions + dependencies CRUD.
_bind_mixin_methods(MetadataStore, SessionRegistryMixin)
# Growth G1 (citations): citations record CRUD for the metering rail.
_bind_mixin_methods(MetadataStore, CitationExtensionsMixin)
# Growth G2 (control plane): control_commands command-channel CRUD.
_bind_mixin_methods(MetadataStore, ControlExtensionsMixin)
# Growth G3 (cost accounting): llm_calls record + GROUP BY aggregation.
_bind_mixin_methods(MetadataStore, CostExtensionsMixin)
# Growth G4 (marketplace): shard_events ledger + expert-vetting CRUD.
_bind_mixin_methods(MetadataStore, ShardExtensionsMixin)
# Unify-Agents: agent_events append-only activity stream (the "Agents"
# surface, the unified interaction log, and cost rollups read this).
_bind_mixin_methods(MetadataStore, EventLogMixin)
# Never-idle convergence: knowledge_conflicts CRUD (the contradiction-scan
# generator's write target + the admin Conflicts surface / backlog read it).
_bind_mixin_methods(MetadataStore, ConflictExtensionsMixin)
# Web-search interaction log (launch-prep sweep): the goldmine's raw side —
# the web_search tool writes one row per search; joins to crystallized
# knowledge by URL via provenance.
_bind_mixin_methods(MetadataStore, WebSearchExtensionsMixin)
# Never-idle convergence: knowledge_gaps reads (the peer of conflicts; the
# memory_gaps MCP tool + knowledge_gaps agent tool consume the full gap rows,
# which the backlog read-model collapses away).
_bind_mixin_methods(MetadataStore, GapExtensionsMixin)
# Never-idle convergence: list_backlog read-model (one ranked view over every
# waiting-work queue — gaps, conflicts, cognition/agent tasks, review, verify).
_bind_mixin_methods(MetadataStore, BacklogExtensionsMixin)
# CRYS session continuity: agent_conversations CRUD (per-scope transcript
# persistence so context survives exit/relaunch; mode-agnostic).
_bind_mixin_methods(MetadataStore, ConversationExtensionsMixin)


__all__ = [
    "VectorStore",
    "MetadataStore",
    "schema",
]
