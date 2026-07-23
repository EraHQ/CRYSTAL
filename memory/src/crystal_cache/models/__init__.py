"""Data model entities — §6 of BUILD_PROPOSAL.md.

Pydantic models for now. When we pick a DB we'll either subclass these with
SQLModel or map them through SQLAlchemy. Keeping them transport-first so
the ingress API can use them directly.
"""

from .customer import Customer, ModelRoutingConfig
from .operator import Operator, OperatorRole, OperatorStatus
from .entity import Entity
from .user import User, UserRole
from .crystal import Crystal, CrystalEdge
from .task_key import TaskKey
from .crystal_type import (
    AclGrant,
    AutosplitPolicy,
    ChainDirection,
    CrystalAcl,
    CrystalChain,
    CrystalScope,
    CrystalType,
    PrincipalType,
)
from .fact import Fact
from .feedback import Feedback, FeedbackSignal
from .query_log import QueryLog
from .verification import VerificationTask
from .document import Document
from .diagnostic import CrystalDiagnostic, CrystalEdit

# Phase 4 (v2 port, May 2026): audit-table models. These cover the
# document ingestion pipeline, Drive sync, HIPAA tracking, push/pull
# review queue, knowledge gaps, and cognition task queue. Per audit
# Decision D1, PhiAccessLog stays dict-in/dict-out at the MetadataStore
# boundary and has no Pydantic model here.
from .document_upload import DocumentUpload, DocumentUploadStatus
from .drive_connection import (
    DriveConnection,
    DriveConnectionStatus,
    WatchedFile,
    WatchedFolder,
    WatchedResourceStatus,
)
from .baa_tracking import BaaTracking
from .push_review import (
    PushReviewItem,
    PushReviewSource,
    PushReviewStatus,
)
from .spend_budget import BudgetPeriod, SpendBudget  # noqa: F401
from .knowledge_gap import (
    GapPriority,
    GapSource,
    GapStatus,
    KnowledgeGap,
)
from .knowledge_conflict import (
    ConflictDetector,
    ConflictResolution,
    ConflictStatus,
    KnowledgeConflict,
)
from .cognition_task import (
    CognitionTask,
    TaskPriority,
    TaskStatus,
    TaskType,
)

# Phase 8.5 (2026-05-27): MCR artifact models per `docs/MCR_FRAMEWORK.md`
# §4.1–§4.3. Three new artifacts: reasoning trace, critique, action
# item. Item-alignment and critique-synthesis tables land in Phase 10
# alongside the metacognitive layer that produces them. Per P0.34, no
# writers / readers / behavior changes ship in Phase 8.5 — only the
# data shapes and CRUD methods.
from .reasoning_trace import ReasoningTrace
from .critique import Critique, CriticRole, ObservationType
from .action_item import ActionItem, ActionItemStatus, ActionType

# Phase 10A (2026-05-27): metacognitive artifact models per
# `docs/MCR_FRAMEWORK.md` §4.4–§4.5. Item alignment + critique
# synthesis. Computed by the new `metacognition/` package; the schema
# tables mirror these 1:1 (ItemAlignmentRow, CritiqueSynthesisRow).
from .item_alignment import AlignmentClass, ItemAlignment
from .critique_synthesis import CritiqueSynthesis

# Phase 10B (2026-05-27): critic calibration per MCR §7. Running
# estimates per (customer_id, critic_role, critic_model). Written
# after each synthesis; not used by the promotion decision in 10B
# (forward-compatible bookkeeping for Phase 11+).
from .critic_calibration import CriticCalibration

__all__ = [
    "TaskKey",
    # Customer + routing
    "Customer",
    "ModelRoutingConfig",
    # Operators (Foundation F1 — team identity layer)
    "Operator",
    "Entity",
    "User",
    "UserRole",
    "OperatorRole",
    "OperatorStatus",
    # Crystal + edges + type registry
    "Crystal",
    "CrystalEdge",
    "CrystalType",
    "CrystalAcl",
    "CrystalChain",
    "CrystalScope",
    "AutosplitPolicy",
    "PrincipalType",
    "AclGrant",
    "ChainDirection",
    # Facts
    "Fact",
    # Feedback
    "Feedback",
    "FeedbackSignal",
    # Query log
    "QueryLog",
    # Verification
    "VerificationTask",
    # Document
    "Document",
    # Diagnostics + edits
    "CrystalDiagnostic",
    "CrystalEdit",
    # Phase 4: document upload pipeline
    "DocumentUpload",
    "DocumentUploadStatus",
    # Phase 4: Drive sync
    "DriveConnection",
    "DriveConnectionStatus",
    "WatchedFile",
    "WatchedFolder",
    "WatchedResourceStatus",
    # Phase 4: HIPAA / BAA tracking
    "BaaTracking",
    # Phase 4: push/pull review queue
    "PushReviewItem",
    "PushReviewSource",
    "PushReviewStatus",
    # Phase 4: knowledge gaps
    "KnowledgeGap",
    "GapPriority",
    "GapStatus",
    "GapSource",
    # Never-idle convergence: knowledge conflicts
    "KnowledgeConflict",
    "ConflictStatus",
    "ConflictResolution",
    "ConflictDetector",
    # Phase 4: cognition task queue
    "CognitionTask",
    "TaskType",
    "TaskPriority",
    "TaskStatus",
    # Phase 8.5: MCR artifacts
    "ReasoningTrace",
    "Critique",
    "CriticRole",
    "ObservationType",
    "ActionItem",
    "ActionType",
    "ActionItemStatus",
    # Phase 10A: metacognitive artifacts
    "ItemAlignment",
    "AlignmentClass",
    "CritiqueSynthesis",
    # Phase 10B: critic calibration
    "CriticCalibration",
]
