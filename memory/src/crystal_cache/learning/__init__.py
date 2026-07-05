"""Learning layer — §7 of BUILD_PROPOSAL.md.

Where facts enter the system and where crystals get smarter over time.
The living subsystems:

  1. Crystallization surface (`crystallizer.py`): success/failure
     crystals from agent tool outputs, plus provenance extractors.
  2. Diagnostic + edit proposal (research §4): telemetry →
     diagnostic_engine → edit_proposer → CrystalEdit queue.
  3. LearningService (`learning_service.py`): failure-derived rule
     distillation through the provider-neutral seam (Wave 7E; the
     BlacklistedReflection store methods replaced its inline SQL).
  4. Bonding (`bonder.py`, imported directly by its consumers).

History note (launch-prep purge, 2026-07-02): v1's scaffolded
fact-ingestion pipeline (FactExtractionQueue → LLMFactExtractor →
VerificationQueue → CrystalWriter), the OfflineCrystalEvaluator, and
the WTrainer were NotImplementedError stubs and were removed. Their
jobs were rebuilt elsewhere in v2 under different shapes: the document
pipeline + crystallization worker do extraction, push_review_queue does
verification, the reflection loop does failure-derived learning, and
the convergence scans do dedup/contradiction. The genuinely-unbuilt
pieces (decay policy, co-query edge population, retrieval weighting by
quality tier) are tracked in docs/BACKLOG.md §13. WTrainer belongs to
the parked hidden-state research line — docs/RESEARCH_DIRECTIONS.md.
"""
from .crystallizer import (
    ToolOutput,
    crystallize_success,
    crystallize_failure,
    extract_web_search_provenance,
    extract_code_exec_output,
)
from .diagnostic_engine import DiagnosticEngine, CrystalEvent
from .edit_proposer import CrystalEditProposer, BankStatistics, CrystalStats
from .learning_service import LearningService, LearningResult, BatchLearningResult
from .diagnostic_loop import run_once as run_diagnostic_loop_once

__all__ = [
    "ToolOutput",
    "crystallize_success",
    "crystallize_failure",
    "extract_web_search_provenance",
    "extract_code_exec_output",
    "DiagnosticEngine",
    "CrystalEvent",
    "CrystalEditProposer",
    "BankStatistics",
    "CrystalStats",
    "LearningService",
    "LearningResult",
    "BatchLearningResult",
    "run_diagnostic_loop_once",
]
