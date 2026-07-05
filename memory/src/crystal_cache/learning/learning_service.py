"""Learning Service — Level B validation + Level F crystal generation.

Production service that extracts the learning pipeline from
scripts/run_level_f.py into an importable, multi-tenant async service.

Pipeline per failure:
  1. Level B: Generate failure reflection (imperative rule)
  2. Level B: Self-check (is the reflection consistent with evidence?)
  3. Level B: Cross-validate against blacklisted reflections
  4. Level F1: Extract domain knowledge + prior assumption
  5. Level F2: Infer behavioral pattern
  6. Generate sparse key for each crystal at ingestion time
  7. Store all crystals in the customer's bank

The service is triggered by:
  - Feedback endpoint (user thumbs-down → learn_from_failure)
  - Batch learner (eval results → learn_from_failures)
  - Manual trigger (admin endpoint)

RELATIONSHIP TO EXISTING CODE:
  - scripts/run_level_f.py — the benchmark version (BCB-specific,
    file-based state, hardcoded customer). Stays for benchmarking.
  - src/crystal_cache/learning/crystallizer.py — the V1 crystallizer
    (simpler reflection, no F1/F2/self-check, no structured output).
    Stays for back-compat. This service supersedes it for V2.
  - This service reads/writes the V2 DB tables (Item 3):
    blacklisted_reflections (not JSON files).

USAGE:
    from crystal_cache.learning.learning_service import LearningService

    service = LearningService(store=store, encoder=encoder, vector_store=vs)
    result = await service.learn_from_failure(
        customer_id="cus_xxx",
        prompt="Write a function that...",
        response="def task_func()...",
        failure_signal="Test failed: expected tuple, got list",
        crystal_type="general:python_general",
    )

PORT NOTE (v2 Wave 7E, 2026-05-20): the two methods
`_is_blacklisted` and `_add_to_blacklist` in v1 carried inline
SQLAlchemy queries against `BlacklistedReflectionRow` — R9
violation. Per Wave 7E AN-7 refactor, those queries moved to
`LearningExtensionsMixin.is_reflection_blacklisted` and
`add_blacklisted_reflection` on the store. The hash-computation
helper stays here. This file is no longer verbatim with v1; the
refactor is documented in PROJECT_LEDGER.md (AN-7, P0.8,
Wave 7E close-out).
"""
from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Optional, TYPE_CHECKING

from ..encoding.sparse_keys import generate_sparse_key
from ..llm import get_llm_client

if TYPE_CHECKING:
    from ..encoding.semantic import SemanticTextEncoder
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_index import VectorIndex
    from ..infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Structured output schema (from run_level_f.py)
# ---------------------------------------------------------------------------

COMBINED_SCHEMA = {
    "type": "object",
    "properties": {
        "reflection": {
            "type": "string",
            "description": (
                "One imperative rule (1-2 sentences) that would prevent "
                "this failure. Start with a verb. Be specific to the "
                "actual error."
            ),
        },
        "category": {
            "type": "string",
            "enum": [
                "API_BEHAVIOR",
                "SPEC_INTERPRETATION",
                "ALGORITHM",
                "LIBRARY_PATTERN",
            ],
            "description": "What category of knowledge gap caused this failure.",
        },
        "knowledge": {
            "type": "string",
            "description": (
                "2-4 sentences filling the specific knowledge gap. "
                "Include a code snippet if about API behavior."
            ),
        },
        "inference": {
            "type": "string",
            "description": (
                "One sentence: what does the code's own behavior imply "
                "about the expected output format? 'NO_INFERENCE' if "
                "nothing can be inferred."
            ),
        },
        "prior_assumption": {
            "type": "string",
            "description": (
                "What did the solution implicitly ASSUME that was wrong? "
                "State the assumption, not the fix. 'NO_ASSUMPTION' if "
                "it was a pure implementation bug."
            ),
        },
        "self_check": {
            "type": "object",
            "properties": {
                "is_consistent": {
                    "type": "boolean",
                    "description": (
                        "true if the reflection is consistent with "
                        "knowledge and inference."
                    ),
                },
                "corrected_reflection": {
                    "type": ["string", "null"],
                    "description": (
                        "If is_consistent is false, the corrected "
                        "reflection. null otherwise."
                    ),
                },
            },
            "required": ["is_consistent"],
            "additionalProperties": False,
        },
    },
    "required": [
        "reflection",
        "category",
        "knowledge",
        "inference",
        "prior_assumption",
        "self_check",
    ],
    "additionalProperties": False,
}

COMBINED_SYSTEM = (
    "You are a Python expert analyzing a code failure. Analyze the "
    "failure and produce structured output with these fields:\n\n"
    "- reflection: An imperative rule that would prevent this failure.\n"
    "- category: The type of knowledge gap (API_BEHAVIOR, "
    "SPEC_INTERPRETATION, ALGORITHM, or LIBRARY_PATTERN)\n"
    "- knowledge: 2-4 sentences filling the knowledge gap.\n"
    "- inference: What the code's own imports/calls imply about "
    "expected output format. 'NO_INFERENCE' if nothing.\n"
    "- prior_assumption: What did the solution implicitly ASSUME "
    "that was wrong? 'NO_ASSUMPTION' if it was a pure bug.\n"
    "- self_check.is_consistent: Does your reflection contradict "
    "your knowledge/inference?\n"
    "- self_check.corrected_reflection: If inconsistent, the "
    "corrected reflection."
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class LearningResult:
    """Result of processing one failure through the learning pipeline."""

    customer_id: str
    prompt: str

    # Level B
    reflection: Optional[str] = None
    reflection_valid: bool = True
    reflection_corrected: Optional[str] = None
    reflection_blacklisted: bool = False

    # Level F1
    category: Optional[str] = None
    knowledge: Optional[str] = None
    prior_assumption: Optional[str] = None

    # Level F2
    inference: Optional[str] = None

    # Ingestion counts
    crystals_written: int = 0
    sparse_keys_generated: int = 0

    # Errors
    error: Optional[str] = None


@dataclass
class BatchLearningResult:
    """Result of processing multiple failures."""

    customer_id: str
    failures_processed: int = 0
    results: list[LearningResult] = field(default_factory=list)
    total_crystals_written: int = 0
    total_blacklisted: int = 0
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Learning Service
# ---------------------------------------------------------------------------

class LearningService:
    """Async service for Level B + Level F learning.

    Extracts the validated learning pipeline from run_level_f.py
    into production code that:
    - Works with any customer (not hardcoded to BCB)
    - Reads/writes DB tables (not JSON files)
    - Generates sparse keys at ingestion time
    - Cross-validates against blacklisted reflections in DB
    """

    def __init__(
        self,
        store: "MetadataStore",
        encoder: "SemanticTextEncoder",
        vector_store: "VectorStore",
        *,
        vector_index: Optional["VectorIndex"] = None,
    ):
        self._store = store
        self._encoder = encoder
        self._vector_store = vector_store
        self._vector_index = vector_index

    # -----------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------

    async def learn_from_failure(
        self,
        customer_id: str,
        prompt: str,
        response: str,
        failure_signal: str,
        *,
        crystal_type: str = "general:python_general",
        prior_rules: Optional[list[str]] = None,
        iteration: int = 0,
    ) -> LearningResult:
        """Process one failure through the full B+F pipeline.

        Args:
            customer_id: Which customer's bank to write to.
            prompt: The original prompt/query that was sent.
            response: The model's response that failed.
            failure_signal: Why it failed (test errors, user feedback, etc.)
            crystal_type: Crystal type for routing.
            prior_rules: Previously generated rules for this prompt
                (to avoid duplicates in the reflection).
            iteration: Learning round number (for pair_type tagging).

        Returns:
            LearningResult with all generated crystals and metadata.
        """
        result = LearningResult(customer_id=customer_id, prompt=prompt)

        # Step 1: Generate B+F structured output in ONE API call
        parsed = await self._call_combined_bf(
            prompt=prompt,
            response=response,
            failure_signal=failure_signal,
            prior_rules=prior_rules or [],
        )
        if parsed is None:
            result.error = "LLM call failed"
            return result

        # Step 2: Extract fields
        reflection = parsed.get("reflection")
        category = parsed.get("category")
        knowledge = parsed.get("knowledge")
        inference = parsed.get("inference")
        prior_assumption = parsed.get("prior_assumption")
        self_check = parsed.get("self_check", {})

        b_valid = self_check.get("is_consistent", True)
        b_corrected = self_check.get("corrected_reflection")

        if inference == "NO_INFERENCE":
            inference = None
        if prior_assumption == "NO_ASSUMPTION":
            prior_assumption = None

        result.category = category
        result.knowledge = knowledge
        result.prior_assumption = prior_assumption
        result.inference = inference
        result.reflection_valid = b_valid

        # Step 3: Level B self-check — use corrected if invalid
        if b_valid and reflection:
            result.reflection = reflection
        elif not b_valid and b_corrected:
            result.reflection = b_corrected
            result.reflection_corrected = b_corrected
        elif not b_valid:
            # Self-check says invalid, no correction available — drop
            result.reflection = None

        # Step 4: Cross-validate against blacklist
        if result.reflection:
            is_blacklisted = await self._is_blacklisted(
                customer_id, result.reflection
            )
            if is_blacklisted:
                result.reflection_blacklisted = True
                result.reflection = None

        # Step 5: Blacklist the original invalid reflection
        if not b_valid and reflection:
            await self._add_to_blacklist(
                customer_id=customer_id,
                reflection_text=reflection,
                reason="self_check_invalid",
            )

        # Step 6: Ingest crystals
        crystals = await self._ingest(
            customer_id=customer_id,
            result=result,
            prompt=prompt,
            crystal_type=crystal_type,
            iteration=iteration,
        )
        result.crystals_written = crystals

        return result

    async def learn_from_failures(
        self,
        customer_id: str,
        failures: list[dict[str, str]],
        *,
        crystal_type: str = "general:python_general",
        iteration: int = 0,
        concurrency: int = 5,
    ) -> BatchLearningResult:
        """Process multiple failures concurrently.

        Each failure dict should have keys:
            prompt, response, failure_signal

        Optional keys:
            prior_rules (list[str])
        """
        batch = BatchLearningResult(customer_id=customer_id)
        sem = asyncio.Semaphore(concurrency)

        async def _worker(failure: dict[str, str]) -> LearningResult:
            async with sem:
                return await self.learn_from_failure(
                    customer_id=customer_id,
                    prompt=failure["prompt"],
                    response=failure["response"],
                    failure_signal=failure["failure_signal"],
                    crystal_type=crystal_type,
                    prior_rules=failure.get("prior_rules"),
                    iteration=iteration,
                )

        tasks = [
            asyncio.create_task(_worker(f)) for f in failures
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for r in results:
            if isinstance(r, Exception):
                logger.error("learn_from_failure error: %s", r)
                batch.results.append(LearningResult(
                    customer_id=customer_id,
                    prompt="",
                    error=str(r),
                ))
            else:
                batch.results.append(r)
                batch.total_crystals_written += r.crystals_written
                if r.reflection_blacklisted:
                    batch.total_blacklisted += 1

        batch.failures_processed = len(failures)
        return batch

    async def cache_success(
        self,
        customer_id: str,
        prompt: str,
        solution: str,
        *,
        crystal_type: str = "general:python_general",
    ) -> bool:
        """Cache a successful solution for future cache-hit retrieval.

        Returns True if cached successfully.
        """
        try:
            sparse_key = generate_sparse_key(prompt)
            await self._store.add_pair_for_customer(
                customer_id=customer_id,
                prompt_text=sparse_key,
                answer_text=solution,
                pair_type="cached_solution",
                source_kind="model_reasoning",
                answer_value=solution,
                encoder=self._encoder,
                vector_store=self._vector_store,
                vector_index=self._vector_index,
                crystal_type=crystal_type,
            )
            return True
        except Exception as e:
            logger.error(
                "cache_success failed: customer=%s error=%s",
                customer_id, e,
            )
            return False

    # -----------------------------------------------------------------
    # Private: LLM call
    # -----------------------------------------------------------------

    async def _call_combined_bf(
        self,
        prompt: str,
        response: str,
        failure_signal: str,
        prior_rules: list[str],
    ) -> Optional[dict[str, Any]]:
        """One structured API call for B + F1 + F2 + self-check.

        Runs through the provider-neutral seam at tier small with json_schema
        structured output; the sync seam call is wrapped in a thread so the
        event loop is never blocked. Fail-safe: any exception returns None.
        """
        prior_text = (
            "Prior rules tried (all failed):\n"
            + "\n".join(f"  - {r}" for r in prior_rules)
        ) if prior_rules else "(no prior rules)"

        user_prompt = (
            f"## Task\n{prompt[:3000]}\n\n"
            f"## Failed Solution\n```python\n{response[:3000]}\n```\n\n"
            f"## Failure Signal\n{failure_signal[:2000]}\n\n"
            f"## Prior Rules\n{prior_text[:500]}\n\n"
            "Analyze this failure."
        )

        try:
            raw = await asyncio.to_thread(
                functools.partial(
                    get_llm_client().complete,
                    system=COMBINED_SYSTEM,
                    messages=[{"role": "user", "content": user_prompt}],
                    max_tokens=768,
                    temperature=0.0,
                    tier="small",
                    json_schema=COMBINED_SCHEMA,
                )
            )
            return json.loads(raw)
        except Exception as e:
            logger.error("Combined B+F call failed: %s", e)
            return None

    # -----------------------------------------------------------------
    # Private: Blacklist operations (DB-backed via store mixin)
    # -----------------------------------------------------------------

    @staticmethod
    def _hash_reflection(text: str) -> str:
        """SHA256 hash of reflection text, truncated to 16 hex chars."""
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]

    async def _is_blacklisted(
        self, customer_id: str, reflection: str
    ) -> bool:
        """Check if a reflection is blacklisted for this customer.

        Per Wave 7E AN-7 refactor: inline SQLAlchemy moved to
        `LearningExtensionsMixin.is_reflection_blacklisted` on the
        store. This method now just hashes + forwards.
        """
        reflection_hash = self._hash_reflection(reflection)
        return await self._store.is_reflection_blacklisted(
            customer_id=customer_id,
            reflection_hash=reflection_hash,
        )

    async def _add_to_blacklist(
        self,
        customer_id: str,
        reflection_text: str,
        reason: str,
    ) -> None:
        """Add a reflection to the blacklist.

        Per Wave 7E AN-7 refactor: inline SQLAlchemy moved to
        `LearningExtensionsMixin.add_blacklisted_reflection` on the
        store. This method now just hashes + forwards.
        """
        reflection_hash = self._hash_reflection(reflection_text)
        await self._store.add_blacklisted_reflection(
            customer_id=customer_id,
            reflection_hash=reflection_hash,
            reflection_text=reflection_text,
            reason=reason,
        )

    # -----------------------------------------------------------------
    # Private: Crystal ingestion
    # -----------------------------------------------------------------

    async def _ingest(
        self,
        customer_id: str,
        result: LearningResult,
        prompt: str,
        crystal_type: str,
        iteration: int,
    ) -> int:
        """Ingest all generated crystals into the customer's bank.

        Returns the number of crystals written.
        """
        count = 0

        # Level B reflection → failure rule crystal
        if result.reflection:
            try:
                sparse_key = generate_sparse_key(prompt)
                result.sparse_keys_generated += 1
                await self._store.add_pair_for_customer(
                    customer_id=customer_id,
                    prompt_text=sparse_key,
                    answer_text=result.reflection,
                    pair_type=f"failure_reflection_iter{iteration}",
                    source_kind="failed_reasoning",
                    encoder=self._encoder,
                    vector_store=self._vector_store,
                    vector_index=self._vector_index,
                    crystal_type=crystal_type,
                )
                count += 1
            except Exception as e:
                logger.error(
                    "B reflection ingestion failed: %s", e
                )

        # F1 knowledge crystal
        if result.knowledge:
            try:
                prior_part = ""
                if result.prior_assumption:
                    prior_part = (
                        f" prior:{result.prior_assumption[:100]} |"
                    )
                f1_source = (
                    f"{result.category or 'python'} knowledge:"
                    f"{prior_part} {result.knowledge[:200]}"
                )
                sparse_key = generate_sparse_key(f1_source)
                result.sparse_keys_generated += 1
                await self._store.add_pair_for_customer(
                    customer_id=customer_id,
                    prompt_text=sparse_key,
                    answer_text=result.knowledge,
                    pair_type=f"knowledge_crystal_iter{iteration}",
                    source_kind="model_reasoning",
                    encoder=self._encoder,
                    vector_store=self._vector_store,
                    vector_index=self._vector_index,
                    crystal_type=crystal_type,
                )
                count += 1
            except Exception as e:
                logger.error(
                    "F1 knowledge ingestion failed: %s", e
                )

        return count
