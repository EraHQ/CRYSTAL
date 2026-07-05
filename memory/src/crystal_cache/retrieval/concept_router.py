"""ConceptRouter — text-query → decomposer → concept-space → config rank.

Parallel to CrystalRouter (which does text-space Fact retrieval), this
router is the concept-space retrieval path for DSL-compiled configs.

Flow:
    text query
        │
        ▼
    decomposer.decompose(text)  →  DecompositionResult
        │
        ▼
    from_decomposer_output(payload, vocab)  →  concept HV
        │
        ▼
    config_store.rank(tenant_id, concept_hv)  →  [(config_name, sim), ...]

The router stays thin — holds references to the decomposer and the
config store. The decomposer is swappable via the Decomposer protocol:
stub for testing, local LLM for production, Sawyer's classifier when/if
that ships.

FAILURE BEHAVIOR
----------------
If the decomposer fails (DecomposerError), we return an empty list
rather than raising. This keeps the concept-path strictly additive:
whatever the text-path retrieves is still returned to the caller.
A failed concept-path never breaks a request.

TENANT_ID IS LOAD-BEARING
-------------------------
Every routing call must specify tenant_id. The concept HV must be
built using the tenant's vocabulary (so it uses the right SHA-256
namespace), and the config ranking must search that tenant's configs
(so there's no cross-tenant leakage). Mismatched tenants produce
guaranteed-wrong results — we assert this rather than hope.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import structlog

from crystal_cache.decomposer.base import (
    DecompositionResult,
    Decomposer,
    DecomposerError,
)
from crystal_cache.decomposer.config_store import DslConfigStore
from crystal_cache.dsl import from_decomposer_output


logger = structlog.get_logger(__name__)


@dataclass
class ConceptRouteOutcome:
    """What the concept router returns.

    Attributes:
        ranked_configs: List of (config_name, similarity) pairs sorted
            descending. Empty if the tenant has no configs, or if the
            decomposer failed, or if the decomposer returned a
            degenerate payload.
        decomposition: The raw DecompositionResult from the decomposer,
            for logging. None if the decomposer wasn't called (empty
            query) or raised.
        query_vector: The concept-space hypervector we ranked against.
            None if decomposition failed.
        decomposer_failed: True if the decomposer raised DecomposerError.
            Callers can use this to downgrade concept-path confidence.
    """

    ranked_configs: list[tuple[str, float]]
    decomposition: Optional[DecompositionResult]
    query_vector: Optional[np.ndarray]
    decomposer_failed: bool = False


class ConceptRouter:
    """Route free-text queries to DSL configs via the decomposer bridge.

    Holds no request state. A single instance can be shared across
    concurrent requests as long as the decomposer and config store it
    wraps are both concurrent-safe (or the caller serializes access).
    """

    def __init__(
        self,
        decomposer: Decomposer,
        config_store: DslConfigStore,
    ) -> None:
        self._decomposer = decomposer
        self._config_store = config_store

    async def route(
        self,
        tenant_id: str,
        query_text: str,
        *,
        context: Optional[dict[str, Any]] = None,
        top_k: int = 3,
    ) -> ConceptRouteOutcome:
        """Run the full concept-path for a query.

        Returns a ConceptRouteOutcome with at most `top_k` ranked configs.
        Never raises; all failure modes produce an empty outcome with
        appropriate flags set.
        """
        if not query_text or not query_text.strip():
            return ConceptRouteOutcome(
                ranked_configs=[],
                decomposition=None,
                query_vector=None,
            )

        # Force a DB load if we're lazy-loading. No-op in the in-memory
        # case where register_source was used directly.
        await self._config_store.ensure_loaded(tenant_id)

        if not self._config_store.has_tenant(tenant_id):
            # No configs registered for this tenant. Concept-path has
            # nothing to contribute. Don't bother calling the decomposer.
            return ConceptRouteOutcome(
                ranked_configs=[],
                decomposition=None,
                query_vector=None,
            )

        # Step 1: decompose the query.
        try:
            decomposition = await self._decomposer.decompose(
                query_text, context=context
            )
        except DecomposerError as e:
            logger.warning(
                "concept_router.decomposer_failed",
                tenant_id=tenant_id,
                error=str(e),
            )
            return ConceptRouteOutcome(
                ranked_configs=[],
                decomposition=None,
                query_vector=None,
                decomposer_failed=True,
            )

        # Step 2: build concept-space query vector.
        env = self._config_store.get_env(tenant_id)
        assert env is not None  # we checked has_tenant above
        try:
            query_hv = from_decomposer_output(decomposition.payload, env.vocab)
        except Exception as e:
            # from_decomposer_output raises RuntimeError on unsupported
            # payload types (nested non-JSON types, etc.). Log and
            # degrade rather than 500.
            logger.warning(
                "concept_router.payload_encoding_failed",
                tenant_id=tenant_id,
                error=str(e),
                payload=decomposition.payload,
            )
            return ConceptRouteOutcome(
                ranked_configs=[],
                decomposition=decomposition,
                query_vector=None,
            )

        # Step 3: rank configs.
        ranked = self._config_store.rank(tenant_id, query_hv)
        ranked = ranked[:top_k]

        logger.debug(
            "concept_router.ranked",
            tenant_id=tenant_id,
            top_config=ranked[0][0] if ranked else None,
            top_score=ranked[0][1] if ranked else 0.0,
            decomposer_model=decomposition.model_name,
        )

        return ConceptRouteOutcome(
            ranked_configs=ranked,
            decomposition=decomposition,
            query_vector=query_hv,
        )
