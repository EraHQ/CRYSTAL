"""Crystal router — encode the query, look up top-K crystals.

Thin composition layer: PromptEncoder + the vector index + a bit of ergonomics.
Given a query (as text OR pre-encoded vector), returns the top-K
matching Crystal entities with their cosine-similarity scores.

Two reasons this module exists instead of calling the vector index directly:
  1. Hydration: the index returns (crystal_id, score). The request
     pipeline wants full Crystal entities so it can pass them to the
     reader and quality gate.
  2. Graph-guided retrieval (future): flat top-K is path (a) from the
     scaffold docstring. Once CrystalEdge is being reinforced by live
     traffic, we add path (b) here — expand the top-K by following
     high-weight edges. That expansion logic belongs in the router,
     not in the vector store.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Union

import numpy as np

from ..encoding.executor import encode_async
from ..encoding.prompt_encoder import PromptEncoder
from ..models import Crystal

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_index import VectorIndex
    from ..models import Operator


class CrystalRouter:
    """Routes queries to candidate crystals via cosine similarity.

    Holds no state beyond its collaborators — a single router instance
    can be shared across requests.
    """

    def __init__(
        self,
        encoder: PromptEncoder,
        vector_index: "VectorIndex",
        metadata_store: "MetadataStore",
    ) -> None:
        self._encoder = encoder
        self._vector_index = vector_index
        self._metadata_store = metadata_store

    async def route(
        self,
        customer_id: str,
        query: Union[str, np.ndarray],
        k: int = 3,
        *,
        crystal_type: str = "customer:legacy",
        general_crystal_types: Optional[list[str]] = None,
        operator: Optional["Operator"] = None,
    ) -> list[tuple[Crystal, float]]:
        """Return up to k (crystal, similarity) pairs for this customer.

        Accepts either a text query (we'll encode it) or a pre-computed
        vector (the pipeline may reuse one vector across multiple steps).

        Phase 3 audit fix #8 (April 2026): `crystal_type` is now
        threaded through to the index's search_routing, which made it
        required. Default 'customer:legacy' matches the migration
        0012 seeded type so existing callers (the chat-completions
        gateway, test fixtures that don't care about type-scoping)
        keep working without changes. Callers that route into
        type-scoped banks (e.g. customer:medical_records) pass the
        explicit type.

        Empty bank → []. No error — retrieval simply doesn't contribute.
        """
        if isinstance(query, str):
            qvec = await encode_async(self._encoder, query)
        else:
            qvec = query

        if qvec is None or qvec.size == 0 or not np.any(qvec):
            # Zero or empty query vector — nothing meaningful to rank.
            return []

        id_score = await self._vector_index.search_routing(
            customer_id=customer_id,
            query_vector=qvec,
            k=k,
            crystal_type=crystal_type,
            general_crystal_types=general_crystal_types,
            operator=operator,
        )
        if not id_score:
            return []

        results: list[tuple[Crystal, float]] = []
        for crystal_id, score in id_score:
            crystal = await self._metadata_store.get_crystal(crystal_id)
            if crystal is None:
                continue
            # Allow customer-owned crystals AND general crystals (customer_id=None)
            if crystal.customer_id is not None and crystal.customer_id != customer_id:
                continue
            results.append((crystal, score))
        return results
