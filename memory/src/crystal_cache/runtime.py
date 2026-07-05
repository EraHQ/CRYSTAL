"""Shared runtime bootstrap for the API process and the worker process.

Both `crystal_cache.app` (the FastAPI lifespan) and
`crystal_cache.workers.__main__` (the standalone worker entrypoint, run as
`python -m crystal_cache.workers`) build the SAME core dependencies here so
the two entrypoints can't drift (A.5 / WS E, decision D3).

What lives here is the intersection the two processes share: the metadata
store (+ its process singleton), the text encoder, both in-memory vector
stores, the schema loader (+ its singleton), and the metacognition worker's
Anthropic client. API-only concerns
(decoder loader, decomposer, shadow evaluator, Mem0, the SPA mount) stay in
the lifespan.

Deliberately imports NO FastAPI app/router code — importing this module must
not construct the web app, or the worker process would drag the whole HTTP
surface in with it.

Schema is owned by Alembic migrations (`alembic upgrade head`), exactly as
for the API process; nothing here creates tables. In a container the
entrypoint runs migrations before either process starts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import structlog

from .config import settings
from .dsl.schema.loader import SchemaLoader, set_schema_loader
from .encoding import build_text_encoder
from .infrastructure import MetadataStore, VectorStore
from .infrastructure.fact_vector_store import FactVectorStore
from .infrastructure.vector_index import build_vector_index
from .infrastructure.metadata_store import set_metadata_store

logger = structlog.get_logger(__name__)


@dataclass
class CoreRuntime:
    """The dependency bundle shared by the API and worker processes."""

    store: MetadataStore
    encoder: Any
    vector_store: VectorStore
    fact_vector_store: FactVectorStore
    vector_index: Any
    schema_loader: SchemaLoader


async def build_core_runtime(
    store: Optional[MetadataStore] = None,
) -> CoreRuntime:
    """Construct the dependencies shared by the API and worker processes.

    Builds (and registers the process singletons for) the metadata store,
    the text encoder, both vector stores, and the schema loader. The caller
    owns the event loop and the store's lifecycle (dispose on shutdown).

    Does NOT create tables or seed the registry — schema and the legacy
    crystal_type rows are both owned by Alembic migrations, matching the API
    lifespan's behavior. A fresh database therefore needs `alembic upgrade
    head` first (the container entrypoint handles this).
    """
    if store is None:
        store = MetadataStore()
    set_metadata_store(store)

    encoder = build_text_encoder()
    vector_store = VectorStore(store=store)
    fact_vector_store = FactVectorStore(store=store)
    vector_index = build_vector_index(
        backend=settings.vector_backend,
        fact_store=fact_vector_store,
        vector_store=vector_store,
        metadata_store=store,
        qdrant_url=settings.qdrant_url,
        qdrant_location=settings.qdrant_location,
        qdrant_collection=settings.qdrant_collection,
        qdrant_routing_collection=settings.qdrant_routing_collection,
        qdrant_routing_oversampling=settings.qdrant_routing_oversampling,
    )

    schema_loader = SchemaLoader(metadata_store=store)
    set_schema_loader(schema_loader)

    return CoreRuntime(
        store=store,
        encoder=encoder,
        vector_store=vector_store,
        fact_vector_store=fact_vector_store,
        vector_index=vector_index,
        schema_loader=schema_loader,
    )
