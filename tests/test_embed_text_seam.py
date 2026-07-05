"""embed_text seam — index a fact by text other than its stored answer.

Locks the D1 decision for code descriptions (June 2026, CRYS): add_pair_to_crystal
and add_pair_for_customer take an optional embed_text. When set, the fact's SEARCH
vector — Fact.vector, the thing FactVectorStore ranks on — encodes embed_text,
while claim_text still stores (and ContentRouter still returns) answer_text.
Default None preserves the historical behavior exactly: index == stored answer.

This is the mechanism that lets a code chunk be indexed by a natural-language
description of what it does, while still returning the verbatim body — fixing the
"NL query matched against raw code" failure without a code-aware encoder.

Integration tests against the in-memory store + deterministic semantic encoder
stub (conftest). The stub's encode_native is a pure function of the text, so the
fact's stored vector is checkable by cosine against encode_native(<the text we
expect was indexed>). Distinct texts map to independent ~random 768-dim
directions, so a wrong index shows up as a near-zero cosine.
"""
from __future__ import annotations

import numpy as np
import pytest

# A verbatim code body and a functional description of it whose wording
# deliberately shares little vocabulary with the identifiers in the code.
CODE = (
    "async def encode_native_async(encoder, text):\n"
    "    loop = asyncio.get_running_loop()\n"
    "    return await loop.run_in_executor(None, encoder.encode_native, text)"
)
DESC = (
    "Runs the text encoder on a background thread so embedding a query does not "
    "block the event loop while other work proceeds."
)
KEY = "Code|src/crystal_cache/encoding/executor.py|encode_native_async"


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b / (na * nb))


@pytest.mark.asyncio
async def test_default_indexes_the_answer(store, semantic_encoder_stub, vector_store):
    """No embed_text: the search vector encodes answer_text — today's behavior."""
    _, fact = await store.add_pair_for_customer(
        customer_id="cust-embed-1", prompt_text=KEY, answer_text=CODE,
        pair_type="content_chunk", encoder=semantic_encoder_stub,
        vector_store=vector_store, crystal_type="customer:legacy",
    )
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(CODE)) > 0.999
    assert fact.claim_text == CODE  # payload is the code


@pytest.mark.asyncio
async def test_embed_text_indexes_description_not_code(store, semantic_encoder_stub, vector_store):
    """With embed_text: the search vector encodes the DESCRIPTION; body stays the code."""
    _, fact = await store.add_pair_for_customer(
        customer_id="cust-embed-2", prompt_text=KEY, answer_text=CODE,
        pair_type="content_chunk", encoder=semantic_encoder_stub,
        vector_store=vector_store, crystal_type="customer:legacy",
        embed_text=DESC,
    )
    # indexed by the description ...
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(DESC)) > 0.999
    # ... NOT by the code (independent stub directions -> near zero) ...
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(CODE)) < 0.2
    # ... but the returned body is still the verbatim code.
    assert fact.claim_text == CODE


@pytest.mark.asyncio
async def test_embed_text_threads_through_normal_path(store, semantic_encoder_stub, vector_store):
    """The non-content_chunk (bond/spawn) write path also honors embed_text."""
    _, fact = await store.add_pair_for_customer(
        customer_id="cust-embed-3", prompt_text="what does X do",
        answer_text="X does the thing.", pair_type="question_answer",
        encoder=semantic_encoder_stub, vector_store=vector_store,
        crystal_type="customer:legacy", embed_text=DESC,
    )
    assert _cos(fact.vector, semantic_encoder_stub.encode_native(DESC)) > 0.999
    assert fact.claim_text == "X does the thing."


@pytest.mark.asyncio
async def test_embed_text_none_equals_omitted(store, semantic_encoder_stub, vector_store):
    """embed_text=None is identical to omitting it — the default-off guarantee."""
    _, omitted = await store.add_pair_for_customer(
        customer_id="cust-embed-4a", prompt_text=KEY, answer_text=CODE,
        pair_type="content_chunk", encoder=semantic_encoder_stub,
        vector_store=vector_store, crystal_type="customer:legacy",
    )
    _, explicit_none = await store.add_pair_for_customer(
        customer_id="cust-embed-4b", prompt_text=KEY, answer_text=CODE,
        pair_type="content_chunk", encoder=semantic_encoder_stub,
        vector_store=vector_store, crystal_type="customer:legacy",
        embed_text=None,
    )
    assert _cos(omitted.vector, explicit_none.vector) > 0.9999
