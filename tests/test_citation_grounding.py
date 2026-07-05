"""Growth G1b — citation grounding (retrieval/citation_grounding.py).

ground_citations embeds each cited claim span + its source in the encoder's
native space and keeps only citations whose cosine clears the threshold. A
deterministic keyword stub encoder stands in for the real encoder: claim and
source on the same topic → cosine 1 (grounded); different topics → cosine 0
(spurious).
"""
from __future__ import annotations

import numpy as np

from crystal_cache.retrieval.citations import CitationSource
from crystal_cache.retrieval.citation_grounding import ground_citations


class _KeywordEncoder:
    """Native vectors keyed on a topic word — same topic → cosine 1, else 0."""

    def encode_native(self, text: str) -> np.ndarray:
        t = (text or "").lower()
        if "director" in t:
            return np.array([1.0, 0.0, 0.0])
        if "budget" in t:
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])


async def test_ground_citations_splits_grounded_and_spurious():
    enc = _KeywordEncoder()
    response = "The director is Jane Doe [[cc:1]]. The budget was huge [[cc:2]]."
    src1 = CitationSource(handle="1", crystal_id="c1")
    src2 = CitationSource(handle="2", crystal_id="c2")
    # Both cite the SAME source content (about the director).
    source_text = "Director: Jane Doe"

    results = await ground_citations(
        enc,
        response,
        [(src1, source_text), (src2, source_text)],
        threshold=0.5,
    )
    by_handle = {r["source"].handle: r for r in results}

    # handle 1: "The director is Jane Doe" vs a director source → grounded.
    assert by_handle["1"]["grounded"] is True
    assert by_handle["1"]["grounding_score"] == 1.0
    # handle 2: "The budget was huge" vs the same director source → spurious.
    assert by_handle["2"]["grounded"] is False
    assert by_handle["2"]["grounding_score"] == 0.0
    assert by_handle["2"]["claim_span"] == "The budget was huge"


async def test_ground_citations_empty_source_not_grounded():
    enc = _KeywordEncoder()
    src = CitationSource(handle="1", crystal_id="c1")
    # No source content → grounds to 0.0 / False without encoding or raising.
    results = await ground_citations(
        enc, "Some claim [[cc:1]].", [(src, "")], threshold=0.25
    )
    assert results[0]["grounded"] is False
    assert results[0]["grounding_score"] == 0.0
