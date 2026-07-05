"""Citation grounding — Growth G1b.

A citation is only trustworthy (and only G4-creditable) if the cited crystal
actually supports the claim it's attached to. This module is the cheap path:
embed the claim span and the cited source in the encoder's native space and
take their cosine. A claim that doesn't clear the threshold is a SPURIOUS
citation — the model attributed a statement to a source that doesn't support
it — and is dropped (recorded with grounded=False for telemetry, never paid).

Cheap by design: cosine between two encodes, no LLM. A future expensive path
(entailment, reusing the verify-loop's grounding machinery) can layer on top
for high-value or contested citations.

The pure pieces — span extraction, marker rewriting — live in citations.py;
this module is the encoder-backed async layer (it imports the executor so the
encodes stay off the event loop, per the no-component-starves-another rule).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import structlog

from ..encoding.executor import encode_native_async
from .citations import CitationSource, extract_claim_span

logger = structlog.get_logger(__name__)


# Grounding threshold for the cheap cosine path. Calibrated for the semantic
# encoder's native space, where on-topic claim↔source pairs land at cosine
# ~0.4–0.6 and off-topic at ~0.05–0.15 (the same regime the retrieval
# thresholds note). 0.25 sits above the noise floor and below typical genuine
# support — it drops clearly-spurious citations while keeping real ones.
# Overridable per call; may promote to a setting if it needs per-customer
# tuning.
CITATION_GROUNDING_THRESHOLD = 0.25


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


async def ground_citations(
    encoder: Any,
    response_text: str,
    cited: list[tuple[CitationSource, str]],
    *,
    threshold: float = CITATION_GROUNDING_THRESHOLD,
) -> list[dict[str, Any]]:
    """Grounding-check each cited source against its supporting content.

    `cited` is a list of (source, source_text) pairs — the source's injected
    content. (G1 v1 cites one crystal, so every pair shares the same
    source_text; the signature already generalizes to per-source content for
    multi-source later.) For each, extract the claim span the handle was
    attached to and cosine it against the source in the encoder's native
    space.

    Returns one dict per cited source:
      {source, claim_span, grounding_score, grounded}
    A missing span or empty source content grounds to 0.0 / False rather than
    raising — grounding is advisory, never fatal to the response.
    """
    results: list[dict[str, Any]] = []
    for source, source_text in cited:
        span = extract_claim_span(response_text, source.handle)
        score = 0.0
        if span and source_text:
            try:
                claim_vec = await encode_native_async(encoder, span)
                source_vec = await encode_native_async(encoder, source_text)
                score = _cosine(claim_vec, source_vec)
            except Exception as e:
                logger.warning(
                    "citations.grounding_encode_failed",
                    handle=source.handle,
                    crystal_id=source.crystal_id,
                    error=str(e),
                )
                score = 0.0
        results.append(
            {
                "source": source,
                "claim_span": span,
                "grounding_score": score,
                "grounded": score >= threshold,
            }
        )
    return results


async def ground_sources_against_answer(
    encoder: Any,
    answer_text: str,
    sources: list[tuple[CitationSource, str]],
    *,
    threshold: float = CITATION_GROUNDING_THRESHOLD,
) -> list[dict[str, Any]]:
    """Ground surfaced sources against the WHOLE answer (G1, agent path).

    The agent has no ``[[cc:N]]`` markers to anchor a claim span on — it
    surfaces sources through its retrieval tools rather than the model citing
    them inline (CC-D11 = grounding-based implicit credit). So each source is
    scored against the ENTIRE answer instead of a marker-delimited span: cosine
    the answer against each source's content in the encoder's native space. The
    answer is encoded ONCE and reused across sources.

    `sources` is a list of (CitationSource, source_text) — the surfaced crystal
    and its representative content. Returns one dict per source:
    {source, claim_span, grounding_score, grounded}; claim_span is "" (there is
    no span). A missing answer or source content grounds to 0.0 / False rather
    than raising — grounding is advisory, never fatal to the response.

    NOTE: the 0.25 threshold was calibrated for claim-span↔source pairs;
    whole-answer↔source cosines sit in a broader regime and this may want its
    own calibration once eyeballed on live runs.
    """
    if not answer_text or not sources:
        return [
            {"source": s, "claim_span": "", "grounding_score": 0.0, "grounded": False}
            for s, _ in sources
        ]
    try:
        answer_vec = await encode_native_async(encoder, answer_text)
    except Exception as e:
        logger.warning("citations.answer_encode_failed", error=str(e))
        return [
            {"source": s, "claim_span": "", "grounding_score": 0.0, "grounded": False}
            for s, _ in sources
        ]
    results: list[dict[str, Any]] = []
    for source, source_text in sources:
        score = 0.0
        if source_text:
            try:
                source_vec = await encode_native_async(encoder, source_text)
                score = _cosine(answer_vec, source_vec)
            except Exception as e:
                logger.warning(
                    "citations.source_encode_failed",
                    crystal_id=source.crystal_id, error=str(e),
                )
                score = 0.0
        results.append(
            {
                "source": source,
                "claim_span": "",
                "grounding_score": score,
                "grounded": score >= threshold,
            }
        )
    return results
