"""Recall — unbind + reverse-project + cleanup against a crystal's codebook.

This module is the Phase 2 read-side primitive. Pairs the Phase 1.1
write-side `add_pair_to_crystal` (which bundles `Σ bind(P_i, A_i)` into
`Crystal.summary_vector` and persists per-pair native answer embeddings
on `Fact.vector`).

Today the routing pipeline picks a top-1 crystal and injects its
`summary_text`. With many pairs per crystal that's wrong — we need to
extract WHICH pair matched. `recall_from_crystal` does that:

    Â = C.summary_vector ⊙ P_q              # unbind in HDC
    Â_native = Â @ P.T / d_hdc              # reverse-project to 768
    matched = argmax_i cosine(Â_native, Fact_i.vector)  # cleanup

If the best cosine clears `cleanup_threshold`, return the matched Fact;
otherwise return None (the routing was right but cleanup couldn't pin
a specific pair, signalling the pipeline should fall through rather
than inject noise).

THE MATH (validated April 2026 spikes, FAQ bank @ 30 pairs)
-----------------------------------------------------------
Bind is approximately self-inverse for elementwise multiplication of
unit-norm bipolar-projected vectors (CLAUDE.md Findings). For a
crystal C = Σ_i (P_i ⊙ A_i):

    C ⊙ P_q = (Σ_i P_i ⊙ A_i) ⊙ P_q
            = Σ_i (P_i ⊙ A_i ⊙ P_q)
            = Σ_i (A_i ⊙ (P_i ⊙ P_q))

When P_q matches some stored P_i*: P_i* ⊙ P_q ≈ 1-vector, so the i*
term recovers A_i* ⊙ 1 ≈ A_i*. Other terms become A_i ⊙ noise where
"noise" is a bipolar pseudo-random vector — they bundle into a
broadband noise floor that doesn't dominate any single direction.

Reverse-project: `Â @ P.T / d_hdc` lifts back to native space (the
gtr-t5-base 768-dim embedding regime that Fact.vector lives in). The
recovered native vector is noisy but cosine-close to the right Fact's
stored answer embedding. Cleanup is the nearest-neighbor lookup that
snaps it back to a clean stored vector.

WHY CLEANUP THRESHOLD MATTERS
-----------------------------
The recovered vector is always SOMETHING — argmax over the codebook
will always return SOMETHING. Without a threshold gate, recall on a
query whose prompt was never bound into the crystal would return a
random Fact with whatever-cosine-the-noise-happens-to-give. That's
worse than no recall: the pipeline would inject confidently-wrong
content.

Cleanup threshold is the noise-floor gate. Real recoveries from
stored pairs sit at cosine ~0.4-0.7 in the FAQ-bank validation
(test_30_pair_crystal_recovers_each_answer_via_unbind_cleanup);
pure crosstalk noise sits below 0.2. The default 0.3 (settings.
cleanup_threshold) is comfortably above noise and below typical
recovery. Per-crystal calibration is Phase 6.3 work — for now,
the global default applies.

WHEN RECALL RETURNS None
------------------------
  - Crystal has no codebook (legacy bank, no Facts).
  - Crystal.summary_vector is empty or wrong shape (pre-Phase-1
    crystal whose summary_vector was a single-encode of summary_text).
  - Best cosine in cleanup is below cleanup_threshold (true noise
    floor: routing brought us to the right crystal but the query
    isn't actually answered by any specific pair).
  - Encoder fingerprint mismatch (the crystal was written with a
    different encoder geometry; recovered vectors wouldn't be
    interpretable to the bind-v1 decoder OR to the codebook
    cosines, so we refuse rather than emit garbage).

In all those cases the pipeline's PERFECT branch falls through to
LOW_CONFIDENCE. The user still gets a useful answer — just from
upstream rather than from injected context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import structlog

from ..models import Crystal, Fact

if TYPE_CHECKING:
    from ..encoding.semantic import SemanticTextEncoder
    from ..infrastructure.metadata_store import MetadataStore
    from .chain_resolver import ChainResolver


logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RecalledFact:
    """The result of a successful recall.

    Carries everything the pipeline needs to make injection +
    cache-hit decisions on a per-pair basis (Phase 2 spec §2.2).

    Attributes:
        fact: the matched Fact entity. Includes claim_text (the prose
            to inject), answer_value (cache-hit canonical short
            answer, may be None), source_kind (voicing — advisory vs.
            imperative), pair_type (telemetry tag).
        score: cosine similarity between the cleaned recovered native
            vector and fact.vector. In [-1, 1]. Above the
            cleanup_threshold by construction (recall returns None
            when no Fact clears the threshold).
        candidates_examined: how many Facts were in the codebook for
            this recall. Surface for the inspector — a 50-pair
            crystal that always recalls the same Fact has different
            calibration concerns from one that switches Facts every
            query.
    """
    fact: Fact
    score: float
    candidates_examined: int
    recovery_ratio: float = 0.0  # Mode 9: ||proj|| / ||summ||


async def recall_from_crystal(
    crystal: Crystal,
    query_vector_hdc: np.ndarray,
    *,
    store: "MetadataStore",
    encoder: "SemanticTextEncoder",
    cleanup_threshold: Optional[float] = None,
    chain_resolver: Optional["ChainResolver"] = None,
    requesting_customer_id: Optional[str] = None,
) -> Optional[RecalledFact]:
    """Unbind + reverse-project + cleanup against the crystal's codebook.

    Phase 2 read-side primitive, Phase 3-extended for chain resolution.
    Companion to Phase 1.1's `add_pair_to_crystal` (which writes the
    bundle and codebook).

    Args:
        crystal: the routing decision's top-1 crystal. Must have a
            non-empty `summary_vector` and a populated
            `encoder_fingerprint` matching the supplied encoder.
            Pre-Phase-1.1 crystals (no fingerprint, summary_vector
            built from a single-encode of summary_text) are not
            recall-capable; the function returns None.
        query_vector_hdc: the user's query encoded to HDC space
            (10000-dim, unit-norm). Same vector the four-way
            classifier used for the routing decision; passed in
            rather than re-encoded so the function is decoupled
            from the encoder lifecycle.
        store: MetadataStore for loading the crystal's codebook
            (Fact rows). Required keyword-only.
        encoder: the SemanticTextEncoder whose .P matrix was used
            to write this crystal. Required keyword-only. The
            crystal.encoder_fingerprint is checked against
            encoder.fingerprint() — mismatch returns None (rather
            than ValueError) because routing already picked this
            crystal as a match and we'd rather log + skip than
            crash a request.
        cleanup_threshold: cosine cutoff. If None, falls back to
            settings.cleanup_threshold (default 0.3). Pass an
            explicit value for testing or for per-customer
            overrides. Phase 6.3 will introduce per-crystal
            cleanup_threshold via offline calibration.
        chain_resolver: optional ChainResolver (Phase 3). When
            provided AND `requesting_customer_id` is also provided,
            the resolver walks outgoing chains from the crystal and
            unions chained crystals' Facts into the cleanup
            codebook (subject to ACL). When None (or when
            requesting_customer_id is None), recall operates only
            on the source crystal's own codebook — same behavior as
            Phase 2. ACL violations are silent: chained Facts are
            absent from the codebook, no error is raised. See
            chain_resolver.py module docstring for the directionality
            and ACL resolution rules.
        requesting_customer_id: the customer whose query is in flight.
            Used by chain_resolver to evaluate read_codebook ACLs on
            chain targets. Required when chain_resolver is provided;
            chain extension is silently disabled when this is None
            even if the resolver is otherwise wired.

    Returns:
        RecalledFact on a successful match, or None on any of:
        - Empty / wrong-shape summary_vector
        - Empty codebook (including chain extension producing nothing)
        - Encoder fingerprint mismatch
        - Best cleanup cosine below cleanup_threshold

    Notes on the design:
        - Unlike synthesize_joint_statement (Phase 1, SPREAD branch),
          recall does NOT use the bind-v1 decoder. Cleanup matches
          the recovered native vector against stored Fact.vector
          entries by cosine; the matched Fact's claim_text is the
          injection text. The decoder is only invoked when there's
          no stored answer to retrieve (which doesn't happen in the
          recall flow — every Fact has a claim_text).
        - Chain extension is opt-in additive: callers that don't
          pass chain_resolver get exactly the Phase 2 behavior. A
          chained Fact that wins cleanup returns just like an own-
          codebook win — the RecalledFact does NOT distinguish
          chained-from-own (the matched Fact's `crystal_id` field
          tells you which crystal it came from if you care).
    """
    # Resolve threshold lazily — settings is process-wide, importing
    # at module load creates a circular dependency at startup.
    if cleanup_threshold is None:
        from ..config import get_settings
        cleanup_threshold = float(get_settings().cleanup_threshold)

    # Guard: empty / wrong-shape summary_vector. Pre-Phase-1.1 crystals
    # whose summary_vector was a single-encode of summary_text might
    # technically have the right dim (depending on encoder) but lack
    # the bundle structure recall expects. The encoder_fingerprint
    # check below catches that case rigorously; this is an early-out
    # for the obvious empty case.
    if not crystal.summary_vector:
        logger.debug(
            "recall.empty_summary_vector",
            crystal_id=crystal.id,
        )
        return None

    summary_vec = np.asarray(crystal.summary_vector, dtype=np.float32)
    d_hdc = summary_vec.shape[0]

    # Guard: encoder fingerprint mismatch. Bind-storage geometry is
    # encoder-specific (P matrix is seeded; different seed → different
    # geometry). Mixing encoders silently corrupts recovered vectors
    # — the matched Fact would be off-manifold relative to the
    # encoder's expected output distribution and cosine values would
    # be uncalibrated. Refuse rather than emit garbage.
    if crystal.encoder_fingerprint is not None:
        expected_fp = encoder.fingerprint()
        if crystal.encoder_fingerprint != expected_fp:
            logger.warning(
                "recall.encoder_fingerprint_mismatch",
                crystal_id=crystal.id,
                crystal_fingerprint=crystal.encoder_fingerprint,
                encoder_fingerprint=expected_fp,
                note=(
                    "Crystal was written with a different encoder "
                    "geometry than the one being used at recall. "
                    "Recovered vectors would not be interpretable "
                    "against this codebook; refusing to recall."
                ),
            )
            return None

    if query_vector_hdc.shape != (d_hdc,):
        logger.warning(
            "recall.query_dim_mismatch",
            crystal_id=crystal.id,
            crystal_dim=d_hdc,
            query_dim=query_vector_hdc.shape,
        )
        return None

    if encoder.P.shape[1] != d_hdc:
        logger.warning(
            "recall.encoder_p_dim_mismatch",
            crystal_id=crystal.id,
            crystal_dim=d_hdc,
            encoder_p_shape=tuple(encoder.P.shape),
        )
        return None

    # Step 1: unbind in HDC space. Elementwise multiply of the bundle
    # against the query's bind-key direction recovers (a noisy
    # approximation of) the answer that was bound to this prompt.
    #
    # Math note: bind's self-inverse property comes from bipolar ±1
    # vectors having pointwise inverse equal to themselves. Our
    # P-projected vectors are unit-norm continuous-valued, not strictly
    # bipolar — but they're close enough that A ⊙ B ⊙ B ≈ A holds
    # statistically across high-dim space (validated empirically on the
    # 30-pair FAQ-bank recall test). The "approximately" is what
    # cleanup is for: the recovered vector points roughly the right
    # direction, and the codebook lookup snaps it onto a stored answer.
    a_hdc = summary_vec * query_vector_hdc

    # Step 2: reverse-project to native dim. Same encoder.P used at
    # write time; transpose is the back-projection. Division by d_hdc
    # follows the research module's `KnowledgeCrystal.read` math
    # (HDC is unitary up to a scale factor of d_hdc; the back-projection
    # picks up that factor and we divide it out). Leaving the divide
    # off would produce a recovered vector with massive magnitude
    # that breaks cosine numerics later.
    a_native = (a_hdc @ encoder.P.T) / float(d_hdc)
    # a_native is shape (native_dim,), float32, NOT unit-norm.

    # Step 3: load the codebook. List ordered by created_at; the order
    # doesn't matter for the cosine search, but it's stable for
    # reproducibility / inspector display.
    facts = await store.list_facts_for_crystal(crystal.id)

    # Step 3b (Phase 3): chain extension. When a resolver and a
    # requesting customer are supplied, walk outgoing chains and union
    # ACL-permitted chained crystals' Facts into the candidate set.
    # When the resolver is None or requesting_customer_id is None,
    # this is a no-op — same behavior as Phase 2.
    #
    # Chain extension can produce a non-empty codebook even when the
    # source crystal's own codebook is empty (rare but possible: a
    # source crystal that exists only to route to a chain target).
    # The downstream emptiness check covers both cases.
    if chain_resolver is not None and requesting_customer_id is not None:
        try:
            extra_facts = await chain_resolver.resolve_extra_facts(
                source_crystal_id=crystal.id,
                requesting_customer_id=requesting_customer_id,
            )
        except Exception as e:
            # Chain resolution must not break recall. The user gets a
            # working result based on the source's own codebook; the
            # chain extension just doesn't fire. Log loudly so we
            # notice the failure mode in production.
            logger.error(
                "recall.chain_resolver_failed",
                crystal_id=crystal.id,
                error=str(e),
                error_type=type(e).__name__,
            )
            extra_facts = []
        if extra_facts:
            # Dedupe by Fact.id in case the chain resolver returns a
            # Fact that's also in the source crystal's own list. The
            # resolver dedupes among chained targets but not against
            # the source.
            own_ids = {f.id for f in facts}
            for fact in extra_facts:
                if fact.id in own_ids:
                    continue
                facts.append(fact)

    if not facts:
        logger.debug(
            "recall.empty_codebook",
            crystal_id=crystal.id,
        )
        return None

    # Filter Facts that have a usable codebook vector. A Fact with an
    # empty `vector` field is from a legacy import path that didn't
    # populate it (or a not-yet-implemented synthetic write). Skipping
    # them silently is correct here — they can't participate in
    # cleanup. If after filtering nothing remains, recall returns None.
    usable_facts: list[Fact] = []
    usable_vectors: list[np.ndarray] = []
    native_dim = encoder.native_dim
    for fact in facts:
        if not fact.vector:
            continue
        v = np.asarray(fact.vector, dtype=np.float32)
        if v.shape != (native_dim,):
            # Dim mismatch — Fact was written with a different encoder.
            # This shouldn't happen if encoder_fingerprint matched the
            # crystal's, but a Fact could have been seeded directly
            # via upsert under a different encoder. Skip it.
            logger.debug(
                "recall.fact_dim_mismatch",
                crystal_id=crystal.id,
                fact_id=fact.id,
                fact_dim=v.shape,
                expected_dim=native_dim,
            )
            continue
        usable_facts.append(fact)
        usable_vectors.append(v)

    if not usable_facts:
        logger.debug(
            "recall.no_usable_codebook_entries",
            crystal_id=crystal.id,
            total_facts=len(facts),
        )
        return None

    # Step 4: cleanup — nearest neighbor by cosine. Stack codebook into
    # a matrix for vectorized cosine. The recovered a_native is
    # NOT unit-norm; codebook vectors ARE unit-norm (encoder.encode_native
    # returns unit-norm); cosine = (a_native @ V.T) / ||a_native||.
    V = np.stack(usable_vectors, axis=0)  # (N, native_dim)
    a_norm = float(np.linalg.norm(a_native))
    if a_norm == 0.0:
        # Pathological — recovered vector is exactly zero. Almost
        # impossible in practice (would require summary * query to
        # cancel everywhere) but the divide would NaN.
        logger.warning(
            "recall.recovered_vector_zero",
            crystal_id=crystal.id,
        )
        return None

    # Mode 9 fix (May 2026): recovery ratio gate.
    #
    # ||a_projected|| / ||summary_vec|| measures how much energy the
    # unbind recovered relative to the crystal's bundle size. This
    # ratio normalizes out crystal-size variance and isolates the
    # recovery-quality signal:
    #   - On-domain queries (query bind-key aligns with a stored key)
    #     produce ratio ~1.3-1.6 on the BCB bank.
    #   - Off-domain queries (no alignment) produce ratio ~0.9-1.2.
    #
    # Cleanup cosine is insensitive to this signal because it
    # normalizes magnitude away (cosine = direction only). The ratio
    # IS the magnitude signal that cleanup discards.
    #
    # We compute the ratio from the PRE-divide vector (a_hdc @ P.T)
    # to avoid the /d_hdc scale factor. The summary_vec norm is the
    # crystal's bundle magnitude.
    #
    # Gate: if recovery_ratio < recovery_ratio_floor, the bank has no
    # usable signal for this query. Skip injection rather than inject
    # noise. Default floor from settings; 1.25 calibrated on BCB bank
    # diagnostic (diagnose_mode9_magnitudes.py, May 2026).
    summary_norm = float(np.linalg.norm(summary_vec))
    a_projected = a_hdc @ encoder.P.T  # before /d_hdc
    proj_norm = float(np.linalg.norm(a_projected))
    recovery_ratio = proj_norm / summary_norm if summary_norm > 0 else 0.0

    # Resolve the floor from settings (same lazy-import pattern as
    # cleanup_threshold to avoid circular imports at module load).
    from ..config import get_settings as _get_settings
    _s = _get_settings()
    recovery_ratio_floor = float(_s.recovery_ratio_floor)

    if recovery_ratio < recovery_ratio_floor:
        logger.debug(
            "recall.below_recovery_ratio_floor",
            crystal_id=crystal.id,
            recovery_ratio=recovery_ratio,
            recovery_ratio_floor=recovery_ratio_floor,
            proj_norm=proj_norm,
            summary_norm=summary_norm,
            note=(
                "Recovery ratio below floor — the unbind did not "
                "recover enough energy relative to the bundle. "
                "Skipping injection to avoid noise."
            ),
        )
        return None

    similarities = (V @ a_native) / a_norm
    # similarities is (N,), values in [-1, 1].

    best_idx = int(np.argmax(similarities))
    best_score = float(similarities[best_idx])

    if best_score < cleanup_threshold:
        # Below noise floor — the routing decision brought us to this
        # crystal, but no specific pair stored here actually answers
        # the query. Pipeline falls through to LOW_CONFIDENCE.
        logger.debug(
            "recall.below_cleanup_threshold",
            crystal_id=crystal.id,
            best_score=best_score,
            cleanup_threshold=cleanup_threshold,
            candidates_examined=len(usable_facts),
        )
        return None

    matched = usable_facts[best_idx]
    logger.debug(
        "recall.matched",
        crystal_id=crystal.id,
        fact_id=matched.id,
        pair_type=matched.pair_type,
        source_kind=matched.source_kind,
        has_answer_value=matched.answer_value is not None,
        score=best_score,
        candidates_examined=len(usable_facts),
    )
    return RecalledFact(
        fact=matched,
        score=best_score,
        candidates_examined=len(usable_facts),
        recovery_ratio=recovery_ratio,
    )
