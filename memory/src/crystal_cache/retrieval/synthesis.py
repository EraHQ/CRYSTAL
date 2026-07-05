"""Synthesis — combine two crystals' answer embeddings, decode the result.

This module is the SPREAD-decision branch of the routing decision table
(see CLAUDE.md "When to use which decoder"). When M2 routing returns
two crystals both above the noise floor with a margin in the spread
band, the right answer is "the user asked about both — synthesize a
joint statement that mentions both."

THE MATH (validated April 2026, Findings 13 + 14 + 15)
-------------------------------------------------------
Given two crystals top1, top2 with stored native answer embeddings
(768-dim, unit-norm, from gtr-t5-base):

  proj_a = answer_a_native @ encoder.P     # (10000,) HDC
  proj_b = answer_b_native @ encoder.P     # (10000,)
  combined = proj_a + proj_b               # elementwise bundle (Finding 15)
  recovered_native = combined @ encoder.P.T  # (768,) back to embedding space

  # Optional cleanup (Finding 14, default alpha=0.0 for bundle):
  Q = orthonormal basis of span(stored answer-embeddings)
  cleaned = (1 - alpha) * recovered + alpha * (Q @ Q.T @ recovered)

  joint_text = bind_v1.decode(cleaned_unit_normed)

WHY BUNDLE INSTEAD OF BIND (Finding 15, the headline result)
------------------------------------------------------------
We previously used elementwise multiplication (bind, `proj_a * proj_b`)
based on Finding 12's reasoning that "summing has loss in dimensions,
we need to multiply/bind the two answers." That reasoning was correct
for the STORAGE primitive (bundling many gratings into one crystal
loses info on dimensional collisions, bind preserves both signals
jointly). It was WRONG for the SYNTHESIS primitive at query time.

The HDC literature is clear: bind is for INTERSECTION queries
("find the thing that has both X AND Y"). Bundle is for DISJUNCTION /
"tell me about both" queries — which is exactly what SPREAD is. We
picked the wrong operator and were trying to fix the symptoms (loops,
"Microsoft Entra ID Microsoft Entra ID Microsoft Entra ID") with cleanup.

The diagnostic that flipped the decision: manifold fraction.
  bind-recovered:   0.11–0.19 (89% off-manifold noise)
  bundle-recovered: 0.965–0.966 (96.5% on-manifold)

Bundle keeps the recovered vector on the bank's manifold; bind
launches it into a part of 768-space the decoder doesn't know how to
read. Bundle wins on 6/6 straddle queries, with the canary SOC2/SSO
loop collapsing entirely:

  bind+0.75    : "...for Okta, Microsoft Entra ID, and OIDC is available
                  in both Pro and Enterprise plans, with SOC 2 Type II
                  certification." (trailing repetition)
  bundle+0.0   : "Lumora's SOC 2 Type II certification is available on
                  Pro and Enterprise plans." (clean, plan-binding correct)

See Finding 15 in CLAUDE.md for the full sweep data.

WHY ALPHA DEFAULTS TO 0.0 NOW
-----------------------------
Bundle vectors are already 96.5% on-manifold. Projection onto bank-span
has nothing to remove. Cleanup is a no-op for the bundle path; we keep
the wiring (alpha + bank_natives kwargs, all the helpers) so future
work can revisit if the failure modes shift, but the production default
is alpha=0.0. The kwargs and helpers also remain useful if anyone wants
to test the bind path again or build new operators on top.

WHAT SYNTHESIS PRODUCES
-----------------------
A joint statement string that mentions both anchor topics. Quality
caveats (preserved verbatim from CLAUDE.md):
  - Specific numbers and plan-bindings can still be wrong (Starter
    getting Pro's webhooks, etc.). bundle is BETTER than bind on this
    axis but not perfect — bind-v1's fact-binding limits from Finding 13
    still apply. Acceptable for hint signal; not for final answer.
  - Use this as a HINT injected alongside the raw FAQ texts. The LLM
    downstream synthesizes the actual answer using the original query
    + both raw FAQs + this joint hint.
  - Bundle outputs occasionally read as instructions ("reset your
    Lumora Slack app from Settings...") because the recovered vector
    lands in a region of the decoder's distribution that frames things
    imperatively. The synthesis-hint framing in the injection prefix
    is load-bearing for guarding against this.

WHEN SYNTHESIS RETURNS None
---------------------------
  - DecoderLoader is None (decoder disabled via CC_ENABLE_DECODER unset)
  - bind-v1 specifically isn't loaded
  - Either crystal lacks answer_embedding_native (legacy bank, hash-encoded)
  - The decoder errors during generation (rare; logged loudly)
  - Either embedding has wrong shape

In all these cases the pipeline should fall back to standard top-1
injection. The synthesis path is a quality-improvement signal, not a
correctness gate. If it can't run, the request still gets answered.

DECODER NAMING NOTE
-------------------
We kept the decoder filename and load helper as `bind_v1` even though
the synthesis path now uses bundle vectors. The decoder weights ARE
the same artifact — bind-v1 was trained on text + bind pairs and
generalizes well to bundle inputs. Renaming the artifact would require
a checkpoint rename and config migration for marginal clarity benefit.
The naming is preserved as a historical artifact; what matters is the
weights file, not the variable name.

A follow-up experiment worth running: compare text-v1 vs bind-v1 on
bundle inputs. text-v1 may decode bundle slightly worse (only saw
single-anchor vectors in training); bind-v1 may decode bundle slightly
better (saw bind pairs that taught it joint-vector structure
generally). Either way, bind-v1 is what's loaded today and works.

WHY synthesis lives in retrieval/, not infrastructure/
-------------------------------------------------------
infrastructure/decoder_loader.py is just a model-load utility. It
holds weights and runs forward passes. synthesis.py is retrieval
business logic — it knows about Crystal entities, the HDC algebra,
and the routing decision table.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import structlog

from ..encoding.semantic import SemanticTextEncoder
from ..infrastructure.decoder_loader import DecoderLoader
from ..models import Crystal


logger = structlog.get_logger(__name__)


def synthesize_joint_statement(
    top1: Crystal,
    top2: Crystal,
    *,
    encoder: SemanticTextEncoder,
    decoder_loader: DecoderLoader,
    bank_natives: Optional[Sequence[Sequence[float]]] = None,
    cleanup_alpha: float = 0.0,
) -> Optional[str]:
    """Synthesize a joint statement that mentions both anchors.

    Args:
        top1, top2: the two crystals selected by the SPREAD branch of
            classify_routing. Both must have answer_embedding_native
            populated (768-dim float list). If either is missing,
            returns None.
        encoder: the process's SemanticTextEncoder. We need its `P`
            matrix to project native embeddings into HDC space and
            back. The encoder must be the SAME instance that produced
            answer_embedding_native at write time — different P matrices
            are different geometries and the math doesn't work across
            them.
        decoder_loader: the process's DecoderLoader. Must have bind_v1
            loaded.
        bank_natives: full list of stored answer_embedding_native vectors
            from the customer's bank. Used to build the cleanup manifold
            basis. With the bundle primitive (Finding 15) cleanup is a
            no-op at default alpha=0; the kwarg is retained for
            experimentation and for cases where bind path is reactivated.
            If None or empty, cleanup is skipped.
        cleanup_alpha: blend weight in [0, 1]. 0.0 = no cleanup; 1.0 =
            full projection onto bank-span. Default 0.0 — bundle vectors
            are already on-manifold (Finding 15), so cleanup has nothing
            to remove. The pipeline reads from
            settings.synthesis_cleanup_alpha (also 0.0 by default).

    Returns:
        A joint statement string on success, or None if synthesis
        couldn't run.
    """
    # Guard rails — every "skip synthesis" branch.
    if decoder_loader.bind_v1 is None:
        logger.debug(
            "synthesis.skipped",
            reason="bind_v1 not loaded",
            top1=top1.id, top2=top2.id,
        )
        return None
    if not top1.answer_embedding_native:
        logger.debug(
            "synthesis.skipped",
            reason="top1 missing answer_embedding_native",
            crystal_id=top1.id,
        )
        return None
    if not top2.answer_embedding_native:
        logger.debug(
            "synthesis.skipped",
            reason="top2 missing answer_embedding_native",
            crystal_id=top2.id,
        )
        return None

    try:
        recovered = _synthesize_combined(
            answer_a_native=top1.answer_embedding_native,
            answer_b_native=top2.answer_embedding_native,
            encoder_P=encoder.P,
        )
    except Exception as e:
        logger.error(
            "synthesis.combine_failed",
            top1=top1.id, top2=top2.id,
            error=str(e), error_type=type(e).__name__,
        )
        return None

    # Cleanup step. With bundle as the primitive (Finding 15) the
    # default alpha=0.0 means this is a no-op for production traffic.
    # We keep the wiring for experimentation and for the (unlikely)
    # case that the bind path is reactivated.
    cleaned = recovered
    if cleanup_alpha > 0.0 and bank_natives:
        try:
            Q = _build_manifold_basis(bank_natives, native_dim=recovered.shape[0])
            if Q is not None:
                cleaned = _apply_cleanup(recovered, Q, cleanup_alpha)
                logger.debug(
                    "synthesis.cleanup_applied",
                    top1=top1.id, top2=top2.id,
                    alpha=cleanup_alpha,
                    manifold_rank=int(Q.shape[1]),
                )
        except Exception as e:
            logger.warning(
                "synthesis.cleanup_failed",
                top1=top1.id, top2=top2.id,
                error=str(e), error_type=type(e).__name__,
            )
            cleaned = recovered

    try:
        decoded = decoder_loader.decode_bind(cleaned)
    except Exception as e:
        # Decoder errors shouldn't fail the user's request. Log loudly
        # and let the caller fall back to non-synthesis injection.
        logger.error(
            "synthesis.decode_failed",
            top1=top1.id, top2=top2.id,
            error=str(e), error_type=type(e).__name__,
        )
        return None

    if decoded is None or not decoded.strip():
        logger.warning(
            "synthesis.empty_decode",
            top1=top1.id, top2=top2.id,
        )
        return None

    logger.debug(
        "synthesis.ok",
        top1=top1.id, top2=top2.id,
        decoded_chars=len(decoded),
    )
    return decoded


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------


def _synthesize_combined(
    answer_a_native: list[float],
    answer_b_native: list[float],
    encoder_P: np.ndarray,
) -> np.ndarray:
    """The HDC combine math. Pure function; testable in isolation.

    Steps:
      1. Native (768) inputs → HDC (10000) via @ P.
      2. Elementwise BUNDLE in HDC space (Finding 15: bundle, not bind).
      3. Reverse-project back to native via @ P.T.

    The decoder will renormalize the result before generating, so we
    don't bother L2-normalizing here. Cleanup happens in the caller
    (synthesize_joint_statement), keeping this function the pure HDC
    primitive.

    Returns:
        np.ndarray of shape (native_dim,) ready for cleanup + decoding.
    """
    a = np.asarray(answer_a_native, dtype=np.float32)
    b = np.asarray(answer_b_native, dtype=np.float32)

    if a.shape != b.shape:
        raise ValueError(
            f"native embedding dim mismatch: top1={a.shape}, top2={b.shape}"
        )
    if a.shape[0] != encoder_P.shape[0]:
        raise ValueError(
            f"native embedding dim {a.shape[0]} doesn't match "
            f"encoder.P input dim {encoder_P.shape[0]}"
        )

    # HDC lift. encoder.P is (native_dim, d_hdc), so a @ P → (d_hdc,).
    proj_a = a @ encoder_P
    proj_b = b @ encoder_P

    # The bundle (Finding 15). Elementwise addition of two HDC vectors
    # produces a third vector that has high cosine similarity to BOTH
    # inputs — geometrically it points to a region centered between the
    # two anchors. The HDC literature calls this "set membership" or
    # "disjunction": the result represents "either of these things,"
    # which is what SPREAD synthesis needs. Bundle stays on the bank's
    # manifold (~96% on-manifold in our spike) where bind launched into
    # off-manifold noise (~15% on-manifold).
    combined = proj_a + proj_b

    # Reverse-projection back to native dim. encoder.P.T is (d_hdc, native).
    # No division by D — the decoder normalizes before generating.
    recovered_native = combined @ encoder_P.T

    return recovered_native.astype(np.float32)


def _build_manifold_basis(
    bank_natives: Sequence[Sequence[float]],
    *,
    native_dim: int,
) -> Optional[np.ndarray]:
    """Return Q, an orthonormal basis for the row-span of the bank.

    Q has shape (native_dim, rank) where rank ≤ min(n_crystals, native_dim).

    With the bundle primitive (Finding 15) this is rarely useful in
    production — bundle stays on-manifold so projection has nothing to
    remove. Retained for experimentation and for the bind path if it's
    ever reactivated.

    Args:
        bank_natives: the customer's full set of stored answer_embedding_native
            vectors. Pass them all — duplicates and irrelevant FAQs are
            fine, QR collapses them.
        native_dim: expected dimensionality (768 for gtr-t5-base). Bank
            entries that don't match this dim are silently dropped.

    Returns:
        Q array of shape (native_dim, rank), or None if no usable bank
        entries were found.
    """
    valid: list[np.ndarray] = []
    for entry in bank_natives:
        if entry is None:
            continue
        v = np.asarray(entry, dtype=np.float32)
        if v.shape == (native_dim,):
            valid.append(v)
    if not valid:
        return None

    S = np.stack(valid, axis=0)
    Q, _ = np.linalg.qr(S.T)
    return Q.astype(np.float32)


def _apply_cleanup(
    recovered: np.ndarray,
    Q: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Alpha-blend the recovered vector with its on-manifold projection.

    alpha=0.0 → recovered (pass-through, no cleanup; default for bundle)
    alpha=1.0 → Q @ (Q.T @ recovered) (full projection onto bank-span)

    Pure function; alpha is clamped to [0, 1] silently for safety
    against misconfigured settings.
    """
    a = max(0.0, min(1.0, float(alpha)))
    if a == 0.0:
        return recovered.astype(np.float32)
    on_manifold = Q @ (Q.T @ recovered)
    blended = (1.0 - a) * recovered + a * on_manifold
    return blended.astype(np.float32)
