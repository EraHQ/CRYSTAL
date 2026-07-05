"""Three-axis bonder for write-time crystal bonding decisions.

Phase 6.3 follow-up #2, May 2026.
Scope: docs/PHASE_6_3_FOLLOWUP_2_DECOMPOSER_BONDING_SCOPE.md
Empirical motivation: docs/PHASE_6_3_FAQ_REIMPORT_FINDING.md
S7 attempt notes: docs/PHASE_6_3_FOLLOWUP_2_FAQ_REIMPORT_NOTES.md

THE PROBLEM
-----------
`add_pair_for_customer` decides whether an incoming (prompt, answer)
pair should bond into the customer's existing top-1 crystal or spawn
a fresh crystal. The pre-followup-2 mechanism was a single-axis cosine
threshold check: bond if `top1_score >= bond_threshold`, else spawn.

On the FAQ corpus (12 FAQs × 3 paraphrases = 36 pairs) with the default
0.65 threshold, this collapses to 1 crystal containing all 36 facts —
every paraphrase bonds into the first FAQ's crystal because the
encoder's domain-clustering bias pushes every Lumora-related prompt
into the 0.65-0.95 cosine range. No single threshold cleanly separates
intra-trio paraphrases (median 0.86) from inter-FAQ neighbors (median
0.70) on this encoder + corpus.

THE FIX
-------
Three independent signals, applied in cost order:

  Axis 1 (cheap)    : routing-vector cosine vs candidate's routing_vector
  Axis 2 (moderate) : per-fact-prompt cosine vs candidate's stored Facts
  Axis 3 (expensive): decomposer payload TEXT cosine

Most pairs short-circuit at axis 1 (clearly above T_high or below
T_low). The gray zone is where the discriminative work happens — first
the per-fact-prompt check (already-encoded text comparison, cheap) and
then, only if that doesn't fire, the decomposer LLM call.

DECISION RULE
-------------
  if candidate_score >= T_high:                  bond ("cosine_clear")
  elif candidate_score < T_low:                  spawn ("cosine_below_floor")
  else (gray zone):
      if max_per_fact_cosine >= T_fact:          bond ("cosine_gray_fact_match")
      else:
          if no payloads available:              spawn ("cosine_gray_no_payload")
          elif payload_agreement >= T_payload:   bond ("cosine_gray_payload_agree")
          else:                                  spawn ("cosine_gray_payload_disagree")

The "spawn fresh on uncertainty" default reflects the conservative-spawn
principle from Lili Hypothesis 4 (the architecture only works if new
compartments are created aggressively enough). Note: this is design
intent borrowed from Lili, not a measured Lili result. We adopt the
principle but the threshold values are unvalidated — S7 measures them.

PAYLOAD AGREEMENT MECHANISM (revised May 3, 2026)
-------------------------------------------------
Originally axis 3 used CONCEPT-HV cosine: render each payload to a
DSL hypervector via `from_decomposer_output()`, compute bipolar cosine.
S7 attempt #2 falsified this primitive on small-model decomposer
output:

  Three paraphrases of one FAQ ("does Lumora support SSO" / "SAML
  login for Lumora" / "can I use Okta with Lumora") under Qwen 2.5
  7B at temperature=0 produced topic values "integration",
  "saml_configuration", "integration" — the unstable middle topic
  hashed to an orthogonal concept-HV, dropping payload_agreement
  below T_payload=0.5. The bonder correctly applied its rule but
  the rule's input was noisy.

Concept-HV's exact-string-match property means "saml_configuration"
and "integration" are treated as fully orthogonal even though they're
semantically close. Small models drift on topic phrasing. The result:
spawn-when-should-bond, false negatives compounding across the bank.

The replacement is TEXT cosine using the production semantic encoder
(gtr-t5-base via `encoder.encode_native`). Payloads render to strings
("intent=X topic=Y domain=Z"); encode both; cosine. The encoder
understands that "saml_configuration" and "integration" are
semantically close — exactly the property concept-HV lacks.

Why this is the right primitive (and concept-HV was the wrong one):
  1. The bonder is doing semantic similarity over LLM-emitted JSON,
     not role-binding composition. Concept-HV was designed for the
     latter (DSL config lookups); text-cosine is right for the former.
  2. Same encoder used by axis 1 (routing_vector) and axis 2
     (per-fact-prompt). Single similarity contract across all three
     axes.
  3. Tolerates the small-model drift S7 attempt #2 surfaced. Empirical
     fix for an empirical problem.

The scope doc anticipated this revision under Decision 2's Caveat:
"This claim [concept-HV is right] is unvalidated. S6 includes a unit
test for the synonym case; S7 measures it on real outputs. If S7
falsifies it, we revisit." S7 falsified it. We revisited.

THRESHOLDS
----------
Defaults pinned by the cosine distribution analysis on the FAQ corpus
(see docs/PHASE_6_3_FAQ_REIMPORT_FINDING.md), but explicitly tagged
"unvalidated, S7 measures" in the scope doc:

  T_high    = 0.85   above intra-trio 50th, below intra-trio 90th
  T_low     = 0.75   above inter-FAQ median + margin
  T_fact    = 0.85   per-fact-prompt cosine target
  T_payload = 0.72   text-cosine of payloads (revised from 0.5; the
                     encoder's floor for "different but related" is
                     higher than concept-HV's, so the threshold moves
                     up. Empirically pinned at 0.72 from the cross-pair
                     diagnostic on Qwen 2.5 7B output: same-FAQ min
                     = 0.7643, cross-FAQ max = 0.7098 on the
                     SSO/SAML/Okta vs password-reset comparison.
                     Midpoint of 0.7370 rounded up to 0.72 to give
                     same-FAQ a slightly larger safety margin than
                     cross-FAQ. Margin is thin (~0.05); S7 attempt #3
                     measures whether this holds across all 66
                     cross-FAQ pairs in the 12-FAQ corpus, not just
                     the one we sampled.)

Tunable per ThreeAxisBonder construction so tests and S7 can vary them.

BACKWARDS COMPATIBILITY
-----------------------
`_CosineOnlyBonder` implements the pre-followup-2 single-axis behavior
exactly. `add_pair_for_customer` defaults to it when no `bonder` kwarg
is passed, so every existing test path produces identical decisions.
The three-axis bonder is opt-in via the metadata store's call site.

The Bonder Protocol no longer takes `vocab`. The DSL Vocabulary was
load-bearing for concept-HV; under text-cosine it's unused. Per
greenfield principle ("we don't have users yet"), the Protocol drops
the parameter rather than carry a deprecated-and-ignored kwarg. The
metadata store and importer call sites are updated in the same commit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

import numpy as np
import structlog

from ..encoding.executor import encode_native_async, run_encoder_bound


if TYPE_CHECKING:
    from crystal_cache.decomposer.base import Decomposer
    from crystal_cache.encoding.base import BindCapableEncoder
    from crystal_cache.models import Crystal, Fact


logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class BondDecision:
    """The bonder's verdict for one bond-vs-spawn decision.

    Carries enough telemetry that the metadata store can log a single
    structured event per decision, and tests can assert on the reason
    code rather than reconstructing the decision from cosines.

    Attributes:
        bond: True = bond into the candidate. False = spawn fresh.
        reason: Short tag describing which axis fired. One of:
          - "cosine_clear"               (axis 1, above T_high)
          - "cosine_below_floor"         (axis 1, below T_low)
          - "cosine_gray_fact_match"     (axis 2 fired in gray zone)
          - "cosine_gray_payload_agree"  (axis 3 fired in gray zone)
          - "cosine_gray_payload_disagree"  (axis 3 vetoed in gray zone)
          - "cosine_gray_no_payload"     (gray zone, no axis-3 data)
          - "empty_bank"                 (no candidate; caller spawns)
          - "cosine_only_above_threshold"  (back-compat path bonded)
          - "cosine_only_below_threshold"  (back-compat path spawned)
        candidate_score: routing_vector cosine for the candidate.
            None for empty_bank.
        best_fact_cosine: Highest cosine of incoming prompt against
            any fact's prompt_text in the candidate. None when axis 2
            wasn't consulted (cosine_clear / cosine_below_floor /
            empty_bank / back-compat).
        payload_agreement: Text-space cosine of payloads (post-
            S7-attempt-2 revision). None when axis 3 wasn't consulted
            or either payload was absent.
        decomposer_called: True if axis 3 actually invoked the LLM.
            False otherwise (including when the decomposer was wired
            but axis 1 or 2 short-circuited first).
    """
    bond: bool
    reason: str
    candidate_score: Optional[float] = None
    best_fact_cosine: Optional[float] = None
    payload_agreement: Optional[float] = None
    decomposer_called: bool = False


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Bonder(Protocol):
    """Strategy interface for the bond-or-spawn decision.

    Implementations are stateless w.r.t. each call. The candidate
    crystal + facts + payloads are passed in by the caller; the bonder
    just runs its rule.

    Why a protocol instead of a single function: lets the metadata
    store accept a `bonder=` kwarg without committing to one
    implementation. `_CosineOnlyBonder` (back-compat) and
    `ThreeAxisBonder` (Phase 6.3 follow-up #2) both satisfy it.
    Future bonders (e.g. EMA-centroid + per-fact-prompt without the
    decomposer call) can land without changing call sites.
    """

    async def should_bond(
        self,
        *,
        candidate_crystal: Optional["Crystal"],
        candidate_score: Optional[float],
        candidate_facts: list["Fact"],
        candidate_payload: Optional[dict[str, Any]],
        incoming_prompt: str,
        incoming_payload: Optional[dict[str, Any]],
        encoder: "BindCapableEncoder",
    ) -> BondDecision:
        ...


# ---------------------------------------------------------------------------
# Cosine-only (back-compat)
# ---------------------------------------------------------------------------


class _CosineOnlyBonder:
    """Pre-followup-2 behavior: bond iff cosine >= threshold.

    Used as the default when `add_pair_for_customer` is called without
    a `bonder` kwarg. Every existing test path produces decisions
    identical to those produced before the bonder protocol existed.

    NOT exposed in the public learning module API — callers that want
    cosine-only behavior should pass `bonder=None` and get this
    automatically. Construction is reserved for the metadata store
    fallback path.
    """

    def __init__(self, *, threshold: float) -> None:
        self.threshold = float(threshold)

    async def should_bond(
        self,
        *,
        candidate_crystal: Optional["Crystal"],
        candidate_score: Optional[float],
        candidate_facts: list["Fact"],
        candidate_payload: Optional[dict[str, Any]],
        incoming_prompt: str,
        incoming_payload: Optional[dict[str, Any]],
        encoder: "BindCapableEncoder",
    ) -> BondDecision:
        if candidate_crystal is None or candidate_score is None:
            return BondDecision(
                bond=False, reason="empty_bank",
                candidate_score=None,
            )
        if candidate_score >= self.threshold:
            return BondDecision(
                bond=True,
                reason="cosine_only_above_threshold",
                candidate_score=candidate_score,
            )
        return BondDecision(
            bond=False,
            reason="cosine_only_below_threshold",
            candidate_score=candidate_score,
        )


# ---------------------------------------------------------------------------
# Payload agreement (axis 3 primitive — revised post-S7-attempt-2)
# ---------------------------------------------------------------------------


# Stable key ordering for payload-string rendering. Pinning the order
# means two payloads with the same content but different dict insertion
# order produce the same string — and therefore the same encoding.
# Keys not in this list get sorted alphabetically and appended; this
# tolerates new fields the decomposer might emit (`tone`, `urgency`,
# anything future) without changing the function.
_PAYLOAD_KEY_ORDER: tuple[str, ...] = ("intent", "topic", "domain")


def _render_payload(payload: dict[str, Any]) -> str:
    """Render a decomposer payload to a stable string for encoding.

    The string is `key=value` pairs separated by spaces, with
    `_PAYLOAD_KEY_ORDER` keys first (in that order) followed by any
    remaining keys alphabetically. Non-string values are stringified
    via `str()`. Missing keys are skipped (no `key=None` pollution
    that would change the encoded representation when a field is
    absent).

    Examples:
        >>> _render_payload({"intent": "ask", "topic": "x", "domain": "y"})
        'intent=ask topic=x domain=y'
        >>> _render_payload({"domain": "y", "topic": "x", "intent": "ask"})
        'intent=ask topic=x domain=y'
        >>> _render_payload({"intent": "ask", "tone": "casual"})
        'intent=ask tone=casual'
    """
    if not payload:
        return ""

    parts: list[str] = []
    seen: set[str] = set()
    for key in _PAYLOAD_KEY_ORDER:
        if key in payload and payload[key] is not None and payload[key] != "":
            parts.append(f"{key}={payload[key]}")
            seen.add(key)
    extra_keys = sorted(k for k in payload.keys() if k not in seen)
    for key in extra_keys:
        if payload[key] is not None and payload[key] != "":
            parts.append(f"{key}={payload[key]}")
    return " ".join(parts)


def payload_agreement(
    a: Optional[dict[str, Any]],
    b: Optional[dict[str, Any]],
    *,
    encoder: "BindCapableEncoder",
) -> Optional[float]:
    """Text-space cosine similarity between two decomposer payloads.

    Returns a float in [-1.0, 1.0] when both payloads are non-empty
    dicts AND each renders to a non-empty string. Returns None when
    either payload is None, empty, or renders to an empty string.

    Implementation:
      1. Render each payload to a stable string via `_render_payload`
         (keys ordered intent → topic → domain → others alphabetical).
      2. Encode both with `encoder.encode_native` (the production
         sentence-transformer producing 768-dim float vectors).
      3. Compute normalized dot product (standard cosine).

    Why text-cosine and not concept-HV (the original Phase 6.3
    follow-up #2 design):
      Concept-HV (DSL `from_decomposer_output()` + `similarity()`) is
      EXACT-MATCH on string identity. "saml_configuration" and
      "integration" hash to orthogonal hypervectors even though
      semantically they overlap.

      S7 attempt #2 (May 3, 2026) surfaced this with Qwen 2.5 7B at
      temperature=0: paraphrases of one FAQ produced different topic
      strings ("integration" vs "saml_configuration"), agreement
      collapsed below T_payload, and bonding correctly-but-uselessly
      spawned a fresh crystal. The mechanism's RULE was correct; its
      INPUT primitive was too brittle for small-model output.

      Text-cosine via the production encoder fixes this. The encoder
      already knows "saml_configuration" and "integration" are
      semantically related (it was trained on that kind of signal).
      Same encoder as axes 1 and 2; one similarity contract across
      the bonder.

    Returns None signals "no axis-3 data" to the caller; the bonder
    treats None the same way it treats `decomposer is None` — the
    conservative-spawn fallback fires.

    The encoder must expose `encode_native` (the unit-norm 768-dim
    sentence-transformer output). Hash encoders don't satisfy this;
    callers passing a non-bind-capable encoder will hit the same
    AttributeError pattern as `add_pair_to_crystal`.
    """
    if not a or not b:
        return None

    text_a = _render_payload(a)
    text_b = _render_payload(b)
    if not text_a or not text_b:
        return None

    vec_a = encoder.encode_native(text_a)
    vec_b = encoder.encode_native(text_b)
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return float(np.dot(vec_a, vec_b)) / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Three-axis bonder
# ---------------------------------------------------------------------------


class ThreeAxisBonder:
    """Cost-ordered three-axis bonding decision.

    Constructor args:
        cosine_threshold_high: T_high. Above this on routing_vector
            cosine, bond unconditionally. Default 0.85.
        cosine_threshold_low: T_low. Below this, spawn unconditionally.
            Default 0.75. Must be < cosine_threshold_high.
        fact_cosine_threshold: T_fact. In the gray zone, bond if
            ANY existing fact's prompt has cosine >= this against
            the incoming prompt. Default 0.85.
        payload_agreement_threshold: T_payload. In the gray zone with
            no fact match, bond if decomposer payload agreement
            (text-space cosine) >= this. Default 0.72.
            Pre-S7-attempt-3 default was 0.5 against concept-HV
            cosine; text-cosine has a higher floor for "different
            but related" so the threshold moved up. Empirically
            pinned at 0.72 from a same-FAQ-vs-cross-FAQ diagnostic
            on Qwen 2.5 7B output (margin 0.05). Re-measure on new
            decomposer/encoder combinations; the threshold is not
            transferable.
        decomposer: Optional Decomposer for axis 3. None disables
            axis 3 — gray zone with no fact match falls through to
            spawn (reason="cosine_gray_no_payload"). Production
            wires this from app.state.decomposer; tests pass a stub
            or None.

    Decision logic per the module-level docstring.

    The bonder is stateless — same inputs produce same outputs. Safe
    to share one ThreeAxisBonder instance across all customers; the
    per-customer state (candidate facts, candidate payload) is
    threaded through the call.
    """

    def __init__(
        self,
        *,
        cosine_threshold_high: float = 0.85,
        cosine_threshold_low: float = 0.75,
        fact_cosine_threshold: float = 0.85,
        payload_agreement_threshold: float = 0.72,
        decomposer: Optional["Decomposer"] = None,
    ) -> None:
        if cosine_threshold_low >= cosine_threshold_high:
            raise ValueError(
                f"cosine_threshold_low ({cosine_threshold_low}) must be "
                f"strictly less than cosine_threshold_high "
                f"({cosine_threshold_high}); otherwise the gray zone is "
                f"empty and the bonder reduces to a single-axis check."
            )
        self.t_high = float(cosine_threshold_high)
        self.t_low = float(cosine_threshold_low)
        self.t_fact = float(fact_cosine_threshold)
        self.t_payload = float(payload_agreement_threshold)
        self.decomposer = decomposer

    async def should_bond(
        self,
        *,
        candidate_crystal: Optional["Crystal"],
        candidate_score: Optional[float],
        candidate_facts: list["Fact"],
        candidate_payload: Optional[dict[str, Any]],
        incoming_prompt: str,
        incoming_payload: Optional[dict[str, Any]],
        encoder: "BindCapableEncoder",
    ) -> BondDecision:
        # Empty bank: no candidate, no decision to make. Caller spawns
        # without consulting the bonder; the empty_bank reason is here
        # for trace consistency in case a caller does invoke the
        # bonder with no candidate.
        if candidate_crystal is None or candidate_score is None:
            return BondDecision(
                bond=False, reason="empty_bank",
                candidate_score=None,
            )

        # Axis 1: routing-vector cosine.
        if candidate_score >= self.t_high:
            return BondDecision(
                bond=True,
                reason="cosine_clear",
                candidate_score=candidate_score,
            )
        if candidate_score < self.t_low:
            return BondDecision(
                bond=False,
                reason="cosine_below_floor",
                candidate_score=candidate_score,
            )

        # Gray zone (t_low <= candidate_score < t_high).

        # Axis 2: per-fact-prompt cosine. Encode the incoming prompt
        # once; compare against every fact's stored prompt_text.
        # An empty fact list is degenerate (a freshly-spawned crystal
        # mid-write?) — treat as no signal.
        best_fact_cosine: Optional[float] = None
        if candidate_facts:
            incoming_native = await encode_native_async(encoder, incoming_prompt)
            in_norm = float(np.linalg.norm(incoming_native))
            if in_norm > 0:
                cosines: list[float] = []
                for fact in candidate_facts:
                    pt = (fact.prompt_text or "").strip()
                    if not pt:
                        continue
                    fact_native = await encode_native_async(encoder, pt)
                    fn_norm = float(np.linalg.norm(fact_native))
                    if fn_norm == 0:
                        continue
                    cos = float(np.dot(incoming_native, fact_native)) / (
                        in_norm * fn_norm
                    )
                    cosines.append(cos)
                if cosines:
                    best_fact_cosine = max(cosines)

        if best_fact_cosine is not None and best_fact_cosine >= self.t_fact:
            return BondDecision(
                bond=True,
                reason="cosine_gray_fact_match",
                candidate_score=candidate_score,
                best_fact_cosine=best_fact_cosine,
            )

        # Axis 3: decomposer payload TEXT cosine.
        # Skip if either side lacks a payload OR the decomposer is
        # unwired. Conservative-spawn fallback covers all three
        # missing-data cases under one reason code so the trace
        # distribution is interpretable.
        if (
            self.decomposer is None
            or candidate_payload is None
            or not candidate_payload
        ):
            return BondDecision(
                bond=False,
                reason="cosine_gray_no_payload",
                candidate_score=candidate_score,
                best_fact_cosine=best_fact_cosine,
            )

        # We need a fresh decomposition of the incoming prompt. The
        # caller passes incoming_payload if they already decomposed;
        # otherwise we decompose here. Either way, mark
        # decomposer_called=True only if WE made the LLM call.
        decomposer_called_here = False
        effective_incoming = incoming_payload
        if effective_incoming is None:
            try:
                result = await self.decomposer.decompose(incoming_prompt)
                effective_incoming = result.payload
                decomposer_called_here = True
            except Exception as e:
                # Wide except covers both DecomposerError (the protocol
                # contract) and any other unexpected failure. Same
                # graceful-degradation path: spawn fresh, log once,
                # don't block the write.
                logger.warning(
                    "bonder.decomposer_failed",
                    error=str(e),
                    error_type=type(e).__name__,
                    candidate_id=candidate_crystal.id,
                )
                return BondDecision(
                    bond=False,
                    reason="cosine_gray_no_payload",
                    candidate_score=candidate_score,
                    best_fact_cosine=best_fact_cosine,
                    decomposer_called=True,
                )

        if not effective_incoming:
            # Decomposer returned an empty payload. Same conservative
            # fallback as no-decomposer.
            return BondDecision(
                bond=False,
                reason="cosine_gray_no_payload",
                candidate_score=candidate_score,
                best_fact_cosine=best_fact_cosine,
                decomposer_called=decomposer_called_here,
            )

        agreement = await run_encoder_bound(
            payload_agreement,
            effective_incoming, candidate_payload, encoder=encoder,
        )
        if agreement is None:
            return BondDecision(
                bond=False,
                reason="cosine_gray_no_payload",
                candidate_score=candidate_score,
                best_fact_cosine=best_fact_cosine,
                decomposer_called=decomposer_called_here,
            )

        if agreement >= self.t_payload:
            return BondDecision(
                bond=True,
                reason="cosine_gray_payload_agree",
                candidate_score=candidate_score,
                best_fact_cosine=best_fact_cosine,
                payload_agreement=agreement,
                decomposer_called=decomposer_called_here,
            )
        return BondDecision(
            bond=False,
            reason="cosine_gray_payload_disagree",
            candidate_score=candidate_score,
            best_fact_cosine=best_fact_cosine,
            payload_agreement=agreement,
            decomposer_called=decomposer_called_here,
        )
