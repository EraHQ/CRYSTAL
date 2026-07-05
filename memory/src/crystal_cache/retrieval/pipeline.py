"""Retrieval + injection pipeline.

Runs BOTH retrieval paths in parallel for every request:

  TEXT PATH (existing)
    PromptEncoder(query) -> CrystalRouter -> top-K crystals by text-space
    cosine -> CrystalReader -> inject summary_text as system prefix

  CONCEPT PATH (new in v0.3)
    Decomposer(query) -> from_decomposer_output -> DslConfigStore.rank
    -> top config(s) by concept-space cosine -> append structured context

v0.3 MERGE POLICY - ADDITIVE ONLY
---------------------------------
The concept path in v0.3 is strictly additive:
  - Text path determines match_type (high / medium / low / none) as before.
  - Text path controls whether crystal text gets injected.
  - If concept path found a matching config AND the text path also
    injected, we append one line describing the matched config's
    intent/domain/tone for the upstream model to condition on.
  - Concept path never overrides text path. Never changes match_type.
  - Concept path failures are silent - text path is the source of truth.

Why so conservative? We don't yet have evidence that concept-path
routing improves upstream-model output quality. Shipping it as an
observation-only enrichment lets us A/B it with real traffic: we log
the concept path's top result, correlate with downstream shadow-helped
/ shadow-hurt counters, and only expand its authority once data says it
helps.

A future v0.4 merge policy could:
  - Let concept path override the crystal choice when it names one
    explicitly (crystal_hint field on a matched config)
  - Treat a high-confidence concept match as a promotion signal for a
    low-confidence text match
  - Replace injected text entirely when concept-path confidence is high
    and text-path has no match
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..decomposer.base import Decomposer
from ..decomposer.config_store import DslConfigStore
from ..encoding.executor import encode_messages_async
from ..encoding.prompt_encoder import PromptEncoder
from ..execution.text_injection import inject_text_context
from ..models import Customer, Operator
from .concept_router import ConceptRouteOutcome, ConceptRouter
from .match_classifier import (
    MatchClassifier,
    MatchType,
    RoutingDecision,
    RoutingResult,
)
from .reader import CrystalContext, CrystalReader, _provenance_header
from .recall import RecalledFact, recall_from_crystal
from .chain_resolver import ChainResolver
from .router import CrystalRouter
from .synthesis import synthesize_joint_statement

if TYPE_CHECKING:
    from ..encoding.semantic import SemanticTextEncoder
    from ..infrastructure.decoder_loader import DecoderLoader
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_index import VectorIndex


logger = structlog.get_logger(__name__)


@dataclass
class RetrievalOutcome:
    """Everything the request pipeline needs to know after retrieval ran."""
    messages: list[dict[str, Any]]
    match_type: MatchType
    # Possible values:
    #   "none"                    — no injection (low match or no candidates)
    #   "text"                    — top-1 crystal text injected
    #   "text+concept"            — top-1 + concept-path hint
    #   "text+synthesis"          — SPREAD branch: top-1 + top-2 + bind-v1 hint
    #   "text+synthesis+concept"  — both signals fired together
    injection_method: str
    matched_crystal_ids: list[str]
    top_score: float
    injected_text: Optional[str] = None

    # Growth G1 (citations): when retrieval injected AND cite=True, the
    # citable sources behind the injection (v1: the primary crystal as
    # [[cc:1]]). None when citations are off or nothing was injected. The
    # injection in `messages` carries the handle + cite instruction;
    # `injected_text` above stays the RAW content for grounding.
    citation_manifest: Optional[list[Any]] = None

    # v0.3 additions: concept-path observations.
    concept_top_config: Optional[str] = None
    concept_top_score: float = 0.0
    concept_payload: Optional[dict[str, Any]] = None
    concept_path_ran: bool = False

    # Four-way routing decision (perfect / spread / low_confidence /
    # no_match) computed alongside the legacy match_type. The pipeline
    # branches on match_type for the inject/passthrough decision;
    # routing_decision drives the synthesis branch on SPREAD.
    #
    # routing_decision is None only when no candidates were returned at all
    # (empty bank). In that case match_type is "none".
    routing_decision: Optional[RoutingDecision] = None
    routing_top1: Optional[float] = None
    routing_top2: Optional[float] = None
    routing_margin: Optional[float] = None

    # Joint statement produced by bind-v1 synthesis when routing_decision
    # was SPREAD AND the decoder loader was available AND both top-2
    # crystals had answer_embedding_native populated. None otherwise.
    # This is a HINT injected alongside the raw FAQ texts — not the
    # answer. The LLM does the actual synthesis using all three signals.
    synthesized_joint_statement: Optional[str] = None

    # Phase 2 (April 2026, recall path). When routing_decision is
    # PERFECT AND the encoder is bind-capable AND top-1 has a
    # codebook AND cleanup matched a Fact above the threshold,
    # `recalled_fact_id` carries the matched Fact's id and
    # `recalled_fact_score` carries the cleanup cosine. The injected
    # text in this branch is the recalled Fact's `claim_text`, not
    # the crystal's `summary_text`.
    #
    # Both fields are None when:
    #   - routing_decision != PERFECT (the recall path didn't run)
    #   - encoder doesn't expose .P / .fingerprint() (hash encoder;
    #     legacy summary_text path runs instead)
    #   - top-1 crystal has no codebook (legacy bank, pre-Phase-1.1)
    #   - cleanup cosine fell below cleanup_threshold (the routing
    #     was right but no specific pair clears the noise floor;
    #     match_type demoted to 'low')
    #
    # Surface them on the outcome so the inspector can render
    # "which pair fired" alongside the routing decision; downstream
    # telemetry (margin/coverage etc.) can also distinguish
    # "recalled-fact PERFECT" from "summary-text PERFECT" without
    # re-running the recall.
    recalled_fact_id: Optional[str] = None
    recalled_fact_score: Optional[float] = None

    # Cache-hit short-circuit (April 2026, GAIA fold-back, item 6).
    #
    # When set, this is the canonical short answer text the gateway
    # should return to the caller WITHOUT invoking the upstream LLM.
    # The pipeline populates this field when:
    #   - routing_decision is PERFECT (top-1 owns the query)
    #   - top-1 crystal's source_kind is "model_reasoning"
    #   - top-1 crystal's answer_value is populated (non-None, non-empty)
    #
    # Validated on the GAIA experiment scaffold: trying to enforce
    # "use the cached answer" via a system-prompt directive is
    # unreliable (the model overrides the directive when the user
    # message contains stronger instructional voice, and even when it
    # complies it burns 4-6 wasted web-search calls verifying first).
    # Bypassing the LLM entirely is deterministic, free, and ~5ms.
    #
    # The gateway is responsible for honoring this field. retrieve_and_inject
    # still populates `messages` and `injection_method` as if the
    # request were going to be proxied; if the gateway opts to honor
    # the cache hit, those values are ignored. This keeps the retrieval
    # path decoupled from the upstream-call path — callers that don't
    # want cache-hit behavior can just ignore this field.
    cache_hit_response: Optional[str] = None
    cache_hit_crystal_id: Optional[str] = None


DEFAULT_TOP_K = 10


async def retrieve_and_inject(
    customer: Customer,
    messages: list[dict[str, Any]],
    store: "MetadataStore",
    vector_index: "VectorIndex",
    encoder: PromptEncoder,
    *,
    classifier: Optional[MatchClassifier] = None,
    top_k: int = DEFAULT_TOP_K,
    decomposer: Optional[Decomposer] = None,
    config_store: Optional[DslConfigStore] = None,
    decoder_loader: Optional["DecoderLoader"] = None,
    crystal_type: str = "customer:legacy",
    operator: Optional[Operator] = None,
    cite: bool = False,
) -> RetrievalOutcome:
    """Run the full retrieval + injection pipeline for a single request.

    If decomposer AND config_store are both provided, the concept path
    runs in parallel with the text path. Either missing, concept path
    is skipped and behavior is identical to pre-v0.3.

    If decoder_loader is provided AND the routing decision comes back
    as SPREAD AND both top-1 and top-2 crystals have answer_embedding_native
    populated, the synthesis path runs (April 2026, item 6). The decoded
    joint statement is added to the injection alongside both raw FAQ
    texts. If synthesis fails for any reason, the pipeline falls back
    to standard top-1 injection — the user still gets a useful answer.

    Phase 3 audit fix #8 (April 2026): `crystal_type` threads through
    to the router and ultimately to the index's search_routing, which made it
    required. Default 'customer:legacy' matches the migration 0012
    seeded type and the gateway's current behavior; callers that
    want to route into a type-scoped bank (e.g. an admin endpoint
    serving the medical-records use case) pass the explicit type.

    Safe to call even if the customer has no bank - falls through to
    match_type='none' without raising. The ingress layer should not
    wrap this in try/except for retrieval errors; structural errors
    (bad bank dimension, etc.) SHOULD bubble up.
    """
    classifier = classifier or MatchClassifier()
    router = CrystalRouter(
        encoder=encoder,
        vector_index=vector_index,
        metadata_store=store,
    )
    reader = CrystalReader(store=store)

    # Step 1: encode the user's query for the text path.
    #
    # The query is encoded directly (windowed). The legacy sparse-key
    # query transformation was removed with the unified-key rebuild:
    # crystals are no longer ingested as 3-8 word semantic keys, so
    # rewriting the query into one no longer matches the bank.
    #
    # Phase 1.5.3: multi-turn routing. Two changes:
    #
    #   a) Tool-turn skip: if the latest message is role="tool", the
    #      encoder should NOT encode tool result text. Walk backward to
    #      the most recent user message and truncate the message list at
    #      that point for encoding purposes. The full message list still
    #      goes to upstream — only the encoding window is trimmed.
    #
    #   b) Windowed context: the encoder considers the last N user turns
    #      (default 3, per-customer override via routing_context_window).
    #      This prevents early-turn context from dominating the vector
    #      in long conversations.
    #
    # System default window. Per spec: "Configurable window (default
    # last 3 user turns)."
    DEFAULT_ROUTING_WINDOW = 3
    window = customer.routing_context_window or DEFAULT_ROUTING_WINDOW

    # Tool-turn skip: if the most recent message is role="tool", trim
    # messages for encoding back to the most recent user message.
    encoding_messages = list(messages)
    if encoding_messages and encoding_messages[-1].get("role") == "tool":
        for idx in range(len(encoding_messages) - 1, -1, -1):
            if encoding_messages[idx].get("role") == "user":
                encoding_messages = encoding_messages[: idx + 1]
                break

    # Encode the (windowed) query directly. The unified sparse key is a
    # wide->specific path built at ingestion / query-classification time,
    # not a semantic word-key derived from the raw prompt, so there is no
    # query rewrite step here anymore.
    query_vector = await encode_messages_async(encoder, encoding_messages, window=window)

    # Extract plain query text for the concept path (from the last user
    # turn). If there's no user turn, concept path gets skipped.
    query_text = _extract_last_user_text(messages)

    # Step 2: text path - route to candidate crystals.
    # Pass customer's general crystal subscriptions for merged search.
    general_types = getattr(customer, 'general_crystal_types', None)
    candidates = await router.route(
        customer_id=customer.id,
        query=query_vector,
        k=top_k,
        crystal_type=crystal_type,
        general_crystal_types=general_types or None,
        operator=operator,
    )

    # Step 3 (parallel): concept path. We don't actually await in parallel
    # yet - sequential calls are simpler and the latency difference is
    # trivial at this scale (one LLM call vs one vector search). If
    # perf becomes an issue we promote to asyncio.gather().
    concept_outcome = await _maybe_run_concept_path(
        customer_id=customer.id,
        query_text=query_text,
        decomposer=decomposer,
        config_store=config_store,
    )

    if not candidates:
        # Text path found nothing. Concept-only injection is explicitly
        # out-of-scope for v0.3 merge policy - we still pass through
        # unchanged. But we record concept observations in the outcome.
        return RetrievalOutcome(
            messages=list(messages),
            match_type="none",
            injection_method="none",
            matched_crystal_ids=[],
            top_score=0.0,
            routing_decision=RoutingDecision.NO_MATCH,
            routing_top1=0.0,
            routing_top2=None,
            routing_margin=None,
            **_cache_hit_fields(None, None),
            **_concept_fields(concept_outcome),
        )

    matched_ids = [c.id for c, _ in candidates]
    top_crystal, top_score = candidates[0]

    # Classify the TEXT-path score. Concept path does not affect this.
    match_type = classifier.classify_for_customer(top_score, customer)

    # Four-way routing decision — drives the SPREAD synthesis branch.
    # Computed from candidates' top-2 scores via classify_routing_for_customer.
    routing_scores = [s for _, s in candidates]
    routing_result = classifier.classify_routing_for_customer(
        routing_scores, customer
    )
    routing_fields = _routing_fields(routing_result)

    # Phase 2 + Phase 7.1 Session 5 extension: recall on PERFECT,
    # SPREAD, and LOW_CONFIDENCE branches.
    #
    # Original Phase 2 contract: recall fires only on PERFECT. The
    # bind-storage architecture stores per-pair claim_text on Fact
    # rows; without recall, SPREAD and LOW_CONFIDENCE branches have
    # nothing to inject (Crystal.summary_text is None on bind-storage
    # crystals — the bonder writes summary_text=None at spawn).
    # Discovered during the SWE-bench eval bootstrap: SPREAD-decision
    # queries on bind-storage banks were producing
    # `match_but_empty_context` and skipping injection entirely,
    # nullifying the treatment arm.
    #
    # Phase 7.1 Session 5 fix: extend recall to all three
    # decision branches that found candidates. The architecture's
    # original intent — "SPREAD injects top-1 + top-2 + synthesis
    # hint" — is preserved by recalling Facts from BOTH top-1 and
    # top-2 when SPREAD fires, with synthesis remaining optional
    # (decoder_loader-gated). LOW_CONFIDENCE recalls top-1 only and
    # uses GAIA-validated hedging language so the upstream model
    # can ignore the injected content if it's not helpful (validated
    # in the GAIA experiment scaffold: hedged advisory voicing
    # outperforms gating-out borderline matches).
    #
    # The score-floor gate (recalled_top1_score_floor below)
    # protects against the noise regime: when top-1 cosine is below
    # 0.5, the bank likely has no relevant content for this query,
    # and even hedged injection would just add noise to the model's
    # context. The floor is an additional gate beyond the routing
    # classifier's NO_MATCH cutoff because the classifier's
    # thresholds are tuned for FAQ-shape data; SWE-bench bug reports
    # produce broader cosine distributions, and the noise/signal
    # boundary lands closer to 0.5 in this regime.
    #
    # Why recall is gated on encoder bind-capability rather than
    # always-on: the legacy hash encoder doesn't have a P matrix,
    # so the unbind math can't run; banks built under it have no
    # codebook and recall would always return None. Hash-encoder
    # banks fall through to the legacy summary_text injection path
    # and the legacy Crystal-level cache-hit detection;
    # semantic-encoder banks land in the new path.
    #
    # Implementation note: recall_from_crystal handles its own
    # error cases (empty codebook, fingerprint mismatch, below-
    # threshold cosine) and returns None. We don't catch exceptions
    # here because any uncaught exception inside the primitive
    # signals a structural problem (encoder geometry mismatch, DB
    # error) that should bubble up rather than silently degrade.
    LOW_CONFIDENCE_INJECTION_FLOOR = 0.5
    recalled: Optional[RecalledFact] = None
    recalled_top2: Optional[RecalledFact] = None
    encoder_is_bind_capable = (
        hasattr(encoder, "P") and hasattr(encoder, "fingerprint")
    )
    runs_recall = (
        encoder_is_bind_capable
        and routing_result.decision in (
            RoutingDecision.PERFECT,
            RoutingDecision.SPREAD,
            RoutingDecision.LOW_CONFIDENCE,
        )
    )
    # LOW_CONFIDENCE skips recall entirely if top-1 is below the
    # floor — at that point the bank's signal is too weak for even
    # hedged injection to be net-positive.
    if (
        runs_recall
        and routing_result.decision == RoutingDecision.LOW_CONFIDENCE
        and top_score < LOW_CONFIDENCE_INJECTION_FLOOR
    ):
        runs_recall = False
        logger.debug(
            "retrieval.low_confidence_below_floor",
            customer_id=customer.id,
            top_score=top_score,
            floor=LOW_CONFIDENCE_INJECTION_FLOOR,
            top_crystal_id=top_crystal.id,
            note=(
                "LOW_CONFIDENCE decision with top-1 cosine below the "
                "injection floor. Skipping recall and injection — "
                "the bank has no usable signal for this query."
            ),
        )

    if runs_recall:
        # Phase 3 (April 2026): pass a ChainResolver so the cleanup
        # codebook is extended with any ACL-permitted chained crystals'
        # Facts. When no chains exist (current state of the dev DB),
        # the resolver returns [] and recall behaves exactly as in
        # Phase 2. Constructing the resolver per call is cheap — it's
        # stateless aside from the store handle.
        recalled = await recall_from_crystal(
            top_crystal,
            query_vector,
            store=store,
            encoder=encoder,  # type: ignore[arg-type]
            chain_resolver=ChainResolver(store=store),
            requesting_customer_id=customer.id,
        )
        # SPREAD also recalls top-2 because the architecture's intent
        # for SPREAD is "both topics are relevant." Two recalled facts
        # give the upstream model two reference points, mirroring the
        # original synthesis branch's top-1 + top-2 design.
        if (
            routing_result.decision == RoutingDecision.SPREAD
            and len(candidates) >= 2
        ):
            top2_crystal = candidates[1][0]
            recalled_top2 = await recall_from_crystal(
                top2_crystal,
                query_vector,
                store=store,
                encoder=encoder,  # type: ignore[arg-type]
                chain_resolver=ChainResolver(store=store),
                requesting_customer_id=customer.id,
            )

        # PERFECT-specific: a None recall demotes match_type to "low"
        # and skips injection. This preserves the original Phase 2
        # spec §2.2: PERFECT routing means "the bank says this query
        # is unambiguously about top-1"; a recall miss at that
        # confidence means cleanup couldn't pin a specific pair, and
        # injecting summary_text would be confidently-wrong noise.
        # SPREAD/LOW_CONFIDENCE don't demote on recall miss because
        # they were never confident to begin with — falling through
        # to no-injection is the right behavior, no demotion needed.
        if (
            routing_result.decision == RoutingDecision.PERFECT
            and recalled is None
        ):
            logger.info(
                "retrieval.perfect_routing_recall_miss",
                customer_id=customer.id,
                crystal_id=top_crystal.id,
                top_score=top_score,
                margin=routing_result.margin,
                note=(
                    "PERFECT routing decision but cleanup found no "
                    "Fact above cleanup_threshold. Demoting match_type "
                    "to 'low' to suppress injection (per spec §2.2). "
                    "routing_decision stays PERFECT for telemetry."
                ),
            )
            match_type = "low"

    # Cache-hit short-circuit detection (Phase 2, Fact-level).
    #
    # Spec §2.2: cache-hit moves from Crystal.{source_kind, answer_value}
    # to Fact.{source_kind, answer_value} — a crystal holds many pairs,
    # and the cache-hit answer comes from the SPECIFIC pair cleanup
    # recovered, not from the crystal's aggregate.
    #
    # Two cases:
    #   1. Recall ran and matched a Fact → read source_kind +
    #      answer_value from the Fact (Phase 2 path).
    #   2. Recall didn't run (hash encoder, no codebook) → fall back
    #      to Crystal-level fields (legacy path, dropped Phase 7.4).
    #
    # We DON'T early-return from this function. The rest of the
    # pipeline still runs: messages get populated with injection so
    # that callers who choose not to honor cache_hit_response (e.g.
    # because they're forcing a shadow eval comparing cache-hit vs
    # upstream-with-injection) get a fully-populated outcome to work
    # with.
    cache_hit_response: Optional[str] = None
    cache_hit_crystal_id: Optional[str] = None

    # Minimum score threshold for cache hits. ONLY serve cached
    # solutions for near-exact matches (same question asked before).
    # Below this, the cached solution becomes REFERENCE MATERIAL
    # that the composer sends to the model as context — the model
    # writes its own solution informed by the reference.
    #
    # 0.90 = high confidence same question. Below this, the cached
    # solution becomes reference material for the composer.
    CACHE_HIT_SCORE_THRESHOLD = 0.90

    if recalled is not None:
        # Phase 2 path: read from the matched Fact.
        if (
            recalled.fact.source_kind == "model_reasoning"
            and recalled.fact.answer_value
            and top_score >= CACHE_HIT_SCORE_THRESHOLD
        ):
            cache_hit_response = recalled.fact.answer_value
            cache_hit_crystal_id = top_crystal.id
            logger.info(
                "retrieval.cache_hit",
                customer_id=customer.id,
                crystal_id=top_crystal.id,
                fact_id=recalled.fact.id,
                pair_type=recalled.fact.pair_type,
                top_score=top_score,
                cleanup_score=recalled.score,
                margin=routing_result.margin,
                answer_chars=len(cache_hit_response),
                source="fact_level",
            )
    elif (
        routing_result.decision == RoutingDecision.PERFECT
        and top_crystal.source_kind == "model_reasoning"
        and top_crystal.answer_value
        and top_score >= CACHE_HIT_SCORE_THRESHOLD
    ):
        # Legacy back-compat: hash-encoder bank or pre-Phase-1.1
        # crystal that never got a codebook. Crystal-level cache-hit
        # is the only signal available. Phase 7.4 drops this branch
        # along with Crystal.answer_embedding_native.
        cache_hit_response = top_crystal.answer_value
        cache_hit_crystal_id = top_crystal.id
        logger.info(
            "retrieval.cache_hit",
            customer_id=customer.id,
            crystal_id=top_crystal.id,
            top_score=top_score,
            margin=routing_result.margin,
            answer_chars=len(cache_hit_response),
            source="crystal_level_legacy",
        )

    # Phase 7.1 Session 5: short-circuit early returns BEFORE the
    # legacy summary_text path. If we have a recalled Fact (from
    # any branch — PERFECT, SPREAD, LOW_CONFIDENCE), use it as the
    # injection source. The crystal-level reader.read() path is the
    # legacy back-compat for hash-encoder banks; bind-storage banks
    # don't go through it.
    #
    # Branch logic for the no-injection early returns:
    #   - match_type == "low" (legacy classifier said low score):
    #     no injection. Same as before.
    #   - LOW_CONFIDENCE below floor + no top-1 recall: no
    #     injection (the runs_recall block above already gated this
    #     by setting runs_recall=False; recalled stays None).
    #   - SPREAD/LOW_CONFIDENCE with successful recall: HEDGED
    #     injection (GAIA-validated voicing — "may or may not be
    #     relevant").
    #   - PERFECT with successful recall: confident injection (no
    #     hedging — the bank is confident this fact answers the
    #     query).
    if match_type == "low":
        logger.debug(
            "retrieval.low_match",
            customer_id=customer.id,
            top_score=top_score,
            top_crystal_id=top_crystal.id,
            routing_decision=routing_result.decision.value,
            routing_margin=routing_result.margin,
        )
        return RetrievalOutcome(
            messages=list(messages),
            match_type="low",
            injection_method="none",
            matched_crystal_ids=matched_ids,
            top_score=top_score,
            **routing_fields,
            **_recalled_fields(recalled),
            **_cache_hit_fields(cache_hit_response, cache_hit_crystal_id),
            **_concept_fields(concept_outcome),
        )

    # Bind-storage path: build injection text from the recalled
    # Fact(s). When no recall produced anything (hash encoder, or
    # SPREAD/LOW_CONFIDENCE on a bank with empty codebooks), fall
    # through to the legacy summary_text reader path.
    context: Optional[CrystalContext] = None
    injection_text: Optional[str] = None
    injection_method: str = "text"

    if recalled is not None:
        # Branch: we have at least top-1 recalled. Build the
        # injection text based on the routing decision.
        if routing_result.decision == RoutingDecision.PERFECT:
            # Confident injection — no hedging. The bank is sure
            # this fact answers the query.
            injection_text = _fact_with_provenance(recalled.fact)
            injection_method = "text"
        elif (
            routing_result.decision == RoutingDecision.SPREAD
            and recalled_top2 is not None
        ):
            # SPREAD with both top-1 and top-2 recalled — hedged
            # framing because the bank couldn't pick a winner.
            # GAIA-validated voicing: tell the model these may not
            # be relevant rather than gating injection out.
            injection_text = (
                "Reference material from prior similar problems "
                "(may or may not be relevant — ignore if not "
                "helpful):\n\n"
                f"Reference 1: {_fact_with_provenance(recalled.fact)}\n\n"
                f"Reference 2: {_fact_with_provenance(recalled_top2.fact)}"
            )
            injection_method = "text+text"
        else:
            # SPREAD with top-2 missing, OR LOW_CONFIDENCE — single
            # hedged reference.
            injection_text = (
                "Reference material from a prior similar problem "
                "(may or may not be relevant — ignore if not "
                "helpful):\n\n"
                f"{_fact_with_provenance(recalled.fact)}"
            )
            injection_method = "text"

        # Voicing for inject_text_context. Bind-storage crystals
        # have summary_text=None (the bonder writes None at spawn),
        # so reader.read() returns None for them — we can't read
        # context.voicing. Default to "advisory" which matches the
        # pre-Phase-1.1 behavior for crystals without explicit
        # voicing metadata. Phase 7.4 work that drops summary_text
        # entirely will need a Crystal.voicing field; for now,
        # "advisory" is the safe default for bind-storage recall.
        context = await reader.read(top_crystal)
    else:
        # No recall. Fall through to legacy summary_text path —
        # same logic as before. Hash-encoder banks land here.
        context = await reader.read(top_crystal)
        if context is None:
            logger.debug(
                "retrieval.match_but_empty_context",
                customer_id=customer.id,
                match_type=match_type,
                top_crystal_id=top_crystal.id,
                routing_decision=routing_result.decision.value,
            )
            return RetrievalOutcome(
                messages=list(messages),
                match_type=match_type,
                injection_method="none",
                matched_crystal_ids=matched_ids,
                top_score=top_score,
                **routing_fields,
                **_recalled_fields(recalled),
                **_cache_hit_fields(
                    cache_hit_response, cache_hit_crystal_id
                ),
                **_concept_fields(concept_outcome),
            )
        injection_text = context.text
        injection_method = "text"

    # Defensive: if we built no injection_text by here, return no-
    # injection. Shouldn't be reachable — the recall branch always
    # builds text, and the no-recall branch returns early on empty
    # context — but a defensive check guards future refactors.
    if injection_text is None:
        logger.warning(
            "retrieval.no_injection_text_built",
            customer_id=customer.id,
            top_crystal_id=top_crystal.id,
            routing_decision=routing_result.decision.value,
            note=(
                "Reached the post-recall section without building "
                "injection_text. This is a logic bug; falling "
                "through to no-injection."
            ),
        )
        return RetrievalOutcome(
            messages=list(messages),
            match_type=match_type,
            injection_method="none",
            matched_crystal_ids=matched_ids,
            top_score=top_score,
            **routing_fields,
            **_recalled_fields(recalled),
            **_cache_hit_fields(cache_hit_response, cache_hit_crystal_id),
            **_concept_fields(concept_outcome),
        )

    synthesized_joint: Optional[str] = None

    # SPREAD branch — synthesis path. Activates only when:
    #   - routing_result.decision is SPREAD (not PERFECT, LOW, or NO_MATCH)
    #   - decoder_loader was passed in (CC_ENABLE_DECODER=true at startup)
    #   - we actually have a top-2 crystal in candidates
    #   - the encoder exposes the P matrix (semantic encoder; not hash).
    #     synthesis_joint_statement requires the SAME P that was used when
    #     answer_embedding_native was written. The hash encoder doesn't
    #     have a P, and its banks have no native embeddings to bind anyway.
    #   - recalled is None (i.e. we built injection_text from the
    #     legacy summary_text path, not from bind-storage recall). When
    #     recall fired, it already built the SPREAD injection text from
    #     two recalled facts; running synthesis on top would either
    #     overwrite that with synthesis-from-summary_text (loses the
    #     specific recalled facts) or stack double-injection. Phase 7.1
    #     Session 5 decision: recall takes precedence over synthesis.
    #     Synthesis only runs as the SPREAD-path injection when there's
    #     no recall to use — i.e. on legacy hash-encoder banks that
    #     never had codebooks.
    # Any failure inside synthesize_joint_statement returns None, and we
    # fall back to standard top-1 injection without surfacing an error
    # to the user. See synthesis.py for the HDC math (bundle as of Finding 15).
    if (
        routing_result.decision == RoutingDecision.SPREAD
        and decoder_loader is not None
        and len(candidates) >= 2
        and hasattr(encoder, "P")
        and recalled is None
        and context is not None
    ):
        top2_crystal = candidates[1][0]
        top2_context = await reader.read(top2_crystal)

        # Fetch the customer's full bank to build the cleanup manifold
        # basis (Finding 14). Bank-span projection strips off-manifold
        # noise from the recovered vector before bind-v1 sees it.
        #
        # Cost: one DB roundtrip on SPREAD-decision queries only
        # (~5–10% of traffic per the routing decision table). The full
        # crystal list is small enough that fetching all of them is
        # cheaper than designing a dedicated "natives only" query right
        # now — we'll specialize if list_crystals_for_customer becomes
        # hot enough to matter.
        #
        # If the fetch fails or the bank is empty, we fall through with
        # bank_natives=None which collapses cleanup to alpha=0
        # (pre-Finding-14 behavior). Synthesis still runs.
        bank_natives: Optional[list[list[float] | None]] = None
        try:
            from ..config import settings as _settings
            cleanup_alpha = float(_settings.synthesis_cleanup_alpha)
        except Exception:
            cleanup_alpha = 0.0

        if cleanup_alpha > 0.0:
            try:
                all_crystals = await store.list_crystals_for_customer(
                    customer.id, include_recall_gated=False,
                )
                bank_natives = [
                    c.answer_embedding_native for c in all_crystals
                ]
            except Exception as e:
                logger.warning(
                    "retrieval.synthesis_bank_fetch_failed",
                    customer_id=customer.id,
                    error=str(e),
                )
                bank_natives = None

        synthesized_joint = synthesize_joint_statement(
            top1=top_crystal,
            top2=top2_crystal,
            encoder=encoder,  # SemanticTextEncoder; has .P matrix
            decoder_loader=decoder_loader,
            bank_natives=bank_natives,
            cleanup_alpha=cleanup_alpha,
        )
        if synthesized_joint is not None and top2_context is not None:
            # Build a richer injection: top-1 text, top-2 text, and the
            # synthesis hint. Wrapped together so the upstream LLM sees
            # all three signals in one system message.
            injection_text = (
                f"{context.text}\n\n"
                f"Additional related context:\n{top2_context.text}\n\n"
                f"Synthesis hint (the system identified both topics as relevant): "
                f"{synthesized_joint}"
            )
            injection_method = "text+synthesis"
            logger.debug(
                "retrieval.synthesis_injected",
                customer_id=customer.id,
                top1_id=top_crystal.id,
                top2_id=top2_crystal.id,
                margin=routing_result.margin,
                synthesis_chars=len(synthesized_joint),
            )
        else:
            # Synthesis was attempted but returned None (decoder missing,
            # native embedding missing, decode error). Log and continue
            # with top-1 only — same as the no-synthesis path.
            logger.debug(
                "retrieval.synthesis_unavailable",
                customer_id=customer.id,
                top1_id=top_crystal.id,
                top2_id=top2_crystal.id,
                top2_context_present=top2_context is not None,
                synthesis_present=synthesized_joint is not None,
            )

    if concept_outcome and concept_outcome.ranked_configs:
        concept_line = _format_concept_context(concept_outcome)
        if concept_line:
            injection_text = f"{concept_line}\n\n{injection_text}"
            # Compose the injection_method string so downstream telemetry
            # can see both signals fired. If synthesis was active, we get
            # "text+synthesis+concept"; otherwise "text+concept".
            injection_method = (
                f"{injection_method}+concept"
                if injection_method != "text"
                else "text+concept"
            )

    # Pick voicing for the inject_text_context wrapper. The CrystalReader
    # tagged the context with a voicing per top-1's source_kind. In
    # Stage 1 of the GAIA fold-back, only "advisory" is reachable in
    # production (no failure crystals yet) but plumbing the voicing
    # through avoids a separate edit when Stage 2 lands.
    #
    # Phase 7.1 Session 5: bind-storage crystals have summary_text=None
    # so reader.read() returns None for them. When context is None
    # (recall fired but reader.read returned nothing), default voicing
    # to "advisory" — the safe pre-Phase-1.1 default for crystals
    # without explicit voicing metadata.
    # Growth G1 (citations): tag the injection with a citation handle +
    # the cite instruction so the model can attribute its source, and
    # build the manifest the post-response step maps citations back
    # through. Off by default (cite=False) → byte-identical to pre-G1.
    # v1 cites the PRIMARY injected crystal only; the SPREAD branch's
    # second reference is a deferred extension. `injected_text` on the
    # outcome stays the RAW content (grounding uses it); only the
    # messages carry the tagged form.
    citation_manifest: Optional[list[Any]] = None
    injection_for_model = injection_text
    if cite and injection_text:
        from .citations import build_primary_citation
        injection_for_model, citation_manifest = build_primary_citation(
            injection_text,
            crystal_id=top_crystal.id,
            version=getattr(top_crystal, "content_hash", None),
            label=_citation_label(top_crystal, recalled),
            origin=top_crystal.source_kind or "",
        )

    voicing: str = context.voicing if context is not None else "advisory"
    # Tier-as-epistemic-signal (RATIFIED 2026-07-02) — the proxy-side
    # legend, appended AFTER citation tagging so the manifest spans are
    # undisturbed. Same single-source helper as the agent tools; silent
    # when everything contributing is whitelist; fail-safe — a tier
    # lookup hiccup never breaks an injection.
    try:
        from .tier_signal import tier_map, tier_note

        _note = tier_note(await tier_map(store, customer.id, matched_ids or []))
        if _note:
            injection_for_model = (
                f"{injection_for_model}\n\n[Knowledge quality] {_note}"
            )
    except Exception:  # noqa: BLE001 — annotation never breaks retrieval
        logger.debug("retrieval.tier_note_failed", exc_info=True)
    new_messages = inject_text_context(
        messages, injection_for_model, voicing=voicing
    )

    logger.debug(
        "retrieval.injected",
        customer_id=customer.id,
        match_type=match_type,
        top_crystal_id=top_crystal.id,
        top_score=top_score,
        context_chars=len(injection_text),
        routing_decision=routing_result.decision.value,
        routing_margin=routing_result.margin,
        concept_config=(
            concept_outcome.ranked_configs[0][0]
            if concept_outcome and concept_outcome.ranked_configs
            else None
        ),
    )
    return RetrievalOutcome(
        messages=new_messages,
        match_type=match_type,
        injection_method=injection_method,
        matched_crystal_ids=matched_ids,
        top_score=top_score,
        injected_text=injection_text,
        citation_manifest=citation_manifest,
        synthesized_joint_statement=synthesized_joint,
        **routing_fields,
        **_recalled_fields(recalled),
        **_cache_hit_fields(cache_hit_response, cache_hit_crystal_id),
        **_concept_fields(concept_outcome),
    )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------


def _extract_last_user_text(messages: list[dict[str, Any]]) -> str:
    """Return the content of the last user turn, or empty string."""
    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, str):
                return content
            # OpenAI v1 content can be a list of parts; take the text parts.
            if isinstance(content, list):
                parts = [
                    p.get("text", "") for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                return "\n".join(parts)
    return ""


def _fact_with_provenance(fact: Any) -> str:
    """Prepend the sparse-key provenance (Source: Locator) to a recalled
    fact's content so identity queries ("where is X defined?") can name
    the source. The address lives in the fact's prompt_text (sparse key);
    claim_text is only the body. Returns the body unchanged for legacy
    facts whose key isn't a structured sparse key.
    """
    body = (fact.claim_text or "").strip()
    header = _provenance_header(fact.prompt_text)
    return f"{header}\n{body}" if header and body else body


def _citation_label(crystal: Any, recalled: Optional[RecalledFact]) -> str:
    """Best-effort human-readable source label for a citation (Growth G1):
    the sparse-key provenance header when a fact was recalled, else the
    crystal's source path / type / id."""
    if recalled is not None and getattr(recalled, "fact", None) is not None:
        header = _provenance_header(recalled.fact.prompt_text)
        if header and header.strip():
            return header.strip()
    for attr in ("source_path", "crystal_type"):
        val = getattr(crystal, attr, None)
        if val:
            return str(val)
    return getattr(crystal, "id", "") or ""


async def _maybe_run_concept_path(
    *,
    customer_id: str,
    query_text: str,
    decomposer: Optional[Decomposer],
    config_store: Optional[DslConfigStore],
) -> Optional[ConceptRouteOutcome]:
    """Run the concept path if both collaborators are present.

    Returns None if concept path is disabled (missing decomposer or
    config store) OR if the query text is empty. Never raises - any
    router-level failure is swallowed and logged by ConceptRouter.
    """
    if decomposer is None or config_store is None:
        return None
    if not query_text.strip():
        return None

    concept_router = ConceptRouter(
        decomposer=decomposer,
        config_store=config_store,
    )
    try:
        return await concept_router.route(
            tenant_id=customer_id,  # Crystal Cache uses customer_id == tenant_id
            query_text=query_text,
            context={"tenant_id": customer_id},
        )
    except Exception as e:
        # ConceptRouter is supposed to swallow its own errors, but if
        # something slips through, don't fail the request over the
        # additive concept path.
        logger.warning(
            "retrieval.concept_path_unhandled_error",
            customer_id=customer_id,
            error=str(e),
        )
        return None


def _concept_fields(outcome: Optional[ConceptRouteOutcome]) -> dict[str, Any]:
    """Build kwargs for the RetrievalOutcome's concept-path fields."""
    if outcome is None:
        return {
            "concept_top_config": None,
            "concept_top_score": 0.0,
            "concept_payload": None,
            "concept_path_ran": False,
        }
    top_config: Optional[str] = None
    top_score = 0.0
    if outcome.ranked_configs:
        top_config, top_score = outcome.ranked_configs[0]
    return {
        "concept_top_config": top_config,
        "concept_top_score": top_score,
        "concept_payload": (
            outcome.decomposition.payload if outcome.decomposition else None
        ),
        "concept_path_ran": True,
    }


def _routing_fields(result: RoutingResult) -> dict[str, Any]:
    """Build kwargs for the RetrievalOutcome's four-way routing fields.

    Mirrors _concept_fields shape — a small dict that splats into the
    RetrievalOutcome constructor at every return site so we don't repeat
    the same four lines four times.
    """
    return {
        "routing_decision": result.decision,
        "routing_top1": result.top1,
        "routing_top2": result.top2,
        "routing_margin": result.margin,
    }


def _cache_hit_fields(
    response: Optional[str], crystal_id: Optional[str]
) -> dict[str, Any]:
    """Build kwargs for the RetrievalOutcome's cache-hit fields.

    Same splat pattern as _routing_fields. Both args default to None
    in the calling sites where no cache hit was detected; this helper
    just bundles them for the constructor.
    """
    return {
        "cache_hit_response": response,
        "cache_hit_crystal_id": crystal_id,
    }


def _recalled_fields(recalled: Optional[RecalledFact]) -> dict[str, Any]:
    """Build kwargs for the RetrievalOutcome's recall-result fields.

    Same splat pattern as _routing_fields and _cache_hit_fields. The
    recall-result fields must be threaded through every early-return
    site so cache-hit telemetry on otherwise-empty crystals (no
    summary_text, no keyword_fingerprint) doesn't silently lose its
    `recalled_fact_id` — the inspector and downstream logs need to
    see which Fact actually fired.

    None recalled → both fields None. Recall succeeded → surface the
    fact id and cleanup score.
    """
    if recalled is None:
        return {
            "recalled_fact_id": None,
            "recalled_fact_score": None,
        }
    return {
        "recalled_fact_id": recalled.fact.id,
        "recalled_fact_score": recalled.score,
    }


def _format_concept_context(outcome: ConceptRouteOutcome) -> Optional[str]:
    """Produce a single line of structured context from concept-path result.

    Example output:
        "Query intent (routing hint): intent=solve_problem domain=math tone=precise"

    Returns None if there's no useful payload to format.
    """
    if not outcome.decomposition:
        return None
    payload = outcome.decomposition.payload
    if not payload:
        return None

    # Skip meta fields like "asks" - those are compound structure, not
    # flat intent descriptors. For compound queries we don't inject
    # a summary line (future work).
    if "asks" in payload:
        return None

    fields = []
    for key in ("intent", "topic", "domain", "tone", "urgency"):
        val = payload.get(key)
        if isinstance(val, str) and val:
            fields.append(f"{key}={val}")
    if not fields:
        return None
    return "Query intent (routing hint): " + " ".join(fields)
