"""Fact entity — §6 of BUILD_PROPOSAL.md.

A Fact is one prompt-answer pair stored in a crystal's codebook. At write
time the (prompt, answer) pair is encoded, bound, and bundled into the
crystal's `summary_vector`; the answer's native-dim embedding is stored
on the Fact row as `vector` so cleanup at recall time can snap a noisy
unbound vector back to a clean stored answer.

A crystal holds many Facts of many `pair_type`s. The `pair_type` is set
at write time and is immutable. At recall, cleanup matches the unbound
vector against the full crystal codebook (one nearest-neighbor pass over
all Facts in the crystal); the matched Fact's `pair_type` is the
inferred query type. No separate pair_type classifier is needed —
the codebook match yields it as a side effect.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from .crystal import SourceKind


class Fact(BaseModel):
    id: str
    crystal_id: str
    claim_text: str  # human-readable answer text, returned on cleanup match

    # Pair_type, set at write time, IMMUTABLE. Examples:
    # "question_answer" (general FAQ), "date_progress_note" (medical
    # records), "section_content" (document ingestion), "medication_dosage".
    # The DSL (Phase 4) declares which pair_types are valid for a given
    # crystal_type, but enforcement at the runtime level is per-Fact via
    # this column.
    #
    # Required. The legacy default ("question_answer") covers the existing
    # FAQ bank; new write paths must specify explicitly.
    pair_type: str = Field(default="question_answer")

    # Native-dim (e.g. 768 for gtr-t5-base) embedding of the answer text.
    # This is the cleanup target at recall time. Phase 1.1 will populate
    # this; Phase 0.1 leaves the field as the cleanup-codebook slot but
    # does not yet wire bind/bundle math at write time.
    vector: list[float] = Field(default_factory=list)

    # Phase 2 (April 2026, migration 0011): source_kind, answer_value,
    # prompt_text. The cache-hit short-circuit and source-kind-aware
    # injection now live on the matched Fact, not on the Crystal —
    # a crystal holds many pairs and the right voicing/answer comes
    # from the SPECIFIC pair cleanup recovered. See the long note on
    # FactRow in infrastructure/schema.py for the rationale.
    #
    # source_kind: what kind of evidence this Fact carries. Defaults
    # to "model_reasoning" so pre-Phase-2 Facts (FAQ bank imports,
    # document-section pairs) participate normally in injection but
    # never trigger the cache-hit path (which requires both
    # source_kind=="model_reasoning" AND a populated answer_value).
    source_kind: SourceKind = "model_reasoning"

    # answer_value: canonical short answer for the cache-hit path.
    # None when claim_text is itself the answer (legacy imports,
    # document-section Facts, anything non-cache-shaped). The
    # pipeline's PERFECT branch checks `is not None and != ""` before
    # serving directly without invoking upstream.
    answer_value: Optional[str] = None

    # prompt_text: the prompt / key text that was bind-paired into
    # this Fact at write time. Default "" preserves pre-Phase-2 rows
    # whose prompts were lost at import (legacy FAQ bank persisted
    # only the answer side onto the Fact). New writes via Phase 1.1's
    # add_pair_to_crystal populate this going forward, unlocking
    # Phase 6.3's per-crystal cleanup_threshold calibrator and
    # inspector display of "what queries match this codebook entry?".
    prompt_text: str = ""

    # Provenance
    source_doc_id: Optional[str] = None
    extracted_by: Optional[str] = None  # LLM name + version
    verified_by: Optional[str] = None   # employee id / email

    # Decay + usage
    grating_strength: float = 1.0
    hit_count: int = 0
    last_hit_at: Optional[datetime] = None

    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
