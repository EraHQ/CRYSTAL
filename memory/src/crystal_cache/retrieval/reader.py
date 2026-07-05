"""Crystal reader — crystal → injectable text context.

For Group D we do NOT read facts. Fact-level retrieval belongs to the
V3 fact lane (FactVectorStore + the routers); what this reader produces
is a compact text summary of what a crystal is "about", using signals
we already have:

  - crystal.summary_text (optional, from bank-construction writers)
  - crystal.keyword_fingerprint (always present after bank construction)
  - crystal's latest CrystalDiagnostic top_help_query_exemplars (when the
    learning loop has had a chance to run)

The result is a short string suitable as a system-message prefix. It's
not brilliant — in particular, the "exemplar questions" snippet is useful
only once the bank has seen real traffic. But it's a real signal that
the model can use, and it improves automatically as the system collects
telemetry and Group E adds summary_text.

STRUCTURE of the output (when we have everything):
    "Relevant context: {summary_text or keyword_fingerprint}.
     Prior example questions handled by this context:
     - {exemplar 1}
     - {exemplar 2}
     - {exemplar 3}"

When we only have the keyword fingerprint:
    "Relevant context: {joined keywords}."

When the crystal has nothing useful, we return None and the caller
treats this as a low-match fallback (no injection).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional

from ..models import Crystal

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore


# Max characters for the context snippet we inject. Above this we start
# bleeding the user's actual prompt out of the model's attention budget.
# Content chunks are exempt from this cap — their value IS the full text.
MAX_INJECTION_CHARS = 800
MAX_EXEMPLARS = 3


# Voicing register for the injected context. Drives how text_injection
# renders the system-message wrapper around this snippet:
#
#   "advisory"   — success-derived crystal (model_reasoning, web_search_result,
#                  code_execution_result). Framed as reference material:
#                  "may or may not be relevant." The model decides whether
#                  to use it.
#
#   "imperative" — failure-derived crystal (failed_reasoning). Framed as
#                  a constraint: "apply these rules to your answer." These
#                  are imperative one-sentence rules extracted by the
#                  failure-reflection helper from prior wrong attempts.
#                  Mixing imperative and advisory voices in the same
#                  injection regressed accuracy on the GAIA benchmark;
#                  keeping them separate is load-bearing.
#
# In Stage 1 of the GAIA fold-back, only "advisory" is reachable in
# production because failure crystals don't get authored yet (Stage 2).
# Plumbed now so Stage 2 doesn't have to touch this module again.
Voicing = Literal["advisory", "imperative", "informational"]


@dataclass
class CrystalContext:
    """The material a crystal contributes to an injected prompt."""
    text: str
    used_summary_text: bool
    used_keyword_fingerprint: bool
    num_exemplars: int
    voicing: Voicing = "advisory"


class CrystalReader:
    """Stateless. Async only because it reads diagnostics from the store."""

    def __init__(self, store: "MetadataStore") -> None:
        self._store = store

    async def read(self, crystal: Crystal) -> Optional[CrystalContext]:
        """Build an injection-ready context snippet for this crystal.

        Returns None if the crystal has nothing useful to contribute.
        """
        # Content chunk crystals: return the verbatim fact text directly.
        # These are document chunks (scenes, sections, passages) stored
        # as a single fact with pair_type='content_chunk'. The value IS
        # the content — no summarization needed.
        if crystal.build_method == "content_chunk" or crystal.source_kind == "document_chunk":
            facts = await self._store.list_facts_for_crystal(crystal.id)
            if facts:
                # Content chunks have exactly one fact with the full text
                chunk_text = facts[0].claim_text or facts[0].answer_value or ""
                if chunk_text.strip():
                    # Prepend a provenance header (Source: Locator) drawn from
                    # the fact's sparse key, which is stored as its prompt_text.
                    # Without this the model sees only verbatim content and
                    # cannot answer identity queries ("where is X defined?").
                    body = chunk_text.strip()
                    header = _provenance_header(facts[0].prompt_text)
                    text = f"{header}\n{body}" if header else body
                    return CrystalContext(
                        text=text,
                        used_summary_text=False,
                        used_keyword_fingerprint=False,
                        num_exemplars=0,
                        voicing="advisory",
                    )
            return None

        # 1. Headline content: summary_text if populated, else keywords,
        #    else fall back to reading facts directly.
        head = None
        used_summary = False
        used_fingerprint = False
        if crystal.summary_text:
            head = crystal.summary_text.strip()
            used_summary = True
        elif crystal.keyword_fingerprint and len(crystal.keyword_fingerprint) > 0:
            keywords = ", ".join(crystal.keyword_fingerprint[:10])
            head = f"this context covers: {keywords}"
            used_fingerprint = True

        if not head:
            # No summary or keywords — build context from facts directly.
            # This is the common case for document-extracted crystals.
            facts = await self._store.list_facts_for_crystal(crystal.id)
            if not facts:
                return None

            lines = []
            for f in facts[:10]:  # Cap at 10 facts per crystal
                key = f.prompt_text or ""
                val = f.claim_text or f.answer_value or ""
                if key and val:
                    lines.append(f"{key}: {val}")
                elif val:
                    lines.append(val)

            if not lines:
                return None

            text = "\n".join(lines)
            if len(text) > MAX_INJECTION_CHARS:
                text = text[:MAX_INJECTION_CHARS - 1].rstrip() + "\u2026"

            return CrystalContext(
                text=text,
                used_summary_text=False,
                used_keyword_fingerprint=False,
                num_exemplars=0,
                voicing=_voicing_for_source_kind(crystal.source_kind),
            )

        # 2. Optional exemplars from the latest diagnostic.
        exemplars: list[str] = []
        diag = await self._store.get_latest_diagnostic(crystal.id)
        if diag and diag.top_help_query_exemplars:
            for ex in diag.top_help_query_exemplars[:MAX_EXEMPLARS]:
                ex = ex.strip()
                if ex:
                    exemplars.append(ex)

        # 3. Assemble
        parts = [f"Relevant context: {head}."]
        if exemplars:
            parts.append("Prior example questions handled by this context:")
            for ex in exemplars:
                parts.append(f"- {ex}")
        text = "\n".join(parts)

        # Hard cap — truncate rather than return nothing.
        if len(text) > MAX_INJECTION_CHARS:
            text = text[: MAX_INJECTION_CHARS - 1].rstrip() + "\u2026"

        return CrystalContext(
            text=text,
            used_summary_text=used_summary,
            used_keyword_fingerprint=used_fingerprint,
            num_exemplars=len(exemplars),
            voicing=_voicing_for_source_kind(crystal.source_kind),
        )


def _voicing_for_source_kind(source_kind: str) -> Voicing:
    """Map a Crystal.source_kind to its rendering voicing.

    Success-derived crystals (model_reasoning, web_search_result,
    code_execution_result) are advisory. Failure-derived crystals
    (failed_reasoning) are imperative. Unknown kinds default to
    advisory — the more conservative choice when the source_kind
    enum grows in future migrations.
    """
    if source_kind == "failed_reasoning":
        return "imperative"
    return "advisory"


def _provenance_header(prompt_text: str) -> str:
    """Build a provenance breadcrumb from a fact's unified sparse key.

    Content-chunk facts store their unified sparse key — a wide->specific
    path, e.g. 'Code | sparse_keys.py | generate_sparse_key' or
    'Film | Corporate Mistletoe | Script | Scene 5' — as prompt_text.
    Surfacing the path lets the model answer identity queries by naming
    where the content lives. Returns "" for depth-1 keys that carry no
    path structure, so the caller injects the body unchanged.
    """
    if not prompt_text:
        return ""
    from .sparse_key import parse_key
    sk = parse_key(prompt_text)
    if sk.depth < 2:
        return ""
    return " > ".join(sk.segments)
