"""Citation primitives — Growth G1 (trust + the metering rail).

The model cites its sources, where a source is an injected crystal. Citations
make grounded answers trustworthy now and — built once — become the signal
G4's shard ledger meters later (a *cited* crystal, not merely an injected
one, is what proves load-bearing).

This module is the PURE substrate: the marker protocol, the parser, handle
assignment, the source model, and the system-prompt instruction. It has no
I/O and no SQL — it's the citation analogue of v3_push_pull's data
definitions. The store-touching manifest assembly (fetch each crystal's
label/version/origin), the grounding check, the proxy wiring, and the ledger
record live in G1b; the uncited-claim→gap dual in G1c.

Protocol: injected knowledge is tagged with handles like ``[[cc:1]]``. The
model cites inline with the same marker. Handles are 1-based indices over the
injected crystals (short, collision-proof, and easy for the model to emit —
far better than leaking internal crystal ids into the prose). The parser maps
the handles back to crystal ids via the manifest.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# The citation marker the model emits, e.g. "[[cc:1]]". Double brackets + the
# ``cc:`` prefix make it collision-proof against ordinary "[1]" footnotes or
# bracketed prose. Handles are short tokens (digits in practice, but the
# pattern tolerates word characters so the protocol can extend).
CITATION_MARKER_RE = re.compile(r"\[\[cc:\s*([A-Za-z0-9_-]+)\s*\]\]")


# System-prompt addendum, appended ONLY when crystals are injected (so a
# no-context turn never sees citation instructions). Mode-agnostic — citations
# work wherever retrieval injects knowledge, not just in coding mode.
CITE_INSTRUCTION = (
    "CITATIONS: The retrieved knowledge below is tagged with a citation "
    "handle like [[cc:1]]. When a statement in your answer is supported by "
    "tagged knowledge, cite it inline immediately after the statement using "
    'its handle — e.g. "The director is Jane Doe [[cc:1]]." Cite only what '
    "the tagged knowledge actually supports. Do NOT attach a citation to "
    "statements that come from your own general knowledge rather than the "
    "tagged context — leaving those uncited is correct and expected. Never "
    "invent a handle that was not provided."
)


@dataclass(frozen=True)
class CitationSource:
    """One citable source — an injected crystal, addressed by its handle.

    handle      the 1-based token the model cites (the N in ``[[cc:N]]``).
    crystal_id  the injected crystal. Under REPLACE semantics a changed
                source becomes a new crystal id, so the id IS the version.
    version     content_hash pin for the audit trail (None for crystals with
                no source hash, e.g. model-reasoning crystals).
    label       human-readable "Source: Locator" from the fact's sparse key.
    origin      the crystal's source_kind (document, model_reasoning, …).
    """
    handle: str
    crystal_id: str
    version: Optional[str] = None
    label: str = ""
    origin: str = ""


def assign_handles(crystal_ids: list[str]) -> dict[str, str]:
    """Map each injected crystal id to a stable 1-based handle string.

    De-duplicates while preserving first-occurrence order, so the same
    crystal injected twice gets one handle. Returns {crystal_id: "1", …}.
    """
    handles: dict[str, str] = {}
    n = 0
    for cid in crystal_ids:
        if not cid or cid in handles:
            continue
        n += 1
        handles[cid] = str(n)
    return handles


def parse_citations(text: str) -> list[str]:
    """Extract the cited handles from model output, in first-seen order,
    de-duplicated. ``"… [[cc:1]] … [[cc:1]] … [[cc:3]]"`` → ``["1", "3"]``.
    Returns [] when the text is empty or carries no markers."""
    if not text:
        return []
    seen: set[str] = set()
    ordered: list[str] = []
    for match in CITATION_MARKER_RE.finditer(text):
        handle = match.group(1)
        if handle not in seen:
            seen.add(handle)
            ordered.append(handle)
    return ordered


def map_citations(
    handles: list[str], manifest: list[CitationSource]
) -> list[CitationSource]:
    """Resolve cited handles to their sources via the manifest, preserving
    citation order. Handles with no matching manifest entry (the model
    invented a handle, or cited one that wasn't injected) are dropped."""
    by_handle = {src.handle: src for src in manifest}
    resolved: list[CitationSource] = []
    for handle in handles:
        src = by_handle.get(handle)
        if src is not None:
            resolved.append(src)
    return resolved


def render_sources_footer(sources: list[CitationSource]) -> str:
    """Format resolved citations as a user-facing provenance footer. Empty
    string when there are no sources (nothing to append)."""
    if not sources:
        return ""
    lines = ["Sources:"]
    for src in sources:
        label = src.label or src.crystal_id
        origin = f" ({src.origin})" if src.origin else ""
        lines.append(f"[{src.handle}] {label}{origin}")
    return "\n".join(lines)


def build_primary_citation(
    injection_text: str,
    *,
    crystal_id: str,
    version: Optional[str] = None,
    label: str = "",
    origin: str = "",
) -> tuple[str, list[CitationSource]]:
    """Tag a single-source injection so the model can cite it.

    Prepends the cite instruction and a ``[[cc:1]]`` handle to the retrieved
    knowledge and returns ``(tagged_text, manifest)``, where the manifest is
    the one citable source. G1 v1 cites the PRIMARY injected crystal only;
    multi-source citation (e.g. a SPREAD branch's second reference) is a
    deferred extension.
    """
    source = CitationSource(
        handle="1",
        crystal_id=crystal_id,
        version=version,
        label=label,
        origin=origin,
    )
    tagged = f"{CITE_INSTRUCTION}\n\n[[cc:1]]\n{injection_text}"
    return tagged, [source]


_SENTENCE_BOUNDARY = ".!?\n"


def extract_claim_span(text: str, handle: str) -> str:
    """Return the sentence the model attached ``[[cc:<handle>]]`` to — the
    claim the grounding check verifies against the cited source.

    Locates the FIRST marker for this handle and walks out to the enclosing
    sentence boundaries (``. ! ? \n`` or string ends), then strips any
    citation markers from the span. Returns "" when the handle isn't cited.
    Best-effort: a coarse sentence window is enough for a cosine grounding
    signal; a future entailment-grade pass can refine it.
    """
    if not text:
        return ""
    m = re.search(r"\[\[cc:\s*" + re.escape(handle) + r"\s*\]\]", text)
    if m is None:
        return ""
    start = m.start()
    while start > 0 and text[start - 1] not in _SENTENCE_BOUNDARY:
        start -= 1
    end = m.end()
    while end < len(text) and text[end] not in _SENTENCE_BOUNDARY:
        end += 1
    span = CITATION_MARKER_RE.sub("", text[start:end])
    return span.strip()


def rewrite_markers(text: str, kept_handles) -> str:
    """Rewrite the model's raw ``[[cc:N]]`` markers for the user-facing answer:
    a kept handle becomes a clean ``[N]`` footnote ref; any other marker
    (a dropped/spurious citation) is removed. `kept_handles` is any container
    of handle strings that cleared grounding."""
    kept = set(kept_handles)

    def _repl(match: "re.Match[str]") -> str:
        handle = match.group(1)
        return f"[{handle}]" if handle in kept else ""

    return CITATION_MARKER_RE.sub(_repl, text)
