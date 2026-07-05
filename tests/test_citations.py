"""Growth G1 — citation primitives (retrieval/citations.py) tests.

The pure substrate: handle assignment, marker parsing, handle→source
mapping, the provenance footer, and the model-facing instruction. No I/O,
so plain sync tests (the store-touching manifest assembly + grounding +
proxy wiring are G1b, tested there).
"""
from __future__ import annotations

from crystal_cache.retrieval.citations import (
    CITATION_MARKER_RE,
    CITE_INSTRUCTION,
    CitationSource,
    assign_handles,
    build_primary_citation,
    extract_claim_span,
    map_citations,
    parse_citations,
    render_sources_footer,
    rewrite_markers,
)


def test_assign_handles_stable_and_deduped():
    handles = assign_handles(["cryst_a", "cryst_b", "cryst_a", "cryst_c"])
    assert handles == {"cryst_a": "1", "cryst_b": "2", "cryst_c": "3"}
    # Empty / falsy ids are skipped, not numbered.
    assert assign_handles(["", "cryst_x"]) == {"cryst_x": "1"}
    assert assign_handles([]) == {}


def test_parse_citations_order_and_dedup():
    text = (
        "The director is Jane Doe [[cc:1]]. The producer is Acme [[cc:2]]. "
        "Jane also wrote it [[cc:1]]."
    )
    assert parse_citations(text) == ["1", "2"]


def test_parse_citations_tolerates_whitespace_and_ignores_malformed():
    assert parse_citations("padded [[cc: 3 ]] here") == ["3"]
    # No handle inside the marker → nothing captured.
    assert parse_citations("broken [[cc:]] marker") == []
    # Ordinary bracketed prose is not a citation.
    assert parse_citations("a list item [1] and [foo]") == []
    assert parse_citations("") == []
    assert parse_citations("no markers at all") == []


def test_map_citations_resolves_in_citation_order_and_drops_unknown():
    manifest = [
        CitationSource(handle="1", crystal_id="cryst_a", label="Script: Scene 5", origin="document"),
        CitationSource(handle="2", crystal_id="cryst_b", label="Policy: 3.2", origin="document"),
        CitationSource(handle="3", crystal_id="cryst_c", label="Note", origin="model_reasoning"),
    ]
    # Cited 3 then 1 then an invented 9 → resolve 3, 1; drop 9.
    resolved = map_citations(["3", "1", "9"], manifest)
    assert [s.crystal_id for s in resolved] == ["cryst_c", "cryst_a"]

    # Nothing cited → nothing resolved.
    assert map_citations([], manifest) == []


def test_render_sources_footer():
    sources = [
        CitationSource(handle="1", crystal_id="cryst_a", label="Script: Scene 5", origin="document"),
        CitationSource(handle="2", crystal_id="cryst_b", label="", origin=""),
    ]
    footer = render_sources_footer(sources)
    assert footer.startswith("Sources:")
    assert "[1] Script: Scene 5 (document)" in footer
    # No label falls back to the crystal id; no origin omits the parenthetical.
    assert "[2] cryst_b" in footer

    # Empty → empty string (nothing to append).
    assert render_sources_footer([]) == ""


def test_marker_regex_and_instruction_are_coherent():
    # The instruction must teach the exact marker the parser recognizes,
    # so the model's emitted form round-trips through parse_citations.
    assert "[[cc:1]]" in CITE_INSTRUCTION
    assert parse_citations("x [[cc:1]] y") == ["1"]
    assert CITATION_MARKER_RE.search("[[cc:1]]") is not None


def test_build_primary_citation_tags_and_manifests():
    tagged, manifest = build_primary_citation(
        "Director: Jane Doe",
        crystal_id="cryst_a",
        version="hash1",
        label="Script: Scene 5",
        origin="document",
    )
    # Manifest carries exactly the one primary source as handle "1".
    assert len(manifest) == 1
    src = manifest[0]
    assert (src.handle, src.crystal_id, src.version) == ("1", "cryst_a", "hash1")
    # Tagged text carries the cite instruction, the handle, and the content,
    # and the handle round-trips through the parser.
    assert "CITATIONS" in tagged
    assert "[[cc:1]]" in tagged
    assert "Director: Jane Doe" in tagged
    assert parse_citations(tagged) == ["1"]


def test_extract_claim_span_returns_enclosing_sentence():
    text = (
        "Intro sentence. The director is Jane Doe [[cc:1]]. "
        "The budget was huge [[cc:2]]."
    )
    # The handle's enclosing sentence, markers stripped.
    assert extract_claim_span(text, "1") == "The director is Jane Doe"
    assert extract_claim_span(text, "2") == "The budget was huge"
    # A handle that wasn't cited yields no span.
    assert extract_claim_span(text, "9") == ""
    assert extract_claim_span("", "1") == ""


def test_rewrite_markers_keeps_grounded_strips_spurious():
    text = "Director is Jane [[cc:1]]. Budget was huge [[cc:2]]. Both true [[cc:1]]."
    # Keep handle 1 (grounded), drop handle 2 (spurious).
    out = rewrite_markers(text, {"1"})
    assert "[1]" in out
    assert "[[cc:" not in out  # no raw markers remain
    assert "[2]" not in out   # spurious handle removed entirely
    # Keeping nothing strips every marker.
    assert "[[cc:" not in rewrite_markers(text, set())
    assert "[1]" not in rewrite_markers(text, set())
