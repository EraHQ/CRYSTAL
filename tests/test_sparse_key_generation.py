"""Tests for unified sparse-key generation (encoding/sparse_keys.py).

The generator derives a wide->specific PATH from free text via an LLM that
returns an ordered segment list. The LLM is stubbed here (no API key needed),
so these tests cover the parse / format / fallback control flow — not live
model output. The keyless path (depth-1 fallback) is what CI exercises by
default; these stubs additionally cover the multi-segment success path that
only fires when an LLM key is present.
"""
from __future__ import annotations

import pytest

from crystal_cache.encoding import sparse_keys
from crystal_cache.encoding.sparse_keys import generate_sparse_key


def _stub_returning(segments):
    """A stand-in for the cached LLM call that returns a fixed segment tuple."""
    def _fn(text_hash, truncated_text):
        return tuple(segments)
    return _fn


def _stub_raising(exc):
    def _fn(text_hash, truncated_text):
        raise exc
    return _fn


def test_multi_segment_path(monkeypatch):
    monkeypatch.setattr(
        sparse_keys, "_cached_generate",
        _stub_returning(["Infrastructure", "Database", "Production", "PostgreSQL 16"]),
    )
    key = generate_sparse_key("We use PostgreSQL 16 for all production services.")
    assert key == "Infrastructure|Database|Production|PostgreSQL 16"
    assert key.count("|") == 3  # depth-4 wide->specific path


def test_pipe_inside_segment_is_sanitized(monkeypatch):
    # The model can echo a '|' from its input; format_key must collapse it to a
    # space so it never reads as a path separator.
    monkeypatch.setattr(
        sparse_keys, "_cached_generate", _stub_returning(["a|b", "c"]),
    )
    assert generate_sparse_key("x") == "a b|c"


def test_empty_segments_fall_back_to_depth1(monkeypatch):
    monkeypatch.setattr(sparse_keys, "_cached_generate", _stub_returning([]))
    key = generate_sparse_key("the team primary database value")
    assert key == "the team primary database value"
    assert "|" not in key


def test_llm_failure_falls_back_to_first_8_words(monkeypatch):
    # Keyless behavior: the client call raises; fallback=True (default) yields a
    # depth-1 key from the first 8 words.
    monkeypatch.setattr(
        sparse_keys, "_cached_generate", _stub_raising(RuntimeError("no api key")),
    )
    key = generate_sparse_key("what database does the team run in prod anyway tell me")
    assert key == "what database does the team run in prod"
    assert "|" not in key


def test_llm_failure_no_fallback_reraises(monkeypatch):
    monkeypatch.setattr(
        sparse_keys, "_cached_generate", _stub_raising(RuntimeError("no api key")),
    )
    with pytest.raises(RuntimeError):
        generate_sparse_key("anything", fallback=False)


def test_parse_segment_array_tolerates_fences_and_prose():
    # The reply parser accepts a bare array, a fenced array, and an array
    # embedded in prose; it drops empties and coerces to stripped strings.
    p = sparse_keys._parse_segment_array
    assert p('["A", "B", "C"]') == ["A", "B", "C"]
    assert p('```json\n["Policy", "PTO"]\n```') == ["Policy", "PTO"]
    assert p('Here you go: ["Code", "module", "symbol"] done') == ["Code", "module", "symbol"]
    assert p("I cannot help with that.") == []
    assert p('["", "  ", "X"]') == ["X"]
