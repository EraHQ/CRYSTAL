"""Unit tests for the hybrid-rank code/prose calibration (stage 1a).

Pure-function tests over `_calibrate_by_subtype` — no DB, no encoder.
The calibration's job: on a mixed `content_chunk` candidate pool, lift
code crystals out from under prose that gtr-t5-base scores systematically
higher (same-language bias), without changing the cosine values that
downstream thresholds depend on.
"""
from crystal_cache.retrieval.v3_routers import _calibrate_by_subtype


def _row(fid, score, key, cid=None, pt="content_chunk"):
    """A FactVectorStore.search(with_keys=True) row: (fid, cid, pt, cos, key)."""
    return (fid, cid or f"crys_{fid}", pt, score, key)


def test_empty_input_returns_empty():
    assert _calibrate_by_subtype([]) == []


def test_strips_keys_to_four_tuples_and_preserves_cosine():
    rows = [_row("a", 0.80, "Document|Passage 1"), _row("b", 0.70, "Code|x.py|f")]
    out = _calibrate_by_subtype(rows)
    assert all(len(r) == 4 for r in out)
    # Original cosine is preserved per fact regardless of reorder.
    by_id = {r[0]: r[3] for r in out}
    assert by_id["a"] == 0.80
    assert by_id["b"] == 0.70


def test_code_lifted_above_weaker_prose():
    # Prose scores systematically higher than code — the bias we cancel.
    rows = [
        _row("p1", 0.80, "Document|Passage 1"),
        _row("p2", 0.78, "Document|Passage 2"),
        _row("p3", 0.76, "Document|Passage 3"),
        _row("c1", 0.70, "Code|app.py|handler"),
        _row("c2", 0.62, "Code|app.py|helper"),
    ]
    out = [r[0] for r in _calibrate_by_subtype(rows)]
    # The best-in-bucket code crystal now ranks above the prose crystals
    # it was buried under by raw cosine.
    assert out.index("c1") < out.index("p2")
    assert out.index("c1") < out.index("p3")
    # But calibration interleaves by relative strength — it does NOT
    # blindly force code to the top; the overall-strongest match leads.
    assert out[0] == "p1"


def test_single_modality_preserves_cosine_order():
    # All prose: within-bucket z-normalization is monotonic in cosine, so
    # the order is exactly the raw-cosine order. Calibration only changes
    # ordering when both modalities are present.
    rows = [
        _row("p1", 0.55, "Document|A"),
        _row("p2", 0.80, "Document|B"),
        _row("p3", 0.40, "Document|C"),
    ]
    out = [r[0] for r in _calibrate_by_subtype(rows)]
    assert out == ["p2", "p1", "p3"]


def test_singleton_bucket_is_neutral_not_promoted():
    # A lone code crystal can't form a distribution, so it gets a neutral
    # 0.0 and sits mid-pack — not spuriously promoted to #1.
    rows = [
        _row("p1", 0.80, "Document|A"),   # prose mean 0.70, std 0.10
        _row("p2", 0.60, "Document|B"),   # -> z(p1)=+1.0, z(p2)=-1.0
        _row("c1", 0.50, "Code|x.py|f"),  # singleton -> neutral 0.0
    ]
    out = [r[0] for r in _calibrate_by_subtype(rows)]
    assert out == ["p1", "c1", "p2"]
