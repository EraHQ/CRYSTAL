"""Cognition-hygiene helpers — C4 (JSON extraction) and C5 (research dedup).

Pure-function coverage for the two sprint fixes:
  - cognition.roles._extract_json_object  (C4)
  - retrieval.v3_signal_handler._topics_duplicate  (C5)

Both were landed against idle-log evidence (2026-06-09). The store-side
pieces (list_open_research_topics, the handle_signals dedup loop, the
roles call sites) are exercised by the existing cognition/signal-handler
suites; these tests pin the decision logic that those paths depend on.
"""
from __future__ import annotations

from types import SimpleNamespace

from crystal_cache.cognition.engine import _should_park_unanswerable
from crystal_cache.cognition.groundedness import assess_groundedness
from crystal_cache.cognition.retrieval_adapter import (
    DEFAULT_SEARCH_PAIR_TYPES,
    _content_text_from_findings,
    _filter_and_cap_findings,
    _normalize_source_output,
    _normalize_web_output,
    _tools_for_pair_types,
)
from crystal_cache.cognition.roles import _extract_json_object
from crystal_cache.retrieval.v3_signal_handler import _topics_duplicate
from crystal_cache.workers.cognition import (
    GAP_BACKOFF_BASE_SECONDS,
    GAP_MAX_ATTEMPTS,
    GAP_QUICK_DELAY_SECONDS,
    GAP_QUICK_RETRIES,
    _record_gap_failure,
)


# ---------------------------------------------------------------------------
# C4 — _extract_json_object
# ---------------------------------------------------------------------------

def test_extract_direct_json():
    assert _extract_json_object('{"approved": true, "score": 0.9}')["approved"] is True


def test_extract_fenced_json():
    # The shape that showed up as a parse failure in the idle log.
    assert _extract_json_object('```json\n{"approved": false, "score": 0.4}\n```') == {
        "approved": False, "score": 0.4
    }


def test_extract_fence_with_surrounding_prose():
    out = _extract_json_object('Here is my evaluation:\n```json\n{"approved": true}\n```\nLet me know!')
    assert out["approved"] is True


def test_extract_preamble_no_fence_brace_match():
    assert _extract_json_object('Sure! {"approved": false, "score": 0.2} done')["score"] == 0.2


def test_extract_handles_braces_inside_strings():
    assert _extract_json_object('{"reasoning": "use {x} carefully", "score": 0.5}')["score"] == 0.5


def test_extract_handles_escaped_quotes():
    out = _extract_json_object('{"reasoning": "the \\"sparse_key\\" entity", "approved": false}')
    assert out["approved"] is False


def test_extract_truncated_returns_none():
    # max_tokens cutoff leaves JSON unterminated → None → caller fails closed.
    raw = '```json\n{\n  "approved": false,\n  "reasoning": "conflates sparse_key with generate_sparse'
    assert _extract_json_object(raw) is None


def test_extract_empty_returns_none():
    assert _extract_json_object("") is None
    assert _extract_json_object("no json here at all") is None


# ---------------------------------------------------------------------------
# C5 — _topics_duplicate
# ---------------------------------------------------------------------------

def test_dup_filler_only_difference_merges():
    assert _topics_duplicate(
        "locate sparse_key in the codebase", "locate sparse_key in codebase"
    ) is True


def test_dup_punctuation_case_merges():
    assert _topics_duplicate("sparse_key definition", "Sparse_key  definition!") is True


def test_dup_distinct_observed_topics_not_merged():
    # The two real topics from the idle log: related but genuinely distinct.
    assert _topics_duplicate(
        "sparse_keys module structure, functions, classes, and DEFAULT_MODEL configuration",
        "sparse_key definition, usage patterns, and file locations in codebase",
    ) is False


def test_dup_distinct_content_word_not_merged():
    assert _topics_duplicate(
        "list functions in sparse_keys", "list classes in sparse_keys"
    ) is False
    assert _topics_duplicate("sparse_key definition", "sparse_key benchmarks") is False


def test_dup_empty_guard():
    assert _topics_duplicate("", "anything") is False


# ---------------------------------------------------------------------------
# C3 — assess_groundedness
# ---------------------------------------------------------------------------

def _step(action, output):
    return SimpleNamespace(action=action, output=output)


def test_grounded_path_present_in_retrieval_corpus():
    steps = {
        1: _step("crystal_key_scan", {
            "content_text": "Code|crystal_cache/encoding/sparse_keys.py::generate_sparse_key",
            "findings": [{
                "key": "Code|crystal_cache/encoding/sparse_keys.py",
                "content": "def generate_sparse_key(text): ...",
            }],
        }),
    }
    deliverable = "generate_sparse_key lives in crystal_cache/encoding/sparse_keys.py and returns str."
    out = assess_groundedness(deliverable, steps)
    assert out["verdict"] == "grounded"
    assert out["ungrounded_paths"] == []


def test_ungrounded_cert_phrases_flagged():
    # The incident: confident "all checks passed" with no retrieval at all.
    deliverable = (
        "Verification status: fully verified. All checks passed: file exists, "
        "function name confirmed, signature confirmed, import path verified."
    )
    out = assess_groundedness(deliverable, {})
    assert out["verdict"] == "ungrounded"
    assert "all checks passed" in out["cert_phrases"]
    assert out["had_retrieval"] is False


def test_ungrounded_path_not_in_corpus():
    steps = {1: _step("crystal_search", {"content_text": "unrelated facts", "findings": []})}
    deliverable = "The function is defined in crystal_cache/encoding/sparse_keys.py."
    out = assess_groundedness(deliverable, steps)
    assert out["verdict"] == "ungrounded"
    assert "crystal_cache/encoding/sparse_keys.py" in out["ungrounded_paths"]


def test_composition_output_does_not_count_as_grounding():
    # A path that only appears in an analyze (LLM) step is NOT corpus.
    steps = {
        1: _step("crystal_search", {"content_text": "unrelated", "findings": []}),
        2: _step("analyze", {"content": "I think it's in crystal_cache/encoding/sparse_keys.py"}),
    }
    deliverable = "It is defined in crystal_cache/encoding/sparse_keys.py."
    out = assess_groundedness(deliverable, steps)
    assert out["verdict"] == "ungrounded"


def test_clean_deliverable_is_grounded():
    steps = {1: _step("crystal_search", {"content_text": "facts about keys", "findings": []})}
    deliverable = "The sparse key generator produces a short semantic key from input text."
    out = assess_groundedness(deliverable, steps)
    assert out["verdict"] == "grounded"


# ---------------------------------------------------------------------------
# C2 — _should_park_unanswerable (answerability gate)
# ---------------------------------------------------------------------------

def _pstep(sid, action_value):
    return SimpleNamespace(id=sid, action=SimpleNamespace(value=action_value))


def _plan(*steps):
    return SimpleNamespace(steps=list(steps))


def _env_with_findings(findings_by_step):
    outs = {
        sid: SimpleNamespace(output={"findings": f})
        for sid, f in findings_by_step.items()
    }
    return SimpleNamespace(step_outputs=outs)


def test_park_when_retrieval_empty_and_composition_remains():
    plan = _plan(_pstep(1, "crystal_search"), _pstep(2, "crystal_key_scan"),
                 _pstep(3, "analyze"), _pstep(4, "format"))
    env = _env_with_findings({1: [], 2: []})  # both retrieval steps ran, 0 findings
    executed = {1, 2}  # composition steps 3,4 not yet run
    assert _should_park_unanswerable(plan, executed, env) is True


def test_no_park_when_retrieval_found_something():
    plan = _plan(_pstep(1, "crystal_search"), _pstep(2, "analyze"))
    env = _env_with_findings({1: [{"fact_id": "f1"}]})
    assert _should_park_unanswerable(plan, {1}, env) is False


def test_no_park_when_retrieval_not_done():
    plan = _plan(_pstep(1, "crystal_search"), _pstep(2, "crystal_key_scan"),
                 _pstep(3, "analyze"))
    env = _env_with_findings({1: []})  # step 2 hasn't run yet
    assert _should_park_unanswerable(plan, {1}, env) is False


def test_no_park_when_no_composition_remains():
    plan = _plan(_pstep(1, "crystal_search"), _pstep(2, "analyze"))
    env = _env_with_findings({1: []})
    assert _should_park_unanswerable(plan, {1, 2}, env) is False


def test_no_park_when_no_retrieval_steps():
    plan = _plan(_pstep(1, "analyze"), _pstep(2, "format"))
    env = _env_with_findings({})
    assert _should_park_unanswerable(plan, {1}, env) is False


# ---------------------------------------------------------------------------
# C1 — _record_gap_failure (two-phase gap-fill backoff)
# ---------------------------------------------------------------------------

def test_gap_backoff_quick_then_exponential_then_park():
    backoff: dict = {}
    gid = "gap_1"
    # Quick-retry phase: the first GAP_QUICK_RETRIES failures use the short delay.
    for n in range(1, GAP_QUICK_RETRIES + 1):
        bo = _record_gap_failure(backoff, gid, now=0.0)
        assert bo["attempts"] == n
        assert bo["next_eligible"] == GAP_QUICK_DELAY_SECONDS
    # First post-quick failure backs off by the base (~1h), not 2x or 4x.
    bo = _record_gap_failure(backoff, gid, now=0.0)
    assert bo["attempts"] == GAP_QUICK_RETRIES + 1
    assert bo["next_eligible"] == GAP_BACKOFF_BASE_SECONDS
    # Next failure reaches the park threshold.
    bo = _record_gap_failure(backoff, gid, now=0.0)
    assert bo["attempts"] >= GAP_MAX_ATTEMPTS


def test_gap_backoff_schedule_reaches_park_after_one_exponential_step():
    # With the default constants the gap should park having spent exactly one
    # exponential (1h) step after the quick probes — not several.
    assert GAP_MAX_ATTEMPTS == GAP_QUICK_RETRIES + 2


def test_gap_backoff_next_eligible_uses_now_offset():
    backoff: dict = {}
    bo = _record_gap_failure(backoff, "gap_2", now=1000.0)
    assert bo["next_eligible"] == 1000.0 + GAP_QUICK_DELAY_SECONDS


# ---------------------------------------------------------------------------
# B — cognition retrieval adapter (pure helpers)
# ---------------------------------------------------------------------------

def test_tools_for_pair_types_default_is_both():
    assert _tools_for_pair_types(DEFAULT_SEARCH_PAIR_TYPES) == {
        "content_search", "knowledge_search",
    }


def test_tools_for_pair_types_content_only():
    assert _tools_for_pair_types(["content_chunk"]) == {"content_search"}


def test_tools_for_pair_types_knowledge_only():
    assert _tools_for_pair_types(["question_answer"]) == {"knowledge_search"}
    assert _tools_for_pair_types(["entity_attribute"]) == {"knowledge_search"}


def test_tools_for_pair_types_exotic_is_empty():
    assert _tools_for_pair_types(["cached_solution"]) == set()


def test_filter_drops_pairtypes_outside_target_and_dedups():
    merged = [
        {"fact_id": "f1", "pair_type": "content_chunk", "content": "a"},
        {"fact_id": "f2", "pair_type": "question_answer", "content": "b"},
        {"fact_id": "f3", "pair_type": "entity_relationship", "content": "c"},
        {"fact_id": "f4", "pair_type": "entity_attribute", "content": "d"},
        {"fact_id": "f1", "pair_type": "content_chunk", "content": "dup"},
    ]
    out = _filter_and_cap_findings(merged, ["content_chunk", "question_answer"], 10)
    assert [f["fact_id"] for f in out] == ["f1", "f2"]


def test_filter_qa_only_drops_content_chunk():
    merged = [
        {"fact_id": "f1", "pair_type": "content_chunk", "content": "a"},
        {"fact_id": "f2", "pair_type": "question_answer", "content": "b"},
    ]
    out = _filter_and_cap_findings(merged, ["question_answer"], 10)
    assert [f["fact_id"] for f in out] == ["f2"]


def test_filter_caps_at_k():
    merged = [
        {"fact_id": "f1", "pair_type": "content_chunk", "content": "a"},
        {"fact_id": "f2", "pair_type": "content_chunk", "content": "b"},
    ]
    out = _filter_and_cap_findings(merged, ["content_chunk"], 1)
    assert [f["fact_id"] for f in out] == ["f1"]


def test_content_text_joins_nonempty_findings():
    findings = [{"content": "one"}, {"content": ""}, {"content": "two"}]
    assert _content_text_from_findings(findings) == "one\n\ntwo"


def test_normalize_web_output_adds_cognition_keys():
    out = _normalize_web_output({"note": "stub", "query": "q", "results": []})
    assert out["findings"] == []
    assert out["results_count"] == 0
    assert out["content_text"] == ""


def test_normalize_source_read_sets_content_text():
    out = _normalize_source_output(
        {"op": "read", "path": "a.py", "content": "def f(): pass"}
    )
    assert "def f()" in out["content_text"]
    assert out["content"] == "def f(): pass"  # structured key left intact


def test_normalize_source_search_summarizes_matches():
    out = _normalize_source_output(
        {"op": "search", "query": "f",
         "matches": [{"path": "a.py", "line": 3, "text": "def f():"}]}
    )
    assert "a.py:3" in out["content_text"]
    assert out["matches"]  # structured key left intact
