"""Bank-probe pins (Q1-Q4=A, 2026-09-04).

The probe's HTTP plumbing runs on the bench; what must not regress silently
is the CLASSIFICATION: which behavior bucket a miss lands in, and when the
presence test is allowed to say PRESENT/ABSENT vs when it must flag REVIEW.
A wrong ABSENT inflates EXTRACTION-LOSS — the bucket that would trigger the
most drastic (and most expensive) response.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")

_PROBE = Path(__file__).resolve().parents[1] / "scripts" / "bank_probe.py"
_spec = importlib.util.spec_from_file_location("bank_probe", _PROBE)
bp = importlib.util.module_from_spec(_spec)
sys.modules["bank_probe"] = bp
_spec.loader.exec_module(bp)


def test_behavior_taxonomy_buckets():
    assert bp.classify_behavior({"correct": True}) == "OK"
    assert bp.classify_behavior(
        {"tool_calls": [], "model_answer": "I couldn't find any information"}
    ) == "A1"
    assert bp.classify_behavior(
        {"tool_calls": [], "model_answer": "It was Roscioli."}
    ) == "A2"
    assert bp.classify_behavior(
        {"tool_calls": ["crystal_recall"],
         "model_answer": "The search didn't return anything about that."}
    ) == "B1"
    assert bp.classify_behavior(
        {"tool_calls": ["crystal_recall"], "question_type": "multi-session",
         "question": "How many weddings have I attended?",
         "expected_answer": "3",
         "model_answer": "You attended at least 2 weddings."}
    ) == "C1"
    assert bp.classify_behavior(
        {"tool_calls": ["crystal_recall"],
         "question_type": "temporal-reasoning",
         "question": "When did I order it?", "expected_answer": "May 5",
         "model_answer": "You ordered it on June 1."}
    ) == "C2"
    assert bp.classify_behavior(
        {"tool_calls": ["crystal_recall"],
         "question_type": "knowledge-update",
         "question": "Where do I live?", "expected_answer": "Denver",
         "model_answer": "You live in Austin."}
    ) == "C3"
    assert bp.classify_behavior(
        {"tool_calls": ["crystal_recall"],
         "question_type": "single-session-user",
         "question": "What degree do I have?",
         "expected_answer": "Business Administration",
         "model_answer": "A Fine Arts degree."}
    ) == "C4"


def test_presence_thresholds_and_review_band():
    score, v = bp.presence("Business Administration degree",
                           "graduated with a business administration degree")
    assert v == "PRESENT" and score >= 0.75
    _, v = bp.presence("Japanese short-grain rice",
                       "the weather in Tokyo is mild in spring")
    assert v == "ABSENT"
    # Partial overlap lands in the adjudication band, never a hard call.
    _, v = bp.presence("red 1968 Ford Mustang convertible",
                       "you drive a Ford")
    assert v == "REVIEW"


def test_presence_short_numeric_never_absent_on_weak_test():
    # No content tokens (e.g. expected "4"): exact-hit PRESENT, else REVIEW —
    # a short answer must not create false EXTRACTION-LOSS.
    _, v = bp.presence("4", "you tried 4 cuisines this year")
    assert v == "PRESENT"
    _, v = bp.presence("4", "you tried several cuisines this year")
    assert v == "REVIEW"


def test_root_cause_mapping():
    f = bp.classify_root_cause
    assert f("B1", "PRESENT", "PRESENT") == "BLIND"
    assert f("C1", "ABSENT", "PRESENT") == "SHORTFALL"
    assert f("C4", "ABSENT", "ABSENT") == "EXTRACTION-LOSS"
    assert f("B1", "REVIEW", "REVIEW") == "REVIEW"
    assert f("A1", "ABSENT", "PRESENT") == "UNSEARCHED-AVAILABLE"
    assert f("A2", "ABSENT", "ABSENT") == "EXTRACTION-LOSS"
