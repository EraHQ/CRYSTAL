"""Judge-audit pins (C ratified 2026-09-06).

What must not regress: the abstention judge prompt must carry the gold
text and credit BOTH correct abstention shapes (v1's omission produced
14-16 false negatives on the headline run); rejudged rows must be honest
about their provenance.
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


lme = _load("longmemeval_harness")
rj = _load("rejudge")


def test_abstention_prompt_carries_gold_and_both_shapes():
    q = {"question_id": "x_abs", "question_type": "multi-session",
         "question": "How many fish in my 30-gallon tank?",
         "answer": "You did not mention a 30-gallon tank."}
    p = lme._judge_prompt(q, "I found your 20-gallon tank; nothing on a 30.")
    # The gold text is IN the prompt (v1 omitted it entirely).
    assert "You did not mention a 30-gallon tank." in p
    # Both correct shapes are named; the incorrect condition is explicit.
    assert "reporting related information" in p
    assert "INCORRECT only if" in p


def test_non_abstention_prompt_unchanged_shape():
    q = {"question_id": "y", "question_type": "single-session-user",
         "question": "What degree do I have?",
         "answer": "Business Administration"}
    p = lme._judge_prompt(q, "A Business Administration degree.")
    assert "Correct answer: Business Administration" in p
    assert "contain or agree" in p


def test_judge_prompt_version_exists_and_dated():
    assert lme.JUDGE_PROMPT_VERSION.startswith("2-abs-gold")


def test_rejudge_row_provenance():
    row = {"question_id": "q1", "correct": False,
           "judge_verdict_raw": "no", "model_answer": "…", "elapsed_s": 1.0}
    out = rj.rejudge_row(row, True, "yes")
    assert out["correct"] is True
    assert out["rejudged"] is True
    assert out["judge_prompt_version"] == lme.JUDGE_PROMPT_VERSION
    assert out["judge_verdict_raw"] == "yes"
    # Original evidence fields untouched; original dict not mutated.
    assert out["elapsed_s"] == 1.0 and row["correct"] is False
