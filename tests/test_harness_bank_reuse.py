"""Bank-reuse mode pins (Q1-Q4=A, 2026-09-04).

The ask/probe plumbing is HTTP-side and exercised on the bench; what must
never regress silently is WHICH rows count as proof of a complete,
reusable bank. build_bank_map is the pure evidence scanner: it feeds the
reuse decision that skips ~$0.35 of ingest per question, so a false
positive re-asks against a partial bank (wrong number) and a false
negative re-pays ingest (wrong bill).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("httpx")  # harness imports it at module top

_HARNESS = Path(__file__).resolve().parents[1] / "scripts" / "longmemeval_harness.py"
_spec = importlib.util.spec_from_file_location("lme_harness", _HARNESS)
lme = importlib.util.module_from_spec(_spec)
sys.modules["lme_harness"] = lme
_spec.loader.exec_module(lme)


def test_bank_map_requires_customer_and_complete_ingest():
    rows = [
        # Ingest finished, ask failed -> the bank is real evidence.
        {"question_id": "q1", "customer_id": "cus_a", "sessions_ingested": 50,
         "error": "AgentTurnError: stop_reason=error"},
        # Died before/inside ingest (no sessions_ingested) -> no evidence.
        {"question_id": "q2", "customer_id": "cus_b",
         "error": "HTTPStatusError: 429 Too Many Requests"},
        # No tenant recorded (died at creation) -> no evidence.
        {"question_id": "q3", "sessions_ingested": 50, "error": "boom"},
    ]
    assert lme.build_bank_map(rows) == {
        "q1": {"customer_id": "cus_a", "bank_sessions": 50},
    }


def test_bank_map_carries_reused_row_evidence_forward():
    # A reused row ingests nothing this attempt (sessions_ingested=0) but
    # carries bank_sessions; an ask-side failure must not demote the bank.
    rows = [{
        "question_id": "q1", "customer_id": "cus_a", "reused_bank": True,
        "bank_sessions": 50, "sessions_ingested": 0,
        "error": "ReadTimeout: timed out",
    }]
    assert lme.build_bank_map(rows) == {
        "q1": {"customer_id": "cus_a", "bank_sessions": 50},
    }


def test_bank_map_zero_sessions_is_not_evidence():
    rows = [{"question_id": "q1", "customer_id": "cus_a",
             "sessions_ingested": 0, "error": "boom"}]
    assert lme.build_bank_map(rows) == {}


def test_bank_map_graded_rows_also_carry_evidence():
    # A graded row proves its bank too (resume skips the question, but the
    # map must not choke on non-error rows).
    rows = [{"question_id": "q1", "customer_id": "cus_a",
             "sessions_ingested": 50, "correct": True}]
    assert lme.build_bank_map(rows)["q1"]["bank_sessions"] == 50
