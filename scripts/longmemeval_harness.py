"""LongMemEval harness — memory benchmark against a live Crystal Cache server.

Backlog §1 "Memory benchmark": per-question loop of ingest-history-into-a-
fresh-bank -> ask via /v1/agent/messages -> LLM-judge grade, with token
accounting from the agent envelope and retrieval telemetry from query_logs.

SURFACE (LAUNCH_PLAN L7-Q1=A, ratified 2026-08-28): the question is asked
through the PRODUCT surface — the agent door, /v1/agent/messages — not the
retiring chat proxy. The number measures what a customer actually gets:
the C2 retrieval preflight on the opening turn plus the agent's own
retrieval tools (knowledge_search / key_scan / content_search /
depth_search), one final answer. `temperature` reaches the agent loop
since audit (e) stage 1.11; the harness pins 0 (Guarantee #6).

WHAT IT MEASURES
  Long-term conversational memory: each LongMemEval question ships a set of
  prior chat sessions (the haystack). The harness ingests those sessions as
  transcript documents into a FRESH customer bank (full crystallization:
  chunk -> extract -> approve), then asks the question once through the
  agent door. The controlling model only wins if Crystal Cache surfaced the
  right memory — via the preflight injection or its retrieval tools.

=============================================================================
BENCHMARK INTEGRITY — how this run defends against "you cheated" claims
=============================================================================
This harness is built so a third party can reproduce and audit the number.
The guarantees, and where each is enforced:

  1. Question-blind ingestion. Sessions are crystallized BEFORE the question
     is asked; the ingest pipeline never sees the question or gold answer.
  2. Blind retrieval. The harness never reads LongMemEval's evidence labels
     (`answer_session_ids`, per-turn `has_answer`) — they are stripped and
     never sent to the server. Retrieval must find the memory on its own.
  3. One final answer. The harness reads exactly ONE final response per
     question (`final_text` of the agent envelope) and grades it once —
     never multiple graded answer attempts, never external info. The agent
     may run several RETRIEVAL rounds inside the turn (the C2 preflight,
     then its retrieval tools across loop iterations); that is the memory
     system doing its job (Mem0/Zep retrieve multi-hop internally too).
     Only multiple graded answers or external information are forbidden.
     External information, concretely: the agent's tool set includes
     web_search (dead unless CC_WEB_SEARCH_PROVIDER is set — it returns an
     explicit error) and web_fetch (NO provider gate — reachable on any
     turn). The manifest records the required server state (provider
     unset), every row records the tool names the agent called, and the
     summary counts rows where an external tool ran — a nonzero count is
     printed as a warning and disqualifies the run as a memory number.
     Disclosure: the agent also carries WRITE tools (crystal_write,
     record_gap, propose_correction, the >=0.9 auto-commit). A mid-turn
     write is the model's own note-taking, not external information; the
     per-row tool list shows exactly what ran. A turn that ends with
     stop_reason="error" is a harness ERROR (counted, never graded — the
     agent's error text is an apology string a judge could mistake for an
     abstention).
  4. Clean room. Each fresh customer has any general-bank subscriptions
     STRIPPED, then verified zero, before ingest — so nothing but the ingested
     sessions can enter the answer context. (New customers may carry a default
     subscription such as 'general:legacy'; removing it only makes the test
     cleaner, and the stripped list is printed + noted in the manifest for
     disclosure.) Prefer a fresh benchmark DB with no general banks, and run
     with background mutation OFF (CC_ENABLE_METACOGNITION_WORKER=0) so the
     bank can't change mid-sweep.
  5. Variant honesty. The `oracle` variant contains ONLY the evidence
     sessions — retrieval is trivial there, so it is reported as a
     retrieval-ISOLATED upper bound, NEVER as a comparable LongMemEval score.
     Headline numbers come from `_s` (~115k-token, ~50-session haystack) or
     `_m` (500 sessions).
  6. Fixed decoding. Answer and judge both request temperature 0. Honoured
     by models that still take sampling params (Haiku 4.5 / Sonnet 4.6,
     the defaults here); Anthropic's adaptive models ignore sampling by
     design and the server seam drops it for them. The manifest records
     both model ids, so the operating point is always reconstructible.
  7. Honest denominator. Errored questions are counted as attempted (not
     silently dropped); the summary shows correct/attempted AND
     correct/graded plus the error count, so nothing can be hidden either way.
  8. Full manifest. The first line of --out (and stdout) records the dataset
     SHA-256, variant, model ids, temperature, counts, and the expected
     server flags — everything needed to reproduce the run. Token cost per
     row is total_input_tokens (uncached + cache_read + cache_creation);
     the uncached `prompt_tokens` alone understates the bill ~100x once
     the system prompt is cached.
  9. Per-question audit trail. Each row records the question, its date and
     the expected answer beside the model answer and verdict; the agent's
     stop_reason, iteration count, every tool call with the INPUT the
     agent chose, and the crystal/fact ids that appeared in tool outputs
     (surfaced_crystal_ids — the live audit column on this lane); plus
     match_type / injection_method / matched_facts from the query log.
     NOTE: matched_facts is citation-derived and citations are OFF on the
     agent lane, so it is structurally empty here — read
     surfaced_crystal_ids instead. match_type on this lane maps from
     citation GROUNDING, not the proxy's routing bands; 2.x rows are not
     comparable on that column. customer_id is the key into the bench
     DB's bank + agent_events for a full post-mortem of any row.
  10. Resumable, inspectable. --out refuses to append to an existing file
     unless --resume, which skips graded question ids and retries errored
     ones; the summary is always computed from the whole file (last row
     per id wins). --report prints it for any results file without a
     server or key; --show-answers prints Q / expected / model / verdict.

  JUDGE CAVEAT (disclosed, not hidden): the default judge (Claude Sonnet 4.6)
  and the answerer are both Anthropic models. This is mitigated — they are
  different capability tiers and the judge is disclosed in the manifest — and
  `--judge-model` swaps to GPT-4o (the LongMemEval paper's judge) in one flag
  once an OpenAI key is available.

DATASET
  https://github.com/xiaowu0162/LongMemEval. Official release (Sept 2025
  "cleaned" set — history sessions scrubbed of answer interference):
  https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned
    longmemeval_s_cleaned.json  — ~50 sessions / question  (HEADLINE config)
    longmemeval_m_cleaned.json  — 500 sessions / question  (stretch, ~2.75 GB)
    longmemeval_oracle.json     — evidence-only sessions   (DIAGNOSTIC ONLY,
                                  retrieval-isolated; not a comparable score)
  detect_variant() keys on the longmemeval_s / _m / oracle substrings, so the
  _cleaned names resolve without a flag. The manifest records the file's
  SHA-256 so a run discloses which release it used.
  Question types: single-session-user, single-session-assistant,
  single-session-preference, multi-session, temporal-reasoning,
  knowledge-update. Abstention variants have question_id ending in "_abs".

SERVER REQUIREMENTS (run before this script)
  CC_TEXT_ENCODER=semantic          as always
  CC_ENABLE_METACOGNITION_WORKER=0  no background bank mutation mid-sweep
  CC_WEB_SEARCH_PROVIDER unset      web_search must be dead (Guarantee #3)
  CC_AGENT_RETRIEVAL_PREFLIGHT unset  default ON = the product default
                                    (L7-Q2=A: ONE number, one config)
  ANTHROPIC_API_KEY in .env         agent turns + extraction
  (and seed NO general banks; the harness strips + verifies per customer)

  server:
    CC_TEXT_ENCODER=semantic CC_ENABLE_METACOGNITION_WORKER=0 \\
      uvicorn crystal_cache.app:app --host 0.0.0.0 --port 8000

USAGE
    export ANTHROPIC_API_KEY=sk-ant-...   # judge + customer upstream key
    # smoke (DIAGNOSTIC, retrieval-isolated oracle set):
    python scripts/longmemeval_harness.py --data data/longmemeval_oracle.json \\
        --limit 5 --out results/lme_smoke_agent.jsonl
    # headline (full _s set, no filters):
    python scripts/longmemeval_harness.py --data data/longmemeval_s_cleaned.json \\
        --server-commit "$(git rev-parse HEAD)" \\
        --out results/lme_s_agent.jsonl

COST NOTE (token economics are first-class): per question the server makes
roughly — one extraction LLM call PER ingested session (crystallization),
the agent loop's model calls (1 + one per tool-call iteration; the row's
`iterations` column is the count), and one MCR self-critique call at
finalize; the harness adds one judge call. Start with --limit 5 on the
oracle variant and read the token + iteration columns before scaling up.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

HARNESS_VERSION = "3.3-bank-reuse-2026-09-04"

# The tools that can bring information from OUTSIDE the ingested sessions
# into an answer (Guarantee #3). web_search is provider-gated; web_fetch is
# not. Any row whose agent turn called one of these is flagged, and the
# summary refuses the run a memory number if the count is nonzero.
EXTERNAL_TOOLS = frozenset({"web_search", "web_fetch"})

# Agent-envelope stop reasons (agent/agent.py `Agent.run` contract +
# the C2 cache-hit short-circuit). `error` never reaches the judge.
_STOP_REASON_ERROR = "error"

BASE = os.environ.get("CC_BASE_URL", "http://localhost:8000").rstrip("/")
DEFAULT_ANSWER_MODEL = "claude-haiku-4-5-20251001"
# GPT-4o's Anthropic equivalent for grading: the balanced-flagship tier.
# A clear step up from a Haiku answerer; disclosed in the manifest.
DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"

TIMEOUT = httpx.Timeout(300.0, connect=10.0)

# Evidence-label keys LongMemEval ships that the system under test must NEVER
# see — stripped defensively so a future edit can't leak them into retrieval.
_EVIDENCE_KEYS = ("answer_session_ids", "has_answer", "answer_evidence")


# ---------------------------------------------------------------------------
# Variant detection
# ---------------------------------------------------------------------------

def detect_variant(path: Path, override: str) -> str:
    """oracle | s | m | unknown. Filename-based; --variant overrides."""
    if override:
        return override
    name = path.name.lower()
    if "oracle" in name:
        return "oracle"
    if re.search(r"(_|-)m(\.|_|$)", name) or "longmemeval_m" in name:
        return "m"
    if re.search(r"(_|-)s(\.|_|$)", name) or "longmemeval_s" in name:
        return "s"
    return "unknown"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Server plumbing
# ---------------------------------------------------------------------------

def create_customer(client: httpx.Client, answer_model: str, upstream_key: str) -> tuple[str, str]:
    r = client.post(f"{BASE}/v1/customers", json={
        "provider": "anthropic",
        "model_id": answer_model,
        "api_key_ref": upstream_key,
    })
    r.raise_for_status()
    body = r.json()
    return body["id"], body["api_key"]


def enforce_clean_room(client: httpx.Client, api_key: str) -> list[str]:
    """Guarantee #4: ensure the customer is subscribed to ZERO general banks,
    so the ONLY knowledge that can enter the answer is the sessions we ingested.

    New customers may carry a default subscription (e.g. 'general:legacy'); we
    STRIP every general-bank subscription, then verify none remain. Removing
    knowledge can only make the test harder/cleaner — never inflate a score —
    and the stripped list is printed + recorded in the manifest for disclosure.

    Returns the list of subscriptions that were removed. Raises RuntimeError
    only if subscriptions still remain after the strip (then we refuse to run).
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    r = client.get(f"{BASE}/v1/subscriptions", headers=headers)
    r.raise_for_status()
    subs = list(r.json().get("general_crystal_types") or [])
    if subs:
        client.post(
            f"{BASE}/v1/unsubscribe", headers=headers,
            json={"crystal_types": subs},
        ).raise_for_status()
    r = client.get(f"{BASE}/v1/subscriptions", headers=headers)
    r.raise_for_status()
    remaining = list(r.json().get("general_crystal_types") or [])
    if remaining:
        raise RuntimeError(
            f"could not clear general-bank subscriptions {remaining}; refusing "
            "to run with extra knowledge in the answer context. Use a fresh "
            "benchmark DB with no general banks."
        )
    return subs


def server_general_bank_count(client: httpx.Client) -> int | None:
    """Best-effort: how many general banks exist on the server (informational
    for the manifest). The per-customer clean-room assert is what actually
    protects the run; this just flags a risky server state."""
    try:
        r = client.get(
            f"{BASE}/admin/api/crystal_types", params={"scope": "general"}
        )
        r.raise_for_status()
        return int(r.json().get("count") or 0)
    except Exception:
        return None


def _wait_for_status(
    client: httpx.Client, headers: dict, doc_id: str, terminal: set[str],
    *, timeout_s: float = 1800.0, every_s: float = 1.0,
) -> str:
    """Poll GET /v1/documents/{id} until its status is in `terminal`
    (L7a gate 5: under CC_INGEST_MODE=worker the ingest endpoints return
    202 and the worker finishes the leg). Returns the terminal status.
    Under inline mode the first poll already sees it, so this costs one
    GET either way. 'error' is always terminal."""
    deadline = time.monotonic() + timeout_s
    while True:
        r = client.get(f"{BASE}/v1/documents/{doc_id}", headers=headers)
        r.raise_for_status()
        status = r.json().get("status")
        if status in terminal or status == "error":
            return status
        if time.monotonic() > deadline:
            raise RuntimeError(f"document {doc_id} still '{status}' after {timeout_s:.0f}s")
        time.sleep(every_s)


def ingest_session(client: httpx.Client, api_key: str, label: str, text: str) -> str:
    """Upload one session transcript and run it to crystallized."""
    headers = {"Authorization": f"Bearer {api_key}"}
    r = client.post(f"{BASE}/v1/documents", headers=headers, json={
        "label": label,
        "text": text,
    })
    r.raise_for_status()
    doc_id = r.json()["id"]

    # pending -> review (chunk + extract). 200 = done inline; 202 = the
    # worker owns it; either way wait for the row to say so.
    r = client.post(f"{BASE}/v1/documents/{doc_id}/crystallize", headers=headers)
    r.raise_for_status()
    status = _wait_for_status(client, headers, doc_id, {"review", "crystallized"})
    if status == "error":
        raise RuntimeError(f"extraction failed for {doc_id}")

    # review -> crystallized (writes crystals; uses the saved review state)
    r = client.post(f"{BASE}/v1/documents/{doc_id}/approve", headers=headers, json={})
    r.raise_for_status()
    status = _wait_for_status(client, headers, doc_id, {"crystallized"})
    if status == "error":
        raise RuntimeError(f"crystallization failed for {doc_id}")
    return doc_id


class AgentTurnError(RuntimeError):
    """The agent turn ended with stop_reason="error" (Guarantee #3/#7): the
    envelope's final_text is an apology string, not an answer, so the
    question is counted as a harness ERROR and never graded."""


def ask(
    client: httpx.Client, api_key: str, model: str, question: str,
) -> tuple[str, dict]:
    """One agent turn through the product surface (L7-Q1=A).

    POST /v1/agent/messages with temperature=0 (Guarantee #6; the agent
    loop honours it since audit (e) 1.11). Returns (final_text, turn) where
    `turn` is the audit slice of the envelope — stop_reason, iterations,
    token totals, the tool names called, and whether any of them was an
    EXTERNAL tool (Guarantee #3). Raises AgentTurnError on stop_reason=error.
    """
    r = client.post(
        f"{BASE}/v1/agent/messages",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1024,
            "temperature": 0,
        },
    )
    r.raise_for_status()
    body = r.json()
    turn = agent_turn_fields(body)
    if turn["stop_reason"] == _STOP_REASON_ERROR:
        raise AgentTurnError(
            f"agent turn stop_reason=error after {turn['iterations']} "
            f"iteration(s): {(body.get('final_text') or '')[:200]!r}"
        )
    return body.get("final_text") or "", turn


def ask_admin(
    client: httpx.Client, customer_id: str, model: str, question: str,
) -> tuple[str, dict]:
    """Bank-reuse ask (Q1=A, 2026-09-04): the SAME agent turn as ask(),
    reached through the keyless admin proxy — /admin/api/customers/{id}/agent
    delegates to the shared run_agent_messages pipeline (auth source differs,
    code path identical). Exists because Key A is hashed at rest
    (no-plaintext, 2026-06-13): a prior leg's tenant key cannot be recovered,
    but its crystallized bank can still be asked. Envelope handling mirrors
    ask() verbatim."""
    r = client.post(
        f"{BASE}/admin/api/customers/{customer_id}/agent",
        json={
            "model": model,
            "messages": [{"role": "user", "content": question}],
            "max_tokens": 1024,
            "temperature": 0,
        },
    )
    r.raise_for_status()
    body = r.json()
    turn = agent_turn_fields(body)
    if turn["stop_reason"] == _STOP_REASON_ERROR:
        raise AgentTurnError(
            f"agent turn stop_reason=error after {turn['iterations']} "
            f"iteration(s): {(body.get('final_text') or '')[:200]!r}"
        )
    return body.get("final_text") or "", turn


def bank_probe_ok(client: httpx.Client, customer_id: str) -> bool:
    """Q1=A sanity probe: the reused tenant's bank is live and non-empty.
    Cheap (limit=1); False is advisory — the caller falls back to the
    fresh-tenant path rather than failing the question."""
    try:
        r = client.get(
            f"{BASE}/admin/api/customers/{customer_id}/crystals",
            params={"limit": 1},
        )
        r.raise_for_status()
        return int(r.json().get("total") or 0) > 0
    except Exception:
        return False


_ID_RE = re.compile(r"\b(crys_[0-9a-f]{8,}|fact_[0-9a-f]{8,})\b")


def _ids_in(obj: Any) -> list[str]:
    """Crystal/fact ids that appear anywhere in a tool output, by their
    fixed prefixes — schema-independent, so it works for every retriever
    tool without the harness knowing each one's output shape."""
    try:
        blob = json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        blob = str(obj)
    return sorted(set(_ID_RE.findall(blob)))


def agent_turn_fields(body: dict) -> dict:
    """The per-row audit slice of an agent envelope (Guarantee #9).

    Pure: no I/O, so it is unit-testable against a literal envelope.
    `tool_calls` entries are {tool_name, input, output, is_error,
    duration_ms, ...} per the agent contract. Kept per call: the name,
    the INPUT (the query the agent chose — the first thing to read when a
    row fails), is_error, duration, and the crystal/fact ids present in
    the output. Raw outputs (chunk text) are NOT kept — they are large
    and already persisted server-side as agent_events under the row's
    customer_id.

    Tokens: `prompt_tokens` is the UNCACHED input slice only; the cached
    system prompt lands in cache_read/cache_creation, so the honest cost
    column is total_input_tokens = all three (Guarantee #8).
    """
    detail = []
    for tc in (body.get("tool_calls") or []):
        tc = tc or {}
        name = str(tc.get("tool_name") or "")
        if not name:
            continue
        detail.append({
            "tool_name": name,
            "input": tc.get("input"),
            "is_error": bool(tc.get("is_error")),
            "duration_ms": tc.get("duration_ms"),
            "ids_in_output": _ids_in(tc.get("output")),
        })
    names = [d["tool_name"] for d in detail]
    surfaced = sorted({i for d in detail for i in d["ids_in_output"]})

    def _int(v):
        return v if isinstance(v, int) else 0

    prompt = body.get("prompt_tokens")
    cache_read = body.get("cache_read_tokens")
    cache_create = body.get("cache_creation_tokens")
    total_input = (
        _int(prompt) + _int(cache_read) + _int(cache_create)
        if any(isinstance(v, int) for v in (prompt, cache_read, cache_create))
        else None
    )
    return {
        "agent_message_id": body.get("id"),
        "stop_reason": body.get("stop_reason"),
        "iterations": body.get("iterations"),
        "prompt_tokens": prompt,
        "cache_read_tokens": cache_read,
        "cache_creation_tokens": cache_create,
        "total_input_tokens": total_input,
        "completion_tokens": body.get("completion_tokens"),
        "tool_calls": names,
        "tool_calls_detail": detail,
        "tool_errors": sum(1 for d in detail if d["is_error"]),
        "surfaced_crystal_ids": [i for i in surfaced if i.startswith("crys_")],
        "surfaced_fact_ids": [i for i in surfaced if i.startswith("fact_")],
        "external_tool_used": any(n in EXTERNAL_TOOLS for n in names),
    }


def last_query_log(client: httpx.Client, customer_id: str) -> dict:
    """The customer's single query_log row (token + retrieval accounting).

    A fresh customer makes exactly ONE chat call, so there is exactly one row
    and items[0] is unambiguous regardless of ordering.
    """
    try:
        r = client.get(
            f"{BASE}/admin/api/customers/{customer_id}/query_logs",
            params={"offset": 0, "limit": 1},
        )
        r.raise_for_status()
        items = r.json().get("items") or []
        return items[0] if items else {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Transcript building (evidence labels stripped — Guarantee #2)
# ---------------------------------------------------------------------------

def session_to_text(session: list[dict], date: str) -> str:
    lines = [f"[Conversation on {date}]", ""]
    for turn in session:
        role = (turn.get("role") or "user").upper()
        content = (turn.get("content") or "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judge — per-question-type prompts (aligned to the LongMemEval rubric)
# ---------------------------------------------------------------------------

def _judge_prompt(q: dict, model_answer: str) -> str:
    qtype = str(q.get("question_type", ""))
    is_abstention = str(q.get("question_id", "")).endswith("_abs")
    question = q.get("question", "")
    gold = q.get("answer", "")

    if is_abstention:
        return (
            "A model was asked a question whose answer was NOT present in its "
            "memory. The correct behavior is to ABSTAIN — to say it does not "
            "know or lacks the information, rather than inventing an answer.\n\n"
            f"Question: {question}\n\n"
            f"Model response: {model_answer}\n\n"
            "Did the model correctly abstain (indicate it does not know / lacks "
            "the information)? Reply with exactly 'yes' or 'no'."
        )

    if qtype == "knowledge-update":
        rubric = (
            "The user's information CHANGED over time. The correct answer "
            "reflects the MOST RECENT (updated) state. The model is correct "
            "ONLY if its response reflects the updated information, not an "
            "outdated earlier value."
        )
    elif qtype == "temporal-reasoning":
        rubric = (
            "This question requires temporal reasoning (dates, durations, or "
            "ordering). The model is correct ONLY if the temporal facts and any "
            "computed dates/intervals match the correct answer."
        )
    elif qtype == "single-session-preference":
        rubric = (
            "This question concerns the user's stated preference. The model is "
            "correct ONLY if its response reflects the preference captured in "
            "the correct answer."
        )
    else:
        rubric = (
            "Minor wording differences are fine; the substance must match."
        )

    return (
        "Judge whether the model's response to a question agrees with the "
        f"correct answer. {rubric}\n\n"
        f"Question: {question}\n\n"
        f"Correct answer: {gold}\n\n"
        f"Model response: {model_answer}\n\n"
        "Does the model response contain or agree with the correct answer? "
        "Reply with exactly 'yes' or 'no'."
    )


def judge(anthropic_client, judge_model: str, q: dict, model_answer: str) -> tuple[bool, str]:
    """Returns (correct, raw_verdict). Raw verdict is logged for audit."""
    prompt = _judge_prompt(q, model_answer)
    msg = anthropic_client.messages.create(
        model=judge_model,
        max_tokens=8,
        # SDK 1.x removed the temperature kwarg; extra_body is the
        # documented path for models that still honour it (Guarantee #6).
        extra_body={"temperature": 0},
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        b.text for b in msg.content if getattr(b, "type", "") == "text"
    ).strip()
    correct = text.lower().lstrip(" '\"`").startswith("yes")
    return correct, text


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Result-file reading + rows-based summary (2026-08-28, L7 gate 3)
# ---------------------------------------------------------------------------
# A week-long headline run must be resumable and inspectable mid-flight, so
# the summary is computed from the ROWS IN THE FILE (last row per question
# id wins), not from counters that only live for one process. `--report`
# prints the same summary for any results file without touching a server.

def load_results_file(path: Path) -> tuple[list[dict], dict]:
    """Returns (rows, last_manifest). Rows are deduplicated by question_id,
    last occurrence wins — a resumed retry of an errored question replaces
    the error row."""
    rows_by_id: dict[str, dict] = {}
    manifest: dict = {}
    if not path.exists():
        return [], manifest
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("record") == "manifest":
                manifest = rec
            elif rec.get("record") == "run_end":
                # Harness 3.2: the wall-clock record the run appends when it
                # finishes; carried on the manifest so callers stay unchanged.
                manifest = dict(manifest, run_end=rec)
            elif rec.get("record") == "result":
                rows_by_id[str(rec.get("question_id"))] = rec
    return list(rows_by_id.values()), manifest


def build_bank_map(rows: list[dict]) -> dict[str, dict]:
    """Q1=A (2026-09-04): question_id -> reusable-bank evidence, from the
    results file itself. A row proves its tenant's bank is complete when it
    records the tenant (customer_id) AND a finished ingest — either this
    attempt ingested every session (sessions_ingested is set only AFTER the
    ingest loop completes) or the row itself reused a proven bank
    (bank_sessions, carried forward so an ask-side failure never demotes
    the evidence). Rows arrive last-wins-deduped from load_results_file;
    the caller still compares the count against the question's haystack
    size before trusting an entry."""
    out: dict[str, dict] = {}
    for row in rows:
        cid = row.get("customer_id")
        n = row.get("bank_sessions") or row.get("sessions_ingested")
        if cid and n:
            out[str(row.get("question_id"))] = {
                "customer_id": str(cid), "bank_sessions": int(n),
            }
    return out


def print_summary(rows: list[dict], *, variant: str, partial: bool,
                  headline_eligible: bool, judge_model: str,
                  out_path: Path | None, run_end: dict | None = None) -> None:
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "correct": 0, "errors": 0}
    )
    total_correct = total_attempted = errors = 0
    tok_sum = tok_n = 0
    surfaced_sum = 0
    iterations_sum = iterations_n = 0
    ingest_s_sum = 0.0
    sessions_sum = 0
    external_tool_rows = 0
    no_tool_rows = 0
    reused_rows = 0
    for row in rows:
        qtype = row.get("question_type", "unknown")
        reused_rows += 1 if row.get("reused_bank") else 0
        total_attempted += 1
        by_type[qtype]["attempted"] += 1
        if "error" in row:
            errors += 1
            by_type[qtype]["errors"] += 1
            continue
        if row.get("correct"):
            total_correct += 1
            by_type[qtype]["correct"] += 1
        if isinstance(row.get("total_input_tokens"), int):
            tok_sum += row["total_input_tokens"]
            tok_n += 1
        surfaced_sum += len(row.get("surfaced_crystal_ids") or [])
        if isinstance(row.get("iterations"), int):
            iterations_sum += row["iterations"]
            iterations_n += 1
        if not row.get("tool_calls"):
            no_tool_rows += 1
        if isinstance(row.get("ingest_s"), (int, float)) and row.get("sessions_ingested"):
            ingest_s_sum += row["ingest_s"]
            sessions_sum += int(row["sessions_ingested"])
        if row.get("external_tool_used"):
            external_tool_rows += 1

    graded = total_attempted - errors
    print("\n=== Results ===")
    print(f"  {'question_type':<28} {'correct/attempted':>18}   {'acc':>5}")
    for qtype in sorted(by_type):
        c = by_type[qtype]
        att = c["attempted"]
        acc = (100.0 * c["correct"] / att) if att else 0.0
        suffix = f"  ({c['errors']} err)" if c["errors"] else ""
        print(f"  {qtype:<28} {c['correct']:>8}/{att:<9} {acc:>5.0f}%{suffix}")
    print("  " + "-" * 50)
    if total_attempted:
        acc_att = 100.0 * total_correct / total_attempted
        print(f"  {'OVERALL (errors as fail)':<28} "
              f"{total_correct:>8}/{total_attempted:<9} {acc_att:>5.0f}%")
    if graded:
        acc_graded = 100.0 * total_correct / graded
        print(f"  {'OVERALL (errors excluded)':<28} "
              f"{total_correct:>8}/{graded:<9} {acc_graded:>5.0f}%")
    if errors:
        print(f"  errors: {errors}  (re-run with --resume to retry them; never drop)")
    if tok_n:
        print(f"  avg input tokens/question (incl. cache): {tok_sum / tok_n:.0f}")
    if graded:
        print(f"  avg surfaced crystals/question: {surfaced_sum / graded:.1f}")
        print(f"  answered with NO retrieval tool call: {no_tool_rows}/{graded}")
    if reused_rows:
        print(f"  reused banks: {reused_rows} row(s) asked via the keyless "
              "admin proxy (see manifest reuse_mode)")
    if iterations_n:
        print(f"  avg agent iterations/question: {iterations_sum / iterations_n:.1f}")
    if sessions_sum:
        per_session = ingest_s_sum / sessions_sum
        conc = int((run_end or {}).get("concurrency") or 1)
        wall_s = (run_end or {}).get("wall_s")
        wall_sessions = int((run_end or {}).get("sessions") or 0)
        if conc == 1 or not wall_s or not wall_sessions:
            print(f"  ingest: {sessions_sum} sessions in {ingest_s_sum:.0f}s "
                  f"= {per_session:.1f}s/session  "
                  f"(full _s set ≈ 25,000 sessions → ~{per_session * 25000 / 3600:.0f}h serial)")
        else:
            # Harness 3.2: with N questions in flight a row's ingest_s includes
            # time spent queued behind the other N-1, so the honest throughput
            # number is wall-clock — sessions this session divided by the time
            # the whole session took.
            per_wall = float(wall_s) / wall_sessions
            print(f"  ingest: {sessions_sum} sessions, {per_session:.1f}s/session per row "
                  f"(includes queueing at concurrency {conc})")
            print(f"  wall:   {wall_sessions} sessions in {float(wall_s):.0f}s "
                  f"= {per_wall:.1f}s/session at concurrency {conc}  "
                  f"(full _s set ≈ 25,000 sessions → ~{per_wall * 25000 / 3600:.0f}h)")
    if external_tool_rows:
        print(f"  ⚠ EXTERNAL TOOL ROWS: {external_tool_rows} — web_search/web_fetch "
              "ran inside an answer turn. This run is NOT a memory number "
              "(Guarantee #3); fix the server config and re-run.")

    # ---- Headline verdict stamp -----------------------------------------
    print()
    if external_tool_rows:
        print("  ▶ DISQUALIFIED: external tool use detected (see above).")
    elif variant == "oracle":
        print("  ▶ RETRIEVAL-ISOLATED (oracle): answerer-only upper bound, "
              "NOT a comparable LongMemEval score.")
    elif partial:
        print("  ▶ PARTIAL / DEV SMOKE — not a headline number.")
    elif headline_eligible:
        print(f"  ▶ HEADLINE-ELIGIBLE: full {variant.upper()} set, "
              f"surface=agent, judge={judge_model}, temp=0. Disclose "
              f"the surface + config when citing.")
    else:
        print("  ▶ NON-HEADLINE: pass --variant s|m on a full set for a citable "
              "number.")
    if out_path:
        print(f"  manifest + per-question rows: {out_path}")


def _print_answer_block(row: dict) -> None:
    """--show-answers: the three things a reviewer compares."""
    def _clip(s: Any, n: int = 500) -> str:
        s = str(s or "").replace("\n", " ")
        return s if len(s) <= n else s[: n - 1] + "…"
    print(f"      Q:        {_clip(row.get('question'))}")
    print(f"      expected: {_clip(row.get('expected_answer'))}")
    print(f"      model:    {_clip(row.get('model_answer'))}")
    print(f"      judge:    {row.get('judge_verdict_raw')}  "
          f"tools={row.get('tool_calls')}  "
          f"surfaced={len(row.get('surfaced_crystal_ids') or [])}")


def run(args: argparse.Namespace) -> int:
    # --report: summarise an existing results file; no server, no key.
    if args.report:
        if not args.out:
            print("--report needs --out <results.jsonl>")
            return 2
        rp = Path(args.out)
        rows, man = load_results_file(rp)
        if not rows:
            print(f"No result rows in {rp}")
            return 2
        if args.show_answers:
            for row in sorted(rows, key=lambda r: str(r.get("question_id"))):
                status = "ERROR" if "error" in row else ("PASS" if row.get("correct") else "FAIL")
                print(f"{status}  {row.get('question_type', ''):<26} {row.get('question_id')}")
                if "error" in row:
                    print(f"      error:    {row['error']}")
                else:
                    _print_answer_block(row)
        print_summary(
            rows, variant=man.get("variant", "?"),
            partial=bool(man.get("partial_run", True)),
            headline_eligible=bool(man.get("headline_eligible", False)),
            judge_model=str(man.get("judge_model", "?")), out_path=rp,
            run_end=man.get("run_end"),
        )
        return 0

    upstream_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not upstream_key:
        print("ANTHROPIC_API_KEY is required (judge + customer upstream key).")
        return 2

    try:
        import anthropic
    except ImportError:
        print("The 'anthropic' package is required (it's in the server venv).")
        return 2
    anthropic_client = anthropic.Anthropic(api_key=upstream_key)

    if not args.data:
        print("--data is required (path to a LongMemEval variant JSON).")
        return 2
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        print("Download a LongMemEval variant JSON first — see module docstring.")
        return 2

    variant = detect_variant(data_path, args.variant)
    dataset_sha = sha256_file(data_path)
    all_questions = json.loads(data_path.read_text(encoding="utf-8"))
    total_in_file = len(all_questions)

    questions = list(all_questions)
    if args.types:
        wanted = {t.strip() for t in args.types.split(",") if t.strip()}
        questions = [q for q in questions if q.get("question_type") in wanted]
    if args.seed is not None:
        random.Random(args.seed).shuffle(questions)
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("No questions selected.")
        return 2

    selected = len(questions)
    partial = bool(args.types) or (selected < total_in_file)
    headline_eligible = (variant in ("s", "m")) and not partial

    # ---- Resume / refuse-to-append (L7 gate 3) -------------------------
    # A results file is one run's evidence. Appending a second run to it
    # silently (the 2.x behaviour) mixed manifests; now: an existing --out
    # is an error unless --resume, and --resume skips every question id
    # that already has a GRADED row (errored rows are retried).
    out_path = Path(args.out) if args.out else None
    resumed_done = 0
    bank_map: dict[str, dict] = {}
    if out_path and out_path.exists():
        if not args.resume:
            print(f"{out_path} exists. Pass --resume to continue it (graded "
                  "questions are skipped, errors retried) or choose a new --out.")
            return 2
        prior_rows, _ = load_results_file(out_path)
        done_ids = {str(r.get("question_id")) for r in prior_rows if "error" not in r}
        before = len(questions)
        questions = [q for q in questions if str(q.get("question_id")) not in done_ids]
        resumed_done = before - len(questions)
        print(f"  resume: {resumed_done} already graded in {out_path}, "
              f"{len(questions)} remaining")
        if args.reuse_banks:
            bank_map = build_bank_map(prior_rows)
            reusable = sum(
                1 for q in questions
                if bank_map.get(str(q.get("question_id")), {}).get("bank_sessions")
                == len(q.get("haystack_sessions") or [])
            )
            print(f"  reuse-banks: {reusable}/{len(questions)} remaining have "
                  f"complete banks (ask-only via keyless admin proxy); "
                  f"{len(questions) - reusable} run the fresh-tenant path")
        if not questions:
            print("  nothing left to run — use --report for the tally.")
            return 0

    if args.reuse_banks and not bank_map:
        print("  reuse-banks: no reusable banks found in the results file — "
              "every question runs the fresh-tenant path")

    # ---- Manifest (Guarantee #8) ----------------------------------------
    with httpx.Client(timeout=TIMEOUT) as probe:
        general_banks = server_general_bank_count(probe)

    manifest = {
        "record": "manifest",
        "harness_version": HARNESS_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(data_path),
        "dataset_sha256": dataset_sha,
        "variant": variant,
        "questions_in_file": total_in_file,
        "questions_selected": selected,
        "types_filter": args.types or None,
        "limit": args.limit or None,
        "seed": args.seed,
        "partial_run": partial,
        "headline_eligible": headline_eligible,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "temperature": 0,
        "server_base": BASE,
        "server_commit": args.server_commit or "unspecified",
        "server_general_banks_present": general_banks,
        # L7-Q1=A (2026-08-28): the product surface, disclosed. Harness
        # 2.x rows were proxy (/v1/chat/completions, passive|active mode)
        # and are not comparable with 3.x rows.
        "surface": "agent",
        "surface_endpoint": "/v1/agent/messages",
        "surface_note": (
            "one agent turn: C2 retrieval preflight on the opening turn + "
            "the agent's retrieval tools across loop iterations -> ONE "
            "final judged answer (final_text). Retrieval may be multi-round; "
            "the graded answer is single. Per-row tool_calls discloses what "
            "ran; external_tool_used flags web_search/web_fetch."
        ),
        "telemetry_note": (
            "match_type on the agent lane maps from citation grounding "
            "(high=grounded, medium=surfaced-ungrounded, none), not the "
            "proxy's routing bands; prompt/completion tokens are the agent "
            "envelope's loop totals."
        ),
        "clean_room_policy": (
            "strip all general-bank subscriptions per customer; verify zero "
            "before ingest (removing knowledge cannot inflate a score)"
        ),
        "expected_server_flags": {
            "CC_TEXT_ENCODER": "semantic",
            "CC_ENABLE_METACOGNITION_WORKER": "0",
            "CC_WEB_SEARCH_PROVIDER": "unset (web_search must be dead)",
            # L7-Q2=A (2026-08-28): one number at the product default.
            "CC_AGENT_RETRIEVAL_PREFLIGHT": "unset (default ON)",
        },
        "judge_caveat": (
            "judge and answerer are both Anthropic (different tiers, "
            "disclosed); swap --judge-model to GPT-4o when available"
        ),
        "note": args.note or None,
        "reuse_banks": bool(args.reuse_banks),
        # Q2=A (2026-09-04): full disclosure of the bank-reuse policy on the
        # manifest; reused rows additionally carry reused_bank /
        # ask_surface / bank_sessions on the row itself.
        "reuse_mode": ({
            "policy": (
                "jsonl-evidence: an errored question is re-asked against its "
                "prior tenant's bank when its latest row records customer_id "
                "and a complete session count (sessions_ingested or "
                "bank_sessions == the question's haystack size), "
                "sanity-probed non-empty via "
                "GET /admin/api/customers/{id}/crystals"
            ),
            "ask_surface_reused_rows": (
                "/admin/api/customers/{id}/agent — keyless admin proxy "
                "delegating to the SAME run_agent_messages pipeline as "
                "/v1/agent/messages; auth source differs, code path identical"
            ),
            "clean_room": (
                "enforced at tenant creation by the original leg (strip + "
                "verify-zero, recorded in that leg's manifest); NOT re-verified "
                "at reuse — Key A is hashed at rest by design and no code path "
                "adds subscriptions between creation and reuse. Inference, not "
                "re-measurement (Q3=A)."
            ),
        } if args.reuse_banks else None),
        "resumed": bool(args.resume),
        "resumed_already_graded": resumed_done,
        "concurrency": max(1, int(getattr(args, "concurrency", 1) or 1)),
        "questions_this_session": len(questions),
    }

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    out_f = out_path.open("a", encoding="utf-8") if out_path else None
    if out_f:
        out_f.write(json.dumps(manifest, ensure_ascii=False) + "\n")
        out_f.flush()

    # ---- Banner ----------------------------------------------------------
    print(f"LongMemEval harness {HARNESS_VERSION}")
    print(f"  dataset: {data_path.name}  variant={variant}  sha256={dataset_sha[:12]}…")
    print(f"  selected {selected}/{total_in_file} questions   "
          f"answer={args.answer_model}   judge={args.judge_model}")
    print(f"  server: {BASE}   commit={manifest['server_commit']}   "
          f"general_banks_on_server={general_banks}")
    print("  surface: AGENT (/v1/agent/messages) — preflight + retrieval "
          "tools, ONE final judged answer")
    print("  REMINDER: launch the server with CC_TEXT_ENCODER=semantic "
          "CC_ENABLE_METACOGNITION_WORKER=0 and "
          "CC_AGENT_DISABLED_TOOLS=web_search,web_fetch (the compose default "
          "configures a search provider); any external_tool_used row "
          "disqualifies the run.\n")
    if variant == "oracle":
        print("  ⚠ ORACLE VARIANT: retrieval-ISOLATED upper bound — the haystack "
              "is evidence-only.\n    This is NOT a comparable LongMemEval score.\n")
    elif variant == "unknown":
        print("  ⚠ UNKNOWN VARIANT: pass --variant s|m|oracle. Treating as "
              "non-headline.\n")
    if partial:
        print("  ⚠ PARTIAL RUN (type filter / limit): DEV SMOKE — not a headline "
              "number.\n")

    # ---- Clean-room pre-flight (strip + verify) -------------------------
    # New customers may carry a default general-bank subscription; strip it and
    # verify zero remain before spending tokens. Removing knowledge only makes
    # the test cleaner. Per-question enforcement below repeats this on each
    # fresh customer (defense-in-depth).
    try:
        with httpx.Client(timeout=TIMEOUT) as preflight:
            _cid, _key = create_customer(preflight, args.answer_model, upstream_key)
            _stripped = enforce_clean_room(preflight, _key)
        if _stripped:
            print(f"  clean-room: stripped default general subscriptions "
                  f"{_stripped} (removed before ingest; disclosed in manifest)\n")
    except RuntimeError as e:
        print(f"\nABORT — {e}")
        if out_f:
            out_f.close()
        return 3
    except Exception as e:
        print(f"\nABORT — clean-room pre-flight could not run: "
              f"{type(e).__name__}: {e}")
        if out_f:
            out_f.close()
        return 3

    session_rows: list[dict] = []
    remaining = len(questions)
    # Harness 3.2: N questions in flight. Every question is its own tenant
    # (fresh customer, own rows, own timings), so questions are independent;
    # extraction is Anthropic-bound, so N in flight divides the wall clock by
    # about N. One lock serializes the file append, the console line and the
    # in-memory row list. Sessions INSIDE a question stay serial. At
    # --concurrency 1 this is the previous loop run through a one-worker pool.
    concurrency = max(1, int(getattr(args, "concurrency", 1) or 1))
    emit_lock = threading.Lock()
    sessions_done = 0
    # Graceful stop for incremental runs (harness 3.2): create
    # `<out>.stop` and the harness starts no new questions, finishes the
    # ones in flight, writes the run record and exits. `--resume` picks
    # up the rest. Ctrl-C would throw away every in-flight question.
    stop_path = Path(str(out_path) + ".stop") if out_path else None
    stop_seen = False

    def _stop_requested() -> bool:
        nonlocal stop_seen
        if stop_path is not None and stop_path.exists():
            if not stop_seen:
                stop_seen = True
                with emit_lock:
                    print(f"  ⏹ stop file {stop_path.name} seen — finishing in-flight questions, "
                          f"starting no more; resume with --resume")
            return True
        return False

    def _run_question(i: int, q: dict) -> None:
        nonlocal sessions_done
        if _stop_requested():
            return
        qid = q.get("question_id", f"q{i}")
        qtype = q.get("question_type", "unknown")
        t0 = time.time()
        row: dict = {
            "record": "result", "question_id": qid, "question_type": qtype,
            # L7 gate 3: what a reviewer compares, on the row itself.
            "question": q.get("question"),
            "question_date": q.get("question_date"),
            "expected_answer": q.get("answer"),
        }
        try:
            with httpx.Client(timeout=TIMEOUT) as client:
                sessions = q.get("haystack_sessions") or []
                # Q1=A (2026-09-04): reuse a prior leg's crystallized bank
                # when the results file proves it complete AND the live probe
                # sees a non-empty bank; otherwise the fresh-tenant path.
                bank = bank_map.get(str(qid)) if args.reuse_banks else None
                reuse = bool(
                    bank
                    and bank.get("bank_sessions") == len(sessions)
                    and bank_probe_ok(client, bank["customer_id"])
                )
                if reuse:
                    customer_id = bank["customer_id"]
                    api_key = None  # Key A unrecoverable by design (hashed)
                    row["customer_id"] = customer_id
                    row["reused_bank"] = True
                    row["ask_surface"] = "admin-proxy"
                    # Carry the evidence forward (build_bank_map reads it) so
                    # an ask-side failure never demotes a proven bank.
                    row["bank_sessions"] = bank["bank_sessions"]
                    row["sessions_ingested"] = 0
                    row["ingest_s"] = 0.0
                else:
                    customer_id, api_key = create_customer(
                        client, args.answer_model, upstream_key
                    )
                    row["customer_id"] = customer_id

                    # Guarantee #4: strip any general-bank subscriptions (a
                    # fresh customer can carry a default) so only the ingested
                    # sessions are in context; verify zero remain before
                    # ingesting.
                    enforce_clean_room(client, api_key)

                    dates = q.get("haystack_dates") or [""] * len(sessions)
                    sids = q.get("haystack_session_ids") or [
                        f"s{j}" for j in range(len(sessions))
                    ]
                    # Phase timing (2026-08-28): the June smoke spent 300-640s
                    # per question on 2-3 sessions while the ask took 2-5s —
                    # ingestion is the wall-clock cost and the S headline is
                    # ~25k sessions. Recording the split per row is what makes
                    # the runtime plan a measurement instead of a guess. Under
                    # concurrency > 1 this per-row number includes queueing;
                    # the summary's wall line is the throughput.
                    t_ingest = time.time()
                    for sess, date, sid in zip(sessions, dates, sids):
                        text = session_to_text(sess, date)
                        ingest_session(
                            client, api_key, label=f"Session {sid} ({date})", text=text
                        )
                    row["sessions_ingested"] = len(sessions)
                    row["ingest_s"] = round(time.time() - t_ingest, 1)

                question_text = q["question"]
                if q.get("question_date"):
                    question_text = (
                        f"Today's date is {q['question_date']}. {question_text}"
                    )
                t_ask = time.time()
                if reuse:
                    answer, turn = ask_admin(
                        client, customer_id, args.answer_model, question_text
                    )
                else:
                    answer, turn = ask(
                        client, api_key, args.answer_model, question_text
                    )
                row["ask_s"] = round(time.time() - t_ask, 1)
                row["model_answer"] = answer
                # Guarantee #9: the agent's own audit slice — what ran,
                # with what inputs, surfacing which ids (L7 gate 3).
                row.update(turn)

                # Retrieval telemetry from the ONE query-log row
                # finalize_agent_turn writes per turn. NOTE: matched_facts
                # is citation-derived and citations are OFF on the agent
                # lane, so it is structurally empty here; the live audit
                # column is surfaced_crystal_ids (from tool outputs).
                qlog = last_query_log(client, customer_id)
                row["match_type"] = qlog.get("match_type")
                row["injection_method"] = qlog.get("injection_method")
                row["matched_facts"] = qlog.get("matched_facts") or []
                row["latency_ms"] = qlog.get("latency_ms")

            t_judge = time.time()
            correct, verdict = judge(
                anthropic_client, args.judge_model, q, answer
            )
            row["judge_s"] = round(time.time() - t_judge, 1)
            row["correct"] = correct
            row["judge_verdict_raw"] = verdict
            status = "PASS" if correct else "FAIL"
        except Exception as e:  # keep the sweep going; counted as attempted
            row["error"] = f"{type(e).__name__}: {e}"
            status = "ERROR"

        row["elapsed_s"] = round(time.time() - t0, 1)
        with emit_lock:
            session_rows.append(row)
            sessions_done += int(row.get("sessions_ingested") or 0)
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
            print(
                f"[{i}/{remaining}] {status}  {qtype:<26} {qid}"
                f"  ({row['elapsed_s']}s"
                + (f" = ingest {row['ingest_s']}s" if row.get("ingest_s") is not None else "")
                + (f" + ask {row['ask_s']}s" if row.get("ask_s") is not None else "")
                + (f" + judge {row['judge_s']}s" if row.get("judge_s") is not None else "")
                + (f", {row.get('total_input_tokens')} in-tok" if row.get("total_input_tokens") else "")
                + (f", {len(row.get('surfaced_crystal_ids') or [])} surfaced" if row.get("surfaced_crystal_ids") is not None else "")
                + (f", {row.get('iterations')} iter" if row.get("iterations") is not None else "")
                + (" ⚠ EXTERNAL TOOL" if row.get("external_tool_used") else "")
                + ")"
            )
            if args.show_answers:
                if "error" in row:
                    print(f"      error:    {row['error']}")
                else:
                    _print_answer_block(row)

    t_run = time.time()
    with ThreadPoolExecutor(max_workers=concurrency, thread_name_prefix="lme-q") as pool:
        for i, q in enumerate(questions, 1):
            pool.submit(_run_question, i, q)
    run_end = {
        "record": "run_end",
        "wall_s": round(time.time() - t_run, 1),
        "concurrency": concurrency,
        "sessions": sessions_done,
        "questions": len(session_rows),
        "stopped": stop_seen,
    }
    if out_f:
        out_f.write(json.dumps(run_end, ensure_ascii=False) + "\n")
        out_f.flush()
    if stop_seen and stop_path is not None:
        stop_path.unlink(missing_ok=True)      # so --resume doesn't stop at once
        print(f"  ⏹ stopped after {len(session_rows)} question(s) this increment; "
              f"{len(questions) - len(session_rows)} remain — rerun with --resume")

    # ---- Results (Guarantee #7: honest denominator) ---------------------
    # From the FILE when there is one (so a resumed run reports the whole
    # run, not just this session); from this session's rows otherwise.
    if out_f:
        out_f.close()
    if out_path:
        rows, man = load_results_file(out_path)
        run_end_rec = man.get("run_end") or run_end
    else:
        rows, run_end_rec = session_rows, run_end
    print_summary(
        rows, variant=variant, partial=partial,
        headline_eligible=headline_eligible, judge_model=args.judge_model,
        out_path=out_path, run_end=run_end_rec,
    )
    return 0


def main(argv: list[str]) -> int:
    # Windows: a piped stdout is cp1252 and the summary uses ≈ and →;
    # never let the report die on an encoding after the rows are written.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", default="",
                    help="Path to a LongMemEval variant JSON (required unless --report)")
    ap.add_argument("--variant", default="",
                    help="Override variant detection: s | m | oracle")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max questions to run (0 = full file; mind the cost)")
    ap.add_argument("--concurrency", type=int, default=1,
                    help="Questions in flight at once (harness 3.2). Each is its "
                         "own tenant; extraction is Anthropic-bound, so N divides "
                         "the wall clock by ~N. Match the worker's "
                         "CC_CRYSTALLIZE_CONCURRENCY. 1 = the serial loop.")
    ap.add_argument("--types", default="",
                    help="Comma-separated question_type filter (marks the run partial)")
    ap.add_argument("--seed", type=int, default=None,
                    help="Shuffle seed for sampling (default: dataset order)")
    ap.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL)
    ap.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                    help="Default: Claude Sonnet 4.6 (GPT-4o's Anthropic tier)")
    ap.add_argument("--server-commit", default="",
                    help="git rev-parse HEAD of the running server, for the manifest")
    ap.add_argument("--note", default="", help="Free-text note recorded in the manifest")
    ap.add_argument("--out", default="",
                    help="JSONL path: manifest line + per-question rows. An "
                         "existing file is refused unless --resume.")
    ap.add_argument("--resume", action="store_true",
                    help="Continue an existing --out: skip question ids that "
                         "already have a graded row, retry errored ones, "
                         "append a new manifest, and report on the whole file.")
    ap.add_argument("--reuse-banks", action="store_true",
                    help="Q1-Q4=A (2026-09-04, default off): with --resume, "
                         "retry an errored question against its prior tenant's "
                         "already-crystallized bank (jsonl evidence: "
                         "customer_id + complete session count) via the "
                         "keyless admin agent proxy, skipping "
                         "create/clean-room/ingest. Questions without complete "
                         "banks run the normal fresh-tenant path.")
    ap.add_argument("--report", action="store_true",
                    help="No run: print the summary for an existing --out "
                         "(with --show-answers, every row). Needs no server/key.")
    ap.add_argument("--show-answers", action="store_true",
                    help="Print question / expected / model answer / judge "
                         "verdict per row.")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
