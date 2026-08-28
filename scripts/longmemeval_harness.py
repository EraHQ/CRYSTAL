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
  6. Fixed decoding. Answer and judge both run at temperature 0.
  7. Honest denominator. Errored questions are counted as attempted (not
     silently dropped); the summary shows correct/attempted AND
     correct/graded plus the error count, so nothing can be hidden either way.
  8. Full manifest. The first line of --out (and stdout) records the dataset
     SHA-256, variant, model ids, temperature, counts, and the expected
     server flags — everything needed to reproduce the run.
  9. Per-question audit trail. Each row records match_type, injection_method,
     the matched crystal ids, plus the agent's stop_reason, iteration count
     and the tool names it called, so a reviewer can inspect exactly which
     memories were surfaced and what ran to surface them. NOTE on
     match_type semantics on the agent lane: it maps from citation
     GROUNDING (high = the answer cites a surfaced crystal; medium =
     surfaced but ungrounded; none = nothing surfaced), not from the
     proxy's routing-confidence bands — rows from harness 2.x are not
     comparable on that column.

  JUDGE CAVEAT (disclosed, not hidden): the default judge (Claude Sonnet 4.6)
  and the answerer are both Anthropic models. This is mitigated — they are
  different capability tiers and the judge is disclosed in the manifest — and
  `--judge-model` swaps to GPT-4o (the LongMemEval paper's judge) in one flag
  once an OpenAI key is available.

DATASET
  https://github.com/xiaowu0162/LongMemEval (also on HuggingFace:
  xiaowu0162/longmemeval). Download a variant JSON and pass it via --data:
    longmemeval_s.json       — ~50 sessions / question  (HEADLINE config)
    longmemeval_m.json       — 500 sessions / question  (stretch)
    longmemeval_oracle.json  — evidence-only sessions   (DIAGNOSTIC ONLY,
                               retrieval-isolated; not a comparable score)
  Question types: single-session-user, single-session-assistant,
  single-session-preference, multi-session, temporal-reasoning,
  knowledge-update. Abstention variants have question_id ending in "_abs".

SERVER REQUIREMENTS (run before this script)
  CC_TEXT_ENCODER=semantic          as always
  CC_ENABLE_METACOGNITION_WORKER=0  no background bank mutation mid-sweep
  CC_WEB_SEARCH_PROVIDER unset      web_search must be dead (Guarantee #3)
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
    python scripts/longmemeval_harness.py --data data/longmemeval_s.json \\
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
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import httpx

HARNESS_VERSION = "3.0-agent-surface-2026-08-28"

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


def ingest_session(client: httpx.Client, api_key: str, label: str, text: str) -> str:
    """Upload one session transcript and run it to crystallized."""
    headers = {"Authorization": f"Bearer {api_key}"}
    r = client.post(f"{BASE}/v1/documents", headers=headers, json={
        "label": label,
        "text": text,
    })
    r.raise_for_status()
    doc_id = r.json()["id"]

    # pending -> review (chunk + extract)
    r = client.post(f"{BASE}/v1/documents/{doc_id}/crystallize", headers=headers)
    r.raise_for_status()

    # review -> crystallized (writes crystals; uses the saved review state)
    r = client.post(f"{BASE}/v1/documents/{doc_id}/approve", headers=headers, json={})
    r.raise_for_status()
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


def agent_turn_fields(body: dict) -> dict:
    """The per-row audit slice of an agent envelope (Guarantee #9).

    Pure: no I/O, so it is unit-testable against a literal envelope.
    `tool_calls` entries are {tool_name, input, output, ...} per the agent
    contract; only the names are kept (inputs can be large and are already
    persisted server-side as agent_events).
    """
    names = [
        str((tc or {}).get("tool_name") or "")
        for tc in (body.get("tool_calls") or [])
    ]
    names = [n for n in names if n]
    return {
        "stop_reason": body.get("stop_reason"),
        "iterations": body.get("iterations"),
        "prompt_tokens": body.get("prompt_tokens"),
        "completion_tokens": body.get("completion_tokens"),
        "tool_calls": names,
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
        temperature=0,
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

def run(args: argparse.Namespace) -> int:
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
        },
        "judge_caveat": (
            "judge and answerer are both Anthropic (different tiers, "
            "disclosed); swap --judge-model to GPT-4o when available"
        ),
        "note": args.note or None,
    }

    out_path = Path(args.out) if args.out else None
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
          "CC_ENABLE_METACOGNITION_WORKER=0 and CC_WEB_SEARCH_PROVIDER unset; "
          "any external_tool_used row disqualifies the run.\n")
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

    # type -> {"attempted", "correct", "errors"}
    by_type: dict[str, dict[str, int]] = defaultdict(
        lambda: {"attempted": 0, "correct": 0, "errors": 0}
    )
    prompt_tokens_sum = 0
    prompt_tokens_n = 0
    matched_facts_sum = 0
    iterations_sum = 0
    iterations_n = 0
    external_tool_rows = 0  # Guarantee #3: nonzero disqualifies the run
    total_correct = 0
    total_attempted = 0
    errors = 0

    with httpx.Client(timeout=TIMEOUT) as client:
        for i, q in enumerate(questions, 1):
            qid = q.get("question_id", f"q{i}")
            qtype = q.get("question_type", "unknown")
            t0 = time.time()
            row: dict = {"record": "result", "question_id": qid, "question_type": qtype}
            total_attempted += 1
            by_type[qtype]["attempted"] += 1
            try:
                customer_id, api_key = create_customer(
                    client, args.answer_model, upstream_key
                )
                row["customer_id"] = customer_id

                # Guarantee #4: strip any general-bank subscriptions (a fresh
                # customer can carry a default) so only the ingested sessions
                # are in context; verify zero remain before ingesting.
                enforce_clean_room(client, api_key)

                sessions = q.get("haystack_sessions") or []
                dates = q.get("haystack_dates") or [""] * len(sessions)
                sids = q.get("haystack_session_ids") or [
                    f"s{j}" for j in range(len(sessions))
                ]
                for sess, date, sid in zip(sessions, dates, sids):
                    text = session_to_text(sess, date)
                    ingest_session(
                        client, api_key, label=f"Session {sid} ({date})", text=text
                    )
                row["sessions_ingested"] = len(sessions)

                question_text = q["question"]
                if q.get("question_date"):
                    question_text = (
                        f"Today's date is {q['question_date']}. {question_text}"
                    )
                answer, turn = ask(
                    client, api_key, args.answer_model, question_text
                )
                row["model_answer"] = answer
                # Guarantee #9: the agent's own audit slice — what ran.
                row["stop_reason"] = turn["stop_reason"]
                row["iterations"] = turn["iterations"]
                row["tool_calls"] = turn["tool_calls"]
                row["external_tool_used"] = turn["external_tool_used"]
                # Tokens come from the envelope (the loop totals the ledger
                # charges from); retrieval telemetry from the ONE query-log
                # row finalize_agent_turn writes per turn.
                row["prompt_tokens"] = turn["prompt_tokens"]
                row["completion_tokens"] = turn["completion_tokens"]
                if turn["external_tool_used"]:
                    external_tool_rows += 1
                if isinstance(turn["iterations"], int):
                    iterations_sum += turn["iterations"]
                    iterations_n += 1

                qlog = last_query_log(client, customer_id)
                row["match_type"] = qlog.get("match_type")
                row["injection_method"] = qlog.get("injection_method")
                row["matched_facts"] = qlog.get("matched_facts") or []
                row["latency_ms"] = qlog.get("latency_ms")
                if isinstance(row["prompt_tokens"], int):
                    prompt_tokens_sum += row["prompt_tokens"]
                    prompt_tokens_n += 1
                matched_facts_sum += len(row["matched_facts"])

                correct, verdict = judge(
                    anthropic_client, args.judge_model, q, answer
                )
                row["correct"] = correct
                row["judge_verdict_raw"] = verdict
                by_type[qtype]["correct"] += int(correct)
                total_correct += int(correct)
                status = "PASS" if correct else "FAIL"
            except Exception as e:  # keep the sweep going; counted as attempted
                errors += 1
                by_type[qtype]["errors"] += 1
                row["error"] = f"{type(e).__name__}: {e}"
                status = "ERROR"

            row["elapsed_s"] = round(time.time() - t0, 1)
            if out_f:
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
            print(
                f"[{i}/{selected}] {status}  {qtype:<26} {qid}"
                f"  ({row['elapsed_s']}s"
                + (f", {row.get('prompt_tokens')} ptok" if row.get("prompt_tokens") else "")
                + (f", {len(row.get('matched_facts') or [])} facts" if row.get("matched_facts") else "")
                + (f", {row.get('iterations')} iter" if row.get("iterations") is not None else "")
                + (" ⚠ EXTERNAL TOOL" if row.get("external_tool_used") else "")
                + ")"
            )

    # ---- Results (Guarantee #7: honest denominator) ---------------------
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
        print(f"  errors: {errors}  (investigate / re-run — do not silently drop)")
    if prompt_tokens_n:
        print(f"  avg prompt tokens/question: {prompt_tokens_sum / prompt_tokens_n:.0f}")
    if total_attempted:
        print(f"  avg matched crystals/question: {matched_facts_sum / total_attempted:.1f}")
    if iterations_n:
        print(f"  avg agent iterations/question: {iterations_sum / iterations_n:.1f}")
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
              f"surface=agent, judge={args.judge_model}, temp=0. Disclose "
              f"the surface + config when citing.")
    else:
        print("  ▶ NON-HEADLINE: pass --variant s|m on a full set for a citable "
              "number.")
    if out_path:
        print(f"  manifest + per-question rows: {out_path}")
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data", required=True, help="Path to a LongMemEval variant JSON")
    ap.add_argument("--variant", default="",
                    help="Override variant detection: s | m | oracle")
    ap.add_argument("--limit", type=int, default=0,
                    help="Max questions to run (0 = full file; mind the cost)")
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
                    help="JSONL path: manifest line + per-question rows (appended)")
    return run(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
