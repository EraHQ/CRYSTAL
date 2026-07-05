#!/usr/bin/env python
"""End-of-C-series live smoke — eyeball the cost+parity wins (C0-C6).

Hits a RUNNING CRYS agent endpoint (real Anthropic calls) and prints what to
look at. This is the integration proof that complements the green unit suite.
C3 (compaction) and C4 (tool-output trimming) are internal/behavioral and
covered by their unit tests; this script focuses on the wins with a clear
single-request runtime signal: C1 (prompt caching — cache_read appears on a
warm prefix) and C6 (model selection — house default, explicit, and the
per-conversation sticky model + client override).

PREREQUISITES
  1. Server running with the agent endpoint and an Anthropic key, e.g.:

       CC_ANTHROPIC_API_KEY=sk-ant-... \\
       CC_AGENT_MODEL=claude-haiku-4-5-20251001 \\
       uvicorn crystal_cache.app:app --host 127.0.0.1 --port 8000

     CC_AGENT_MODEL is optional: set it to see the [C6] house-default scenario
     resolve to it; leave it unset to see the built-in DEFAULT_MODEL instead.
  2. A customer API key (Bearer Key A) for that server.

RUN
  CC_SMOKE_BASE_URL=http://127.0.0.1:8000 \\
  CC_SMOKE_API_KEY=<customer key> \\
  CC_SMOKE_HOUSE_DEFAULT=claude-haiku-4-5-20251001 \\
  python scripts/smoke_c_series.py

Reads only env; makes a handful of cheap Haiku calls. Eyeball each printed
"model used"/"tokens" line against the EXPECT line beneath it.
"""
from __future__ import annotations

import os
import sys
import uuid

import httpx

BASE_URL = os.environ.get("CC_SMOKE_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
API_KEY = os.environ.get("CC_SMOKE_API_KEY", "")
# What CC_AGENT_MODEL is set to on the server, if any (for the house-default
# EXPECT line). Empty = the server is using the built-in DEFAULT_MODEL.
HOUSE_DEFAULT = os.environ.get("CC_SMOKE_HOUSE_DEFAULT", "")

HAIKU = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-5-20250929"
ENDPOINT = f"{BASE_URL}/v1/agent/messages"


def _post(*, prompt: str, model=None, sequence_id=None) -> dict:
    body: dict = {"messages": [{"role": "user", "content": prompt}], "max_tokens": 256}
    if model is not None:
        body["model"] = model
    if sequence_id is not None:
        body["metadata"] = {"sequence_id": sequence_id}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    try:
        r = httpx.post(ENDPOINT, json=body, headers=headers, timeout=120.0)
    except httpx.ConnectError:
        raise SystemExit(
            f"ERROR: can't reach {ENDPOINT}. Start uvicorn with the agent "
            f"endpoint (and CC_ANTHROPIC_API_KEY) first."
        )
    if r.status_code >= 400:
        raise SystemExit(
            f"ERROR: HTTP {r.status_code} from {ENDPOINT}\n  {r.text[:500]}\n"
            f"(A 400 about ANTHROPIC_API_KEY means the server has no Anthropic "
            f"key configured.)"
        )
    return r.json()


def _line(label: str, value) -> None:
    print(f"    {label:<26} {value}")


def _tokens(res: dict) -> str:
    return (
        f"in={res.get('prompt_tokens')} out={res.get('completion_tokens')} "
        f"cache_write={res.get('cache_creation_tokens')} "
        f"cache_read={res.get('cache_read_tokens')}"
    )


def _rid(tag: str) -> str:
    return f"smoke-{tag}-{uuid.uuid4().hex[:6]}"


def main() -> int:
    if not API_KEY:
        print("ERROR: set CC_SMOKE_API_KEY to a customer (Bearer Key A) key.", file=sys.stderr)
        return 2
    print(f"CRYS C-series live smoke -> {ENDPOINT}\n")

    # --- C1: prompt caching (warm prefix across two requests) -----------------
    print("[C1] Prompt caching - two requests, same customer (shared system+tools prefix).")
    r1 = _post(prompt="In one word, the capital of France?", model=HAIKU, sequence_id=_rid("c1a"))
    _line("request 1 model:", r1.get("model"))
    _line("request 1 tokens:", _tokens(r1))
    r2 = _post(prompt="In one word, the capital of Japan?", model=HAIKU, sequence_id=_rid("c1b"))
    _line("request 2 model:", r2.get("model"))
    _line("request 2 tokens:", _tokens(r2))
    print("    EXPECT: request 2 cache_read > 0 (system+tools prefix cached on request 1).\n")

    # --- C6.1: explicit model -------------------------------------------------
    print("[C6] Explicit model - body.model is honored.")
    re = _post(prompt="Say OK.", model=HAIKU, sequence_id=_rid("explicit"))
    _line("model used:", re.get("model"))
    _line("EXPECT:", f"{HAIKU}\n")

    # --- C6.2: per-conversation sticky + client override ----------------------
    print("[C6] Sticky model - set once, reused on a no-model follow-up; client can re-override.")
    sid = _rid("sticky")
    s1 = _post(prompt="Say OK.", model=HAIKU, sequence_id=sid)
    _line("turn 1 (model=haiku):", s1.get("model"))
    s2 = _post(prompt="Say OK again.", model=None, sequence_id=sid)  # reuse saved
    _line("turn 2 (no model):", s2.get("model"))
    _line("EXPECT:", f"both {HAIKU} (turn 2 reused the saved model)")
    s3 = _post(prompt="Say OK once more.", model=SONNET, sequence_id=sid)  # override + re-save
    _line("turn 3 (model=sonnet):", s3.get("model"))
    s4 = _post(prompt="And again.", model=None, sequence_id=sid)  # reuse the new save
    _line("turn 4 (no model):", s4.get("model"))
    _line("EXPECT:", f"turns 3 & 4 {SONNET} (client override re-saved; last-writer-wins)\n")

    # --- C6.3: house default --------------------------------------------------
    print("[C6] House default - no model, fresh conversation.")
    h = _post(prompt="Say OK.", model=None, sequence_id=_rid("house"))
    _line("model used:", h.get("model"))
    if HOUSE_DEFAULT:
        _line("EXPECT:", f"{HOUSE_DEFAULT} (the server's CC_AGENT_MODEL)")
    else:
        _line("EXPECT:", f"{SONNET} (built-in DEFAULT_MODEL; set CC_AGENT_MODEL to change)")
    print()

    print("Note: C3 (compaction) and C4 (tool-output trimming) are internal and")
    print("proven by their unit suites - no clean single-request signal. Enable")
    print("CC_AGENT_COMPACTION / CC_AGENT_TOOL_OUTPUT_MAX_CHARS and watch a long")
    print("multi-tool session if you want to eyeball them live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
