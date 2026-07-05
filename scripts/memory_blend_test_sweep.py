#!/usr/bin/env python3
"""Memory-blend live test sweep — exercises Inc 1-3 through the chat proxy.

Fire-and-watch: run this against a live server and read the uvicorn logs.
It drives single-turn and multi-turn conversations that trigger each
memory-blend behavior, printing a labeled header + what to watch before
each turn. Multi-turn scenarios capture the real assistant reply and feed
it into the next turn, so sequence_id / follow-up detection behave as in
production.

Pass criteria for every scenario are in docs/MEMORY_BLEND_TEST_SWEEP.md.

Usage
-----
    # 1. Start the server (with the #3 + memory-blend code loaded):
    CC_TEXT_ENCODER=semantic uvicorn crystal_cache.app:app --host 0.0.0.0 --port 8000

    # 2. In another shell:
    export CC_API_KEY=<the customer's Bearer key>
    export CC_BASE_URL=http://localhost:8000      # optional, this is the default
    python scripts/memory_blend_test_sweep.py      # run all
    python scripts/memory_blend_test_sweep.py 3 4  # run only scenarios 3 and 4

Only stdlib is used (urllib) — no extra deps.

NOTE: prompts assume the bank has crystal-cache content (crystallization,
sparse_keys, cognition). Tweak the TOPIC prompts to match your bank if a
retrieval scenario doesn't match anything — the point is the PATH behavior
(retrieve vs passthrough), not exact match quality.
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request

BASE = os.environ.get("CC_BASE_URL", "http://localhost:8000").rstrip("/")
KEY = os.environ.get("CC_API_KEY", "")
MODEL = os.environ.get("CC_TEST_MODEL", "claude-sonnet-4-5-20250929")


def _post(messages: list[dict], sequence_id: str) -> dict:
    body = {
        "model": MODEL,
        "messages": messages,
        "metadata": {"sequence_id": sequence_id},
    }
    req = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {KEY}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        print(f"    !! HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}")
        raise
    except urllib.error.URLError as e:
        print(f"    !! connection error: {e}. Is the server running at {BASE}?")
        raise


def _assistant_text(resp: dict) -> str:
    try:
        return resp["choices"][0]["message"].get("content") or ""
    except Exception:
        return ""


def turn(history: list[dict], user_text: str, seq: str, watch: str) -> list[dict]:
    """Send one user turn; print what to watch + the reply; return new history."""
    msgs = history + [{"role": "user", "content": user_text}]
    print(f"\n  -> user: {user_text!r}")
    print(f"     WATCH: {watch}")
    resp = _post(msgs, seq)
    reply = _assistant_text(resp)
    print(f"  <- assistant: {reply[:160]!r}")
    return msgs + [{"role": "assistant", "content": reply}]


def header(n: int, title: str, covers: str) -> None:
    print("\n" + "=" * 72)
    print(f"SCENARIO {n}: {title}")
    print(f"covers: {covers}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

def s1_resemblance():
    header(1, "Resemblance retrieval (baseline)", "retrieval still works")
    turn([], "how does crystallization work in this system", "sweep_s1",
         "recall.matched + retrieval.injected; match_type; push_pull.tools_injected=5")


def s2_identity():
    header(2, "Identity query (key-scan routing)",
           "Inc 4 / D-MB6 — 'where is X' routes to a sparse-key scan")
    turn([], "where is generate_sparse_key defined", "sweep_s2",
         "if symbol in bank: navigation_dispatch.identity_hit + "
         "retrieval.identity_routed (recall SKIPPED); injected 'Source: Locator' "
         "header; answer names the module. Else falls through to recall.")


def s3_followup_skip():
    header(3, "Follow-up skip (short)", "Inc 2 — follow-up gate")
    h = turn([], "how does crystallization work in this system", "sweep_s3",
             "TURN 1: recall.matched + retrieval.injected (retrieves)")
    turn(h, "got it — give me a bit more on that", "sweep_s3",
         "TURN 2: retrieval.followup_passthrough; NO recall.matched (skipped)")


def s4_followup_subject():
    header(4, "Follow-up skip via carried subject (long)",
           "Inc 3 — session consumption strengthens detection")
    h = turn([], "how does crystallization work in this system", "sweep_s4",
             "TURN 1: matches → subject carried to query_logs")
    turn(h,
         "okay and then walk me through the next part after that in quite a "
         "lot more detail than you just gave me right now please",
         "sweep_s4",
         "TURN 2: long + no intent, but prior turn matched → "
         "retrieval.followup_passthrough (subject carried forward)")


def s5_midconv_lookup():
    header(5, "Mid-conversation new lookup still retrieves",
           "intent overrides follow-up")
    h = turn([], "how does crystallization work in this system", "sweep_s5",
             "TURN 1: retrieves")
    turn(h, "what is the cognition loop", "sweep_s5",
         "TURN 2: 'what is' = intent → RETRIEVES (retrieval.injected), "
         "NOT passthrough")


def s6_pull_on_demand():
    header(6, "Pull-on-demand", "skipped retrieval + model pulls if it needs to")
    h = turn([], "how does crystallization work in this system", "sweep_s6",
             "TURN 1: retrieves")
    turn(h, "now tell me the exact default poll interval for the drive worker",
         "sweep_s6",
         "TURN 2: likely passthrough; watch for push_pull (crystal_pull_research "
         "or crystal_push_gap) — the model pulling rather than us pre-injecting")


def s7_compaction():
    header(7, "Compaction (12 turns)", "Inc 1 — compaction + long-term")
    print("  (12 real turns; watch for compaction.triggered around turn 10,\n"
          "   then compaction.complete with summary_chars; prompt_tokens should\n"
          "   stop growing with full history. mem0 off → rule-based summary.)")
    h: list[dict] = []
    prompts = [
        "let's talk about this codebase. what is a crystal",
        "what is a fact inside a crystal",
        "what are sparse keys",
        "how does the crystallization worker pick up documents",
        "what is the cognition worker for",
        "what does the validator do",
        "what is a knowledge gap",
        "how does the push/pull protocol work",
        "what is mem0 used for here",
        "what is compaction",        # ~turn 10 — compaction should trigger here
        "what is the routing window",
        "summarize everything we covered about memory",  # relies on summary+recent
    ]
    for i, p in enumerate(prompts, start=1):
        watch = ("compaction.triggered/compaction.complete expected from here"
                 if i >= 10 else f"turn {i} (building history)")
        h = turn(h, p, "sweep_s7", watch)


def s8_gap_trigger():
    header(8, "Trigger a gap (then watch idle logs)",
           "Track C — cognition backoff after restart")
    turn([], "what is the exact absolute filesystem path of the sparse_keys "
             "module on this machine", "sweep_s8",
         "model likely emits crystal_push_gap (push_pull.gap_identified). "
         "THEN watch idle logs ~30 min: gap_unfilled should now carry "
         "attempts= and parked=; after 3 attempts the gap parks (no more "
         "every-~11-min retries).")


SCENARIOS = {
    1: s1_resemblance, 2: s2_identity, 3: s3_followup_skip,
    4: s4_followup_subject, 5: s5_midconv_lookup, 6: s6_pull_on_demand,
    7: s7_compaction, 8: s8_gap_trigger,
}


def main(argv: list[str]) -> int:
    if not KEY:
        print("Set CC_API_KEY to the customer's Bearer key first. Example:")
        print("  export CC_API_KEY=sk-...")
        return 2
    wanted = [int(a) for a in argv if a.isdigit()] or sorted(SCENARIOS)
    print(f"Target: {BASE}  model={MODEL}  scenarios={wanted}")
    print("Watch the uvicorn logs in your other shell as this runs.")
    for n in wanted:
        fn = SCENARIOS.get(n)
        if fn is None:
            print(f"  (no scenario {n})")
            continue
        try:
            fn()
        except Exception as e:
            print(f"  !! scenario {n} aborted: {e}")
    print("\n" + "=" * 72)
    print("Sweep complete. Cross-check against docs/MEMORY_BLEND_TEST_SWEEP.md.")
    print("Quick log greps:")
    print("  retrieval.followup_passthrough   (scenarios 3,4,6 turn 2)")
    print("  compaction.triggered             (scenario 7, ~turn 10)")
    print("  push_pull.gap_identified         (scenario 8)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
