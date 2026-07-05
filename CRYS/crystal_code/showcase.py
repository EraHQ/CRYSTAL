"""CRYS Showcase — `python -m crystal_code --showcase`.

One command that spins up a fresh in-process CRYS and walks the whole surface,
for showing others "the true power of CRYS and crystals." It orchestrates the
same headless entry points the CLI already uses (no server, no setup), against
a fresh temp store + the inline "Helios" fixture, printing a panel per act and
saving a report. Design + act list: docs/SHOWCASE_PLAN.md.

Acts 1-8 are filled in incrementally; today Act 0 (ingest) is live and the rest
announce themselves so the one command runs end-to-end. Run a subset with
`--showcase-acts 0,5` and keep the workspace with `--showcase-keep`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

# Flags the demo needs ON, set BEFORE any crystal_cache import builds Settings
# (get_settings is lru_cached). Done at import time because this module is only
# imported on the --showcase path.
for _k, _v in {
    "CC_TEXT_ENCODER": "semantic",
    "CC_ENABLE_CITATIONS": "1",
    "CC_ENABLE_MARKETPLACE_METERING": "1",
    "CC_ENABLE_COST_ACCOUNTING": "1",
    "CC_AGENT_RECALL": "1",
}.items():
    os.environ.setdefault(_k, _v)

from . import config_store, style  # noqa: E402


# ---------------------------------------------------------------------------
# The "Helios" fixture — a small, coherent SaaS, materialized per run.
# ---------------------------------------------------------------------------

_HELIOS_FILES: dict[str, str] = {
    "README.md": (
        "# Helios\n\n"
        "Helios is a small order-management service: it prices carts, validates\n"
        "orders, and authorizes requests with short-lived tokens. This repo is a\n"
        "demo fixture for the CRYS showcase.\n\n"
        "## Modules\n"
        "- `pricing.py` — cart totals and discounts\n"
        "- `orders.py` — order creation and validation\n"
        "- `auth.py` — token authorization\n"
    ),
    "docs/ARCHITECTURE.md": (
        "# Helios Architecture\n\n"
        "Helios has three layers. The **pricing layer** (`pricing.py`) computes a\n"
        "cart total from line items and applies a percentage discount. The\n"
        "**orders layer** (`orders.py`) turns a validated cart into an Order and is\n"
        "the single place order validation lives. The **auth layer** (`auth.py`)\n"
        "authorizes a request by checking a bearer token against the active set.\n\n"
        "Order validation is centralized in `orders.validate_cart` so every entry\n"
        "point enforces the same rules: non-empty cart, positive quantities, and a\n"
        "known currency.\n"
    ),
    "docs/SECURITY.md": (
        "# Helios Security\n\n"
        "Tokens are short-lived and opaque. `auth.authorize` rejects a request\n"
        "when the token is missing, unknown, or expired; it never logs the token\n"
        "value. Authorization failures return a generic message so a caller can't\n"
        "distinguish 'unknown' from 'expired'.\n"
    ),
    "pricing.py": (
        '"""Cart pricing for Helios."""\n'
        "from __future__ import annotations\n\n\n"
        "def line_total(unit_price: float, quantity: int) -> float:\n"
        '    """Total for one line item."""\n'
        "    return round(unit_price * quantity, 2)\n\n\n"
        "def cart_total(items: list[dict]) -> float:\n"
        '    """Sum of line totals for a cart of {unit_price, quantity} items."""\n'
        "    return round(sum(line_total(i['unit_price'], i['quantity']) for i in items), 2)\n\n\n"
        "def apply_discount(total: float, percent: float) -> float:\n"
        '    """Apply a percentage discount (0-100) to a total."""\n'
        "    return round(total * (1 - percent / 100.0), 2)\n"
    ),
    "orders.py": (
        '"""Order creation and validation for Helios."""\n'
        "from __future__ import annotations\n\n"
        "from dataclasses import dataclass\n\n"
        "KNOWN_CURRENCIES = {'USD', 'EUR', 'GBP'}\n\n\n"
        "@dataclass\n"
        "class Order:\n"
        "    items: list[dict]\n"
        "    currency: str\n"
        "    total: float\n\n\n"
        "def validate_cart(items: list[dict], currency: str) -> None:\n"
        '    """Raise ValueError if the cart is invalid. The one place order\n'
        '    rules live (see docs/ARCHITECTURE.md)."""\n'
        "    if not items:\n"
        "        raise ValueError('cart is empty')\n"
        "    if currency not in KNOWN_CURRENCIES:\n"
        "        raise ValueError(f'unknown currency: {currency}')\n\n\n"
        "def create_order(items: list[dict], currency: str) -> Order:\n"
        '    """Validate a cart and build an Order."""\n'
        "    from pricing import cart_total\n"
        "    validate_cart(items, currency)\n"
        "    return Order(items=items, currency=currency, total=cart_total(items))\n"
    ),
    "auth.py": (
        '"""Token authorization for Helios."""\n'
        "from __future__ import annotations\n\n"
        "_ACTIVE_TOKENS = {'tok_live_demo'}\n\n\n"
        "def authorize(token: str | None) -> bool:\n"
        '    """True if the token is active. Never logs the token value."""\n'
        "    return bool(token) and token in _ACTIVE_TOKENS\n"
    ),
    "test_helios.py": (
        '"""Helios test suite (the showcase verify command runs this)."""\n'
        "from pricing import cart_total, apply_discount\n"
        "from orders import create_order, validate_cart\n"
        "from auth import authorize\n"
        "import pytest\n\n\n"
        "def test_cart_total():\n"
        "    items = [{'unit_price': 10.0, 'quantity': 2}, {'unit_price': 5.0, 'quantity': 1}]\n"
        "    assert cart_total(items) == 25.0\n\n\n"
        "def test_discount():\n"
        "    assert apply_discount(100.0, 20) == 80.0\n\n\n"
        "def test_create_order_ok():\n"
        "    order = create_order([{'unit_price': 3.0, 'quantity': 4}], 'USD')\n"
        "    assert order.total == 12.0 and order.currency == 'USD'\n\n\n"
        "def test_validate_rejects_empty():\n"
        "    with pytest.raises(ValueError):\n"
        "        validate_cart([], 'USD')\n\n\n"
        "def test_authorize():\n"
        "    assert authorize('tok_live_demo') is True\n"
        "    assert authorize(None) is False\n"
    ),
    ".gitignore": "__pycache__/\n*.pyc\n",
    ".crystal-code.json": json.dumps({"verify_command": "python -m pytest -q"}, indent=2) + "\n",
}


def _materialize_helios(dest: Path) -> None:
    for rel, content in _HELIOS_FILES.items():
        p = dest / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=60)


# ---------------------------------------------------------------------------
# Shared context + metrics
# ---------------------------------------------------------------------------

@dataclass
class Showcase:
    workspace: Path
    project_dir: Path
    db_path: Path
    customer_id: str
    creds: Any
    store: Any = None
    encoder: Any = None
    vector_store: Any = None
    fact_vector_store: Any = None
    client: Any = None
    models: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    panels: list = field(default_factory=list)  # (act_name, status, [lines])

    def db_arg(self) -> str:
        return str(self.db_path)


# ---------------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------------

def _banner(text: str) -> None:
    print("\n" + style.rule())
    print(style.bold(text))
    print(style.rule())


def _panel(sc: Showcase, act_name: str, status: str, lines: list[str]) -> None:
    sc.panels.append((act_name, status, lines))
    for ln in lines:
        print("  " + ln)


# ---------------------------------------------------------------------------
# Bootstrap — fresh store + fixture, all in-process
# ---------------------------------------------------------------------------

async def _bootstrap(workspace: Path) -> Showcase:
    from crystal_cache.encoding import build_text_encoder
    from crystal_cache.infrastructure import VectorStore
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.metadata_store import set_metadata_store
    from .runtime import (
        _make_store, _resolve_db_url, _seed_legacy_crystal_types,
        build_llm_client, resolve_models,
    )

    creds = config_store.resolve_credentials()
    if creds is None:
        raise SystemExit(
            "No credentials found. Launch `crys` once (or set ANTHROPIC_API_KEY) "
            "before running the showcase."
        )

    project_dir = workspace / "helios"
    _materialize_helios(project_dir)
    # A clean initial commit so the background-agent acts have the clean
    # worktree they require.
    if _git(["rev-parse", "--git-dir"], project_dir).returncode != 0:
        _git(["init"], project_dir)
        _git(["add", "-A"], project_dir)
        _git(["-c", "user.name=helios", "-c", "user.email=helios@demo.local",
              "commit", "-m", "Helios fixture (initial)"], project_dir)

    db_path = workspace / "showcase.db"
    print(style.dim("  setting up a fresh knowledge store..."))
    store = _make_store(_resolve_db_url(str(db_path)))
    await store.init()
    set_metadata_store(store)
    await _seed_legacy_crystal_types(store)

    print(style.dim("  loading the semantic encoder (first run downloads gtr-t5-base)..."))
    encoder = build_text_encoder()
    vector_store = VectorStore(store=store)
    fact_vector_store = FactVectorStore(store=store)

    # A real customer row so later acts (operators/ACLs/credit) have a team.
    customer = await store.create_customer(
        provider=creds.provider, model_id=creds.model, api_key_ref="config:showcase",
    )
    models = resolve_models(creds, {})
    # Provider-neutral per-user client through the seam (anthropic or any
    # OpenAI-compatible endpoint); tiers map fast->small, main->large.
    client = build_llm_client(creds, models)

    return Showcase(
        workspace=workspace, project_dir=project_dir, db_path=db_path,
        customer_id=customer.id, creds=creds, store=store, encoder=encoder,
        vector_store=vector_store, fact_vector_store=fact_vector_store,
        client=client, models=models,
    )


# ---------------------------------------------------------------------------
# Act 0 — Seed: ingest Helios (docs + code) into a fresh bank
# ---------------------------------------------------------------------------

async def act_seed(sc: Showcase) -> None:
    """Ingest Helios through CRYS's real ingestion engine.

    scan_project + ingest_files is exactly what `crys --ingest` runs. We drive
    it against the SHARED components (the way the production crystallization
    worker uses the app's startup singletons — not a fresh per-command store),
    and the pipeline self-invalidates the fact index after writing, so the next
    act's retrieval sees the new bank with no manual refresh. Genuine engine; we
    only observe the result.
    """
    from .ingest import scan_project, ingest_files

    scan = scan_project(sc.project_dir, include_docs=True)
    summary = await ingest_files(
        sc.project_dir, scan.files,
        store=sc.store, encoder=sc.encoder, vector_store=sc.vector_store,
        fact_vector_store=sc.fact_vector_store, customer_id=sc.customer_id,
        client=sc.client,
    )
    total = await sc.store.count_crystals_for_customer(sc.customer_id)
    sc.metrics["crystals_total"] = total
    sc.metrics["files_ingested"] = summary.written

    lines = [
        f"ingested {summary.written} files "
        f"({len(scan.code_files)} code, {len(scan.doc_files)} docs) — "
        f"{summary.crystals} crystal writes, {summary.unchanged} unchanged, {summary.failed} failed",
        f"write-side bonding consolidated those into {total} distinct crystals "
        f"(related facts bond into one crystal at ingest, not after)",
        style.dim("  (the sparse keys these are filed under are shown in Act 1)"),
    ]
    _panel(sc, "Seed", "ok", lines)


# ---------------------------------------------------------------------------
# Acts 1-7 — filled in next (announce for now so the tour runs end-to-end)
# ---------------------------------------------------------------------------

async def _todo(sc: Showcase, name: str, what: str) -> None:
    _panel(sc, name, "todo", [style.dim(f"(coming next) {what}")])


async def _make_agent(sc: Showcase, *, max_tokens: int = 2048):
    """Construct a library Agent from the SHARED bootstrap components (the same
    store + vector stores + encoder + client every act uses — mirroring the web
    app's startup singletons and how /v1/agent/messages builds an agent over
    them). Constructing the Agent registers the tool registry and injects the
    tool state the retrievers read."""
    from crystal_cache.agent import Agent
    customer = await sc.store.get_customer_by_id(sc.customer_id)
    return Agent(
        customer=customer,
        llm=sc.client,
        tool_state={
            "store": sc.store, "vector_store": sc.vector_store,
            "fact_vector_store": sc.fact_vector_store, "encoder": sc.encoder,
            "decomposer": None,
        },
        model=sc.models["main"],
        max_tokens=max_tokens,
    )


def _surfaced_crystal_ids(result: dict) -> list[str]:
    """Crystal ids the agent surfaced through retrieval tools this run."""
    ids: list[str] = []
    for call in (result.get("tool_calls") or []):
        out = call.get("output")
        if isinstance(out, dict):
            for cid in (out.get("matched_crystal_ids") or []):
                if cid and cid not in ids:
                    ids.append(cid)
    return ids


async def _crystal_key(sc: Showcase, crystal_id: str) -> str:
    """A crystal's sparse key for display = its first fact's prompt_text."""
    try:
        facts = await sc.store.list_facts_for_crystal(crystal_id)
        if facts and getattr(facts[0], "prompt_text", None):
            return str(facts[0].prompt_text)
    except Exception:
        pass
    return crystal_id


def _short_key(key: str, n: int = 30) -> str:
    """Compact a sparse key for inline display."""
    key = " ".join(str(key).split())
    return key if len(key) <= n else key[: n - 1] + "…"


async def _grounding_scores(sc: Showcase, result: dict) -> list[tuple[str, float, bool]]:
    """(sparse key, grounding score, grounded?) for each crystal CRYS surfaced
    this turn — read back from the recorded P3 citation rows (newest citation
    per crystal), sorted by score descending.

    Surfaces the actual whole-answer↔source cosine SPREAD the agent grounding
    produced, so the threshold's discrimination is visible (which crystals fall
    above/below the bar) instead of just a grounded/not count."""
    rows: list[tuple[str, float, bool]] = []
    for cid in _surfaced_crystal_ids(result):
        try:
            cites = await sc.store.list_citations_for_crystal(
                sc.customer_id, cid, grounded_only=False,
            )
        except Exception:  # noqa: BLE001
            cites = []
        if not cites:
            continue
        latest = cites[0]  # newest first
        score = latest.get("grounding_score")
        rows.append((
            await _crystal_key(sc, cid),
            float(score) if score is not None else 0.0,
            bool(latest.get("grounded")),
        ))
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


async def act_ask(sc: Showcase) -> None:
    # Front-door behavior: we do NOT call the retrieval tools ourselves. We ASK
    # the agent two differently-shaped questions — the way an operator does —
    # and OBSERVE which retrieval tool CRYS reaches for, plus the answer and the
    # citations its post-turn layer grounds. CRYS's genuine system prompt + the
    # same knowledge-bank affordance the REPL injects (_BANK_ADDENDUM) steer the
    # routing; nothing hand-holds the tool choice, and a wrong choice is shown
    # as a MISS, not papered over (no fallback that masks it).
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.turn_finalize import finalize_agent_turn
    from crystal_cache.config import get_settings
    from .cli import _BANK_ADDENDUM

    customer = await sc.store.get_customer_by_id(sc.customer_id)
    ground_thr = get_settings().agent_citation_grounding_threshold

    def _system(agent: Any) -> str:
        return build_system_prompt(agent.customer, agent.tools) + _BANK_ADDENDUM

    def _tools_used(result: dict) -> list[str]:
        used: list[str] = []
        for c in (result.get("tool_calls") or []):
            n = c.get("tool_name")
            if n and n not in used:
                used.append(n)
        return used

    # Semantic = meaning-ranked retrieval; structural = key-registry scan.
    _SEMANTIC = {"knowledge_search", "content_search", "depth_search"}

    async def _ask(question: str, max_tokens: int = 2048) -> tuple[dict, dict]:
        """One operator-style question through the genuine agent + the genuine
        post-turn layer (finalize_agent_turn: grounds citations at the
        configured agent threshold, records the cost row, emits the MCR trace)."""
        agent = await _make_agent(sc, max_tokens=max_tokens)
        result = await agent.run(
            messages=[{"role": "user", "content": question}],
            system=_system(agent),
        )
        finalized = await finalize_agent_turn(
            store=sc.store, encoder=sc.encoder, customer=customer,
            anthropic_client=sc.client, result=result,
            user_query=question, sequence_id=result.get("id"), origin="agent",
        )
        return result, finalized

    cost_total = 0

    # --- Q1 — resemblance: a meaning question. CRYS should reach for a semantic
    # retrieval tool (knowledge_search / content_search), not a structural one.
    q1 = "How does Helios keep order validation rules consistent across entry points?"
    r1, f1 = await _ask(q1)
    used1 = _tools_used(r1)
    a1 = " ".join((r1.get("final_text") or "").split())
    scores1 = await _grounding_scores(sc, r1)
    grounded1 = [k for (k, s, g) in scores1 if g]
    cost_total += f1.get("cost_micro_usd") or 0
    routed1_ok = bool(set(used1) & _SEMANTIC)
    verdict1 = (
        "routed to semantic retrieval"
        if routed1_ok else
        f"MISS — expected semantic retrieval, CRYS used "
        f"{', '.join(used1) or 'no retrieval'}"
    )
    spread1 = "; ".join(
        f"{_short_key(k)} {s:.2f}{'✓' if g else '✗'}" for k, s, g in scores1[:10]
    ) + (f"  (+{len(scores1) - 10} more)" if len(scores1) > 10 else "")
    _panel(sc, "Ask · resemblance", "ok" if routed1_ok else "error", [
        f'q: "{q1}"',
        f"CRYS chose: {', '.join(used1) or '(none)'} — {verdict1}",
        ("answer: " + a1[:220] + ("…" if len(a1) > 220 else "")) if a1 else "answer: (none)",
        (f"grounding vs the answer (threshold {ground_thr:.2f}) — "
         f"{len(grounded1)} of {len(scores1)} surfaced crystals grounded:"
         if scores1 else "no crystals surfaced to ground against the answer"),
        *([style.dim("  " + spread1)] if scores1 else []),
    ])

    # --- Q2 — identity / enumeration: CRYS should reach for key_scan (the
    # enumeration primitive its own bank affordance names for "list every X").
    q2 = "List every function defined in pricing.py."
    r2, f2 = await _ask(q2)
    used2 = _tools_used(r2)
    a2 = " ".join((r2.get("final_text") or "").split())
    cost_total += f2.get("cost_micro_usd") or 0
    routed2_ok = "key_scan" in used2
    verdict2 = (
        "routed to key_scan"
        if routed2_ok else
        f"MISS — expected key_scan, CRYS used {', '.join(used2) or 'no retrieval'}"
    )
    _panel(sc, "Ask · identity / enumeration", "ok" if routed2_ok else "error", [
        f'q: "{q2}"',
        f"CRYS chose: {', '.join(used2) or '(none)'} — {verdict2}",
        ("answer: " + a2[:220] + ("…" if len(a2) > 220 else "")) if a2 else "answer: (none)",
    ])

    # Self-traffic note: these are the customer's OWN crystals, so grounded
    # citations mint no marketplace credit by design — cross-owner credit is
    # Act 6/7. Routing verdicts are recorded plainly so a miss shows in the
    # dashboard and report rather than being smoothed over.
    sc.metrics["ask_resemblance_tool"] = ", ".join(used1) or "none"
    sc.metrics["ask_resemblance_routing"] = "ok" if routed1_ok else "MISS"
    sc.metrics["ask_enumeration_tool"] = ", ".join(used2) or "none"
    sc.metrics["ask_enumeration_routing"] = "ok" if routed2_ok else "MISS"
    sc.metrics["ask_grounding_threshold"] = f"{ground_thr:.2f}"
    sc.metrics["ask_surfaced_crystals"] = len(scores1)
    sc.metrics["ask_grounded_citations"] = len(grounded1)
    if scores1:
        sc.metrics["ask_grounding_spread"] = f"{scores1[0][1]:.2f}..{scores1[-1][1]:.2f}"
    if cost_total:
        sc.metrics["ask_cost_micro_usd"] = cost_total


# The task Act 2 hands the background agent — a small, genuinely verifiable
# Helios feature. CRYS plans it, writes it, runs the verify command (pytest),
# and commits to its own branch; we only observe the outcome.
_ACT2_TASK = (
    "Add a function total_with_tax(items, tax_percent) to pricing.py. It should "
    "compute the cart total using the existing cart_total function and apply "
    "tax_percent percent on top, rounded to 2 decimals. Add a focused test for "
    "it to test_helios.py covering one simple case. Once the tests pass, record "
    "a concise note in your long-term memory about what total_with_tax does and "
    "how it is implemented, so a future task can reuse it without re-reading the "
    "code."
)
_ACT2_BRANCH = "agent/showcase-act2"


def _diffstat_summary(diffstat: str) -> str:
    """Last line of `git diff --stat` (the 'N files changed, ...' summary)."""
    lines = [ln.strip() for ln in (diffstat or "").splitlines() if ln.strip()]
    return lines[-1] if lines else "(no changes)"


def _event_after(created_at: Any, t0: datetime) -> bool:
    """True if an event's created_at is at/after t0 (tz-robust; SQLite can hand
    back a naive datetime or an ISO string)."""
    dt = created_at
    if dt is None:
        return False
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except ValueError:
            return True  # unparseable -> don't exclude
    if getattr(dt, "tzinfo", None) is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt >= t0


async def act_build(sc: Showcase) -> None:
    """Hand the background agent a real task and observe the build.

    Drives run_background_task — the genuine `--task` headless front door
    (preflight -> plan -> execute -> verify-loop -> ground-truth verify ->
    commit -> restore branch). It builds its own store/agent over the shared DB
    (faithful to the daemon being a separate process), prints its own report
    inline, and writes session/events/cost/reflection to the DB. We then read
    the outcome back through the shared store + git — never reaching past the
    front door, never fabricating success (a non-zero exit shows as a miss).
    """
    from .background import run_background_task

    # Where the agent branches from (git's default branch for the fixture repo;
    # don't assume master vs main). run_background_task restores it at the end.
    original = (
        _git(["rev-parse", "--abbrev-ref", "HEAD"], sc.project_dir).stdout.strip()
        or "HEAD"
    )
    crystals_before = await sc.store.count_crystals_for_customer(sc.customer_id)
    facts_before = await sc.store.list_recent_facts_for_customer(sc.customer_id, limit=None)
    fact_ids_before = {f.id for f in facts_before}
    t0 = datetime.now(timezone.utc)

    rc = await run_background_task(
        sc.project_dir, _ACT2_TASK, _ACT2_BRANCH,
        str(sc.db_path), sc.customer_id,
    )

    # --- Observe the outcome through the store + git (no reaching past) ---
    verdict = {
        0: "built + verified (committed)",
        1: "verify FAILED — committed on the branch for review",
        2: "aborted before completing",
    }.get(rc, f"exited rc={rc}")

    diffstat = _git(
        ["diff", "--stat", f"{original}..{_ACT2_BRANCH}"], sc.project_dir
    ).stdout.strip()
    commit_subj = _git(
        ["log", "-1", "--format=%s", _ACT2_BRANCH], sc.project_dir
    ).stdout.strip()

    crystals_after = await sc.store.count_crystals_for_customer(sc.customer_id)
    delta = crystals_after - crystals_before
    all_facts = await sc.store.list_recent_facts_for_customer(sc.customer_id, limit=None)
    new_facts = [f for f in all_facts if f.id not in fact_ids_before]
    if new_facts:
        nf = new_facts[0]  # newest-first; the note CRYS just recorded
        recorded = (nf.claim_text or nf.answer_value or "").strip()
        tail = "…" if len(recorded) > 120 else ""
        body = f": {recorded[:120]}{tail}" if recorded else ""
        learned_note = (
            f'CRYS recorded {len(new_facts)} new fact(s) from this work — "{nf.prompt_text}"{body}'
        )
    else:
        learned_note = "CRYS recorded nothing new this run"

    # Cost / tokens / caching from this run's turn_completed events (DB-direct,
    # so no fact-index staleness; Act 1 emits no such events, so these are this
    # act's passes). cache_read climbs as the trajectory is re-sent across
    # passes when prompt caching is active.
    evs = await sc.store.list_events_for_team(
        sc.customer_id, event_types=["turn_completed"], limit=20,
    )
    passes = [e for e in evs if _event_after(e.get("created_at"), t0)] or evs
    cost_micro = sum((e.get("cost_micro_usd") or 0) for e in passes)
    tok_in = sum((e.get("tokens_input") or 0) for e in passes)
    tok_out = sum((e.get("tokens_output") or 0) for e in passes)
    cache_read = sum(
        ((e.get("payload") or {}).get("cache_read_tokens") or 0) for e in passes
    )
    cache_note = (
        f"; prompt caching active — {cache_read} cached-read tok across {len(passes)} pass(es)"
        if cache_read else ""
    )

    status = "ok" if rc == 0 else "error"
    _panel(sc, "Build", status, [
        f'task: "{_ACT2_TASK[:150]}{"…" if len(_ACT2_TASK) > 150 else ""}"',
        "flow: plan → execute → verify-loop → ground-truth verify → commit (own branch)",
        f"verdict: {verdict}",
        (f"committed to {_ACT2_BRANCH}: {commit_subj}"
         if commit_subj else f"branch {_ACT2_BRANCH}: no commit"),
        f"changes: {_diffstat_summary(diffstat)}",
        f"learned: {learned_note}",
        f"recorded cost: {cost_micro} micro-USD over {len(passes)} pass(es); "
        f"tokens in={tok_in} out={tok_out}{cache_note}",
    ])

    sc.metrics["build_verdict"] = {
        0: "verified", 1: "verify_failed", 2: "aborted",
    }.get(rc, f"rc_{rc}")
    sc.metrics["build_branch"] = _ACT2_BRANCH
    sc.metrics["build_changes"] = _diffstat_summary(diffstat)
    sc.metrics["build_crystal_delta"] = delta
    sc.metrics["build_facts_written"] = len(new_facts)
    sc.metrics["build_cost_micro_usd"] = cost_micro
    if cache_read:
        sc.metrics["build_cache_read_tokens"] = cache_read


async def act_get_smarter(sc: Showcase) -> None:
    """CRYS recalls and reuses the crystal it wrote in Act 2 — cold vs warm.

    Act 2 recorded that note in the background task's OWN process, so this
    process's shared fact index is stale — we invalidate it first, exactly the
    cross-process reload the web app does to see a separate agent's writes
    (FactVectorStore.invalidate: "Call after writing new facts"). Then the same
    task runs twice: COLD (no pre-turn recall — CRYS must work it out) and WARM
    (the genuine maybe_recall pre-injects the saved note). We observe whether
    recall fired, whether it surfaced CRYS's OWN note, and the cold/warm
    difference — a recall miss shows plainly rather than being papered over.
    """
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.turn_finalize import finalize_agent_turn
    from .cli import _BANK_ADDENDUM
    from .recall import maybe_recall

    customer = await sc.store.get_customer_by_id(sc.customer_id)

    # Cross-process reload: make Act 2's crystal (written in the background
    # task's process) visible to this process's shared index.
    sc.fact_vector_store.invalidate(sc.customer_id)

    def _system(agent: Any) -> str:
        return build_system_prompt(agent.customer, agent.tools) + _BANK_ADDENDUM

    def _tools_used(result: dict) -> list[str]:
        used: list[str] = []
        for c in (result.get("tool_calls") or []):
            n = c.get("tool_name")
            if n and n not in used:
                used.append(n)
        return used

    # Phrased as approaching a task that RESEMBLES prior work (not "what's in
    # file X") so the recall gate engages — the gate declines project-file
    # questions by design.
    task = (
        "I'm about to add another money calculation to Helios, similar to the "
        "tax total. What have you already worked out about computing a cart "
        "total with tax here that I can build on?"
    )

    async def _turn(agent: Any, *, recall_block: Optional[str] = None) -> tuple[dict, int]:
        result = await agent.run(
            messages=[{"role": "user", "content": task}],
            system=_system(agent) + (recall_block or ""),
        )
        fin = await finalize_agent_turn(
            store=sc.store, encoder=sc.encoder, customer=customer,
            anthropic_client=sc.client, result=result, user_query=task,
            sequence_id=result.get("id"), origin="agent",
        )
        return result, (fin.get("cost_micro_usd") or 0)

    # COLD — no pre-turn recall: CRYS has to find/derive it (or answer thin).
    r_cold, cold_cost = await _turn(await _make_agent(sc))
    cold_tools = _tools_used(r_cold)
    cold_answer = " ".join((r_cold.get("final_text") or "").split())

    # WARM — the genuine pre-turn self-recall pre-injects the saved note.
    agent_w = await _make_agent(sc)
    recall_block = await maybe_recall(
        agent=agent_w, user_input=task, fast_model=sc.models["fast"],
    )
    r_warm, warm_cost = await _turn(agent_w, recall_block=recall_block)
    warm_tools = _tools_used(r_warm)
    warm_answer = " ".join((r_warm.get("final_text") or "").split())

    recalled = bool(recall_block)
    surfaced_note = recalled and ("total_with_tax" in recall_block)
    warm_reused = "total_with_tax" in warm_answer.lower()
    cold_found = "total_with_tax" in cold_answer.lower()

    if surfaced_note:
        warm_line = "recall fired and surfaced CRYS's OWN Act-2 note (pre-injected — no search needed)"
    elif recalled:
        warm_line = "recall fired but did NOT surface the Act-2 note"
    else:
        warm_line = "recall DECLINED — nothing pre-injected"

    status = "ok" if (surfaced_note and warm_reused) else "error"
    _panel(sc, "Get smarter", status, [
        f'task: "{task[:130]}{"…" if len(task) > 130 else ""}"',
        f"COLD (recall off) — CRYS used tools {', '.join(cold_tools) or '(none)'}; "
        f"answer references total_with_tax: {'yes' if cold_found else 'no'}; "
        f"cost {cold_cost} micro-USD",
        f"WARM (recall on) — {warm_line}",
        f"   CRYS used tools {', '.join(warm_tools) or '(none)'}; reused the note: "
        f"{'yes' if warm_reused else 'no'}; cost {warm_cost} micro-USD",
        ("warm answer: " + warm_answer[:200] + ("…" if len(warm_answer) > 200 else ""))
        if warm_answer else "warm answer: (none)",
    ])

    sc.metrics["smarter_recall_fired"] = "yes" if recalled else "no"
    sc.metrics["smarter_recall_surfaced_act2_note"] = "yes" if surfaced_note else "no"
    sc.metrics["smarter_warm_reused_note"] = "yes" if warm_reused else "no"
    sc.metrics["smarter_cold_tools"] = ", ".join(cold_tools) or "none"
    sc.metrics["smarter_warm_tools"] = ", ".join(warm_tools) or "none"
    sc.metrics["smarter_cold_cost_micro_usd"] = cold_cost
    sc.metrics["smarter_warm_cost_micro_usd"] = warm_cost


# Act 4 enqueues a one-shot + a recurring series, cancels the series, then
# runs ONE bounded daemon pass that CLAIMS and executes the one-shot — the
# real claim→execute→verify→commit the --daemon loop runs, just not looping.
_ACT4_ONESHOT = (
    "Add a function currency_symbol(code) to pricing.py that returns '$' for "
    "'USD', '€' for 'EUR', '£' for 'GBP', and the code itself otherwise. Add a "
    "focused test for it to test_helios.py covering USD and an unknown code. "
    "Make the tests pass."
)
_ACT4_RECURRING = "Run the Helios test suite and report any failures."


async def act_delegate(sc: Showcase) -> None:
    """Delegate to the background queue, then let the daemon drain it.

    Drives the genuine --queue and --cancel front doors (enqueue_cli /
    cancel_cli), then runs ONE bounded daemon pass — the daemon's real
    claim → execute → finish machinery (the body of the --daemon loop) — so the
    queued one-shot is built + verified + committed UNATTENDED, not just parked.
    A gap is also planted (narrated DATA setup) to show the surface a failed run
    lands on for the daemon's one idle-pass retry.
    """
    from .daemon import _execute_task, cancel_cli, enqueue_cli

    # --- Enqueue through the real --queue front door (each prints its own line) ---
    await enqueue_cli(
        str(sc.db_path), sc.project_dir, _ACT4_ONESHOT, None, sc.customer_id,
    )
    await enqueue_cli(
        str(sc.db_path), sc.project_dir, _ACT4_RECURRING, None, sc.customer_id,
        every="daily",
    )

    tasks = await sc.store.list_agent_tasks(limit=30)
    recurring = next((t for t in tasks if t.get("recur_seconds")), None)
    oneshot = next(
        (t for t in tasks if not t.get("recur_seconds") and t["status"] == "queued"),
        None,
    )

    # --- Cancel the recurring series through the real --cancel front door ---
    cancelled_status = None
    if recurring:
        await cancel_cli(str(sc.db_path), recurring["id"])
        row = await sc.store.get_agent_task(recurring["id"])
        cancelled_status = row["status"] if row else "(gone)"

    # --- Plant a gap (narrated DATA setup) so the gap surface is visible ---
    await sc.store.create_agent_gap(
        sc.customer_id,
        task="(planted) a task whose verify failed, to show the gap surface",
        task_id="(planted)",
        branch="agent/planted-gap",
        failing_tail="AssertionError: expected 110.0, got 100.0",
        project_dir=str(sc.project_dir),
    )
    gaps = await sc.store.list_agent_gaps(
        statuses=["open", "retrying", "needs_operator"], limit=20,
    )

    # --- Drive ONE real daemon pass: claim the queued one-shot and run it
    # through the daemon's own claim → execute → finish machinery (the body of
    # the --daemon loop, bounded to a single task so the tour doesn't block on a
    # 5s poll). The one-shot is built + verified + committed UNATTENDED. ---
    print(style.dim(
        "  starting one daemon pass — it claims the queued task and runs it "
        "unattended (the full build streams to the task's own log)…"
    ))
    claimed = await sc.store.claim_next_agent_task()
    drain_status = None
    drain_report = ""
    if claimed is not None:
        drain_status, drain_report, drain_log = await _execute_task(
            claimed, str(sc.db_path),
        )
        await sc.store.finish_agent_task(
            claimed["id"], status=drain_status, report=drain_report,
            error=None if drain_status == "done" else "see report/log",
            log_path=drain_log,
        )
    drain_verified = drain_status == "done" and "PASS" in drain_report

    status = "ok" if (
        cancelled_status == "cancelled" and drain_verified
    ) else "error"
    _panel(sc, "Delegate", status, [
        f'enqueued a one-shot ("{_ACT4_ONESHOT[:46]}…") and a daily recurring '
        f'series ("{_ACT4_RECURRING[:42]}…")',
        (f"--cancel on the recurring series → status now '{cancelled_status}', "
         "future occurrences stopped"
         if recurring else "no recurring task to cancel"),
        (f"the daemon then CLAIMED the one-shot and ran it unattended → "
         f"{drain_status}; ground-truth verify "
         f"{'PASSED' if drain_verified else 'did NOT pass'}"
         if claimed is not None else "no due task for the daemon to claim"),
        f"a separate failed run is parked as a knowledge gap ({len(gaps)} open) "
        "for the daemon's one idle-pass retry",
        style.dim(
            "  the real --daemon runs this same claim→execute→verify→commit on a "
            "5s poll loop; the tour runs one bounded pass"
        ),
    ])

    sc.metrics["delegate_enqueued"] = 2
    sc.metrics["delegate_recurring_cancelled"] = (
        "yes" if cancelled_status == "cancelled" else "no"
    )
    sc.metrics["delegate_daemon_drained"] = drain_status or "nothing_claimed"
    sc.metrics["delegate_daemon_verified"] = "yes" if drain_verified else "no"
    sc.metrics["delegate_gaps_open"] = len(gaps)


# Act 5 plants a crisp same-Subject contradiction (narrated DATA setup) so the
# audit has something real to catch: two facts whose sparse keys share the
# Subject segment "token expiry" but whose claims are opposite — the kind of
# drift that accumulates when a bank ingests sources written at different times.
_ACT5_KEY_A = "Helios|auth.py|token expiry|Security"
_ACT5_KEY_B = "Helios|SECURITY.md|token expiry|Security"
_ACT5_CLAIM_A = "Helios bearer tokens expire 15 minutes after issuance and must be refreshed."
_ACT5_CLAIM_B = "Helios bearer tokens never expire; once issued they remain valid permanently."


async def act_self_heal(sc: Showcase) -> None:
    """Plant a contradiction, then let the real --audit front door surface it.

    DATA setup (narrated): two facts under one sparse-key Subject with opposite
    claims, written through the genuine fact-write path (add_pair_for_customer —
    the same primitive ingestion and crystal_write use). The MECHANISM under
    test is the operator --audit front door (run_audit → scan_for_contradictions):
    it pairs same-Subject facts, asks a Haiku CONTRADICTS/CONSISTENT
    discriminator, and writes a knowledge_conflict. Surfacing-only — nothing is
    deleted or overwritten; both facts stay in the bank, flagged for review. We
    read the surfaced conflict back through the shared store.
    """
    from .runtime import run_audit

    # --- Plant the contradictory pair (genuine fact-write path) ---
    for key, claim in ((_ACT5_KEY_A, _ACT5_CLAIM_A), (_ACT5_KEY_B, _ACT5_CLAIM_B)):
        await sc.store.add_pair_for_customer(
            customer_id=sc.customer_id,
            prompt_text=key,
            answer_text=claim,
            pair_type="entity_attribute",
            encoder=sc.encoder,
            vector_store=sc.vector_store,
            crystal_type="customer:legacy",
            source_kind="document_chunk",
        )

    # --- Run the genuine --audit front door (prints its own scan report) ---
    rc = await run_audit(
        str(sc.db_path), sc.customer_id, max_pairs=200, max_calls=50,
    )

    # --- Observe the surfaced conflict through the shared store ---
    conflicts = await sc.store.list_knowledge_conflicts(
        sc.customer_id, status="open", limit=20,
    )
    planted = next(
        (c for c in conflicts if (c.subject or "").lower() == "token expiry"),
        None,
    )

    status = "ok" if planted is not None else "error"
    lines = [
        'planted a contradiction under Subject "token expiry" (DATA setup): '
        f'"{_ACT5_CLAIM_A[:44]}…" vs "{_ACT5_CLAIM_B[:44]}…"',
        "ran the real --audit front door: the scan pairs same-Subject facts → a "
        "Haiku CONTRADICTS/CONSISTENT discriminator → writes a knowledge_conflict",
    ]
    if planted is not None:
        lines.append(
            f"CRYS surfaced the conflict [{planted.subject}]: "
            f'"{(planted.claim_a or "")[:44]}…"  ⟷  "{(planted.claim_b or "")[:44]}…"'
        )
        lines.append(style.dim(
            "  surfacing-only — both facts remain in the bank (non-destructive), "
            "flagged for a curation decision (qualify · supersede · blacklist)"
        ))
    else:
        lines.append(style.red(
            f"MISS — the audit surfaced no 'token expiry' conflict "
            f"(rc={rc}, {len(conflicts)} open conflict(s) total)"
        ))
    _panel(sc, "Self-heal", status, lines)

    sc.metrics["selfheal_conflicts_open"] = len(conflicts)
    sc.metrics["selfheal_planted_surfaced"] = "yes" if planted is not None else "no"
    sc.metrics["selfheal_audit_rc"] = rc


# Act 6 imports a tiny SHARED (general, cross-owner) pattern bank, then asks a
# question only that bank answers. The patterns are pattern-form imperative rules
# (the form the BCB finding showed transfers); the money one is what the query
# pulls. Keys MUST start with "General|" (the seed importer enforces it).
_ACT6_GENERAL_TYPE = "general:swe_patterns"
_ACT6_SEED = [
    {"key": "General|SWE Patterns|Money|Minor units",
     "claim": "Represent money as integer minor units (e.g. cents) or a Decimal type, "
              "never a binary float — float rounding silently corrupts currency math, "
              "so 0.1 + 0.2 != 0.3."},
    {"key": "General|SWE Patterns|APIs|Idempotency",
     "claim": "Make state-changing POST endpoints idempotent with a client-supplied "
              "idempotency key, so a network retry can't double-charge or double-create "
              "the resource."},
    {"key": "General|SWE Patterns|Concurrency|Optimistic locking",
     "claim": "Guard concurrent updates with an optimistic version column: read the "
              "version, write only if it is unchanged, retry on conflict — avoids the "
              "lost-update race without long-held locks."},
]
_ACT6_QUESTION = (
    "Search your saved engineering patterns: what's the recommended way to "
    "represent money in code, and why?"
)


async def act_share(sc: Showcase) -> None:
    """Merge a shared general bank, subscribe, and retrieve it — no retraining.

    The accommodation / marketplace foundation: general (cross-owner) knowledge
    is a BANK you subscribe to, not weights you retrain. We import a tiny general
    seed through the genuine --seed-general front door (run_seed_import), which
    registers the type and subscribes the customer, then reload the shared index
    (invalidate_general + invalidate — the cross-process refresh the web app does)
    and ask CRYS a question only the general bank answers. CRYS surfaces a
    'General|…' crystal it never had before — gained by merge, instantly, with no
    model retraining.
    """
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.turn_finalize import finalize_agent_turn
    from .cli import _BANK_ADDENDUM
    from .general_seed import run_seed_import

    customer = await sc.store.get_customer_by_id(sc.customer_id)

    # --- Write a tiny general seed + import it through the real front door ---
    seed_path = sc.workspace / "general_seed.jsonl"
    seed_path.write_text(
        "\n".join(json.dumps(e) for e in _ACT6_SEED) + "\n", encoding="utf-8",
    )
    rc = await run_seed_import(
        str(sc.db_path), seed_path,
        crystal_type=_ACT6_GENERAL_TYPE, customer_id=sc.customer_id,
    )

    # --- Cross-process reload: the import wrote via its OWN store; refresh the
    # shared index so the new subscription + general bank are visible here. ---
    sc.fact_vector_store.invalidate_general(_ACT6_GENERAL_TYPE)
    sc.fact_vector_store.invalidate(sc.customer_id)

    # --- Ask a question only the general bank answers, through the agent ---
    agent = await _make_agent(sc)
    result = await agent.run(
        messages=[{"role": "user", "content": _ACT6_QUESTION}],
        system=build_system_prompt(agent.customer, agent.tools) + _BANK_ADDENDUM,
    )
    await finalize_agent_turn(
        store=sc.store, encoder=sc.encoder, customer=customer,
        anthropic_client=sc.client, result=result,
        user_query=_ACT6_QUESTION, sequence_id=result.get("id"), origin="agent",
    )
    answer = " ".join((result.get("final_text") or "").split())

    # --- Observe: did a cross-owner General| crystal surface + get used? ---
    general_keys: list[str] = []
    for cid in _surfaced_crystal_ids(result):
        key = await _crystal_key(sc, cid)
        if key.startswith("General|"):
            general_keys.append(key)
    used_general = any(
        w in answer.lower() for w in ("minor unit", "decimal", "float", "cents")
    )
    subs = await sc.store.get_customer_general_types(sc.customer_id)

    status = "ok" if (rc == 0 and general_keys) else "error"
    lines = [
        f"imported {len(_ACT6_SEED)} shared patterns into general bank "
        f"'{_ACT6_GENERAL_TYPE}' and subscribed the customer "
        f"(subscriptions: {', '.join(subs) or 'none'})",
        f'asked: "{_ACT6_QUESTION}"',
    ]
    if general_keys:
        lines.append(
            "CRYS retrieved cross-owner knowledge it never had before: "
            + "; ".join(_short_key(k) for k in general_keys[:3])
        )
        lines.append(
            ("answer: " + answer[:200] + ("…" if len(answer) > 200 else ""))
            if answer else "answer: (none)"
        )
        lines.append(style.dim(
            "  gained by bank merge + subscription — instantly, no model retraining "
            "(imperative patterns transfer where raw code examples don't — the BCB finding)"
        ))
    else:
        lines.append(style.red(
            f"MISS — no 'General|' crystal surfaced (import rc={rc}); the "
            "merge/subscribe/reload didn't make the shared bank retrievable here"
        ))
    _panel(sc, "Share", status, lines)

    sc.metrics["share_general_type"] = _ACT6_GENERAL_TYPE
    sc.metrics["share_patterns_imported"] = len(_ACT6_SEED)
    sc.metrics["share_subscribed"] = "yes" if _ACT6_GENERAL_TYPE in subs else "no"
    sc.metrics["share_general_crystal_surfaced"] = "yes" if general_keys else "no"
    sc.metrics["share_reused_in_answer"] = "yes" if used_general else "no"


# Act 7 governs two rails. Part A (F2): a crystal is an OWNED resource — an
# operator-private note (mode 0o600) is readable by its owner-operator but
# denied to a same-team coworker, enforced at the retrieval primitive itself.
# Part B (G4): the marketplace economy is metered on GROUNDED citations — a
# different team grounding an expert's general crystal mints the expert a
# shard; a team's own crystals and self-traffic earn nothing (anti-gaming).
_ACT7_PRIVATE_PROMPT = "Alice's private staging-access runbook"
_ACT7_PRIVATE_ANSWER = (
    "Personal note — reach the Helios staging environment through the "
    "bastion-3.helios.internal jump host, then rotate the read-replica "
    "password every Monday. Keep within my own access; do not share."
)
_ACT7_PRIVATE_QUERY = "How do I get into the Helios staging environment?"

_ACT7_EXPERT_TYPE = "general:reliability"
_ACT7_EXPERT_SEED = [{
    "key": "General|Reliability|Deploys|Canary gate",
    "claim": "Gate every production deploy behind a canary: shift 5% of "
             "traffic to the new build for 10 minutes and auto-rollback if "
             "the error rate exceeds 0.5% before promoting to 100%.",
}]
_ACT7_EXPERT_QUESTION = (
    "Search your saved engineering patterns: what's the recommended way to "
    "safely roll out a production deploy?"
)


async def act_govern(sc: Showcase) -> None:
    """Govern: F2 permission-checked retrieval + the G4 shard ledger.

    Two governance rails. (A) A crystal is an owned resource: an operator-
    private note is readable by its owner-operator but DENIED to a same-team
    coworker — enforced at the retrieval primitive (FactVectorStore.search
    with an operator context → permissions.can_read on POSIX mode bits), not by
    trusting the caller. (B) The marketplace economy is metered on GROUNDED
    citations: when a different team grounds an expert's general crystal, a
    shard credit mints to the expert — and self-traffic plus a team's own
    private/team crystals earn nothing, the anti-gaming rule.
    """
    import numpy as np
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.turn_finalize import finalize_agent_turn
    from .cli import _BANK_ADDENDUM

    lines: list[str] = []

    # ============ Part A — F2 permission-checked retrieval ============
    # Two real operators under the SAME team (the showcase customer).
    alice, _ = await sc.store.create_operator(
        team_id=sc.customer_id, display_name="Alice (owner)", role="operator",
    )
    bob, _ = await sc.store.create_operator(
        team_id=sc.customer_id, display_name="Bob (coworker)", role="operator",
    )
    # Alice authors an operator-private crystal (mode 0o600 — owner rw, group
    # ---, other ---). The owner + mode ride the write primitive (F2.2).
    priv_crystal, _ = await sc.store.add_pair_for_customer(
        customer_id=sc.customer_id,
        prompt_text=_ACT7_PRIVATE_PROMPT,
        answer_text=_ACT7_PRIVATE_ANSWER,
        pair_type="question_answer",
        encoder=sc.encoder,
        vector_store=sc.vector_store,
        owner_operator_id=alice.id,
        group_team_id=sc.customer_id,
        mode=0o600,
    )
    sc.fact_vector_store.invalidate(sc.customer_id)

    # Same query, two operators on one team. The candidate set is identical;
    # the only difference is who is asking.
    q = np.asarray(
        sc.encoder.encode_native(_ACT7_PRIVATE_QUERY), dtype=np.float32,
    )
    owner_hits = await sc.fact_vector_store.search(
        sc.customer_id, q, k=10, operator=alice,
    )
    coworker_hits = await sc.fact_vector_store.search(
        sc.customer_id, q, k=10, operator=bob,
    )
    owner_sees = any(cid == priv_crystal.id for _, cid, _, _ in owner_hits)
    coworker_sees = any(cid == priv_crystal.id for _, cid, _, _ in coworker_hits)
    f2_ok = owner_sees and not coworker_sees and priv_crystal.mode == 0o600

    lines.append(
        f"F2 — Alice authored an operator-private note (mode 0{priv_crystal.mode:o}, "
        "owner-only). Same query, two operators on the same team:"
    )
    lines.append(
        f"  owner Alice: {'sees it ✓' if owner_sees else 'CANNOT see it ✗'}    "
        f"coworker Bob: {'DENIED ✓' if not coworker_sees else 'sees it ✗ (leak)'}"
    )
    lines.append(style.dim(
        "  enforced at the retrieval primitive (FactVectorStore.search → "
        "permissions.can_read on POSIX mode bits), not by trusting the caller"
    ))

    # ============ Part B — G4 grounded-citation shard ledger ============
    # A separate expert org authors a general (marketplace) crystal.
    expert_team = await sc.store.create_customer(
        provider="anthropic", model_id="expert", api_key_ref="local",
    )
    dana, _ = await sc.store.create_operator(
        team_id=expert_team.id, display_name="Dana (reliability expert)",
        role="operator",
    )
    await sc.store.import_general_bank(
        crystal_type=_ACT7_EXPERT_TYPE, entries=_ACT7_EXPERT_SEED,
        encoder=sc.encoder,
    )
    # Stamp the general crystal with the expert's ownership (import leaves
    # general crystals owner-less; F3 promotion is where this normally lands).
    expert_cid: Optional[str] = None
    expert_gcs = await sc.store.list_general_crystals(_ACT7_EXPERT_TYPE)
    if expert_gcs:
        gc = expert_gcs[0]
        gc.owner_operator_id = dana.id
        gc.group_team_id = expert_team.id
        await sc.store.upsert_crystal(gc)
        expert_cid = gc.id
    # Subscribe the SHOWCASE customer (a different team) to the expert bank.
    subs = await sc.store.get_customer_general_types(sc.customer_id)
    if _ACT7_EXPERT_TYPE not in subs:
        await sc.store.set_customer_general_types(
            sc.customer_id, [*subs, _ACT7_EXPERT_TYPE],
        )
    sc.fact_vector_store.invalidate_general(_ACT7_EXPERT_TYPE)
    sc.fact_vector_store.invalidate(sc.customer_id)

    balance_before = await sc.store.shard_balance(dana.id)

    # The showcase customer grounds the expert's crystal through a genuine
    # agent turn — finalize_agent_turn's citation rail mints the credit.
    customer = await sc.store.get_customer_by_id(sc.customer_id)
    agent = await _make_agent(sc)
    result = await agent.run(
        messages=[{"role": "user", "content": _ACT7_EXPERT_QUESTION}],
        system=build_system_prompt(agent.customer, agent.tools) + _BANK_ADDENDUM,
    )
    await finalize_agent_turn(
        store=sc.store, encoder=sc.encoder, customer=customer,
        anthropic_client=sc.client, result=result,
        user_query=_ACT7_EXPERT_QUESTION, sequence_id=result.get("id"),
        origin="agent",
    )
    balance_after = await sc.store.shard_balance(dana.id)
    minted = balance_after - balance_before
    cited_expert = any(
        cid == expert_cid for cid in _surfaced_crystal_ids(result)
    )

    lines.append("")
    lines.append(
        "G4 — Dana (a different org) authored a general pattern; the showcase "
        "team grounded it in an answer."
    )
    if minted > 0:
        lines.append(
            f"  cross-owner grounded citation → Dana earned {minted} shard"
            f"{'s' if minted != 1 else ''} (balance {balance_before} → {balance_after})"
        )
        lines.append(style.dim(
            "  metered on GROUNDING, not mere injection; a team's own crystals "
            "and self-traffic earn nothing — Act 1 cited the team's OWN crystals → 0"
        ))
    else:
        lines.append(style.red(
            f"  MISS — no shard minted (cited_expert={cited_expert}, balance "
            f"{balance_before}→{balance_after}); the grounded-citation credit "
            "rail didn't fire this run"
        ))

    status = "ok" if (f2_ok and minted > 0) else "error"
    _panel(sc, "Govern", status, lines)

    sc.metrics["govern_f2_owner_sees"] = "yes" if owner_sees else "no"
    sc.metrics["govern_f2_coworker_denied"] = "yes" if not coworker_sees else "no"
    sc.metrics["govern_shards_minted"] = minted
    sc.metrics["govern_expert_balance"] = balance_after
    sc.metrics["govern_cross_owner_cited"] = "yes" if cited_expert else "no"


# Act 8 researches a synthesis the bank has the pieces for, validated end to end.
_ACT_RESEARCH_GOAL = (
    "Write a concise onboarding brief on Helios's request lifecycle for a new "
    "engineer: how a request is authorized, how a cart is validated, and how an "
    "order total is computed — and name the module each rule lives in."
)


async def act_research(sc: Showcase) -> None:
    """Research a synthesis through the validated cognition loop.

    Drives run_cognition_workflow — the engine behind the agent's cognition_run
    tool. An orchestrator derives acceptance criteria from the goal; workers
    gather from the bank and synthesize (they never see the goal, so they can't
    pander to it — the information barrier that stops validation poisoning); a
    validator scores the deliverable against the criteria and sends it back up
    to two times. CRYS doesn't just retrieve a fact — it produces a NEW,
    validator-checked synthesis the bank's atomic facts didn't directly hold.

    Honest scope: research here is over the bank + the model's synthesis under
    validator review. The web_search worker is a v2 placeholder, so this does
    NOT reach the open web — it is validated synthesis of what CRYS already
    knows, not a live external lookup.
    """
    from crystal_cache.cognition.engine import run_cognition_workflow

    result = await run_cognition_workflow(
        goal=_ACT_RESEARCH_GOAL,
        customer_id=sc.customer_id,
        store=sc.store,
        fact_store=sc.fact_vector_store,
        encoder=sc.encoder,
        conversation_context="",
        source_crystal_id="",
        output_type="report",
        trigger_type="agent",
        trigger_id="",
        max_attempts=2,
    )
    text = " ".join((getattr(result, "text", None) or "").split())
    confidence = float(getattr(result, "confidence", 0.0) or 0.0)
    success = bool(getattr(result, "success", False))
    reason = getattr(result, "reason", None)
    cost_usd = float(getattr(result, "cost_usd", 0.0) or 0.0)

    status = "ok" if (success and text) else "error"
    lines = [
        f'goal: "{_ACT_RESEARCH_GOAL}"',
        "flow: orchestrator derives acceptance criteria → workers gather + "
        "synthesize (blind to the goal) → validator scores the deliverable",
    ]
    if success and text:
        lines.append(
            f"validator: APPROVED (confidence {confidence:.2f}) after the "
            "orchestrate → work → validate loop"
        )
        lines.append("brief: " + text[:240] + ("…" if len(text) > 240 else ""))
        lines.append(style.dim(
            f"  ~${cost_usd:.3f} for the multi-step loop — validated synthesis of "
            "what the bank knows (no open-web lookup; web_search is a v2 stub)"
        ))
    else:
        lines.append(style.red(
            f"  validation did not pass (confidence {confidence:.2f}"
            + (f"; {reason}" if reason else "")
            + ") — the loop couldn't satisfy the goal from the bank in the "
            "retry budget"
        ))
    _panel(sc, "Research", status, lines)

    sc.metrics["research_validated"] = "yes" if success else "no"
    sc.metrics["research_confidence"] = f"{confidence:.2f}"
    sc.metrics["research_cost_usd"] = f"{cost_usd:.3f}"


# ---------------------------------------------------------------------------
# The act registry + the dashboard
# ---------------------------------------------------------------------------

_ACTS: list[tuple[str, Callable[[Showcase], Awaitable[None]]]] = [
    ("0  Seed",        act_seed),
    ("1  Ask",         act_ask),
    ("2  Build",       act_build),
    ("3  Get smarter", act_get_smarter),
    ("4  Delegate",    act_delegate),
    ("5  Self-heal",   act_self_heal),
    ("6  Share",       act_share),
    ("7  Govern",      act_govern),
    ("8  Research",    act_research),
]


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(s: str) -> str:
    """Strip ANSI color codes for the saved report."""
    return _ANSI_RE.sub("", s)


def _render_report(sc: Showcase) -> Path:
    lines = ["# CRYS Showcase report", "",
             f"_generated {datetime.now(timezone.utc).isoformat(timespec='seconds')}_", "",
             f"- workspace: `{sc.workspace}`",
             f"- customer: `{sc.customer_id}`", ""]
    for name, status, panel_lines in sc.panels:
        mark = {"ok": "✓", "todo": "…", "error": "✗"}.get(status, "•")
        lines.append(f"## {mark} {name}")
        for pl in panel_lines:
            lines.append(f"- {_plain(pl)}")
        lines.append("")
    if sc.metrics:
        lines.append("## Metrics")
        for k, v in sc.metrics.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    report = sc.workspace / "showcase_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    return report


# Self-contained styling for the saved .html report (no external assets, so the
# file opens anywhere). Brand palette: indigo→cyan.
_REPORT_CSS = """
:root{--indigo:#4f46e5;--cyan:#06b6d4;--ink:#0f172a;--muted:#64748b;--bg:#f8fafc;
--card:#fff;--line:#e2e8f0;--ok:#16a34a;--okbg:#f0fdf4;--todo:#94a3b8;
--todobg:#f1f5f9;--err:#dc2626;--errbg:#fef2f2;}
*{box-sizing:border-box;}
body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;}
.wrap{max-width:880px;margin:0 auto;padding:0 20px 56px;}
header{background:linear-gradient(120deg,var(--indigo),var(--cyan));color:#fff;
padding:38px 0 32px;margin-bottom:26px;}
header .wrap{padding-bottom:0;}
header h1{margin:0 0 6px;font-size:27px;letter-spacing:-.02em;}
header .sub{opacity:.92;font-size:14px;}
header .meta{margin-top:16px;font-size:12px;opacity:.85;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;word-break:break-all;}
.summary{display:flex;gap:10px;margin:0 0 24px;flex-wrap:wrap;}
.chip{border:1px solid var(--line);background:var(--card);border-radius:10px;
padding:10px 16px;font-size:13px;display:flex;align-items:baseline;gap:8px;}
.chip b{font-size:20px;}
.chip.ok b{color:var(--ok);}.chip.err b{color:var(--err);}.chip.todo b{color:var(--todo);}
.act{background:var(--card);border:1px solid var(--line);border-radius:12px;
padding:15px 18px;margin-bottom:13px;}
.act.ok{border-left:3px solid var(--ok);}
.act.todo{border-left:3px solid var(--todo);}
.act.err{border-left:3px solid var(--err);}
.act h2{margin:0 0 10px;font-size:16px;display:flex;align-items:center;gap:10px;}
.badge{font-size:11px;font-weight:600;padding:3px 9px;border-radius:999px;
text-transform:uppercase;letter-spacing:.04em;}
.badge.ok{color:var(--ok);background:var(--okbg);}
.badge.todo{color:var(--todo);background:var(--todobg);}
.badge.err{color:var(--err);background:var(--errbg);}
.act ul{margin:0;padding-left:18px;}
.act li{margin:4px 0;color:#1e293b;}
.act li.dim{color:var(--muted);font-size:13px;list-style:none;margin-left:-18px;}
.act li.err{color:var(--err);}
h3.sec{font-size:12px;text-transform:uppercase;letter-spacing:.07em;
color:var(--muted);margin:30px 0 12px;}
table{width:100%;border-collapse:collapse;background:var(--card);
border:1px solid var(--line);border-radius:12px;overflow:hidden;}
td{padding:9px 14px;border-bottom:1px solid var(--line);font-size:13px;}
tr:last-child td{border-bottom:none;}
td.k{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
color:var(--muted);width:46%;}
footer{color:var(--muted);font-size:12px;text-align:center;margin-top:34px;}
"""


def _render_html_report(sc: Showcase) -> Path:
    """Write a self-contained, browser-openable HTML report of the run.

    Mirrors the .md report's data (per-act status + panel lines + metrics) but
    styled for sharing. Reuses the dim/error ANSI markers the panels already
    carry to tint note lines and misses, so a failure reads the same on the
    page as it did in the terminal — nothing is smoothed over.
    """
    import html as _html

    def esc(s: Any) -> str:
        return _html.escape(_plain(str(s)))

    ok = sum(1 for _, s, _ in sc.panels if s == "ok")
    todo = sum(1 for _, s, _ in sc.panels if s == "todo")
    failed = sum(1 for _, s, _ in sc.panels if s == "error")
    gen = datetime.now(timezone.utc).isoformat(timespec="seconds")

    out: list[str] = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>CRYS Showcase report</title><style>", _REPORT_CSS,
        "</style></head><body>",
        "<header><div class='wrap'><h1>CRYS Showcase</h1>",
        "<div class='sub'>One command, the whole surface — where knowledge "
        "takes shape.</div>",
        f"<div class='meta'>generated {esc(gen)}  ·  workspace {esc(sc.workspace)}"
        f"  ·  customer {esc(sc.customer_id)}</div>",
        "</div></header><div class='wrap'>",
        "<div class='summary'>",
        f"<div class='chip ok'><b>{ok}</b> passed</div>",
        f"<div class='chip todo'><b>{todo}</b> pending</div>",
        f"<div class='chip err'><b>{failed}</b> failed</div>",
        "</div>",
    ]
    for name, status, panel_lines in sc.panels:
        cls = {"ok": "ok", "todo": "todo", "error": "err"}.get(status, "todo")
        label = {"ok": "pass", "todo": "pending", "error": "fail"}.get(status, "—")
        out.append(
            f"<div class='act {cls}'><h2><span class='badge {cls}'>{label}</span>"
            f"{esc(name)}</h2><ul>"
        )
        for pl in panel_lines:
            text = esc(pl)
            if not text.strip():
                continue
            li_cls = ""
            if "\x1b[31m" in pl:      # style.red — a miss
                li_cls = " class='err'"
            elif "\x1b[2m" in pl:     # style.dim — a note
                li_cls = " class='dim'"
            out.append(f"<li{li_cls}>{text}</li>")
        out.append("</ul></div>")
    if sc.metrics:
        out.append("<h3 class='sec'>Metrics</h3><table>")
        for k, v in sc.metrics.items():
            out.append(f"<tr><td class='k'>{esc(k)}</td><td>{esc(v)}</td></tr>")
        out.append("</table>")
    out.append("<footer>CRYS · Crystal Cache — generated by the showcase</footer>")
    out.append("</div></body></html>")

    report = sc.workspace / "showcase_report.html"
    report.write_text("".join(out), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def _run(acts: Optional[list[int]], keep: bool) -> int:
    ts = time.strftime("%Y%m%d-%H%M%S")
    workspace = config_store.CONFIG_DIR / "showcase" / ts
    workspace.mkdir(parents=True, exist_ok=True)

    _banner("CRYS Showcase — one command, the whole surface")
    print(f"  workspace : {workspace}")
    print(f"  acts      : {acts if acts is not None else 'all'}")

    sc = await _bootstrap(workspace)

    selected = _ACTS if acts is None else [a for i, a in enumerate(_ACTS) if i in acts]
    for name, fn in selected:
        _banner(name)
        try:
            await fn(sc)
        except Exception as e:  # noqa: BLE001 — the tour continues; the report flags it
            _panel(sc, name, "error", [style.red(f"act failed: {type(e).__name__}: {e}")])

    _banner("Dashboard")
    # At-a-glance outcome of the whole tour: one line per panel, then a tally.
    ok = sum(1 for _, s, _ in sc.panels if s == "ok")
    todo = sum(1 for _, s, _ in sc.panels if s == "todo")
    failed = sum(1 for _, s, _ in sc.panels if s == "error")
    for name, status, _ in sc.panels:
        mark = {
            "ok": style.green("✓"), "todo": style.dim("…"), "error": style.red("✗"),
        }.get(status, "•")
        print(f"  {mark} {name}")
    print(
        "\n  " + style.bold(f"{ok} passed · {todo} pending · {failed} failed")
        + style.dim(f"   ({len(sc.panels)} panels)")
    )
    if sc.metrics:
        print(style.dim("\n  metrics"))
        for k, v in sc.metrics.items():
            print(f"    {k:<30} {v}")
    md_report = _render_report(sc)
    html_report = _render_html_report(sc)
    print(f"\n  report saved: {md_report}")
    print(f"  report saved: {html_report}")
    if sc.store is not None:
        await sc.store.dispose()
    if not keep:
        print(style.dim(f"  (workspace kept at {workspace}; --showcase-keep is the default for now)"))
    return 0


def run_showcase(acts: Optional[list[int]] = None, keep: bool = True) -> int:
    """Synchronous entry for the --showcase CLI flag."""
    return asyncio.run(_run(acts, keep))
