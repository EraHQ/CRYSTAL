"""Cognition retrieval adapter — routes cognition's tool-style worker
steps onto the shared agent tool registry (B / §6.5.5 unification).

Cognition's worker plans emit tool-style steps (crystal_search,
crystal_key_scan, web_search). Their `step.input` shape and the output
shape cognition's analyze/synthesize steps consume both differ from the
agent's registry tools. This module is the single seam that reconciles
them, so there is ONE retrieval implementation and one place that knows
the shape difference:

  * INPUT  — maps step.input to the right shared tool(s) + kwargs.
  * SEMANTICS — crystal_search fans out across content_search
    (content_chunk) and knowledge_search (entity/qa/relationship) and
    filters the merged facts to the plan's requested pair_types,
    because no single agent tool covers content_chunk + question_answer
    the way cognition's v1 crystal_search did.
  * OUTPUT — hydrates per-fact `findings` (with content) from the
    tools' matched_fact_ids via the store, and assembles the
    `content_text` / `results_count` keys cognition reads.
    crystal_key_scan dispatches to the `key_scan` tool and web_search
    to the `web_search` tool — both already return cognition's shape,
    so they pass through with light normalization.

The routers/tools are untouched: they keep returning their
injection-oriented contract for the agent + composer + chat_proxy.
This adapter expands that contract into the cognition view.

If the shared registry can't be loaded (cognition-in-isolation tests
with the agent package absent), `dispatch_cognition_retrieval` raises
`RegistryUnavailable` and the caller falls back to the v1 worker
helpers in roles.py.
"""
from __future__ import annotations

from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# Pair-type coverage of the two shared vector tools. Mirrors
# ContentRouter.PAIR_TYPES / KnowledgeRouter.PAIR_TYPES in
# retrieval/v3_routers.py.
_CONTENT_PAIR_TYPES = frozenset({"content_chunk"})
_KNOWLEDGE_PAIR_TYPES = frozenset(
    {"entity_attribute", "question_answer", "entity_relationship"}
)

# v1 crystal_search default (see _worker_crystal_search in roles.py):
# content chunks plus Q&A pairs.
DEFAULT_SEARCH_PAIR_TYPES = ["content_chunk", "question_answer"]


class RegistryUnavailable(Exception):
    """Raised when the shared tool registry can't be loaded or doesn't
    cover the requested action, so the caller falls back to the v1
    worker helpers (cognition-in-isolation tests)."""


# ---------------------------------------------------------------------------
# Pure helpers (no registry, no DB) — unit-testable
# ---------------------------------------------------------------------------

def _tools_for_pair_types(pair_types: list[str]) -> set[str]:
    """Which shared vector tools cover the requested pair_types.

    content_search covers content_chunk; knowledge_search covers
    entity_attribute / question_answer / entity_relationship. Returns
    the subset of tool names whose coverage intersects the request.
    """
    requested = set(pair_types)
    tools: set[str] = set()
    if _CONTENT_PAIR_TYPES & requested:
        tools.add("content_search")
    if _KNOWLEDGE_PAIR_TYPES & requested:
        tools.add("knowledge_search")
    return tools


# Bank relevance floor (2026-07-11, ratified: "the bank should never be
# prefiring... the gate should be extremely high"). Rematch #4 evidence:
# a video-infrastructure research task's crystal_search returned 20
# Python-tutorial facts (nearest neighbors in an unrelated bank), which
# (a) flooded ~10K chars of noise into the composition context, starving
# the real web findings out of the truncation window, and (b) counted as
# "grounding" for the C2 answerability gate, so the park built for
# exactly this could never fire. The gate is ALL-OR-NOTHING per search
# on the tools' top_score: if even the BEST match doesn't clear the
# floor, the bank has nothing on this topic and the search contributes
# NOTHING (findings=[], zero grounding — C2 semantics restored). A tool
# output without a numeric top_score (legacy fakes) is not gated.
# Per-fact score gating needs scores threaded through the tool contract
# — that rides the STR/Phase-A work (CCA plan); top_score is the
# contract-change-free gate available today.
COGNITION_BANK_RELEVANCE_FLOOR = 0.60


def _filter_and_cap_findings(
    findings: list[dict], target_pair_types: list[str], k: int
) -> list[dict]:
    """Filter merged findings to the requested pair_types, dedup by
    fact_id, and cap at k.

    A finding is kept only if its pair_type is in the requested set —
    this is what makes a request for pair_types=["question_answer"]
    drop the entity/relationship facts knowledge_search also returns,
    faithfully reproducing v1's fact_store.search(pair_types=...).
    """
    target = set(target_pair_types)
    out: list[dict] = []
    seen: set[str] = set()
    for f in findings:
        fid = f.get("fact_id")
        if fid in seen:
            continue
        if target and f.get("pair_type") not in target:
            continue
        seen.add(fid)
        out.append(f)
        if len(out) >= k:
            break
    return out


def _content_text_from_findings(findings: list[dict]) -> str:
    """Assemble the analyze/synthesize input text from findings."""
    return "\n\n".join(f.get("content", "") for f in findings if f.get("content"))


def _normalize_web_output(out: dict) -> dict:
    """Add the findings/content_text/results_count keys cognition's
    downstream + C2's answerability gate expect to the web_search
    stub's {note, query, results} shape."""
    results = out.get("results", []) or []
    normalized = dict(out)
    normalized.setdefault("findings", results)
    normalized.setdefault("results_count", len(results))
    normalized.setdefault("content_text", "")
    return normalized


def _normalize_source_output(out: dict) -> dict:
    """Add a content_text summary to a source_lookup result.

    cognition's composition steps read content_text / content / findings.
    A source read already exposes `content`, but search (matches) and list
    (entries) would otherwise never reach the analyze/synthesize steps.
    This derives a content_text summary so every source op feeds
    composition. The structured keys (content/matches/entries/path) are
    left intact for the C2 grounding count and the C3 corpus.
    """
    if not isinstance(out, dict) or out.get("content_text"):
        return out
    op = out.get("op", "")
    lines: list[str] = []
    if op == "read" and out.get("content"):
        lines.append(f"Source file {out.get('path', '')}:")
        lines.append(str(out.get("content", "")))
    elif op == "search":
        ms = out.get("matches", []) or []
        lines.append(f"Source search for {out.get('query', '')!r} — {len(ms)} match(es):")
        lines += [
            f"- {m.get('path', '')}:{m.get('line', '')}: {m.get('text', '')}"
            for m in ms if isinstance(m, dict)
        ]
    elif op == "list":
        es = out.get("entries", []) or []
        lines.append(f"Source listing {out.get('path', '')} — {len(es)} entr(ies):")
        lines += [
            f"- {e.get('name', '')} ({e.get('type', '')})"
            for e in es if isinstance(e, dict)
        ]
    if out.get("error"):
        lines.append(f"(source_lookup note: {out['error']})")
    enriched = dict(out)
    enriched["content_text"] = "\n".join(lines)
    return enriched


# ---------------------------------------------------------------------------
# Registry + DB shell (impure)
# ---------------------------------------------------------------------------

def _load_registry(store: Any, fact_store: Any, encoder: Any):
    """Load the shared registry and inject tool state. Raises
    RegistryUnavailable if the shared surface can't be loaded (agent
    package not importable, a raising get_registry/import_all_tools, or
    an empty registry) so the caller degrades to the v1 helpers (AN-13).
    """
    try:
        from ..agent.tool_registry import get_registry, import_all_tools
        from ..agent.tools.retrievers import set_tool_state
        from ..infrastructure.vector_index import InMemoryVectorIndex
        # Tool state lets the registered tools reach store/fact_store/
        # encoder. vector_store/decomposer are agent-loop-only
        # concerns; set to None so a misrouted cognition step fails fast
        # rather than silently misbehaving.
        set_tool_state({
            "store": store,
            "fact_vector_store": fact_store,
            "vector_index": InMemoryVectorIndex(
                fact_store=fact_store, vector_store=None, metadata_store=store
            ),
            "encoder": encoder,
            "vector_store": None,
            "decomposer": None,
        })
        import_all_tools()
        registry = get_registry()
    except Exception as e:
        # Any failure loading the shared surface (import error, a
        # raising get_registry/import_all_tools, etc.) means the
        # registry is unavailable. This is the AN-13 graceful-
        # degradation path: the caller falls back to the v1 worker
        # helpers rather than failing the step.
        raise RegistryUnavailable(f"registry load failed: {e}")
    if len(registry) == 0:
        raise RegistryUnavailable("registry empty after import_all_tools")
    return registry


async def _hydrate_findings(
    store: Any, fact_ids: list[str], crystal_ids: list[str]
) -> list[dict]:
    """Expand the tools' matched_fact_ids into per-fact findings.

    The tools return matched_fact_ids (score-ordered) and
    matched_crystal_ids (a deduped set) separately, not paired. So we
    fetch each matched crystal once, index its facts by id, then map
    the fact_ids back to findings preserving the tools' order. Each
    finding carries pair_type so the caller can filter by it.
    """
    index: dict[str, tuple] = {}
    for cid in dict.fromkeys(crystal_ids):  # dedup, preserve order
        try:
            for f in await store.list_facts_for_crystal(cid):
                index[f.id] = (f, cid)
        except Exception:
            continue

    findings: list[dict] = []
    seen: set[str] = set()
    for fid in fact_ids:
        if fid in seen or fid not in index:
            continue
        seen.add(fid)
        f, cid = index[fid]
        content = f.claim_text or f.answer_value or ""
        findings.append({
            "fact_id": f.id,
            "crystal_id": cid,
            "key": f.prompt_text or "",
            "content": content[:1500],
            "pair_type": f.pair_type or "",
        })
    return findings


async def _do_crystal_search(
    registry: Any, store: Any, customer_id: str, step_input: dict
) -> dict:
    """crystal_search → content_search (+) knowledge_search, hydrate,
    filter by pair_types, merge."""
    query = step_input.get("query", "")
    k = step_input.get("k", 10)
    pair_types = step_input.get("pair_types") or list(DEFAULT_SEARCH_PAIR_TYPES)

    if not query:
        return {
            "query": query, "results_count": 0, "findings": [],
            "content_text": "", "matched_fact_ids": [],
            "matched_crystal_ids": [], "fact_count": 0,
            "error": "No query provided for crystal_search",
        }

    # If the requested pair_types don't map to either tool (an exotic
    # type), search both rather than returning nothing — the pair_type
    # filter below still constrains the result.
    tool_names = _tools_for_pair_types(pair_types) or {"content_search", "knowledge_search"}

    fact_ids: list[str] = []
    crystal_ids: list[str] = []
    top_scores: list[float] = []
    for name in sorted(tool_names):  # deterministic order
        tool = registry.get(name)
        if tool is None:
            continue
        out = await tool.impl(customer_id=customer_id, query=query, k=k)
        fact_ids.extend(out.get("matched_fact_ids", []) or [])
        crystal_ids.extend(out.get("matched_crystal_ids", []) or [])
        ts = out.get("top_score")
        if isinstance(ts, (int, float)):
            top_scores.append(float(ts))

    # Bank relevance gate (see COGNITION_BANK_RELEVANCE_FLOOR above):
    # when every tool reported a top_score and the best of them is under
    # the floor, the bank has no material on this topic — return an
    # explicitly empty result (zero C2 grounding) instead of the k
    # nearest unrelated neighbors. Hydration is skipped entirely.
    if top_scores and max(top_scores) < COGNITION_BANK_RELEVANCE_FLOOR:
        return {
            "query": query,
            "pair_types": pair_types,
            "results_count": 0,
            "findings": [],
            "content_text": "",
            "matched_fact_ids": [],
            "matched_crystal_ids": [],
            "fact_count": 0,
            "gated_top_score": round(max(top_scores), 4),
            "note": (
                "bank relevance gate: best match scored "
                f"{max(top_scores):.2f} < {COGNITION_BANK_RELEVANCE_FLOOR} "
                "floor; the bank has no material on this topic"
            ),
        }

    raw_findings = await _hydrate_findings(store, fact_ids, crystal_ids)
    findings = _filter_and_cap_findings(raw_findings, pair_types, k)

    return {
        "query": query,
        "pair_types": pair_types,
        "results_count": len(findings),
        "findings": findings,
        "content_text": _content_text_from_findings(findings),
        "matched_fact_ids": [f["fact_id"] for f in findings],
        "matched_crystal_ids": list({f["crystal_id"] for f in findings}),
        "fact_count": len(findings),
    }


async def dispatch_cognition_retrieval(
    *,
    action_value: str,
    step_input: dict,
    customer_id: str,
    store: Any,
    fact_store: Any,
    encoder: Any,
) -> dict:
    """Dispatch a cognition tool-style step onto the shared registry and
    return cognition's findings shape.

    Raises RegistryUnavailable when the registry can't serve the action,
    so roles._dispatch_tool_via_registry can fall back to the v1 helpers.
    """
    registry = _load_registry(store, fact_store, encoder)

    if action_value == "crystal_search":
        return await _do_crystal_search(registry, store, customer_id, step_input)

    if action_value == "crystal_key_scan":
        tool = registry.get_by_cognition_action("crystal_key_scan")
        if tool is None or "cognition" not in tool.contexts:
            raise RegistryUnavailable("no cognition tool for crystal_key_scan")
        return await tool.impl(
            customer_id=customer_id,
            key_prefix=step_input.get("key_prefix", ""),
            subject_contains=step_input.get("subject_contains", ""),
        )

    if action_value == "web_search":
        tool = registry.get_by_cognition_action("web_search")
        if tool is None or "cognition" not in tool.contexts:
            raise RegistryUnavailable("no cognition tool for web_search")
        out = await tool.impl(customer_id=customer_id, query=step_input.get("query", ""))
        return _normalize_web_output(out)

    if action_value == "source_lookup":
        tool = registry.get_by_cognition_action("source_lookup")
        if tool is None or "cognition" not in tool.contexts:
            raise RegistryUnavailable("no cognition tool for source_lookup")
        # Passthrough: the tool's params (op/path/query/path_prefix) are
        # already the cognition plan's input shape. Normalize so search/
        # list results (not just read) reach the composition steps.
        out = await tool.impl(
            customer_id=customer_id,
            op=step_input.get("op", "search"),
            path=step_input.get("path", ""),
            query=step_input.get("query", ""),
            path_prefix=step_input.get("path_prefix", ""),
        )
        return _normalize_source_output(out)

    # Any other tool-style action has no adapter mapping — let the
    # caller fall back to the v1 helper path.
    raise RegistryUnavailable(f"unhandled cognition action {action_value!r}")
