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

import asyncio

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

# Instruction-prose words that poison a search-engine query. Rematch #5:
# the orchestrator wrote worker instructions into input.query ("Extract
# WhisperX release data: latest stable version, recent releases (last 6
# months), changelog, and commit activity from GitHub API endpoints.")
# — 25 words of prose into SearXNG returns zero results every time. The
# prompt now tells it not to (keyword-query rule); this is the code
# backstop for the residue.
_QUERY_STOPWORDS = frozenset({
    "extract", "fetch", "retrieve", "find", "search", "identify", "get",
    "the", "a", "an", "and", "or", "of", "for", "from", "with", "via",
    "using", "into", "onto", "per", "all", "each", "any", "that", "this",
    "data", "information", "endpoints", "specific", "targeted",
    "recent", "last", "months", "month", "days", "weeks",
})
_QUERY_MAX_TERMS = 8
# Batch tool language: max queries fanned out inside one web_search step.
_WEB_BATCH_MAX_QUERIES = 5


def _keywordize(query: str) -> str:
    """Reduce instruction prose to a search-engine keyword query:
    strip punctuation, drop instruction verbs/stopwords, cap terms."""
    import re
    tokens = re.split(r"[^\w.+-]+", query.lower())
    kept = [t for t in tokens if t and t not in _QUERY_STOPWORDS]
    return " ".join(kept[:_QUERY_MAX_TERMS])


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


import re as _re

_GITHUB_REPO_RE = _re.compile(
    r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)"
)


def _github_api_targets(url: str) -> Optional[list[tuple[str, str]]]:
    """Map a github.com repo URL onto REST API endpoints (2026-07-13,
    ratified Q3A after rematch #8: GitHub HTML beat both static fetch
    and the render fallback across two consecutive runs — the pages
    assemble releases/contributor data with JS behind persistent
    connections and rate-limit scrapers, while api.github.com serves
    the same data as clean JSON in milliseconds). Returns
    [(label, api_url), ...] or None for non-github URLs. Unauthenticated
    quota (60 req/hr/IP) covers a cognition run; CC_GITHUB_TOKEN in the
    environment raises it to 5000 when set."""
    m = _GITHUB_REPO_RE.match(url.strip())
    if not m:
        return None
    org, repo = m.group(1), m.group(2)
    repo = repo.removesuffix(".git")
    base = f"https://api.github.com/repos/{org}/{repo}"
    return [
        ("repo", base),
        ("releases", f"{base}/releases?per_page=5"),
        ("contributors", f"{base}/contributors?per_page=1&anon=true"),
    ]


def _render_github_json(label: str, data: Any) -> str:
    """Flatten the API JSON into the compact factual text a composer
    cites: versions, dates, counts, URLs — no scraping artifacts."""
    lines: list[str] = []
    if label == "repo" and isinstance(data, dict):
        lines.append(f"Repository: {data.get('html_url', '')}")
        lines.append(f"Description: {data.get('description') or ''}")
        lines.append(f"Stars: {data.get('stargazers_count')}")
        lines.append(f"Forks: {data.get('forks_count')}")
        lines.append(f"Open issues: {data.get('open_issues_count')}")
        lines.append(f"Last push: {data.get('pushed_at')}")
        lines.append(f"Created: {data.get('created_at')}")
        lines.append(f"Default branch: {data.get('default_branch')}")
    elif label == "releases" and isinstance(data, list):
        for rel in data:
            lines.append(
                f"Release {rel.get('tag_name')} — "
                f"name: {rel.get('name') or ''} — "
                f"published: {rel.get('published_at')} — "
                f"url: {rel.get('html_url')}"
            )
            body = (rel.get("body") or "").strip()
            if body:
                lines.append(f"  notes: {body[:1200]}")
    elif label == "contributors":
        # per_page=1&anon=true: the Link header carries the total, but
        # we only have the body here — report what one page shows.
        if isinstance(data, list) and data:
            lines.append(
                f"Top contributor: {data[0].get('login', 'anonymous')} "
                f"({data[0].get('contributions')} contributions)"
            )
    return "\n".join(x for x in lines if x is not None)


async def _fetch_github_via_api(url: str) -> Optional[dict]:
    """Fetch a github repo's data through api.github.com. Returns a
    finding dict or None (fall back to the HTML pipeline)."""
    targets = _github_api_targets(url)
    if not targets:
        return None
    import os

    import httpx
    headers = {"Accept": "application/vnd.github+json",
               "User-Agent": "crystal-cache-cognition"}
    token = os.environ.get("CC_GITHUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    def _pull() -> Optional[str]:
        sections: list[str] = []
        with httpx.Client(timeout=10.0, headers=headers) as client:
            for label, api_url in targets:
                try:
                    resp = client.get(api_url)
                    if resp.status_code != 200:
                        sections.append(
                            f"[{label}: api status {resp.status_code}]"
                        )
                        continue
                    text = _render_github_json(label, resp.json())
                    if text:
                        sections.append(f"== {label} ==\n{text}")
                except Exception as e:  # noqa: BLE001
                    sections.append(f"[{label}: {str(e)[:120]}]")
        joined = "\n\n".join(sections).strip()
        return joined or None

    content = await asyncio.to_thread(_pull)
    if not content:
        return None
    return {
        "title": f"GitHub API data for {url}",
        "url": url,
        "content": content,
        "source": "github_api",
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
    # web_fetch (2026-07-13, rematch #7): retrieve-KNOWN pages — the
    # second verb of the batch tool language ({"queries"} = discover,
    # {"urls"} = retrieve). Runs the guarded fetch pipeline directly
    # (fill_missing_content: SSRF guard, wall-clock deadline, headless
    # render fallback); needs no registry tool, so it dispatches before
    # the registry load and never raises RegistryUnavailable.
    if action_value == "web_fetch":
        raw_urls = step_input.get("urls")
        # noqa: E501 — github routing docs below
        if isinstance(raw_urls, list):
            urls = [
                u.strip() for u in raw_urls
                if isinstance(u, str) and u.strip()
            ][:_WEB_BATCH_MAX_QUERIES]
        else:
            single = step_input.get("url", "")
            urls = [single.strip()] if isinstance(single, str) and single.strip() else []
        if not urls:
            return {
                "urls": [], "results_count": 0, "findings": [],
                "content_text": "",
                "note": "web_fetch: no urls provided",
            }
        # GitHub routing (Q3A): repo URLs go to api.github.com — same
        # data the JS page renders, as JSON, in milliseconds. Non-github
        # URLs (and github URLs the API can't serve) continue through
        # the HTML pipeline below.
        api_findings: list[dict] = []
        html_urls: list[str] = []
        for u in urls:
            f = await _fetch_github_via_api(u)
            if f is not None:
                api_findings.append(f)
            else:
                html_urls.append(u)
        if not html_urls:
            return {
                "urls": urls,
                "results_count": len(api_findings),
                "findings": api_findings,
                "content_text": "",
            }

        from ..config import get_settings
        from ..search.fetch import fill_missing_content
        from ..search.render import render_available
        settings = get_settings()
        payload = {"results": [
            {"title": "", "url": u, "snippet": "", "content": None}
            for u in html_urls
        ]}
        payload = await asyncio.to_thread(
            fill_missing_content,
            payload,
            max_pages=len(urls),
            content_cap=30_000,
            deadline_seconds=settings.web_search_fetch_deadline_seconds,
            render_enabled=(
                settings.web_render_enabled and render_available()
            ),
            render_timeout_seconds=settings.web_render_timeout_seconds,
        )
        findings = api_findings + [
            {"title": r.get("title") or r["url"], "url": r["url"],
             "content": r["content"],
             "rendered": bool(r.get("rendered"))}
            for r in payload["results"] if r.get("content")
        ]
        return {
            "urls": urls,
            "results_count": len(findings),
            "findings": findings,
            "content_text": "",
        }

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

        async def _search_one(query: str) -> dict:
            out = await tool.impl(customer_id=customer_id, query=query)
            normalized = _normalize_web_output(out)
            # Zero-result backstop (2026-07-11, ratified Q1C): a search
            # that found nothing is retried ONCE with the query reduced
            # to keywords — the common cause is instruction prose in
            # the query, which search engines answer with nothing.
            if not normalized.get("findings"):
                reduced = _keywordize(query)
                if reduced and reduced != query.lower().strip():
                    retry_out = await tool.impl(
                        customer_id=customer_id, query=reduced,
                    )
                    retried = _normalize_web_output(retry_out)
                    if retried.get("findings"):
                        retried["retried_query"] = reduced
                        retried["original_query"] = query
                        return retried
            return normalized

        # Batch tool language (2026-07-11, ratified): one web_search
        # step accepts input {"queries": [...]} — up to
        # _WEB_BATCH_MAX_QUERIES keyword queries fanned out
        # CONCURRENTLY and merged, each finding tagged with the query
        # that produced it. Rematch #6 attempt 1 planned "execute three
        # parallel searches" inside ONE step; the tool language now
        # matches how the orchestrator thinks instead of forcing
        # intent into tool-shaped fragments. {"query": str} remains
        # the single-target form.
        raw_queries = step_input.get("queries")
        if isinstance(raw_queries, list):
            queries = [
                q.strip() for q in raw_queries
                if isinstance(q, str) and q.strip()
            ][:_WEB_BATCH_MAX_QUERIES]
        else:
            single = step_input.get("query", "")
            queries = [single] if single else []
        if not queries:
            return _normalize_web_output(
                await tool.impl(customer_id=customer_id, query="")
            )
        if len(queries) == 1:
            return await _search_one(queries[0])

        results = await asyncio.gather(
            *[_search_one(q) for q in queries]
        )
        merged_findings: list[dict] = []
        per_query: dict[str, int] = {}
        for q, r in zip(queries, results):
            findings = r.get("findings") or []
            for f in findings:
                f["query"] = q
            merged_findings.extend(findings)
            per_query[q] = len(findings)
        # content_text stays empty by design: web findings carry their
        # content per-finding (the composition renderer joins them via
        # _finding_to_text, title+URL+content, now with per-query
        # provenance) — same shape as a single search.
        return {
            "query": " | ".join(queries),
            "queries": queries,
            "provider": results[0].get("provider", ""),
            "results_count": len(merged_findings),
            "per_query_counts": per_query,
            "findings": merged_findings,
            "content_text": "",
        }

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
