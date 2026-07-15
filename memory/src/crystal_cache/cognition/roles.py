"""Cognition roles: orchestrator, worker, validator.

Each role is a function that takes the environment's visible state
(enforcing information barriers) and returns structured output.

v2 port (Phase 6 Wave C): one targeted refactor against the v1
verbatim — `_worker_crystal_key_scan` now calls
`store.list_facts_by_key_prefix(...)` instead of inline SQL via
`sqlalchemy.select`. Per R9 (no SQL outside the store).

Phase 7.5 refactor (§6.5.5, locked per P0.26 + P0.28):
- `run_worker`'s dispatch chain now delegates tool-style actions
  (crystal_search, crystal_key_scan, web_search) to the shared
  agent tool registry. The COMPOSITION_ACTIONS set explicitly
  identifies the cognition-only actions (analyze, synthesize,
  format) that stay in `_worker_llm_step`.
- The existing `_worker_crystal_search`, `_worker_crystal_key_scan`,
  and `_worker_web_search` helper functions REMAIN as test-callable
  fallbacks but are no longer the primary dispatch path. If the
  agent tool registry is empty (e.g. tests with no imports), the
  dispatcher falls back to the v1 helpers so cognition still works
  in isolation.
- The orchestrator's prompt text stays hardcoded for Phase 7.5
  (P0.28). Making it registry-derived requires its own design pass
  on prompt-injection / length-budget concerns; Phase 11 work.
- Wire-format strings (StepAction enum values, action names in the
  orchestrator prompt) are preserved per R3 — they appear in
  persisted `cognition_tasks.result` JSON rows.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

import structlog

from ..cost.emit import record_model_call
from ..encoding.executor import encode_native_async
from ..llm import get_llm_client
from .models import (
    CognitionEnvironment, CriterionEval, GoalDocument, OutputType,
    Plan, PlanStep, StepAction, StepOutput, StepStatus, ValidationResult,
)

if TYPE_CHECKING:
    from ..infrastructure.fact_vector_store import FactVectorStore
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# Cognition model routing runs through the provider-neutral seam. The
# wire-keys "haiku"/"sonnet" are preserved (they appear in persisted plan
# JSON + StepOutput.model_used — R3) and map onto the seam tiers below;
# CC_LLM_MODEL_SMALL / CC_LLM_MODEL_LARGE override which models the tiers
# resolve to.
_TIER_BY_KEY = {"haiku": "small", "sonnet": "large"}

# Composition actions: LLM-driven steps that take prior step outputs
# and produce structured intermediates. These stay in `_worker_llm_step`
# per §6.5.2 / D-A10 (composition cognition-only). The set is the source
# of truth for "is this a composition action" checks.
COMPOSITION_ACTIONS: frozenset[StepAction] = frozenset({
    StepAction.ANALYZE,
    StepAction.SYNTHESIZE,
    StepAction.FORMAT,
})


# C4: robust JSON extraction for orchestrator/validator responses.
#
# The earlier inline strip only handled a ```/```json fence at the very
# start and end of the text. The idle-log parse failures (2026-06-09)
# came from two other shapes: JSON wrapped in preamble/trailing prose,
# and — the common case — responses truncated by max_tokens, which leaves
# the JSON unterminated. This helper handles the first two robustly
# (direct parse → fenced block anywhere → first balanced brace span);
# truncation is addressed by raising the token ceilings at the call
# sites. A still-unparseable response returns None, and the caller
# fails closed (validator) or falls back to a default plan
# (orchestrator) exactly as before.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def _extract_json_object(raw: str) -> Optional[dict]:
    """Best-effort parse of a single JSON object from an LLM response.

    Tries, in order: a direct parse of the stripped text; the contents
    of the first ```/```json fenced block (tolerating surrounding
    prose); and the first balanced ``{...}`` span. Returns the parsed
    dict, or None if nothing parseable is found. Never raises.
    """
    if not raw:
        return None
    text = raw.strip()

    # 1. Direct parse.
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. First fenced block, even with prose around it.
    m = _JSON_FENCE_RE.search(text)
    if m:
        try:
            obj = json.loads(m.group(1).strip())
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. First balanced brace span (handles a preamble + trailing tail).
    start = text.find("{")
    if start != -1:
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        obj = json.loads(text[start:i + 1])
                        return obj if isinstance(obj, dict) else None
                    except (json.JSONDecodeError, ValueError):
                        break
    return None


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------

async def _source_bank_findings(
    env: CognitionEnvironment,
    store: Any,
    fact_store: Any,
    encoder: Any,
) -> list[dict]:
    """One gated bank lookup on the task, run by CODE before the
    orchestrator's LLM call (2026-07-11, ratified Q1A).

    Replaces the old "crystal_search always comes first" plan rule — a
    blind mandatory step that prefired off-topic banks into every run
    (rematch #4). Now the orchestrator SEES what the bank holds (already
    relevance-gated: an off-topic bank yields an honest empty) and
    curates which findings ride the plan. Never raises — sourcing
    failure degrades to "no bank material", it never blocks planning.
    """
    query = (env.task_goal or env.conversation_context or "").strip()[:300]
    if not query or store is None:
        return []
    try:
        from .retrieval_adapter import dispatch_cognition_retrieval
        out = await dispatch_cognition_retrieval(
            action_value="crystal_search",
            step_input={"query": query, "k": 10},
            customer_id=env.customer_id,
            store=store,
            fact_store=fact_store,
            encoder=encoder,
        )
        return list(out.get("findings") or [])
    except Exception as e:  # noqa: BLE001
        logger.info(
            "orchestrator.bank_sourcing_unavailable",
            env_id=env.id, error=str(e),
            note="planning proceeds without bank material",
        )
        return []


async def run_orchestrator(
    env: CognitionEnvironment,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any = None,
) -> tuple[GoalDocument, Plan]:
    """Orchestrator: reviews the trigger, creates goal contract + execution plan.

    Sees: trigger context; a gated bank check on the task (sourced by
    code, curated by the orchestrator onto plan.bank_findings —
    2026-07-11); on retry also the rejection history, the
    carried-findings inventory, and the rejected deliverable (trimmed) —
    the revision-aware retry amendment (2026-07-10): it classifies the
    failure into a route (compose_only / gap_fill / replan / give_up)
    and proposes the composition token budget.
    Writes: goal.json, plan.json
    Never sees: the CURRENT attempt's step outputs or deliverables
    """
    sourced = await _source_bank_findings(env, store, fact_store, encoder)
    if sourced:
        lines = []
        for i, f in enumerate(sourced, 1):
            lines.append(
                f"  {i}. [{f.get('fact_id', '?')}] "
                f"{(f.get('key') or '')[:80]}: "
                f"{(f.get('content') or '')[:200]}"
            )
        bank_context = (
            "\nBANK MATERIAL (a relevance-gated check of the crystal bank "
            "was already run on this task; these are the matches):\n"
            + "\n".join(lines)
            + "\nCurate: list the fact ids that are RELEVANT to this task "
            "in \"bank_finding_ids\" — workers will receive their full "
            "text as source material. Omit anything off-topic. Do not "
            "plan search steps for information already here.\n"
        )
    else:
        bank_context = (
            "\nBANK MATERIAL: the relevance-gated bank check found no "
            "material on this task. Plan accordingly (external sources "
            "or targeted lookups).\n"
        )

    # Q2B (2026-07-15): operator critiques — human judgment pinned to
    # prior work on this run or this trigger. The orchestrator treats
    # them as authoritative direction when writing the contract and
    # plan; workers never see them (barrier discipline unchanged).
    if getattr(env, "operator_critiques", None):
        crit_lines = []
        for c in env.operator_critiques[:10]:
            crit_lines.append(
                f"  - [{c.get('target_path', 'run')}] "
                f"{(c.get('text') or '')[:300]}"
            )
        bank_context += (
            "\nOPERATOR CRITIQUES (a human reviewed prior work on this "
            "task and pinned these critiques — treat them as "
            "authoritative direction; address each one in your goal "
            "and plan):\n" + "\n".join(crit_lines) + "\n"
        )
    # Revision-aware retry (2026-07-10, ratified Q1A/Q2A/Q4A/Q5A): on a
    # retry the orchestrator sees the verdict, the rejected deliverable
    # (trimmed head+tail), and an inventory of the findings already
    # gathered — then CLASSIFIES the failure into a route instead of
    # cold-replanning by default. Attempts are revisions, not
    # independent samples.
    # amend_contract availability (2026-07-14, ratified Q2A): the route
    # exists ONLY when the last verdict flagged criteria as possibly
    # infeasible — the appeal is adjudicated on documented evidence at
    # this seat, never negotiated with the validator.
    _flagged: list[tuple[int, str]] = []
    if env.attempt_history:
        _last_val = (env.attempt_history[-1] or {}).get("validation") or {}
        for _i, _c in enumerate(_last_val.get("criteria_evaluation") or []):
            if isinstance(_c, dict) and _c.get("possibly_infeasible"):
                _flagged.append((_i, str(_c.get("criterion", ""))[:160]))
    if _flagged:
        _flag_lines = "\n".join(
            f"    {i + 1}. (index {i}) {text}" for i, text in _flagged
        )
        _amendable_block = f"""
- "amend_contract": the validator flagged these criteria as POSSIBLY
  INFEASIBLE AS WRITTEN (on documented search evidence):
{_flag_lines}
  You may amend ONLY those criteria: set retry_route to
  "amend_contract" and provide "contract_amendments":
  [{{"criterion_index": <index above>, "amended": "<new criterion
  text>", "evidence": "<the documented evidence justifying it>"}}].
  Amend to what the evidence supports (e.g. a fixed count becomes
  "all that verifiably exist, with documented search breadth") —
  never simply delete a criterion. Amendments are permanently
  audit-trailed on the goal document. Plan the revision steps in the
  same plan; unflagged criteria are untouchable."""
    else:
        _amendable_block = ""

    rejection_context = ""
    if env.rejection_log:
        rejection_context = "\n\nPREVIOUS ATTEMPT(S) FAILED. Validator feedback:\n"
        for entry in env.rejection_log:
            rejection_context += f"\nAttempt {entry['attempt']}:\n"
            rejection_context += f"  Reasoning: {entry['reasoning']}\n"
            for issue in entry.get("issues", []):
                rejection_context += f"  Issue: {issue}\n"
            for sug in entry.get("suggestions", []):
                rejection_context += f"  Suggestion: {sug}\n"

        if env.carried_findings:
            rejection_context += (
                "\nFINDINGS ALREADY GATHERED (carried across attempts — "
                "workers will see their full text; do NOT re-plan searches "
                "for information already here):\n"
            )
            for f in env.carried_findings:
                rejection_context += (
                    f"  - [{f.get('action', '?')}] "
                    f"{f.get('description', '')[:100]}: "
                    f"{(f.get('text', '') or '')[:200]}\n"
                )
        if env.prior_deliverable:
            rejection_context += (
                "\nREJECTED DELIVERABLE (trimmed):\n"
                + _trim_head_tail(
                    env.prior_deliverable, _REVISION_DELIVERABLE_CHARS
                )
                + "\n"
            )
        rejection_context += f"""
THIS IS A REVISION. Classify the failure and set "retry_route" in your plan:
- "compose_only": the findings above already contain what the deliverable
  needs; the failure is in the composition (structure, missing sections,
  placeholders, unsupported claims). Plan ONLY analyze/synthesize/format
  steps that revise the rejected deliverable using the carried findings.
- "gap_fill": specific information is missing. Plan TARGETED search steps
  for exactly the named gaps (never repeat searches whose findings are
  already carried), then composition steps that revise.
- "replan": the prior attempt's approach was wrong at the root; carried
  material would anchor a revision to the same hole. Plan from scratch
  (carried findings and the rejected deliverable will be discarded).
- "give_up": the goal cannot be achieved with the available tools (e.g.
  required external search is unconfigured, or the needed information
  demonstrably does not exist in any reachable source). Set retry_route
  to "give_up" and put the explanation in the plan's "reasoning"; steps
  may be an empty list. Do not burn budget on attempts that cannot
  succeed.{_amendable_block}
Fix the named deficiencies without regressing what was adequate."""

    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    # The research action is agentic (ratified Q2A) — advertised only
    # when the agentic-workers flag is on, so a classic deployment's
    # orchestrator never plans steps the workers execute degraded.
    try:
        from ..config import get_settings as _gs
        _agentic = bool(getattr(_gs(), "cognition_agentic_workers", False))
    except Exception:  # noqa: BLE001
        _agentic = False
    _research_enum = "|research" if _agentic else ""
    _research_action = (
        "\n- research: an agentic retrieve-and-verify step. Input "
        '{{"targets": ["<name> — <what to verify>", ...]}} (up to 5). '
        "Hand it NAMES, never URLs — it discovers the canonical "
        "source, fetches it, CONFIRMS IDENTITY (the fetched repo/page "
        "is actually the named thing), and verifies the asked-for "
        "property from the primary source. Plan one research step per "
        "group of ~3-5 targets so verification capacity scales with "
        "target count. Use it for every acceptance criterion that "
        "names specific projects/entities."
    ) if _agentic else ""
    prompt = f"""You are a research orchestrator. TODAY'S DATE IS {_today} (UTC). You receive a goal and must produce:
1. A GOAL DOCUMENT (contract for the validator)
2. An EXECUTION PLAN (instructions for workers)

TASK: {env.task_goal or env.conversation_context or "No additional context"}

CONTEXT: {(env.conversation_context[:600] if env.task_goal else "") or "None"}

SOURCE CRYSTAL: {env.source_crystal_id or "None specified"}
OUTPUT TYPE: {env.output_type.value}
{bank_context}{rejection_context}

Respond with ONLY valid JSON matching this structure:
{{
  "goal": {{
    "title": "short title",
    "description": "what the deliverable must contain",
    "acceptance_criteria": ["criterion 1", "criterion 2", "..."]
  }},
  "plan": {{
    "reasoning": "why this plan will work",
    "steps": [
      {{
        "id": 1,
        "action": "crystal_search|crystal_key_scan|web_search|web_fetch{_research_enum}|analyze|synthesize|format",
        "description": "what this step does",
        "input": {{"query": "...", "instruction": "..."}},
        "depends_on": [],
        "parallel_group": "A or null"
      }}
    ],
    "expected_output": "what the final deliverable should look like",
    "suggested_key": "wide|...|specific unified sparse key (general to specific)",
    "parent_crystal_id": "{env.source_crystal_id}",
    "retry_route": "\"\" on a first attempt; on a revision one of compose_only|gap_fill|replan|give_up|amend_contract",
    "bank_finding_ids": ["fact ids from BANK MATERIAL that are relevant to this task; [] when none are"]
  }}
}}

Available worker actions:
- crystal_search: Vector similarity search. Finds content by MEANING.
  Best for: "what happens in the break room scene", "find dialogue about Christmas"
  Bad for: counting, listing, enumerating. Vector search returns top-k results,
  not exhaustive lists. If you need ALL of something, use crystal_key_scan.
  Input: {{"query": "search terms", "pair_types": ["content_chunk"], "k": 10}}
- crystal_key_scan: Prefix scan on sparse keys. Returns ALL matching facts.
  Sparse keys are wide->specific paths (e.g. "Film|Corporate Mistletoe|Script|Scene 5").
  key_prefix matches the WIDE (left) end; subject_contains matches ANY segment
  (use it to find something regardless of where it sits in the path).
  Best for: counting ("how many scenes"), listing ("what chapters exist"),
  enumerating ("all characters mentioned"), structural questions about the corpus.
  Bad for: semantic meaning. It matches key text literally, not by meaning.
  Input: {{"key_prefix": "Film|Corporate Mistletoe|Script", "subject_contains": "Scene"}}
- web_search: Search the web for external information (requires operator configuration; unconfigured runs return an explicit error the plan can react to).
  Best for: questions that can't be answered from the crystal bank.
  Bad for: anything already in the crystal bank. Always check crystals first.
  Input: {{"query": "search terms"}}
- web_fetch: Retrieve EXACT page URLs you already know (rendered if the page needs JavaScript). When you know where the data lives — GitHub releases (https://github.com/{{org}}/{{repo}}/releases), official changelogs, project homepages — fetch it directly instead of searching and hoping the right page ranks. Search is for discovery; fetch is for known sources.
  Input: {{"urls": ["https://github.com/org/repo/releases"]}}{_research_action}
- source_lookup: Read ACTUAL source code — never reconstruct code or file paths from memory.
  ops: "search" (find a symbol/string across files), "read" (one file's contents), "list" (a directory).
  Best for: "where is X defined", "what does the code at path P do", verifying a path exists before asserting it.
  Bad for: anything answerable from the crystal bank (check crystal_search first). Needs a configured source backend.
  Input: {{"op": "search", "query": "generate_sparse_key"}} or {{"op": "read", "path": "src/crystal_cache/encoding/sparse_keys.py"}}
- analyze: LLM reasoning over prior step outputs.
  Best for: extracting insights, comparing, interpreting source material.
  Input: {{"instruction": "what to analyze"}}
- synthesize: Combine multiple sources into one coherent output.
  Best for: merging findings from multiple search steps into a single answer.
  Input: {{"instruction": "what to combine"}}
- format: Structure the final deliverable for output.
  Input: {{"format": "markdown", "sections": ["...", "..."]}}

Rules:
- The bank has ALREADY been checked (BANK MATERIAL above) — never plan a
  crystal_search step to re-check it. Plan crystal_search/crystal_key_scan
  steps only for TARGETED lookups the check can't cover.
- Use crystal_key_scan when the task involves COUNTING or LISTING items (scenes, chapters, etc.)
- web_search input takes {{"queries": ["...", "..."]}} — one to five
  SEARCH-ENGINE KEYWORD QUERIES (3-8 words each, like you'd type into a
  search box), fanned out concurrently inside the ONE step. Each query
  targets ONE thing; related targets are separate queries in the same
  step. Instruction prose belongs in the step description, never in a
  query.
  GOOD: {{"queries": ["WhisperX latest release changelog", "MLT Framework latest release", "OpenTimelineIO release notes"]}}
  BAD:  {{"queries": ["Extract WhisperX release data: latest stable version, recent releases, changelog, and commit activity from GitHub API endpoints"]}}
- web_fetch input takes {{"urls": ["...", "..."]}} — one to five exact page
  URLs fetched concurrently inside the ONE step.
- URL DISCIPLINE: you DIRECT the work, you do not do it. Any URL you
  write into a step input must be copied VERBATIM from the material
  provided to you (bank material, carried findings). NEVER compose a
  URL from memory — a guessed repo org that happens to exist returns
  convincing data about the WRONG project. When you only know a NAME,
  hand the name to a research step (or plan a web_search first).
- VALIDATOR ALIGNMENT: you wrote the acceptance criteria — plan
  against them. Every criterion must have a step that produces or
  verifies it. Criteria about named projects/entities (versions,
  dates, "newly launched") require verification from primary sources
  that CONFIRMS IDENTITY — the fetched source must actually be the
  named thing, not a look-alike.
- Read-only steps (crystal_search, crystal_key_scan, web_search, web_fetch, research, source_lookup) can share a parallel_group.
- Write steps (analyze, synthesize, format) must have parallel_group: null.
- Maximum 5 steps.
- acceptance_criteria must be specific and testable.
- CRITERIA MUST BE SATISFIABLE regardless of what the world turns out
  to contain: for "find/discover X" goals, write "all X that
  verifiably exist, with documented search breadth" — do not invent a
  fixed count the world isn't obligated to contain ("at least 3") —
  fixed counts are allowed ONLY when the user's request explicitly
  demanded a number.
- suggested_key is a wide->specific path (general to specific), variable length.
- Be terse in every string field — the JSON must be complete and well-formed."""

    t0 = time.time()
    llm = await asyncio.to_thread(
        get_llm_client().complete_detailed,
        system=None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_ORCHESTRATOR_MAX_TOKENS,
        temperature=1.0,
        tier=_TIER_BY_KEY["haiku"],
    )
    duration_ms = int((time.time() - t0) * 1000)

    raw = llm.text
    tokens_in = llm.input_tokens or 0
    tokens_out = llm.output_tokens or 0
    env.record_tokens(tokens_in, tokens_out, "haiku")
    await record_model_call(
        customer_id=env.customer_id,
        model=llm.model,
        input_tokens=llm.input_tokens,
        output_tokens=llm.output_tokens,
        cache_read_tokens=llm.cache_read_tokens,
        cache_creation_tokens=llm.cache_creation_tokens,
        origin="cognition",
        session_id=env.id,
    )

    data = _extract_json_object(raw)
    if data is None:
        logger.warning(
            "orchestrator.json_parse_failed",
            raw_head=raw[:200],
            raw_tail=raw[-200:],
            raw_len=len(raw),
            note="falling back to the default plan",
        )
        data = _fallback_plan(env)

    goal_data = data.get("goal", {})
    goal = GoalDocument(
        title=goal_data.get("title", "Research task"),
        description=goal_data.get("description", env.conversation_context[:200]),
        acceptance_criteria=goal_data.get("acceptance_criteria", ["Deliverable addresses the original request"]),
        output_type=env.output_type,
        output_metadata={
            "parent_crystal_id": env.source_crystal_id,
            "suggested_key": data.get("plan", {}).get("suggested_key", ""),
        },
        source_context={
            "trigger_type": env.trigger_type,
            "trigger_id": env.trigger_id,
            "conversation_context": env.conversation_context[:500],
        },
    )

    plan_data = data.get("plan", {})
    steps = []
    for s in plan_data.get("steps", []):
        try:
            action = StepAction(s.get("action", "analyze"))
        except ValueError:
            action = StepAction.ANALYZE
        steps.append(PlanStep(
            id=s.get("id", len(steps) + 1),
            action=action,
            description=s.get("description", ""),
            input=s.get("input", {}),
            depends_on=s.get("depends_on", []),
            parallel_group=s.get("parallel_group"),
            model=s.get("model", "haiku"),
        ))

    _route = str(plan_data.get("retry_route", "") or "").strip().lower()
    if _route not in ("", "compose_only", "gap_fill", "replan", "give_up",
                      "amend_contract"):
        _route = ""
    # amend_contract payload (Q2A): shape-validated here; the ENGINE
    # enforces the flagged-only rule against the last verdict.
    _amendments: list[dict] = []
    for a in plan_data.get("contract_amendments") or []:
        if not isinstance(a, dict):
            continue
        try:
            idx = int(a.get("criterion_index"))
        except (TypeError, ValueError):
            continue
        amended = str(a.get("amended", "") or "").strip()
        if amended:
            _amendments.append({
                "criterion_index": idx,
                "amended": amended,
                "evidence": str(a.get("evidence", "") or "")[:500],
            })
    # Q1A curation: the orchestrator names the sourced findings it wants;
    # unknown ids are ignored (it can only carry what code actually
    # sourced). Empty/omitted → nothing rides the plan.
    _wanted = plan_data.get("bank_finding_ids") or []
    _by_id = {f.get("fact_id"): f for f in sourced}
    bank_findings = [
        _by_id[fid] for fid in _wanted
        if isinstance(fid, str) and fid in _by_id
    ]
    plan = Plan(
        reasoning=plan_data.get("reasoning", ""),
        steps=steps,
        expected_output=plan_data.get("expected_output", ""),
        suggested_key=plan_data.get("suggested_key", ""),
        parent_crystal_id=plan_data.get("parent_crystal_id", env.source_crystal_id),
        retry_route=_route,
        contract_amendments=_amendments,
        bank_findings=bank_findings,
    )

    logger.info(
        "orchestrator.complete",
        env_id=env.id,
        goal_title=goal.title[:40],
        step_count=len(plan.steps),
        duration_ms=duration_ms,
        tokens=tokens_in + tokens_out,
    )

    return goal, plan


def _fallback_plan(env: CognitionEnvironment) -> dict:
    """Fallback plan when orchestrator JSON fails to parse."""
    task = env.task_goal or env.conversation_context or ""
    return {
        "goal": {
            "title": "Research task",
            "description": task[:400] if task else "Complete the requested research",
            "acceptance_criteria": ["Deliverable addresses the original request", "All claims supported by source material"],
        },
        "plan": {
            # 2026-07-09: the fallback now also searches the WEB. It used
            # to be bank-only, so an orchestrator parse failure on an
            # external-knowledge task guaranteed an evidence-free stub
            # (or a C2 park). Where search is unconfigured the web step
            # returns its explicit error and the C2 gate behaves as
            # before; where configured, the degraded path can research.
            "reasoning": "Fallback plan: bank + web search, analyze, format",
            "steps": [
                {"id": 1, "action": "crystal_search", "description": "Vector search for relevant content",
                 "input": {"query": task[:100]}, "depends_on": [], "parallel_group": "A"},
                {"id": 2, "action": "web_search", "description": "Web search for external information",
                 "input": {"query": task[:100]}, "depends_on": [], "parallel_group": "A"},
                {"id": 3, "action": "analyze", "description": "Analyze findings from the searches",
                 "input": {"instruction": "Analyze the source material and extract relevant information to answer the question"},
                 "depends_on": [1, 2], "parallel_group": None},
                {"id": 4, "action": "format", "description": "Format deliverable",
                 "input": {"format": "markdown"}, "depends_on": [3], "parallel_group": None},
            ],
            "suggested_key": "",
            "parent_crystal_id": env.source_crystal_id,
        },
    }


# ---------------------------------------------------------------------------
# WORKERS
# ---------------------------------------------------------------------------

async def run_worker(
    env: CognitionEnvironment,
    step: PlanStep,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any,
) -> StepOutput:
    """Execute a single worker step.

    Sees: plan, prior step outputs, read-only resources; on a revision
    route, composition steps also see the PRIOR attempt's verdict +
    rejected deliverable (the revision-aware retry amendment,
    2026-07-10 — the verdict is the work order)
    Writes: its own step output
    Never sees: goal.json, or the CURRENT attempt's validation

    Phase 7.5 §6.5.5 refactor (P0.26):
      - Composition actions (analyze/synthesize/format) → `_worker_llm_step`
      - Other actions → shared agent tool registry via
        `_dispatch_tool_via_registry`. The registry is filtered to
        the "cognition" context per D-A10. If the registry has no
        matching tool, the dispatcher falls back to the v1 worker
        helpers (`_worker_crystal_search`, `_worker_crystal_key_scan`,
        `_worker_web_search`) so cognition keeps working in
        isolated tests where the agent package isn't imported.
    """
    result = StepOutput(
        step_id=step.id,
        action=step.action.value,
        status=StepStatus.RUNNING,
        started_at=datetime.now(timezone.utc),
    )

    t0 = time.time()

    try:
        if step.action == StepAction.RESEARCH:
            # Q2A (2026-07-14): a plannable retrieve-and-verify step.
            from .agentic import run_research_step
            result.output = await run_research_step(
                env=env, step=step,
                store=store, fact_store=fact_store, encoder=encoder,
            )
            result.status = StepStatus.COMPLETE
        elif step.action in COMPOSITION_ACTIONS:
            # Composition actions stay cognition-only — see D-A10 +
            # §6.5.3. Each composition action reads `prior_context`
            # built from dependency step outputs, which is the shape
            # cognition's plan-execution model assumes; agent-side
            # llm_invoke doesn't have that shape.
            result = await _worker_llm_step(
                env, step, result,
                store=store, fact_store=fact_store, encoder=encoder,
            )
        else:
            # Tool-style action — dispatch via the shared registry
            # if available; fall back to the v1 helper if not.
            result = await _dispatch_tool_via_registry(
                env=env,
                step=step,
                result=result,
                store=store,
                fact_store=fact_store,
                encoder=encoder,
            )

    except Exception as e:
        result.status = StepStatus.FAILED
        result.error = str(e)
        logger.error("worker.step_failed", env_id=env.id, step_id=step.id, error=str(e))

    result.duration_ms = int((time.time() - t0) * 1000)
    result.completed_at = datetime.now(timezone.utc)

    logger.info(
        "worker.step_complete",
        env_id=env.id,
        step_id=step.id,
        action=step.action.value,
        status=result.status.value,
        duration_ms=result.duration_ms,
        tokens=result.tokens_in + result.tokens_out,
    )

    return result


async def _dispatch_tool_via_registry(
    *,
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any,
) -> StepOutput:
    """Dispatch a tool-style worker step onto the shared agent tool registry.

    Per §6.5.5 + B: cognition retrieval is unified onto the shared
    tools through `cognition.retrieval_adapter`. The adapter maps the
    plan's step.input to the right tool(s), fans crystal_search across
    content_search + knowledge_search (filtering by the requested
    pair_types), and hydrates the per-fact findings / content_text
    shape cognition's analyze/synthesize steps consume. The agent
    tools' own contract (injection_text) is unchanged — the adapter
    expands it into the cognition view.

    When the registry can't serve the action (agent package not
    importable in cognition-in-isolation tests, or no adapter mapping
    for the action), the adapter raises RegistryUnavailable and we
    fall back to the v1 worker helpers, which accept the plan's input
    shape natively.
    """
    # The adapter does the registry load + tool-state injection and
    # raises RegistryUnavailable when the agent package isn't present
    # or the action has no adapter mapping.
    try:
        from .retrieval_adapter import (
            RegistryUnavailable,
            dispatch_cognition_retrieval,
        )
    except Exception as e:
        logger.info(
            "worker.adapter_unavailable",
            env_id=env.id,
            action=step.action.value,
            error=str(e),
            note="falling back to v1 worker helper",
        )
        return await _dispatch_via_fallback(
            env=env, step=step, result=result,
            store=store, fact_store=fact_store, encoder=encoder,
        )

    try:
        output = await dispatch_cognition_retrieval(
            action_value=step.action.value,
            step_input=step.input,
            customer_id=env.customer_id,
            store=store,
            fact_store=fact_store,
            encoder=encoder,
        )
    except RegistryUnavailable as e:
        # Designed graceful-degradation path (INFO, not ERROR): the
        # v1 worker helpers accept the plan's input shape natively.
        logger.info(
            "worker.registry_unavailable",
            env_id=env.id,
            action=step.action.value,
            error=str(e),
            note="falling back to v1 worker helper",
        )
        return await _dispatch_via_fallback(
            env=env, step=step, result=result,
            store=store, fact_store=fact_store, encoder=encoder,
        )
    except Exception as e:
        result.status = StepStatus.FAILED
        result.error = f"cognition retrieval adapter failed: {e}"
        logger.error(
            "worker.adapter_failed",
            env_id=env.id,
            action=step.action.value,
            error=str(e),
        )
        return result

    result.output = output if isinstance(output, dict) else {"result": output}
    result.status = StepStatus.COMPLETE
    result.model_used = f"registry_adapter:{step.action.value}"
    return result


async def _dispatch_via_fallback(
    *,
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any,
) -> StepOutput:
    """Fall back to the v1 worker helpers when the registry isn't available.

    Preserves behavior for cognition-in-isolation tests and for any
    deployment where the agent package isn't loaded.
    """
    if step.action == StepAction.CRYSTAL_SEARCH:
        return await _worker_crystal_search(env, step, result, store, fact_store, encoder)
    if step.action == StepAction.CRYSTAL_KEY_SCAN:
        return await _worker_crystal_key_scan(env, step, result, store)
    if step.action == StepAction.WEB_SEARCH:
        return _worker_web_search(env, step, result)
    # Unknown action — same failure path as v1.
    result.status = StepStatus.FAILED
    result.error = f"Unknown action: {step.action}"
    return result


async def _worker_crystal_search(
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
    encoder: Any,
) -> StepOutput:
    """Pure tool call, no model needed.

    Phase 7.5 §6.5.5: this helper REMAINS as a v1-compatible fallback
    for `_dispatch_via_fallback`. The primary dispatch path now goes
    through the agent tool registry (knowledge_search has cognition
    alias "crystal_search" → routes here at the API level). Behavior
    unchanged.
    """
    query = step.input.get("query", "")
    pair_types = step.input.get("pair_types", ["content_chunk", "question_answer"])
    k = step.input.get("k", 10)

    if not query:
        result.status = StepStatus.FAILED
        result.error = "No query provided for crystal_search"
        return result

    query_vector = await encode_native_async(encoder, query)
    search_results = await fact_store.search(
        customer_id=env.customer_id,
        query_vector=query_vector,
        pair_types=pair_types,
        k=k,
    )

    # Bank relevance floor (2026-07-11) — same gate as the registry
    # adapter (COGNITION_BANK_RELEVANCE_FLOOR there; per-fact here since
    # this path still has per-fact scores). Sub-floor matches are the
    # k nearest UNRELATED neighbors of an off-topic bank; they poison
    # the composition context and fake C2 grounding.
    from .retrieval_adapter import COGNITION_BANK_RELEVANCE_FLOOR

    findings = []
    seen = set()
    for fact_id, crystal_id, pair_type, score in search_results[:8]:
        if score < COGNITION_BANK_RELEVANCE_FLOOR:
            continue
        if fact_id in seen:
            continue
        seen.add(fact_id)
        facts = await store.list_facts_for_crystal(crystal_id)
        for f in facts:
            if f.id == fact_id:
                content = f.claim_text or f.answer_value or ""
                findings.append({
                    "fact_id": f.id,
                    "crystal_id": crystal_id,
                    "key": f.prompt_text or "",
                    "content": content[:1500],
                    "pair_type": pair_type,
                    "score": round(score, 4),
                })
                break

    result.output = {
        "query": query,
        "results_count": len(findings),
        "findings": findings,
        "content_text": "\n\n".join(
            (f.get("content") or "") for f in findings
        ),
    }
    result.status = StepStatus.COMPLETE
    result.model_used = "none (tool call)"
    return result


async def _worker_crystal_key_scan(
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
    store: "MetadataStore",
) -> StepOutput:
    """Key prefix scan. No model, no vector search.

    Finds ALL facts whose prompt_text (sparse key) starts with a prefix.
    Use for enumeration tasks: count scenes, list chapters, etc.

    v2 port (Phase 6 Wave C): the v1 implementation reached into
    SQLAlchemy directly to build a (FactRow JOIN CrystalRow) prefix
    scan with optional subject filter. v2 routes through
    `store.list_facts_by_key_prefix` from CognitionExtensionsMixin so
    no SQL leaks outside the store layer (R9). The returned Fact
    objects carry the same fields the v1 raw-row path produced, and
    the findings-dict shape below matches v1 exactly so downstream
    LLM steps see identical inputs.

    Phase 7.5 §6.5.5: this helper REMAINS as a v1-compatible fallback
    for `_dispatch_via_fallback`. The primary dispatch path now goes
    through the agent tool registry (navigation_search has cognition
    alias "crystal_key_scan" → routes via registry at the API level).
    Behavior unchanged.
    """
    key_prefix = step.input.get("key_prefix", "")
    subject_contains = step.input.get("subject_contains", "")

    # A scan needs at least one filter. Either a key prefix (e.g.
    # "Code|") OR a subject substring (e.g. "generate_sparse_key") is
    # enough — an empty prefix with a subject filter scans all keys for
    # that substring, which is how identity lookups ("where is X
    # defined?") find their target.
    if not key_prefix and not subject_contains:
        result.status = StepStatus.FAILED
        result.error = "crystal_key_scan needs a key_prefix or subject_contains"
        return result

    facts = await store.list_facts_by_key_prefix(
        env.customer_id,
        key_prefix=key_prefix,
        subject_contains=subject_contains or None,
    )

    findings = []
    for fact in facts:
        content = fact.claim_text or fact.answer_value or ""
        findings.append({
            "fact_id": fact.id,
            "crystal_id": fact.crystal_id,
            "key": fact.prompt_text or "",
            "pair_type": fact.pair_type or "",
            "content_preview": content[:300],
            "content_length": len(content),
        })

    # Build a summary that the analyze step can work with.
    # Identical to v1's format.
    key_list = "\n".join(f"- {f['key']}" for f in findings)
    summary = f"Key prefix scan for '{key_prefix}' found {len(findings)} facts:\n{key_list}"

    result.output = {
        "key_prefix": key_prefix,
        "subject_contains": subject_contains,
        "results_count": len(findings),
        "findings": findings,
        "content_text": summary,
    }
    result.status = StepStatus.COMPLETE
    result.model_used = "none (key scan)"
    return result


def _worker_web_search(
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
) -> StepOutput:
    """Web search fallback dispatch.

    The PRIMARY path is the registry web_search tool (§6.5.5), which routes
    through the provider seam (search/web.py) and logs the interaction.
    This fallback fires only when the registry lookup fails; it stays
    network-free and reports the situation honestly so a plan step never
    silently produces empty success.
    """
    result.output = {
        "note": (
            "web_search dispatched via the network-free fallback — the "
            "registry tool (provider-seam path) was unavailable"
        ),
        "query": step.input.get("query", ""),
    }
    result.status = StepStatus.COMPLETE
    result.model_used = "none"
    return result


def _finding_to_text(f: object) -> str:
    """One finding → one text block, None-proof (2026-07-09 rematch:
    a web result carrying content=None detonated the old
    `f.get("content", "")` join — .get's default only fires when the
    KEY is absent, not when the value is None — killing analyze at
    0ms and cascading template deliverables through three attempts).
    Also carries title/url forward: the old join dropped them, which
    made cite-every-claim criteria structurally unsatisfiable — the
    analyst never saw a URL."""
    if not isinstance(f, dict):
        return str(f or "")
    text = f.get("content") or f.get("snippet") or f.get("text") or ""
    title = f.get("title") or ""
    url = f.get("url") or f.get("source_url") or ""
    head = " — ".join(x for x in (title, url) if x)
    return "\n".join(x for x in (head, text) if x)


def _render_step_output_text(out: "StepOutput") -> str:
    """Render one step output's usable text: prefer content_text/content,
    else join findings via _finding_to_text. Shared by the composition
    prompt assembly AND the engine's carryover harvest (revision-aware
    retry, 2026-07-10) so what a retry carries is exactly what a
    composition step would have read."""
    content = (
        out.output.get("content_text")
        or out.output.get("content")
        or ""
    )
    if not content:
        findings = out.output.get("findings") or []
        pieces = [t for t in (_finding_to_text(f) for f in findings) if t]
        content = "\n\n".join(pieces)
    return content


def _fair_share_allocations(sizes: list[int], budget: int) -> list[int]:
    """Max-min fair allocation of a character budget across parts.

    Rematch #4 (2026-07-11): the composition context was assembled in
    dependency order then sliced — so an early oversized part starved
    every later one (10K of bank noise truncated the emerging-projects
    findings to ZERO, and the run failed for data it already had).
    Fair share instead: parts at or under the equal share keep
    everything; the surplus redistributes to larger parts. No part can
    starve another — the "no component starves another" principle
    applied to the prompt window.

    Ascending pass: for each remaining part, share = remaining budget /
    remaining parts; a part takes min(its size, share). Processing
    smallest-first makes the shares only grow, which is what makes the
    allocation max-min fair.
    """
    n = len(sizes)
    if n == 0 or budget <= 0:
        return [0] * n
    order = sorted(range(n), key=lambda i: sizes[i])
    alloc = [0] * n
    remaining_budget = budget
    remaining_parts = n
    for idx in order:
        share = remaining_budget // remaining_parts
        take = min(sizes[idx], share)
        alloc[idx] = take
        remaining_budget -= take
        remaining_parts -= 1
    return alloc


def _assemble_prior_context(env: CognitionEnvironment, step: PlanStep) -> str:
    """Build the source-material block a composition step reads.

    Per-dependency: prefer content_text/content; else render findings
    via _finding_to_text. A FAILED dependency contributes an explicit
    marker (downstream models must KNOW a step died — the rematch's
    synthesize/format steps received silence and correctly refused to
    fabricate, but with the marker they can also say WHICH input is
    missing). Every piece is None-proofed before joining.

    Revision-aware retry (2026-07-10, ratified Q1A): on attempt>1 the
    prior attempt's harvested findings (env.carried_findings) PREFIX the
    block — research already paid for feeds the revision instead of
    being re-bought. The "replan" route arrives here with the carryover
    already dropped by the engine.

    Budgeting (2026-07-11): every part — each carried finding and each
    dependency block — gets a max-min fair share of
    _PRIOR_CONTEXT_MAX_CHARS (head-kept within its share), so no part
    can starve another. See _fair_share_allocations for the rematch-#4
    evidence."""
    headers: list[str] = []
    bodies: list[str] = []
    for f in (env.plan.bank_findings if env.plan else []):
        content = f.get("content") or ""
        if not content.strip():
            continue
        headers.append(
            f"--- Bank finding ({(f.get('key') or '')[:80]}) ---"
        )
        bodies.append(content)
    for f in env.carried_findings:
        text = f.get("text") or ""
        if not text.strip():
            continue
        headers.append(
            f"--- Carried finding (attempt {f.get('attempt', '?')}, "
            f"{f.get('action', '?')}: {f.get('description', '')[:80]}) ---"
        )
        bodies.append(text)
    for dep_id in step.depends_on:
        dep_output = env.step_outputs.get(dep_id)
        if dep_output is None:
            continue
        if dep_output.status == StepStatus.COMPLETE:
            content = _render_step_output_text(dep_output)
            if content:
                headers.append(f"--- Step {dep_id} output ---")
                bodies.append(content)
        elif dep_output.status == StepStatus.FAILED:
            headers.append(
                f"--- Step {dep_id} FAILED: "
                f"{dep_output.error or 'no error recorded'} ---"
            )
            bodies.append("")
    if not headers:
        return "(No prior output available)"
    overhead = sum(len(h) + 2 for h in headers)  # headers + joins ride free-ish
    budget = max(0, _PRIOR_CONTEXT_MAX_CHARS - overhead)
    allocs = _fair_share_allocations([len(b) for b in bodies], budget)
    parts = []
    for header, body, alloc in zip(headers, bodies, allocs):
        parts.append(f"{header}\n{body[:alloc]}" if body else header)
    return "\n\n".join(parts)


async def _worker_llm_step(
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
    store: Any = None,
    fact_store: Any = None,
    encoder: Any = None,
) -> StepOutput:
    """LLM-based step: analyze, synthesize, or format.

    Phase 7.5 §6.5.5: composition actions stay cognition-only per
    D-A10. This function is the worker dispatcher's primary path
    for ANALYZE / SYNTHESIZE / FORMAT actions; unchanged from v1
    except for clarifying the §6.5.5 boundary.
    """
    prior_context = _assemble_prior_context(env, step)

    instruction = step.input.get("instruction", step.description)
    sections = step.input.get("sections", [])

    if step.action == StepAction.FORMAT and sections:
        instruction += f"\n\nOrganize the output with these sections: {', '.join(sections)}"

    # Revision block (2026-07-10, ratified Q1A/Q3A): on a revision route
    # the composer sees the rejected deliverable + the verdict — the
    # verdict IS the work order. Barrier amendment, deliberate: within an
    # attempt workers still never see the goal or the CURRENT validation;
    # what pierces here is the PRIOR attempt's outcome, without which a
    # "revision" is just a re-roll. The "replan" route arrives with
    # prior_deliverable already dropped by the engine.
    revision_block = ""
    if env.prior_deliverable and env.rejection_log:
        last = env.rejection_log[-1]
        issues = "\n".join(
            f"  - {i}" for i in (last.get("issues") or [])
        ) or "  (none listed)"
        suggestions = "\n".join(
            f"  - {sug}" for sug in (last.get("suggestions") or [])
        )
        revision_block = f"""

THIS IS A REVISION of a rejected deliverable.
VALIDATOR VERDICT: {last.get('reasoning', '')}
ISSUES TO FIX:
{issues}
{("SUGGESTIONS:" + chr(10) + suggestions) if suggestions else ""}
REJECTED DELIVERABLE (trimmed):
{_trim_head_tail(env.prior_deliverable, _REVISION_DELIVERABLE_CHARS)}

Fix the named deficiencies without regressing what was adequate. Never
output placeholders — if the source material lacks something, state the
gap explicitly."""

    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""You are a research worker executing step {step.id}. TODAY'S DATE IS {_today} (UTC).

YOUR TASK: {step.description}

INSTRUCTION: {instruction}

PRIOR WORK (from previous steps):
{prior_context[:_PRIOR_CONTEXT_MAX_CHARS]}{revision_block}

Rules:
- Work ONLY on your assigned task
- Base your output on the source material provided, not your own knowledge
- If source material doesn't contain what you need, say so explicitly
- Document your reasoning
- Cite the ORIGINAL external URL for every factual claim (the url in the
  source material), never internal step numbers — "per Step 3" is not a
  citation
- When source material presents repo/page data for a NAMED project,
  confirm the FETCHED REPOSITORY (or page identity) actually matches
  the project before using it; a mismatch means the material is about
  the WRONG thing — say so rather than substituting a look-alike
- Write structured text that can be used as a deliverable or by the next worker"""

    model_key = step.model if step.model in _TIER_BY_KEY else "haiku"
    if step.action == StepAction.SYNTHESIZE:
        model_key = "sonnet"

    max_tokens = _COMPOSITION_MAX_TOKENS

    # Workers-as-CRYS (ratified 2026-07-13, Q1A/Q5A): behind the flag,
    # composition steps run as bounded agent sessions — the worker can
    # react to a 404, pivot on an empty result, verify before
    # asserting (rematch #9's decision-shaped residue). ANY failure of
    # the agentic path — timeout, loop error, empty output — falls
    # through to the classic single-call path below: the new machinery
    # can never lose an attempt.
    try:
        from ..config import get_settings
        _agentic_on = bool(
            getattr(get_settings(), "cognition_agentic_workers", False)
        )
    except Exception:  # noqa: BLE001
        _agentic_on = False
    if _agentic_on:
        try:
            from .agentic import run_agentic_composition
            agentic = await run_agentic_composition(
                env=env, step=step, prompt=prompt,
                store=store, fact_store=fact_store, encoder=encoder,
            )
            if (agentic.get("content") or "").strip():
                env.record_event("agentic_step", step_id=step.id,
                                 iterations=agentic.get("iterations"),
                                 tool_calls=len(
                                     agentic.get("tool_calls") or []))
                result.output = {
                    "content": agentic["content"],
                    "agentic": True,
                    "tool_calls": agentic.get("tool_calls") or [],
                    "iterations": agentic.get("iterations"),
                }
                result.model_used = str(agentic.get("model") or "agent")
                result.status = StepStatus.COMPLETE
                result.completed_at = datetime.now(timezone.utc)
                return result
            logger.warning("cognition.agentic_empty_falling_back",
                           step_id=step.id)
        except Exception as e:  # noqa: BLE001
            logger.warning("cognition.agentic_failed_falling_back",
                           step_id=step.id, error=str(e)[:200],
                           error_type=type(e).__name__)
            env.record_event("agentic_fallback", step_id=step.id,
                             error=str(e)[:160])

    client = get_llm_client()
    tier = _TIER_BY_KEY[model_key]
    total_in = 0
    total_out = 0

    async def _meter(llm_r) -> None:
        # Both ledgers, PER CALL: the env's UI estimate and the
        # authoritative record_model_call rows. Retry + continuation
        # calls are real spend and must each land a row.
        nonlocal total_in, total_out
        total_in += llm_r.input_tokens or 0
        total_out += llm_r.output_tokens or 0
        env.record_tokens(llm_r.input_tokens or 0, llm_r.output_tokens or 0,
                          model_key)
        await record_model_call(
            customer_id=env.customer_id,
            model=llm_r.model,
            input_tokens=llm_r.input_tokens,
            output_tokens=llm_r.output_tokens,
            cache_read_tokens=llm_r.cache_read_tokens,
            cache_creation_tokens=llm_r.cache_creation_tokens,
            origin="cognition",
            session_id=env.id,
        )

    # Q2A retry: one same-prompt retry on empty output or exception
    # before the step goes FAILED (rematch #6 attempt 3: an 85.9s
    # transient empty synthesize cost the whole attempt).
    llm = None
    for _try in range(_COMPOSITION_LLM_TRIES):
        try:
            llm = await asyncio.to_thread(
                client.complete_detailed,
                system=None,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=1.0,
                tier=tier,
            )
        except Exception as e:  # noqa: BLE001
            if _try + 1 >= _COMPOSITION_LLM_TRIES:
                raise
            logger.warning("cognition.composition_call_failed_retrying",
                           step_id=step.id, error=str(e))
            continue
        await _meter(llm)
        if (llm.text or "").strip():
            break
        logger.warning("cognition.composition_empty_retrying",
                       step_id=step.id, attempt=_try + 1)
        env.record_event("composition_empty_retry", step_id=step.id,
                         attempt=_try + 1)

    # 2026-07-11 (rematch #5, attempt 2): a composition call that
    # returns EMPTY text (after the retry above) is a step FAILURE, not
    # a completion — an empty synthesize must not sail to the validator.
    # FAILED status makes the hole visible: downstream steps see the
    # explicit FAILED marker and the tracker shows where the run died.
    if llm is None or not (llm.text or "").strip():
        result.status = StepStatus.FAILED
        result.error = "model returned empty output"
        result.tokens_in = total_in
        result.tokens_out = total_out
        result.model_used = model_key
        return result

    content = llm.text

    # Q1A continuation (rematch #6): the format step built the real
    # report and hit the output ceiling MID-SENTENCE — the validator
    # correctly rejected an amputated deliverable. When the stop reason
    # is max_tokens, continue the SAME assistant turn (partial text +
    # an explicit continue instruction; provider-neutral shape) and
    # concatenate, up to _COMPOSITION_MAX_CONTINUATIONS extra calls.
    continuations = 0
    while (
        getattr(llm, "stop_reason", None) == "max_tokens"
        and continuations < _COMPOSITION_MAX_CONTINUATIONS
    ):
        continuations += 1
        llm = await asyncio.to_thread(
            client.complete_detailed,
            system=None,
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": content},
                {"role": "user", "content": (
                    "Continue EXACTLY where you left off. Do not repeat "
                    "anything already written, do not add a preamble — "
                    "resume mid-sentence if needed and run to completion."
                )},
            ],
            max_tokens=max_tokens,
            temperature=1.0,
            tier=tier,
        )
        await _meter(llm)
        piece = (llm.text or "")
        if not piece.strip():
            break
        content = content + piece
    if continuations:
        logger.info("cognition.composition_continued",
                    step_id=step.id, continuations=continuations,
                    chars=len(content))
        env.record_event("composition_continued", step_id=step.id,
                         continuations=continuations, chars=len(content))

    result.tokens_in = total_in
    result.tokens_out = total_out
    result.model_used = model_key

    result.output = {"content": content}
    result.status = StepStatus.COMPLETE

    if step.action == StepAction.FORMAT:
        result.output["is_deliverable"] = True

    return result


# ---------------------------------------------------------------------------
# VALIDATOR
# ---------------------------------------------------------------------------

# Validator sizing (2026-07-09, the video-infra research run): the old
# max_tokens=1500 truncated Sonnet's per-criterion JSON on goals with
# large criteria sets — unbalanced braces, parse failure, fail-closed
# reject, three identical retries ("Failed after 3 attempts"). And the
# 4000-CHAR deliverable window meant a 7KB report was judged on its
# first half. Ceilings, not behavior, were the bug; fail-closed stays.
_VALIDATOR_MAX_TOKENS = 4000
# 2026-07-13 (rematch #7): 24000 made the validator reject its own
# truncated VIEW — long continued reports were judged "cut off
# mid-section / Section 3 missing / no sources" because the validator
# read the first 24K chars of a 30-60K-char deliverable. Same disease
# as the composer's old [:4000], one seat over. The validator must see
# the WHOLE document to judge completeness; deliverables beyond even
# this window get the envelope loop below (never truncate — add an
# envelope, same philosophy as the composition continuation loop).
_VALIDATOR_DELIVERABLE_CHARS = 120_000
_VALIDATOR_ENVELOPE_DIGEST_TOKENS = 1500
# Same disease, two more sites (2026-07-09): the orchestrator's
# goal+plan JSON truncated at 2000 tokens on large tasks (parse fail →
# bank-only fallback plan ×3 attempts), and composition steps — format
# WRITES THE FINAL DELIVERABLE — were capped at 1500 tokens, which is
# why attempt 1's report "terminates mid-sentence".
_ORCHESTRATOR_MAX_TOKENS = 4000
# 2026-07-13 (ratified): ONE flat cap for every composition call —
# 16000 output tokens (a cap is blast radius, not a target; you pay
# only for what's generated). This DELETED the orchestrator budget
# proposal, the floor, the clamp, and the escalation ladder: four
# mechanisms fighting one number whose root problem was that the
# orchestrator cannot know how many tokens a thinking model spends
# before writing. Adaptive-thinking models get room; the continuation
# loop still covers reports beyond even this.
_COMPOSITION_MAX_TOKENS = 16000
# Revision-aware retry sizing (2026-07-10, ratified Q3A):
#   _REVISION_DELIVERABLE_CHARS — the rejected deliverable, head+tail
#     trimmed, injected into the retry's orchestrator + composition
#     prompts (Q3A: 8,000 chars).
#   _PRIOR_CONTEXT_MAX_CHARS — the composition prompt's source-material
#     window. The old inline [:4000] silently starved the composer of
#     findings on research-scale tasks (the rematch's placeholder
#     reports). Rematch #4 (2026-07-11) proved 16000 was STILL too
#     small AND blind: dependency order let 10K of gated-off bank noise
#     eat the window and truncate the emerging-projects findings to
#     zero — a report failed for data that was sitting in step outputs.
#     Raised to 48000 (~12K tokens; pennies on the composition tiers vs
#     a wasted Sonnet retry) and allocated MAX-MIN FAIR across parts
#     (_fair_share_allocations) so no dependency can starve another —
#     the no-starvation principle applied to the prompt itself.
# Q1A continuation (2026-07-11, rematch #6): a composition call cut at
# max_tokens continues (assistant partial + "continue exactly") up to
# this many extra calls — reports get to be as long as the task needs;
# the per-plan budget becomes per-CALL, this caps total calls.
_COMPOSITION_MAX_CONTINUATIONS = 3
# Q2A (2026-07-11, rematch #6): one same-prompt retry before an
# empty-output/exception composition call goes FAILED — attempt 3's
# synthesize returned empty after 85.9s (transient), and one glitch
# should not cost an entire attempt.
_COMPOSITION_LLM_TRIES = 2
_REVISION_DELIVERABLE_CHARS = 8000
_PRIOR_CONTEXT_MAX_CHARS = 48000


def _trim_head_tail(text: str, limit: int) -> str:
    """Head+tail trim: keep the opening and the ending, elide the middle.
    A rejected deliverable's structure (head) and conclusion (tail) carry
    most of the revision signal; the middle is what the trimming spends."""
    if len(text) <= limit:
        return text
    half = max(1, (limit - 40) // 2)
    return (
        text[:half]
        + "\n\n[... middle elided for length ...]\n\n"
        + text[-half:]
    )


async def _digest_envelopes(
    env: CognitionEnvironment,
    deliverable_text: str,
    criteria_block: str,
) -> str:
    """Map an oversized deliverable into per-envelope digests the final
    verdict call can judge (2026-07-13). Each envelope call reports
    which sections exist and what criteria evidence appears IN THAT
    PART — presence/absence judgments then happen over the union, so a
    complete 200K-char report is never rejected for what the window
    couldn't show."""
    chunks = [
        deliverable_text[i:i + _VALIDATOR_DELIVERABLE_CHARS]
        for i in range(0, len(deliverable_text), _VALIDATOR_DELIVERABLE_CHARS)
    ]
    digests: list[str] = []
    for i, chunk in enumerate(chunks, 1):
        dprompt = f"""You are assisting a quality validator. This is PART {i} of {len(chunks)} of one deliverable (split only for length). The acceptance criteria for the WHOLE deliverable:
{criteria_block}

PART {i}/{len(chunks)}:
{chunk}

Report tersely, as plain text:
1. Which sections/headings appear in this part.
2. Concrete evidence in this part relevant to each criterion (cite the detail, e.g. version numbers, URLs, word counts).
3. Quality problems visible in this part (unsupported claims, placeholders).
Do NOT judge completeness of the whole deliverable — other parts exist."""
        llm = await asyncio.to_thread(
            get_llm_client().complete_detailed,
            system=None,
            messages=[{"role": "user", "content": dprompt}],
            max_tokens=_VALIDATOR_ENVELOPE_DIGEST_TOKENS,
            temperature=1.0,
            tier=_TIER_BY_KEY["sonnet"],
        )
        env.record_tokens(llm.input_tokens or 0, llm.output_tokens or 0,
                          "sonnet")
        await record_model_call(
            customer_id=env.customer_id,
            model=llm.model,
            input_tokens=llm.input_tokens,
            output_tokens=llm.output_tokens,
            cache_read_tokens=llm.cache_read_tokens,
            cache_creation_tokens=llm.cache_creation_tokens,
            origin="cognition",
            session_id=env.id,
        )
        digests.append(f"--- DIGEST OF PART {i}/{len(chunks)} ---\n{llm.text}")
    logger.info("validator.envelopes_digested", env_id=env.id,
                parts=len(chunks), chars=len(deliverable_text))
    env.record_event("validator_envelopes", parts=len(chunks),
                     chars=len(deliverable_text))
    return (
        "(The deliverable is "
        f"{len(deliverable_text)} characters — longer than one validation "
        "window. Below are structured digests of each part, produced by "
        "an assistant that saw the full text. Judge completeness and "
        "criteria coverage over the UNION of the parts.)\n\n"
        + "\n\n".join(digests)
    )


async def run_validator(
    env: CognitionEnvironment,
) -> ValidationResult:
    """Validator: compares deliverables against the goal contract.

    Sees: goal.json, deliverables/
    Writes: validation.json
    Never sees: plan.json, step outputs
    """
    goal = env.goal
    if not goal:
        return ValidationResult(
            approved=False, score=0.0,
            reasoning="No goal document found",
            model_used="none",
        )

    deliverable_text = env.get_final_deliverable() or "(No deliverable produced)"

    criteria_block = "\n".join(
        f"  {i+1}. {c}" for i, c in enumerate(goal.acceptance_criteria)
    )

    # Contract amendment audit (2026-07-14, Q2A): when earlier attempts
    # amended criteria on evidence, the validator judges against the
    # CURRENT criteria and can SEE that (and why) they were amended —
    # the bend is never silent.
    _amendments_block = ""
    if getattr(goal, "amendments", None):
        _amend_lines = [
            (f"  - criterion {a.get('index', '?') + 1 if isinstance(a.get('index'), int) else '?'}: "
             f"\"{a.get('original', '')}\" -> \"{a.get('amended', '')}\" "
             f"(evidence: {str(a.get('evidence', ''))[:200]})")
            for a in goal.amendments
        ]
        _amendments_block = (
            "  CONTRACT AMENDMENTS (applied on documented evidence in "
            "earlier attempts — judge against the CURRENT criteria "
            "above):\n" + "\n".join(_amend_lines) + "\n"
        )

    # Envelope loop (2026-07-13, ratified): a deliverable longer than
    # the window is NEVER truncated — like the composition continuation
    # loop, we add envelopes. Each envelope gets a digest call ("what
    # sections/criteria evidence appears in THIS part"), and the final
    # verdict call judges the digests. Deliverables that fit take the
    # single-call path unchanged.
    deliverable_block = deliverable_text
    if len(deliverable_text) > _VALIDATOR_DELIVERABLE_CHARS:
        deliverable_block = await _digest_envelopes(
            env, deliverable_text, criteria_block,
        )

    _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"""You are a quality validator. TODAY'S DATE IS {_today} (UTC) — judge date plausibility against it, not against your training data.
You evaluate deliverables against a goal contract.
You have NO knowledge of how the work was done. You only see what was asked for and what was produced.

GOAL CONTRACT:
  Title: {goal.title}
  Description: {goal.description}
  Acceptance Criteria:
{criteria_block}
{_amendments_block}
DELIVERABLE:
{deliverable_block[:_VALIDATOR_DELIVERABLE_CHARS + 20_000]}

For each acceptance criterion, evaluate:
- MET: the deliverable clearly satisfies this criterion
- PARTIALLY_MET: partially addressed but with gaps
- NOT_MET: the criterion is not satisfied

Respond with ONLY valid JSON:
{{
  "approved": true or false,
  "score": 0.0 to 1.0,
  "reasoning": "overall assessment",
  "criteria_evaluation": [
    {{"criterion": "...", "status": "MET|PARTIALLY_MET|NOT_MET", "evidence": "...", "possibly_infeasible": false}}
  ],
  "issues": ["specific issue 1", "..."],
  "suggestions": ["improvement 1", "..."]
}}

Rules:
- APPROVED if all criteria MET or PARTIALLY_MET with minor gaps and score >= 0.7
- REJECTED if any criterion NOT_MET or score < 0.7
- Be strict about hallucination: claims not in the deliverable's source material are failures
- "possibly_infeasible": true ONLY when the deliverable DOCUMENTS real search breadth (the queries run, the sources fetched, the candidates examined) and that evidence suggests the criterion may be unsatisfiable AS WRITTEN — e.g. the world may not contain the demanded count. Absence of effort is NEVER infeasibility; an undocumented "couldn't find any" gets NOT_MET with possibly_infeasible false. The flag does not soften your verdict — status stays NOT_MET; the flag only licenses the next attempt's planner to propose an evidence-based contract amendment.
- Your issues must be specific enough for a planner to create a better plan
- Keep each reasoning string under 30 words; be terse — the JSON must be complete and well-formed"""

    t0 = time.time()
    llm = await asyncio.to_thread(
        get_llm_client().complete_detailed,
        system=None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_VALIDATOR_MAX_TOKENS,
        temperature=1.0,
        tier=_TIER_BY_KEY["sonnet"],
    )
    duration_ms = int((time.time() - t0) * 1000)

    raw = llm.text
    tokens_in = llm.input_tokens or 0
    tokens_out = llm.output_tokens or 0
    env.record_tokens(tokens_in, tokens_out, "sonnet")
    await record_model_call(
        customer_id=env.customer_id,
        model=llm.model,
        input_tokens=llm.input_tokens,
        output_tokens=llm.output_tokens,
        cache_read_tokens=llm.cache_read_tokens,
        cache_creation_tokens=llm.cache_creation_tokens,
        origin="cognition",
        session_id=env.id,
    )

    data = _extract_json_object(raw)
    if data is None:
        logger.warning(
            "validator.json_parse_failed",
            raw_head=raw[:200],
            raw_tail=raw[-200:],
            raw_len=len(raw),
        )
        # Fail CLOSED. An unparseable validator response is not an
        # approval. The previous behavior approved whenever the
        # deliverable was longer than 100 chars, which let a
        # process-report "deliverable" through and committed a garbage
        # crystal (idle-log incident, 2026-06-08). Without a valid
        # evaluation we cannot certify the contract was met, so reject
        # and let the orchestrator retry with the parse-failure note.
        data = {
            "approved": False,
            "score": 0.0,
            "reasoning": "Validator response could not be parsed; failing closed (no approval without a valid evaluation).",
            "criteria_evaluation": [],
            "issues": ["Validator JSON parse failure — could not evaluate the deliverable."],
            "suggestions": ["Ensure the validator returns a single valid JSON object with no surrounding prose."],
        }

    result = ValidationResult(
        approved=data.get("approved", False),
        score=data.get("score", 0.0),
        reasoning=data.get("reasoning", ""),
        criteria_evaluation=[
            CriterionEval(
                criterion=c.get("criterion", ""),
                status=c.get("status", "NOT_MET"),
                evidence=c.get("evidence", ""),
                possibly_infeasible=bool(c.get("possibly_infeasible", False)),
            )
            for c in data.get("criteria_evaluation", [])
        ],
        issues=data.get("issues", []),
        suggestions=data.get("suggestions", []),
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        model_used="sonnet",
    )

    logger.info(
        "validator.complete",
        env_id=env.id,
        approved=result.approved,
        score=result.score,
        issues=len(result.issues),
        duration_ms=duration_ms,
    )

    return result
