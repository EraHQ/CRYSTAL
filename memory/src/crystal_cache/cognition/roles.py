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

async def run_orchestrator(
    env: CognitionEnvironment,
    store: "MetadataStore",
    fact_store: "FactVectorStore",
) -> tuple[GoalDocument, Plan]:
    """Orchestrator: reviews the trigger, creates goal contract + execution plan.

    Sees: trigger context, rejection history (on retry)
    Writes: goal.json, plan.json
    Never sees: step outputs, deliverables
    """
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

    prompt = f"""You are a research orchestrator. You receive a goal and must produce:
1. A GOAL DOCUMENT (contract for the validator)
2. An EXECUTION PLAN (instructions for workers)

TASK: {env.task_goal or env.conversation_context or "No additional context"}

CONTEXT: {(env.conversation_context[:600] if env.task_goal else "") or "None"}

SOURCE CRYSTAL: {env.source_crystal_id or "None specified"}
OUTPUT TYPE: {env.output_type.value}
{rejection_context}

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
        "action": "crystal_search|crystal_key_scan|web_search|analyze|synthesize|format",
        "description": "what this step does",
        "input": {{"query": "...", "instruction": "..."}},
        "depends_on": [],
        "parallel_group": "A or null"
      }}
    ],
    "expected_output": "what the final deliverable should look like",
    "suggested_key": "wide|...|specific unified sparse key (general to specific)",
    "parent_crystal_id": "{env.source_crystal_id}"
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
- crystal_search or crystal_key_scan always comes first. Check existing knowledge.
- Use crystal_key_scan when the task involves COUNTING or LISTING items (scenes, chapters, etc.)
- Read-only steps (crystal_search, crystal_key_scan, web_search, source_lookup) can share a parallel_group.
- Write steps (analyze, synthesize, format) must have parallel_group: null.
- Maximum 5 steps.
- acceptance_criteria must be specific and testable.
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

    plan = Plan(
        reasoning=plan_data.get("reasoning", ""),
        steps=steps,
        expected_output=plan_data.get("expected_output", ""),
        suggested_key=plan_data.get("suggested_key", ""),
        parent_crystal_id=plan_data.get("parent_crystal_id", env.source_crystal_id),
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

    Sees: plan, prior step outputs, read-only resources
    Writes: its own step output
    Never sees: goal.json, validation.json

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
        if step.action in COMPOSITION_ACTIONS:
            # Composition actions stay cognition-only — see D-A10 +
            # §6.5.3. Each composition action reads `prior_context`
            # built from dependency step outputs, which is the shape
            # cognition's plan-execution model assumes; agent-side
            # llm_invoke doesn't have that shape.
            result = await _worker_llm_step(env, step, result)
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

    findings = []
    seen = set()
    for fact_id, crystal_id, pair_type, score in search_results[:8]:
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


def _assemble_prior_context(env: CognitionEnvironment, step: PlanStep) -> str:
    """Build the source-material block a composition step reads.

    Per-dependency: prefer content_text/content; else render findings
    via _finding_to_text. A FAILED dependency contributes an explicit
    marker (downstream models must KNOW a step died — the rematch's
    synthesize/format steps received silence and correctly refused to
    fabricate, but with the marker they can also say WHICH input is
    missing). Every piece is None-proofed before joining."""
    parts: list[str] = []
    for dep_id in step.depends_on:
        dep_output = env.step_outputs.get(dep_id)
        if dep_output is None:
            continue
        if dep_output.status == StepStatus.COMPLETE:
            content = (
                dep_output.output.get("content_text")
                or dep_output.output.get("content")
                or ""
            )
            if not content:
                findings = dep_output.output.get("findings") or []
                pieces = [t for t in (_finding_to_text(f) for f in findings) if t]
                content = "\n\n".join(pieces)
            if content:
                parts.append(f"--- Step {dep_id} output ---\n{content}")
        elif dep_output.status == StepStatus.FAILED:
            parts.append(
                f"--- Step {dep_id} FAILED: "
                f"{dep_output.error or 'no error recorded'} ---"
            )
    return "\n\n".join(parts) if parts else "(No prior output available)"


async def _worker_llm_step(
    env: CognitionEnvironment,
    step: PlanStep,
    result: StepOutput,
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

    prompt = f"""You are a research worker executing step {step.id}.

YOUR TASK: {step.description}

INSTRUCTION: {instruction}

PRIOR WORK (from previous steps):
{prior_context[:4000]}

Rules:
- Work ONLY on your assigned task
- Base your output on the source material provided, not your own knowledge
- If source material doesn't contain what you need, say so explicitly
- Document your reasoning
- Write structured text that can be used as a deliverable or by the next worker"""

    model_key = step.model if step.model in _TIER_BY_KEY else "haiku"
    if step.action == StepAction.SYNTHESIZE:
        model_key = "sonnet"

    llm = await asyncio.to_thread(
        get_llm_client().complete_detailed,
        system=None,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=_COMPOSITION_MAX_TOKENS,
        temperature=1.0,
        tier=_TIER_BY_KEY[model_key],
    )

    content = llm.text
    result.tokens_in = llm.input_tokens or 0
    result.tokens_out = llm.output_tokens or 0
    result.model_used = model_key
    env.record_tokens(result.tokens_in, result.tokens_out, model_key)
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
_VALIDATOR_DELIVERABLE_CHARS = 24000
# Same disease, two more sites (2026-07-09): the orchestrator's
# goal+plan JSON truncated at 2000 tokens on large tasks (parse fail →
# bank-only fallback plan ×3 attempts), and composition steps — format
# WRITES THE FINAL DELIVERABLE — were capped at 1500 tokens, which is
# why attempt 1's report "terminates mid-sentence".
_ORCHESTRATOR_MAX_TOKENS = 4000
_COMPOSITION_MAX_TOKENS = 4000


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

    prompt = f"""You are a quality validator. You evaluate deliverables against a goal contract.
You have NO knowledge of how the work was done. You only see what was asked for and what was produced.

GOAL CONTRACT:
  Title: {goal.title}
  Description: {goal.description}
  Acceptance Criteria:
{criteria_block}

DELIVERABLE:
{deliverable_text[:_VALIDATOR_DELIVERABLE_CHARS]}

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
    {{"criterion": "...", "status": "MET|PARTIALLY_MET|NOT_MET", "evidence": "..."}}
  ],
  "issues": ["specific issue 1", "..."],
  "suggestions": ["improvement 1", "..."]
}}

Rules:
- APPROVED if all criteria MET or PARTIALLY_MET with minor gaps and score >= 0.7
- REJECTED if any criterion NOT_MET or score < 0.7
- Be strict about hallucination: claims not in the deliverable's source material are failures
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
