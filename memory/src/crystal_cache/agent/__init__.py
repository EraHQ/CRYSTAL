"""Agent package — Phase 7.5 of the v2 port.

Crystal Cache's flagship surface per the agent reframe (D-A1 through
D-A10 in docs/AGENT_ARCHITECTURE.md). The agent is a conversational
entity with a flat tool surface; one of those tools is the upstream
LLM (`llm_invoke`), but the LLM is a peer among Memory (mem0_*,
crystal_*), Planning (cognition_run), Retrievers (the four V3
routers exposed flat), and external services.

Module structure (per §5 of the design doc):

    agent/
    ├── __init__.py          (this file — public surface)
    ├── agent.py             (conversation loop)
    ├── system_prompt.py     (registry-derived prompt builder)
    ├── tool_registry.py     (single source of truth for tools)
    ├── mcr_emitter.py       (Phase 9A — trace + self-critique emission)
    ├── shadow_critic.py     (Phase 9.5 — shadow critic, sampled second voice)
    ├── tools/
    │   ├── retrievers.py    (content_search, knowledge_search, ...)
    │   ├── memory.py        (mem0_*, crystal_*)
    │   ├── llm.py           (llm_invoke)
    │   ├── cognition.py     (cognition_run)
    │   └── external.py      (web_search, document_upload, decompose)
    └── adapters/
        ├── anthropic.py     (full, used by the agent loop)
        ├── openai.py        (skeleton — wire format only for now)
        └── mcp.py           (skeleton — wire format only for now)

Public exports (the surface importers should depend on):

- `Agent`: the conversation loop class. One instance per request.
- `build_system_prompt`: registry-derived prompt builder.
- `get_registry`, `ToolRegistry`, `Tool`: the tool registry and
  its types. The decorator `register_tool` is exported too in case
  customers add their own tools at runtime.
- `set_tool_state`: state injection for the tool implementations.
  The Agent class calls this automatically; tests can call it
  manually to override individual dependencies.
- `import_all_tools`: trigger registration of every built-in tool.
  Called automatically when an Agent is constructed; exposed
  publicly so tests / scripts can pre-warm the registry.
- `emit_mcr_artifacts` (Phase 9A): persist trace + self-critique
  + action items for one agent turn. Called from endpoint
  handlers after `Agent.run(...)` returns.
- `shadow_review_trace`, `ShadowSamplingPolicy`, `run_shadow_critique`
  (Phase 9.5): the MCR shadow critic. `shadow_review_trace` is the
  manual/triggered entry point that loads a persisted trace + its
  agent_self critique and persists a Critique(critic_role="shadow").
  The automatic scheduling that calls it on a cadence is Phase 10's
  metacognitive-layer job.

WAVE 7E + 7F LIBRARY READINESS:
The agent depends on the full Wave 7B + 7C + 7D + 7E + 7F surface
working as a library. Specifically: the four V3 routers (Wave 7A),
the V2 retrieval pipeline (Wave 7B — for crystal_recall),
upstream_client (Wave 7C — for llm_invoke), the bonder + store
methods (Wave 7D — for crystal_write), the LearningService and
ConsolidationService (Wave 7E — for the SDK endpoints, but the
agent doesn't call them directly), and the Mem0 session-memory
module (Wave 7F — for mem0_recall/mem0_write).

CROSS-REFERENCE TO MCR (Phase 9):
Phase 7.5 built the agent. **Phase 9A (2026-05-27) added the MCR
trace + self-critique emission via `mcr_emitter.py`**, called from
`endpoints/agent.py` after `Agent.run(...)` returns. Phase 9B
extended emission to the push/pull signal handler (resolves BD-3
and BD-11 per P0.42 + P0.43); Phase 9C extended emission to the
chat_proxy after characterization tests landed per CU-17.
**Phase 9.5 (2026-05-27) added the shadow critic via
`shadow_critic.py`** — the sampled second critic per D-MCR-10 +
MCR §5.2, reviewing persisted traces + their self-critiques and
emitting Critique(critic_role="shadow"). The shadow is ADDITIVE to
(not a replacement for) `execution/shadow_evaluator.py`, which is
v1's orthogonal response-quality delta still wired into chat_proxy.

CROSS-REFERENCE TO COGNITION REFACTOR (§6.5.5):
The cognition package (`cognition/roles.py::run_worker`) gets a
companion refactor in Phase 7.5 to delegate non-composition actions
to the shared tool registry. That refactor reads from this
package's tool_registry (via `get_registry().get_by_cognition_action`)
and lives in `cognition/roles.py`, not here.
"""
from .agent import Agent
from .mcr_emitter import emit_mcr_artifacts
from .shadow_critic import (
    ShadowSamplingPolicy,
    run_shadow_critique,
    shadow_review_trace,
)
from .system_prompt import build_system_prompt
from .tool_registry import (
    Tool,
    ToolRegistry,
    get_registry,
    import_all_tools,
    register_tool,
    reset_registry,
)

__all__ = [
    "Agent",
    "build_system_prompt",
    "emit_mcr_artifacts",
    "ShadowSamplingPolicy",
    "run_shadow_critique",
    "shadow_review_trace",
    "Tool",
    "ToolRegistry",
    "get_registry",
    "import_all_tools",
    "register_tool",
    "reset_registry",
]
