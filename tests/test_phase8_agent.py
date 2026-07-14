"""Phase 8 smoke tests for the agent package.

Per the locked Phase 8 decisions (P0.29–P0.32):

  Test 1: agent package imports cleanly + all tools register
  Test 2: Agent loop with simple "hi" → 0 tool calls, end_turn
  Test 3: Agent loop with one tool call (knowledge_search empty bank)
  Test 4: Agent loop with a write (crystal_write) then final answer
  Test 5: Unknown tool name → is_error=True surfaced to LLM
  Test 6: System prompt contains all registered agent tools
  Test 7: Tool dispatch sanitizes customer_id from LLM input
  Test 8: Multiple tool_use blocks in one response → all dispatched

Cognition §6.5.5 refactor tests live in tests/test_phase8_cognition_refactor.py.

ALL tests use in-memory SQLite (P0.30) and the FakeAnthropic client
(P0.31). None of these tests make a real network call.

R14 note: every assertion below corresponds to a runtime check
performed by `pytest`. These tests have not yet been run; the
assertions describe expected behavior. Whether they pass is verified
in the runtime + ledger update at the end of Phase 8 execution.
"""
from __future__ import annotations

from typing import Any

import pytest

from crystal_cache.agent import (
    Agent,
    Tool,
    build_system_prompt,
    get_registry,
    import_all_tools,
)


# ===========================================================================
# Test 1 — Imports + registrations
# ===========================================================================

def test_agent_package_imports_and_all_tools_register():
    """Importing the agent package + calling import_all_tools() must
    populate the registry with the full Phase 7.5 tool surface.

    The expected tool set is the 19 tools defined across
    agent/tools/*.py per D-A3/D-A5/D-A6/§4.1/§4.6/§6.5.5/P4d, plus the
    three curation tools promoted into the registry in WS C (the MCP
    memory server exposes the same impls as memory_learn /
    memory_conflicts / memory_gaps).
    """
    import_all_tools()
    registry = get_registry()

    expected = {
        # D-A3 — four V3 routers as flat tools
        "content_search",
        "knowledge_search",
        "navigation_search",
        "depth_search",
        # B / §6.5.5 — enumeration primitive (agent+cognition since 2026-06-11)
        "key_scan",
        # D-A5 — split memory tools
        "mem0_recall",
        "mem0_write",
        "crystal_recall",
        "crystal_write",
        # §4.1
        "llm_invoke",
        # D-A6
        "cognition_run",
        "cognition_status",
        # §4.6
        "web_search",
        # 2026-07-07 — the browsing half of the search+fetch pair:
        # web_search discovers, web_fetch reads one URL through the
        # same SSRF-guarded fetcher (agent+cognition).
        "web_fetch",
        "document_upload",
        "decompose",
        # VS-D5 — read-only source access for cognition (cognition-context)
        "source_lookup",
        # P4d — document artifact generation (agent-only; CRYS's web
        # surface renders the result as a download/preview card)
        "create_document",
        # WS C — curation tools promoted into the registry (also exposed on
        # the MCP memory surface as memory_learn / memory_conflicts /
        # memory_gaps). crystal_learn is write-side (agent-only); the two
        # reads are agent+cognition.
        "crystal_learn",
        "knowledge_conflicts",
        "knowledge_gaps",
    }
    actual = set(registry.names())
    missing = expected - actual
    extra = actual - expected
    assert not missing, f"missing tools: {missing}"
    assert not extra, f"unexpected tools: {extra}"

    # Spot-check context assignments per D-A10.
    knowledge = registry.get("knowledge_search")
    assert knowledge is not None
    assert knowledge.contexts == frozenset({"agent", "cognition"})

    llm = registry.get("llm_invoke")
    assert llm is not None
    assert llm.contexts == frozenset({"agent"})  # agent-only

    cog = registry.get("cognition_run")
    assert cog is not None
    assert cog.contexts == frozenset({"agent"})  # no recursion

    # Verify cognition_action_alias mapping per P0.26 + B (§6.5.5).
    assert knowledge.cognition_action_alias == "crystal_search"
    # Post-B, crystal_key_scan resolves to the key_scan enumeration
    # tool, not navigation_search — which is now overview-only and no
    # longer carries a cognition alias.
    nav = registry.get("navigation_search")
    assert nav is not None
    assert nav.cognition_action_alias is None
    key_scan = registry.get("key_scan")
    assert key_scan is not None
    assert key_scan.cognition_action_alias == "crystal_key_scan"
    # 2026-06-11: key_scan widened to agent+cognition — the coding
    # agent's bank demo surfaced an identity query ('what does <file>
    # define') that resemblance top-matching answers incompletely; the
    # agent needs the raw enumeration primitive. This exercised the
    # documented seam ("widen contexts to add 'agent' if the agent ever
    # needs raw enumeration").
    assert key_scan.contexts == frozenset({"agent", "cognition"})
    # VS-D5 source_lookup: cognition-context, aliased to its own name.
    src = registry.get("source_lookup")
    assert src is not None
    assert src.cognition_action_alias == "source_lookup"
    assert src.contexts == frozenset({"cognition"})
    # P4d — create_document is agent-only (generation lives on the agent
    # surface, not in cognition workers) and carries no cognition alias.
    create_doc = registry.get("create_document")
    assert create_doc is not None
    assert create_doc.contexts == frozenset({"agent"})
    assert create_doc.cognition_action_alias is None


# ===========================================================================
# Test 2 — Simple "hi" → no tool calls
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_simple_greeting_no_tool_calls(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """The agent's controlling LLM returns a text-only response.
    The agent loop should produce final_text after one iteration,
    stop_reason='end_turn', and zero tool calls in the log.
    """
    fake_anthropic.script_text("Hi! How can I help you today?")

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "hi"}],
    )

    assert result["final_text"] == "Hi! How can I help you today?"
    assert result["stop_reason"] == "end_turn"
    assert result["iterations"] == 1
    assert result["tool_calls"] == []
    # Token accumulation should reflect the single model call.
    assert result["prompt_tokens"] == 100
    assert result["completion_tokens"] == 50

    # Verify the fake saw exactly one call with the expected shape.
    call = fake_anthropic.assert_called_once()
    assert call["model"] == "claude-sonnet-4-5-20250929"
    # H1 (2026-06-13): default output budget raised 4096 -> 8192 (settings-
    # tunable via CC_AGENT_MAX_TOKENS) so a large write can't truncate.
    assert call["max_tokens"] == 8192
    # C1: system is now a cached content-block list, not a bare string.
    assert call["system"] is not None
    assert isinstance(call["system"], list)
    assert "Crystal Cache" in call["system"][0]["text"]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}


# ===========================================================================
# Test 3 — One tool call (knowledge_search on empty bank)
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_one_tool_call_empty_bank(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """Agent emits a knowledge_search tool_use block, receives the
    (empty) result, then emits its final text answer. Two iterations
    total.
    """
    # Iteration 1: emit knowledge_search.
    fake_anthropic.script_tool_use(
        name="knowledge_search",
        input_dict={"query": "test query"},
        tool_use_id="tu_001",
    )
    # Iteration 2: produce final text based on the (empty) result.
    fake_anthropic.script_text(
        "I couldn't find anything about that in the bank.",
    )

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "what do we know about X?"}],
    )

    assert result["iterations"] == 2
    assert result["stop_reason"] == "end_turn"
    assert "couldn't find" in result["final_text"]

    # One tool call recorded.
    assert len(result["tool_calls"]) == 1
    tc = result["tool_calls"][0]
    assert tc["tool_name"] == "knowledge_search"
    assert tc["is_error"] is False
    assert tc["input"] == {"query": "test query"}
    # The output is a dict per P0.23 with the V3 router result shape.
    assert isinstance(tc["output"], dict)
    assert "injection_text" in tc["output"]
    assert "matched_fact_ids" in tc["output"]
    # Empty bank — no matches.
    assert tc["output"]["matched_fact_ids"] == []
    assert tc["output"]["fact_count"] == 0


# ===========================================================================
# Test 4 — Write a crystal then return final answer
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_crystal_write_then_final_text(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
    store: Any,
):
    """Agent emits a crystal_write, then a final text response. The
    write must actually land in the DB — verified via a direct store
    query post-run.
    """
    # Iteration 1: write a fact.
    fake_anthropic.script_tool_use(
        name="crystal_write",
        input_dict={
            "key": "What is the deadline?",
            "value": "April 1, 2027",
        },
        tool_use_id="tu_write_001",
    )
    # Iteration 2: final text.
    fake_anthropic.script_text(
        "Got it — I'll remember the deadline is April 1, 2027.",
    )

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{
            "role": "user",
            "content": "Remember: the deadline is April 1, 2027.",
        }],
    )

    assert result["iterations"] == 2
    assert result["stop_reason"] == "end_turn"
    assert "April 1, 2027" in result["final_text"]

    # Verify the write actually landed.
    crystals = await store.list_crystals_for_customer(customer.id)
    assert len(crystals) == 1
    facts = await store.list_facts_for_crystal(crystals[0].id)
    assert len(facts) == 1
    assert facts[0].prompt_text == "What is the deadline?"
    assert facts[0].claim_text == "April 1, 2027"
    assert facts[0].pair_type == "question_answer"

    # Tool call telemetry carries the crystal_id + fact_id.
    tc = result["tool_calls"][0]
    assert tc["tool_name"] == "crystal_write"
    assert tc["is_error"] is False
    assert tc["output"]["crystal_id"] == crystals[0].id
    assert tc["output"]["fact_id"] == facts[0].id


# ===========================================================================
# Test 5 — Unknown tool name → is_error surfaced
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_unknown_tool_name_returns_is_error(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """If the LLM hallucinates a tool name, the agent loop must
    surface `is_error: True` in the tool_result block (not crash).
    The agent then gets to recover on the next iteration.
    """
    # Iteration 1: call a nonexistent tool.
    fake_anthropic.script_tool_use(
        name="nonexistent_tool",
        input_dict={"foo": "bar"},
        tool_use_id="tu_bad_001",
    )
    # Iteration 2: agent recovers, emits text.
    fake_anthropic.script_text(
        "Sorry, I confused myself. Let me try again differently.",
    )

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "do something"}],
    )

    assert result["iterations"] == 2
    assert result["stop_reason"] == "end_turn"

    tc = result["tool_calls"][0]
    assert tc["tool_name"] == "nonexistent_tool"
    assert tc["is_error"] is True
    # The output should be an error string per Agent._dispatch_tool.
    assert isinstance(tc["output"], str)
    assert "nonexistent_tool" in tc["output"]
    assert "not registered" in tc["output"]

    # The next iteration's input messages must include the
    # tool_result block with is_error=True.
    second_call_messages = fake_anthropic.calls[1]["messages"]
    last_msg = second_call_messages[-1]
    assert last_msg["role"] == "user"
    assert isinstance(last_msg["content"], list)
    tool_results = [
        b for b in last_msg["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 1
    assert tool_results[0]["is_error"] is True
    assert tool_results[0]["tool_use_id"] == "tu_bad_001"


# ===========================================================================
# Test 6 — System prompt contains every registered agent tool
# ===========================================================================

def test_system_prompt_lists_all_agent_tools(customer: Any):
    """build_system_prompt should include every agent-context tool's
    name in the TOOLS section. This is the P0.24 contract — adding
    a new agent tool surfaces in the system prompt without manual
    edits.
    """
    import_all_tools()
    registry = get_registry()
    agent_tools = registry.list_for_context("agent")
    prompt = build_system_prompt(customer, agent_tools)

    # Every agent-context tool name must appear in the prompt body.
    for tool in agent_tools:
        assert tool.name in prompt, (
            f"tool {tool.name!r} not found in system prompt"
        )

    # The three required sections are present.
    assert "Crystal Cache" in prompt  # ROLE
    assert "TOOLS AVAILABLE" in prompt  # TOOLS header
    assert "POLICIES" in prompt  # POLICIES header

    # P0.20 cognition-delegation heuristic appears in POLICIES.
    assert "cognition_run" in prompt
    assert "save" in prompt.lower() or "deliverable" in prompt.lower()


# ===========================================================================
# Test 7 — Tool dispatch sanitizes customer_id from LLM input
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_dispatch_sanitizes_customer_id_from_llm_input(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """Per P0.23, the agent injects customer_id from the request
    context. An LLM that emits `customer_id` in tool_input must NOT
    be able to spoof another customer's identity.

    The Agent._dispatch_tool path strips `customer_id` from
    sanitized_input before calling the tool impl. We verify by
    scripting a tool_use that includes a fake customer_id, then
    inspect the actual write to confirm it landed under the real
    customer.
    """
    # The LLM tries to write under a different customer.
    fake_anthropic.script_tool_use(
        name="crystal_write",
        input_dict={
            "customer_id": "cus_attacker_does_not_exist",  # SPOOF
            "key": "spoofed key",
            "value": "spoofed value",
        },
        tool_use_id="tu_spoof_001",
    )
    fake_anthropic.script_text("Done.")

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "remember spoofed"}],
    )

    # Should succeed without is_error=True (the spoof attempt was
    # silently sanitized away, not surfaced as an error).
    assert result["iterations"] == 2
    tc = result["tool_calls"][0]
    assert tc["is_error"] is False
    # The actual write must have landed under the real customer.
    assert tc["output"]["crystal_id"]
    # Sanity check: list crystals for the spoof id returns nothing.
    crystals_real = await tool_state["store"].list_crystals_for_customer(
        customer.id,
    )
    crystals_spoof = await tool_state["store"].list_crystals_for_customer(
        "cus_attacker_does_not_exist",
    )
    assert len(crystals_real) == 1
    assert len(crystals_spoof) == 0


# ===========================================================================
# Test 8 — Multiple tool_use blocks in one response
# ===========================================================================

@pytest.mark.asyncio
async def test_agent_dispatches_multiple_parallel_tool_calls(
    customer: Any,
    tool_state: dict[str, Any],
    fake_anthropic: Any,
):
    """Anthropic supports parallel tool calling — one response with
    multiple tool_use blocks. The agent loop must dispatch all of
    them in one iteration and append a single user message with the
    N tool_result blocks.
    """
    # Iteration 1: two tool calls in one response.
    fake_anthropic.script_multi_tool_use(
        calls=[
            ("knowledge_search", {"query": "alpha"}, "tu_a"),
            ("navigation_search", {"query_text": "beta"}, "tu_b"),
        ],
    )
    fake_anthropic.script_text("Both lookups came back empty.")

    agent = Agent(
        customer=customer,
        llm=fake_anthropic,
        tool_state=tool_state,
    )
    result = await agent.run(
        messages=[{"role": "user", "content": "lookup alpha and beta"}],
    )

    assert result["iterations"] == 2
    # Both tool calls recorded in the same iteration.
    assert len(result["tool_calls"]) == 2
    iters = {tc["iteration"] for tc in result["tool_calls"]}
    assert iters == {1}, (
        f"both tool calls must be iteration 1, got {iters}"
    )
    names = {tc["tool_name"] for tc in result["tool_calls"]}
    assert names == {"knowledge_search", "navigation_search"}
    for tc in result["tool_calls"]:
        assert tc["is_error"] is False

    # The second model call's last message must be a user message
    # carrying two tool_result blocks.
    second_call = fake_anthropic.calls[1]
    last = second_call["messages"][-1]
    assert last["role"] == "user"
    tool_results = [
        b for b in last["content"]
        if isinstance(b, dict) and b.get("type") == "tool_result"
    ]
    assert len(tool_results) == 2
    ids = {tr["tool_use_id"] for tr in tool_results}
    assert ids == {"tu_a", "tu_b"}


# --- runtime tool availability (2026-07-07): hide what isn't live ---------------

def test_list_for_context_hides_unavailable_tools():
    from crystal_cache.agent.tool_registry import Tool, ToolRegistry

    async def _impl(customer_id: str) -> dict:  # pragma: no cover
        return {}

    reg = ToolRegistry()
    reg.register(Tool(name="always", description="d",
                      contexts=frozenset({"agent"}),
                      parameters_schema={"type": "object", "properties": {}},
                      impl=_impl))
    flag = {"on": False}
    reg.register(Tool(name="gated", description="d",
                      contexts=frozenset({"agent"}),
                      parameters_schema={"type": "object", "properties": {}},
                      impl=_impl, available=lambda: flag["on"]))

    assert [t.name for t in reg.list_for_context("agent")] == ["always"]
    flag["on"] = True
    assert [t.name for t in reg.list_for_context("agent")] == \
        ["always", "gated"]


def test_mem0_tools_hidden_when_backend_uninitialized(monkeypatch):
    """Hosted posture: mem0 extra absent / never initialized -> the mem0
    tools vanish from the agent's list (registration itself unchanged —
    the manifest test above still sees them)."""
    import crystal_cache.retrieval.mem0_session as m0
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools

    import_all_tools()
    monkeypatch.setattr(m0, "_mem0_instance", None, raising=False)
    names = {t.name for t in get_registry().list_for_context("agent")}
    assert "mem0_recall" not in names
    assert "mem0_write" not in names

    monkeypatch.setattr(m0, "_mem0_instance", object(), raising=False)
    names = {t.name for t in get_registry().list_for_context("agent")}
    assert "mem0_recall" in names and "mem0_write" in names


def test_system_prompt_omits_mem0_guidance_when_hidden(monkeypatch):
    import crystal_cache.retrieval.mem0_session as m0
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools
    from crystal_cache.models.customer import Customer, ModelRoutingConfig

    import_all_tools()
    customer = Customer(
        id="cus_prompt_test",
        model_routing_config=ModelRoutingConfig(
            provider="anthropic", model_id="m", api_key_ref=""),
    )

    monkeypatch.setattr(m0, "_mem0_instance", None, raising=False)
    tools = get_registry().list_for_context("agent")
    prompt = build_system_prompt(customer, tools)
    assert "mem0_recall" not in prompt
    assert "{MEM0_GUIDANCE}" not in prompt  # placeholder never leaks

    monkeypatch.setattr(m0, "_mem0_instance", object(), raising=False)
    tools = get_registry().list_for_context("agent")
    prompt = build_system_prompt(customer, tools)
    assert "Multi-turn awareness" in prompt


# --- Workstream A: extraction, not blobs (2026-07-08) ---------------------------

async def test_crystal_write_refuses_content_blobs(monkeypatch):
    """P1 TRIPWIRE: a value past the atomic-fact ceiling is refused with
    an error that names document_upload — the model self-corrects; the
    bank never gains a jumbo fact."""
    import crystal_cache.agent.tools.memory as mem

    monkeypatch.setattr(mem, "_get_state",
                        lambda: (_ for _ in ()).throw(
                            AssertionError("store touched on refused write")))
    out = await mem.crystal_write(
        "cus_x", key="Company|Era HQ|Overview", value="x" * 1200)
    assert "document_upload" in out["error"]
    assert "1200" in out["error"]


async def test_crystal_write_accepts_atomic_facts(monkeypatch, store):
    """At/under the ceiling the write proceeds unchanged."""
    import crystal_cache.agent.tools.memory as mem

    seen = {}

    class _Crystal:
        id = "crys_1"

    class _Fact:
        id = "fact_1"
        pair_type = "entity_attribute"

    class _Store:
        async def add_pair_for_customer(self, **kw):
            seen.update(kw)
            return _Crystal(), _Fact()

    monkeypatch.setattr(mem, "_get_state", lambda: {
        "store": _Store(), "encoder": object(),
        "vector_store": object(), "vector_index": None,
    })
    out = await mem.crystal_write(
        "cus_x", key="Company|Era HQ|HQ city", value="Raleigh, NC",
        pair_type="entity_attribute", source_kind="document_chunk")
    assert out == {"crystal_id": "crys_1", "fact_id": "fact_1",
                   "pair_type": "entity_attribute"}
    assert seen["answer_text"] == "Raleigh, NC"


def test_system_prompt_steers_learn_to_document_upload():
    from crystal_cache.agent.system_prompt import build_system_prompt
    from crystal_cache.agent.tool_registry import get_registry, import_all_tools
    from crystal_cache.models.customer import Customer, ModelRoutingConfig

    import_all_tools()
    customer = Customer(
        id="cus_steer",
        model_routing_config=ModelRoutingConfig(
            provider="anthropic", model_id="m", api_key_ref=""),
    )
    prompt = build_system_prompt(
        customer, get_registry().list_for_context("agent"))
    assert "document_upload" in prompt
    assert "ONE atomic fact" in prompt
