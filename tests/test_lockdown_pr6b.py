"""Lockdown PR-6b pins (Q46=A field limits, the tool side, 2026-10-13).

The agent tool registry validates every call against the tool's own
JSON schema, once, in the wrapper `register_tool` puts around `impl`,
so the agent loop, `mcp_server._dispatch` and the cognition retrieval
adapter are all covered by construction. These pins drive the real
registry, the real MCP tools and the real adapter:

  1. structure: every registered tool's schema is closed
     (`additionalProperties: false`) and every parameter carries a
     bound (maxLength / enum, maximum, maxItems, a closed object), with
     a floor on the tool count;
  2. the guard: one over-limit argument per tool through `tool.impl`
     is `{"error", "code": "invalid_argument"}` and the tool never ran;
     an unexpected argument is refused; the message names the argument
     and the rule and never echoes the value; a None optional is left
     to the tool's default; the bare function is untouched;
  3. the three callers: MCP (`_dispatch`, the typed wrapper and the
     protocol boundary), the cognition adapter, and the agent loop;
  4. the MCP-only tools bound their own ids and clamp their pages;
  5. key_scan: `%` and `_` in the filters are literal (LIKE-escaped in
     the store, on the real SQL) and the scan is capped.
"""
from __future__ import annotations

import inspect

import pytest
from jsonschema import Draft202012Validator

from crystal_cache.agent import mcp_server
from crystal_cache.agent import tool_registry as reg
from crystal_cache.agent.tool_registry import (
    INVALID_ARGUMENT,
    get_registry,
    guard_arguments,
    import_all_tools,
)
from crystal_cache.agent.tools import retrievers as retrievers_mod
from crystal_cache.infrastructure import metadata_store_cognition_ext as cog_ext

MIN_TOOLS = 25
LONG = "Z" * 9000
LONG_ID = "x" * 65


def _registry():
    import_all_tools()
    r = get_registry()
    assert len(r) >= MIN_TOOLS, len(r)
    return r


@pytest.fixture
def mcp_state(store, customer, semantic_encoder_stub, vector_store, fact_vector_store, monkeypatch):
    from crystal_cache.agent.tools.retrievers import set_tool_state

    state = {
        "store": store, "encoder": semantic_encoder_stub, "vector_store": vector_store,
        "fact_vector_store": fact_vector_store, "vector_index": None,
    }
    monkeypatch.setattr(mcp_server, "_get_state", lambda: state)
    set_tool_state(state)
    token = mcp_server._current_customer_id.set(customer.id)
    yield state
    mcp_server._current_customer_id.reset(token)


# ---------------------------------------------------------------------------
# 1. Structure
# ---------------------------------------------------------------------------

def _bounded(spec: dict) -> bool:
    t = spec.get("type")
    if t == "string":
        return "maxLength" in spec or "enum" in spec
    if t in ("integer", "number"):
        return "maximum" in spec
    if t == "array":
        return "maxItems" in spec and _bounded(spec.get("items", {}))
    if t == "object":
        return spec.get("additionalProperties") is False and all(
            _bounded(p) for p in spec.get("properties", {}).values()
        )
    return t == "boolean"


def test_every_tool_schema_is_closed_and_every_parameter_bounded():
    loose: list[str] = []
    for tool in (_registry()._tools.values()):
        schema = tool.parameters_schema
        Draft202012Validator.check_schema(schema)
        if schema.get("additionalProperties") is not False:
            loose.append(f"{tool.name}: additionalProperties")
        for pname, spec in schema.get("properties", {}).items():
            if not _bounded(spec):
                loose.append(f"{tool.name}.{pname}: {spec.get('type')} unbounded")
    assert not loose, "\n".join(loose)


# ---------------------------------------------------------------------------
# 2. The guard
# ---------------------------------------------------------------------------

# (tool, over-limit arguments, the argument the message must name)
OVER_LIMIT = [
    ("content_search", {"query": LONG}, "query"),
    ("content_search", {"query": "q", "k": 51}, "k"),
    ("content_search", {"query": "q", "hints": {"subject": "s", "bogus": 1}}, "hints"),
    ("content_search", {"query": "q", "hints": {"depth_mode": "sideways"}}, "hints.depth_mode"),
    ("knowledge_search", {"query": "q", "k": 0}, "k"),
    ("navigation_search", {"query_text": LONG}, "query_text"),
    ("depth_search", {"query": "q", "hints": {"locator_prefix": "p" * 257}}, "hints.locator_prefix"),
    ("key_scan", {"key_prefix": "k" * 257}, "key_prefix"),
    ("key_scan", {"subject_contains": "s" * 257}, "subject_contains"),
    ("crystal_recall", {"query": "q", "k": 999}, "k"),
    ("crystal_write", {"key": "k", "value": "v" * 100_001}, "value"),
    ("crystal_write", {"key": "k" * 5001, "value": "v"}, "key"),
    ("crystal_write", {"key": "k", "value": "v", "scope": "public"}, "scope"),
    ("crystal_write", {"key": "k", "value": "v", "source_kind": "document_extraction"}, "source_kind"),
    ("crystal_write", {"key": "k", "value": "v", "crystal_type": "Customer:Legacy"}, "crystal_type"),
    ("crystal_write", {"key": "k", "value": "v", "pair_type": "p" * 65}, "pair_type"),
    ("mem0_recall", {"query": "q", "limit": 21}, "limit"),
    ("mem0_write", {"query": "q", "response": "r" * 20_001}, "response"),
    ("crystal_learn", {"prompt": "p", "response": "r", "outcome": "meh"}, "outcome"),
    ("crystal_learn", {"prompt": "p" * 50_001, "response": "r"}, "prompt"),
    ("knowledge_conflicts", {"status": "weird"}, "status"),
    ("knowledge_conflicts", {"limit": 201}, "limit"),
    ("knowledge_gaps", {"status": "open", "limit": 0}, "limit"),
    ("resolve_conflict", {"conflict_id": LONG_ID, "resolution": "qualified", "user_confirmation": "yes"}, "conflict_id"),
    ("resolve_conflict", {"conflict_id": "c1", "resolution": "blacklisted", "user_confirmation": "yes"}, "resolution"),
    ("resolve_conflict", {"conflict_id": "c1", "resolution": "superseded", "user_confirmation": "yes", "loser": "c"}, "loser"),
    ("record_gap", {"question": "q" * 2001, "disposition": "researchable"}, "question"),
    ("record_gap", {"question": "q", "disposition": "maybe"}, "disposition"),
    ("record_gap", {"question": "q", "disposition": "workable", "priority": "urgent"}, "priority"),
    ("assume", {"crystal_a_id": "a", "crystal_b_id": "b c"}, "crystal_b_id"),
    ("propose_correction", {"crystal_id": "c", "proposed_change": "p" * 4001, "rationale": "r"}, "proposed_change"),
    ("web_search", {"query": "q" * 2001}, "query"),
    ("web_fetch", {"url": "https://x/" + "a" * 2048}, "url"),
    ("web_fetch", {"url": "https://x/", "offset": -1}, "offset"),
    ("source_lookup", {"op": "search", "query": LONG}, "query"),
    ("source_lookup", {"op": "grep"}, "op"),
    ("document_upload", {"label": "l", "text": "t", "crystal_type": "general:python"}, "crystal_type"),
    ("document_upload", {"label": "l" * 257, "text": "t"}, "label"),
    ("decompose", {"text": "t" * 100_001}, "text"),
    ("cognition_run", {"goal": "g" * 8001}, "goal"),
    ("cognition_run", {"goal": "g", "max_attempts": 11}, "max_attempts"),
    ("cognition_run", {"goal": "g", "output_type": "poem"}, "output_type"),
    ("cognition_status", {"task_id": LONG_ID}, "task_id"),
    ("create_document", {"content": "c", "filename": "f" * 257}, "filename"),
    ("llm_invoke", {"messages": [{"role": "user", "content": "hi"}] * 51}, "messages"),
    ("llm_invoke", {"messages": [{"role": "user", "content": "hi"}], "temperature": 1.5}, "temperature"),
]


@pytest.mark.asyncio
async def test_registry_refuses_over_limit_arguments_without_running_the_tool():
    """No tool state is set in this test, so a tool that RAN would raise
    or fail on the missing state; a clean refusal dict is the proof the
    guard answered first."""
    r = _registry()
    covered = set()
    for name, args, param in OVER_LIMIT:
        tool = r.get(name)
        assert tool is not None, name
        out = await tool.impl(customer_id="cus_pin", **args)
        assert isinstance(out, dict) and out.get("code") == INVALID_ARGUMENT, (name, args, out)
        assert out["error"].startswith(f"{name}: {param} "), (name, out["error"])
        assert "ZZZZ" not in out["error"] and LONG_ID not in out["error"], out["error"]
        covered.add(name)
    # Every tool in the registry has at least one over-limit case above.
    assert covered == set(r.names()), sorted(set(r.names()) - covered)


@pytest.mark.asyncio
async def test_unexpected_and_wrong_typed_arguments_are_refused():
    r = _registry()
    out = await r.get("crystal_recall").impl(customer_id="c", query="q", surprise=1)
    assert out["code"] == INVALID_ARGUMENT and "surprise" in out["error"]
    out = await r.get("crystal_recall").impl(customer_id="c", query="q", k="ten")
    assert out["code"] == INVALID_ARGUMENT and "k must be of type integer" in out["error"]
    out = await r.get("crystal_recall").impl(customer_id="c", k=3)
    assert out["code"] == INVALID_ARGUMENT and "missing required argument(s): query" in out["error"]
    # A boolean is not an integer.
    out = await r.get("crystal_recall").impl(customer_id="c", query="q", k=True)
    assert out["code"] == INVALID_ARGUMENT, out


@pytest.mark.asyncio
async def test_guard_leaves_none_to_the_tool_and_the_bare_function_alone():
    calls: list[dict] = []

    async def impl(customer_id: str, query: str, k: int = 5, hints: dict | None = None) -> dict:
        calls.append({"customer_id": customer_id, "query": query, "k": k, "hints": hints})
        return {"ok": True}

    schema = {
        "type": "object", "additionalProperties": False,
        "properties": {
            "query": {"type": "string", "maxLength": 10},
            "k": {"type": "integer", "minimum": 1, "maximum": 50},
            "hints": {"type": "object", "additionalProperties": False,
                      "properties": {"subject": {"type": "string", "maxLength": 4}}},
        },
        "required": ["query"],
    }
    guarded = guard_arguments("t", schema, impl)
    assert inspect.unwrap(guarded) is impl
    assert inspect.signature(guarded) == inspect.signature(impl)
    # None is "not provided": the tool's own default applies.
    assert await guarded(customer_id="c", query="q", k=None, hints=None) == {"ok": True}
    assert calls[-1] == {"customer_id": "c", "query": "q", "k": None, "hints": None}
    # Positional customer_id (mcp_server._dispatch's shape) is never validated.
    assert await guarded("c", query="q") == {"ok": True}
    out = await guarded("c", query="q" * 11)
    assert out == {"error": "t: query must be at most 10 characters", "code": INVALID_ARGUMENT}
    assert len(calls) == 2  # the refused call never ran
    # A malformed schema fails at registration, not on the first call.
    with pytest.raises(Exception):
        guard_arguments("bad", {"type": "object", "properties": {"q": {"type": "strung"}}}, impl)


def test_every_registered_impl_is_the_guard_over_the_bare_function():
    for tool in _registry()._tools.values():
        bare = inspect.unwrap(tool.impl)
        assert bare is not tool.impl, tool.name
        assert tool.impl.__name__ == bare.__name__ == tool.impl.__wrapped__.__name__


# ---------------------------------------------------------------------------
# 3. The three callers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_dispatch_typed_wrapper_and_boundary_carry_the_refusal(mcp_state):
    from mcp.server.fastmcp.exceptions import ToolError

    out = await mcp_server._dispatch("crystal_recall", query="q", k=999)
    assert out["code"] == INVALID_ARGUMENT and "k must be at most 50" in out["error"]
    out = await mcp_server.memory_recall.fn(query=LONG, k=5)
    assert out["code"] == INVALID_ARGUMENT and "ZZZZ" not in out["error"]
    with pytest.raises(ToolError) as exc:
        await mcp_server.memory_recall(query="q", k=999)
    assert "[invalid_argument]" in str(exc.value)
    # The 800-character working limit stays the tool's own refusal (it names
    # memory_ingest; RC-15, pinned in test_mcp_contract); the schema's
    # ceiling is the wire bound above it.
    out = await mcp_server.memory_store.fn(key="k", value="v" * 801)
    assert out["code"] == "content_too_long" and "memory_ingest" in out["error"]
    out = await mcp_server.memory_store.fn(key="k", value="v" * 100_001)
    assert out["code"] == INVALID_ARGUMENT and "value must be at most 100000" in out["error"]
    out = await mcp_server.memory_keys.fn(key_prefix="k" * 257)
    assert out["code"] == INVALID_ARGUMENT
    out = await mcp_server.remember.fn(fact="f" * 801)
    assert out["code"] == "content_too_long" and "memory_ingest" in out["error"]
    out = await mcp_server.recall.fn(query="q", mode="gaps", k=0)
    assert out["code"] == INVALID_ARGUMENT and "limit must be at least 1" in out["error"]


@pytest.mark.asyncio
async def test_cognition_adapter_goes_through_the_guard(
    store, customer, semantic_encoder_stub, vector_store, fact_vector_store,
):
    from crystal_cache.cognition.retrieval_adapter import dispatch_cognition_retrieval

    out = await dispatch_cognition_retrieval(
        action_value="crystal_key_scan",
        step_input={"key_prefix": "k" * 257},
        customer_id=customer.id, store=store, fact_store=fact_vector_store,
        encoder=semantic_encoder_stub,
    )
    assert out["code"] == INVALID_ARGUMENT and "key_prefix must be at most 256" in out["error"]
    # A step_input that leaves a value None takes the tool's default.
    out = await dispatch_cognition_retrieval(
        action_value="crystal_key_scan",
        step_input={"key_prefix": "Nothing|Here", "subject_contains": None},
        customer_id=customer.id, store=store, fact_store=fact_vector_store,
        encoder=semantic_encoder_stub,
    )
    assert out["results_count"] == 0 and "code" not in out


def test_agent_loop_calls_the_registry_impl_not_the_bare_function():
    """The agent loop reaches tools through `tool.impl` (the guarded
    callable) and nowhere unwraps it."""
    from crystal_cache.agent import agent as agent_mod
    from crystal_cache.cognition import retrieval_adapter

    for mod in (agent_mod, mcp_server, retrieval_adapter):
        src = inspect.getsource(mod)
        assert "tool.impl(" in src, mod.__name__
        assert "__wrapped__" not in src and "inspect.unwrap" not in src, mod.__name__


# ---------------------------------------------------------------------------
# 4. MCP-only tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_mcp_only_tools_bound_their_ids_and_clamp_their_pages(mcp_state, store, customer):
    out = await mcp_server.memory_forget.fn(crystal_id=LONG_ID)
    assert out["code"] == INVALID_ARGUMENT and out["deleted"] is False
    out = await mcp_server.memory_forget.fn(fact_id="has space")
    assert out["code"] == INVALID_ARGUMENT
    out = await mcp_server.forget.fn(crystal_id=LONG_ID)
    assert out["code"] == INVALID_ARGUMENT and out["retired"] is False
    out = await mcp_server.memory_list.fn(crystal_id=LONG_ID)
    assert out["code"] == INVALID_ARGUMENT
    out = await mcp_server.memory_list.fn(limit=999_999, offset=-5)
    assert out["limit"] == mcp_server.MCP_LIST_MAX_LIMIT == 200 and out["offset"] == 0
    out = await mcp_server.memory_list.fn(limit=0, offset=10**9)
    assert out["limit"] == 1 and out["offset"] == mcp_server.MCP_LIST_MAX_OFFSET


# ---------------------------------------------------------------------------
# 5. key_scan: literal filters, capped scan
# ---------------------------------------------------------------------------

async def _seed(store, customer, encoder, vector_store, keys: list[str]) -> None:
    for i, key in enumerate(keys):
        await store.add_pair_for_customer(
            customer_id=customer.id, prompt_text=key, answer_text=f"value {i}",
            pair_type="question_answer", encoder=encoder, vector_store=vector_store,
            vector_index=None, crystal_type="customer:legacy",
            source_kind="model_reasoning", answer_value=None,
        )


@pytest.mark.asyncio
async def test_key_scan_filters_are_literal_and_the_scan_is_capped(
    mcp_state, store, customer, semantic_encoder_stub, vector_store, monkeypatch,
):
    await _seed(store, customer, semantic_encoder_stub, vector_store, [
        "Sales|Q1_2026|north",   # a literal underscore
        "Sales|Q12026|south",    # what a wildcard `_` would also match
        "Sales|100%|east",       # a literal percent
        "Sales|100x|west",       # what a wildcard `%` would also match
        "Sales|back\\slash",     # a literal backslash (the escape itself)
    ])
    assert cog_ext.like_escape("a%b_c\\d") == "a\\%b\\_c\\\\d"
    keys = lambda facts: sorted(f.prompt_text for f in facts)  # noqa: E731

    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|Q1_")
    assert keys(facts) == ["Sales|Q1_2026|north"]
    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|100%")
    assert keys(facts) == ["Sales|100%|east"]
    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|", subject_contains="%")
    assert keys(facts) == ["Sales|100%|east"]
    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|", subject_contains="_2026")
    assert keys(facts) == ["Sales|Q1_2026|north"]
    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|back\\")
    assert keys(facts) == ["Sales|back\\slash"]
    facts = await store.list_facts_by_key_prefix(customer.id, key_prefix="Sales|")
    assert len(facts) == 5

    # The tool passes the cap; a lowered cap cuts the scan.
    out = await mcp_server.memory_keys.fn(key_prefix="Sales|")
    assert out["results_count"] == 5
    monkeypatch.setattr(retrievers_mod, "KEY_SCAN_MAX_RESULTS", 2)
    out = await mcp_server.memory_keys.fn(key_prefix="Sales|")
    assert out["results_count"] == 2
    src = inspect.getsource(retrievers_mod.key_scan)
    assert "limit=KEY_SCAN_MAX_RESULTS" in src


def test_jsonschema_is_a_declared_dependency_with_a_floor():
    import importlib.metadata
    from pathlib import Path

    root = Path(reg.__file__).resolve().parents[4]  # memory/src/crystal_cache/agent/x.py
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert '"jsonschema>=4.26"' in pyproject
    major, minor, *_ = importlib.metadata.version("jsonschema").split(".")
    assert (int(major), int(minor)) >= (4, 26)
