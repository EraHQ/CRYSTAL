"""Tool annotations pin (2026-09-27, Connectors Directory requirement).

Every MCP tool must carry a title and the applicable readOnlyHint /
destructiveHint (Anthropic directory submission requirement, and good
client hygiene besides: clients drive confirmation prompts off these).
The table lives in agent/mcp_server.py and is applied at import with a
loud drift check; this pin holds the contract from the wire side.

R14 note: verified by pytest; describes expected behavior.
"""
from crystal_cache.agent import mcp_server as srv


async def test_every_tool_carries_directory_grade_annotations():
    tools = await srv.mcp.list_tools()
    assert len(tools) >= 21, [t.name for t in tools]
    for t in tools:
        a = t.annotations
        assert a is not None, f"{t.name}: no annotations"
        assert a.title, f"{t.name}: no title"
        assert a.readOnlyHint is not None, f"{t.name}: readOnlyHint unset"
        assert a.destructiveHint is not None, f"{t.name}: destructiveHint unset"


async def test_classification_spot_checks():
    by_name = {t.name: t.annotations for t in await srv.mcp.list_tools()}
    # The destructive pair: ledgered retire, destructive from the client.
    assert by_name["forget"].destructiveHint is True
    assert by_name["memory_forget"].destructiveHint is True
    # Reads say so; writes say so.
    assert by_name["memory_search"].readOnlyHint is True
    assert by_name["recall"].readOnlyHint is True
    assert by_name["memory_store"].readOnlyHint is False
    assert by_name["memory_ingest"].readOnlyHint is False
    # The unaudited composer never claims purity.
    assert by_name["memory_synthesize"].readOnlyHint is False
