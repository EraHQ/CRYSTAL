#!/usr/bin/env python3
"""Live connection smoke for the CRYS MCP memory server (WS C).

Exercises the one path the pytest suite can't reach: the real HTTP mount
at /mcp + the session-manager lifespan + the Bearer-key auth middleware +
the MCP streamable-HTTP protocol, end to end against a running server.

Usage:
    # 1. In one shell, with the FULL .venv (real gtr encoder):
    #       uvicorn crystal_cache.app:app --port 8000
    # 2. In another shell (any venv with `mcp` + `httpx` installed):
    #       python scripts/mcp_smoke.py
    #    or point it elsewhere:
    #       python scripts/mcp_smoke.py --base-url http://127.0.0.1:8000

The script mints its own throwaway customer via POST /v1/customers (so it
needs no pre-existing key), then connects an MCP client with that key and
runs a store -> search -> stats -> conflicts round-trip. Exits non-zero on
the first hard failure.

Only LLM-free tools are exercised (store / search / stats / conflicts), so
no upstream provider key is required — the dummy api_key_ref below is never
called. A 0-result search prints WARN (retrieval tuning), not FAIL — the
point of this smoke is wiring, not recall quality.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


EXPECTED_TOOLS = {
    "memory_search", "memory_search_documents", "memory_outline",
    "memory_keys", "memory_synthesize", "memory_recall", "memory_store",
    "memory_forget", "memory_ingest", "memory_learn", "memory_stats",
    "memory_list", "memory_export", "memory_import",
    "memory_conflicts", "memory_gaps",
}

_GREEN, _RED, _YELLOW, _RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}PASS{_RESET} {msg}")


def _warn(msg: str) -> None:
    print(f"  {_YELLOW}WARN{_RESET} {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}FAIL{_RESET} {msg}")
    sys.exit(1)


def _payload(result, label: str) -> dict:
    """Pull the tool's dict return from a CallToolResult, tolerant of whether
    FastMCP populated structuredContent or only a JSON text block."""
    if result.isError:
        _fail(f"{label} -> tool reported isError: {result.content}")
    if result.structuredContent is not None:
        return result.structuredContent
    for block in result.content:
        text = getattr(block, "text", None)
        if text:
            try:
                return json.loads(text)
            except (ValueError, TypeError):
                pass
    _fail(f"{label} -> no structured or JSON-text payload")
    return {}  # unreachable (_fail exits)


async def _mint_customer(base_url: str) -> str:
    """Create a throwaway customer; return its CRYS API key (Key A)."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{base_url}/v1/customers",
            json={
                "provider": "anthropic",
                "model_id": "claude-sonnet-4-5-20250929",
                "api_key_ref": "sk-smoke-upstream-unused",
            },
        )
    if resp.status_code != 201:
        _fail(f"create customer -> HTTP {resp.status_code}: {resp.text[:200]}")
    key = resp.json().get("api_key")
    if not key:
        _fail("create customer response missing 'api_key'")
    _ok(f"minted customer, key {key[:10]}...")
    return key


async def run(base_url: str) -> None:
    print(f"CRYS MCP smoke -> {base_url}")
    key = await _mint_customer(base_url)

    # Trailing slash hits the mounted endpoint directly (the mount is at
    # /mcp with inner path "/"), avoiding the cosmetic /mcp -> /mcp/ 307.
    mcp_url = f"{base_url}/mcp/"
    headers = {"Authorization": f"Bearer {key}"}

    async with streamablehttp_client(mcp_url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _ok("initialize handshake")

            tools = {t.name for t in (await session.list_tools()).tools}
            missing = EXPECTED_TOOLS - tools
            if missing:
                _fail(f"missing tools: {sorted(missing)}")
            _ok(f"list_tools -> {len(tools)} tools; all 16 memory_* present")

            r = await session.call_tool("memory_store", {
                "key": "Capital|France",
                "value": "Paris is the capital of France.",
            })
            sc = _payload(r, "memory_store")
            if not sc.get("crystal_id"):
                _fail(f"memory_store -> no crystal_id in {sc}")
            _ok(f"memory_store -> crystal {str(sc['crystal_id'])[:12]}...")

            r = await session.call_tool("memory_search", {
                "query": "What is the capital of France?", "k": 5,
            })
            sc = _payload(r, "memory_search")
            fc = sc.get("fact_count", 0)
            if fc >= 1:
                _ok(f"memory_search -> {fc} fact(s) retrieved")
            else:
                _warn("memory_search -> 0 facts (retrieval tuning, not a wiring failure)")

            sc = _payload(await session.call_tool("memory_stats", {}), "memory_stats")
            _ok(f"memory_stats -> {sc.get('crystal_count')} crystal(s), "
                f"{sc.get('fact_count')} fact(s)")

            sc = _payload(await session.call_tool("memory_conflicts", {}), "memory_conflicts")
            _ok(f"memory_conflicts -> count {sc.get('count')}")

    print(f"\n{_GREEN}SMOKE PASSED{_RESET} — mount + lifespan + auth + protocol all green.")


def main() -> None:
    ap = argparse.ArgumentParser(description="CRYS MCP live connection smoke")
    ap.add_argument(
        "--base-url", default="http://127.0.0.1:8000",
        help="CRYS server base URL (default http://127.0.0.1:8000)",
    )
    args = ap.parse_args()
    try:
        asyncio.run(run(args.base_url))
    except SystemExit:
        raise
    except KeyboardInterrupt:
        sys.exit(130)
    except Exception as e:  # noqa: BLE001 - turn any transport error into a clear hint
        print(f"  {_RED}FAIL{_RESET} could not complete smoke: {e!r}")
        print(f"  hint: is the server running? -> uvicorn crystal_cache.app:app --port 8000")
        sys.exit(1)


if __name__ == "__main__":
    main()
