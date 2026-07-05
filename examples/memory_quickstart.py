#!/usr/bin/env python3
"""Crystal Cache memory quickstart — the create → store → retrieve loop.

Start a Crystal Cache memory server in one terminal:

    uvicorn crystal_cache.app:app --port 8000

then run this in another:

    python examples/memory_quickstart.py

It creates a fresh customer (account), stores a few facts, and retrieves them
by MEANING — no upstream LLM key required (store + retrieve never call a
model). Point any agent at the same /v1/store and /v1/retrieve endpoints and
it has durable, self-curating memory.

Override the server URL with CRYS_BASE_URL (default http://localhost:8000).
"""
from __future__ import annotations

import os
import sys

import httpx

BASE_URL = os.environ.get("CRYS_BASE_URL", "http://localhost:8000").rstrip("/")


def main() -> int:
    with httpx.Client(base_url=BASE_URL, timeout=60.0) as http:
        # 1) Create a customer (an account). Returns the Crystal Cache API
        #    key ("Key A") ONCE — save it. `api_key_ref` is the UPSTREAM
        #    provider key; the store/retrieve loop never uses it, so a
        #    placeholder is fine here (set a real one only for the chat
        #    proxy — see the documentation on the website).
        print(f"-> creating a customer at {BASE_URL} ...")
        r = http.post("/v1/customers", json={
            "provider": "openai",
            "model_id": "gpt-4o",
            "api_key_ref": "unused-for-memory-only",
        })
        r.raise_for_status()
        customer = r.json()
        api_key = customer["api_key"]
        auth = {"Authorization": f"Bearer {api_key}"}
        print(f"   customer id: {customer['id']}")
        print(f"   api key:     {api_key[:12]}... (save this -- shown once)\n")

        # 2) Store a few facts. `key` is what a query should match (the
        #    retrieval handle); `value` is the knowledge content.
        facts = [
            ("the team's primary database",
             "We use PostgreSQL 16 for all production services."),
            ("how we deploy",
             "Deploys go out via GitHub Actions on merge to main; no manual steps."),
            ("the on-call rotation tool",
             "On-call is managed in PagerDuty; the schedule is named 'core-eng'."),
        ]
        print("-> storing facts ...")
        for key, value in facts:
            r = http.post("/v1/store", headers=auth,
                          json={"key": key, "value": value})
            r.raise_for_status()
            out = r.json()
            print(f"   stored {out['crystal_id']}  <-  {key!r}")
        print()

        # 3) Retrieve by MEANING (not keyword): a paraphrased query still
        #    matches. No LLM call — this returns composed context you can
        #    drop straight into your own model prompt.
        query = "what database does the team run in prod?"
        print(f"-> retrieving: {query!r}")
        r = http.post("/v1/retrieve", headers=auth, json={"query": query})
        r.raise_for_status()
        res = r.json()
        print(f"   routing: {res['routing']}   score: {res['score']:.3f}")
        print(f"   matched crystals: {res['matched_crystal_ids']}")
        print("   -- injection --")
        injection = res.get("injection") or res.get("answer") or "(no match)"
        print("   " + injection.replace("\n", "\n   "))

    print("\nDone. That's the whole loop: create -> store -> retrieve.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except httpx.HTTPStatusError as e:
        print(f"\nHTTP {e.response.status_code}: {e.response.text}", file=sys.stderr)
        print("Tip: start the server with  uvicorn crystal_cache.app:app --port 8000",
              file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"\nCould not connect to {BASE_URL}.", file=sys.stderr)
        print("Start the server first:  uvicorn crystal_cache.app:app --port 8000",
              file=sys.stderr)
        sys.exit(1)
