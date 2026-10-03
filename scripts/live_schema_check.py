"""Live structured-output check (2026-10-02).

Sends the respond schema through the REAL client code path to the REAL
API, once, from the developer's machine, before any deploy that touches
a json_schema. Three production deploys of the respond feature failed on
shape rules a single live call would have caught.

Usage (the key is read from the environment of this one process only):

    read -rs ANTHROPIC_LIVE_KEY
    ANTHROPIC_API_KEY="$ANTHROPIC_LIVE_KEY" python scripts/live_schema_check.py
    unset ANTHROPIC_LIVE_KEY

Prints the parsed JSON and the token counts. Any 400 from the API prints
its message (redacted) and exits non-zero.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "memory", "src"))

from crystal_cache.endpoints.admin import _RESPOND_SCHEMA, _RESPOND_SYSTEM  # noqa: E402
from crystal_cache.hygiene import redact  # noqa: E402
from crystal_cache.llm.client import LLMClient  # noqa: E402


def main() -> int:
    key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CC_ANTHROPIC_API_KEY")
    if not key:
        print("set ANTHROPIC_API_KEY for this process (see the module docstring)")
        return 2
    client = LLMClient(
        provider="anthropic", api_key=key, base_url=None,
        model_small="claude-haiku-4-5", model_large="claude-sonnet-5",
        model_frontier="claude-opus-4-8",
    )
    sample = {
        "current_assumption": {"statement": "Payments are due on the 31st", "confidence": 0.6},
        "evidence": [
            {"id": "ev_a", "summary": "Invoices are sent on the 1st", "facts": ["Invoices go out on the 1st"]},
            {"id": "ev_b", "summary": "Net-30 terms apply", "facts": ["Payment terms are net 30"]},
        ],
        "thread": [],
        "operator_response": "Due 30 days after the invoice on the 1st, and there is a 2% late fee.",
    }
    try:
        result = client.complete_detailed(
            system=_RESPOND_SYSTEM,
            messages=[{"role": "user", "content": json.dumps(sample)}],
            max_tokens=800,
            temperature=0.2,
            tier="large",
            json_schema=_RESPOND_SCHEMA,
        )
    except Exception as e:  # noqa: BLE001
        print("API rejected the request:", redact(str(e))[:600])
        return 1
    parsed = json.loads(result.text)
    print(json.dumps(parsed, indent=2))
    print(f"\nmodel={result.model} in={result.input_tokens} out={result.output_tokens}")
    assert isinstance(parsed.get("assumptions"), list) and parsed["assumptions"], "no assumptions returned"
    for a in parsed["assumptions"]:
        assert set(a) == {"statement", "subject", "confidence", "rationale", "replaces_parent"}, a
    print("\nOK: schema accepted, shape as designed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
