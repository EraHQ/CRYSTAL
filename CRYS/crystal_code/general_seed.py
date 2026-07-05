"""General-bank seeding for CRYS: `--seed-general <file.jsonl>`.

Imports curated pattern-form knowledge into the GENERAL bank
(customer_id NULL) and subscribes the active customer to it. This is
the operator write-path for general crystals — customers are read-only
consumers (their learning lands in their own bank, never here).

Why patterns, not code: the BCB Hard findings
(crystal-cache-v1/docs/BCB_BENCHMARK_FINDINGS.md, 0.318→0.493 pass@1)
showed imperative pattern rules transfer where raw code examples
don't. The seed file format enforces the form: one JSON object per
line, {"key": "General|Domain|Topic|Slug", "claim": "the pattern"}.

Re-running is a replace, not an accumulation — same contract as
document re-ingestion.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from . import style

DEFAULT_GENERAL_TYPE = "general:swe_patterns"


def load_seed_entries(path: Path) -> list[dict]:
    """Parse and validate a seed JSONL file. Raises ValueError with the
    offending line number — a seed file is operator-authored, so a bad
    line is a fix-it-now error, not something to skip silently."""
    entries: list[dict] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path.name} line {n}: not valid JSON ({e.msg})") from e
        key = (obj.get("key") or "").strip()
        claim = (obj.get("claim") or "").strip()
        if not key or not claim:
            raise ValueError(f"{path.name} line {n}: needs non-empty 'key' and 'claim'")
        if not key.startswith("General|"):
            raise ValueError(
                f"{path.name} line {n}: keys must start with 'General|' — "
                "the namespace is how the agent tells general knowledge "
                "from project knowledge."
            )
        entries.append({"key": key, "claim": claim})
    if not entries:
        raise ValueError(f"{path.name}: no entries found")
    return entries


async def run_seed_import(
    db: Optional[str],
    seed_path: Path,
    *,
    crystal_type: str = DEFAULT_GENERAL_TYPE,
    customer_id: Optional[str] = None,
) -> int:
    """Import a seed file into the general bank — a PLATFORM write.

    General banks are platform-wide: importing registers the type in the
    crystal_types registry (scope='general'), which makes it visible to
    EVERY customer's Knowledge page, where each customer opts in or out
    with the subscription toggles. Importing does not require a customer
    and does not subscribe anyone, with two deliberate exceptions:

      - explicit ``--customer ID``: the operator asked for that customer
        to be subscribed as part of the import — honored.
      - the local CRYS store (no --db, no --customer): the operator IS
        the local user, so the local customer is auto-subscribed as a
        convenience — without it the agent wouldn't see patterns it
        just seeded.

    Seeding a foreign store (``--db path``) without ``--customer`` is
    pure import+register — in particular it must NEVER plant the
    the agent's local customer row in a server database (the bug
    this guard exists for, found 2026-06-12).

    Builds the minimal stack (store + encoder — no agent, no hands),
    refuses stale schemas like every other store-opening entry point,
    and prints a per-domain summary.
    """
    from crystal_cache.encoding import build_text_encoder

    from .daemon import _refuse_if_schema_stale
    from .runtime import LOCAL_CUSTOMER_ID, _make_store, _resolve_db_url

    try:
        entries = load_seed_entries(seed_path)
    except (OSError, ValueError) as e:
        print(style.yellow(f"not imported: {e}"))
        return 2

    store = _make_store(_resolve_db_url(db))
    try:
        await store.init()
        if await _refuse_if_schema_stale(store, "not imported"):
            return 2
        cid = customer_id
        if cid is None and db is None:
            # Local CRYS store only: runtime's local customer is
            # in-memory; subscriptions need the real row (see
            # store.ensure_customer_row docstring).
            cid = LOCAL_CUSTOMER_ID
            await store.ensure_customer_row(cid, api_key="local-coding-agent")

        print(style.dim(f"  … encoding {len(entries)} patterns (model load on first run)"))
        encoder = build_text_encoder()
        result = await store.import_general_bank(
            crystal_type=crystal_type, entries=entries, encoder=encoder,
        )

        # Subscribe only when there's a customer in play (idempotent
        # union — never drop existing subs). Platform-wide imports leave
        # subscription to each customer's own toggle on the Knowledge page.
        if cid is not None:
            subs = await store.get_customer_general_types(cid)
            if crystal_type not in subs:
                await store.set_customer_general_types(cid, [*subs, crystal_type])
                sub_note = f"subscribed {cid}"
            else:
                sub_note = f"{cid} already subscribed"
        else:
            sub_note = "platform-wide — customers opt in from the Knowledge page"

        domains = sorted({e["key"].split("|")[1] for e in entries})
        print(
            f"general bank updated: {result['facts']} patterns in "
            f"{result['crystals']} domains ({', '.join(domains)})"
        )
        print(style.dim(f"  type: {crystal_type} — {sub_note}"))
        print(style.dim(
            "  the agent now sees these alongside project knowledge; "
            "ask it to key_scan 'General|' to browse them"
        ))
        return 0
    finally:
        await store.dispose()
