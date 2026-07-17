"""Code description generation for retrieval (CRYS, June 2026).

Code chunks are stored as verbatim source, and a content_chunk fact's SEARCH
vector is encode_native(text) — so a natural-language query is matched against
raw code and only hits on incidental identifier overlap (the symbol you can't
name is unfindable). This module generates a functional natural-language
description for each symbol plus a file-level summary — phrased the way someone
searching for the behavior would describe it, NOT a restatement of the
signature. The crystallizer then indexes each chunk by its description (via
add_pair_*'s embed_text) while still returning the verbatim body.

Two modes, chosen by file size:

  * WHOLE-FILE (file fits in one call): the entire file is sent once with the
    symbols listed by index. The model describes every symbol with the full
    file in view — connective tissue, imports, neighbours and all. One call.
  * BATCHED (file too large): the file can't fit, so its symbols are processed
    in SEQUENCE in code-budget batches, each symbol carrying its own body, and
    each batch after the first is handed the running summary + the descriptions
    already produced (chaining) so later symbols are described with awareness of
    earlier ones. After the last batch, ONE synopsis call sees ALL the
    descriptions together and writes the file summary — synthesized from the
    complete set rather than guessed incrementally.

Generation is deterministic (temperature 0) so an ingest is reproducible.

Gated by CC_ENABLE_CODE_DESCRIPTIONS; only the ingest path (which holds an
Anthropic client) generates them. Best-effort by design: any call that errors
or returns unparseable JSON is skipped (those symbols fall back to indexing the
verbatim code); other batches, the synopsis, and the rest proceed.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Optional

from ..cost.emit import record_model_call

logger = logging.getLogger(__name__)

# Deterministic so two ingests of the same source produce the same index.
DESCRIBE_TEMPERATURE = 0.0

# If the whole file is within this many chars, describe it in ONE call with the
# full file in view (best quality). Larger files fall to batched mode.
WHOLE_FILE_BUDGET = 18_000
# Batched mode: cap each symbol's code (its head carries most of the intent) so
# one giant symbol can't blow the budget, and split symbols into batches within
# this combined-code budget so every symbol's body is actually seen.
PER_SYMBOL_CODE_CHARS = 4_000
BATCH_CODE_BUDGET = 18_000
# Batched mode carry-forward: the running summary + prior descriptions handed to
# the next batch, bounded (most-recent-first) so a huge file's context stays
# bounded.
CARRY_CONTEXT_CHARS = 4_000
# Synopsis pass: cap the descriptions listing fed to the final summary call.
SYNOPSIS_LISTING_CHARS = 16_000

DESCRIBE_MAX_TOKENS = 4_000
SYNOPSIS_MAX_TOKENS = 200

DESCRIBE_SYSTEM = """You write retrieval descriptions for code symbols.

You are given symbols from a source file — sometimes the whole file, sometimes
one part of a larger file processed in sequence, with context about the parts
already covered. For EACH symbol listed, write a 1-2 sentence description of
WHAT IT DOES and WHEN SOMEONE WOULD LOOK FOR IT — phrased the way a developer
describes behavior in plain language when they don't yet know what it's called.
Name the symbol, then describe its job and the problem it solves. When a symbol
relates to one already described, say so. Do NOT restate the signature,
parameter names, or types. Do NOT quote the code. Prefer the words someone
would search for ("retry on failure", "limit the request rate", "parse the
config file") over implementation nouns.

Also write file_summary: one sentence on what this file is for in the system.

Return ONLY a JSON object, no markdown and no prose:
{"file_summary": "...", "symbols": {"0": "...", "1": "...", ...}}
where each key is the [index] shown next to the symbol in THIS message."""

SYNOPSIS_SYSTEM = """You summarize what a source file is for. Given the file's
symbols and a short description of what each does, reply with ONE sentence
describing the file's role in the system. Return only the sentence — no
preamble, no markdown, no list."""


def _has_text(chunks: list[dict]) -> bool:
    return any((c.get("text") or "").strip() for c in chunks)


def _label(c: dict, idx: int) -> str:
    return c.get("label") or c.get("locator") or f"symbol {idx}"


def _call_describe(client: Any, user: str) -> tuple[Optional[dict], Any]:
    """One description call → (parsed JSON object or None, usage or None).
    Gate B: prefers complete_detailed so callers can stamp the ledger;
    fakes exposing only complete() run unmetered but identical."""
    _kwargs = dict(
        system=DESCRIBE_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=DESCRIBE_MAX_TOKENS,
        temperature=DESCRIBE_TEMPERATURE,
        tier="small",
    )
    try:
        _detailed = getattr(client, "complete_detailed", None)
        if _detailed is not None:
            _result = _detailed(**_kwargs)
            return _parse_json_object((_result.text or "").strip()), _result
        text = client.complete(**_kwargs)
        return _parse_json_object((text or "").strip()), None
    except Exception as e:  # noqa: BLE001 — best-effort; caller falls back
        logger.warning(
            "code_describer.call_failed",
            extra={"error": str(e), "error_type": type(e).__name__},
        )
        return None, None


def _collect(parsed: dict, entries: list[tuple[int, str]]) -> tuple[dict[int, str], list[tuple[str, str]]]:
    """Map the model's per-[index] descriptions back to GLOBAL chunk indices.

    `entries` is (global_index, label) in the order shown to the model — the
    local [index] is the position. Returns {global_index: desc} and an
    order-preserving (label, desc) list for carry-forward / synopsis.
    """
    raw = parsed.get("symbols")
    raw = raw if isinstance(raw, dict) else {}
    by_index: dict[int, str] = {}
    described: list[tuple[str, str]] = []
    for local, (gidx, label) in enumerate(entries):
        v = raw.get(str(local))
        if isinstance(v, str) and v.strip():
            desc = v.strip()
            by_index[gidx] = desc
            described.append((label, desc))
    return by_index, described


# ---------------------------------------------------------------------------
# Whole-file mode
# ---------------------------------------------------------------------------

async def _describe_whole_file(
    file_text: str, chunks: list[dict], client: Any, file_label: str,
    customer_id: Optional[str] = None, store: Any = None,
) -> dict[str, Any]:
    symbol_list = "\n".join(f"[{i}] {_label(c, i)}" for i, c in enumerate(chunks))
    user = (
        f"File: {file_label or 'unknown'}\n\n"
        f"Symbols:\n{symbol_list}\n\n"
        f"Source:\n```\n{file_text}\n```"
    )
    parsed, _usage = _call_describe(client, user)
    if _usage is not None and customer_id:
        await record_model_call(
            customer_id=customer_id,
            origin="code_descriptions",
            model=_usage.model,
            input_tokens=_usage.input_tokens,
            output_tokens=_usage.output_tokens,
            cache_creation_tokens=_usage.cache_creation_tokens,
            cache_read_tokens=_usage.cache_read_tokens,
            store=store,
        )
    if not isinstance(parsed, dict):
        return {"file_summary": "", "by_index": {}}
    entries = [(i, _label(c, i)) for i, c in enumerate(chunks)]
    by_index, _ = _collect(parsed, entries)
    fs = parsed.get("file_summary")
    return {"file_summary": fs.strip() if isinstance(fs, str) else "", "by_index": by_index}


# ---------------------------------------------------------------------------
# Batched mode
# ---------------------------------------------------------------------------

def _batch_by_budget(chunks: list[dict]) -> list[list[tuple[int, dict, str]]]:
    """Split chunks (in source order) into batches whose combined per-symbol
    code stays within BATCH_CODE_BUDGET. Each entry is (global_index, chunk,
    capped_code)."""
    batches: list[list[tuple[int, dict, str]]] = []
    cur: list[tuple[int, dict, str]] = []
    cur_size = 0
    for gidx, c in enumerate(chunks):
        code = (c.get("text") or "")[:PER_SYMBOL_CODE_CHARS]
        size = len(code)
        if cur and cur_size + size > BATCH_CODE_BUDGET:
            batches.append(cur)
            cur, cur_size = [], 0
        cur.append((gidx, c, code))
        cur_size += size
    if cur:
        batches.append(cur)
    return batches


def _carry_context(running_summary: str, described: list[tuple[str, str]]) -> str:
    """The compressed state handed to the NEXT batch: the running file summary
    plus the symbols already described (most-recent-first, capped)."""
    if not running_summary and not described:
        return ""
    lines: list[str] = []
    if running_summary:
        lines.append(f"File summary so far: {running_summary}")
    if described:
        picked: list[str] = []
        acc = 0
        for label, desc in reversed(described):
            entry = f"- {label}: {desc}"
            if acc + len(entry) > CARRY_CONTEXT_CHARS:
                break
            picked.append(entry)
            acc += len(entry)
        if picked:
            lines.append("Symbols already described in earlier parts:")
            lines.extend(reversed(picked))  # back into source order
    return "\n".join(lines)


def _batch_prompt(
    file_label: str, b: int, n_batches: int,
    carry: str, batch: list[tuple[int, dict, str]],
) -> str:
    part = f" (part {b + 1} of {n_batches})" if n_batches > 1 else ""
    symbol_block = "\n\n".join(
        f"[{local}] {_label(c, gidx)}\n```\n{code}\n```"
        for local, (gidx, c, code) in enumerate(batch)
    )
    ctx = f"Context from earlier parts of this file:\n{carry}\n\n" if carry else ""
    return (
        f"File: {file_label or 'unknown'}{part}\n\n"
        f"{ctx}"
        "Describe each symbol below; reference earlier ones when relevant. "
        "Key the JSON by the [index] shown here.\n\n"
        f"{symbol_block}"
    )


def _synopsize(client: Any, file_label: str, described: list[tuple[str, str]]) -> tuple[str, Any]:
    """End-of-file pass: one call over ALL the file's descriptions → a single
    synthesized file summary. Returns ("", None) on no input or failure;
    (text, usage) otherwise (Gate B metering)."""
    if not described:
        return "", None
    lines: list[str] = []
    acc = 0
    for label, desc in described:
        line = f"- {label}: {desc}"
        if acc + len(line) > SYNOPSIS_LISTING_CHARS:
            break
        lines.append(line)
        acc += len(line)
    user = (
        f"File: {file_label or 'unknown'}\n\n"
        "Symbols in this file and what each does:\n"
        + "\n".join(lines)
        + "\n\nIn ONE sentence, what is this file for in the system?"
    )
    _kwargs = dict(
        system=SYNOPSIS_SYSTEM,
        messages=[{"role": "user", "content": user}],
        max_tokens=SYNOPSIS_MAX_TOKENS,
        temperature=DESCRIBE_TEMPERATURE,
        tier="small",
    )
    try:
        _detailed = getattr(client, "complete_detailed", None)
        if _detailed is not None:
            _result = _detailed(**_kwargs)
            return (_result.text or "").strip(), _result
        text = client.complete(**_kwargs)
        return (text or "").strip(), None
    except Exception as e:  # noqa: BLE001 — best-effort; caller falls back
        logger.warning(
            "code_describer.synopsis_failed",
            extra={"label": file_label, "error": str(e), "error_type": type(e).__name__},
        )
        return "", None


async def _describe_batched(
    chunks: list[dict], client: Any, file_label: str,
    customer_id: Optional[str] = None, store: Any = None,
) -> dict[str, Any]:
    batches = _batch_by_budget(chunks)
    n = len(batches)
    by_index: dict[int, str] = {}
    described: list[tuple[str, str]] = []
    running_summary = ""

    for b, batch in enumerate(batches):
        carry = _carry_context(running_summary, described)
        user = _batch_prompt(file_label, b, n, carry, batch)
        parsed, _usage = _call_describe(client, user)
        if _usage is not None and customer_id:
            await record_model_call(
                customer_id=customer_id,
                origin="code_descriptions",
                model=_usage.model,
                input_tokens=_usage.input_tokens,
                output_tokens=_usage.output_tokens,
                cache_creation_tokens=_usage.cache_creation_tokens,
                cache_read_tokens=_usage.cache_read_tokens,
                store=store,
            )
        if not isinstance(parsed, dict):
            logger.warning(
                "code_describer.batch_unparsed",
                extra={"label": file_label, "batch": b, "of": n},
            )
            continue
        fs = parsed.get("file_summary")
        if isinstance(fs, str) and fs.strip():
            running_summary = fs.strip()
        entries = [(gidx, _label(c, gidx)) for (gidx, c, _code) in batch]
        bi, desc = _collect(parsed, entries)
        by_index.update(bi)
        described.extend(desc)

    # End-of-file synopsis over the complete description set; fall back to the
    # last batch's running summary if the synopsis call fails.
    _syn, _usage = _synopsize(client, file_label, described)
    if _usage is not None and customer_id:
        await record_model_call(
            customer_id=customer_id,
            origin="code_descriptions",
            model=_usage.model,
            input_tokens=_usage.input_tokens,
            output_tokens=_usage.output_tokens,
            cache_creation_tokens=_usage.cache_creation_tokens,
            cache_read_tokens=_usage.cache_read_tokens,
            store=store,
        )
    file_summary = _syn or running_summary
    return {"file_summary": file_summary, "by_index": by_index}


async def describe_code_file(
    *,
    file_text: str = "",
    chunks: list[dict],
    client: Any,
    file_label: str = "",
    customer_id: Optional[str] = None,
    store: Any = None,
) -> dict[str, Any]:
    """Functional descriptions for a file's symbols.

    Whole-file mode when the file fits in one call (full file in view), else
    sequential code-budget batches with carry-forward + an end-of-file synopsis
    pass. Returns {"file_summary": str, "by_index": {global_chunk_index: str}}.
    Best-effort: failures leave symbols undescribed so they fall back to
    indexing the verbatim code. No client / no text → empty.
    """
    empty: dict[str, Any] = {"file_summary": "", "by_index": {}}
    if client is None or not _has_text(chunks):
        return empty
    if len(file_text or "") <= WHOLE_FILE_BUDGET:
        return await _describe_whole_file(
            file_text or "", chunks, client, file_label,
            customer_id=customer_id, store=store,
        )
    return await _describe_batched(
        chunks, client, file_label, customer_id=customer_id, store=store,
    )


def _parse_json_object(text: str) -> Optional[dict]:
    """Lenient JSON-object parse (object sibling of
    document_pipeline._parse_json_array): plain, ```json-fenced, or the
    first {...} span embedded in surrounding prose."""
    try:
        r = json.loads(text)
        if isinstance(r, dict):
            return r
    except json.JSONDecodeError:
        pass
    if "```" in text:
        inner = text.split("```")[1]
        if inner.startswith("json"):
            inner = inner[4:]
        try:
            r = json.loads(inner.strip())
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            r = json.loads(text[start:end + 1])
            if isinstance(r, dict):
                return r
        except json.JSONDecodeError:
            pass
    return None
