"""Structural critics — store-signal scanners that critique the bank's
ARTIFACTS (S6, 2026-07-08).

The MCR critics (agent_self, shadow) review reasoning; nothing reviewed
what ingestion actually produced. This module adds the first structural
critic: the blob-fact detector. A single-fact crystal whose one claim
exceeds the atomic ceiling is the fingerprint of content that skipped
extraction (the erahq.ai learn incident — a whole site jammed into one
fact). Each finding is filed as a Critique(critic_role='structural',
critic_model='store-signal') + ActionItem(action_type=
'substrate_observation', subsystem='ingestion') — the same channel the
agent's own in-trace complaints use, so the Critiques surface presents
reasoning-critiques and artifact-critiques side by side, grouped by
subsystem.

Per Principle 9 / D-MCR-15: observations are recorded and surfaced,
NEVER auto-acted. The detector does not fix, delete, or re-extract
anything.

Scope note (Anthony, 2026-07-08): the substrate channel reaches EVERY
part of the system that affects outcomes — tool capability wishes,
metacognition misses, retrieval quality. This detector is one writer
among what should become many; keep new structural critics in this
module.

Idempotence: one open substrate_observation per crystal — the scan
skips crystals already named in a pending/deferred observation's
content.crystal_id.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from ..agent.tools.memory import CRYSTAL_WRITE_MAX_VALUE_CHARS

if TYPE_CHECKING:
    from ..infrastructure.metadata_store import MetadataStore

logger = structlog.get_logger(__name__)

# One scan pass caps its findings so a legacy bank can't flood the
# critique stream in one cycle; the next cadence picks up the rest.
_SCAN_BATCH = 10


async def run_structural_ingestion_scan(
    *,
    store: "MetadataStore",
    min_claim_chars: int = CRYSTAL_WRITE_MAX_VALUE_CHARS,
    limit: int = _SCAN_BATCH,
) -> dict[str, int]:
    """One blob-fact scan pass. Store-signal only — no model calls, so
    no budget row needed. Never raises."""
    out = {"found": 0, "filed": 0, "skipped_existing": 0}
    try:
        blobs = await store.list_blob_facts(
            min_claim_chars=min_claim_chars, limit=limit
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("structural_scan.read_failed", error=str(e))
        return out
    if not blobs:
        return out
    out["found"] = len(blobs)

    # Idempotence set: crystals already named by an open observation.
    seen: set[str] = set()
    try:
        pending = await store.list_substrate_action_items(limit=500)
        for item in pending:
            cid = (item.content or {}).get("crystal_id")
            if cid and item.status in ("pending", "deferred"):
                seen.add(cid)
    except Exception as e:  # noqa: BLE001
        logger.warning("structural_scan.dedupe_read_failed", error=str(e))

    for blob in blobs:
        if blob["crystal_id"] in seen:
            out["skipped_existing"] += 1
            continue
        complaint = (
            f"Ingestion produced a malformed artifact: crystal "
            f"{blob['crystal_id']} holds ONE fact of "
            f"{blob['claim_chars']} chars (atomic ceiling "
            f"{min_claim_chars}). This is content that should have been "
            f"extracted into individual facts plus a context chunk "
            f"(key: {blob['sample_key'][:120]})."
        )
        try:
            critique = await store.create_critique(
                blob["customer_id"],
                "structural",
                "store-signal",
                observations=[{
                    "type": "substrate_complaint",
                    "text": complaint,
                    "confidence": 1.0,
                    "anchors": [],
                }],
                summary_text="Structural scan: blob-shaped fact detected.",
                total_action_items=1,
            )
            item = await store.create_action_item(
                critique.id,
                blob["customer_id"],
                "substrate_observation",
                content={
                    "subsystem": "ingestion",
                    "complaint": complaint,
                    "severity": "medium",
                    "crystal_id": blob["crystal_id"],
                    "fact_id": blob["fact_id"],
                    "claim_chars": blob["claim_chars"],
                },
                critic_confidence=1.0,
            )
            # Substrate items are recorded, DEFERRED, surfaced (Principle
            # 9). Agent-written ones get deferred by synthesis; a
            # structural critic's are born deferred — nothing to promote.
            await store.update_action_item_status(item.id, "deferred")
            seen.add(blob["crystal_id"])
            out["filed"] += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "structural_scan.file_failed",
                crystal_id=blob["crystal_id"], error=str(e),
            )
    if out["filed"]:
        logger.info("structural_scan.pass_complete", **out)
    return out
