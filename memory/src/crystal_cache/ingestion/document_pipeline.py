"""Document Ingestion Pipeline — LLM-powered document crystallization.

Extracts structured knowledge from documents using an LLM, then
stores each piece of knowledge as a crystal in the customer's bank.

Pipeline per document:
  1. Extract raw text from file (PDF, docx, txt)
  2. Chunk into sections (by paragraph/page/heading)
  3. For each chunk, LLM extracts structured knowledge:
     - Key facts and rules
     - Entities and relationships
     - Q&A pairs (what questions does this content answer?)
     - Procedural steps
  4. Each extracted item becomes a crystal via add_pair_for_customer
  5. Document status updated to 'crystallized'
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, TYPE_CHECKING

from ..retrieval.sparse_key import format_key
from ..llm import get_llm_client
from .injection_screen import scan_for_injection

if TYPE_CHECKING:
    from ..encoding.semantic import SemanticTextEncoder
    from ..infrastructure.metadata_store import MetadataStore
    from ..infrastructure.vector_store import VectorStore

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM = """You are a knowledge extraction engine. You receive a
section of a document and extract structured knowledge from it.

For each section, produce a JSON array of knowledge items. Each item has:
- "key": A short, specific retrieval key (what question would someone ask
  to find this information? 5-15 words)
- "segments": An ordered list of 2-5 short strings naming WHERE this
  knowledge sits, from GENERAL (first) to SPECIFIC (last). Each segment is
  1-4 plain words, no "|" character. Broad category first, exact subject
  last. Examples:
    ["Film", "Corporate Mistletoe", "Characters", "Shawna"]
    ["Healthcare", "Employee Handbook", "PTO", "accrual rate"]
- "value": The complete answer/fact (1-3 sentences, self-contained,
  includes enough context to be useful without the original document)
- "type": One of:
  - "fact" — a specific rule, policy, number, or requirement
  - "entity" — information about a person, organization, or place
  - "relationship" — how two entities relate to each other
  - "process" — a step-by-step procedure or workflow
  - "definition" — what a term or concept means in this context
  - "qa" — a natural question-answer pair this content answers

Extract EVERY piece of useful, specific knowledge. Be thorough.
Do NOT include:
- Boilerplate headers/footers
- Page numbers or formatting artifacts
- Vague or generic statements that don't contain specific information

Each item must be self-contained — someone reading just the "value"
should understand it without needing the original document.

Return ONLY a JSON array, no markdown, no explanation:
[
  {"key": "...", "segments": ["General", "...", "Specific"], "value": "...", "type": "fact"},
  {"key": "...", "segments": ["General", "...", "Specific"], "value": "...", "type": "entity"},
  ...
]"""


@dataclass
class ExtractionItem:
    key: str
    value: str
    item_type: str
    sparse_key: str = ""
    chunk_index: int = 0
    crystal_id: Optional[str] = None
    fact_id: Optional[str] = None


@dataclass
class CrystallizationResult:
    document_id: str
    customer_id: str
    chunks_processed: int = 0
    items_extracted: int = 0
    crystals_written: int = 0
    errors: int = 0
    items: list[ExtractionItem] = field(default_factory=list)
    error: Optional[str] = None


def stamps_for_source(
    scope, owner_operator_id, customer_id: str,
) -> dict:
    """P2 scope-on-sources (ratified 2026-07-02): the add_pair stamp kwargs
    for crystals born from a source with the given scope. None scope =
    legacy source → team-scoped unowned crystals (today's exact behavior);
    'personal'/'team' → owned crystals at 0o600/0o640 with the customer as
    the POSIX group. One helper so every pipeline write site stamps
    identically."""
    if scope is None:
        return {}
    from ..infrastructure.permissions import mode_for_scope

    return {
        "owner_operator_id": owner_operator_id,
        "group_team_id": customer_id,
        "mode": mode_for_scope(scope),
    }


def recall_stamps(origin: str) -> dict:
    """The add_pair stamp kwargs for recall-gating + birth attribution
    (2026-07-03). Background-worker output is born recall_gated (held out of
    recall until approved) and origin-tagged; everything else is born
    ungated/direct exactly as before. One helper so every pipeline write
    site stamps identically, mirroring stamps_for_source.

    'direct' (the default) => {} => add_pair defaults (recall_gated=False,
    origin='direct'), i.e. zero behavior change. Any non-direct origin (the
    only one today being 'background_worker') => born gated.
    """
    if origin == "direct":
        return {}
    return {"origin": origin, "recall_gated": True}


class DocumentPipeline:
    def __init__(self, store, encoder, vector_store, *, vector_index=None, fact_vector_store=None, client=None):
        self._store = store
        self._encoder = encoder
        self._vector_store = vector_store
        self._vector_index = vector_index
        # Optional FactVectorStore handle. When provided, the approval
        # path invalidates it after deletes/writes so replaced facts
        # stop surfacing and new facts appear without a restart.
        self._fact_vector_store = fact_vector_store
        # Optional injected LLM client (LLMClient-shaped: exposes complete()).
        # Callers that already hold one pass it here; None falls back to the
        # shared provider-neutral seam via get_llm_client().
        self._client = client

    def _get_client(self):
        # Injected client wins (tests, or a caller holding its own);
        # otherwise the shared provider-neutral seam.
        return self._client if self._client is not None else get_llm_client()

    async def crystallize_document(
        self, customer_id, document_id, text, *,
        label="", crystal_type="customer:legacy", chunk_size=3000,
        scope=None, owner_operator_id=None,
    ) -> CrystallizationResult:
        result = CrystallizationResult(document_id=document_id, customer_id=customer_id)

        chunks = self._chunk_text(text, chunk_size)
        result.chunks_processed = len(chunks)
        logger.info("document_pipeline.chunked", extra={"document_id": document_id, "chunks": len(chunks)})

        for i, chunk in enumerate(chunks):
            try:
                # Offload the synchronous LLM extraction off the event loop
                # (the whole sync helper in one hop, like executor.run_encoder_bound)
                # so inline crystallize endpoints + the worker don't block the API.
                items = await asyncio.to_thread(
                    self._extract_knowledge, chunk, label, i
                )
                for item in items:
                    item.chunk_index = i
                    result.items.append(item)
                    result.items_extracted += 1
            except Exception as e:
                logger.error("document_pipeline.extraction_failed", extra={"chunk": i, "error": str(e)})
                result.errors += 1

        for item in result.items:
            try:
                pair_type_map = {
                    "fact": "question_answer", "entity": "entity_attribute",
                    "relationship": "entity_relationship", "process": "question_answer",
                    "definition": "question_answer", "qa": "question_answer",
                }
                pair_type = pair_type_map.get(item.item_type, "question_answer")

                # Unified sparse key from extraction (already clean); fall
                # back to a single clean segment from the retrieval key.
                sk = item.sparse_key or format_key(" ".join(item.key.split()[:8]))

                crystal, fact = await self._store.add_pair_for_customer(
                    customer_id=customer_id, prompt_text=sk,
                    answer_text=item.value, pair_type=pair_type,
                    encoder=self._encoder, vector_store=self._vector_store,
                    vector_index=self._vector_index,
                    crystal_type=crystal_type, source_kind="model_reasoning",
                    **stamps_for_source(scope, owner_operator_id, customer_id),
                    **recall_stamps(origin),
                )
                item.crystal_id = crystal.id
                item.fact_id = fact.id
                result.crystals_written += 1
            except Exception as e:
                import traceback
                print(f"STORE_FAILED: {type(e).__name__}: {e}")
                traceback.print_exc()
                result.errors += 1

        logger.info("document_pipeline.complete", extra={
            "document_id": document_id, "crystals": result.crystals_written,
            "items": result.items_extracted, "errors": result.errors,
        })
        return result

    async def extract_items(
        self, text: str, *, label: str = "",
        crystal_type: str = "customer:legacy", chunk_size: int = 3000,
    ) -> list[ExtractionItem]:
        """Extract knowledge items from text WITHOUT writing crystals.

        Returns the extracted items for review. The user can edit/delete
        items before calling approve_and_crystallize to write them.
        """
        chunks = self._chunk_text(text, chunk_size)
        all_items: list[ExtractionItem] = []

        for i, chunk in enumerate(chunks):
            try:
                # Offload the synchronous LLM extraction off the event loop
                # (see crystallize_document above) so the extraction loop can't
                # freeze the API while a document is being processed.
                items = await asyncio.to_thread(
                    self._extract_knowledge, chunk, label, i
                )
                for item in items:
                    item.chunk_index = i
                    all_items.append(item)
            except Exception as e:
                logger.error("document_pipeline.extraction_failed", extra={"chunk": i, "error": str(e)})

        logger.info("document_pipeline.extracted", extra={
            "label": label, "items": len(all_items), "chunks": len(chunks),
        })
        return all_items

    async def approve_and_crystallize(
        self, customer_id: str, document_id: str,
        items: list[dict], content_chunks: list[dict],
        *, crystal_type: str = "customer:legacy",
        scope=None, owner_operator_id=None, origin: str = "direct",
    ) -> CrystallizationResult:
        """Write approved items and content chunks as crystals.

        Called after user reviews and approves extracted items.
        Content chunks become crystals with pair_type='content_chunk'.
        Knowledge items become crystals with their original pair_types.

        origin (2026-07-03): 'direct' (default) => crystals born usable,
        exactly as before. 'background_worker' => crystals born recall_gated
        (held out of recall until a human or a system_rules promotion rule
        approves them), because autonomous workers ran networked and
        unattended and their output must not be relied on until reviewed.
        """
        result = CrystallizationResult(document_id=document_id, customer_id=customer_id)

        # --- Content chunks (Layer 1: verbatim text) ---
        # Source versioning + dedup (VS-D2/D3, REPLACE semantics, locked
        # 2026-06-10): every content chunk carries a source_path — the
        # file path for code (locators are 'path::symbol'), the document
        # label otherwise — and all chunks of a path share one content
        # hash. On re-ingest: unchanged hash -> skip the path entirely
        # (dedup); changed hash -> DELETE the prior crystals for that
        # path (facts die with them; the HDC codebook dies with the
        # crystal row, so no grating surgery) and write a fresh set.
        # No stale crystals are ever kept — there is no is_current flag
        # and no supersede chain. One-crystal-per-file bundling (VS-D1)
        # is the follow-up grain change.
        doc_row = await self._store.get_document_upload(document_id, customer_id)
        doc_label = (getattr(doc_row, "label", "") or "") if doc_row else ""
        doc_source_modified_at = (
            getattr(doc_row, "source_modified_at", None) if doc_row else None
        )

        def _source_path(chunk: dict) -> str:
            if chunk.get("doc_type") == "code":
                return _file_path_for_chunk(chunk)
            return doc_label or chunk.get("label", "") or "unknown"

        # Group every non-empty chunk by source path; one hash per path.
        by_path: dict[str, list[dict]] = {}
        for chunk in content_chunks:
            if (chunk.get("text") or "").strip():
                by_path.setdefault(_source_path(chunk), []).append(chunk)
        path_hashes: dict[str, str] = {
            p: _content_hash_for_chunks(cs) for p, cs in by_path.items()
        }

        # Resolve skip-vs-replace per path BEFORE writing anything.
        skip_paths: set[str] = set()
        if by_path:
            existing_crystals = await self._store.list_crystals_for_customer(
                customer_id
            )
            for file_path, file_hash in path_hashes.items():
                current = [
                    c for c in existing_crystals if c.source_path == file_path
                ]
                if not current:
                    continue
                if all(c.content_hash == file_hash for c in current):
                    # Unchanged source — keep the existing crystals and
                    # skip re-writing this path (dedup).
                    skip_paths.add(file_path)
                    logger.info("document_pipeline.source_unchanged_skipped", extra={
                        "source_path": file_path,
                        "existing_crystals": len(current),
                    })
                    continue
                # Changed source — REPLACE: delete the prior crystals.
                # delete_crystal invalidates the routing cache per call so
                # the write loop below can't bond into a deleted crystal.
                deleted = 0
                for old in current:
                    if await self._store.delete_crystal(
                        old.id,
                        customer_id,
                        vector_store=self._vector_store,
                        fact_vector_store=self._fact_vector_store,
                    ):
                        deleted += 1
                logger.info("document_pipeline.source_replaced", extra={
                    "source_path": file_path,
                    "crystals_deleted": deleted,
                })

        # Write content chunks. Chunks of an unchanged source path are
        # skipped; everything else is written, and every content-chunk
        # crystal is stamped with its source-version fields.
        for chunk in content_chunks:
            try:
                label = chunk.get("label", f"Chunk {chunk.get('index', 0)}")
                text = chunk.get("text", "")
                if not text.strip():
                    continue

                # Build the unified sparse key, wide -> specific.
                locator = chunk.get("locator", label)
                doc_type = chunk.get("doc_type", "general")
                source_map = {
                    "script": "Script", "policy": "Policy", "contract": "Contract",
                    "transcript": "Transcript", "technical": "Docs", "general": "Document",
                    "code": "Code",
                }
                source = source_map.get(doc_type, "Document")
                subject = chunk.get("subject") or ""
                domain = chunk.get("domain", "")
                if doc_type == "code":
                    # Code locators are "path::symbol": Code | <file path> | <symbol>.
                    sk = format_key(["Code", *str(locator).split("::")])
                else:
                    # domain | subject | source | locator (empties dropped).
                    sk = format_key([domain, subject, source, locator])

                file_path = _source_path(chunk)
                if file_path in skip_paths:
                    continue  # unchanged source; existing crystals kept as-is

                crystal, fact = await self._store.add_pair_for_customer(
                    customer_id=customer_id, prompt_text=sk,
                    answer_text=text, pair_type="content_chunk",
                    encoder=self._encoder, vector_store=self._vector_store,
                    vector_index=self._vector_index,
                    crystal_type=crystal_type, source_kind="document_chunk",
                    embed_text=chunk.get("description") or None,
                    **stamps_for_source(scope, owner_operator_id, customer_id),
                    **recall_stamps(origin),
                )

                # VS-D2: stamp source-version fields on EVERY content-chunk
                # crystal so a later re-ingest can dedup/replace it. Don't
                # steal a crystal already stamped for a DIFFERENT source
                # (cross-file/document bonding): overwriting would make a
                # later replace of the original source delete this one's
                # facts too. Log and leave the original stamp; the true
                # fix is VS-D1 (one crystal per file, no shared routing).
                if crystal.source_path and crystal.source_path != file_path:
                    logger.warning("document_pipeline.shared_crystal_stamp_skipped", extra={
                        "crystal_id": crystal.id,
                        "stamped_source": crystal.source_path,
                        "this_source": file_path,
                    })
                else:
                    crystal.source_path = file_path
                    crystal.content_hash = path_hashes.get(file_path)
                    crystal.source_modified_at = doc_source_modified_at
                    await self._store.upsert_crystal(crystal)

                result.crystals_written += 1

                # C2 mitigation (2026-07-03): screen ingested chunk text for
                # prompt-injection shapes. A hit quarantines the crystal (the
                # tier signal then tells the model to distrust it) rather than
                # blocking ingestion — a heuristic layer atop the C1 fence, not
                # a content filter. Fail-safe: a screening error never breaks
                # the write.
                try:
                    _hits = scan_for_injection(text)
                    if _hits:
                        await self._store.set_crystal_quality_tier(
                            crystal.id, customer_id, "quarantine",
                        )
                        logger.warning(
                            "document_pipeline.chunk_quarantined_injection",
                            extra={
                                "crystal_id": crystal.id,
                                "source_path": file_path,
                                "patterns": _hits,
                            },
                        )
                except Exception as _scan_err:  # noqa: BLE001
                    logger.error(
                        "document_pipeline.injection_scan_failed",
                        extra={"crystal_id": crystal.id,
                               "error": str(_scan_err)},
                    )

                logger.info("document_pipeline.chunk_written", extra={
                    "label": label, "crystal_id": crystal.id, "sparse_key": sk,
                })
            except Exception as e:
                logger.error("document_pipeline.chunk_write_failed", extra={
                    "label": label, "error": str(e), "error_type": type(e).__name__,
                })
                import traceback
                traceback.print_exc()
                result.errors += 1

        # Write knowledge items as crystals (Layer 2: extracted knowledge)
        for item in items:
            try:
                pair_type_map = {
                    "fact": "question_answer", "entity": "entity_attribute",
                    "relationship": "entity_relationship", "process": "question_answer",
                    "definition": "question_answer", "qa": "question_answer",
                }
                pair_type = pair_type_map.get(item.get("type", ""), "question_answer")

                sk = item.get("sparse_key", "")
                if not sk:
                    sk = format_key(" ".join(item.get("key", "").split()[:8]))

                crystal, fact = await self._store.add_pair_for_customer(
                    customer_id=customer_id, prompt_text=sk,
                    answer_text=item.get("value", ""), pair_type=pair_type,
                    encoder=self._encoder, vector_store=self._vector_store,
                    vector_index=self._vector_index,
                    crystal_type=crystal_type, source_kind="model_reasoning",
                    **stamps_for_source(scope, owner_operator_id, customer_id),
                    **recall_stamps(origin),
                )
                # Share-source provenance (P4, ratified 2026-07-02): record
                # which crystal each approved item landed in, so 'share this
                # document' can resolve its full crystal set. The caller
                # persists the mutated items back onto the upload row.
                item["crystal_id"] = crystal.id
                result.crystals_written += 1
                result.items_extracted += 1
            except Exception as e:
                logger.error("document_pipeline.item_write_failed", extra={"key": item.get("key"), "error": str(e)})
                result.errors += 1

        # New facts were written (and possibly replaced sources deleted):
        # drop the fact-search cache so the next query sees the current
        # bank rather than a snapshot from before this approval.
        if self._fact_vector_store is not None:
            self._fact_vector_store.invalidate(customer_id)

        logger.info("document_pipeline.crystallized", extra={
            "document_id": document_id, "crystals": result.crystals_written,
            "items": result.items_extracted, "errors": result.errors,
        })
        return result

    def _chunk_text(self, text, chunk_size=3000):
        paragraphs = text.split("\n\n")
        chunks, current = [], ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) + 2 <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) > chunk_size:
                    sentences = para.replace(". ", ".\n").split("\n")
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) + 1 <= chunk_size:
                            current = (current + " " + sent).strip()
                        else:
                            if current:
                                chunks.append(current)
                            current = sent
                else:
                    current = para
        if current:
            chunks.append(current)
        return chunks

    def _extract_knowledge(self, chunk, label, chunk_index):
        client = self._get_client()
        context = (f"Document: {label}\n" if label else "") + f"Section {chunk_index + 1}:\n\n{chunk}"
        try:
            text = client.complete(
                system=EXTRACTION_SYSTEM,
                messages=[{"role": "user", "content": context}],
                max_tokens=4000,
                temperature=0.0,
                tier="small",
            )
            items_data = self._parse_json_array(text)
            if items_data is None:
                return []
            items: list[ExtractionItem] = []
            for d in items_data:
                if not (d.get("key") and d.get("value")):
                    continue
                segments = d.get("segments")
                if isinstance(segments, str):
                    segments = [segments]
                # The key is always format_key output -> sanitized, '|'-free,
                # wide->specific. Freeform pipe junk can no longer leak in.
                sk = format_key(segments or [])
                if not sk:
                    # Fallback: one clean segment from the retrieval key.
                    sk = format_key(" ".join(str(d.get("key", "")).split()[:8]))
                items.append(ExtractionItem(
                    key=d.get("key", ""),
                    value=d.get("value", ""),
                    item_type=d.get("type", "fact"),
                    sparse_key=sk,
                ))
            return items
        except Exception as e:
            logger.error("document_pipeline.llm_failed", extra={"chunk": chunk_index, "error": str(e)})
            return []

    @staticmethod
    def _parse_json_array(text):
        try:
            r = json.loads(text)
            if isinstance(r, list):
                return r
        except json.JSONDecodeError:
            pass
        if "```" in text:
            inner = text.split("```")[1]
            if inner.startswith("json"):
                inner = inner[4:]
            try:
                r = json.loads(inner.strip())
                if isinstance(r, list):
                    return r
            except json.JSONDecodeError:
                pass
        start, end = text.find("["), text.rfind("]")
        if start >= 0 and end > start:
            try:
                r = json.loads(text[start:end+1])
                if isinstance(r, list):
                    return r
            except json.JSONDecodeError:
                pass
        return None


def _file_path_for_chunk(chunk: dict) -> str:
    """Source file path for a code chunk: the part of the locator before
    '::' (locators are 'path::symbol'), falling back to the chunk label.
    Stable across re-uploads of the same file, so it keys dedup/supersede.
    """
    locator = chunk.get("locator", "") or ""
    head = locator.split("::", 1)[0].strip()
    return head or chunk.get("label", "") or "unknown"


def _content_hash_for_chunks(chunks: list[dict]) -> str:
    """Stable SHA-256 over a file's chunks for change-detection / dedup.

    Sorted by locator so chunk ordering doesn't affect the result; each
    entry combines locator + text so renaming a symbol or editing its
    body both change the hash.
    """
    parts = sorted(
        f"{c.get('locator', '')}\n{c.get('text', '')}" for c in chunks
    )
    return hashlib.sha256("\n--\n".join(parts).encode("utf-8")).hexdigest()
