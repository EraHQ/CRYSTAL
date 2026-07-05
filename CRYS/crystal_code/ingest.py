"""Codebase ingestion for the coding agent CLI — /ingest and --ingest.

Drives the SAME pipeline the server uses, with no server running:
per file, create a DocumentUpload row, run the chunk/extract workflow
(workers.crystallization.crystallize_document), then auto-approve via
DocumentPipeline.approve_and_crystallize. Reusing that exact path means
source stamping and the VS-D2/D3 REPLACE semantics apply unchanged: an
unchanged file is skipped by content hash, a changed file replaces its
prior crystals, and a killed run is naturally resumable by re-running.

Cost shape worth knowing: code files skip LLM extraction entirely (the
verbatim per-symbol chunks ARE the code knowledge), so ingesting a
codebase is mostly LLM-free. Only prose docs (md/txt/rst) hit the
extraction model. The wizard's cost preview reflects that split.

File selection honors .gitignore the robust way — `git ls-files` when
the folder is a git repo (tracked + untracked-not-ignored), falling
back to a directory walk with default excludes elsewhere. User globs
and the guard's block_paths are excluded in both modes.
"""
from __future__ import annotations

import fnmatch
import hashlib
import math
import os
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from crystal_cache.ingestion.document_chunker import CODE_EXTENSIONS

DOC_EXTENSIONS = (".md", ".txt", ".rst")
MAX_FILE_BYTES = 1_000_000
# Mirrors DocumentPipeline's default chunk_size — used only to ESTIMATE
# extraction calls for the wizard's cost preview, never to chunk.
EXTRACTION_CHUNK_CHARS = 3000

# Walk-mode excludes (non-git folders). Git mode doesn't need these —
# .gitignore already covers them in any sane repo — but they're applied
# there too as a backstop for repos that track junk.
DEFAULT_EXCLUDE_DIRS = {
    ".git", ".hg", ".svn", "node_modules", ".venv", "venv", "env",
    "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    "dist", "build", "target", ".next", ".nuxt", "coverage",
    "htmlcov", ".tox", ".idea", ".vscode", ".eggs",
}


# ---------------------------------------------------------------------------
# Scanning — which files are ingestible
# ---------------------------------------------------------------------------

@dataclass
class ScanResult:
    code_files: list[str] = field(default_factory=list)   # rel posix paths
    doc_files: list[str] = field(default_factory=list)    # rel posix paths
    doc_chars: int = 0      # total bytes of prose docs, for the cost estimate
    oversized: int = 0      # skipped: larger than MAX_FILE_BYTES
    used_git: bool = False  # whether .gitignore was honored via git ls-files

    @property
    def files(self) -> list[str]:
        return self.code_files + self.doc_files

    @property
    def estimated_extraction_calls(self) -> int:
        return math.ceil(self.doc_chars / EXTRACTION_CHUNK_CHARS) if self.doc_chars else 0


def _git_file_list(root: Path) -> Optional[list[str]]:
    """Candidate files via git: tracked + untracked-but-not-ignored.

    This is how .gitignore is honored without reimplementing its
    semantics. Returns rel posix paths, or None when the folder isn't a
    git repo (or git isn't available) — the caller falls back to a walk.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "--cached", "--others",
             "--exclude-standard", "-z"],
            capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    seen: set[str] = set()
    files: list[str] = []
    for rel in out.stdout.split("\0"):
        rel = rel.strip()
        if rel and rel not in seen:
            seen.add(rel)
            files.append(rel)
    return files


def _walk_file_list(root: Path) -> list[str]:
    """Directory-walk fallback for non-git folders, pruning junk dirs."""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(root).as_posix()
            files.append(rel)
    return files


def _matches_any(rel: str, patterns: list[str]) -> bool:
    """Glob match against a rel posix path. A bare name or dir pattern
    ("tests", "docs/legacy") excludes the whole subtree, so users don't
    have to remember to type the `/**`."""
    for raw in patterns:
        p = raw.strip().rstrip("/")
        if not p:
            continue
        if fnmatch.fnmatch(rel, p) or fnmatch.fnmatch(rel, p + "/*") \
                or rel == p or rel.startswith(p + "/"):
            return True
    return False


def scan_project(
    root: Path,
    *,
    extra_excludes: Optional[list[str]] = None,
    include_docs: bool = True,
) -> ScanResult:
    """Enumerate ingestible files under root.

    Selection: code extensions (the chunker's CODE_EXTENSIONS — single
    source of truth) plus prose docs (md/txt/rst) when include_docs.
    Excluded: user globs, anything in a default-excluded dir, and files
    over MAX_FILE_BYTES.
    """
    excludes = list(extra_excludes or [])
    result = ScanResult()

    candidates = _git_file_list(root)
    result.used_git = candidates is not None
    if candidates is None:
        candidates = _walk_file_list(root)

    for rel in sorted(candidates):
        parts = rel.split("/")
        if any(part in DEFAULT_EXCLUDE_DIRS for part in parts[:-1]):
            continue
        if excludes and _matches_any(rel, excludes):
            continue
        low = rel.lower()
        is_code = low.endswith(CODE_EXTENSIONS)
        is_doc = include_docs and low.endswith(DOC_EXTENSIONS)
        if not (is_code or is_doc):
            continue
        full = root / rel
        try:
            size = full.stat().st_size
        except OSError:
            continue
        if size > MAX_FILE_BYTES:
            result.oversized += 1
            continue
        if is_code:
            result.code_files.append(rel)
        else:
            result.doc_files.append(rel)
            result.doc_chars += size
    return result


def read_project_file(path: Path) -> Optional[str]:
    """Read a file as text; None means binary/unreadable (skip it).

    Line endings are normalized to LF so ingestion is line-ending
    invariant: on Windows, a `git checkout` can rewrite an untouched
    file with CRLF (autocrlf), and without normalization that reads as
    a content change — re-ingesting and re-crystallizing a file whose
    text never meaningfully changed."""
    try:
        data = path.read_bytes()
    except OSError:
        return None
    if b"\x00" in data[:8192]:
        return None  # binary masquerading behind a text extension
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


# ---------------------------------------------------------------------------
# Ingestion — drive the server pipeline per file
# ---------------------------------------------------------------------------

@dataclass
class FileOutcome:
    path: str
    status: str            # "written" | "unchanged" | "failed"
    chunks: int = 0
    items: int = 0
    crystals: int = 0
    error: str = ""


@dataclass
class IngestSummary:
    outcomes: list[FileOutcome] = field(default_factory=list)

    @property
    def written(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "written")

    @property
    def unchanged(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "unchanged")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")

    @property
    def crystals(self) -> int:
        return sum(o.crystals for o in self.outcomes)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _first_line(err: object) -> str:
    """One readable line for an outcome's error — SQLAlchemy errors carry
    the full SQL statement, which is noise at the progress line."""
    s = str(err).strip()
    return s.splitlines()[0][:300] if s else "unknown error"


async def _prior_doc_hashes(store: Any, customer_id: str) -> dict[str, set[str]]:
    """label -> hashes of previously crystallized document texts.

    The upfront unchanged-file skip: identical text under the same label
    means no new row, no re-chunk, and — for prose — no re-extraction
    LLM calls. (The pipeline's own content-hash dedup still protects the
    changed/partial cases downstream.) Best-effort: any failure here
    just disables the optimization.
    """
    prior: dict[str, set[str]] = {}
    try:
        docs = await store.list_document_uploads(
            customer_id=customer_id, status="crystallized", limit=None,
        )
        for d in docs:
            text = getattr(d, "text", None)
            label = getattr(d, "label", None)
            if text and label:
                prior.setdefault(label, set()).add(_text_hash(text))
    except Exception:
        return {}
    return prior


async def ingest_files(
    root: Path,
    rel_paths: list[str],
    *,
    store: Any,
    encoder: Any,
    vector_store: Any,
    fact_vector_store: Any,
    customer_id: str,
    client: Any = None,
    progress: Optional[Callable[[int, int, FileOutcome], None]] = None,
) -> IngestSummary:
    """Ingest files into the customer's bank via the server pipeline.

    Per file: create_document_upload (label = rel posix path, which the
    code chunker uses as the path in `path::symbol` locators — the key
    that makes replace-on-change work across re-ingests) → the worker's
    crystallize_document (chunk + extract) → auto-approve →
    approve_and_crystallize → mark crystallized. Failures are recorded
    and skipped; the run never aborts on one bad file.
    """
    from crystal_cache.ingestion.document_pipeline import DocumentPipeline
    from crystal_cache.workers.crystallization import crystallize_document

    pipeline = DocumentPipeline(
        store=store,
        encoder=encoder,
        vector_store=vector_store,
        fact_vector_store=fact_vector_store,
        client=client,
    )
    prior = await _prior_doc_hashes(store, customer_id)
    summary = IngestSummary()
    total = len(rel_paths)

    for i, rel in enumerate(rel_paths):
        outcome = FileOutcome(path=rel, status="failed")
        try:
            text = read_project_file(root / rel)
            if text is None:
                outcome.error = "binary or unreadable"
            elif not text.strip():
                outcome.status = "unchanged"
                outcome.error = "empty file"
            elif _text_hash(text) in prior.get(rel, set()):
                outcome.status = "unchanged"
            else:
                outcome = await _ingest_one(
                    store=store, pipeline=pipeline, encoder=encoder,
                    vector_store=vector_store, customer_id=customer_id,
                    rel=rel, text=text, client=client,
                    crystallize_document=crystallize_document,
                )
        except Exception as e:  # noqa: BLE001 — one bad file never kills the run
            outcome = FileOutcome(
                path=rel, status="failed",
                error=f"{type(e).__name__}: {_first_line(e)}",
            )
        summary.outcomes.append(outcome)
        if progress is not None:
            progress(i + 1, total, outcome)
    return summary


async def _ingest_one(
    *, store: Any, pipeline: Any, encoder: Any, vector_store: Any,
    customer_id: str, rel: str, text: str, client: Any,
    crystallize_document: Any,
) -> FileOutcome:
    """One file through the full server flow. Mirrors endpoints/documents.py
    upload + approve, minus the human review stop."""
    doc = await store.create_document_upload(
        customer_id=customer_id, label=rel, text=text,
        crystal_type="customer:legacy",
    )

    # Chunk + extract (sets status to 'review' on success, 'error' on
    # failure). Code files skip the LLM extractor inside this call.
    await crystallize_document(
        store=store, encoder=encoder, vector_store=vector_store,
        document_id=doc.id, client=client,
    )
    doc_after = await store.get_document_upload(doc.id, customer_id)
    if doc_after is None or doc_after.status == "error":
        msg = getattr(doc_after, "error_message", None) or "extraction failed"
        return FileOutcome(path=rel, status="failed", error=_first_line(msg))

    items = doc_after.extracted_items or []
    content_chunks = doc_after.content_chunks or []

    # Auto-approve: same atomic transition + pipeline call as the
    # /v1/documents/{id}/approve endpoint.
    await store.save_approval_edits_and_mark_crystallizing(
        document_id=doc.id, items=items, content_chunks=content_chunks,
    )
    try:
        result = await pipeline.approve_and_crystallize(
            customer_id=customer_id, document_id=doc.id,
            items=items, content_chunks=content_chunks,
            crystal_type=doc_after.confirmed_type or doc_after.crystal_type,
        )
        await store.mark_document_crystallized(
            document_id=doc.id,
            crystals_written=result.crystals_written,
            items_extracted=result.items_extracted,
            crystallized_at=datetime.now(timezone.utc),
        )
    except Exception as e:  # noqa: BLE001
        await store.mark_document_error(doc.id, str(e))
        return FileOutcome(
            path=rel, status="failed",
            error=f"{type(e).__name__}: {_first_line(e)}",
        )

    return FileOutcome(
        path=rel, status="written",
        chunks=len(content_chunks), items=len(items),
        crystals=result.crystals_written,
    )


async def resync_written_files(
    project_dir: Path,
    abs_paths: list[str],
    *,
    store: Any,
    encoder: Any,
    vector_store: Any,
    fact_vector_store: Any,
    customer_id: str,
    client: Any = None,
) -> list[str]:
    """Bank freshness: re-sync agent-edited files at turn end.

    The staleness this closes (observed in the F9 bank demo): the agent
    edits a file, the bank keeps serving the pre-edit knowledge until
    the next manual /ingest. Now the CLI calls this after every turn
    with the guard's written-paths list.

    Policy (v1, each line deliberate):
      * TRACKED-ONLY — a file re-syncs only if a crystallized document
        with the same label already exists. An un-ingested project
        stays un-ingested, and a NEW agent-created file waits for an
        explicit /ingest — auto-ingesting it would spend extractor
        tokens the user never opted into. (Known edge: a moved file's
        old label lingers until the next /ingest replaces the view.)
      * HASH-SKIP — ingest_files' unchanged-skip still applies, so an
        edit that round-trips to identical content costs nothing.
      * REPL-ONLY — headless runs leave work on a review branch; the
        bank must not get ahead of what the user has merged. The
        background runner does not call this.

    Returns human-readable lines for the activity trace; empty list
    when there was nothing to do.
    """
    root = project_dir.resolve()
    rels: list[str] = []
    for p in dict.fromkeys(abs_paths):  # dedup, preserve order
        try:
            rel = Path(p).resolve().relative_to(root).as_posix()
        except (ValueError, OSError):
            continue  # outside the project — not bank material
        rels.append(rel)
    if not rels:
        return []

    prior = await _prior_doc_hashes(store, customer_id)
    tracked = [r for r in rels if r in prior]
    if not tracked:
        return []

    summary = await ingest_files(
        root, tracked,
        store=store, encoder=encoder, vector_store=vector_store,
        fact_vector_store=fact_vector_store, customer_id=customer_id,
        client=client,
    )
    lines: list[str] = []
    for o in summary.outcomes:
        if o.status == "written":
            lines.append(f"updated the knowledge bank: {o.path} ({o.crystals} crystals)")
        elif o.status == "failed":
            lines.append(f"bank sync failed for {o.path}: {o.error}")
        # 'unchanged' stays silent — nothing happened, say nothing.
    return lines


# ---------------------------------------------------------------------------
# The /ingest wizard (REPL) and --ingest (headless)
# ---------------------------------------------------------------------------

def _progress_printer(print_fn: Callable[[str], None]) -> Callable[[int, int, FileOutcome], None]:
    def show(i: int, total: int, o: FileOutcome) -> None:
        if o.status == "written":
            print_fn(
                f"  [{i}/{total}] {o.path} — {o.chunks} chunks"
                + (f" · {o.items} items" if o.items else "")
                + f" · {o.crystals} crystals"
            )
        elif o.status == "unchanged":
            print_fn(f"  [{i}/{total}] {o.path} — unchanged, skipped")
        else:
            print_fn(f"  [{i}/{total}] {o.path} — FAILED: {o.error}")
    return show


def _print_summary(summary: IngestSummary, print_fn: Callable[[str], None]) -> None:
    print_fn(
        f"\n  Done: {summary.written} files ingested, "
        f"{summary.crystals} crystals written, "
        f"{summary.unchanged} unchanged (skipped), {summary.failed} failed.\n"
    )
    for o in summary.outcomes:
        if o.status == "failed":
            print_fn(f"    failed: {o.path} — {o.error}")


def _cost_preview(scan: ScanResult, describe_code: bool) -> str:
    """The wizard's LLM-cost line.

    Prose docs always cost ~1 extraction call per ~3000 chars. With code
    descriptions on (CC_ENABLE_CODE_DESCRIPTIONS), each code file also costs
    ~1 description call; with it off, code ingests with no model calls.
    """
    extraction = scan.estimated_extraction_calls
    describe = len(scan.code_files) if describe_code else 0
    parts: list[str] = []
    if extraction:
        parts.append(f"~{extraction} extraction calls (prose docs)")
    if describe:
        parts.append(f"~{describe} description calls (one per code file)")
    head = f"  Cost: {' + '.join(parts)}" if parts else "  Cost: no LLM calls"
    tail = (
        " — code is indexed by generated descriptions, one model call per file."
        if describe_code
        else " — code ingests without the extraction model."
    )
    return head + tail + " Crystal writes are local."


async def run_ingest_wizard(
    default_root: Path,
    agent: Any,
    guard: Any,
    *,
    db_label: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
) -> None:
    """The interactive /ingest flow: folder → scan → excludes → docs?
    → target + cost preview → confirm → run with progress."""
    raw = input_fn(f"  Folder to ingest [{default_root}]: ").strip()
    root = Path(raw).resolve() if raw else default_root
    if not root.is_dir():
        print_fn(f"  That folder doesn't exist: {root}\n")
        return

    base_excludes = list(getattr(guard, "block_paths", None) or [])

    scan = scan_project(root, extra_excludes=base_excludes, include_docs=True)
    gitnote = ".gitignore honored" if scan.used_git else "not a git repo — default excludes applied"
    print_fn(
        f"  Found {len(scan.files)} ingestible files "
        f"({len(scan.code_files)} code, {len(scan.doc_files)} docs) — {gitnote}."
    )
    if scan.oversized:
        print_fn(f"  ({scan.oversized} files over 1MB skipped)")
    if base_excludes:
        print_fn(f"  (blocked paths excluded: {', '.join(base_excludes)})")

    extra = input_fn("  Extra excludes? (globs or folders, comma-separated, blank for none): ").strip()
    user_excludes = [p.strip() for p in extra.split(",") if p.strip()] if extra else []

    docs_in = input_fn("  Include prose docs (md/txt/rst)? [Y/n]: ").strip().lower()
    include_docs = docs_in not in ("n", "no")

    if user_excludes or not include_docs:
        scan = scan_project(
            root, extra_excludes=base_excludes + user_excludes,
            include_docs=include_docs,
        )
        print_fn(
            f"  Now {len(scan.files)} files "
            f"({len(scan.code_files)} code, {len(scan.doc_files)} docs)."
        )
    if not scan.files:
        print_fn("  Nothing to ingest with those filters.\n")
        return

    customer_id = agent.customer.id
    print_fn(f"  Target: customer {customer_id} in {db_label}")
    from crystal_cache.config import settings as _settings
    print_fn(_cost_preview(scan, _settings.enable_code_descriptions))

    if input_fn(f"  Ingest {len(scan.files)} files? [y/N]: ").strip().lower() not in ("y", "yes"):
        print_fn("  (ingest cancelled)\n")
        return

    state = agent.tool_state
    summary = await ingest_files(
        root, scan.files,
        store=state["store"], encoder=state["encoder"],
        vector_store=state["vector_store"],
        fact_vector_store=state.get("fact_vector_store"),
        customer_id=customer_id,
        client=getattr(agent, "llm", None),
        progress=_progress_printer(print_fn),
    )
    _print_summary(summary, print_fn)


async def run_ingest_headless(
    root: Path,
    *,
    excludes: list[str],
    include_docs: bool,
    db: Optional[str],
    customer_id: Optional[str],
    print_fn: Callable[[str], None] = print,
) -> int:
    """--ingest: scan and ingest without the REPL or the agent.

    Builds only what ingestion needs (store, encoder, vector stores)
    via the runtime helpers. Login precedence mirrors the REPL: explicit
    flags > saved /login > local default. No confirmation prompt — the
    flag IS the confirmation. Returns a process exit code.
    """
    from crystal_cache.encoding import build_text_encoder
    from crystal_cache.infrastructure import VectorStore
    from crystal_cache.infrastructure.fact_vector_store import FactVectorStore
    from crystal_cache.infrastructure.metadata_store import set_metadata_store

    from . import config_store
    from .runtime import (
        LOCAL_CUSTOMER_ID,
        _make_store,
        _resolve_db_url,
        _seed_legacy_crystal_types,
        build_llm_client,
    )

    if not (db or customer_id):
        saved_db, saved_customer = config_store.load_login()
        db = db or saved_db
        customer_id = customer_id or saved_customer
    customer_id = customer_id or LOCAL_CUSTOMER_ID

    # The extraction client, from the same credentials the REPL uses —
    # the pipeline's settings/env fallback can't see a dotenv-only key.
    # Provider-neutral: the pipeline and describer call .complete on it.
    creds = config_store.resolve_credentials()
    client = None
    if creds is not None:
        client = build_llm_client(creds)

    scan = scan_project(root, extra_excludes=excludes, include_docs=include_docs)
    gitnote = ".gitignore honored" if scan.used_git else "not a git repo — default excludes applied"
    print_fn(
        f"Ingesting {len(scan.files)} files from {root} "
        f"({len(scan.code_files)} code, {len(scan.doc_files)} docs) — {gitnote}."
    )
    if not scan.files:
        print_fn("Nothing to ingest.")
        return 0
    print_fn(f"Target: customer {customer_id} in {db or 'the local default store'}")

    print_fn("Setting up the database...")
    store = _make_store(_resolve_db_url(db))
    await store.init()
    set_metadata_store(store)
    await _seed_legacy_crystal_types(store)
    print_fn("Loading the language model (first run can take a minute)...")
    encoder = build_text_encoder()
    vector_store = VectorStore(store=store)
    fact_vector_store = FactVectorStore(store=store)

    summary = await ingest_files(
        root, scan.files,
        store=store, encoder=encoder, vector_store=vector_store,
        fact_vector_store=fact_vector_store, customer_id=customer_id,
        client=client,
        progress=_progress_printer(print_fn),
    )
    _print_summary(summary, print_fn)
    return 1 if (summary.failed and not summary.written) else 0
