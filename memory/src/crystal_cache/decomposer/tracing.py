"""Decomposer trace logging - seed corpus for the eventual distilled classifier.

Every successful LLM-decomposer call is a labeled data point:
  (query_text, structured_payload)

Log those consistently and after a few weeks you have thousands of
examples, all labeled by a teacher model that's doing a reasonable job.
That's the training set for a much smaller student classifier.

v0.3 SCOPE - JSONL FILE
-----------------------
We write one JSON object per line to a file configured by
settings.decomposer_trace_path. JSONL because:
  - Zero infrastructure (no DB table to migrate, no service to run).
  - Trivially replayable into any training pipeline (pandas, HF datasets,
    scikit-learn - they all read JSONL).
  - Append-only, so concurrent writers can both append without locking
    if the OS supports O_APPEND (which Linux does; Windows is tricky).

If decomposer_trace_path is None, logging is a no-op.

v0.4+ PATH
----------
Once traffic warrants, move to a proper table in metadata store with
tenant_id indexed. Enables per-tenant training data partitioning and
classification-as-a-service per customer. The JSONL file format here
is a superset of what a future table would hold, so migration is a
one-time replay.

PRIVACY NOTE
------------
The trace contains raw user queries. If deployed for customers whose
queries are sensitive (health, legal, financial), either:
  - Disable tracing (set decomposer_trace_path=None)
  - Scrub/hash queries before logging
  - Keep trace files in a customer-isolated location with access
    controls matching the rest of their data.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import structlog

from crystal_cache.decomposer.base import (
    DecompositionResult,
    Decomposer,
)


logger = structlog.get_logger(__name__)


@dataclass
class TraceRecord:
    """One row of training data from a decomposer call.

    Required fields (always populated):
        tenant_id, query_text, payload, model_name, confidence,
        latency_ms, timestamp.

    Distillation fields (added 2026-04-24, optional/None when unknown):
        sub_queries: When the decomposition was compound (Shape A multi-
            query), the list of sub-payloads. None for non-compound.
            Critical for distillation — a student model needs to learn
            both single-query and compound-query decomposition shapes.
        raw_output: Raw text output from the LLM before JSON parsing.
            Useful when distillation training surfaces schema-compliance
            failures and we need to know what the teacher actually said.
            None for stub/programmatic decomposers.
        context: The conversation context dict passed to the decomposer.
            Carries prior-turn information needed to disambiguate
            references like "what about last Tuesday?" Optional and
            potentially large; truncated by the writer if so.
        routing_outcome: Filled in by the router AFTER retrieval has
            run. Captures whether this decomposition produced a useful
            retrieval. The single highest-leverage field for
            distillation — it lets a student learn from outcomes, not
            just imitate the teacher's outputs. See RoutingOutcome
            dataclass below for the exact shape.

    Why all the new fields are optional: TracingDecomposer (the
    decomposer-only path) doesn't have access to routing outcome.
    Setting these to None there is correct. Router-level tracing fills
    them in when it has the data.
    """

    tenant_id: str
    query_text: str
    payload: dict[str, Any]
    model_name: Optional[str]
    confidence: Optional[float]
    latency_ms: float
    timestamp: str  # ISO 8601 UTC

    # Distillation fields, all optional.
    sub_queries: Optional[list[dict[str, Any]]] = None
    raw_output: Optional[str] = None
    context: Optional[dict[str, Any]] = None
    routing_outcome: Optional["RoutingOutcome"] = None

    def to_json_line(self) -> str:
        """Serialize as a single JSON line (no trailing newline).

        Optional fields are emitted as null when unset rather than
        omitted from the JSON. Stable schema makes downstream training
        scripts simpler — they can assume every line has every key.
        """
        return json.dumps(
            {
                "tenant_id": self.tenant_id,
                "query_text": self.query_text,
                "payload": self.payload,
                "model_name": self.model_name,
                "confidence": self.confidence,
                "latency_ms": self.latency_ms,
                "timestamp": self.timestamp,
                "sub_queries": self.sub_queries,
                "raw_output": self.raw_output,
                "context": self.context,
                "routing_outcome": (
                    self.routing_outcome.to_dict()
                    if self.routing_outcome is not None
                    else None
                ),
            },
            ensure_ascii=False,
        )


@dataclass
class RoutingOutcome:
    """Captured after retrieval runs. Tells the distillation pipeline
    whether the decomposition actually produced a useful match.

    A decomposition can be 'correct-looking' to a teacher LLM (good
    JSON, sensible payload) but still wrong in the sense that nothing
    matched. That's the highest-signal training example for a student:
    'this is what NOT to produce.' Without routing_outcome we can't
    distinguish those from genuinely useful decompositions.

    Fields:
        match_type: 'high', 'medium', 'low', or 'none' — the highest
            match level achieved by retrieval after this decomposition.
            'none' is the strongest negative signal.
        top_match_score: The actual cosine score of the best match.
            Continuous-valued companion to match_type for fine-grained
            analysis.
        matched_crystal_id: The id of the top-matched crystal, if any.
            For Shape A compound queries, only the FIRST sub-query's
            top match is captured here; per-sub-query outcomes belong
            in a future schema extension if/when we need them.
        injection_method: How (if at all) injection happened: 'text',
            'fact', 'none'. 'none' means we matched but didn't inject
            (low confidence, gated by skill rules, etc.).
    """

    match_type: str  # 'high' | 'medium' | 'low' | 'none'
    top_match_score: Optional[float] = None
    matched_crystal_id: Optional[str] = None
    injection_method: Optional[str] = None  # 'text' | 'fact' | 'none'

    def to_dict(self) -> dict[str, Any]:
        return {
            "match_type": self.match_type,
            "top_match_score": self.top_match_score,
            "matched_crystal_id": self.matched_crystal_id,
            "injection_method": self.injection_method,
        }


class JsonlTraceWriter:
    """Append-only JSONL writer. One line per decomposer call.

    Two modes:

    **Static path** (legacy mode):
        writer = JsonlTraceWriter("traces/decomposer.jsonl")
        writer.write(record)

    All records land in one file. Fine for development, breaks at
    scale because there's no per-tenant isolation and the file grows
    unboundedly.

    **Path template** (production mode, added 2026-04-24):
        writer = JsonlTraceWriter("traces/{tenant_id}/{date}.jsonl")
        writer.write(record)

    The path is rendered per-record with `{tenant_id}` substituted from
    `record.tenant_id` and `{date}` substituted from `record.timestamp`
    (the date portion in YYYY-MM-DD). This gives natural partitioning
    by tenant and day, which means:
      - Per-tenant retention is `rm -rf traces/<tenant_id>/`
      - Date-based rotation is `rm traces/<tenant_id>/2026-01-*.jsonl`
      - Customer-policy isolation is filesystem-level

    The placeholders are recognized by literal substring match. If the
    path contains neither, static mode is used. If it contains either,
    the writer creates parent directories as needed per-record.

    File is opened per-write (no lingering handles). Slow for high-
    throughput but correct across concurrent processes and robust to
    crashes. If throughput becomes an issue we buffer and flush
    periodically.
    """

    # Recognized placeholders. Add more here if we need finer-grained
    # partitioning (e.g. {hour} for very high-volume tenants).
    _PLACEHOLDERS = ("{tenant_id}", "{date}")

    def __init__(self, path: str | Path) -> None:
        # Keep the raw template string — we render it per-write when
        # placeholders are present. Don't normalize to Path here
        # because Path('foo/{tenant_id}/bar') is a valid Path on
        # Linux but the {tenant_id} segment confuses Path.parent on
        # static-mode resolution. Cleaner to keep the string and
        # convert to Path only after rendering.
        self._raw_path = str(path)
        self._is_template = any(
            placeholder in self._raw_path
            for placeholder in self._PLACEHOLDERS
        )

        # In static mode, ensure parent dir exists once at construction.
        # In template mode, we ensure parent per-write because the dir
        # depends on the record.
        if not self._is_template:
            Path(self._raw_path).parent.mkdir(parents=True, exist_ok=True)

    @property
    def path(self) -> Path:
        """Backward-compat accessor. Returns the raw template path.

        For template-mode writers, this is the unrendered string —
        callers can use it for logging but should NOT use it for
        filesystem operations. Use `path_for(record)` instead.
        """
        return Path(self._raw_path)

    def path_for(self, record: "TraceRecord") -> Path:
        """Render the path for a specific record.

        In static mode, returns the same path regardless of record.
        In template mode, substitutes `{tenant_id}` and `{date}` from
        the record's fields. Sanitizes tenant_id to prevent path
        traversal: forward slashes, backslashes, and parent-directory
        references are replaced with underscores.
        """
        if not self._is_template:
            return Path(self._raw_path)

        rendered = self._raw_path
        if "{tenant_id}" in rendered:
            rendered = rendered.replace(
                "{tenant_id}", _sanitize_path_segment(record.tenant_id)
            )
        if "{date}" in rendered:
            # Extract date portion of ISO 8601 timestamp. The first 10
            # chars are always YYYY-MM-DD if the timestamp is valid.
            # If timestamp is malformed, fall back to 'unknown_date'
            # rather than crashing.
            date_part = (record.timestamp or "")[:10]
            if not _is_iso_date(date_part):
                date_part = "unknown_date"
            rendered = rendered.replace("{date}", date_part)
        return Path(rendered)

    def write(self, record: "TraceRecord") -> None:
        line = record.to_json_line()
        target = self.path_for(record)
        try:
            if self._is_template:
                # Per-write parent-dir creation only in template mode.
                # In static mode we did this once at construction.
                target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
        except OSError as e:
            # Don't fail the request over a trace write failure.
            logger.warning(
                "decomposer.trace_write_failed",
                path=str(target),
                error=str(e),
            )


def _sanitize_path_segment(value: str) -> str:
    """Strip characters that could create path-traversal attacks.

    Tenant IDs come from the customers table and are nominally safe
    (UUID-shaped). But we treat them as untrusted at this boundary
    — a misconfigured ingest path that lets a customer pick their
    own id shouldn't enable filesystem escape.

    Replaces any char that's not alphanumeric, dash, underscore, or
    dot with underscore. Empty strings become 'unknown_tenant'.
    """
    if not value:
        return "unknown_tenant"
    safe_chars = []
    for ch in value:
        if ch.isalnum() or ch in ("-", "_", "."):
            safe_chars.append(ch)
        else:
            safe_chars.append("_")
    sanitized = "".join(safe_chars)
    # Reject leading dots (hidden files / parent-dir refs)
    sanitized = sanitized.lstrip(".") or "unknown_tenant"
    return sanitized


def _is_iso_date(value: str) -> bool:
    """True iff value looks like YYYY-MM-DD. Cheap regex-free check."""
    if len(value) != 10:
        return False
    if value[4] != "-" or value[7] != "-":
        return False
    return value[:4].isdigit() and value[5:7].isdigit() and value[8:10].isdigit()


class TracingDecomposer:
    """Wraps any Decomposer, logging successful calls as training data.

    Pass-through on failure - if the inner decomposer raises, we don't
    log (no payload to log) and re-raise unchanged.

    Usage:
        inner = HostedLLMDecomposer()
        writer = JsonlTraceWriter("traces/decomposer.jsonl")
        traced = TracingDecomposer(inner, writer, tenant_id_fn=...)

    tenant_id_fn: the tenant_id isn't part of the Decomposer.decompose()
    signature (it's a router-level concept), so we require callers to
    either supply a fixed tenant_id at construction time or a callable
    that extracts it from the context dict. The router already has
    tenant_id in hand, so the simplest pattern is to construct a
    per-request TracingDecomposer. Or use the tenant_id_fn to pull it
    from context["tenant_id"].
    """

    def __init__(
        self,
        inner: Decomposer,
        writer: JsonlTraceWriter,
        *,
        tenant_id: Optional[str] = None,
        tenant_id_fn: Optional[Any] = None,
    ) -> None:
        if tenant_id is None and tenant_id_fn is None:
            raise ValueError(
                "TracingDecomposer requires either tenant_id or tenant_id_fn"
            )
        self._inner = inner
        self._writer = writer
        self._tenant_id = tenant_id
        self._tenant_id_fn = tenant_id_fn

    async def decompose(
        self,
        text: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> DecompositionResult:
        start = _now_monotonic_ms()
        result = await self._inner.decompose(text, context=context)
        latency = _now_monotonic_ms() - start

        tenant_id = self._resolve_tenant_id(context)
        record = TraceRecord(
            tenant_id=tenant_id,
            query_text=text,
            payload=result.payload,
            model_name=result.model_name,
            confidence=result.confidence,
            latency_ms=latency,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
        self._writer.write(record)
        return result

    def _resolve_tenant_id(self, context: Optional[dict[str, Any]]) -> str:
        if self._tenant_id is not None:
            return self._tenant_id
        assert self._tenant_id_fn is not None
        if context is None:
            return "unknown"
        try:
            return str(self._tenant_id_fn(context))
        except Exception:
            return "unknown"


def _now_monotonic_ms() -> float:
    """Monotonic wall-ish time in milliseconds, for latency measurement."""
    import time
    return time.monotonic() * 1000.0


def build_trace_writer_from_settings() -> Optional[JsonlTraceWriter]:
    """Construct a JsonlTraceWriter from settings, or None if disabled."""
    from crystal_cache.config import settings
    if not settings.decomposer_trace_path:
        return None
    return JsonlTraceWriter(settings.decomposer_trace_path)
