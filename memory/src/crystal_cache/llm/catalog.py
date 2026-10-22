"""Token Factory's model catalog, read from the platform's own key.

Hackathon item 1 (2026-10-22, Q95 B): a tenant may pick ANY text model
Nebius Token Factory serves, not only the three Nemotron ids in config.
Token Factory is one OpenAI-compatible endpoint for its whole catalog
and `GET /v1/models?verbose=true` describes every model the same way
(context length, pricing, per-request limits, modality), so "any model"
is a catalog read plus the lane that already exists:

  * `fetch_catalog()` reads the listing once per process and keeps it
    for `settings.nebius_catalog_ttl_seconds`; after that a stale copy is
    served while a daemon thread refreshes it, so no model call ever
    waits on the listing. The first read of a process is synchronous
    (one GET, ten seconds at most); the api warms it at startup.
  * `catalog_ids()` is what llm/lanes.family_of consults: a model is in
    the Token Factory family when the catalog lists it (or its id starts
    with nvidia/, the rule before the catalog existed).
  * `price_rows()` turns the listing's pricing into the ledger's
    micro-USD-per-million rows so the capacity walls meter any pick at
    its real price (an unknown id ledgers 0 and never trips a wall).
    The listing's unit is not documented, so the rows are CALIBRATED
    against one anchor whose price Anthony verified by hand on
    2026-10-19 (Lightning, $0.06 per million input tokens): the scale
    that lands the anchor within 2% is the one applied to every row;
    no anchor, no rows (the hard-coded estimates in cost/pricing.py
    stay in force) and one log line says so.
  * `probe_model()` asks a model the two questions the listing cannot
    answer (the docs say to "test several models"): does it honour
    `response_format: json_schema`, and does it make a tool call when
    told to. Two tiny calls, reasoning off. The flags are stored with
    the tenant's pick; the seam reads them (a model without json_schema
    gets json_object plus the schema in the prompt; an answers pick
    without tools is refused at the door, code model_no_tools).

Tenants never see dollars: the console gets a cost BAND per model
(light / standard / heavy) derived here from the input price.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

# structlog, not stdlib: the lockdown's stdlib root handler passes
# WARNING and up only, so a stdlib INFO line (catalog.loaded) never
# reached Cloud Logging on the first deploy (2026-10-22).
logger = structlog.get_logger(__name__)

# The anchor: id and verified micro-USD per million input tokens.
_ANCHOR_ID = "nvidia/nemotron-3_5-lightning"
_ANCHOR_INPUT_MICRO = 60_000  # $0.06 / Mtok (Series K1, 2026-10-19)
# Candidate meanings of the listing's price strings, as multipliers to
# micro-USD per million tokens: USD per million, USD per thousand, USD per token.
_UNIT_SCALES: tuple[float, ...] = (1e6, 1e9, 1e12)
_ANCHOR_TOLERANCE = 0.02

_CATALOG_TIMEOUT_S = 10.0
_PROBE_MAX_TOKENS = 60


@dataclass(frozen=True)
class CatalogModel:
    id: str
    name: str
    context_length: int
    modality: str                      # "text->text", "image+text->text", ...
    input_micro_per_mtok: Optional[int]   # None when the unit could not be calibrated
    output_micro_per_mtok: Optional[int]
    cached_input_micro_per_mtok: Optional[int] = None  # the listing's input_cache_read, when priced
    requests_per_minute: Optional[int] = None
    tokens_per_minute: Optional[int] = None

    @property
    def is_text(self) -> bool:
        return self.modality.replace(" ", "").lower().startswith("text->text")

    @property
    def is_fast(self) -> bool:
        return self.id.lower().endswith("-fast")

    @property
    def band(self) -> str:
        """light / standard / heavy, by input price; 'standard' when unknown."""
        p = self.input_micro_per_mtok
        if p is None:
            return "standard"
        if p < 200_000:
            return "light"
        if p < 1_000_000:
            return "standard"
        return "heavy"


@dataclass
class _Cache:
    models: tuple[CatalogModel, ...] = ()
    fetched_at: float = 0.0
    attempted: bool = False
    refreshing: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


_cache = _Cache()


def _as_int(v: Any) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _as_float(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f >= 0 else None


def _calibrate(raw: list[dict[str, Any]]) -> Optional[float]:
    """The scale under which the anchor's input price lands on its verified
    value, or None when the anchor is missing or no scale fits."""
    for m in raw:
        if str(m.get("id", "")).lower() != _ANCHOR_ID:
            continue
        p = _as_float((m.get("pricing") or {}).get("prompt"))
        if not p:
            return None
        for scale in _UNIT_SCALES:
            if abs(p * scale - _ANCHOR_INPUT_MICRO) <= _ANCHOR_INPUT_MICRO * _ANCHOR_TOLERANCE:
                return scale
        return None
    return None


def parse_catalog(payload: dict[str, Any]) -> tuple[CatalogModel, ...]:
    """The verbose listing -> CatalogModel rows. Pure; tested on a recorded
    payload. Prices are None for every row when calibration fails."""
    raw = [m for m in (payload.get("data") or []) if isinstance(m, dict) and m.get("id")]
    scale = _calibrate(raw)
    if scale is None and raw:
        logger.warning("catalog.price_unit_unknown", anchor=_ANCHOR_ID, rows=len(raw))
    out: list[CatalogModel] = []
    for m in raw:
        pricing = m.get("pricing") or {}
        limits = m.get("per_request_limits") or {}
        arch = m.get("architecture") or {}
        pin = _as_float(pricing.get("prompt")) if scale else None
        pout = _as_float(pricing.get("completion")) if scale else None
        pcache = _as_float(pricing.get("input_cache_read")) if scale else None
        out.append(CatalogModel(
            id=str(m["id"]),
            name=str(m.get("name") or m["id"]),
            context_length=_as_int(m.get("context_length")) or 0,
            modality=str(arch.get("modality") or "text->text"),
            input_micro_per_mtok=int(round(pin * scale)) if (scale and pin is not None) else None,
            output_micro_per_mtok=int(round(pout * scale)) if (scale and pout is not None) else None,
            cached_input_micro_per_mtok=int(round(pcache * scale)) if (scale and pcache is not None) else None,
            requests_per_minute=_as_int(limits.get("requests_per_minute")),
            tokens_per_minute=_as_int(limits.get("tokens_per_minute")),
        ))
    return tuple(out)


def _fetch_payload() -> dict[str, Any]:
    import httpx

    from ..config import settings

    base = (settings.nebius_base_url or "").rstrip("/")
    key = (settings.nebius_api_key or "").strip()
    resp = httpx.get(
        f"{base}/models",
        params={"verbose": "true"},
        headers={"Authorization": f"Bearer {key}"},
        timeout=_CATALOG_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()


def verify_key(api_key: str) -> int:
    """Hackathon item 2: one GET /v1/models with a tenant's own Token
    Factory key, before it is stored. Returns the number of models the
    key can see; raises (httpx error) when the key is refused."""
    import httpx

    from ..config import settings

    base = (settings.nebius_base_url or "").rstrip("/")
    resp = httpx.get(
        f"{base}/models",
        headers={"Authorization": f"Bearer {api_key.strip()}"},
        timeout=_CATALOG_TIMEOUT_S,
    )
    resp.raise_for_status()
    return len((resp.json() or {}).get("data") or [])


def _load(reason: str) -> None:
    try:
        models = parse_catalog(_fetch_payload())
    except Exception as e:  # noqa: BLE001 — the catalog is best-effort; stale or empty is never fatal
        logger.warning("catalog.fetch_failed", reason=reason, error=str(e)[:200])
        with _cache.lock:
            _cache.attempted = True
            _cache.refreshing = False
        return
    with _cache.lock:
        _cache.models = models
        _cache.fetched_at = time.monotonic()
        _cache.attempted = True
        _cache.refreshing = False
    try:
        from ..cost.pricing import register_runtime_prices

        register_runtime_prices(price_rows(models))
    except Exception as e:  # noqa: BLE001
        logger.warning("catalog.price_register_failed", error=str(e)[:200])
    logger.info("catalog.loaded", reason=reason, models=len(models),
                priced=sum(1 for m in models if m.input_micro_per_mtok is not None))


def _ttl() -> float:
    from ..config import settings

    return float(getattr(settings, "nebius_catalog_ttl_seconds", 900))


def fetch_catalog(*, force: bool = False) -> tuple[CatalogModel, ...]:
    """The catalog (empty when the lane is off). Fresh within the TTL;
    stale copies are served while a daemon thread refreshes; the first
    read of a process (or `force`) fetches synchronously."""
    from .lanes import nemotron_enabled

    if not nemotron_enabled():
        return ()
    with _cache.lock:
        have = _cache.attempted
        stale = (time.monotonic() - _cache.fetched_at) > _ttl()
        refreshing = _cache.refreshing
    if force or not have:
        _load("force" if force else "first")
    elif stale and not refreshing:
        with _cache.lock:
            _cache.refreshing = True
        threading.Thread(target=_load, args=("refresh",), daemon=True).start()
    with _cache.lock:
        return _cache.models


def catalog_ids() -> frozenset[str]:
    """Lower-cased ids of every listed model (empty when the lane is off)."""
    return frozenset(m.id.lower() for m in fetch_catalog())


def text_models() -> tuple[CatalogModel, ...]:
    return tuple(m for m in fetch_catalog() if m.is_text)


def find_model(model_id: str) -> Optional[CatalogModel]:
    want = (model_id or "").lower()
    for m in fetch_catalog():
        if m.id.lower() == want:
            return m
    return None


def price_rows(models: Optional[tuple[CatalogModel, ...]] = None) -> dict[str, Any]:
    """{model id: ModelPrice} for every calibrated row."""
    from ..cost.pricing import ModelPrice

    rows: dict[str, Any] = {}
    for m in (models if models is not None else fetch_catalog()):
        if m.input_micro_per_mtok is None or m.output_micro_per_mtok is None:
            continue
        rows[m.id] = ModelPrice(
            m.input_micro_per_mtok, m.output_micro_per_mtok,
            cache_read_micro_per_mtok=m.cached_input_micro_per_mtok or 0,
        )
    return rows


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelCaps:
    json_schema: bool
    tools: bool
    error: Optional[str] = None

    def as_dict(self) -> dict[str, Any]:
        from datetime import datetime, timezone

        return {
            "json_schema": self.json_schema,
            "tools": self.tools,
            "error": self.error,
            "probed_at": datetime.now(timezone.utc).isoformat(),
        }


_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_PROBE_TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "Echo one word back.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
            "additionalProperties": False,
        },
    },
}


def probe_model(client, model_id: str) -> ModelCaps:
    """Two small calls through `client.raw_chat` (the Token Factory
    client). Each failure, transport or shape, marks that flag False and
    the first error text is kept for the console."""
    errors: list[str] = []
    json_ok = False
    tools_ok = False
    try:
        data = client.raw_chat({
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with JSON: {\"ok\": true}"}],
            "max_tokens": _PROBE_MAX_TOKENS,
            "temperature": 0,
            "reasoning_effort": "none",
            "response_format": {"type": "json_schema", "json_schema": {"name": "probe", "schema": _PROBE_SCHEMA}},
        }, max_tokens=_PROBE_MAX_TOKENS)
        content = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        parsed = json.loads(content.strip())
        json_ok = isinstance(parsed, dict) and parsed.get("ok") is True
    except Exception as e:  # noqa: BLE001
        errors.append(f"json_schema: {str(e)[:120]}")
    try:
        data = client.raw_chat({
            "model": model_id,
            "messages": [{"role": "user", "content": "Call the ping tool with the word hello."}],
            "max_tokens": _PROBE_MAX_TOKENS,
            "temperature": 0,
            "reasoning_effort": "none",
            "tools": [_PROBE_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "ping"}},
        }, max_tokens=_PROBE_MAX_TOKENS)
        calls = ((data.get("choices") or [{}])[0].get("message") or {}).get("tool_calls") or []
        tools_ok = bool(calls) and (calls[0].get("function") or {}).get("name") == "ping"
    except Exception as e:  # noqa: BLE001
        errors.append(f"tools: {str(e)[:120]}")
    return ModelCaps(json_schema=json_ok, tools=tools_ok, error="; ".join(errors) or None)


def reset_for_tests() -> None:
    with _cache.lock:
        _cache.models = ()
        _cache.fetched_at = 0.0
        _cache.attempted = False
        _cache.refreshing = False
    from ..cost.pricing import RUNTIME_PRICE_TABLE

    RUNTIME_PRICE_TABLE.clear()
