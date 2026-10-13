"""Crystal Cache — semantic preprocessing cache for enterprise LLM stacks.

This is the v2 rewrite. See PROJECT_LEDGER.md at the repo root for status,
decisions, and the porting plan.
"""

__version__ = "0.2.0"


def _configure_logging() -> None:
    """N7 (security sweep 2026-09-30): structlog's default dev processor
    chain plus one redaction processor, so no log line from any of the
    three services (api, worker, cognition) can carry a key, token, JWT
    or URL password. The chain mirrors structlog's defaults exactly
    (contextvars, level, stack info, exc_info, timestamp, console) with
    redact_event_dict inserted before rendering. Idempotent."""
    import os

    import structlog

    if structlog.is_configured():
        return
    from .hygiene import redact_event_dict

    # RC-08 (2026-10-05): in production (CC_LOG_FORMAT=json) every line
    # is one JSON object with `severity` and `message`, the two fields
    # Cloud Logging reads for log-based alerts. ConsoleRenderer lines
    # carried no severity, so an ERROR could not be alerted on. Local
    # runs keep the readable console.
    def _add_severity(_logger, _method, event_dict):
        event_dict["severity"] = {
            "debug": "DEBUG", "info": "INFO", "warning": "WARNING",
            "error": "ERROR", "critical": "CRITICAL",
        }.get(str(event_dict.get("level", "")).lower(), "DEFAULT")
        return event_dict

    json_logs = (os.environ.get("CC_LOG_FORMAT") or "").strip().lower() == "json"
    renderers = (
        [structlog.processors.EventRenamer("message"), structlog.processors.JSONRenderer()]
        if json_logs else [structlog.dev.ConsoleRenderer()]
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            _add_severity,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            # Render tracebacks to plain text BEFORE redaction so the
            # exception message passes through redact() too, and no rich
            # renderer ever prints local variables.
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(
                fmt="iso" if json_logs else "%Y-%m-%d %H:%M:%S", utc=json_logs,
            ),
            redact_event_dict,
            *renderers,
        ],
    )


def _configure_stdlib_logging() -> None:
    """B6-2 (Lockdown PR-7, 2026-10-13): the stdlib loggers (nine
    `logging.getLogger` modules in this package, httpx, the SDKs) and
    uvicorn's `uvicorn.error` tracebacks never passed through redact():
    an LLM exception that quoted a BYOK key reached Cloud Logging whole.

    Two parts, both idempotent:

    1. A LogRecord factory that redacts EVERY stdlib record at creation —
       the formatted message (args folded in) and the exception text
       (rendered once, cached on `exc_text`, which every Formatter uses
       instead of re-rendering). A factory covers every logger and every
       handler, including the ones uvicorn installs before this package
       is imported and the ones pytest attaches later, with no handler
       plumbing.
    2. A root handler when nothing configured one, so a stdlib WARNING or
       traceback is a structured line (`severity`, `message`) under
       CC_LOG_FORMAT=json like the structlog lines the alert reads,
       instead of lastResort's bare text. WARNING, not INFO: INFO would
       add every httpx request line to the hosted logs.
    """
    import logging
    import os
    from datetime import datetime, timezone

    from .hygiene import redact

    if not getattr(logging, "_crystal_redacting_factory", False):
        base_factory = logging.getLogRecordFactory()

        def _redacting_factory(*args, **kwargs):
            record = base_factory(*args, **kwargs)
            try:
                record.msg = redact(record.getMessage())
                record.args = ()
            except Exception:  # noqa: BLE001 — a bad format string is the caller's bug; keep the record
                record.msg = redact(str(record.msg))
                record.args = ()
            exc_info = record.exc_info
            if isinstance(exc_info, tuple) and exc_info[1] is not None:
                try:
                    record.exc_text = redact(logging.Formatter().formatException(exc_info))
                except Exception:  # noqa: BLE001
                    record.exc_text = "[traceback unavailable]"
            return record

        logging.setLogRecordFactory(_redacting_factory)
        logging._crystal_redacting_factory = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    if root.handlers:
        return
    json_logs = (os.environ.get("CC_LOG_FORMAT") or "").strip().lower() == "json"

    class _JsonFormatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            import json

            message = record.getMessage()
            if record.exc_info and not record.exc_text:
                record.exc_text = redact(self.formatException(record.exc_info))
            if record.exc_text:
                message = f"{message}\n{record.exc_text}"
            return json.dumps({
                "severity": record.levelname,
                "message": message,
                "logger": record.name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    handler = logging.StreamHandler()
    handler.setFormatter(
        _JsonFormatter() if json_logs
        else logging.Formatter("%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
    )
    root.addHandler(handler)
    root.setLevel(logging.WARNING)


_configure_logging()
_configure_stdlib_logging()
