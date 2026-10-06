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


_configure_logging()
