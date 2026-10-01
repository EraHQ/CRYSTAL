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
    import structlog

    if structlog.is_configured():
        return
    from .hygiene import redact_event_dict

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            # Render tracebacks to plain text BEFORE redaction so the
            # exception message passes through redact() too, and no rich
            # renderer ever prints local variables.
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            redact_event_dict,
            structlog.dev.ConsoleRenderer(),
        ],
    )


_configure_logging()
