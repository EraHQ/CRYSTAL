"""Secret hygiene and safe error egress (security sweep 2026-09-30).

Two incidents on one day drove this module:

1. A Stripe secret stored with a trailing newline made the SDK raise an
   error quoting the FULL live key, which the billing wrapper logged and
   returned in the HTTP response body (v104 fixed billing only).
2. The sweep found the identical mechanism on the LLM path: a customer's
   own BYOK key with a stray space or newline is echoed by httpx into
   responses, SSE frames, tool results and Cloud Logging (B1), and the
   platform keys and GitHub token had no strip at use (S1).

Rules this module enforces, for every secret and every error:

- `clean_secret` strips and REFUSES whitespace or control characters
  inside a secret, at write time and at config load.
- `redact` scrubs every known key shape and any Authorization-style
  header value from text destined for a log line.
- `safe_error` is the only way exception text should reach a client:
  the client gets a fixed message plus a reference id, and the redacted
  detail goes to the log keyed by that reference (Q6=C, generalized).
"""
from __future__ import annotations

import re
import uuid
from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)

SUPPORT_EMAIL = "hello@erahq.ai"

# Key shapes we hold or handle on behalf of customers, each with its
# replacement template. Header-shaped patterns run first so a
# `Bearer <key>` is scrubbed whole; URL credentials keep the scheme and
# user so a log line stays diagnosable.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"Bearer\s+\S+"), "[redacted]"),
    (re.compile(r"eyJ[A-Za-z0-9_\-]{8,}(?:\.[A-Za-z0-9_\-]+){0,2}"), "[redacted]"),  # JWTs
    (re.compile(r"(\w+://[^:/\s@]+:)[^@\s]+@"), r"\1[redacted]@"),     # URL passwords
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]+"), "[redacted]"),                # Anthropic
    (re.compile(r"sk-proj-[A-Za-z0-9_\-]+"), "[redacted]"),               # OpenAI project
    (re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), "[redacted]"),                # OpenAI legacy
    (re.compile(r"(?:sk|rk|pk)_(?:live|test)_[A-Za-z0-9]+"), "[redacted]"),  # Stripe
    (re.compile(r"whsec_[A-Za-z0-9]+"), "[redacted]"),                    # Stripe webhook
    (re.compile(r"gsk_[A-Za-z0-9]+"), "[redacted]"),                      # Groq
    (re.compile(r"tvly-[A-Za-z0-9_\-]+"), "[redacted]"),                  # Tavily
    (re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]+"), "[redacted]"),  # GitHub
    (re.compile(r"github_pat_[A-Za-z0-9_]+"), "[redacted]"),
    (re.compile(r"ya29\.[A-Za-z0-9_\-\.]+"), "[redacted]"),               # Google OAuth
    (re.compile(r"AIza[0-9A-Za-z_\-]{35}"), "[redacted]"),                # Google API key
    (re.compile(r"cc_sk_[A-Za-z0-9]+"), "[redacted]"),                    # Crystal Key A
    (re.compile(r"(?i)(authorization|x-api-key|api[-_]?key)(['\"]?\s*[:=]\s*['\"]?)\S+"), r"\1\2[redacted]"),
)


def redact(text: Any) -> str:
    """Scrub every known secret shape from `text` for logging."""
    out = str(text)
    for pat, template in _SECRET_PATTERNS:
        out = pat.sub(template, out)
    return out


def redact_event_dict(logger, method_name, event_dict):  # noqa: ANN001
    """structlog processor (N7, 2026-09-30): every string value in every
    log line passes through redact(), one level into lists and dicts, so
    a key that reaches a logger call by any path never reaches Cloud
    Logging. Wired in crystal_cache/__init__.py."""
    for key, value in list(event_dict.items()):
        if isinstance(value, str):
            event_dict[key] = redact(value)
        elif isinstance(value, dict):
            event_dict[key] = {
                k: (redact(v) if isinstance(v, str) else v) for k, v in value.items()
            }
        elif isinstance(value, (list, tuple)):
            event_dict[key] = [redact(v) if isinstance(v, str) else v for v in value]
    return event_dict


class SecretFormatError(ValueError):
    """A secret contains whitespace or control characters after stripping."""


def clean_secret(value: Optional[str], *, field: str = "secret") -> str:
    """Strip a secret and refuse anything that would break or leak.

    A stray newline from `echo` (S1) or a pasted trailing space (B1) is
    what turns a secret into an httpx InvalidHeader error that quotes the
    whole value. Empty input stays empty (the "not configured" state).
    """
    if value is None:
        return ""
    cleaned = str(value).strip()
    if any(ord(ch) < 32 or ord(ch) == 127 or ch.isspace() for ch in cleaned):
        raise SecretFormatError(
            f"{field} contains whitespace or control characters"
        )
    return cleaned


def new_ref() -> str:
    return uuid.uuid4().hex[:10]


def safe_error(
    event: str,
    exc: BaseException,
    *,
    user_message: str = "Something went wrong on our side.",
    **context: Any,
) -> tuple[str, str]:
    """Log a redacted exception under a reference id; return the pair
    (ref, message) the client may see. The message never contains
    exception text. `context` is logged as-is, so pass ids, not payloads.
    """
    ref = new_ref()
    logger.error(
        event,
        ref=ref,
        error_type=type(exc).__name__,
        error=redact(str(exc))[:500],
        **context,
    )
    return ref, (
        f"{user_message} Please try again, or contact support at "
        f"{SUPPORT_EMAIL} and mention reference {ref}."
    )


def client_message(ref: str, user_message: str = "Something went wrong on our side.") -> str:
    return (
        f"{user_message} Please try again, or contact support at "
        f"{SUPPORT_EMAIL} and mention reference {ref}."
    )
