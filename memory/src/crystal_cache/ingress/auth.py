"""Bearer-token authentication dependencies for FastAPI.

Reads the Authorization header, extracts the api_key, and resolves it to
the caller. Two callers exist:

  - require_customer  -> a Customer (the TEAM). "Key A" in the design: the
    key the team uses to authenticate TO Crystal Cache. Not to be confused
    with "Key B" (api_key_ref in model_routing_config), the team's key to
    their upstream LLM provider.
  - require_operator  -> an Operator (an authenticated human under a team,
    Foundation F1). Per-operator scoped key; a suspended operator is
    rejected at this boundary without its row being deleted.

Both hash the presented key and look it up by that hash — no plaintext key
is ever stored (see infrastructure/credentials.py). Raise 401/403 otherwise.
"""
from __future__ import annotations

import hmac
import os
import re
from typing import Annotated, Optional

from fastapi import Depends, HTTPException, Request, status

from ..config import get_settings
from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..models import Customer, Operator


def _extract_bearer_token(request: Request) -> str:
    """Return the Bearer token from the Authorization header, or raise 401.

    Shared by require_customer and require_operator so the header parsing
    (and its 401 messages) stay identical across both.
    """
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header must be 'Bearer <api_key>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = parts[1].strip()
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Empty bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token


async def require_customer(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> Customer:
    """FastAPI dependency: validate the Bearer team key, return the Customer
    (team) or 401. The store hashes the presented key and matches the stored
    hash."""
    api_key = _extract_bearer_token(request)
    customer = await store.get_customer_by_api_key(api_key)
    if customer is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return customer


async def require_operator(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> Operator:
    """FastAPI dependency: validate the Bearer operator key, return the
    Operator or raise.

    401 when the key matches no operator; 403 when the operator is suspended.
    The store returns suspended operators (the row survives a suspension), so
    this boundary is what denies them — keeping 'suspended' distinguishable
    from 'unknown key'.
    """
    api_key = _extract_bearer_token(request)
    operator = await store.get_operator_by_api_key(api_key)
    if operator is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid operator api_key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if operator.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Operator is suspended",
        )
    return operator


async def resolve_principal(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> tuple[Customer, Optional[Operator]]:
    """Resolve the Bearer token to a (Customer team, Operator).

    The token may be EITHER an operator key (→ that operator + its team as
    the Customer) or a team customer key (→ that Customer + its DEFAULT
    ADMIN operator). P1 identity chain (ratified 2026-07-02): the team key
    ACTS AS the default admin, so every request has an operator and every
    write can stamp an owner — there is no operator-less path anymore. The
    default admin is created at customer creation and lazily self-healed
    here for pre-P1 customers (ensure_default_admin is an idempotent
    get-or-create). Operator keys are tried first so an operator never
    falls through to the team path.

    The return type keeps Optional[Operator] for signature compatibility
    with existing consumers, but the operator is now ALWAYS present.

    A suspended operator is rejected (403); an operator whose team row is
    missing is an integrity error (401). Used by the operator-aware SDK
    endpoints; team-only endpoints keep require_customer.
    """
    token = _extract_bearer_token(request)

    operator = await store.get_operator_by_api_key(token)
    if operator is not None:
        if operator.status != "active":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Operator is suspended",
            )
        team = await store.get_customer_by_id(operator.team_id)
        if team is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Operator's team not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return team, operator

    customer = await store.get_customer_by_api_key(token)
    if customer is not None:
        return customer, await store.ensure_default_admin(customer.id)

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid api_key",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Role-gated authorization (Foundation F1)
# ---------------------------------------------------------------------------

# Operator roles form a total order: a higher rank subsumes every lower
# one (admin can do whatever operator can, etc.). The team (customer) key
# is the team ROOT credential and outranks every named role, so it is
# always admitted regardless of the gate.
_ROLE_RANK: dict[str, int] = {"viewer": 0, "operator": 1, "admin": 2}


def require_role(min_role: str):
    """Dependency factory: admit the request iff the principal meets min_role.

    Returns a FastAPI dependency that resolves the bearer to a principal
    (via resolve_principal) and enforces a minimum operator role:

      - A TEAM key (operator is None) is the team root credential and is
        ALWAYS admitted -- root outranks every named role. This keeps the
        bootstrap path open: a brand-new team holds only its team key and
        must be able to provision its first operator.
      - An OPERATOR key is admitted iff its role rank >= min_role's rank,
        else 403.

    The resolved (Customer, Optional[Operator]) principal is returned so the
    endpoint can both team-scope its work and, when present, attribute the
    acting operator. min_role must be one of viewer/operator/admin.
    """
    if min_role not in _ROLE_RANK:
        raise ValueError(f"unknown role: {min_role!r}")
    min_rank = _ROLE_RANK[min_role]

    async def _require_role(
        request: Request,
        store: Annotated[MetadataStore, Depends(get_metadata_store)],
    ) -> tuple[Customer, Optional[Operator]]:
        customer, operator = await resolve_principal(request, store)
        if operator is None:
            return customer, operator  # team root -- always admitted
        if _ROLE_RANK.get(operator.role, -1) < min_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"operator role {operator.role!r} is below the required "
                    f"role {min_role!r}"
                ),
            )
        return customer, operator

    return _require_role


# ---------------------------------------------------------------------------
# Principal projections (Foundation F2 -- proxy operator-scoping)
# ---------------------------------------------------------------------------
# The chat proxy needs BOTH the team Customer and the optional Operator from
# one bearer. These two deps each read the SAME resolve_principal result
# (FastAPI caches a dependency's result within a request), so the bearer is
# resolved once. Splitting it this way lets the proxy wrapper keep a plain
# `customer` parameter -- direct-call tests that inject `customer=` stay
# valid, and the `operator` parameter simply defaults to None when omitted.

async def principal_customer(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
) -> Customer:
    """Project the team Customer out of the resolved principal."""
    return principal[0]


async def principal_operator(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal)
    ],
) -> Optional[Operator]:
    """Project the optional Operator out of the resolved principal."""
    return principal[1]


# ---------------------------------------------------------------------------
# Task-scoped principals (Phase 3 G3, 2026-07-03, ratified)
# ---------------------------------------------------------------------------
# The disposable box's only credential. Resolves ONLY through these two
# projections, which ONLY the public chat proxy uses — task keys are never
# accepted by resolve_principal / require_customer, so the whole SDK,
# document, and control surface rejects them naturally (restriction by
# routing, not by flags). Budget is checked HERE, at the door: a task whose
# ledger spend has reached its budget gets 429 on the next call, making the
# key self-limiting even if the remote-task monitor lags. The tenant's
# default admin is the acting operator (P1 identity chain), and the task_id
# is stashed on request.state so the proxy's cost row lands under
# session_id = task_id — which is exactly what the budget reads.

async def resolve_principal_or_task(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> tuple[Customer, Optional[Operator]]:
    """resolve_principal, extended to accept task-scoped keys (proxy only)."""
    token = _extract_bearer_token(request)
    if token.startswith("ck_task_"):
        task = await store.resolve_task_key(token)
        if task is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid, expired, or revoked task key",
            )
        spent = await store.task_spend_micro_usd(task.task_id)
        if task.budget_micro_usd > 0 and spent >= task.budget_micro_usd:
            raise HTTPException(
                status_code=429,
                detail="Task budget exhausted",
                headers={"Retry-After": "3600"},
            )
        team = await store.get_customer_by_id(task.customer_id)
        if team is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Task key tenant not found",
            )
        operator = await store.ensure_default_admin(team.id)
        request.state.task_key_task_id = task.task_id
        return team, operator
    return await resolve_principal(request, store)


async def task_principal_customer(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_task)
    ],
) -> Customer:
    """Project the team Customer (proxy-only dependency)."""
    return principal[0]


async def task_principal_operator(
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_task)
    ],
) -> Optional[Operator]:
    """Project the acting Operator (proxy-only dependency)."""
    return principal[1]


# ---------------------------------------------------------------------------
# Platform-admin credential (WS D / D.1) — the deployment-wide superuser gate
# ---------------------------------------------------------------------------
# Distinct from team/operator keys: a single static key (CC_ADMIN_API_KEY) that
# authorizes the cross-customer operator surface (/admin/api/*) and, in
# production, customer minting. Enforcement lives in the platform_admin_guard
# middleware in app.py — centralized so EVERY /admin/api router (admin,
# cognition, metacognition, and any added later) is covered without per-router
# wiring, and so the protected surface is auditable in one place. These two
# helpers are the single source of truth that the boot guard and the middleware
# share: "is the gate on" and "is this the admin key".


def _is_loopback_host(host: str) -> bool:
    """True when the bind host is loopback-only (nothing off-machine can
    reach the server). Used by the admin-gate fail-closed rule.

    Covers the common spellings: 127.0.0.0/8, ::1, and the literal
    'localhost'. An empty/unknown host is treated as NON-loopback
    (fail-closed — if we cannot prove it is loopback, assume networked).
    """
    import ipaddress

    h = (host or "").strip().lower()
    if not h:
        return False
    if h == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


def _is_hosted_platform_env() -> bool:
    """True when the process is running on a managed serverless platform
    where the container is reachable off-machine REGARDLESS of the app's
    `host` setting.

    This closes the Cloud Run fail-open gap (2026-07-06). The runtime binds
    the socket itself (the container CMD is `uvicorn --host 0.0.0.0`), while
    Settings.host keeps its `127.0.0.1` default — so `_is_loopback_host`
    reads a stale, misleading value and the gate wrongly concludes "loopback,
    nothing can reach us." The bind setting is NOT a trustworthy proxy for
    reachability on these platforms; a platform marker is.

    Google Cloud Run and Cloud Run Jobs always inject `K_SERVICE` (the
    service name) into the container environment; `K_REVISION` is likewise
    always present. Either marker means "hosted and networked" with
    certainty. A self-hoster on their own box / VM / laptop has neither, so
    local-dev ergonomics are untouched.

    Isolated in its own helper so tests can monkeypatch it, mirroring how the
    gate's other inputs are injected via get_settings.
    """
    return bool(os.environ.get("K_SERVICE") or os.environ.get("K_REVISION"))


def platform_admin_gate_active() -> bool:
    """True when the platform-admin gate is enforced.

    Fail-closed on any NETWORKED deployment (B2 hardening, 2026-07-03;
    Cloud Run fix, 2026-07-06). The gate is ENFORCED when ANY of these hold:
      * production (also refuses to boot without a key — the lifespan guard);
      * an admin key is configured (an operator opted in explicitly);
      * the process runs on a hosted serverless platform (Cloud Run &c.),
        where the container is reachable off-machine no matter what the
        `host` setting says;
      * the server is bound to a NON-loopback address (0.0.0.0, a LAN/public
        IP, etc.) — i.e. anything off-machine could reach /admin/api.

    The ONLY case the gate stays OFF is a loopback-bound dev server, not on a
    hosted platform, with no key configured — where nothing off the machine
    can reach the admin surface anyway, so the zero-config local-dev
    ergonomics are preserved. A networked deployment that genuinely wants the
    surface open without a key must opt out CONSCIOUSLY with
    CC_ADMIN_GATE_DISABLE=1 (documented in the deployment guide; strongly
    discouraged). The hatch applies ONLY to the non-production, non-hosted,
    no-key case — it can never disable production, hosted, or key-based
    enforcement.
    """
    s = get_settings()
    # Production and an explicit admin key ALWAYS enforce — the escape hatch
    # can never disable these (a hatch left on from dev must not open a
    # production admin surface). Production also refuses to boot without a
    # key via the lifespan guard.
    if s.is_production:
        return True
    if (getattr(s, "admin_api_key", "") or "").strip():
        return True
    # A hosted serverless platform is networked by definition and enforces
    # ABOVE the escape hatch: a CC_ADMIN_GATE_DISABLE left on from local dev
    # must never open a real Cloud Run admin surface. This is the clause that
    # the stale-`host`-default fooled before — the platform marker is trusted
    # over the (misleading) bind setting.
    if _is_hosted_platform_env():
        return True
    # Below here it is non-production, non-hosted, no key. The hatch applies
    # ONLY to this case: a conscious opt-out for keyless networked dev.
    if getattr(s, "admin_gate_disable", False):
        return False
    # Networked (non-loopback) bind with no key → fail closed.
    return not _is_loopback_host(getattr(s, "host", "") or "")


def is_platform_admin_token(token: Optional[str]) -> bool:
    """Constant-time check that `token` equals the configured admin key.

    The admin key is a process-side static secret (env, not DB), so a direct
    constant-time compare is the right tool — the DB-key pepper/hash exists for
    stored customer/operator keys, not for this. False when no key is
    configured or no token is presented.
    """
    configured = (getattr(get_settings(), "admin_api_key", "") or "").strip()
    if not configured or not token:
        return False
    return hmac.compare_digest(
        token.strip().encode("utf-8"), configured.encode("utf-8")
    )


def _bearer_token_from_header(authorization: Optional[str]) -> Optional[str]:
    """Extract the Bearer token from a raw Authorization header value, or None.

    Lenient (returns None rather than raising): the platform-admin gate treats
    a missing/!bearer header and a wrong key identically (both deny), so a
    parse failure simply yields no token.
    """
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def path_needs_platform_admin(method: str, path: str) -> bool:
    """Whether a request falls inside the platform-admin surface.

    Two protected regions: the entire cross-customer operator API
    (/admin/api/*) and customer minting (POST /v1/customers). The /admin SPA
    shell and its assets are deliberately NOT here — only the /admin/api
    prefix — so the Inspector bundle loads publicly and then authenticates its
    API calls. Team-scoped customer routes (GET/PATCH /v1/customers/{id}) are
    NOT platform surface; they want team auth, handled elsewhere.
    """
    if path.startswith("/admin/api"):
        return True
    if method.upper() == "POST" and path.rstrip("/") == "/v1/customers":
        return True
    return False


def platform_admin_error(
    method: str, path: str, authorization: Optional[str]
) -> Optional[tuple[int, str]]:
    """The whole platform-admin decision in one testable place.

    Returns (status_code, detail) when the request must be rejected, else None
    (allow). The app-side middleware is thin glue over this — it parses nothing
    and only renders the result. Order: gate inactive -> allow; outside the
    protected surface -> allow; inside it -> require the admin key.
    """
    if not platform_admin_gate_active():
        return None
    if not path_needs_platform_admin(method, path):
        return None
    if not is_platform_admin_token(_bearer_token_from_header(authorization)):
        return (401, "platform admin credential required")
    return None


# ---------------------------------------------------------------------------
# Hosted-identity principals + tenant admin guard (Accounts Phase A,
# 2026-07-06, ratified). Three principal kinds, one surface:
#   * the static platform-admin key (stage 1, above) — platform root;
#   * Firebase/Identity-Platform JWTs — hosted sign-in (users table);
#   * Key A — a tenant's own API credential.
# Stage 2 below runs ONLY when stage 1 (platform_admin_error) would deny:
# it grants tenant principals their OWN console slice — the tenant-pathed
# /admin/api/customers/{own-id}/* routes, plus read-only cognition /
# metacognition views force-pinned to their tenant (amended D3). Foreign
# tenant ids return 404 (never confirm existence); everything else under
# /admin/api stays platform-admin-only.
# ---------------------------------------------------------------------------


def _looks_like_firebase_jwt(token: str) -> bool:
    """Cheap shape check: JWTs are three dot-joined base64url segments
    starting 'eyJ' (base64 of '{"'). Key A / task keys never match, so the
    principal kinds are disjoint by construction (D2)."""
    if not token or not token.startswith("eyJ"):
        return False
    return token.count(".") == 2


def _verify_firebase_jwt(token: str, project_id: str) -> Optional[dict]:
    """Verify an Identity Platform JWT; return its claims or None.

    Isolated as a module function so tests monkeypatch it (the same seam
    pattern as get_settings). Real path uses google-auth's
    verify_firebase_token — JWKS fetch/cache/rotation + issuer/audience
    checks handled by Google's own library; hand-rolling JWT crypto is
    explicitly rejected (ratified D1). Lazy import: self-host deployments
    that never set CC_FIREBASE_PROJECT_ID never import google.auth here.
    """
    try:
        import google.auth.transport.requests
        from google.oauth2 import id_token as google_id_token

        request = google.auth.transport.requests.Request()
        return google_id_token.verify_firebase_token(
            token, request, audience=project_id
        )
    except Exception:
        # Bad signature / wrong audience / expired / malformed — all deny.
        return None


def _admin_bootstrap_emails() -> set[str]:
    raw = (getattr(get_settings(), "platform_admin_emails", "") or "")
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


async def resolve_firebase_user(store: MetadataStore, token: str):
    """Resolve a Firebase JWT bearer to a User, or None.

    None on: identity disabled (no CC_FIREBASE_PROJECT_ID — D4 presence-
    as-switch), invalid token, or an unknown uid whose email is not in the
    admin bootstrap allowlist. First login of an allowlisted email
    auto-provisions a platform_admin account (customer_id=None) — the
    ratified admin bootstrap. Regular-user provisioning is the signup flow
    (Phase B/C), NOT this resolver: an unknown non-allowlisted JWT is
    denied, it never conjures a tenant.
    """
    project_id = (getattr(get_settings(), "firebase_project_id", "") or "").strip()
    if not project_id:
        return None
    claims = _verify_firebase_jwt(token, project_id)
    if not claims:
        return None
    uid = claims.get("sub") or claims.get("user_id") or ""
    email = (claims.get("email") or "").strip().lower()
    if not uid:
        return None
    user = await store.get_user_by_id(uid)
    if user is not None:
        return user
    if email and email in _admin_bootstrap_emails():
        return await store.create_user(
            user_id=uid, email=email, customer_id=None, role="platform_admin"
        )
    return None


# Tenant-pathed console routes: /admin/api/customers/{cid}[/...]
_TENANT_PATH_RE = re.compile(r"^/admin/api/customers/([^/]+)(?:/|$)")

# Read-only cognition/metacognition views tenants may see PINNED to their
# own tenant (amended D3). Exact paths or prefixes; GET only. The list is
# deliberately explicit — a new admin route is platform-only until someone
# consciously adds it here.
# NOTE (C1, 2026-07-08): the substrate-observation endpoints were REMOVED
# from this list — System Critiques are a PLATFORM-ADMIN surface (Anthony:
# super-admin only). Tenants never see the system's complaints about
# itself.
_TENANT_READ_EXACT = frozenset({
    "/admin/api/cognition/environments",
    # S7 (2026-07-08): playground chat history — the tenant's own
    # sessions, pinned like every console read.
    "/admin/api/chat/sessions",
    # 2026-07-07 tenant-console sweep: the Cognition, Conflicts, and Bank
    # tabs' reads. All customer_id-parameterized handlers OVERRIDE the
    # param with the pin (same contract as cognition/api.py); the
    # crystal-detail handler enforces ownership against the pin (404 on
    # foreign — never an existence oracle). Found live: a signed-in
    # tenant 401'd browsing its OWN crystals and cognition views.
    "/admin/api/push-queue",
    "/admin/api/cognition-tasks",
    "/admin/api/knowledge-gaps",
    "/admin/api/conflicts",
    "/admin/api/backlog",
    "/admin/api/crystal_types",
})
_TENANT_READ_PREFIXES = (
    "/admin/api/cognition/environments/",
    "/admin/api/chat/sessions/",  # S7: one session's transcript
    "/admin/api/crystals/",  # detail: handler checks ownership vs pin
)


def _tenant_readable(method: str, path: str) -> bool:
    if method.upper() != "GET":
        return False
    p = path.rstrip("/") or path
    if p in _TENANT_READ_EXACT:
        return True
    return any(path.startswith(pre) for pre in _TENANT_READ_PREFIXES)


async def tenant_admin_error(
    method: str,
    path: str,
    authorization: Optional[str],
    store: MetadataStore,
) -> tuple[Optional[tuple[int, str]], Optional[str]]:
    """Stage 2 of the admin guard: the tenant slice.

    Called ONLY when stage 1 (platform_admin_error) would deny. Returns
    (error, tenant_pin): error None = allow; tenant_pin, when set, is the
    customer_id the request is force-scoped to (stashed on request.state
    by the middleware; pinned routes trust it over any query parameter).

    Decision order:
      * resolve the bearer to a principal — a platform_admin USER passes
        everything (equivalent to the static key); an owner user or a
        Key A customer yields a tenant id; anything else -> 401.
      * tenant-pathed route (/admin/api/customers/{cid}/*): own id ->
        allow unpinned (the path itself scopes); foreign id -> 404, the
        same shape as 'no such customer' (never an existence oracle).
      * tenant-readable cognition/metacognition GET -> allow with pin.
      * everything else stays platform-admin-only -> 401.
    """
    deny: tuple[int, str] = (401, "platform admin credential required")
    bearer = _bearer_token_from_header(authorization)
    if not bearer:
        return deny, None

    tenant_id: Optional[str] = None
    if _looks_like_firebase_jwt(bearer):
        user = await resolve_firebase_user(store, bearer)
        if user is None:
            return deny, None
        if user.role == "platform_admin":
            return None, None  # platform root — full surface, no pin
        tenant_id = user.customer_id
    else:
        customer = await store.get_customer_by_api_key(bearer)
        if customer is not None:
            tenant_id = customer.id

    if not tenant_id:
        return deny, None

    m = _TENANT_PATH_RE.match(path)
    if m:
        if m.group(1) == tenant_id:
            return None, None
        return (404, "Customer not found"), None

    if _tenant_readable(method, path):
        return None, tenant_id

    return deny, None


# ---------------------------------------------------------------------------
# Self-or-admin authorization for the customer-record routes (B1 fix,
# 2026-07-03). GET/PATCH /v1/customers/{id} were unauthenticated: anyone
# who knew a customer_id could read that customer's routing config and
# OVERWRITE their upstream key. These routes are now least-privilege:
# the caller must authenticate as the customer itself (its own Key A) OR
# as the platform admin. A mismatch returns 404 (not 403) so the endpoint
# never confirms whether a foreign customer_id exists.
# ---------------------------------------------------------------------------


async def require_customer_self_or_admin(
    customer_id: str,
    request: Request,
    store: MetadataStore,
) -> Customer:
    """Authorize access to a specific customer record.

    Allowed callers:
      * the customer itself — a valid Key A whose customer.id == customer_id;
      * the platform admin — a valid CC_ADMIN_API_KEY bearer token;
      * a hosted JWT principal (Accounts Phase C fix, 2026-07-06): the
        OWNER user of this customer, or a platform_admin user. The
        Settings surface authenticates with the session JWT, not Key A —
        without this path a signed-in tenant could not manage their own
        record (found live: the pinned console's key-save/mode-toggle
        404'd).

    Any other caller (no token, wrong customer's token, unknown token) gets
    404, identical to "customer not found", so the route is not an oracle
    for which customer ids exist. Not a FastAPI dependency (it needs the
    path param); handlers call it directly.
    """
    token = (
        request.headers.get("authorization")
        or request.headers.get("Authorization")
    )
    bearer = _bearer_token_from_header(token)

    # Admin path: a valid admin token may manage any customer, mirroring
    # POST /v1/customers which is already platform-admin gated.
    if bearer is not None and is_platform_admin_token(bearer):
        customer = await store.get_customer_by_id(customer_id)
        if customer is None:
            raise HTTPException(status_code=404, detail="Customer not found")
        return customer

    # Hosted-identity path: a Firebase JWT resolving to this customer's
    # OWNER, or to a platform_admin user (equivalent to the static key).
    if bearer is not None and _looks_like_firebase_jwt(bearer):
        user = await resolve_firebase_user(store, bearer)
        if user is not None:
            if user.role == "platform_admin" or user.customer_id == customer_id:
                customer = await store.get_customer_by_id(customer_id)
                if customer is None:
                    raise HTTPException(
                        status_code=404, detail="Customer not found")
                return customer
        # A valid-looking JWT that resolves to nobody (or a foreign
        # tenant) falls through to the uniform 404.

    # Self path: the presented Key A must resolve to THIS customer.
    if bearer is not None:
        caller = await store.get_customer_by_api_key(bearer)
        if caller is not None and caller.id == customer_id:
            return caller

    # Everything else is indistinguishable from "no such customer".
    raise HTTPException(status_code=404, detail="Customer not found")
