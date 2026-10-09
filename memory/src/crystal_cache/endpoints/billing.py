"""Billing surface (L2-S3, Q4=B remedy A, 2026-09-08; Scale-solo and
plan lifecycle 2026-09-30).

- POST /v1/billing/checkout: authed (console or keys); creates a hosted
  Stripe Checkout session for a validated plan ("starter" $29 or
  "scale" $49, solo and uncapped). FREE tenants only (Q4=A): a tenant
  already on a paid plan gets 409 and changes plans in the portal, so a
  second subscription can never be created. The chosen price id rides
  in session metadata because checkout.session.completed carries no
  line items. Card data never touches this server (no PCI surface).
- POST /v1/billing/webhook: Stripe is the caller and the SIGNATURE is
  the auth (construct_event verifies HMAC + timestamp; no bearer).
  checkout.session.completed maps the price id to its tier, clears the
  trial clock, and persists the Stripe customer join.
  customer.subscription.updated (portal plan switches) maps the
  subscription's price id to its tier; customer.subscription.deleted
  drops the tenant to free (reads intact, writes walled past the free
  cap). An unknown price id is logged and changes nothing: never
  default a stranger's price to a paid tier.
- POST /v1/billing/portal: Stripe's hosted portal for paid tenants.

Settings empty (the self-host default) => the routes 404: no Stripe
surface exists unless the deployment opted in.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Annotated, Literal, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from .. import hygiene
from ..config import get_settings
from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..ingress.auth import resolve_principal_or_session
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()


def _require_billing_principal(operator: Optional[Operator]) -> None:
    """Lockdown PR-3 (B4-3, 2026-10-09): checkout and the portal change
    what the workspace pays. resolve_principal_or_session already holds a
    console session to owner-side roles (operator=None here); a key
    principal must be an admin (Key A acts as the Default Admin). A
    member or viewer key gets 403 admin_required."""
    if operator is None or getattr(operator, "role", None) == "admin":
        return
    raise HTTPException(
        status_code=403,
        detail={"error": {
            "message": "Only a workspace admin can manage billing.",
            "type": "permission_error", "code": "admin_required",
        }},
    )

# The tiers the webhook stamps. Named once; S2's expiry logic treats any
# non-"trial*" tier as never-degrading. SCALE_TIER keeps its historical
# stamped key (see admission.TIER_TABLE) though Scale is solo at launch.
STARTER_TIER = "starter_29"
SCALE_TIER = "scale_49_seat"
FREE_TIER = "free"
PAID_TIERS: tuple[str, ...] = (STARTER_TIER, SCALE_TIER)

# Subscription statuses that keep the paid tier. past_due keeps it on
# purpose: Stripe retries the card, and a failed retry run ends in
# canceled/unpaid, which drops to free below.
_KEEP_STATUSES = ("active", "trialing", "past_due")
_DROP_STATUSES = ("canceled", "unpaid", "incomplete_expired")


def price_tier_map(settings) -> dict[str, str]:
    """Configured price id -> stamped tier. Only configured prices map;
    anything else is unknown by construction."""
    out: dict[str, str] = {}
    if settings.stripe_price_starter:
        out[settings.stripe_price_starter] = STARTER_TIER
    if settings.stripe_price_scale:
        out[settings.stripe_price_scale] = SCALE_TIER
    return out


def _is_paid(customer) -> bool:
    return (
        getattr(customer, "subscription_tier", None) in PAID_TIERS
        and bool(getattr(customer, "stripe_customer_id", None))
    )


def _get(obj, key, default=None):
    """Tolerant field read across dicts and StripeObjects."""
    try:
        val = obj.get(key, default) if hasattr(obj, "get") else obj[key]
    except (KeyError, TypeError, AttributeError):
        return default
    return default if val is None else val


def _subscription_price_id(sub) -> Optional[str]:
    items = _get(_get(sub, "items", {}), "data", []) or []
    if not items:
        return None
    return _get(_get(items[0], "price", {}), "id")


async def _stripe_call(fn):
    """Run a blocking Stripe SDK call off-thread and convert provider
    rejections into a clean 502 carrying Stripe's own message (2026-09-22:
    a Managed-Payments tax-code rejection surfaced as a bare 500 and cost
    a log dive that a user could never do).

    2026-09-30 incident: a secret stored with a trailing newline made the
    SDK raise InvalidHeader with the FULL live key in the message, which
    this wrapper then logged AND returned to the browser. Q6=C: provider
    text NEVER reaches the response. The user gets a fixed support
    message plus a reference id; the redacted detail goes to the logs
    only, findable by that id."""
    try:
        return await asyncio.to_thread(fn)
    except HTTPException:
        raise
    except Exception as e:
        ref = uuid.uuid4().hex[:10]
        logger.error(
            "billing.stripe_error", ref=ref,
            error_type=type(e).__name__, error=_redact(str(e))[:500],
        )
        raise HTTPException(
            status_code=502,
            detail=(
                "We couldn't reach our payment provider. Please try again "
                f"in a few minutes, or contact support at {SUPPORT_EMAIL} "
                f"and mention reference {ref}."
            ),
        )


SUPPORT_EMAIL = hygiene.SUPPORT_EMAIL


# N7 (security sweep 2026-09-30): one redactor for the whole codebase.
_redact = hygiene.redact


def _clean(secret: str) -> str:
    """Secrets arrive from Secret Manager byte-exact; a stray newline from
    `echo` breaks every Stripe call (2026-09-30). Settings strips at load
    (S1); this is the belt for a Settings built any other way."""
    return (secret or "").strip()


class CheckoutRequest(BaseModel):
    # The console knows its own origin; it supplies where Stripe should
    # land the user afterward.
    success_url: str
    cancel_url: str
    # Validated by the Literal (422 on anything else). Default starter
    # keeps pre-v103 consoles working unchanged.
    plan: Literal["starter", "scale"] = "starter"


def _require_return_url(url: str, settings, *, field: str) -> str:
    """S8 (security sweep 2026-09-30): the console supplies the URLs
    Stripe sends the user back to. Unvalidated, a crafted request could
    bounce a paying customer from Stripe's page to any site. Only https
    URLs on an allow-listed host pass (http only on loopback for dev)."""
    from urllib.parse import urlparse
    try:
        u = urlparse(url)
    except ValueError:
        u = None
    hosts = {
        h.strip().lower()
        for h in (settings.billing_return_hosts or "").split(",") if h.strip()
    }
    loopback = {"localhost", "127.0.0.1", "[::1]"}
    ok = (
        u is not None
        and u.hostname
        and (
            u.scheme == "https"
            or (u.scheme == "http" and u.hostname in loopback)
        )
        and (u.hostname.lower() in hosts or u.hostname in loopback)
    )
    if not ok:
        raise HTTPException(
            status_code=400,
            detail=f"{field} must be an https URL on the console host",
        )
    return url


@router.post("/v1/billing/checkout")
async def create_checkout(
    body: CheckoutRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_session)
    ],
) -> dict:
    settings = get_settings()
    if not (settings.stripe_secret_key and settings.stripe_price_starter):
        raise HTTPException(status_code=404, detail="Billing is not enabled")
    customer, operator = principal
    _require_billing_principal(operator)
    success_url = _require_return_url(body.success_url, settings, field="success_url")
    cancel_url = _require_return_url(body.cancel_url, settings, field="cancel_url")
    if _is_paid(customer):
        # Q4=A: one subscription per tenant. Plan changes go through the
        # portal (customer.subscription.updated maps the new price).
        raise HTTPException(
            status_code=409,
            detail="You already have a paid plan. Change plans in the billing portal.",
        )
    price = (
        settings.stripe_price_scale if body.plan == "scale"
        else settings.stripe_price_starter
    )
    if not price:
        raise HTTPException(
            status_code=400, detail=f"The {body.plan} plan is not available",
        )

    import stripe

    def _create():
        stripe.api_key = _clean(settings.stripe_secret_key)
        kwargs = dict(
            mode="subscription",
            line_items=[{"price": price, "quantity": 1}],
            client_reference_id=customer.id,
            # checkout.session.completed has no line items: the webhook
            # reads the tier back from this.
            metadata={"price_id": price},
            success_url=success_url,
            cancel_url=cancel_url,
        )
        if settings.stripe_managed_payments:
            # MoR contract (Stripe onboarding, 2026-09-23): explicit opt-in
            # per session + the basil API version or later. Forbidden
            # params under MoR (automatic_tax, payment_method_*, …) are
            # already absent from this call by construction.
            stripe.api_version = "2025-03-31.basil"
            kwargs["managed_payments"] = {"enabled": True}
        return stripe.checkout.Session.create(**kwargs)

    session = await _stripe_call(_create)
    return {"checkout_url": session.url, "session_id": session.id}


@router.post("/v1/billing/webhook")
async def stripe_webhook(
    request: Request,
    store: Annotated[MetadataStore, Depends(get_metadata_store)],
) -> dict:
    """No auth dependency BY DESIGN: Stripe calls this, and the verified
    signature (HMAC over timestamp.payload with the webhook secret) is
    the authentication. Everything else is a 400."""
    settings = get_settings()
    if not settings.stripe_webhook_secret:
        raise HTTPException(status_code=404, detail="Billing is not enabled")
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")

    import stripe

    try:
        event = stripe.Webhook.construct_event(
            payload, sig, _clean(settings.stripe_webhook_secret)
        )
    except Exception:
        raise HTTPException(status_code=400, detail="invalid signature")

    etype = event["type"]
    obj = event["data"]["object"] or {}
    prices = price_tier_map(settings)
    event_id = str(event.get("id") or "")
    created = int(event.get("created") or 0)

    # RC-13 (2026-10-05): Stripe retries and does not order deliveries.
    # A replayed event id is a no-op; an event older than the newest one
    # already processed for the customer is stale and must not overwrite.
    if event_id and await store.billing_event_seen(event_id):
        logger.info("billing.webhook_replayed", event_type=etype, event_id=event_id)
        return {"received": True, "duplicate": True}

    async def _done(customer_id: Optional[str]) -> dict:
        if event_id:
            await store.record_billing_event(
                event_id, customer_id=customer_id, event_type=etype, created=created,
            )
        return {"received": True}

    async def _stale(customer_id: str) -> bool:
        return created < await store.latest_billing_event_created(customer_id)

    async def _after_tier_change(customer_id: str) -> None:
        # A downgrade must not leave a model the plan no longer includes.
        from ..control.admission import clamp_stored_model_to_tier

        c = await store.get_customer_by_id(customer_id)
        moved = await clamp_stored_model_to_tier(store, c)
        if moved:
            logger.info("billing.model_clamped", customer_id=customer_id, model=moved)

    if etype == "checkout.session.completed":
        cid = _get(obj, "client_reference_id")
        if not cid:
            logger.warning("billing.webhook_missing_reference")
            return await _done(None)
        # Only a PAID session grants a tier (no_payment_required covers a
        # 100% coupon or a free-trial checkout).
        pay = _get(obj, "payment_status")
        if pay not in ("paid", "no_payment_required"):
            logger.warning("billing.checkout_unpaid", customer_id=cid, payment_status=pay)
            return await _done(cid)
        if await _stale(cid):
            logger.info("billing.webhook_stale", event_type=etype, customer_id=cid)
            return await _done(cid)
        price_id = _get(_get(obj, "metadata", {}), "price_id")
        if price_id:
            tier = prices.get(price_id)
            if tier is None:
                logger.error(
                    "billing.unknown_price", customer_id=cid, price_id=price_id,
                )
                return await _done(cid)
        else:
            # Sessions created before v103 carried no metadata and were
            # Starter-only by construction.
            tier = STARTER_TIER
        updated = await store.set_customer_subscription(
            cid,
            tier,
            trial_expires_at=None,
            # L2-S4=B: persist Stripe's customer join for the hosted
            # portal (plan management/invoices/cancel live on Stripe's
            # page, not ours).
            stripe_customer_id=_get(obj, "customer"),
        )
        logger.info(
            "billing.tier_upgraded", customer_id=cid, tier=tier,
            found=bool(updated),
        )
        await _after_tier_change(cid)
        return await _done(cid)

    if etype in ("customer.subscription.updated", "customer.subscription.deleted"):
        c = await store.get_customer_by_stripe_customer_id(_get(obj, "customer"))
        if c is None:
            logger.warning(
                "billing.lifecycle_unknown_customer", event_type=etype,
            )
            return await _done(None)
        if await _stale(c.id):
            logger.info("billing.webhook_stale", event_type=etype, customer_id=c.id)
            return await _done(c.id)
        status = _get(obj, "status")
        if etype == "customer.subscription.deleted" or status in _DROP_STATUSES:
            tier = FREE_TIER
        elif status in _KEEP_STATUSES:
            price_id = _subscription_price_id(obj)
            tier = prices.get(price_id) if price_id else None
            if tier is None:
                logger.error(
                    "billing.unknown_price", customer_id=c.id, price_id=price_id,
                )
                return await _done(c.id)
        else:
            # incomplete and friends: no tier change until Stripe settles.
            return await _done(c.id)
        if tier != c.subscription_tier:
            await store.set_customer_subscription(c.id, tier, trial_expires_at=None)
            logger.info(
                "billing.tier_changed", customer_id=c.id,
                tier=tier, event_type=etype,
            )
            await _after_tier_change(c.id)
        return await _done(c.id)

    return await _done(None)


class PortalRequest(BaseModel):
    return_url: str


async def cancel_subscriptions_for_customer(customer) -> int:
    """Account deletion (2026-10-03): cancel every active Stripe
    subscription on the tenant's Stripe customer, immediately. Returns
    how many were cancelled; 0 when billing is off or the tenant never
    paid. Failures are logged (redacted) and never block the deletion:
    the webhook's deleted event will still drop the tier if Stripe
    cancels later, and the purge removes the tenant regardless."""
    settings = get_settings()
    stripe_cid = getattr(customer, "stripe_customer_id", None)
    if not (settings.stripe_secret_key and stripe_cid):
        return 0

    import stripe

    def _cancel():
        stripe.api_key = _clean(settings.stripe_secret_key)
        subs = stripe.Subscription.list(customer=stripe_cid, status="active", limit=10)
        n = 0
        for sub in subs.auto_paging_iter():
            stripe.Subscription.cancel(sub.id)
            n += 1
        return n

    try:
        return int(await asyncio.to_thread(_cancel))
    except Exception as e:  # noqa: BLE001
        hygiene.safe_error(
            "billing.cancel_on_delete_failed", e,
            customer_id=getattr(customer, "id", None),
        )
        return 0


@router.post("/v1/billing/portal")
async def customer_portal(
    body: PortalRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_session)
    ],
) -> dict:
    """L2-S4=B: open Stripe's hosted customer portal — plan management,
    invoices, cancellation, all Stripe's UI. 409 for tenants that never
    paid (no stripe_customer_id yet): the console shows the upgrade
    button instead."""
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=404, detail="Billing is not enabled")
    customer, operator = principal
    _require_billing_principal(operator)
    return_url = _require_return_url(body.return_url, settings, field="return_url")
    if not customer.stripe_customer_id:
        raise HTTPException(
            status_code=409,
            detail="No billing account yet — complete an upgrade first",
        )

    import stripe

    def _create():
        stripe.api_key = _clean(settings.stripe_secret_key)
        return stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            return_url=return_url,
        )

    session = await _stripe_call(_create)
    return {"portal_url": session.url}
