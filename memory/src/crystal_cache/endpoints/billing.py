"""Billing surface (L2-S3, Q4=B remedy A, 2026-09-08).

Two routes and nothing else:

- POST /v1/billing/checkout — authed (console or keys); creates a hosted
  Stripe Checkout session for the $29 starter tier. Card data never
  touches this server — Stripe's hosted page does (no PCI surface).
- POST /v1/billing/webhook — Stripe is the caller and the SIGNATURE is
  the auth (construct_event verifies HMAC + timestamp; no bearer). On
  checkout.session.completed the tenant flips to the paid tier and the
  trial clock clears — set_customer_subscription(cid, tier, None): the
  exact upgrade shape pinned in test_trial_expiry.

Settings empty (the self-host default) => both routes 404: no Stripe
surface exists unless the deployment opted in.
"""
from __future__ import annotations

import asyncio
from typing import Annotated, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from ..config import get_settings
from ..infrastructure.metadata_store import MetadataStore, get_metadata_store
from ..ingress.auth import resolve_principal_or_console
from ..models import Customer, Operator

logger = structlog.get_logger(__name__)

router = APIRouter()

# The paid tier the webhook stamps. Named once; S2's expiry logic treats
# any non-"trial*" tier as never-degrading.
STARTER_TIER = "starter_29"


class CheckoutRequest(BaseModel):
    # The console knows its own origin; it supplies where Stripe should
    # land the user afterward.
    success_url: str
    cancel_url: str


@router.post("/v1/billing/checkout")
async def create_checkout(
    body: CheckoutRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_console)
    ],
) -> dict:
    settings = get_settings()
    if not (settings.stripe_secret_key and settings.stripe_price_starter):
        raise HTTPException(status_code=404, detail="Billing is not enabled")
    customer, _operator = principal

    import stripe

    def _create():
        stripe.api_key = settings.stripe_secret_key
        return stripe.checkout.Session.create(
            mode="subscription",
            line_items=[{"price": settings.stripe_price_starter, "quantity": 1}],
            client_reference_id=customer.id,
            success_url=body.success_url,
            cancel_url=body.cancel_url,
        )

    session = await asyncio.to_thread(_create)
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
            payload, sig, settings.stripe_webhook_secret
        )
    except Exception:
        raise HTTPException(status_code=400, detail="invalid signature")

    if event["type"] == "checkout.session.completed":
        obj = event["data"]["object"] or {}
        cid = obj.get("client_reference_id")
        if cid:
            updated = await store.set_customer_subscription(
                cid,
                STARTER_TIER,
                trial_expires_at=None,
                # L2-S4=B: persist Stripe's customer join for the hosted
                # portal (plan management/invoices/cancel live on Stripe's
                # page, not ours).
                stripe_customer_id=obj.get("customer"),
            )
            logger.info(
                "billing.tier_upgraded",
                customer_id=cid,
                tier=STARTER_TIER,
                found=bool(updated),
            )
        else:
            logger.warning("billing.webhook_missing_reference")
    return {"received": True}


class PortalRequest(BaseModel):
    return_url: str


@router.post("/v1/billing/portal")
async def customer_portal(
    body: PortalRequest,
    principal: Annotated[
        tuple[Customer, Optional[Operator]], Depends(resolve_principal_or_console)
    ],
) -> dict:
    """L2-S4=B: open Stripe's hosted customer portal — plan management,
    invoices, cancellation, all Stripe's UI. 409 for tenants that never
    paid (no stripe_customer_id yet): the console shows the upgrade
    button instead."""
    settings = get_settings()
    if not settings.stripe_secret_key:
        raise HTTPException(status_code=404, detail="Billing is not enabled")
    customer, _operator = principal
    if not customer.stripe_customer_id:
        raise HTTPException(
            status_code=409,
            detail="No billing account yet — complete an upgrade first",
        )

    import stripe

    def _create():
        stripe.api_key = settings.stripe_secret_key
        return stripe.billing_portal.Session.create(
            customer=customer.stripe_customer_id,
            return_url=body.return_url,
        )

    session = await asyncio.to_thread(_create)
    return {"portal_url": session.url}
