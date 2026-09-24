"""The OAuth authorization server provider (L2-S5a, 2026-09-24).

Implements the mcp SDK's OAuthAuthorizationServerProvider protocol
against the oauth_records store (D12 mixin). This is what makes
Claude's "Add custom connector" dialog work: the SDK's create_auth_routes
mounts /authorize, /token, /register, and discovery metadata around
this class; we supply identity (subject = operator_id, minted by the
S5b consent bridge) and storage.

Token shapes (Q2=A, Q3=A ratified 2026-09-24): opaque random strings —
access `cco_…` (1 hour), refresh `ccr_…` (30 days, ROTATED on use:
the old refresh token is revoked the moment it is exchanged, so a
stolen-and-replayed one dies loudly). Static `cc_…` keys are a
different door entirely and never touched by any of this.
"""
import secrets
import time
from typing import Optional

from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    OAuthAuthorizationServerProvider,
    RefreshToken,
    TokenError,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ..config import settings

ACCESS_TTL_SECONDS = 3600           # Q3=A: 1 hour
REFRESH_TTL_SECONDS = 30 * 86400    # Q3=A: 30 days
CODE_TTL_SECONDS = 600              # authorization codes live 10 minutes


def _now() -> int:
    return int(time.time())


class CrystalOAuthProvider(
    OAuthAuthorizationServerProvider[AuthorizationCode, RefreshToken, AccessToken]
):
    def __init__(self, store) -> None:
        # Accepts a MetadataStore instance (tests) OR a zero-arg getter
        # (app wiring passes get_metadata_store so nothing constructs an
        # engine at import time).
        self._store_ref = store

    @property
    def _store(self):
        s = self._store_ref
        return s() if callable(s) else s

    # ── clients (Q1=A: open dynamic registration) ──────────────────────

    async def get_client(
        self, client_id: str
    ) -> Optional[OAuthClientInformationFull]:
        rec = await self._store.oauth_get("client", client_id)
        if rec is None:
            return None
        return OAuthClientInformationFull.model_validate_json(rec["data"])

    async def register_client(
        self, client_info: OAuthClientInformationFull
    ) -> None:
        await self._store.oauth_put(
            "client",
            client_info.client_id,
            client_info.model_dump_json(),
            client_id=client_info.client_id,
        )

    # ── authorize (S5b's consent page finishes this hop) ───────────────

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        """Send the user to the Inspector-hosted sign-in/consent page
        (Q4=A). The page authenticates via the existing Firebase stack,
        then calls the S5b bridge endpoint, which mints the code via
        create_authorization_code below and redirects back to the
        client's redirect_uri."""
        from urllib.parse import urlencode

        q = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "state": params.state or "",
            "code_challenge": params.code_challenge,
            "scopes": " ".join(params.scopes or []),
            "explicit_redirect": "1" if params.redirect_uri_provided_explicitly else "0",
        }
        if params.resource:
            q["resource"] = str(params.resource)
        return f"{settings.oauth_consent_url}?{urlencode(q)}"

    async def create_authorization_code(
        self,
        client_id: str,
        operator_id: str,
        redirect_uri: str,
        code_challenge: str,
        scopes: list[str],
        redirect_uri_provided_explicitly: bool,
        resource: Optional[str] = None,
    ) -> str:
        """S5b bridge helper (not part of the SDK protocol): mint the
        code after the consent page proves identity."""
        code = "ccc_" + secrets.token_urlsafe(32)
        model = AuthorizationCode(
            code=code,
            scopes=scopes,
            expires_at=_now() + CODE_TTL_SECONDS,
            client_id=client_id,
            code_challenge=code_challenge,
            redirect_uri=redirect_uri,  # type: ignore[arg-type]
            redirect_uri_provided_explicitly=redirect_uri_provided_explicitly,
            resource=resource,  # type: ignore[arg-type]
            subject=operator_id,
        )
        await self._store.oauth_put(
            "code", code, model.model_dump_json(),
            client_id=client_id, subject=operator_id,
        )
        return code

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> Optional[AuthorizationCode]:
        rec = await self._store.oauth_get("code", authorization_code)
        if rec is None:
            return None
        model = AuthorizationCode.model_validate_json(rec["data"])
        if model.client_id != client.client_id or model.expires_at < _now():
            return None
        return model

    async def exchange_authorization_code(
        self,
        client: OAuthClientInformationFull,
        authorization_code: AuthorizationCode,
    ) -> OAuthToken:
        # Single use: the code dies before the tokens are born.
        await self._store.oauth_delete("code", authorization_code.code)
        return await self._mint_pair(
            client.client_id,
            authorization_code.subject,
            authorization_code.scopes,
            authorization_code.resource,
        )

    # ── refresh (rotation on every use) ────────────────────────────────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> Optional[RefreshToken]:
        rec = await self._store.oauth_get("refresh", refresh_token)
        if rec is None:
            return None
        model = RefreshToken.model_validate_json(rec["data"])
        if model.client_id != client.client_id:
            return None
        if model.expires_at is not None and model.expires_at < _now():
            return None
        return model

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        await self._store.oauth_revoke("refresh", refresh_token.token)
        return await self._mint_pair(
            client.client_id,
            refresh_token.subject,
            scopes or refresh_token.scopes,
            refresh_token.resource,
        )

    # ── access tokens (what the MCP door verifies in S5c) ──────────────

    async def load_access_token(self, token: str) -> Optional[AccessToken]:
        rec = await self._store.oauth_get("access", token)
        if rec is None:
            return None
        model = AccessToken.model_validate_json(rec["data"])
        if model.expires_at is not None and model.expires_at < _now():
            return None
        return model

    async def revoke_token(
        self, token: AccessToken | RefreshToken
    ) -> None:
        kind = "refresh" if isinstance(token, RefreshToken) else "access"
        await self._store.oauth_revoke(kind, token.token)

    # ── internals ──────────────────────────────────────────────────────

    async def _mint_pair(
        self,
        client_id: str,
        subject: Optional[str],
        scopes: list[str],
        resource,
    ) -> OAuthToken:
        if not subject:
            raise TokenError("invalid_grant", "code carries no subject")
        access = AccessToken(
            token="cco_" + secrets.token_urlsafe(32),
            client_id=client_id,
            scopes=scopes,
            expires_at=_now() + ACCESS_TTL_SECONDS,
            resource=resource,
            subject=subject,
        )
        refresh = RefreshToken(
            token="ccr_" + secrets.token_urlsafe(32),
            client_id=client_id,
            scopes=scopes,
            expires_at=_now() + REFRESH_TTL_SECONDS,
            resource=resource,
            subject=subject,
        )
        await self._store.oauth_put(
            "access", access.token, access.model_dump_json(),
            client_id=client_id, subject=subject,
        )
        await self._store.oauth_put(
            "refresh", refresh.token, refresh.model_dump_json(),
            client_id=client_id, subject=subject,
        )
        return OAuthToken(
            access_token=access.token,
            token_type="Bearer",
            expires_in=ACCESS_TTL_SECONDS,
            refresh_token=refresh.token,
            scope=" ".join(scopes) if scopes else None,
        )
