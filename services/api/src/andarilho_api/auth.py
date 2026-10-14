"""Supabase access-token verification against the configured JWKS.

Signature, issuer, audience, expiry and the algorithm allowlist (RS256/ES256)
are always checked. Symmetric (HS*) and ``none`` tokens are rejected. There is
no bypass flag: tests inject their own signing keys through ``fetcher``.
Project membership is checked separately (``domain/membership.py``).
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import jwt
from fastapi import Request

from andarilho_api.errors import ApiError, unauthenticated

logger = logging.getLogger("andarilho_api.auth")

ALLOWED_ALGORITHMS = ("RS256", "ES256")
_ALLOWED_KTY = {"RSA", "EC"}

JwksFetcher = Callable[[str], dict[str, Any]]


def http_fetch_jwks(url: str) -> dict[str, Any]:
    response = httpx.get(url, timeout=5.0, follow_redirects=False)
    response.raise_for_status()
    data = response.json()
    if not isinstance(data, dict):
        raise ValueError("JWKS document is not an object")
    return data


class JwksCache:
    """Caches signing keys with a bounded TTL; refreshes on an unknown ``kid`` at most
    once per ``min_refresh_seconds`` so random kids cannot hammer the provider."""

    def __init__(
        self,
        url: str,
        *,
        fetcher: JwksFetcher | None = None,
        ttl_seconds: int = 600,
        min_refresh_seconds: int = 30,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = url
        self._fetcher = fetcher or http_fetch_jwks
        self._ttl = ttl_seconds
        self._min_refresh = min_refresh_seconds
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._last_attempt: float | None = None
        self._lock = threading.Lock()
        self.fetch_count = 0

    def _refresh(self, now: float) -> None:
        self._last_attempt = now
        self.fetch_count += 1
        document = self._fetcher(self.url)
        keys: dict[str, jwt.PyJWK] = {}
        for raw in document.get("keys", []):
            if not isinstance(raw, dict):
                continue
            kid = raw.get("kid")
            if not isinstance(kid, str) or raw.get("kty") not in _ALLOWED_KTY:
                continue
            if raw.get("use", "sig") != "sig":
                continue
            alg = raw.get("alg")
            if alg is not None and alg not in ALLOWED_ALGORITHMS:
                continue
            try:
                keys[kid] = jwt.PyJWK(raw)
            except jwt.PyJWTError:
                continue
        self._keys = keys
        self._fetched_at = now

    def get(self, kid: str) -> jwt.PyJWK:
        with self._lock:
            now = self._clock()
            failed = False

            def may_fetch() -> bool:
                return self._last_attempt is None or now - self._last_attempt >= self._min_refresh

            try:
                stale = self._fetched_at is None or now - self._fetched_at > self._ttl
                if stale and may_fetch():
                    self._refresh(now)
                if kid not in self._keys and may_fetch():
                    self._refresh(now)
            except (httpx.HTTPError, ValueError, TypeError, AttributeError) as exc:
                # Keep serving known keys during a provider outage.
                logger.warning("JWKS refresh failed: %s", type(exc).__name__)
                failed = True
            key = self._keys.get(kid)
            if key is None and (failed or self._fetched_at is None):
                raise ApiError(
                    503,
                    "AUTH_UNAVAILABLE",
                    "Sign-in verification is temporarily unavailable.",
                    retryable=True,
                    suggested_action="Try again in a moment.",
                )
            if key is None:
                raise unauthenticated("The access token is not valid.")
            return key


class TokenVerifier:
    def __init__(self, jwks: JwksCache, *, issuer: str, audience: str, leeway: int = 10) -> None:
        self.jwks = jwks
        self.issuer = issuer
        self.audience = audience
        self.leeway = leeway

    def verify(self, token: str) -> dict[str, Any]:
        invalid = unauthenticated("The access token is not valid.")
        try:
            header = jwt.get_unverified_header(token)
        except jwt.PyJWTError:
            raise invalid from None
        alg = header.get("alg")
        kid = header.get("kid")
        if alg not in ALLOWED_ALGORITHMS or not isinstance(kid, str) or not kid:
            raise invalid
        key = self.jwks.get(kid)
        if key.algorithm_name != alg:
            raise invalid
        try:
            claims = jwt.decode(
                token,
                key=key.key,
                algorithms=[alg],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.leeway,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.ExpiredSignatureError:
            raise unauthenticated("The access token has expired.") from None
        except jwt.PyJWTError:
            raise invalid from None
        sub = claims.get("sub")
        if not isinstance(sub, str) or not sub or len(sub) > 255:
            raise invalid
        return claims


def current_subject(request: Request) -> str:
    """FastAPI dependency: the verified ``sub`` claim of the bearer token."""
    header = request.headers.get("authorization")
    if not header:
        raise unauthenticated()
    scheme, _, token = header.partition(" ")
    token = token.strip()
    if scheme.lower() != "bearer" or not token or len(token) > 8192:
        raise unauthenticated()
    verifier: TokenVerifier = request.app.state.token_verifier
    return verifier.verify(token)["sub"]
