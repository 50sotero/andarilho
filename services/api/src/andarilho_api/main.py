"""FastAPI application factory."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from andarilho_api.ai.client import AIClient
from andarilho_api.auth import JwksCache, JwksFetcher, TokenVerifier
from andarilho_api.db import make_engine, make_session_factory
from andarilho_api.domain.imports import fetch_listing
from andarilho_api.errors import RequestContextMiddleware, install_error_handlers
from andarilho_api.ratelimit import RateLimiter
from andarilho_api.routes import capture_sets, health, imports, projects, runs, scenes, uploads
from andarilho_api.settings import Settings, get_settings
from andarilho_api.storage import Storage


def create_app(
    settings: Settings | None = None,
    *,
    jwks_fetcher: JwksFetcher | None = None,
    ai_client: AIClient | None = None,
    listing_fetcher: Callable[[str], str] | None = None,
) -> FastAPI:
    """Build the app. ``jwks_fetcher`` replaces the HTTP JWKS download (tests inject
    their own signing keys); verification itself is never skipped. ``ai_client``
    replaces the AI provider client and ``listing_fetcher`` the guarded listing
    page fetch (tests use mock transports)."""
    settings = settings or get_settings()
    app = FastAPI(
        title="Andarilho API",
        version="0.1.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url=None,
    )
    app.state.settings = settings
    app.state.engine = make_engine(settings.database_url)
    app.state.session_factory = make_session_factory(app.state.engine)
    app.state.storage = Storage(settings)
    app.state.token_verifier = TokenVerifier(
        JwksCache(
            settings.auth_jwks_url,
            fetcher=jwks_fetcher,
            ttl_seconds=settings.auth_jwks_cache_seconds,
            min_refresh_seconds=settings.auth_jwks_min_refresh_seconds,
        ),
        issuer=settings.auth_issuer,
        audience=settings.auth_audience,
        leeway=settings.auth_leeway_seconds,
    )
    app.state.public_rate_limiter = RateLimiter(settings.public_delivery_rate_limit_per_minute)
    app.state.ai_client = ai_client or AIClient(settings)
    app.state.ai_rate_limiter = RateLimiter(settings.ai_requests_per_minute)
    app.state.listing_fetcher = listing_fetcher or fetch_listing

    install_error_handlers(app)
    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
            allow_headers=["Authorization", "Content-Type", "If-Match", "Idempotency-Key"],
            expose_headers=["ETag", "X-Request-ID", "Retry-After"],
        )
    app.add_middleware(RequestContextMiddleware)

    app.include_router(health.router)
    for module in (projects, uploads, capture_sets, runs, imports):
        app.include_router(module.router, prefix="/v1")
    app.include_router(scenes.router, prefix="/v1")
    app.include_router(scenes.public_router, prefix="/v1")
    return app


def app_factory() -> FastAPI:
    """Entry point for ``uvicorn --factory andarilho_api.main:app_factory``."""
    return create_app()
