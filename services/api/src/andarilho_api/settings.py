"""Environment-driven configuration (README section 14).

Variable names match ``.env.example``. ``APP_ENV=production`` rejects the
example/local credentials and insecure auth endpoints, and always disables the
synthetic recipe.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIB = 1024 * 1024
GIB = 1024 * MIB

# Values that only ever appear in examples or local development.
_EXAMPLE_MARKERS = (
    "example.supabase.co",
    "example.com",
    "local-password",
    "local-development-only",
)
_LOCAL_S3_CREDENTIALS = {"local-development-only", "minioadmin", "minio", "testing", "test"}


class Settings(BaseSettings):
    # hide_input_in_errors: validation errors must never print credential values.
    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=False, hide_input_in_errors=True
    )

    app_env: Literal["development", "test", "production"] = "development"

    database_url: str = "postgresql+psycopg://andarilho:local-password@localhost:5432/andarilho"

    auth_issuer: str = "https://example.supabase.co/auth/v1"
    auth_audience: str = "authenticated"
    auth_jwks_url: str = "https://example.supabase.co/auth/v1/.well-known/jwks.json"
    auth_jwks_cache_seconds: int = Field(default=600, ge=30, le=86400)
    auth_jwks_min_refresh_seconds: int = Field(default=30, ge=1, le=3600)
    auth_leeway_seconds: int = Field(default=10, ge=0, le=120)

    # Internal endpoint for server operations; public endpoint only signs browser URLs.
    # Leave both unset for ordinary AWS S3.
    s3_endpoint_url: str | None = None
    s3_public_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "andarilho-local"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None

    # Intake limits (README section 2). Configurable starting limits.
    upload_max_total_bytes: int = Field(default=2 * GIB, gt=0)
    upload_max_video_bytes: int | None = Field(default=None, gt=0)
    upload_max_photo_bytes: int = Field(default=50 * MIB, gt=0)
    upload_max_floor_plan_bytes: int = Field(default=20 * MIB, gt=0)
    upload_max_photos: int = Field(default=300, gt=0)
    upload_part_size: int = Field(default=16 * MIB, ge=5 * MIB, le=5 * GIB)
    upload_max_parts: int = Field(default=1000, ge=1, le=10000)
    upload_ttl_seconds: int = Field(default=86400, ge=300)
    upload_part_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    run_max_seconds: int = Field(default=3600, gt=0)
    delivery_url_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    idempotency_ttl_seconds: int = Field(default=86400, ge=60)

    # None means "default for the environment": true outside production.
    allow_synthetic_recipe: bool | None = None

    public_delivery_rate_limit_per_minute: int = Field(default=60, ge=1)
    # Only enable when the API sits behind a proxy that overwrites X-Forwarded-For.
    public_delivery_trust_forwarded_for: bool = False

    # CORS is off by default; the web app proxies requests same-origin.
    cors_allowed_origins: list[str] = Field(default_factory=list)

    @property
    def video_max_bytes(self) -> int:
        return self.upload_max_video_bytes or self.upload_max_total_bytes

    @property
    def synthetic_recipe_enabled(self) -> bool:
        return bool(self.allow_synthetic_recipe)

    @model_validator(mode="after")
    def _environment_rules(self) -> Settings:
        if self.app_env != "production":
            if self.allow_synthetic_recipe is None:
                self.allow_synthetic_recipe = True
            return self

        # Production: the synthetic fixture recipe is never available.
        self.allow_synthetic_recipe = False
        problems: list[str] = []
        for name in ("auth_issuer", "auth_jwks_url"):
            value = getattr(self, name)
            if urlparse(value).scheme != "https":
                problems.append(f"{name.upper()} must use https in production")
        checked = {
            "DATABASE_URL": self.database_url,
            "AUTH_ISSUER": self.auth_issuer,
            "AUTH_JWKS_URL": self.auth_jwks_url,
            "S3_ACCESS_KEY_ID": self.s3_access_key_id or "",
            "S3_SECRET_ACCESS_KEY": self.s3_secret_access_key or "",
        }
        for name, value in checked.items():
            if any(marker in value.lower() for marker in _EXAMPLE_MARKERS):
                problems.append(f"{name} contains an example/local value")
        for name in ("s3_access_key_id", "s3_secret_access_key"):
            value = getattr(self, name)
            if value is not None and value.lower() in _LOCAL_S3_CREDENTIALS:
                problems.append(f"{name.upper()} is a local development credential")
        if problems:
            # Names only: never echo credential values.
            raise ValueError("invalid production configuration: " + "; ".join(sorted(problems)))
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
