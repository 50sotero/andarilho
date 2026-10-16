"""Environment-driven configuration (README section 14).

Variable names match ``.env.example``. ``APP_ENV=production`` rejects the
example/local credentials and insecure auth and browser-facing storage endpoints,
and always disables the synthetic recipe.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

MIB = 1024 * 1024
GIB = 1024 * MIB

# A capture set lists at most this many videos and photos, and a run reads at most
# RUN_MAX_VIDEOS videos (the recipes' maxVideos). The photo limit leaves room for
# them, so any selection the published intake policy allows fits in a capture set.
CAPTURE_SET_MAX_INPUTS = 400
RUN_MAX_VIDEOS = 8
# The API freezes each completed upload with one server-side copy, which S3 (and
# MinIO) perform only up to 5 GiB, so no single file may be larger.
S3_SINGLE_COPY_MAX_BYTES = 5 * GIB
# Idempotency records last at least as long as the web keeps a key to replay them.
IDEMPOTENCY_MIN_TTL_SECONDS = 24 * 60 * 60

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
    upload_max_video_bytes: int | None = Field(default=None, gt=0, le=S3_SINGLE_COPY_MAX_BYTES)
    upload_max_photo_bytes: int = Field(default=50 * MIB, gt=0, le=S3_SINGLE_COPY_MAX_BYTES)
    upload_max_floor_plan_bytes: int = Field(default=20 * MIB, gt=0, le=S3_SINGLE_COPY_MAX_BYTES)
    upload_max_photos: int = Field(default=300, gt=0, le=CAPTURE_SET_MAX_INPUTS - RUN_MAX_VIDEOS)
    upload_part_size: int = Field(default=16 * MIB, ge=5 * MIB, le=5 * GIB)
    upload_max_parts: int = Field(default=1000, ge=1, le=10000)
    upload_ttl_seconds: int = Field(default=86400, ge=300)
    upload_part_url_ttl_seconds: int = Field(default=900, ge=60, le=3600)

    run_max_seconds: int = Field(default=3600, gt=0)
    delivery_url_ttl_seconds: int = Field(default=300, ge=30, le=3600)
    # The web keeps an upload's creation key this long and sends it again after a lost
    # response (CREATION_KEY_TTL_MS in apps/web/src/lib/upload-resume.ts). A record that
    # expired sooner would let that retry create, and charge, a second upload.
    idempotency_ttl_seconds: int = Field(
        default=IDEMPOTENCY_MIN_TTL_SECONDS, ge=IDEMPOTENCY_MIN_TTL_SECONDS
    )

    # None means "default for the environment": true outside production.
    allow_synthetic_recipe: bool | None = None

    public_delivery_rate_limit_per_minute: int = Field(default=60, ge=1)
    # Only enable when the API sits behind a proxy that overwrites X-Forwarded-For.
    public_delivery_trust_forwarded_for: bool = False

    # CORS is off by default; the web app proxies requests same-origin.
    cors_allowed_origins: list[str] = Field(default_factory=list)

    # Optional AI enrichment (README section 16), off unless AI_PROVIDER is set. The
    # key comes from OPENCODE_API_KEY or the file OPENCODE_API_KEY_FILE names; it is
    # read when used and never logged, returned or passed to a subprocess.
    ai_provider: Literal["opencode"] | None = None
    # OpenCode Go (the subscription); pay-as-you-go balance is .../inference/openai/v1.
    ai_base_url: str = "https://opencode.ai/inference/go/v1"
    ai_model: str = "deepseek-v4.1-flash"
    opencode_api_key: SecretStr | None = None
    opencode_api_key_file: str | None = None
    ai_max_output_tokens: int = Field(default=2048, ge=64, le=8192)
    ai_timeout_seconds: float = Field(default=60, gt=0, le=300)
    ai_max_concurrency: int = Field(default=2, ge=1, le=16)
    ai_requests_per_minute: int = Field(default=10, ge=1)

    @field_validator("ai_provider", "opencode_api_key", "opencode_api_key_file", mode="before")
    @classmethod
    def _blank_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def video_max_bytes(self) -> int:
        """Per video: its own limit, else the project total, never above one copy."""
        limit = self.upload_max_video_bytes or self.upload_max_total_bytes
        return min(limit, S3_SINGLE_COPY_MAX_BYTES)

    @property
    def s3_browser_endpoint_url(self) -> str | None:
        """The endpoint presigned browser URLs use; None is the provider's default (AWS S3)."""
        return self.s3_public_endpoint_url or self.s3_endpoint_url

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
        if self.ai_provider is not None and urlparse(self.ai_base_url).scheme != "https":
            problems.append("AI_BASE_URL must use https in production")
        # Presigned upload and download URLs carry write capabilities and private scene
        # assets to the browser. The internal endpoint may stay plain HTTP on a private
        # network when a separate public endpoint is set.
        browser = self.s3_browser_endpoint_url
        if browser is not None and urlparse(browser).scheme != "https":
            source = "S3_PUBLIC_ENDPOINT_URL" if self.s3_public_endpoint_url else "S3_ENDPOINT_URL"
            problems.append(
                f"{source} signs browser URLs and must use https in production "
                "(set S3_PUBLIC_ENDPOINT_URL to an https endpoint)"
            )
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
