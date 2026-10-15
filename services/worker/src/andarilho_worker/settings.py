"""Worker configuration from the environment (README section 14)."""

from __future__ import annotations

import math
import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

LOCAL_SECRETS = frozenset({"local-development-only", "local-password", "example"})
# The OpenAI-compatible gateway the API's AI advice also uses (README section 16): OpenCode
# Go, the subscription. Pay-as-you-go balance is https://opencode.ai/inference/openai/v1.
DEFAULT_AI_BASE_URL = "https://opencode.ai/inference/go/v1"
DEFAULT_AI_MODEL = "deepseek-v4.1-flash"
DEFAULT_AI_SCENE_TIMEOUT_SECONDS = 180.0
DEFAULT_AI_SCENE_MAX_OUTPUT_TOKENS = 8192
TRUE_WORDS = frozenset({"1", "true", "yes"})
FALSE_WORDS = frozenset({"", "0", "false", "no", "off"})


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _flag(env: Mapping[str, str], name: str) -> bool:
    word = (env.get(name) or "").strip().lower()
    if word in TRUE_WORDS:
        return True
    if word in FALSE_WORDS:
        return False
    raise ValueError(f"{name} must be true or false")


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a positive number of seconds") from None
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number of seconds")
    return value


def _upload_horizon(env: Mapping[str, str]) -> int:
    """The reconcile horizon from the upload variables the API signs part URLs with."""
    part_size = _int(env, "UPLOAD_PART_SIZE", DEFAULT_UPLOAD_PART_SIZE)
    if not 5 * 1024**2 <= part_size <= 5 * 1024**3:  # the API's own range
        raise ValueError("UPLOAD_PART_SIZE must be between 5 MiB and 5 GiB")
    ttl = _int(env, "UPLOAD_PART_URL_TTL_SECONDS", DEFAULT_UPLOAD_PART_URL_TTL_SECONDS)
    if not 60 <= ttl <= 3600:
        raise ValueError("UPLOAD_PART_URL_TTL_SECONDS must be between 60 and 3600")
    rate = _int(env, "UPLOAD_MIN_BYTES_PER_SECOND", DEFAULT_UPLOAD_MIN_BYTES_PER_SECOND)
    if rate < 1000:
        raise ValueError("UPLOAD_MIN_BYTES_PER_SECOND must be at least 1000")
    return reconcile_horizon(part_size, ttl, rate)


def _text(env: Mapping[str, str], name: str) -> str | None:
    """A trimmed variable; blank or unset means None."""
    return (env.get(name) or "").strip() or None


def reconcile_horizon(part_size: int, part_url_ttl: int, min_bytes_per_second: int) -> int:
    """How long after an abort a part upload already under way can still land.

    Storage cannot limit how long a part upload takes, so this is the bound the app
    supports: a part upload must start within its URL's lifetime and move at least
    ``min_bytes_per_second``. No part URL is signed once an upload is removed, rejected
    or expired, so a check this long after the abort sees every part such an upload can
    still add. A part that lands later anyway is still found and aborted by the
    multipart checks, which then charge its upload's quota again until it is gone.
    """
    return part_url_ttl + math.ceil(part_size / min_bytes_per_second)


# The API's defaults (UPLOAD_PART_SIZE, UPLOAD_PART_URL_TTL_SECONDS) and the slowest part
# upload supported, 16,000 bytes/s (128 kbit/s): 900 + 1049 seconds.
DEFAULT_UPLOAD_PART_SIZE = 16 * 1024**2
DEFAULT_UPLOAD_PART_URL_TTL_SECONDS = 900
DEFAULT_UPLOAD_MIN_BYTES_PER_SECOND = 16_000
DEFAULT_RECONCILE_HORIZON_SECONDS = reconcile_horizon(
    DEFAULT_UPLOAD_PART_SIZE,
    DEFAULT_UPLOAD_PART_URL_TTL_SECONDS,
    DEFAULT_UPLOAD_MIN_BYTES_PER_SECOND,
)


@dataclass(frozen=True)
class Settings:
    database_url: str
    worker_capability: str = "cpu"
    worker_max_concurrency: int = 1
    worker_scratch_dir: Path = Path("/scratch")
    worker_id: str = field(default_factory=lambda: f"{socket.gethostname()}-{os.getpid()}")
    run_max_seconds: int = 3600
    upload_max_total_bytes: int = 2 * 1024**3
    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_bucket: str = "andarilho-local"
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    local_store_dir: Path | None = None
    app_env: str = "development"
    lease_seconds: int = 90
    heartbeat_seconds: float = 20.0
    poll_seconds: float = 2.0
    reaper_seconds: float = 15.0
    max_scratch_bytes: int = 50 * 1024**3
    retry_base_seconds: float = 30.0
    retry_max_seconds: float = 900.0
    # The write-grant horizon (reconcile_horizon). from_env derives it from the API's own
    # upload variables, so the two services cannot disagree on it.
    cleanup_reconcile_horizon_seconds: int = DEFAULT_RECONCILE_HORIZON_SECONDS
    # Scene analysis (scene_analysis stage), off unless AI_SCENE_ANALYSIS is set. The key is
    # read from the file OPENCODE_API_KEY_FILE names at each call; it is never logged. The
    # timeout and the reply budget have their own variables (AI_SCENE_TIMEOUT_SECONDS and
    # AI_SCENE_MAX_OUTPUT_TOKENS): one call carries a dozen images and answers with every
    # object's parts, far more than the API's short AI advice.
    ai_scene_analysis: bool = False
    ai_base_url: str = DEFAULT_AI_BASE_URL
    ai_model: str = DEFAULT_AI_MODEL
    opencode_api_key_file: str | None = None
    ai_timeout_seconds: float = DEFAULT_AI_SCENE_TIMEOUT_SECONDS
    ai_max_output_tokens: int = DEFAULT_AI_SCENE_MAX_OUTPUT_TOKENS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        env = os.environ if env is None else env
        url = env.get("DATABASE_URL")
        if not url:
            raise ValueError("DATABASE_URL is required")
        capability = env.get("WORKER_CAPABILITY", "cpu")
        if capability not in ("cpu", "gpu"):
            raise ValueError("WORKER_CAPABILITY must be cpu or gpu")
        concurrency = _int(env, "WORKER_MAX_CONCURRENCY", 1)
        if concurrency < 1:
            raise ValueError("WORKER_MAX_CONCURRENCY must be at least 1")
        local_dir = env.get("WORKER_LOCAL_STORE_DIR")
        settings = cls(
            database_url=url,
            worker_capability=capability,
            worker_max_concurrency=concurrency,
            worker_scratch_dir=Path(env.get("WORKER_SCRATCH_DIR", "/scratch")),
            worker_id=env.get("WORKER_ID") or f"{socket.gethostname()}-{os.getpid()}",
            run_max_seconds=_int(env, "RUN_MAX_SECONDS", 3600),
            upload_max_total_bytes=_int(env, "UPLOAD_MAX_TOTAL_BYTES", 2 * 1024**3),
            s3_endpoint_url=env.get("S3_ENDPOINT_URL") or None,
            s3_region=env.get("S3_REGION", "us-east-1"),
            s3_bucket=env.get("S3_BUCKET", "andarilho-local"),
            s3_access_key_id=env.get("S3_ACCESS_KEY_ID") or None,
            s3_secret_access_key=env.get("S3_SECRET_ACCESS_KEY") or None,
            local_store_dir=Path(local_dir) if local_dir else None,
            app_env=env.get("APP_ENV", "development"),
            max_scratch_bytes=_int(env, "WORKER_MAX_SCRATCH_BYTES", 50 * 1024**3),
            cleanup_reconcile_horizon_seconds=_upload_horizon(env),
            ai_scene_analysis=_flag(env, "AI_SCENE_ANALYSIS"),
            ai_base_url=_text(env, "AI_BASE_URL") or DEFAULT_AI_BASE_URL,
            ai_model=_text(env, "AI_MODEL") or DEFAULT_AI_MODEL,
            opencode_api_key_file=_text(env, "OPENCODE_API_KEY_FILE"),
            ai_timeout_seconds=_seconds(
                env, "AI_SCENE_TIMEOUT_SECONDS", DEFAULT_AI_SCENE_TIMEOUT_SECONDS
            ),
            ai_max_output_tokens=_int(
                env, "AI_SCENE_MAX_OUTPUT_TOKENS", DEFAULT_AI_SCENE_MAX_OUTPUT_TOKENS
            ),
        )
        if settings.ai_max_output_tokens < 1:
            raise ValueError("AI_SCENE_MAX_OUTPUT_TOKENS must be at least 1")
        if settings.app_env == "production":
            secrets = {settings.s3_access_key_id, settings.s3_secret_access_key}
            if secrets & LOCAL_SECRETS or "local-password" in url:
                raise ValueError("local/example credentials are not allowed in production")
            if settings.local_store_dir is not None:
                raise ValueError("the local object store is not allowed in production")
            if settings.ai_scene_analysis and urlparse(settings.ai_base_url).scheme != "https":
                raise ValueError(
                    "AI_BASE_URL must use https in production when AI_SCENE_ANALYSIS is on"
                )
        return settings

    def secrets(self) -> list[str]:
        """Values that must never appear in logs."""
        values = [self.s3_secret_access_key, self.s3_access_key_id]
        if "@" in self.database_url and ":" in self.database_url.split("@", 1)[0]:
            values.append(self.database_url.split("@", 1)[0].rsplit(":", 1)[-1])
        return [v for v in values if v and len(v) >= 6]
