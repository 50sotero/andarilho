"""Worker configuration from the environment (README section 14)."""

from __future__ import annotations

import os
import socket
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

LOCAL_SECRETS = frozenset({"local-development-only", "local-password", "example"})


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw == "":
        return default
    value = int(raw)
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


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
    cleanup_reconcile_horizon_seconds: int = 900

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
        )
        if settings.app_env == "production":
            secrets = {settings.s3_access_key_id, settings.s3_secret_access_key}
            if secrets & LOCAL_SECRETS or "local-password" in url:
                raise ValueError("local/example credentials are not allowed in production")
            if settings.local_store_dir is not None:
                raise ValueError("the local object store is not allowed in production")
        return settings

    def secrets(self) -> list[str]:
        """Values that must never appear in logs."""
        values = [self.s3_secret_access_key, self.s3_access_key_id]
        if "@" in self.database_url and ":" in self.database_url.split("@", 1)[0]:
            values.append(self.database_url.split("@", 1)[0].rsplit(":", 1)[-1])
        return [v for v in values if v and len(v) >= 6]
