"""Settings: production validation and environment defaults; migrations match models."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from pydantic import ValidationError

from andarilho_api.domain.recipes import RECIPES
from andarilho_api.settings import (
    CAPTURE_SET_MAX_INPUTS,
    GIB,
    RUN_MAX_VIDEOS,
    S3_SINGLE_COPY_MAX_BYTES,
    Settings,
)

PROD = {
    "app_env": "production",
    "database_url": "postgresql+psycopg://app:s3cr3t-value@db.internal:5432/andarilho",
    "auth_issuer": "https://abc.supabase.co/auth/v1",
    "auth_jwks_url": "https://abc.supabase.co/auth/v1/.well-known/jwks.json",
    "s3_bucket": "andarilho-prod",
    "s3_access_key_id": None,
    "s3_secret_access_key": None,
}


def test_development_defaults():
    settings = Settings(app_env="development")
    assert settings.synthetic_recipe_enabled is True
    assert settings.upload_part_size == 16 * 1024 * 1024
    assert settings.upload_max_photos == 300
    assert settings.upload_max_floor_plan_bytes == 20 * 1024 * 1024
    assert settings.cors_allowed_origins == []


def test_photo_limit_leaves_room_for_videos_in_a_capture_set():
    # The worker defaults to 8 videos when a recipe sets no maxVideos.
    assert all(r.params.get("maxVideos", 8) <= RUN_MAX_VIDEOS for r in RECIPES.values())
    most = CAPTURE_SET_MAX_INPUTS - RUN_MAX_VIDEOS
    assert Settings(app_env="development", upload_max_photos=most).upload_max_photos == most
    with pytest.raises(ValidationError):
        Settings(app_env="development", upload_max_photos=most + 1)


def test_no_single_file_may_exceed_one_storage_copy():
    # A project total above the copy limit still allows several files, not a bigger one.
    roomy = Settings(app_env="development", upload_max_total_bytes=10 * GIB)
    assert roomy.video_max_bytes == S3_SINGLE_COPY_MAX_BYTES
    for field in (
        "upload_max_video_bytes",
        "upload_max_photo_bytes",
        "upload_max_floor_plan_bytes",
    ):
        with pytest.raises(ValidationError):
            Settings(app_env="development", **{field: S3_SINGLE_COPY_MAX_BYTES + 1})


def test_production_accepts_real_values_and_forces_synthetic_off():
    settings = Settings(**{**PROD, "allow_synthetic_recipe": True})
    assert settings.synthetic_recipe_enabled is False


@pytest.mark.parametrize(
    "override",
    [
        {"database_url": "postgresql+psycopg://andarilho:local-password@postgres:5432/andarilho"},
        {"auth_issuer": "http://abc.supabase.co/auth/v1"},
        {"auth_jwks_url": "http://abc.supabase.co/jwks.json"},
        {"auth_issuer": "https://example.supabase.co/auth/v1"},
        {"s3_access_key_id": "local-development-only", "s3_secret_access_key": "x" * 40},
        {"s3_access_key_id": "AKIAREAL", "s3_secret_access_key": "minioadmin"},
        # Browser-facing storage URLs: the public endpoint, else the internal one.
        {"s3_public_endpoint_url": "http://files.andarilho.app"},
        {"s3_endpoint_url": "http://storage.internal:9000"},
    ],
)
def test_production_rejects_example_values(override):
    with pytest.raises(ValidationError) as excinfo:
        Settings(**{**PROD, **override})
    message = str(excinfo.value)
    assert "local-password" not in message and "minioadmin" not in message


def test_production_allows_plain_http_only_behind_an_https_public_endpoint():
    settings = Settings(
        **PROD,
        s3_endpoint_url="http://storage.internal:9000",
        s3_public_endpoint_url="https://files.andarilho.app",
    )
    assert settings.s3_browser_endpoint_url == "https://files.andarilho.app"
    with pytest.raises(ValidationError, match="S3_ENDPOINT_URL signs browser URLs"):
        Settings(**PROD, s3_endpoint_url="http://storage.internal:9000")


def test_env_names(monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("UPLOAD_MAX_TOTAL_BYTES", "1234")
    monkeypatch.setenv("DELIVERY_URL_TTL_SECONDS", "120")
    monkeypatch.setenv("ALLOW_SYNTHETIC_RECIPE", "false")
    settings = Settings()
    assert settings.upload_max_total_bytes == 1234
    assert settings.delivery_url_ttl_seconds == 120
    assert settings.synthetic_recipe_enabled is False


def test_alembic_migrations_match_models(database_url):
    config = Config(str(Path(__file__).resolve().parents[1] / "alembic.ini"))
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = database_url
    try:
        command.check(config)
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
