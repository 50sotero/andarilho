"""Test fixtures: real PostgreSQL (migrated with Alembic), moto S3, test-only JWKS."""

from __future__ import annotations

import os
import re
import time
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import jwt
import psycopg
import pytest
from alembic import command
from alembic.config import Config
from cryptography.hazmat.primitives.asymmetric import ec, rsa
from fastapi.testclient import TestClient
from moto import mock_aws
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from andarilho_api.main import create_app
from andarilho_api.models import Base
from andarilho_api.settings import Settings

API_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TEST_DB = "postgresql+psycopg://andarilho:local-password@localhost:55432/andarilho_api_test"
# Only a private test database may be dropped: ``andarilho_<service>_test`` with an optional
# ``_<suffix>``. The dev database (``andarilho``) and look-alikes (``andarilho_testing``,
# ``prod_test_mirror``) do not match.
TEST_DATABASE_NAME = re.compile(r"andarilho(_[a-z0-9]+)*_test(_[a-z0-9]+)?")
ISSUER = "https://auth.test.invalid/auth/v1"
AUDIENCE = "authenticated"
BUCKET = "andarilho-test"


def _recreate_database(url: str) -> None:
    parsed = make_url(url)
    name = parsed.database
    assert name and TEST_DATABASE_NAME.fullmatch(name), "refusing to recreate a non-test database"
    admin = parsed.set(drivername="postgresql", database="postgres")
    conninfo = admin.render_as_string(hide_password=False)
    with psycopg.connect(conninfo, autocommit=True) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{name}"')


@pytest.fixture(scope="session")
def database_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DB)
    _recreate_database(url)
    config = Config(str(API_ROOT / "alembic.ini"))
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        command.upgrade(config, "head")
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous
    return url


@pytest.fixture(scope="session")
def engine(database_url: str):
    eng = create_engine(database_url)
    yield eng
    eng.dispose()


_TABLES = ", ".join(t.name for t in Base.metadata.sorted_tables)


@pytest.fixture(autouse=True)
def clean_db(engine) -> Iterator[None]:
    yield
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))


@pytest.fixture
def db(engine) -> Iterator[Session]:
    with Session(engine, expire_on_commit=False) as session:
        yield session


# -- signing keys (test-only) -----------------------------------------------


class Keys:
    def __init__(self) -> None:
        self.rsa = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        self.ec = ec.generate_private_key(ec.SECP256R1())
        self.rsa_kid = "test-rsa-1"
        self.ec_kid = "test-ec-1"
        self.extra: list[dict[str, Any]] = []
        self.fetches = 0

    def jwks(self) -> dict[str, Any]:
        rsa_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(self.rsa.public_key(), as_dict=True)
        ec_jwk = jwt.algorithms.ECAlgorithm.to_jwk(self.ec.public_key(), as_dict=True)
        rsa_jwk.update(kid=self.rsa_kid, alg="RS256", use="sig")
        ec_jwk.update(kid=self.ec_kid, alg="ES256", use="sig")
        return {"keys": [rsa_jwk, ec_jwk, *self.extra]}

    def fetch(self, url: str) -> dict[str, Any]:
        self.fetches += 1
        return self.jwks()

    def token(
        self,
        sub: str = "user-a",
        *,
        alg: str = "RS256",
        kid: str | None = None,
        key: Any = None,
        **claims: Any,
    ) -> str:
        now = int(time.time())
        payload = {"sub": sub, "iss": ISSUER, "aud": AUDIENCE, "iat": now, "exp": now + 600}
        payload.update(claims)
        payload = {k: v for k, v in payload.items() if v is not None}
        if key is None:
            key = self.ec if alg == "ES256" else self.rsa
        kid = kid or (self.ec_kid if alg == "ES256" else self.rsa_kid)
        return jwt.encode(payload, key, algorithm=alg, headers={"kid": kid})


@pytest.fixture(scope="session")
def _session_keys() -> Keys:
    return Keys()


@pytest.fixture
def keys(_session_keys: Keys) -> Keys:
    _session_keys.extra = []
    _session_keys.fetches = 0
    return _session_keys


# -- app ------------------------------------------------------------------


@pytest.fixture
def settings(database_url: str) -> Settings:
    return Settings(
        app_env="test",
        database_url=database_url,
        auth_issuer=ISSUER,
        auth_audience=AUDIENCE,
        auth_jwks_url="https://auth.test.invalid/jwks.json",
        s3_endpoint_url=None,
        s3_public_endpoint_url="http://localhost:9000",
        s3_region="us-east-1",
        s3_bucket=BUCKET,
        s3_access_key_id="testing",
        s3_secret_access_key="testing",
    )


@pytest.fixture
def aws() -> Iterator[None]:
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        import boto3

        boto3.client("s3", region_name="us-east-1").create_bucket(Bucket=BUCKET)
        yield


@pytest.fixture
def app(aws, settings: Settings, keys: Keys):
    application = create_app(settings, jwks_fetcher=keys.fetch)
    yield application
    application.state.engine.dispose()


@pytest.fixture
def client(app) -> Iterator[TestClient]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth(keys: Keys):
    def headers(sub: str = "user-a", **extra: str) -> dict[str, str]:
        return {"Authorization": f"Bearer {keys.token(sub)}", **extra}

    return headers


@pytest.fixture
def s3(app):
    return app.state.storage.internal


def new_id() -> str:
    return str(uuid.uuid4())
