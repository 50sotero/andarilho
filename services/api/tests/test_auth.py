"""Token verification: signature, issuer, audience, expiry, algorithms, key rotation."""

from __future__ import annotations

import time

import jwt
import pytest
from conftest import AUDIENCE, ISSUER
from cryptography.hazmat.primitives.asymmetric import rsa

from andarilho_api.auth import JwksCache
from andarilho_api.errors import ApiError


def _get(client, token: str):
    return client.get("/v1/projects", headers={"Authorization": f"Bearer {token}"})


def _assert_401(response):
    assert response.status_code == 401
    body = response.json()
    assert body["error"]["code"] == "UNAUTHENTICATED"
    assert response.headers["WWW-Authenticate"] == "Bearer"


def test_valid_rs256_and_es256_tokens(client, keys):
    assert _get(client, keys.token("user-a")).status_code == 200
    assert _get(client, keys.token("user-a", alg="ES256")).status_code == 200


def test_missing_and_malformed_headers(client):
    _assert_401(client.get("/v1/projects"))
    _assert_401(client.get("/v1/projects", headers={"Authorization": "Basic abc"}))
    _assert_401(client.get("/v1/projects", headers={"Authorization": "Bearer not-a-jwt"}))


def test_bad_signature(client, keys):
    other = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    _assert_401(_get(client, keys.token("user-a", key=other)))


def test_wrong_issuer_and_audience(client, keys):
    _assert_401(_get(client, keys.token(iss="https://evil.example/auth/v1")))
    _assert_401(_get(client, keys.token(aud="someone-else")))


def test_expired(client, keys):
    now = int(time.time())
    response = _get(client, keys.token(iat=now - 7200, exp=now - 3600))
    _assert_401(response)
    assert "expired" in response.json()["error"]["message"]


def test_missing_sub_or_exp(client, keys):
    _assert_401(_get(client, keys.token(sub=None)))
    _assert_401(_get(client, keys.token(exp=None)))


def test_alg_none_rejected(client, keys):
    payload = {"sub": "user-a", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 600}
    token = jwt.encode(payload, None, algorithm="none", headers={"kid": keys.rsa_kid})
    _assert_401(_get(client, token))


def test_hs256_rejected_even_with_public_key_material(client, keys):
    payload = {"sub": "user-a", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 600}
    token = jwt.encode(
        payload,
        "a-shared-secret-of-sufficient-length!",
        algorithm="HS256",
        headers={"kid": keys.rsa_kid},
    )
    _assert_401(_get(client, token))


def test_alg_must_match_key_type(client, keys):
    # An ES256 header pointing at the RSA kid must not verify.
    token = keys.token(alg="ES256", kid=keys.rsa_kid)
    _assert_401(_get(client, token))


def test_symmetric_jwks_entries_ignored(client, keys):
    keys.extra = [{"kty": "oct", "kid": "hmac", "k": "c2VjcmV0", "alg": "HS256"}]
    payload = {"sub": "user-a", "iss": ISSUER, "aud": AUDIENCE, "exp": int(time.time()) + 600}
    token = jwt.encode(
        payload, "secret-secret-secret-secret-secret!", algorithm="HS256", headers={"kid": "hmac"}
    )
    _assert_401(_get(client, token))


def test_unknown_kid_refreshes_once_then_rate_limited(client, keys):
    assert _get(client, keys.token()).status_code == 200
    assert keys.fetches == 1
    # Rotate in a new key: the first request with the new kid triggers a refresh... but
    # only after the minimum refresh interval. Simulate by moving the cache clock.
    cache = client.app.state.token_verifier.jwks
    rotated = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    new_jwk = jwt.algorithms.RSAAlgorithm.to_jwk(rotated.public_key(), as_dict=True)
    new_jwk.update(kid="rotated", alg="RS256", use="sig")
    keys.extra = [new_jwk]
    token = keys.token(kid="rotated", key=rotated)

    # Within 30 s of the last fetch: no refresh, rejected.
    _assert_401(_get(client, token))
    assert keys.fetches == 1

    base = time.monotonic()
    cache._clock = lambda: base + 31
    assert _get(client, token).status_code == 200
    assert keys.fetches == 2
    # Garbage kids do not trigger more fetches inside the interval.
    _assert_401(_get(client, keys.token(kid="random-kid")))
    _assert_401(_get(client, keys.token(kid="random-kid-2")))
    assert keys.fetches == 2


def test_jwks_cache_ttl_and_outage():
    calls = {"n": 0}
    now = {"t": 0.0}

    def fetch(url):
        calls["n"] += 1
        if calls["n"] > 1:
            raise ValueError("down")
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        jwk = jwt.algorithms.RSAAlgorithm.to_jwk(key.public_key(), as_dict=True)
        jwk.update(kid="k1", alg="RS256")
        return {"keys": [jwk]}

    cache = JwksCache(
        "https://x", fetcher=fetch, ttl_seconds=60, min_refresh_seconds=30, clock=lambda: now["t"]
    )
    assert cache.get("k1").key_id == "k1"
    now["t"] = 120  # TTL expired, provider down: stale known key still served
    assert cache.get("k1").key_id == "k1"
    now["t"] = 200
    with pytest.raises(ApiError) as excinfo:
        cache.get("unknown")
    assert excinfo.value.status_code == 503
