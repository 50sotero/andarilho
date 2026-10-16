"""Scene revisions: delivery envelopes and share tokens (README section 10)."""

from __future__ import annotations

import hashlib
import secrets
from datetime import timedelta
from typing import Any

from pydantic import ValidationError

from andarilho_api.deps import utcnow
from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.errors import invalid_state, scene_outdated
from andarilho_api.models import SceneRevision
from andarilho_api.schemas.scene import DeliveryEnvelope, SceneManifest
from andarilho_api.serializers import iso
from andarilho_api.settings import Settings
from andarilho_api.storage import Storage, scene_prefix

DELIVERY_ASSETS = ("render", "collision", "poster")


def new_share_token() -> tuple[str, bytes]:
    """Return ``(raw_token, sha256_digest)``. Only the digest is stored."""
    token = secrets.token_urlsafe(32)
    return token, hash_token(token)


def hash_token(token: str) -> bytes:
    return hashlib.sha256(token.encode("utf-8")).digest()


def manifest_sha256(revision: SceneRevision) -> str:
    """SHA-256 of the canonical manifest JSON (see ``domain/canonical.py``)."""
    if revision.manifest_sha256:
        return revision.manifest_sha256
    return sha256_hex(revision.manifest)


def validated_manifest(revision: SceneRevision) -> SceneManifest:
    if revision.manifest is None:
        raise invalid_state("This scene revision has no packaged manifest yet.")
    # Checked before the current schema so an old document gets a clear reason, not a
    # list of validation errors the owner cannot act on.
    if revision.manifest.get("schemaVersion") == "1.0":
        raise scene_outdated()
    try:
        manifest = SceneManifest.model_validate(revision.manifest)
    except ValidationError:
        raise invalid_state("This scene revision's manifest is invalid.") from None
    if manifest.scene_revision_id != str(revision.id):
        raise invalid_state("This scene revision's manifest is invalid.")
    prefix = scene_prefix(revision.project_id, revision.id)
    for name in DELIVERY_ASSETS:
        asset = getattr(manifest.assets, name)
        if asset is not None and not asset.key.startswith(prefix):
            raise invalid_state("This scene revision's manifest is invalid.")
    return manifest


def build_envelope(storage: Storage, settings: Settings, revision: SceneRevision) -> dict[str, Any]:
    """Derivative-only delivery envelope with short-lived URLs for one immutable revision."""
    manifest = validated_manifest(revision)
    ttl = settings.delivery_url_ttl_seconds
    expires_at = iso(utcnow() + timedelta(seconds=ttl))
    asset_urls: dict[str, dict[str, Any]] = {}
    for name in DELIVERY_ASSETS:
        asset = getattr(manifest.assets, name)
        if asset is None:
            continue
        asset_urls[name] = {
            "url": storage.presign_get(asset.key, ttl),
            "sha256": asset.sha256,
            "bytes": asset.bytes,
            "format": asset.format,
        }
    envelope = {
        "sceneRevisionId": str(revision.id),
        "manifestSha256": manifest_sha256(revision),
        "expiresAt": expires_at,
        # The stored manifest verbatim so clients can verify manifestSha256.
        "manifest": revision.manifest,
        "assetUrls": asset_urls,
    }
    DeliveryEnvelope.model_validate(envelope)
    return envelope
