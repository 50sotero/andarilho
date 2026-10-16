"""Canonical JSON and hashing used for request hashes, recipes and manifests.

Canonical form: ``json.dumps(value, sort_keys=True, separators=(",", ":"),
ensure_ascii=False)`` encoded as UTF-8. The worker must use the same form when
it computes ``scene_revisions.manifest_sha256``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256_hex(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()
