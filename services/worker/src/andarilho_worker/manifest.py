"""Persistent scene manifest construction (README section 10)."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from andarilho_worker.contracts import SCHEMA_VERSION
from andarilho_worker.geometry.layout import layout_bounds


def scene_prefix(project_id: str, scene_revision_id: str) -> str:
    return f"tenants/{project_id}/scenes/{scene_revision_id}/"


def asset_entry(
    key: str, data: bytes | Path, fmt: str, *, sha256: str | None = None
) -> dict[str, Any]:
    """Manifest entry for an asset given as bytes or as a local file (hashed in a stream)."""
    if isinstance(data, Path):
        size = data.stat().st_size
        if sha256 is None:
            digest = hashlib.sha256()
            with data.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    digest.update(chunk)
            sha256 = digest.hexdigest()
    else:
        size, sha256 = len(data), sha256 or hashlib.sha256(data).hexdigest()
    return {"key": key, "format": fmt, "bytes": size, "sha256": sha256}


def _clean_number(v: float) -> float | int:
    """Render integral floats as ints so JSON output is stable and compact."""
    f = float(v)
    return int(f) if f.is_integer() else f


def navigation_for(geometry_source: str, layout_revision: int) -> dict[str, Any]:
    """Navigation for a packaged scene: walkable at once, whatever its geometry source.

    The collision mesh is built from the same layout as the model, so an automatic proposal
    is walkable without review. Review is optional editing, and approval only publishes. The
    source is kept in the provenance, not here.
    """
    return {
        "mode": "approved_collision",
        "layoutRevision": layout_revision,
        "approved": True,
        "playerRadius": 0.25,
        "playerHeight": 1.7,
        "eyeHeight": 1.6,
    }


def build_manifest(
    *,
    scene_id: str,
    scene_revision_id: str,
    revision: int,
    run_id: str,
    layout: Mapping[str, Any],
    layout_revision: int,
    render: Mapping[str, Any],
    collision: Mapping[str, Any],
    poster: Mapping[str, Any] | None,
    scale_status: str,
    geometry_source: str,
    recipe_digest: str,
    capture_set_id: str,
    quality: Mapping[str, Any],
) -> dict[str, Any]:
    spawn = layout["spawn"]
    assets: dict[str, Any] = {"render": dict(render), "collision": dict(collision)}
    if poster is not None:
        assets["poster"] = dict(poster)
    bounds = layout_bounds(layout)
    return {
        "schemaVersion": SCHEMA_VERSION,
        "sceneId": scene_id,
        "sceneRevisionId": scene_revision_id,
        "revision": revision,
        "runId": run_id,
        "renderMode": "mesh",
        "units": "m",
        "upAxis": "+Y",
        "cameraForward": "-Z",
        "scaleStatus": scale_status,
        "geometrySpace": "world",
        "bounds": {
            "min": [_clean_number(v) for v in bounds["min"]],
            "max": [_clean_number(v) for v in bounds["max"]],
        },
        "spawn": {
            "feet": [_clean_number(v) for v in spawn["feet"]],
            "yawRadians": spawn["yawRadians"],
            "floorId": spawn["floorId"],
        },
        "floors": [
            {"id": f["id"], "elevation": _clean_number(f["elevation"])} for f in layout["floors"]
        ],
        "assets": assets,
        "navigation": navigation_for(geometry_source, layout_revision),
        "quality": {
            "registeredViews": int(quality.get("registeredViews", 0)),
            "selectedViews": int(quality.get("selectedViews", 0)),
            "disconnectedComponents": int(quality.get("disconnectedComponents", 0)),
            "warnings": list(quality.get("warnings", []))[:100],
        },
        "provenance": {
            "recipeDigest": recipe_digest,
            "captureSetId": capture_set_id,
            "geometrySource": geometry_source,
        },
    }
