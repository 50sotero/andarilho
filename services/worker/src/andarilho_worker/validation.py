"""Scene manifest validation: JSON Schema plus semantic and asset checks.

Used by the ``scene_validation`` stage and by ``scripts/check-scene.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from andarilho_worker.contracts import SCHEMA_VERSION, scene_validator
from andarilho_worker.geometry.collide import capsule_clearance
from andarilho_worker.geometry.glb import (
    MAX_GLB_BYTES,
    GlbError,
    inspect_textured_glb,
    read_glb,
)
from andarilho_worker.geometry.layout import (
    Finding,
    floor_supports,
    furniture_footprint,
    validate_layout,
)
from andarilho_worker.geometry.polygon import point_in_polygon

BOUNDS_TOLERANCE = 1e-3
SPAWN_FLOOR_TOLERANCE = 0.1
# Leading bytes read from a poster to check its image magic.
MAX_HEADER_BYTES = 64 * 1024
# The textured model: a layout has hundreds of triangles, so these leave wide headroom
# while keeping a phone's GPU memory bounded (README section 11).
MAX_RENDER_TRIANGLES = 200_000
MAX_RENDER_TEXTURE_SIDE = 4096
# Rendered surfaces follow the layout; the margin absorbs float32 rounding.
RENDER_BOUNDS_MARGIN = 0.05


def canonical_json(value: Any) -> bytes:
    """Canonical JSON shared with the API: sorted keys, compact, UTF-8 (not ASCII-escaped)."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(manifest)).hexdigest()


def loads_strict(data: bytes | str) -> Any:
    """json.loads that rejects NaN/Infinity literals."""

    def reject(token: str) -> Any:
        raise ValueError(f"non-finite JSON literal {token}")

    return json.loads(data, parse_constant=reject)


def _non_finite(value: Any, path: str = "") -> str | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return path or "<root>"
    if isinstance(value, Mapping):
        for k, v in value.items():
            found = _non_finite(v, f"{path}/{k}")
            if found:
                return found
    if isinstance(value, list | tuple):
        for i, v in enumerate(value):
            found = _non_finite(v, f"{path}/{i}")
            if found:
                return found
    return None


def validate_manifest(
    manifest: Mapping[str, Any], *, project_id: str | None = None
) -> list[Finding]:
    """Schema + semantic checks that need no asset bytes."""
    bad = _non_finite(manifest)
    if bad:
        return [Finding("NON_FINITE", f"non-finite number at {bad}")]
    if manifest.get("schemaVersion") != SCHEMA_VERSION:
        return [Finding("SCHEMA_VERSION", "unknown schemaVersion")]
    errors = sorted(scene_validator().iter_errors(manifest), key=lambda e: list(e.path))
    if errors:
        return [
            Finding("SCHEMA", f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}")
            for e in errors[:20]
        ]
    findings: list[Finding] = []
    bounds = manifest["bounds"]
    if any(lo >= hi for lo, hi in zip(bounds["min"], bounds["max"], strict=True)):
        findings.append(Finding("BOUNDS", "bounds.min must be below bounds.max on every axis"))
    floor_ids = [f["id"] for f in manifest["floors"]]
    if len(set(floor_ids)) != len(floor_ids):
        findings.append(Finding("FLOORS", "floor ids must be unique"))
    spawn = manifest["spawn"]
    floors = {f["id"]: f for f in manifest["floors"]}
    if spawn["floorId"] not in floors:
        findings.append(Finding("SPAWN", "spawn.floorId must reference a declared floor"))
    elif abs(spawn["feet"][1] - floors[spawn["floorId"]]["elevation"]) > SPAWN_FLOOR_TOLERANCE:
        findings.append(Finding("SPAWN", "spawn feet must be at its floor elevation"))
    if not all(
        lo <= p <= hi for lo, p, hi in zip(bounds["min"], spawn["feet"], bounds["max"], strict=True)
    ):
        findings.append(Finding("SPAWN", "spawn.feet must lie within bounds"))
    rev = manifest["sceneRevisionId"]
    projects = set()
    for name, asset in manifest["assets"].items():
        if asset is None:
            continue
        parts = asset["key"].split("/")
        if len(parts) != 5 or parts[0] != "tenants" or parts[2] != "scenes":
            findings.append(Finding("ASSET_PREFIX", f"{name} key has an unexpected layout"))
            continue
        if parts[3] != rev:
            findings.append(Finding("ASSET_PREFIX", f"{name} key belongs to another revision"))
        projects.add(parts[1])
    if len(projects) > 1:
        findings.append(Finding("ASSET_PREFIX", "asset keys must share one project prefix"))
    if project_id is not None and projects and projects != {project_id}:
        findings.append(Finding("ASSET_PREFIX", "asset keys are outside this project's prefix"))
    nav = manifest["navigation"]
    radius, height = nav.get("playerRadius", 0.25), nav.get("playerHeight", 1.7)
    if height <= 2 * radius or nav.get("eyeHeight", 1.6) > height:
        findings.append(Finding("NAVIGATION", "player capsule dimensions are inconsistent"))
    if nav["mode"] == "approved_collision" and not nav["approved"]:
        findings.append(Finding("NAVIGATION", "approved_collision requires approved=true"))
    return findings


def check_collision_glb(
    data: bytes, manifest: Mapping[str, Any], layout: Mapping[str, Any] | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    try:
        _doc, nodes = read_glb(data)
    except (GlbError, KeyError, IndexError, ValueError) as exc:
        return [Finding("COLLISION_GLB", f"collision GLB rejected: {exc}")]
    if not nodes:
        return [Finding("COLLISION_GLB", "collision GLB has no geometry")]
    lo = np.array(manifest["bounds"]["min"]) - BOUNDS_TOLERANCE
    hi = np.array(manifest["bounds"]["max"]) + BOUNDS_TOLERANCE
    for node in nodes:
        if not np.all(np.isfinite(node.positions)):
            findings.append(Finding("COLLISION_GLB", f"node {node.name} has non-finite vertices"))
        elif np.any(node.positions < lo) or np.any(node.positions > hi):
            findings.append(Finding("COLLISION_GLB", f"node {node.name} lies outside bounds"))
    if findings:
        return findings
    nav = manifest["navigation"]
    radius, height = nav.get("playerRadius", 0.25), nav.get("playerHeight", 1.7)
    feet = manifest["spawn"]["feet"]
    clearance = capsule_clearance(nodes, feet, radius=radius, height=height)
    if clearance < -1e-6:
        findings.append(Finding("SPAWN_BLOCKED", "spawn capsule intersects collision geometry"))
    if not _floor_under(nodes, feet):
        findings.append(Finding("SPAWN_NOT_SUPPORTED", "no collision floor under the spawn point"))
    if layout is not None:
        floor = next((f for f in layout["floors"] if f["id"] == manifest["spawn"]["floorId"]), None)
        if floor is None or not floor_supports(floor, (feet[0], feet[2])):
            findings.append(Finding("SPAWN_NOT_SUPPORTED", "spawn is outside the approved floor"))
        for obstacle in layout.get("obstacles", []):
            # Only obstacles on the spawn floor can block it; others are above or below.
            if obstacle["floorId"] != manifest["spawn"]["floorId"]:
                continue
            if point_in_polygon((feet[0], feet[2]), obstacle["footprint"]):
                findings.append(Finding("SPAWN_BLOCKED", "spawn is inside an obstacle"))
        for item in layout.get("furniture", []):
            # Solid furniture blocks the spawn like an obstacle; a piece that is not solid
            # is not in the collision mesh and does not.
            solid_here = item.get("solid", True) and item["floorId"] == manifest["spawn"]["floorId"]
            if solid_here and point_in_polygon((feet[0], feet[2]), furniture_footprint(item)):
                findings.append(Finding("SPAWN_BLOCKED", "spawn is inside a piece of furniture"))
    return findings


def _floor_under(nodes, feet: Sequence[float], tolerance: float = 0.05) -> bool:
    x, y, z = feet
    for node in nodes:
        if node.kind != "floor":
            continue
        for tri in node.positions[node.triangles]:
            ys = tri[:, 1]
            if abs(float(ys.mean()) - y) > tolerance:
                continue
            if point_in_polygon((x, z), [(float(p[0]), float(p[2])) for p in tri]):
                return True
    return False


def check_render_asset(data: bytes, fmt: str, manifest: Mapping[str, Any]) -> list[Finding]:
    """Check the textured model: a self-contained, unlit GLB inside the scene's bounds."""
    if fmt != "glb":
        return [Finding("RENDER_ASSET", "unknown render format")]
    try:
        summary = inspect_textured_glb(data)
    except (GlbError, KeyError, IndexError, ValueError) as exc:
        return [Finding("RENDER_ASSET", f"render model rejected: {exc}")]
    findings: list[Finding] = []
    if not summary["unlit"]:
        findings.append(Finding("RENDER_ASSET", "render model materials must be unlit"))
    if summary["triangles"] > MAX_RENDER_TRIANGLES:
        findings.append(Finding("RENDER_ASSET", "render model has too many triangles"))
    if any(max(size) > MAX_RENDER_TEXTURE_SIDE for size in summary["images"]):
        findings.append(Finding("RENDER_ASSET", "render model texture is too large"))
    lo = np.array(manifest["bounds"]["min"]) - RENDER_BOUNDS_MARGIN
    hi = np.array(manifest["bounds"]["max"]) + RENDER_BOUNDS_MARGIN
    low, high = np.array(summary["bounds"]["min"]), np.array(summary["bounds"]["max"])
    if np.any(low < lo) or np.any(high > hi):
        findings.append(Finding("RENDER_ASSET", "render model lies outside the scene bounds"))
    return findings


POSTER_MAGIC = {"png": (0, b"\x89PNG"), "jpeg": (0, b"\xff\xd8\xff"), "webp": (8, b"WEBP")}


def check_poster_asset(data: bytes, fmt: str) -> list[Finding]:
    """Check that a poster's leading bytes match its declared image format."""
    offset, magic = POSTER_MAGIC[fmt]
    if data[offset : offset + len(magic)] != magic or (fmt == "webp" and data[:4] != b"RIFF"):
        return [Finding("POSTER_ASSET", f"poster is not a {fmt} image")]
    return []


def _measure(source: bytes | Path) -> tuple[int, str]:
    """Byte count and SHA-256 of an asset; a file is hashed in a stream."""
    if not isinstance(source, Path):
        return len(source), hashlib.sha256(source).hexdigest()
    digest = hashlib.sha256()
    with source.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return source.stat().st_size, digest.hexdigest()


def _head(source: bytes | Path) -> bytes:
    if not isinstance(source, Path):
        return source[:MAX_HEADER_BYTES]
    with source.open("rb") as fh:
        return fh.read(MAX_HEADER_BYTES)


def validate_scene(
    manifest: Mapping[str, Any],
    fetch: Callable[[str], bytes | Path],
    *,
    project_id: str | None = None,
    layout: Mapping[str, Any] | None = None,
) -> list[Finding]:
    """Full validation: manifest semantics, asset hashes/bytes, GLB and spawn checks.

    ``fetch`` returns an asset's bytes or a local file holding them; files are
    hashed in a stream. The render model is parsed whole, so it is read only when
    it is within ``MAX_GLB_BYTES``.
    """
    findings = validate_manifest(manifest, project_id=project_id)
    if findings:
        return findings
    if layout is not None:
        findings += validate_layout(layout)
    sources: dict[str, bytes | Path] = {}
    sizes: dict[str, int] = {}
    for name, asset in manifest["assets"].items():
        if asset is None:
            continue
        try:
            source = fetch(asset["key"])
        except FileNotFoundError:
            findings.append(Finding("ASSET_MISSING", f"{name} asset is missing"))
            continue
        size, sha256 = _measure(source)
        if size != asset["bytes"]:
            findings.append(Finding("ASSET_BYTES", f"{name} byte count does not match"))
        if sha256 != asset["sha256"]:
            findings.append(Finding("ASSET_SHA256", f"{name} sha256 does not match"))
        sources[name] = source
        sizes[name] = size
    if "render" in sources:
        render = sources["render"]
        if sizes["render"] > MAX_GLB_BYTES:
            findings.append(Finding("RENDER_ASSET", "render model is too large"))
        else:
            data = render.read_bytes() if isinstance(render, Path) else render
            render_format = manifest["assets"]["render"]["format"]
            findings += check_render_asset(data, render_format, manifest)
    if "poster" in sources:
        poster_format = manifest["assets"]["poster"]["format"]
        findings += check_poster_asset(_head(sources["poster"]), poster_format)
    if "collision" in sources:
        collision = sources["collision"]  # generated, small: parsed in memory
        data = collision.read_bytes() if isinstance(collision, Path) else collision
        findings += check_collision_glb(data, manifest, layout)
    return findings
