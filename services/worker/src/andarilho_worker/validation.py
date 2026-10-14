"""Scene manifest validation: JSON Schema plus semantic and asset checks.

Used by the ``scene_validation`` stage and by ``scripts/check-scene.py``.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from andarilho_worker.contracts import SCHEMA_VERSION, scene_validator
from andarilho_worker.geometry.collide import capsule_clearance
from andarilho_worker.geometry.glb import GlbError, read_glb
from andarilho_worker.geometry.layout import Finding, floor_supports, validate_layout
from andarilho_worker.geometry.ply import PlyError, parse_header
from andarilho_worker.geometry.polygon import point_in_polygon

BOUNDS_TOLERANCE = 1e-3
SPAWN_FLOOR_TOLERANCE = 0.1


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


def similarity_matrix_problem(m: Sequence[float]) -> str | None:
    """Accept only rotation + translation + positive uniform scale (column-major)."""
    if len(m) != 16:
        return "matrix must have 16 entries"
    if abs(m[3]) > 1e-9 or abs(m[7]) > 1e-9 or abs(m[11]) > 1e-9 or abs(m[15] - 1) > 1e-9:
        return "worldFromDecodedRender must be affine"
    cols = [np.array(m[0:3]), np.array(m[4:7]), np.array(m[8:11])]
    norms = [float(np.linalg.norm(c)) for c in cols]
    scale = norms[0]
    if scale <= 1e-12 or any(abs(n - scale) > 1e-6 * max(1.0, scale) for n in norms):
        return "worldFromDecodedRender must use a positive uniform scale"
    a, b, c = (col / scale for col in cols)
    if max(abs(a @ b), abs(a @ c), abs(b @ c)) > 1e-6:
        return "worldFromDecodedRender rotation must be orthogonal"
    if float(np.linalg.det(np.column_stack([a, b, c]))) <= 0:
        return "worldFromDecodedRender must not contain a reflection"
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
    problem = similarity_matrix_problem(manifest["worldFromDecodedRender"])
    if problem:
        findings.append(Finding("MATRIX", problem))
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


def check_render_asset(data: bytes, fmt: str) -> list[Finding]:
    if fmt == "ply":
        try:
            parse_header(data)
        except (PlyError, UnicodeDecodeError) as exc:
            return [Finding("RENDER_ASSET", f"render PLY rejected: {exc}")]
        return []
    if fmt in ("sog", "spz"):
        # Bundled SOG is a zip archive; SPZ is gzip-compressed.
        magic = data[:4]
        if fmt == "sog" and magic != b"PK\x03\x04":
            return [Finding("RENDER_ASSET", "render SOG is not a bundled archive")]
        if fmt == "spz" and magic[:2] != b"\x1f\x8b":
            return [Finding("RENDER_ASSET", "render SPZ is not gzip data")]
        return []
    return [Finding("RENDER_ASSET", "unknown render format")]


def validate_scene(
    manifest: Mapping[str, Any],
    fetch: Callable[[str], bytes],
    *,
    project_id: str | None = None,
    layout: Mapping[str, Any] | None = None,
) -> list[Finding]:
    """Full validation: manifest semantics, asset hashes/bytes, GLB and spawn checks."""
    findings = validate_manifest(manifest, project_id=project_id)
    if findings:
        return findings
    if layout is not None:
        findings += validate_layout(layout)
    blobs: dict[str, bytes] = {}
    for name, asset in manifest["assets"].items():
        if asset is None:
            continue
        try:
            data = fetch(asset["key"])
        except FileNotFoundError:
            findings.append(Finding("ASSET_MISSING", f"{name} asset is missing"))
            continue
        if len(data) != asset["bytes"]:
            findings.append(Finding("ASSET_BYTES", f"{name} byte count does not match"))
        if hashlib.sha256(data).hexdigest() != asset["sha256"]:
            findings.append(Finding("ASSET_SHA256", f"{name} sha256 does not match"))
        blobs[name] = data
    if "render" in blobs:
        findings += check_render_asset(blobs["render"], manifest["assets"]["render"]["format"])
    if "collision" in blobs:
        findings += check_collision_glb(blobs["collision"], manifest, layout)
    return findings
