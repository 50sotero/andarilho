from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from andarilho_worker import validation
from andarilho_worker.geometry.glb import _pack_glb, _split_glb, write_textured_glb
from andarilho_worker.geometry.layout import (
    collision_glb,
    layout_bounds,
    rectangle_layout,
)
from andarilho_worker.geometry.texture import TextureParams, procedural_texture_layout
from andarilho_worker.geometry.textured import TexturedMesh, TexturedPart
from andarilho_worker.manifest import asset_entry, build_manifest, navigation_for, scene_prefix
from andarilho_worker.validation import (
    canonical_json,
    check_poster_asset,
    check_render_asset,
    loads_strict,
    manifest_sha256,
    validate_manifest,
    validate_scene,
)

REPO = Path(__file__).resolve().parents[3]
FIXTURES = REPO / "packages" / "scene-contract" / "fixtures"


@pytest.fixture
def manifest() -> dict:
    return json.loads((FIXTURES / "synthetic-two-room.manifest.json").read_text())


def fetch(key: str) -> bytes:
    name = key.rsplit("/", 1)[-1]
    local = {
        "model.glb": "synthetic-two-room.model.glb",
        "collision.glb": "synthetic-two-room.collision.glb",
    }
    return (FIXTURES / local[name]).read_bytes()


def codes(findings) -> set[str]:
    return {f.code for f in findings}


def _quad(pages: list[np.ndarray]) -> TexturedMesh:
    """One unit square on the floor (normal +Y), textured from the first page."""
    positions = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1]], dtype=np.float64)
    part = TexturedPart(
        name="floor",
        kind="floor",
        positions=positions,
        normals=np.tile([0.0, 1.0, 0.0], (4, 1)),
        uvs=np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64),
        indices=np.array([[0, 1, 2], [2, 1, 3]]),
        page=0,
    )
    return TexturedMesh(parts=[part], pages=pages, coverage={"all": 1.0}, texel_size=0.01)


def _square_glb(width: int = 8, height: int = 8) -> bytes:
    return write_textured_glb(_quad([np.zeros((height, width, 3), dtype=np.uint8)]))


def _bounds(low: list[float], high: list[float]) -> dict:
    return {"bounds": {"min": low, "max": high}}


UNIT_BOUNDS = _bounds([-0.1, -0.1, -0.1], [1.1, 0.1, 1.1])


def _mutated_glb(data: bytes, mutate) -> bytes:
    doc, bin_blob = _split_glb(data)
    mutate(doc)
    return _pack_glb(doc, bin_blob)


def test_fixture_manifest_valid(manifest):
    layout = json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())
    assert validate_scene(manifest, fetch, project_id="fixture-project", layout=layout) == []


def test_fixture_manifest_is_a_v2_textured_mesh(manifest):
    assert manifest["schemaVersion"] == "2.0"
    assert manifest["renderMode"] == "mesh"
    assert "worldFromDecodedRender" not in manifest
    render = manifest["assets"]["render"]
    assert render["format"] == "glb" and render["key"].endswith("/model.glb")


def test_rejects_nan(manifest):
    bad = copy.deepcopy(manifest)
    bad["bounds"]["min"][0] = math.nan
    assert codes(validate_manifest(bad)) == {"NON_FINITE"}
    with pytest.raises(ValueError):
        loads_strict('{"a": NaN}')


def test_rejects_the_previous_manifest_version(manifest):
    bad = copy.deepcopy(manifest)
    bad["schemaVersion"] = "1.0"
    assert codes(validate_manifest(bad)) == {"SCHEMA_VERSION"}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda m: m.update(renderMode="gaussian"),
        lambda m: m.update(worldFromDecodedRender=[1.0] + [0.0] * 15),
        lambda m: m["assets"]["render"].update(format="ply"),
    ],
)
def test_rejects_splat_era_fields(manifest, mutate):
    bad = copy.deepcopy(manifest)
    mutate(bad)
    assert codes(validate_manifest(bad)) == {"SCHEMA"}


def test_rejects_foreign_prefixes(manifest):
    bad = copy.deepcopy(manifest)
    bad["assets"]["collision"]["key"] = (
        "tenants/other-project/scenes/fixture-revision/collision.glb"
    )
    assert "ASSET_PREFIX" in codes(validate_manifest(bad))
    bad = copy.deepcopy(manifest)
    bad["assets"]["render"]["key"] = "tenants/fixture-project/scenes/other-revision/model.glb"
    assert "ASSET_PREFIX" in codes(validate_manifest(bad))
    assert "ASSET_PREFIX" in codes(validate_manifest(manifest, project_id="someone-else"))
    bad = copy.deepcopy(manifest)
    bad["assets"]["render"]["key"] = "tenants/fixture-project/runs/x/model.glb"
    assert codes(validate_manifest(bad))


def test_rejects_unknown_schema_version(manifest):
    bad = copy.deepcopy(manifest)
    bad["schemaVersion"] = "3.0"
    assert codes(validate_manifest(bad)) == {"SCHEMA_VERSION"}


def test_rejects_spawn_outside_bounds_and_bad_hash(manifest):
    bad = copy.deepcopy(manifest)
    bad["spawn"]["feet"] = [50, 0, 0]
    assert "SPAWN" in codes(validate_manifest(bad))
    bad = copy.deepcopy(manifest)
    bad["assets"]["render"]["sha256"] = "0" * 64
    assert "ASSET_SHA256" in codes(validate_scene(bad, fetch))


WEBP = b"RIFF" + (4).to_bytes(4, "little") + b"WEBP"


def test_poster_is_checked_against_its_declared_format(manifest):
    assert check_poster_asset(WEBP, "webp") == []
    assert check_poster_asset(b"\xff\xd8\xff\xe0", "jpeg") == []
    assert codes(check_poster_asset(b"\xff\xd8\xff\xe0", "webp")) == {"POSTER_ASSET"}
    assert codes(check_poster_asset(b"RIFF\x04\x00\x00\x00WAVE", "webp")) == {"POSTER_ASSET"}
    with_poster = copy.deepcopy(manifest)
    prefix = with_poster["assets"]["render"]["key"].rsplit("/", 1)[0]
    poster = {"key": f"{prefix}/poster.webp", "format": "webp", "bytes": len(WEBP),
              "sha256": hashlib.sha256(WEBP).hexdigest()}  # fmt: skip
    with_poster["assets"]["poster"] = poster

    def fetch_with(data: bytes):
        return lambda key: data if key.endswith("/poster.webp") else fetch(key)

    assert validate_scene(with_poster, fetch_with(WEBP)) == []
    jpeg = b"\xff\xd8\xff" + WEBP[3:]
    poster.update(sha256=hashlib.sha256(jpeg).hexdigest())
    assert codes(validate_scene(with_poster, fetch_with(jpeg))) == {"POSTER_ASSET"}


def test_spawn_blocked_detected(manifest):
    bad = copy.deepcopy(manifest)
    bad["spawn"]["feet"] = [0.0, 0.0, 1.5]  # inside the divider wall
    assert "SPAWN_BLOCKED" in codes(validate_scene(bad, fetch))


def test_spawn_obstacle_check_uses_the_spawn_floor(manifest):
    layout = json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())
    assert validate_scene(manifest, fetch, layout=layout) == []
    feet = manifest["spawn"]["feet"]
    x, z = feet[0], feet[2]
    square = [[x - 0.2, z - 0.2], [x + 0.2, z - 0.2], [x + 0.2, z + 0.2], [x - 0.2, z + 0.2]]
    upstairs = copy.deepcopy(layout)
    upstairs["floors"].append({**copy.deepcopy(layout["floors"][0]), "id": "upper", "elevation": 3})
    upstairs["obstacles"] = [{"id": "desk", "floorId": "upper", "footprint": square, "height": 0.8}]
    assert validate_scene(manifest, fetch, layout=upstairs) == []
    same_floor = copy.deepcopy(upstairs)
    same_floor["obstacles"][0]["floorId"] = manifest["spawn"]["floorId"]
    assert "SPAWN_BLOCKED" in codes(validate_scene(manifest, fetch, layout=same_floor))


def _table(item_id: str, position: list[float], **changes) -> dict:
    """A table with a raised top and a leg standing on the floor."""
    item = {
        "id": item_id,
        "floorId": "floor-0",
        "category": "table",
        "position": position,
        "yawRadians": 0.0,
        "size": [1.0, 0.75, 0.6],
        "solid": True,
        "parts": [
            {"shape": "box", "min": [0, 0.9, 0], "max": [1, 1, 1], "color": "#c0a080"},
            {"shape": "box", "min": [0.05, 0, 0.05], "max": [0.12, 0.9, 0.12], "color": "#444444"},
        ],
    }
    item.update(changes)
    return item


PACKAGE_PROJECT, PACKAGE_REVISION = "project-1", "revision-1"


def _fetch(blobs: dict[str, bytes]):
    def fetch(key: str) -> bytes:
        return blobs[key]

    return fetch


def _package(layout: dict, geometry_source: str = "proposal") -> tuple[dict, dict[str, bytes]]:
    """The manifest and assets that packaging would write for a layout, as bytes by key."""
    prefix = scene_prefix(PACKAGE_PROJECT, PACKAGE_REVISION)
    model = write_textured_glb(
        procedural_texture_layout(layout, TextureParams(texel_size=0.02, page_size=512))
    )
    collision = collision_glb(layout)
    manifest = build_manifest(
        scene_id="scene-1",
        scene_revision_id=PACKAGE_REVISION,
        revision=1,
        run_id="run-1",
        layout=layout,
        layout_revision=1,
        render=asset_entry(prefix + "model.glb", model, "glb"),
        collision=asset_entry(prefix + "collision.glb", collision, "glb"),
        poster=None,
        scale_status="estimated",
        geometry_source=geometry_source,
        recipe_digest="recipe-digest",
        capture_set_id="capture-1",
        quality={"registeredViews": 0, "selectedViews": 0, "disconnectedComponents": 0},
    )
    return manifest, {prefix + "model.glb": model, prefix + "collision.glb": collision}


@pytest.mark.parametrize("source", ["user_confirmed_trace", "synthetic_fixture", "proposal"])
def test_every_geometry_source_is_walkable_at_once(source):
    nav = navigation_for(source, 4)
    assert nav["mode"] == "approved_collision"
    assert nav["approved"] is True
    assert nav["layoutRevision"] == 4
    assert (nav["playerRadius"], nav["playerHeight"], nav["eyeHeight"]) == (0.25, 1.7, 1.6)


def test_an_automatic_proposal_packages_walkable_and_valid():
    layout = rectangle_layout((-2.0, -2.0), (2.0, 2.0), elevation=0.0, height=2.5)
    manifest, blobs = _package(layout, geometry_source="proposal")
    assert manifest["provenance"]["geometrySource"] == "proposal"
    assert manifest["navigation"]["mode"] == "approved_collision"
    assert manifest["navigation"]["approved"] is True
    findings = validate_scene(manifest, _fetch(blobs), project_id=PACKAGE_PROJECT, layout=layout)
    assert findings == []


def test_a_furnished_scene_validates_with_its_bounds_and_collision():
    layout = rectangle_layout((-2.0, -2.0), (2.0, 2.0), elevation=0.0, height=2.5)
    # The lamp is not solid and reaches 0.45 m beyond the walls. Its parts are drawn, so the
    # scene bounds must take them in or the model would lie outside them.
    lamp = {
        "id": "lamp",
        "floorId": "floor-0",
        "category": "lamp",
        "position": [2.2, 0.0],
        "yawRadians": 0.0,
        "size": [0.5, 1.8, 0.5],
        "solid": False,
        "parts": [{"shape": "cylinder", "min": [0, 0, 0], "max": [1, 1, 1], "color": "#ffe9a0"}],
    }
    layout["furniture"] = [_table("table", [-1.0, 0.5]), lamp]
    manifest, blobs = _package(layout)
    assert manifest["bounds"]["max"][0] >= 2.45
    findings = validate_scene(manifest, _fetch(blobs), project_id=PACKAGE_PROJECT, layout=layout)
    assert findings == []


def test_solid_furniture_over_the_spawn_blocks_the_scene_and_a_passable_piece_does_not():
    base = rectangle_layout((-2.0, -2.0), (2.0, 2.0), elevation=0.0, height=2.5)
    manifest, blobs = _package(base)  # packaged without the furniture, validated with it
    fetch = _fetch(blobs)
    blocking = {**base, "furniture": [_table("table", [0.0, 0.0])]}  # the spawn is at (0, 0)
    findings = validate_scene(manifest, fetch, project_id=PACKAGE_PROJECT, layout=blocking)
    assert "SPAWN_BLOCKED" in codes(findings)
    passable = {**base, "furniture": [_table("table", [0.0, 0.0], solid=False)]}
    assert validate_scene(manifest, fetch, project_id=PACKAGE_PROJECT, layout=passable) == []


def test_canonical_sha_matches_api_rule():
    value = {"b": 1, "a": "ç"}
    assert canonical_json(value) == '{"a":"ç","b":1}'.encode()
    assert manifest_sha256(value) == __import__("hashlib").sha256(canonical_json(value)).hexdigest()


def test_render_model_passes_on_the_fixture_and_on_a_generated_room(manifest):
    fixture = (FIXTURES / "synthetic-two-room.model.glb").read_bytes()
    assert check_render_asset(fixture, "glb", manifest) == []
    layout = rectangle_layout((-1.5, -1.25), (1.5, 1.25), elevation=0.0, height=2.6)
    mesh = procedural_texture_layout(layout, TextureParams(texel_size=0.02, page_size=512))
    generated = write_textured_glb(mesh)
    assert check_render_asset(generated, "glb", {"bounds": layout_bounds(layout)}) == []


def test_render_model_must_be_unlit():
    def lit(doc):
        for material in doc["materials"]:
            material.pop("extensions", None)
        doc["extensionsUsed"] = []

    findings = check_render_asset(_mutated_glb(_square_glb(), lit), "glb", UNIT_BOUNDS)
    assert codes(findings) == {"RENDER_ASSET"}
    assert "unlit" in findings[0].message


def test_render_model_triangle_budget(monkeypatch):
    glb = _square_glb()  # two triangles
    monkeypatch.setattr(validation, "MAX_RENDER_TRIANGLES", 2)
    assert check_render_asset(glb, "glb", UNIT_BOUNDS) == []
    monkeypatch.setattr(validation, "MAX_RENDER_TRIANGLES", 1)
    assert codes(check_render_asset(glb, "glb", UNIT_BOUNDS)) == {"RENDER_ASSET"}


@pytest.mark.parametrize("side,accepted", [(4096, True), (4097, False)])
def test_render_texture_side_limit_is_inclusive(side, accepted):
    glb = write_textured_glb(_quad([np.zeros((4, side, 3), dtype=np.uint8)]))
    findings = check_render_asset(glb, "glb", UNIT_BOUNDS)
    assert (findings == []) is accepted
    assert accepted or "texture is too large" in findings[0].message


def test_render_model_must_lie_within_the_scene_bounds():
    glb = _square_glb()
    inside = _bounds([-0.1, -0.1, -0.1], [1.1, 0.1, 1.1])
    assert check_render_asset(glb, "glb", inside) == []
    outside = _bounds([-0.1, -0.1, -0.1], [0.5, 0.1, 1.1])
    assert codes(check_render_asset(glb, "glb", outside)) == {"RENDER_ASSET"}


@pytest.mark.parametrize(
    "data,fmt",
    [
        (b"not a glb at all, just text", "glb"),
        (b"glTF" + b"\x00" * 16, "glb"),
        (_square_glb(), "ply"),
    ],
)
def test_render_asset_must_be_a_glb(data, fmt):
    assert codes(check_render_asset(data, fmt, UNIT_BOUNDS)) == {"RENDER_ASSET"}


def test_render_model_size_limit(manifest, monkeypatch):
    monkeypatch.setattr(validation, "MAX_GLB_BYTES", 100)
    findings = validate_scene(manifest, fetch)
    assert "RENDER_ASSET" in codes(findings)
    assert any("too large" in f.message for f in findings if f.code == "RENDER_ASSET")


def test_check_scene_script_on_fixture():
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(REPO / "scripts" / "check-scene.py"),
         str(FIXTURES / "synthetic-two-room.manifest.json"), "--project-id", "fixture-project"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr


def test_fixtures_up_to_date():
    proc = subprocess.run(  # noqa: S603
        [sys.executable, str(REPO / "scripts" / "build-fixture.py"), "--check"],
        capture_output=True, text=True, timeout=120, check=False,
    )  # fmt: skip
    assert proc.returncode == 0, proc.stderr
