from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from andarilho_worker.validation import (
    canonical_json,
    check_poster_asset,
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
        "scene.ply": "synthetic-two-room.splat.ply",
        "collision.glb": "synthetic-two-room.collision.glb",
    }
    return (FIXTURES / local[name]).read_bytes()


def codes(findings) -> set[str]:
    return {f.code for f in findings}


def test_fixture_manifest_valid(manifest):
    layout = json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())
    assert validate_scene(manifest, fetch, project_id="fixture-project", layout=layout) == []


def test_rejects_nan(manifest):
    bad = copy.deepcopy(manifest)
    bad["bounds"]["min"][0] = math.nan
    assert codes(validate_manifest(bad)) == {"NON_FINITE"}
    with pytest.raises(ValueError):
        loads_strict('{"a": NaN}')


@pytest.mark.parametrize(
    "matrix",
    [
        [2, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # non-uniform scale
        [-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # reflection
        [1, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # projective
        [1, 0.3, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # shear
    ],
)
def test_rejects_bad_matrix(manifest, matrix):
    bad = copy.deepcopy(manifest)
    bad["worldFromDecodedRender"] = matrix
    assert "MATRIX" in codes(validate_manifest(bad))


def test_accepts_rotation_scale_translation(manifest):
    good = copy.deepcopy(manifest)
    c, s, k = math.cos(0.3), math.sin(0.3), 2.0
    good["worldFromDecodedRender"] = [
        k * c,
        0,
        -k * s,
        0,
        0,
        k,
        0,
        0,
        k * s,
        0,
        k * c,
        0,
        1,
        2,
        3,
        1,
    ]
    assert validate_manifest(good) == []


def test_rejects_foreign_prefixes(manifest):
    bad = copy.deepcopy(manifest)
    bad["assets"]["collision"]["key"] = (
        "tenants/other-project/scenes/fixture-revision/collision.glb"
    )
    assert "ASSET_PREFIX" in codes(validate_manifest(bad))
    bad = copy.deepcopy(manifest)
    bad["assets"]["render"]["key"] = "tenants/fixture-project/scenes/other-revision/scene.ply"
    assert "ASSET_PREFIX" in codes(validate_manifest(bad))
    assert "ASSET_PREFIX" in codes(validate_manifest(manifest, project_id="someone-else"))
    bad = copy.deepcopy(manifest)
    bad["assets"]["render"]["key"] = "tenants/fixture-project/runs/x/scene.ply"
    assert codes(validate_manifest(bad))


def test_rejects_unknown_schema_version(manifest):
    bad = copy.deepcopy(manifest)
    bad["schemaVersion"] = "2.0"
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


def test_canonical_sha_matches_api_rule():
    value = {"b": 1, "a": "ç"}
    assert canonical_json(value) == '{"a":"ç","b":1}'.encode()
    assert manifest_sha256(value) == __import__("hashlib").sha256(canonical_json(value)).hexdigest()


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
