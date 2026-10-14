from __future__ import annotations

import copy
import json
import math
import struct
from pathlib import Path

import numpy as np
import pytest

from andarilho_worker.geometry.collide import capsule_clearance, capsule_is_clear
from andarilho_worker.geometry.glb import GlbError, read_glb, write_glb
from andarilho_worker.geometry.layout import (
    LayoutError,
    collision_glb,
    collision_parts,
    rectangle_layout,
    validate_layout,
)
from andarilho_worker.geometry.ply import (
    PlyError,
    clean_splat_ply_file,
    drop_non_finite,
    inspect_splat_ply_file,
    read_splat_ply,
    recenter,
    write_splat_ply,
)
from andarilho_worker.geometry.polygon import triangulate
from andarilho_worker.geometry.synthetic import synthetic_splats

FIXTURES = Path(__file__).resolve().parents[3] / "packages" / "scene-contract" / "fixtures"


@pytest.fixture
def layout() -> dict:
    return json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())


@pytest.fixture
def nodes(layout):
    return read_glb(collision_glb(layout))[1]


def test_fixture_layout_is_valid(layout):
    assert validate_layout(layout) == []


def test_door_gap_is_clear_and_walls_block(nodes):
    # The doorway in the divider wall (x = 0) spans z in [-0.45, 0.45].
    for x in (-0.6, -0.3, 0.0, 0.3, 0.6):
        assert capsule_is_clear(nodes, [x, 0.0, 0.0]), x
    # Elsewhere the divider blocks a capsule standing on it.
    for z in (-1.5, -0.8, 0.8, 1.5):
        assert not capsule_is_clear(nodes, [0.0, 0.0, z]), z
    # Outer walls block, the table blocks, open floor is clear.
    assert not capsule_is_clear(nodes, [3.9, 0.0, 0.0])
    assert not capsule_is_clear(nodes, [-2.0, 0.0, -0.5])
    assert capsule_is_clear(nodes, [2.0, 0.0, 0.0])
    assert capsule_clearance(nodes, [2.0, 0.0, 0.0]) > 1.5


def test_lintel_above_opening_and_no_wall_across_it(nodes):
    lintels = [n for n in nodes if n.kind == "lintel"]
    assert len(lintels) == 1
    pos = lintels[0].positions
    assert pos[:, 1].min() == pytest.approx(2.1) and pos[:, 1].max() == pytest.approx(2.6)
    assert pos[:, 2].min() == pytest.approx(-0.45) and pos[:, 2].max() == pytest.approx(0.45)
    # No full-height wall piece of the divider overlaps the opening span.
    for n in nodes:
        if n.name.startswith("wall-divider-seg"):
            zs = n.positions[:, 2]
            assert zs.max() <= -0.45 + 1e-6 or zs.min() >= 0.45 - 1e-6


def test_identity_nodes_and_world_vertices(layout):
    glb = collision_glb(layout)
    doc, nodes = read_glb(glb)
    for node in doc["nodes"]:
        assert not {"matrix", "translation", "rotation", "scale"} & set(node)
    allpos = np.vstack([n.positions for n in nodes])
    assert np.all(np.isfinite(allpos))
    assert allpos[:, 0].min() == pytest.approx(-4.05, abs=1e-5)
    assert allpos[:, 1].max() == pytest.approx(2.6, abs=1e-5)


def test_glb_reader_rejects_non_identity_node(layout):
    glb = collision_glb(layout)
    json_len = struct.unpack_from("<I", glb, 12)[0]
    doc = json.loads(glb[20 : 20 + json_len])
    doc["nodes"][0]["translation"] = [0, 1, 0]
    new_json = json.dumps(doc).encode()
    new_json += b" " * ((4 - len(new_json) % 4) % 4)
    rest = glb[20 + json_len :]
    total = 12 + 8 + len(new_json) + len(rest)
    patched = struct.pack("<III", 0x46546C67, 2, total) + struct.pack(
        "<II", len(new_json), 0x4E4F534A
    )
    patched += new_json + rest
    with pytest.raises(GlbError, match="non-identity"):
        read_glb(patched)


def test_floor_with_hole_is_triangulated(layout):
    layout = copy.deepcopy(layout)
    layout["floors"][0]["holes"] = [[[1, 1], [2, 1], [2, 2], [1, 2]]]
    floor = next(p for p in collision_parts(layout) if p.kind == "floor")
    tris = floor.positions[floor.indices]
    area = 0.0
    for a, b, c in tris:
        area += np.linalg.norm(np.cross(b - a, c - a)) / 2
        centroid = (a + b + c) / 3
        assert not (1 < centroid[0] < 2 and 1 < centroid[2] < 2)  # nothing inside the hole
        assert np.cross(b - a, c - a)[1] > 0  # faces up
    assert area == pytest.approx(40 - 1)


def test_triangulate_simple():
    pts, tris = triangulate([(0, 0), (1, 0), (1, 1), (0, 1)])
    assert len(pts) == 4 and len(tris) == 2


@pytest.mark.parametrize(
    "mutate,code",
    [
        (
            lambda l: l["floors"][0].__setitem__("polygons", [[[0, 0], [2, 2], [2, 0], [0, 2]]]),
            "INVALID_FLOOR",
        ),
        (
            lambda l: l["floors"][0].__setitem__("polygons", [[[0, 0], [1, 0], [2, 0]]]),
            "INVALID_FLOOR",
        ),
        (lambda l: l["walls"][0].__setitem__("thickness", -0.1), "SCHEMA"),
        (lambda l: l["walls"][0].__setitem__("thickness", 5.0), "INVALID_WALL"),
        (lambda l: l["walls"][0].__setitem__("height", 50.0), "INVALID_WALL"),
        (lambda l: l["walls"][0].__setitem__("end", l["walls"][0]["start"]), "INVALID_WALL"),
        (lambda l: l["openings"][0].__setitem__("top", 3.0), "INVALID_OPENING"),
        (lambda l: l["openings"][0].__setitem__("offset", 4.5), "INVALID_OPENING"),
        (lambda l: l["spawn"].__setitem__("feet", [10, 0, 0]), "SPAWN_NOT_ON_FLOOR"),
        (lambda l: l["spawn"].__setitem__("feet", [-2, 0, -0.5]), "SPAWN_BLOCKED"),
        (lambda l: l["walls"][0].__setitem__("floorId", "nope"), "UNKNOWN_FLOOR"),
        (lambda l: l["obstacles"][0].__setitem__("height", math.inf), "NON_FINITE"),
    ],
)
def test_invalid_layouts_rejected(layout, mutate, code):
    bad = copy.deepcopy(layout)
    mutate(bad)
    findings = validate_layout(bad)
    assert code in {f.code for f in findings}, findings
    with pytest.raises(LayoutError):
        collision_glb(bad)


def test_closed_door_blocks(layout):
    layout = copy.deepcopy(layout)
    layout["openings"][0]["state"] = "closed"
    nodes = read_glb(collision_glb(layout))[1]
    assert not capsule_is_clear(nodes, [0.0, 0.0, 0.0])


def test_rectangle_proposal_valid():
    layout = rectangle_layout([-2, -3], [2, 3], elevation=0.1, height=2.5)
    assert validate_layout(layout) == []
    assert collision_parts(layout)


def test_glb_roundtrip_deterministic(layout):
    assert collision_glb(layout) == collision_glb(layout)
    with pytest.raises(ValueError):
        from andarilho_worker.geometry.glb import MeshPart

        write_glb([MeshPart("x", "wall", np.array([[math.nan, 0, 0]] * 3), np.array([[0, 1, 2]]))])


def test_synthetic_splats_and_ply_roundtrip(layout):
    cloud = synthetic_splats(layout)
    assert 1000 < cloud.count <= 4000
    blob = write_splat_ply(cloud)
    assert len(blob) < 300_000
    again = read_splat_ply(blob)
    assert again.count == cloud.count
    data = again.data.copy()
    data["x"][0] = np.nan
    cleaned, dropped = drop_non_finite(type(again)(data))
    assert dropped == 1 and cleaned.count == cloud.count - 1
    centered, center = recenter(cleaned)
    assert np.allclose(np.median(centered.positions(), axis=0), 0, atol=1e-3)


@pytest.mark.parametrize("chunk", [7, 1000, 1 << 20])
def test_chunked_cleaning_matches_the_in_memory_pipeline(layout, tmp_path, chunk):
    cloud = synthetic_splats(layout)
    data = cloud.data.copy()
    data["x"][[0, 6, 7, 500]] = np.nan  # around and across chunk boundaries
    data["opacity"][-1] = np.inf
    src = tmp_path / "export.ply"
    src.write_bytes(write_splat_ply(type(cloud)(data)))
    expected, dropped = drop_non_finite(type(cloud)(data))
    expected, center = recenter(expected)

    dst = tmp_path / "normalized.ply"
    cleaned = clean_splat_ply_file(src, dst, comments=["c"], chunk=chunk)
    assert dst.read_bytes() == write_splat_ply(expected, comments=["c"])
    assert (cleaned.count, cleaned.dropped, cleaned.center) == (expected.count, 5, center)
    assert dropped == 5 and np.array_equal(cleaned.positions, expected.positions())
    src.unlink()  # the memory map is released


def test_ply_files_are_validated_without_reading_records(layout, tmp_path):
    blob = write_splat_ply(synthetic_splats(layout))
    path = tmp_path / "scene.ply"
    path.write_bytes(blob)
    layout_info = inspect_splat_ply_file(path)
    assert layout_info.count == synthetic_splats(layout).count
    bad_count = b"ply\nformat binary_little_endian 1.0\nelement vertex x\nend_header\n"
    not_ascii = b"ply\n\xff\xfe\nend_header\n"
    for bad in (b"", bad_count, not_ascii, blob[:-1]):
        path.write_bytes(bad)
        with pytest.raises(PlyError):
            inspect_splat_ply_file(path)
