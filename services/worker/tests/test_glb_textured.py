from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from andarilho_worker.geometry import glb as glb_module
from andarilho_worker.geometry.glb import (
    GLB_MAGIC,
    GlbError,
    MeshPart,
    _frame,
    _pack_glb,
    _split_glb,
    inspect_textured_glb,
    read_glb,
    write_glb,
    write_textured_glb,
)
from andarilho_worker.geometry.layout import collision_parts, layout_bounds, rectangle_layout
from andarilho_worker.geometry.texture import TextureParams, procedural_texture_layout
from andarilho_worker.geometry.textured import TexturedMesh, TexturedPart
from andarilho_worker.validation import check_collision_glb, check_render_asset

FIXTURES = Path(__file__).resolve().parents[3] / "packages" / "scene-contract" / "fixtures"
COLLISION_SHA256 = "76f09ecbe81a4c85f1a3c379d488019009808b4e325a3c6cd7c09b5134f5508f"
COLLISION_BYTES = 7008


def _layout() -> dict:
    return json.loads((FIXTURES / "synthetic-two-room.layout.json").read_text())


def _manifest() -> dict:
    return json.loads((FIXTURES / "synthetic-two-room.manifest.json").read_text())


def _pages() -> list[np.ndarray]:
    """Deterministic pages of the sizes the structure tests use: 8x8 and 16x4 (width x height)."""
    floor = (np.arange(8 * 8 * 3, dtype=np.int64) * 7 % 256).astype(np.uint8).reshape(8, 8, 3)
    wall = (np.arange(4 * 16 * 3, dtype=np.int64) * 13 % 256).astype(np.uint8).reshape(4, 16, 3)
    return [floor, wall]


def _gradient_page(width: int, height: int) -> np.ndarray:
    """A smooth RGB page with a different ramp per channel, so flips and swaps show up."""
    x = np.arange(width)[None, :] / (width - 1)
    y = np.arange(height)[:, None] / (height - 1)
    red = 30 + 200 * np.broadcast_to(x, (height, width))
    green = 40 + 180 * np.broadcast_to(y, (height, width))
    blue = 220 - 150 * np.broadcast_to((x + y) / 2, (height, width))
    return np.stack([red, green, blue], axis=-1).round().astype(np.uint8)


def _two_quads(pages: list[np.ndarray] | None = None) -> TexturedMesh:
    """A floor quad on page 0 and a wall quad on page 1 (by default 8x8 and 16x4)."""
    floor = TexturedPart(
        name="floor-0",
        kind="floor",
        positions=np.array([[0, 0, 0], [0, 0, 2], [2, 0, 2], [2, 0, 0]], dtype=np.float64),
        normals=np.tile(np.array([0.0, 1.0, 0.0]), (4, 1)),
        uvs=np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float64),
        indices=np.array([[0, 1, 2], [0, 2, 3]]),
        page=0,
        extras={"floorId": "f0"},
    )
    wall = TexturedPart(
        name="wall-0",
        kind="wall",
        positions=np.array([[0, 0, 0], [0, 2.5, 0], [0, 2.5, 2], [0, 0, 2]], dtype=np.float64),
        normals=np.tile(np.array([1.0, 0.0, 0.0]), (4, 1)),
        uvs=np.array([[1, 1], [1, 0], [0, 0], [0, 1]], dtype=np.float64),
        indices=np.array([[0, 1, 2], [0, 2, 3]]),
        page=1,
        extras={"wallId": "w0"},
    )
    return TexturedMesh(
        parts=[floor, wall],
        pages=_pages() if pages is None else pages,
        coverage={"floor": 1.0, "wall": 0.75, "all": 0.875},
        texel_size=0.0625,
    )


def _doc(data: bytes) -> dict:
    return _split_glb(data)[0]


def _image_bytes(data: bytes, index: int) -> bytes:
    doc, bin_chunk = _split_glb(data)
    view = doc["bufferViews"][doc["images"][index]["bufferView"]]
    return bin_chunk[view["byteOffset"] : view["byteOffset"] + view["byteLength"]]


def _float_accessor(data: bytes, index: int) -> np.ndarray:
    """Values of a float32 VEC2/VEC3 accessor, read back through its bufferView in the BIN chunk.

    The view must hold exactly the accessor's bytes, so a wrong bufferView index shows up here.
    """
    doc, bin_chunk = _split_glb(data)
    acc = doc["accessors"][index]
    view = doc["bufferViews"][acc["bufferView"]]
    width = {"VEC2": 2, "VEC3": 3}[acc["type"]]
    assert acc["componentType"] == glb_module.FLOAT
    assert view["byteLength"] == acc["count"] * width * 4
    values = np.frombuffer(
        bin_chunk, dtype="<f4", count=acc["count"] * width, offset=view.get("byteOffset", 0)
    )
    return values.reshape(acc["count"], width)


def _bin_start(data: bytes) -> int:
    """Offset of the BIN payload in a file laid out as JSON chunk then BIN chunk."""
    return 20 + struct.unpack_from("<I", data, 12)[0] + 8


def _repack(data: bytes, mutate) -> bytes:
    """Rebuild a GLB after editing its JSON document; the BIN chunk is kept as is."""
    doc, bin_chunk = _split_glb(data)
    mutate(doc)
    return _pack_glb(doc, bin_chunk)


def _with_trailing(data: bytes, extra: bytes) -> bytes:
    """Append bytes after the last chunk and make the header's total length match."""
    buf = bytearray(data + extra)
    struct.pack_into("<I", buf, 8, len(buf))
    return bytes(buf)


def _without_bin(data: bytes) -> bytes:
    """The same GLB with its BIN chunk dropped, so only the header and JSON chunk remain."""
    json_end = 20 + struct.unpack_from("<I", data, 12)[0]
    head = bytearray(data[:json_end])
    struct.pack_into("<I", head, 8, json_end)
    return bytes(head)


def _grid_mesh(side: int) -> tuple[np.ndarray, np.ndarray]:
    """A flat side x side grid of quads in the plane y=0: 2 * side**2 triangles."""
    xs, zs = np.meshgrid(np.arange(side + 1.0), np.arange(side + 1.0))
    positions = np.column_stack([xs.ravel(), np.zeros(xs.size), zs.ravel()])
    rows, cols = np.meshgrid(np.arange(side), np.arange(side), indexing="ij")
    a = (rows * (side + 1) + cols).ravel()
    b, c = a + 1, a + side + 1
    d = c + 1
    return positions, np.vstack([np.column_stack([a, b, d]), np.column_stack([a, d, c])])


def _instanced_glb(positions: np.ndarray, indices: np.ndarray, copies: int) -> bytes:
    """A collision GLB whose ``copies`` nodes all reference the one mesh built from the arrays."""
    data = write_glb([MeshPart(name="shared", kind="wall", positions=positions, indices=indices)])

    def instance(doc):
        doc["nodes"] = [{"name": f"copy-{i}", "mesh": 0} for i in range(copies)]
        doc["scenes"] = [{"name": "collision", "nodes": list(range(copies))}]

    return _repack(data, instance)


def _sliver_part(count: int, offset: float, seed: int) -> TexturedPart:
    """``count`` triangles whose three vertices are collinear in the plane y=0 (zero area).

    Their normals are +Y, so the noise float32 rounding leaves in the cross product must not
    decide the winding.
    """
    rng = np.random.default_rng(seed)
    base = rng.uniform(-3, 3, (count, 3)) * [1, 0, 1] + [offset, 0, offset]
    direction = rng.uniform(-1, 1, (count, 3)) * [1, 0, 1]
    steps = rng.uniform(-2, 2, (count, 2))
    corners = np.stack(
        [base, base + steps[:, :1] * direction, base + steps[:, 1:] * direction], axis=1
    )
    positions = corners.reshape(-1, 3)
    return TexturedPart(
        name="sliver-0",
        kind="floor",
        positions=positions,
        normals=np.tile(np.array([0.0, 1.0, 0.0]), (len(positions), 1)),
        uvs=np.zeros((len(positions), 2)),
        indices=np.arange(len(positions)).reshape(-1, 3),
        page=0,
    )


def _patch_float(data: bytes, accessor: int, element: int, component: int, value: float) -> bytes:
    """Overwrite one float32 component of an accessor in the BIN chunk."""
    doc, _ = _split_glb(data)
    acc = doc["accessors"][accessor]
    view = doc["bufferViews"][acc["bufferView"]]
    width = {"VEC2": 2, "VEC3": 3}[acc["type"]]
    offset = view.get("byteOffset", 0) + (element * width + component) * 4
    patched = bytearray(data)
    struct.pack_into("<f", patched, _bin_start(data) + offset, value)
    return bytes(patched)


def _with_image(data: bytes, index: int, raw: bytes, mime: str) -> bytes:
    """Point image ``index`` at ``raw`` bytes appended to the BIN chunk, declared as ``mime``."""
    doc, bin_chunk = _split_glb(data)
    blob = bin_chunk + raw
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    doc["bufferViews"].append({"buffer": 0, "byteOffset": len(bin_chunk), "byteLength": len(raw)})
    doc["images"][index] = {"bufferView": len(doc["bufferViews"]) - 1, "mimeType": mime}
    doc["buffers"] = [{"byteLength": len(blob)}]
    return _pack_glb(doc, blob)


def _jpeg_segments(raw: bytes) -> list[tuple[int, bytes]]:
    """(marker code, payload) for each header segment of a JPEG, up to and including SOS."""
    assert raw[:2] == b"\xff\xd8"
    segments: list[tuple[int, bytes]] = []
    pos = 2
    while True:
        assert raw[pos] == 0xFF
        code = raw[pos + 1]
        if code == 0xDA:  # SOS: entropy-coded scan data follows, nothing more to parse
            segments.append((code, b""))
            return segments
        (length,) = struct.unpack_from(">H", raw, pos + 2)
        segments.append((code, raw[pos + 4 : pos + 2 + length]))
        pos += 2 + length


def _png_bytes(page: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(page).save(buf, format="PNG")
    return buf.getvalue()


def _decode_rgb(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as img:
        return np.asarray(img.convert("RGB"), dtype=np.float64)


def _mean_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).mean())


def test_collision_glb_bytes_are_unchanged():
    layout = _layout()
    data = write_glb(collision_parts(layout))
    assert len(data) == COLLISION_BYTES
    assert hashlib.sha256(data).hexdigest() == COLLISION_SHA256
    assert data == (FIXTURES / "synthetic-two-room.collision.glb").read_bytes()


def test_writes_glb_container_and_json():
    data = write_textured_glb(_two_quads())
    magic, version, total = struct.unpack_from("<III", data, 0)
    assert (magic, version, total) == (GLB_MAGIC, 2, len(data))
    assert len(data) % 4 == 0
    assert struct.unpack_from("<I", data, 16)[0] == glb_module.CHUNK_JSON
    assert struct.unpack_from("<I", data, _bin_start(data) - 4)[0] == glb_module.CHUNK_BIN
    doc = _doc(data)
    assert doc["extensionsUsed"] == ["KHR_materials_unlit"]
    assert "extensionsRequired" not in doc
    assert doc["scenes"] == [{"name": "visual", "nodes": [0, 1]}]
    assert doc["samplers"] == [
        {"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}
    ]
    assert doc["textures"] == [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}]
    assert [image["mimeType"] for image in doc["images"]] == ["image/jpeg", "image/jpeg"]
    assert all("uri" not in image for image in doc["images"])
    assert doc["materials"] == [
        {
            "name": "atlas-0",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0,
                "roughnessFactor": 1,
            },
            "extensions": {"KHR_materials_unlit": {}},
        },
        {
            "name": "atlas-1",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 1},
                "metallicFactor": 0,
                "roughnessFactor": 1,
            },
            "extensions": {"KHR_materials_unlit": {}},
        },
    ]
    assert doc["extras"]["andarilho"] == {
        "geometrySpace": "world",
        "units": "m",
        "upAxis": "+Y",
        "coverage": {"floor": 1.0, "wall": 0.75, "all": 0.875},
        "texelSize": 0.0625,
    }
    assert [node["name"] for node in doc["nodes"]] == ["floor-0", "wall-0"]
    assert [node["extras"] for node in doc["nodes"]] == [
        {"kind": "floor", "floorId": "f0"},
        {"kind": "wall", "wallId": "w0"},
    ]
    for node in doc["nodes"]:
        assert not {"matrix", "translation", "rotation", "scale"} & set(node)
    assert doc["meshes"][1]["primitives"] == [
        {
            "attributes": {"POSITION": 4, "NORMAL": 5, "TEXCOORD_0": 6},
            "indices": 7,
            "material": 1,
            "mode": 4,
        }
    ]
    wall_position = doc["accessors"][4]
    assert wall_position["type"] == "VEC3"
    assert wall_position["min"] == [0.0, 0.0, 0.0]
    assert wall_position["max"] == [0.0, 2.5, 2.0]
    assert doc["accessors"][6]["type"] == "VEC2"
    assert all(view["byteOffset"] % 4 == 0 for view in doc["bufferViews"])


def test_images_are_baseline_420_jpegs_without_metadata():
    data = write_textured_glb(_two_quads())
    for index in range(2):
        segments = _jpeg_segments(_image_bytes(data, index))
        codes = [code for code, _ in segments]
        assert 0xC0 in codes  # SOF0: baseline DCT
        assert 0xC2 not in codes  # no SOF2: not progressive
        assert not [code for code in codes if 0xE0 <= code <= 0xEF]  # no APPn segment at all
        assert 0xFE not in codes  # no COM comment
        sof = dict(segments)[0xC0]
        assert sof[5] == 3  # three components: Y, Cb, Cr
        assert (sof[7], sof[10], sof[13]) == (0x22, 0x11, 0x11)  # 4:2:0 sampling factors
        with Image.open(io.BytesIO(_image_bytes(data, index))) as img:
            assert img.format == "JPEG"


def test_jpeg_pixels_decode_to_the_pages_without_flip_or_channel_swap():
    pages = [_gradient_page(64, 48), _gradient_page(96, 32)]
    data = write_textured_glb(_two_quads(pages))
    for index, page in enumerate(pages):
        source = page.astype(np.float64)
        decoded = _decode_rgb(_image_bytes(data, index))
        assert decoded.shape == source.shape
        error = _mean_error(decoded, source)
        assert error < 3.0
        assert _mean_error(decoded, np.flipud(source)) > 10 * error
        assert _mean_error(decoded, source[..., ::-1]) > 10 * error
    assert inspect_textured_glb(data)["images"] == [[64, 48], [96, 32]]


def test_lower_jpeg_quality_gives_smaller_images():
    low = write_textured_glb(_two_quads(), jpeg_quality=20)
    high = write_textured_glb(_two_quads(), jpeg_quality=95)
    assert len(low) < len(high)
    for index in range(2):
        assert len(_image_bytes(low, index)) < len(_image_bytes(high, index))
    assert inspect_textured_glb(low)["images"] == inspect_textured_glb(high)["images"]


def test_inspect_reports_counts_sizes_bounds_and_unlit():
    summary = inspect_textured_glb(write_textured_glb(_two_quads()))
    assert summary == {
        "triangles": 4,
        "vertices": 8,
        "images": [[8, 8], [16, 4]],
        "materials": 2,
        "bounds": {"min": [0.0, 0.0, 0.0], "max": [2.0, 2.5, 2.0]},
        "unlit": True,
    }


def test_bounds_span_negative_and_positive_coordinates_across_nodes():
    mesh = _two_quads()
    mesh.parts[0].positions = mesh.parts[0].positions + np.array([-5.0, 0.0, -4.0])
    summary = inspect_textured_glb(write_textured_glb(mesh))
    assert summary["bounds"] == {"min": [-5.0, 0.0, -4.0], "max": [0.0, 2.5, 2.0]}


def test_read_glb_still_returns_the_textured_nodes():
    _, nodes = read_glb(write_textured_glb(_two_quads()))
    assert [n.name for n in nodes] == ["floor-0", "wall-0"]
    assert [n.kind for n in nodes] == ["floor", "wall"]
    for node, part in zip(nodes, _two_quads().parts, strict=True):
        assert np.array_equal(node.positions, part.positions)
        assert np.array_equal(node.triangles, part.indices)


def test_attributes_read_back_exactly_from_their_buffer_views():
    mesh = _two_quads()
    data = write_textured_glb(mesh)
    primitives = [m["primitives"][0] for m in _doc(data)["meshes"]]
    for prim, part in zip(primitives, mesh.parts, strict=True):
        attrs = prim["attributes"]
        assert np.array_equal(
            _float_accessor(data, attrs["POSITION"]), part.positions.astype("<f4")
        )
        assert np.array_equal(_float_accessor(data, attrs["NORMAL"]), part.normals.astype("<f4"))
        assert np.array_equal(_float_accessor(data, attrs["TEXCOORD_0"]), part.uvs.astype("<f4"))


def test_uv_origin_is_top_left_and_normals_face_the_visible_side():
    data = write_textured_glb(_two_quads())
    floor, wall = (m["primitives"][0]["attributes"] for m in _doc(data)["meshes"])
    # The wall's first vertex is on the floor line (y=0), the bottom edge of the atlas: v=1.
    # Seen from +X the viewer's right is -Z, so that z=0 edge is the atlas's right edge: u=1.
    assert _float_accessor(data, wall["TEXCOORD_0"])[0].tolist() == [1.0, 1.0]
    # Its second vertex is on the top edge (y=2.5): v=0.
    assert _float_accessor(data, wall["TEXCOORD_0"])[1].tolist() == [1.0, 0.0]
    assert _float_accessor(data, floor["NORMAL"])[0].tolist() == [0.0, 1.0, 0.0]
    assert _float_accessor(data, wall["NORMAL"])[0].tolist() == [1.0, 0.0, 0.0]


def test_wall_u_runs_to_the_viewers_right_and_v_runs_down():
    data = write_textured_glb(_two_quads())
    wall = _doc(data)["meshes"][1]["primitives"][0]["attributes"]
    positions = _float_accessor(data, wall["POSITION"]).astype(np.float64)
    uvs = _float_accessor(data, wall["TEXCOORD_0"]).astype(np.float64)
    normal = _float_accessor(data, wall["NORMAL"])[0].astype(np.float64)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(-normal, up)  # the viewer looks along -normal; forward x up is right
    assert right.tolist() == [0.0, 0.0, -1.0]
    for a in range(len(positions)):
        for b in range(len(positions)):
            step = positions[a] - positions[b]
            if step @ right:
                assert np.sign(uvs[a, 0] - uvs[b, 0]) == np.sign(step @ right)
            if step @ up:
                assert np.sign(uvs[a, 1] - uvs[b, 1]) == -np.sign(step @ up)


@pytest.mark.parametrize(
    "value, accepted",
    [(1 + 5e-7, True), (1 + 1e-5, False), (-5e-7, True), (-1e-5, False)],
)
def test_uv_tolerance_edge_is_the_same_in_writer_and_reader(value, accepted):
    mesh = _two_quads()
    mesh.parts[0].uvs[2, 0] = value
    clean = write_textured_glb(_two_quads())
    if accepted:
        assert inspect_textured_glb(write_textured_glb(mesh))["triangles"] == 4
        assert inspect_textured_glb(_patch_float(clean, 2, 2, 0, value))["triangles"] == 4
    else:
        with pytest.raises(ValueError, match="uvs must lie in"):
            write_textured_glb(mesh)
        with pytest.raises(GlbError, match="must lie in"):
            inspect_textured_glb(_patch_float(clean, 2, 2, 0, value))


def test_two_writes_are_byte_identical():
    assert write_textured_glb(_two_quads()) == write_textured_glb(_two_quads())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda m: m.parts[0].uvs.__setitem__((2, 0), 1.5), "uvs must lie in"),
        (lambda m: m.parts[0].uvs.__setitem__((2, 0), -0.5), "uvs must lie in"),
        (lambda m: m.parts[0].uvs.__setitem__((2, 0), np.nan), "uvs must be finite"),
        (lambda m: setattr(m.parts[1], "page", 2), "page index 2"),
        (lambda m: setattr(m.parts[1], "page", -1), "page index -1"),
        (lambda m: setattr(m.parts[1], "page", True), "page index True"),
        (lambda m: m.parts[0].positions.__setitem__((0, 1), np.nan), "non-finite"),
        (lambda m: setattr(m.parts[0], "positions", np.zeros((0, 3))), "non-empty"),
        (lambda m: m.parts[0].normals.__setitem__((1, 0), np.inf), "normals must be finite"),
        (lambda m: m.parts[0].normals.__setitem__((1, 0), 2.0), "unit length"),
        (lambda m: m.parts[0].indices.__setitem__((0, 0), 4), "out of range"),
        (lambda m: m.parts[0].indices.__setitem__((0, 0), -1), "out of range"),
        (
            lambda m: setattr(m.parts[0], "indices", np.array([[0, 1, 2]], dtype=np.float64)),
            "indices must be",
        ),
        (
            lambda m: setattr(m, "pages", [m.pages[0].astype(np.float32), m.pages[1]]),
            "uint8",
        ),
        (
            lambda m: setattr(m, "pages", [np.zeros((8, 8, 4), np.uint8), m.pages[1]]),
            "uint8",
        ),
        (
            lambda m: setattr(m, "pages", [np.zeros((8193, 8, 3), np.uint8), m.pages[1]]),
            "each side",
        ),
        (lambda m: setattr(m, "pages", [np.zeros((8, 8), np.uint8), m.pages[1]]), "uint8"),
        (lambda m: setattr(m, "pages", [np.zeros((0, 8, 3), np.uint8), m.pages[1]]), "each side"),
        (lambda m: setattr(m, "pages", [np.zeros((8, 0, 3), np.uint8), m.pages[1]]), "each side"),
        (lambda m: setattr(m, "parts", []), "no parts"),
        (lambda m: setattr(m.parts[0], "uvs", np.zeros((4, 3))), r"uvs must be finite \(N, 2\)"),
        (lambda m: setattr(m.parts[0], "normals", np.zeros((4, 2))), "normals must be finite"),
        (
            lambda m: setattr(m.parts[0], "indices", np.zeros((0, 3), dtype=np.int64)),
            "non-empty integer",
        ),
        (lambda m: setattr(m.parts[0], "indices", np.array([0, 1, 2])), "indices must be"),
        (
            lambda m: setattr(m.parts[0], "indices", np.array([[0, 1, 2], [0, 3, 2]])),
            "winding disagrees",
        ),
        (
            lambda m: setattr(m.parts[0], "indices", m.parts[0].indices[:, ::-1]),
            "winding disagrees",
        ),
        (lambda m: setattr(m.parts[1], "normals", -m.parts[1].normals), "winding disagrees"),
        (lambda m: m.coverage.update(floor=1.5), r"coverage\['floor'\] must lie in"),
        (lambda m: m.coverage.update(wall=-0.1), r"coverage\['wall'\] must lie in"),
        (lambda m: m.coverage.update(all=math.nan), r"coverage\['all'\] must be finite"),
        (lambda m: m.coverage.update(all=True), r"coverage\['all'\] must be a number"),
        (lambda m: m.coverage.update(all="0.5"), r"coverage\['all'\] must be a number"),
        (lambda m: setattr(m, "texel_size", 0.0), "texel_size must be positive"),
        (lambda m: setattr(m, "texel_size", -0.01), "texel_size must be positive"),
        (lambda m: setattr(m, "texel_size", math.inf), "texel_size must be finite"),
        (lambda m: setattr(m, "texel_size", None), "texel_size must be a number"),
    ],
)
def test_rejects_invalid_meshes(mutate, message):
    mesh = _two_quads()
    mutate(mesh)
    with pytest.raises(ValueError, match=message):
        write_textured_glb(mesh)


@pytest.mark.parametrize("offset", [0.0, 1000.0])
def test_collinear_triangles_pass_the_winding_check(offset):
    mesh = TexturedMesh(parts=[_sliver_part(500, offset, seed=3)], pages=_pages())
    assert inspect_textured_glb(write_textured_glb(mesh))["triangles"] == 500


@pytest.mark.parametrize("order, accepted", [((0, 2, 1), True), ((0, 1, 2), False)])
def test_winding_is_still_judged_far_from_the_origin(order, accepted):
    # Edges (2, 0, 0) and (0, 0, 2) cross to -Y, so (0, 1, 2) is the flipped order for +Y.
    positions = np.array([[1000.0, 0.0, 1000.0], [1002.0, 0.0, 1000.0], [1000.0, 0.0, 1002.0]])
    part = TexturedPart(
        name="far-0",
        kind="floor",
        positions=positions,
        normals=np.tile(np.array([0.0, 1.0, 0.0]), (3, 1)),
        uvs=np.zeros((3, 2)),
        indices=np.array([order]),
        page=0,
    )
    mesh = TexturedMesh(parts=[part], pages=_pages())
    if accepted:
        assert inspect_textured_glb(write_textured_glb(mesh))["triangles"] == 1
    else:
        with pytest.raises(ValueError, match="winding disagrees"):
            write_textured_glb(mesh)


def test_positions_beyond_float32_fail_without_numpy_warnings():
    mesh = _two_quads()
    mesh.parts[0].positions[1, 1] = 1e39
    collision = [
        MeshPart(
            name="big",
            kind="wall",
            positions=np.array([[0.0, 0.0, 0.0], [1e39, 0.0, 0.0], [0.0, 1.0, 0.0]]),
            indices=np.array([[0, 1, 2]]),
        )
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="float32 range"):
            write_textured_glb(mesh)
        with pytest.raises(ValueError, match="float32 range"):
            write_glb(collision)


@pytest.mark.parametrize("quality", [0, 101, -5])
def test_rejects_bad_jpeg_quality(quality):
    with pytest.raises(ValueError, match="jpeg_quality"):
        write_textured_glb(_two_quads(), jpeg_quality=quality)


@pytest.mark.parametrize("quality", [1, 100])
def test_accepts_the_jpeg_quality_bounds(quality):
    data = write_textured_glb(_two_quads(), jpeg_quality=quality)
    assert inspect_textured_glb(data)["images"] == [[8, 8], [16, 4]]


def test_page_side_limits_are_inclusive():
    pages = [np.zeros((1, 1, 3), np.uint8), np.zeros((glb_module.MAX_TEXTURE_SIDE, 1, 3), np.uint8)]
    summary = inspect_textured_glb(write_textured_glb(_two_quads(pages)))
    assert summary["images"] == [[1, 1], [1, glb_module.MAX_TEXTURE_SIDE]]


def test_rejects_output_over_the_size_limit(monkeypatch):
    monkeypatch.setattr(glb_module, "MAX_GLB_BYTES", 100)
    with pytest.raises(ValueError, match="limit"):
        write_textured_glb(_two_quads())


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda d: d["images"][0].update(uri="floor.jpg"), "external images"),
        (lambda d: d["buffers"][0].update(uri="other.bin"), "external buffers"),
        (lambda d: d.update(extensionsRequired=["KHR_materials_unlit"]), "must not require"),
        (lambda d: d.update(extensionsUsed=[]), "must be listed in extensionsUsed"),
        (lambda d: d.update(extensionsUsed="KHR_materials_unlit"), "must be an array"),
        (lambda d: d["images"][1].update(mimeType="image/webp"), "unsupported image type"),
        (lambda d: d["images"][1].update(mimeType=None), "unsupported image type None"),
        (lambda d: d["images"][0].update(bufferView=20), "bufferView index 20"),
        (lambda d: d["images"][0].pop("bufferView"), "bufferView index None"),
        (lambda d: d["meshes"][1]["primitives"][0].update(material=5), "material index 5"),
        (lambda d: d["textures"][0].update(source=9), "image index 9"),
        (lambda d: d["textures"][0].pop("source"), "image index None"),
        (lambda d: d["textures"][0].update(sampler=3), "sampler index 3"),
        (
            lambda d: d["materials"][1]["pbrMetallicRoughness"]["baseColorTexture"].update(index=7),
            "texture index 7",
        ),
        (
            lambda d: d["materials"][1].update(pbrMetallicRoughness=[]),
            "pbrMetallicRoughness must be an object",
        ),
        (
            lambda d: d["meshes"][0]["primitives"][0]["attributes"].pop("TEXCOORD_0"),
            "need NORMAL and TEXCOORD_0",
        ),
        (
            lambda d: d["meshes"][0]["primitives"][0]["attributes"].pop("NORMAL"),
            "need NORMAL and TEXCOORD_0",
        ),
        (
            lambda d: d["meshes"][1]["primitives"][0]["attributes"].update(TEXCOORD_0=0),
            "accessor 0 must be VEC2",
        ),
        (lambda d: d["accessors"][6].update(count=3), "count differs"),
    ],
)
def test_inspect_rejects_unsupported_glb(mutate, message):
    data = _repack(write_textured_glb(_two_quads()), mutate)
    with pytest.raises(GlbError, match=message):
        inspect_textured_glb(data)


# Malformed geometry must fail as GlbError in both readers, never as KeyError/IndexError/etc.
GEOMETRY_MUTATIONS = [
    (
        lambda d: d["meshes"][0]["primitives"][0]["attributes"].pop("POSITION"),
        "no POSITION attribute",
    ),
    (
        lambda d: d["meshes"][0]["primitives"][0]["attributes"].update(POSITION=99),
        "accessor index 99",
    ),
    (lambda d: d["meshes"][0]["primitives"][0].update(indices=99), "accessor index 99"),
    (lambda d: d["nodes"][0].update(mesh=42), "mesh index 42"),
    (lambda d: d["meshes"][0].update(primitives=[5]), "primitives must be an array of objects"),
    (lambda d: d["meshes"][0]["primitives"][0].update(mode=1), "only triangle"),
    (lambda d: d["accessors"][0].update(componentType=5130), "componentType 5130"),
    (lambda d: d["accessors"][0].update(componentType=5125), "componentType 5125"),
    (lambda d: d["accessors"][0].update(type="MAT4"), "must be VEC3"),
    (lambda d: d["accessors"][3].update(type="VEC2"), "must be SCALAR"),
    (lambda d: d["accessors"][0].pop("count"), "count is missing"),
    (lambda d: d["accessors"][0].update(count=1.5), "count must be a non-negative integer"),
    (lambda d: d["accessors"][0].update(count=10**9), "accessor out of range"),
    (lambda d: d["accessors"][0].update(bufferView=99), "bufferView index 99"),
    (lambda d: d["accessors"][0].pop("bufferView"), "bufferView is missing"),
    (lambda d: d["bufferViews"][0].update(byteLength="8"), "byteLength must be"),
    (lambda d: d["nodes"][0].update(translation=[0, 1]), "malformed translation"),
    (lambda d: d["nodes"][0].update(translation=[0, 1, 0]), "non-identity"),
    (lambda d: d.update(nodes=5), "nodes must be an array"),
]


@pytest.mark.parametrize("mutate, message", GEOMETRY_MUTATIONS)
def test_read_and_inspect_reject_malformed_geometry(mutate, message):
    data = _repack(write_textured_glb(_two_quads()), mutate)
    with pytest.raises(GlbError, match=message):
        read_glb(data)
    with pytest.raises(GlbError, match=message):
        inspect_textured_glb(data)


@pytest.mark.parametrize(
    "body, message",
    [
        (b"{not json", "not valid"),
        (b'{"asset": NaN}', "non-finite JSON literal NaN"),
        (b"[]", "must be an object"),
    ],
)
def test_rejects_a_bad_json_chunk(body, message):
    data = write_textured_glb(_two_quads())
    framed = _frame(body, _split_glb(data)[1])
    with pytest.raises(GlbError, match=message):
        read_glb(framed)
    with pytest.raises(GlbError, match=message):
        inspect_textured_glb(framed)


@pytest.mark.parametrize(
    "accessor, element, component, value, message",
    [
        (2, 0, 0, math.nan, "TEXCOORD_0 values must be finite"),  # part 0 UVs
        (6, 1, 1, math.inf, "TEXCOORD_0 values must be finite"),  # part 1 UVs
        (1, 0, 1, math.nan, "NORMAL values must be finite"),
        (0, 0, 1, math.nan, "POSITION values must be finite"),
        (4, 2, 2, -math.inf, "POSITION values must be finite"),
    ],
)
def test_inspect_rejects_non_finite_attributes(accessor, element, component, value, message):
    data = _patch_float(write_textured_glb(_two_quads()), accessor, element, component, value)
    with pytest.raises(GlbError, match=message):
        inspect_textured_glb(data)


def test_inspect_rejects_non_finite_positions_on_a_primitive_without_material():
    data = _patch_float(write_textured_glb(_two_quads()), 0, 0, 1, math.nan)

    def strip_materials(doc):
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                prim.pop("material")

    with pytest.raises(GlbError, match="non-finite positions"):
        inspect_textured_glb(_repack(data, strip_materials))


def test_inspect_rejects_uvs_outside_the_unit_square():
    data = _patch_float(write_textured_glb(_two_quads()), 2, 0, 0, 2.0)
    with pytest.raises(GlbError, match="must lie in"):
        inspect_textured_glb(data)


def test_inspect_rejects_a_textured_glb_without_triangles():
    def empty_indices(doc):
        for index in (3, 7):
            doc["accessors"][index]["count"] = 0

    with pytest.raises(GlbError, match="no triangles"):
        inspect_textured_glb(_repack(write_textured_glb(_two_quads()), empty_indices))


def test_unlit_is_false_when_there_are_no_materials():
    def strip(doc):
        for mesh in doc["meshes"]:
            for prim in mesh["primitives"]:
                prim.pop("material")
        doc.update(materials=[], textures=[], samplers=[], images=[], extensionsUsed=[])

    summary = inspect_textured_glb(_repack(write_textured_glb(_two_quads()), strip))
    assert summary["materials"] == 0
    assert summary["images"] == []
    assert summary["unlit"] is False


def test_unlit_needs_every_material_to_be_unlit():
    def unlit_only_the_first(doc):
        doc["materials"][1]["extensions"] = {}

    summary = inspect_textured_glb(_repack(write_textured_glb(_two_quads()), unlit_only_the_first))
    assert summary["materials"] == 2
    assert summary["unlit"] is False


@pytest.mark.parametrize("stride, message", [(12, None), (16, "interleaved buffers")])
def test_byte_stride_must_match_the_element_size(stride, message):
    def set_stride(doc):
        view = doc["bufferViews"][doc["accessors"][0]["bufferView"]]
        view["byteStride"] = stride

    data = _repack(write_textured_glb(_two_quads()), set_stride)
    if message is None:
        assert [node.name for node in read_glb(data)[1]] == ["floor-0", "wall-0"]
        assert inspect_textured_glb(data)["vertices"] == 8
    else:
        with pytest.raises(GlbError, match=message):
            read_glb(data)
        with pytest.raises(GlbError, match=message):
            inspect_textured_glb(data)


def test_inspect_rejects_an_image_bufferview_past_the_bin_chunk():
    def overrun(doc):
        doc["bufferViews"][doc["images"][0]["bufferView"]]["byteLength"] = 10**7

    with pytest.raises(GlbError, match="bufferView out of range"):
        inspect_textured_glb(_repack(write_textured_glb(_two_quads()), overrun))


def test_a_glb_without_a_bin_chunk_cannot_supply_accessor_data():
    doc, _ = _split_glb(write_textured_glb(_two_quads()))
    body = json.dumps(doc, separators=(",", ":"), sort_keys=True).encode()
    body += b" " * ((4 - len(body) % 4) % 4)
    data = struct.pack("<III", GLB_MAGIC, 2, 20 + len(body))
    data += struct.pack("<II", len(body), glb_module.CHUNK_JSON) + body
    with pytest.raises(GlbError, match="out of range"):
        read_glb(data)
    with pytest.raises(GlbError, match="out of range"):
        inspect_textured_glb(data)


@pytest.mark.parametrize("extra", [1, 4, 7])
def test_trailing_bytes_that_do_not_form_a_chunk_are_rejected(extra):
    data = write_textured_glb(_two_quads())
    tail = b"\x00" * extra
    for base in (_without_bin(data), data):
        bad = _with_trailing(base, tail)
        with pytest.raises(GlbError, match="trailing bytes"):
            read_glb(bad)
        with pytest.raises(GlbError, match="trailing bytes"):
            inspect_textured_glb(bad)


def test_a_complete_chunk_of_unknown_type_after_bin_is_ignored():
    data = write_textured_glb(_two_quads())
    unknown = struct.pack("<I", 8) + b"XTRA" + b"\x01" * 8
    assert [node.name for node in read_glb(_with_trailing(data, unknown))[1]] == [
        "floor-0",
        "wall-0",
    ]


@pytest.mark.parametrize("delta, accepted", [(-3, True), (0, True), (-4, False), (1, False)])
def test_bin_chunk_may_pad_buffer_byte_length_by_up_to_three_bytes(delta, accepted):
    def shift_length(doc):
        doc["buffers"][0]["byteLength"] += delta

    data = _repack(write_textured_glb(_two_quads()), shift_length)
    if accepted:
        assert inspect_textured_glb(data)["triangles"] == 4
    else:
        with pytest.raises(GlbError, match="byteLength"):
            read_glb(data)


def test_a_deeply_nested_json_chunk_is_a_glb_error_not_a_crash():
    body = b"[" * 200_000 + b"]" * 200_000
    framed = _frame(body, _split_glb(write_textured_glb(_two_quads()))[1])
    with pytest.raises(GlbError, match="nested too deeply"):
        read_glb(framed)
    with pytest.raises(GlbError, match="nested too deeply"):
        inspect_textured_glb(framed)
    findings = check_collision_glb(framed, _manifest())
    assert [f.code for f in findings] == ["COLLISION_GLB"]
    assert "nested too deeply" in findings[0].message


def test_instanced_nodes_share_one_decode_per_accessor(monkeypatch):
    positions, indices = _grid_mesh(20)
    data = _instanced_glb(positions, indices, copies=200)
    decoded: list[int] = []
    original = glb_module._read_accessor

    def counting(doc, bin_chunk, index, accessor_type, components):
        decoded.append(index)
        return original(doc, bin_chunk, index, accessor_type, components)

    monkeypatch.setattr(glb_module, "_read_accessor", counting)
    _, nodes = read_glb(data)
    assert len(nodes) == 200
    assert sorted(decoded) == [0, 1]  # one POSITION and one index accessor, however many nodes
    assert all(node.positions is nodes[0].positions for node in nodes)
    assert all(node.triangles is nodes[0].triangles for node in nodes)


def test_triangle_total_is_capped_across_instanced_nodes():
    # 2 * 142**2 = 40,328 triangles per copy; 60 copies is about 2.4 million, over the cap.
    data = _instanced_glb(*_grid_mesh(142), copies=60)
    tracemalloc.start()
    try:
        with pytest.raises(GlbError, match="triangles"):
            read_glb(data)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Per-node copies peaked at about 84 MB here; the decode-once reader stays far below.
    assert peak < 32 * 1024 * 1024


def test_vertex_total_is_capped_across_instanced_nodes():
    count = 40_001  # 101 copies make 4,040,101 vertices, over the cap
    positions = np.column_stack([np.arange(count, dtype=np.float64), np.zeros((count, 2))])
    data = _instanced_glb(positions, np.array([[0, 1, 2]]), copies=101)
    with pytest.raises(GlbError, match="vertices"):
        read_glb(data)


def test_png_images_are_accepted_when_declared_as_png():
    mesh = _two_quads()
    data = _with_image(write_textured_glb(mesh), 0, _png_bytes(mesh.pages[0]), "image/png")
    summary = inspect_textured_glb(data)
    assert summary["images"] == [[8, 8], [16, 4]]
    assert np.array_equal(_decode_rgb(_image_bytes(data, 0)), mesh.pages[0].astype(np.float64))


@pytest.mark.parametrize(
    "raw_factory, mime, message",
    [
        (lambda: _png_bytes(_pages()[0]), "image/jpeg", "image data is PNG, not image/jpeg"),
        (
            lambda: _image_bytes(write_textured_glb(_two_quads()), 0),
            "image/png",
            "image data is JPEG, not image/png",
        ),
        (lambda: b"not an image at all", "image/jpeg", "cannot be decoded"),
    ],
)
def test_image_bytes_must_match_their_declared_type(raw_factory, mime, message):
    data = _with_image(write_textured_glb(_two_quads()), 0, raw_factory(), mime)
    with pytest.raises(GlbError, match=message):
        inspect_textured_glb(data)


def test_collision_glb_with_non_vec3_positions_is_a_finding():
    layout = _layout()
    data = write_glb(collision_parts(layout))
    assert check_collision_glb(data, _manifest()) == []
    patched = _repack(data, lambda d: d["accessors"][0].update(type="VEC2"))
    with pytest.raises(GlbError, match="must be VEC3"):
        read_glb(patched)
    findings = check_collision_glb(patched, _manifest())
    assert [f.code for f in findings] == ["COLLISION_GLB"]
    assert "must be VEC3" in findings[0].message


def test_collision_glb_with_non_scalar_indices_is_a_finding():
    data = write_glb(collision_parts(_layout()))
    patched = _repack(data, lambda d: d["accessors"][1].update(type="VEC2"))
    findings = check_collision_glb(patched, _manifest())
    assert [f.code for f in findings] == ["COLLISION_GLB"]
    assert "must be SCALAR" in findings[0].message


def _furnished_layout() -> dict:
    """A 7 x 7 m room with a sofa: a seat on the floor, a raised back and a cylinder leg."""
    layout = rectangle_layout((-3.0, -3.0), (4.0, 4.0), elevation=0.0, height=2.5)
    sofa = {
        "id": "sofa",
        "floorId": "floor-0",
        "category": "sofa",
        "position": [1.0, 2.0],
        "yawRadians": 0.3,
        "size": [2.0, 0.9, 0.9],
        "solid": True,
        "parts": [
            {"shape": "box", "min": [0, 0, 0], "max": [1, 0.4, 1], "color": "#7a4a2a"},
            {"shape": "box", "min": [0, 0.4, 0], "max": [1, 1, 0.15], "color": "#6a3a20"},
            {"shape": "cylinder", "min": [0.02, 0, 0.02], "max": [0.08, 0.1, 0.08],
             "color": "#222222"},
        ],
    }  # fmt: skip
    layout["furniture"] = [sofa]
    return layout


def test_a_furnished_model_is_an_unlit_glb_with_its_parts_in_colour():
    layout = _furnished_layout()
    mesh = procedural_texture_layout(layout, TextureParams(texel_size=0.02, page_size=512))
    data = write_textured_glb(mesh)
    summary = inspect_textured_glb(data)
    assert summary["unlit"] and summary["triangles"] > 0
    doc = _doc(data)
    back = next(node for node in doc["nodes"] if node["name"] == "sofa:p1:top")
    assert back["extras"] == {
        "kind": "furniture",
        "furnitureId": "sofa",
        "partIndex": 1,
        "color": [106, 58, 32],
    }
    kinds = {node["extras"]["kind"] for node in doc["nodes"]}
    assert {"furniture", "floor", "wall"} <= kinds
    assert doc["extras"]["andarilho"]["coverage"]["furniture"] == 1.0


def test_a_furnished_model_is_a_valid_render_asset_inside_its_bounds():
    layout = _furnished_layout()
    data = write_textured_glb(procedural_texture_layout(layout))
    assert check_render_asset(data, "glb", {"bounds": layout_bounds(layout)}) == []
