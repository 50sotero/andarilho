"""Minimal deterministic glTF 2.0 binary (GLB) writer and reader.

Collision geometry needs only one mesh per node, float32 VEC3 positions, uint32
triangle indices and no materials. The textured writer (``write_textured_glb``)
adds float32 normals and UVs and one unlit JPEG material per atlas page. Nodes
never carry a transform, so every node transform is identity and vertex
positions are canonical world coordinates.
"""

from __future__ import annotations

import io
import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any, NoReturn

import numpy as np
from PIL import Image

from andarilho_worker.geometry.textured import TexturedMesh, TexturedPart

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT = 5126
UINT32 = 5125
UINT16 = 5123
UINT8 = 5121
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
MODE_TRIANGLES = 4
LINEAR = 9729
LINEAR_MIPMAP_LINEAR = 9987
CLAMP_TO_EDGE = 33071
IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
MAX_GLB_BYTES = 64 * 1024 * 1024
# Totals over every node, so instanced meshes count once per node that references them.
MAX_GLB_TRIANGLES = 2_000_000
MAX_GLB_VERTICES = 4_000_000
MAX_TEXTURE_SIDE = 8192
UV_TOLERANCE = 1e-6
NORMAL_TOLERANCE = 1e-3
# A face whose geometric normal has a cosine below this with its vertex normals is wound the
# wrong way (-0.5 is 120 degrees apart); degenerate faces have no direction and are not judged.
WINDING_COSINE = -0.5
# A face is degenerate, and not judged, when |cross| <= DEGENERATE_AREA * L * max(L, C), where L
# is its longest edge and C its largest absolute coordinate. Float32 rounding of collinear vertices
# stays within about 4e-7 of that bound, so this margin covers it.
DEGENERATE_AREA = 1e-6
F32_MAX = float(np.finfo(np.float32).max)
JPEG_MIME = "image/jpeg"
PNG_MIME = "image/png"
IMAGE_FORMATS = {JPEG_MIME: "JPEG", PNG_MIME: "PNG"}
UNLIT_EXTENSION = "KHR_materials_unlit"


@dataclass
class MeshPart:
    name: str
    kind: str
    positions: np.ndarray  # (N, 3) float
    indices: np.ndarray  # (M, 3) int
    extras: dict[str, Any] = field(default_factory=dict)


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def _frame(json_bytes: bytes, bin_bytes: bytes) -> bytes:
    """Frame a JSON document and one BIN chunk as a glTF 2.0 container (chunks padded)."""
    json_bytes = _pad(json_bytes, b" ")
    bin_bytes = _pad(bin_bytes, b"\x00")
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(json_bytes), CHUNK_JSON) + json_bytes
    out += struct.pack("<II", len(bin_bytes), CHUNK_BIN) + bin_bytes
    return bytes(out)


def _pack_glb(doc: dict[str, Any], bin_blob: bytes) -> bytes:
    """Frame a JSON document and one BIN chunk as a glTF 2.0 container."""
    json_bytes = json.dumps(doc, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return _frame(json_bytes.encode(), bin_blob)


def _positions_f32(name: str, positions: Any) -> np.ndarray:
    """Return (N, 3) positions as float32, checked in float64 so nothing overflows silently."""
    src = np.asarray(positions, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3 or len(src) == 0:
        raise ValueError(f"{name}: positions must be a non-empty (N, 3) array")
    if not np.all(np.isfinite(src)):
        raise ValueError(f"{name}: non-finite vertex")
    if np.any(np.abs(src) > F32_MAX):
        raise ValueError(f"{name}: vertex outside the float32 range")
    return np.ascontiguousarray(src, dtype="<f4")


def _finite_number(name: str, value: Any) -> float:
    """A finite real number as a float; anything else is a ValueError that names ``name``."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise ValueError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def write_glb(parts: list[MeshPart], *, generator: str = "andarilho-worker") -> bytes:
    bin_blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for part in parts:
        pos = _positions_f32(part.name, part.positions)
        idx = np.ascontiguousarray(part.indices, dtype="<u4")
        if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0 or int(idx.max()) >= len(pos):
            raise ValueError(f"{part.name}: invalid triangle indices")
        pos_offset = len(bin_blob)
        bin_blob += pos.tobytes()
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": pos_offset,
                "byteLength": pos.nbytes,
                "target": ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": FLOAT,
                "count": len(pos),
                "type": "VEC3",
                "min": [float(v) for v in pos.min(axis=0)],
                "max": [float(v) for v in pos.max(axis=0)],
            }
        )
        pos_accessor = len(accessors) - 1
        idx_offset = len(bin_blob)
        bin_blob += idx.tobytes()
        buffer_views.append(
            {
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx.nbytes,
                "target": ELEMENT_ARRAY_BUFFER,
            }
        )
        accessors.append(
            {
                "bufferView": len(buffer_views) - 1,
                "componentType": UINT32,
                "count": int(idx.size),
                "type": "SCALAR",
            }
        )
        meshes.append(
            {
                "name": part.name,
                "primitives": [
                    {"attributes": {"POSITION": pos_accessor}, "indices": pos_accessor + 1}
                ],
            }
        )
        extras = {"kind": part.kind, **part.extras}
        nodes.append({"name": part.name, "mesh": len(meshes) - 1, "extras": extras})
    doc = {
        "asset": {"version": "2.0", "generator": generator},
        "extras": {"andarilho": {"geometrySpace": "world", "units": "m", "upAxis": "+Y"}},
        "scene": 0,
        "scenes": [{"name": "collision", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    return _pack_glb(doc, bytes(bin_blob))


def _check_page(index: int, page: np.ndarray) -> np.ndarray:
    arr = np.asarray(page)
    if arr.dtype != np.uint8 or arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(f"page {index} must be a uint8 (H, W, 3) array")
    height, width = arr.shape[:2]
    if not (1 <= height <= MAX_TEXTURE_SIDE and 1 <= width <= MAX_TEXTURE_SIDE):
        raise ValueError(
            f"page {index} is {width}x{height}; each side must be 1..{MAX_TEXTURE_SIDE}"
        )
    return arr


def _check_winding(name: str, pos: np.ndarray, nrm: np.ndarray, idx: np.ndarray) -> None:
    """Reject faces whose counter-clockwise side faces away from their vertex normals.

    A face wound counter-clockwise seen from its visible side has a geometric normal (the cross
    product of two edges) that agrees with the normals of its vertices. Degenerate faces are not
    judged: their rounding noise can point either way.
    """
    corners = pos.astype(np.float64)[idx]  # (M, 3, 3)
    face = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    area = np.linalg.norm(face, axis=1)
    longest = np.linalg.norm(corners[:, [1, 2, 0]] - corners, axis=2).max(axis=1)
    magnitude = np.abs(pos).max(axis=1)[idx].max(axis=1)  # largest coordinate of each face
    judged = area > DEGENERATE_AREA * longest * np.maximum(longest, magnitude)
    mean = nrm.astype(np.float64)[idx].sum(axis=1)  # (M, 3)
    limit = WINDING_COSINE * area * np.linalg.norm(mean, axis=1)
    if np.any(judged & (np.einsum("ij,ij->i", face, mean) < limit)):
        raise ValueError(f"{name}: triangle winding disagrees with the vertex normals")


def _check_part(part: TexturedPart, page_count: int) -> tuple[np.ndarray, ...]:
    """Validate one part; return its float32 positions, normals, UVs, uint32 indices and page."""
    pos = _positions_f32(part.name, part.positions)
    src_nrm = np.asarray(part.normals, dtype=np.float64)
    if src_nrm.shape != pos.shape or not np.all(np.isfinite(src_nrm)):
        raise ValueError(f"{part.name}: normals must be finite and match the positions")
    if np.any(np.abs(np.linalg.norm(src_nrm, axis=1) - 1) > NORMAL_TOLERANCE):
        raise ValueError(f"{part.name}: normals must be unit length")
    src_uv = np.asarray(part.uvs, dtype=np.float64)
    if src_uv.shape != (len(pos), 2) or not np.all(np.isfinite(src_uv)):
        raise ValueError(f"{part.name}: uvs must be finite (N, 2) matching the positions")
    if np.any(src_uv < -UV_TOLERANCE) or np.any(src_uv > 1 + UV_TOLERANCE):
        raise ValueError(f"{part.name}: uvs must lie in [0, 1]")
    raw = np.asarray(part.indices)
    if (
        raw.ndim != 2
        or raw.shape[1] != 3
        or len(raw) == 0
        or not np.issubdtype(raw.dtype, np.integer)
    ):
        raise ValueError(f"{part.name}: indices must be a non-empty integer (M, 3) array")
    idx = raw.astype(np.int64)
    if int(idx.min()) < 0 or int(idx.max()) >= len(pos):
        raise ValueError(f"{part.name}: triangle indices out of range")
    page = part.page
    if (
        isinstance(page, bool)
        or not isinstance(page, int | np.integer)
        or not 0 <= page < page_count
    ):
        raise ValueError(f"{part.name}: page index {page!r} is out of range")
    _check_winding(part.name, pos, src_nrm, idx)
    return (
        pos,
        np.ascontiguousarray(src_nrm, dtype="<f4"),
        np.ascontiguousarray(src_uv, dtype="<f4"),
        np.ascontiguousarray(idx, dtype="<u4"),
        int(page),
    )


def _strip_jfif(raw: bytes) -> bytes:
    """Drop the JFIF APP0 segment that Pillow always writes first; JPEG does not require it.

    Without it the file has no application segment at all. Decoders then treat the three
    components as YCbCr, which is what Pillow encodes.
    """
    if raw[2:4] != b"\xff\xe0" or raw[6:11] != b"JFIF\x00":
        return raw
    (length,) = struct.unpack_from(">H", raw, 4)
    return raw[:2] + raw[4 + length :]


def _encode_jpeg(page: np.ndarray, quality: int) -> bytes:
    """Baseline 4:2:0 JPEG with fixed encoder settings and no metadata (reproducible bytes)."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(page)).save(
        buf,
        format="JPEG",
        quality=quality,
        subsampling="4:2:0",
        optimize=False,
        progressive=False,
    )
    return _strip_jfif(buf.getvalue())


def _add_view(
    bin_blob: bytearray, views: list[dict[str, Any]], data: bytes, target: int | None
) -> int:
    """Append ``data`` to the BIN chunk as a bufferView that starts on a 4-byte boundary."""
    view: dict[str, Any] = {"buffer": 0, "byteOffset": len(bin_blob), "byteLength": len(data)}
    if target is not None:
        view["target"] = target
    views.append(view)
    bin_blob += _pad(data, b"\x00")
    return len(views) - 1


def write_textured_glb(
    mesh: TexturedMesh, *, jpeg_quality: int = 85, generator: str = "andarilho-worker"
) -> bytes:
    """Serialize a textured layout: UV'd triangles, one unlit JPEG material per atlas page.

    Normals must already be unit length (within ``NORMAL_TOLERANCE``). The writer validates
    them instead of renormalizing, so a producer bug is reported rather than hidden.
    """
    if not mesh.parts:
        raise ValueError("textured mesh has no parts")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    coverage: dict[str, float] = {}
    for key, value in mesh.coverage.items():
        fraction = _finite_number(f"coverage[{key!r}]", value)
        if not 0 <= fraction <= 1:
            raise ValueError(f"coverage[{key!r}] must lie in [0, 1]")
        coverage[str(key)] = fraction
    texel_size = _finite_number("texel_size", mesh.texel_size)
    if texel_size <= 0:
        raise ValueError("texel_size must be positive")
    pages = [_check_page(i, page) for i, page in enumerate(mesh.pages)]
    bin_blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for part in mesh.parts:
        pos, nrm, uvs, idx, page = _check_part(part, len(pages))
        first = len(accessors)
        pos_view = _add_view(bin_blob, buffer_views, pos.tobytes(), ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": pos_view,
                "componentType": FLOAT,
                "count": len(pos),
                "type": "VEC3",
                "min": [float(v) for v in pos.min(axis=0)],
                "max": [float(v) for v in pos.max(axis=0)],
            }
        )
        nrm_view = _add_view(bin_blob, buffer_views, nrm.tobytes(), ARRAY_BUFFER)
        accessors.append(
            {"bufferView": nrm_view, "componentType": FLOAT, "count": len(nrm), "type": "VEC3"}
        )
        uv_view = _add_view(bin_blob, buffer_views, uvs.tobytes(), ARRAY_BUFFER)
        accessors.append(
            {"bufferView": uv_view, "componentType": FLOAT, "count": len(uvs), "type": "VEC2"}
        )
        idx_view = _add_view(bin_blob, buffer_views, idx.tobytes(), ELEMENT_ARRAY_BUFFER)
        accessors.append(
            {
                "bufferView": idx_view,
                "componentType": UINT32,
                "count": int(idx.size),
                "type": "SCALAR",
            }
        )
        meshes.append(
            {
                "name": part.name,
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": first,
                            "NORMAL": first + 1,
                            "TEXCOORD_0": first + 2,
                        },
                        "indices": first + 3,
                        "material": page,
                        "mode": MODE_TRIANGLES,
                    }
                ],
            }
        )
        extras = {"kind": part.kind, **part.extras}
        nodes.append({"name": part.name, "mesh": len(meshes) - 1, "extras": extras})
    images: list[dict[str, Any]] = []
    textures: list[dict[str, Any]] = []
    materials: list[dict[str, Any]] = []
    for i, page_pixels in enumerate(pages):
        view = _add_view(bin_blob, buffer_views, _encode_jpeg(page_pixels, jpeg_quality), None)
        images.append({"bufferView": view, "mimeType": JPEG_MIME})
        textures.append({"sampler": 0, "source": i})
        materials.append(
            {
                "name": f"atlas-{i}",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": i},
                    "metallicFactor": 0,
                    "roughnessFactor": 1,
                },
                "extensions": {UNLIT_EXTENSION: {}},
            }
        )
    doc = {
        "asset": {"version": "2.0", "generator": generator},
        "extensionsUsed": [UNLIT_EXTENSION],
        "extras": {
            "andarilho": {
                "geometrySpace": "world",
                "units": "m",
                "upAxis": "+Y",
                "coverage": coverage,
                "texelSize": texel_size,
            }
        },
        "scene": 0,
        "scenes": [{"name": "visual", "nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "textures": textures,
        "samplers": [
            {
                "magFilter": LINEAR,
                "minFilter": LINEAR_MIPMAP_LINEAR,
                "wrapS": CLAMP_TO_EDGE,
                "wrapT": CLAMP_TO_EDGE,
            }
        ],
        "images": images,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": len(bin_blob)}],
    }
    glb = _pack_glb(doc, bytes(bin_blob))
    if len(glb) > MAX_GLB_BYTES:
        raise ValueError(f"textured GLB is {len(glb)} bytes; the limit is {MAX_GLB_BYTES}")
    return glb


class GlbError(ValueError):
    pass


@dataclass
class GlbNode:
    name: str
    kind: str
    positions: np.ndarray
    triangles: np.ndarray  # (M, 3) int indices into positions


# glTF componentType codes the reader accepts, with the numpy dtype and component count of each.
_COMPONENT_DTYPES = {FLOAT: "<f4", UINT32: "<u4", UINT16: "<u2", UINT8: "u1"}
_COMPONENT_COUNTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
FLOAT_COMPONENTS = (FLOAT,)
INDEX_COMPONENTS = (UINT8, UINT16, UINT32)


def _reject_constant(token: str) -> NoReturn:
    raise GlbError(f"non-finite JSON literal {token}")


def _chunk_header(data: bytes, offset: int) -> tuple[int, int]:
    """The (length, type) of the chunk header at ``offset``; fewer than 8 bytes is a GlbError."""
    if offset + 8 > len(data):
        raise GlbError("trailing bytes do not form a chunk")
    return struct.unpack_from("<II", data, offset)


def _split_frame(data: bytes) -> tuple[bytes, bytes]:
    """Check the GLB framing; return the raw JSON chunk and the BIN chunk (empty if absent).

    Chunks after the BIN chunk have unknown types, which a reader ignores, but each must still be
    whole: a header and its payload that run past the end of the file are an error.
    """
    if len(data) < 20 or len(data) > MAX_GLB_BYTES:
        raise GlbError("GLB size out of range")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or length != len(data):
        raise GlbError("not a glTF 2.0 binary")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != CHUNK_JSON or 20 + json_len > len(data):
        raise GlbError("missing JSON chunk")
    offset = 20 + json_len
    bin_chunk = b""
    if offset < len(data):
        bin_len, bin_type = _chunk_header(data, offset)
        if bin_type != CHUNK_BIN or offset + 8 + bin_len > len(data):
            raise GlbError("invalid BIN chunk")
        bin_chunk = data[offset + 8 : offset + 8 + bin_len]
        offset += 8 + bin_len
    while offset < len(data):
        chunk_len, _ = _chunk_header(data, offset)
        if offset + 8 + chunk_len > len(data):
            raise GlbError("invalid chunk: its length runs past the end of the GLB")
        offset += 8 + chunk_len
    return data[20 : 20 + json_len], bin_chunk


def _check_bin_length(doc: dict[str, Any], bin_chunk: bytes) -> None:
    """buffers[0] must describe the BIN chunk; the writer pads the chunk to 4 bytes, so up to 3."""
    buffers = _objects(doc, "buffers")
    if not buffers:
        return
    declared = _uint(buffers[0], "byteLength")
    if not 0 <= len(bin_chunk) - declared <= 3:
        raise GlbError(f"BIN chunk is {len(bin_chunk)} bytes; buffers[0].byteLength is {declared}")


def _split_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Check the GLB framing; return the JSON document and the BIN chunk (empty if absent)."""
    json_bytes, bin_chunk = _split_frame(data)
    try:
        doc = json.loads(json_bytes, parse_constant=_reject_constant)
    except RecursionError as exc:
        raise GlbError("JSON chunk is nested too deeply") from exc
    except ValueError as exc:
        raise GlbError(f"JSON chunk is not valid: {exc}") from exc
    if not isinstance(doc, dict):
        raise GlbError("JSON chunk must be an object")
    if bin_chunk:
        _check_bin_length(doc, bin_chunk)
    return doc, bin_chunk


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _objects(parent: dict[str, Any], key: str) -> list[dict[str, Any]]:
    """A glTF array of objects; a missing array is empty and anything else is a GlbError."""
    value = parent.get(key, [])
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise GlbError(f"{key} must be an array of objects")
    return value


def _object(parent: dict[str, Any], key: str) -> dict[str, Any]:
    """A glTF object member; a missing object is empty and anything else is a GlbError."""
    value = parent.get(key, {})
    if not isinstance(value, dict):
        raise GlbError(f"{key} must be an object")
    return value


def _element(items: list[dict[str, Any]], index: Any, what: str) -> dict[str, Any]:
    """The object at ``index``; a missing, non-integer or out-of-range index is a GlbError."""
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(items):
        raise GlbError(f"{what} index {index!r} is out of range")
    return items[index]


def _uint(obj: dict[str, Any], key: str, default: int | None = None) -> int:
    """A non-negative integer member; a missing member takes ``default``, or is an error."""
    if key not in obj:
        if default is None:
            raise GlbError(f"{key} is missing")
        return default
    value = obj[key]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GlbError(f"{key} must be a non-negative integer")
    return value


def _all_close(values: list[Any], expected: tuple[float, ...]) -> bool:
    return all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(values, expected, strict=True))


def _vector(node: dict[str, Any], key: str, default: tuple[float, ...]) -> list[Any]:
    """A node's matrix or TRS member as a list of numbers of the right length."""
    value = node.get(key, list(default))
    if (
        not isinstance(value, list)
        or len(value) != len(default)
        or not all(_is_number(v) for v in value)
    ):
        raise GlbError(f"node has a malformed {key}")
    return value


def _node_is_identity(node: dict[str, Any]) -> bool:
    return (
        _all_close(_vector(node, "matrix", IDENTITY), IDENTITY)
        and _all_close(_vector(node, "translation", (0, 0, 0)), (0, 0, 0))
        and _all_close(_vector(node, "rotation", (0, 0, 0, 1)), (0, 0, 0, 1))
        and _all_close(_vector(node, "scale", (1, 1, 1)), (1, 1, 1))
    )


def _buffer_view(doc: dict[str, Any], index: Any) -> dict[str, Any]:
    """The bufferView at ``index``; views must point into the GLB's own BIN chunk."""
    view = _element(_objects(doc, "bufferViews"), index, "bufferView")
    if view.get("buffer", 0) != 0:
        raise GlbError("external buffers are not allowed")
    return view


def _read_accessor(
    doc: dict[str, Any],
    bin_chunk: bytes,
    index: Any,
    accessor_type: str,
    components: tuple[int, ...],
) -> np.ndarray:
    """Return accessor values as (count, n) for VEC2/VEC3 and (count,) for SCALAR.

    Only ``accessor_type`` with one of the allowed componentTypes is accepted, so every caller
    gets the layout it expects; anything else raises GlbError.
    """
    acc = _element(_objects(doc, "accessors"), index, "accessor")
    kind = acc.get("type")
    if kind != accessor_type:
        raise GlbError(f"accessor {index} must be {accessor_type}, not {kind!r}")
    component = acc.get("componentType")
    if component not in components:
        raise GlbError(f"accessor {index} has unsupported componentType {component!r}")
    view = _buffer_view(doc, _uint(acc, "bufferView"))
    count = _uint(acc, "count")
    comps = _COMPONENT_COUNTS[accessor_type]
    dtype = np.dtype(_COMPONENT_DTYPES[component])
    itemsize = dtype.itemsize * comps
    start = _uint(view, "byteOffset", 0) + _uint(acc, "byteOffset", 0)
    if view.get("byteStride") not in (None, itemsize):
        raise GlbError("interleaved buffers are not supported")
    end = start + itemsize * count
    if end > len(bin_chunk) or end > _uint(view, "byteOffset", 0) + _uint(view, "byteLength"):
        raise GlbError("accessor out of range")
    if count == 0:
        return np.empty((0, comps) if comps > 1 else (0,), dtype=dtype)
    arr = np.frombuffer(bin_chunk, dtype=dtype, count=count * comps, offset=start)
    return arr.reshape(-1, comps) if comps > 1 else arr


def _accessor_index(index: Any) -> int:
    """An accessor index as an int, so it can key a cache; anything else is out of range."""
    if isinstance(index, bool) or not isinstance(index, int):
        raise GlbError(f"accessor index {index!r} is out of range")
    return index


def _triangles(
    doc: dict[str, Any], bin_chunk: bytes, pos: np.ndarray, index: int | None
) -> np.ndarray:
    """Triangle indices (M, 3) into ``pos``: the index accessor's values, or vertices in order."""
    if index is None:
        idx = np.arange(len(pos), dtype=np.int64)
    else:
        idx = _read_accessor(doc, bin_chunk, index, "SCALAR", INDEX_COMPONENTS).astype(np.int64)
    if idx.size % 3 or (idx.size and int(idx.max()) >= len(pos)):
        raise GlbError("invalid indices")
    return idx.reshape(-1, 3)


def _read_nodes(doc: dict[str, Any], bin_chunk: bytes) -> list[GlbNode]:
    """Return the world-space triangle meshes of every node that references a mesh.

    Each accessor is decoded once and shared by every node that instances it, so memory follows
    the distinct data in the file. The triangle and vertex totals count every node, instances
    included, and stop a small file from asking for an unbounded amount of work.
    """
    nodes_out: list[GlbNode] = []
    meshes = _objects(doc, "meshes")
    positions: dict[int, np.ndarray] = {}  # float64 POSITION values by accessor index
    triangles: dict[tuple[int, int | None], np.ndarray] = {}  # by (POSITION, indices) accessors
    triangle_total = vertex_total = 0
    for node in _objects(doc, "nodes"):
        name = str(node.get("name", ""))
        if not _node_is_identity(node):
            raise GlbError(f"node {name!r} has a non-identity transform")
        if "mesh" not in node:
            continue
        mesh = _element(meshes, node["mesh"], "mesh")
        for prim in _objects(mesh, "primitives"):
            if prim.get("mode", 4) != 4:
                raise GlbError("only triangle primitives are supported")
            attrs = _object(prim, "attributes")
            if "POSITION" not in attrs:
                raise GlbError("primitive has no POSITION attribute")
            pos_index = _accessor_index(attrs["POSITION"])
            if pos_index not in positions:
                values = _read_accessor(doc, bin_chunk, pos_index, "VEC3", FLOAT_COMPONENTS)
                positions[pos_index] = values.astype(np.float64)
            pos = positions[pos_index]
            idx_index = _accessor_index(prim["indices"]) if "indices" in prim else None
            key = (pos_index, idx_index)
            if key not in triangles:
                triangles[key] = _triangles(doc, bin_chunk, pos, idx_index)
            tris = triangles[key]
            triangle_total += len(tris)
            vertex_total += len(pos)
            if triangle_total > MAX_GLB_TRIANGLES:
                raise GlbError(f"nodes reference more than {MAX_GLB_TRIANGLES} triangles in total")
            if vertex_total > MAX_GLB_VERTICES:
                raise GlbError(f"nodes reference more than {MAX_GLB_VERTICES} vertices in total")
            kind = str(_object(node, "extras").get("kind", "unknown"))
            nodes_out.append(GlbNode(name, kind, pos, tris))
    return nodes_out


def read_glb(data: bytes) -> tuple[dict[str, Any], list[GlbNode]]:
    """Parse a GLB, enforce identity node transforms, and return world-space meshes."""
    doc, bin_chunk = _split_glb(data)
    return doc, _read_nodes(doc, bin_chunk)


def _view_bytes(doc: dict[str, Any], bin_chunk: bytes, index: Any) -> bytes:
    view = _buffer_view(doc, index)
    start = _uint(view, "byteOffset", 0)
    end = start + _uint(view, "byteLength")
    if end > len(bin_chunk):
        raise GlbError("bufferView out of range")
    return bin_chunk[start:end]


def inspect_textured_glb(data: bytes) -> dict[str, Any]:
    """Summarize a textured GLB and reject anything the viewer must not load."""
    doc, nodes = read_glb(data)
    _, bin_chunk = _split_frame(data)
    if "extensionsRequired" in doc:
        raise GlbError("textured GLBs must not require extensions")
    used = doc.get("extensionsUsed", [])
    if not isinstance(used, list):
        raise GlbError("extensionsUsed must be an array")
    if any("uri" in buffer for buffer in _objects(doc, "buffers")):
        raise GlbError("external buffers are not allowed")
    materials = _objects(doc, "materials")
    textures = _objects(doc, "textures")
    images = _objects(doc, "images")
    samplers = _objects(doc, "samplers")
    for texture in textures:
        _element(images, texture.get("source"), "image")
        if "sampler" in texture:
            _element(samplers, texture["sampler"], "sampler")
    for material in materials:
        pbr = _object(material, "pbrMetallicRoughness")
        if "baseColorTexture" in pbr:
            _element(textures, _object(pbr, "baseColorTexture").get("index"), "texture")
        if UNLIT_EXTENSION in _object(material, "extensions") and UNLIT_EXTENSION not in used:
            raise GlbError(f"{UNLIT_EXTENSION} must be listed in extensionsUsed")
    for mesh in _objects(doc, "meshes"):
        for prim in _objects(mesh, "primitives"):
            if "material" not in prim:
                continue
            _element(materials, prim["material"], "material")
            attrs = _object(prim, "attributes")
            if "NORMAL" not in attrs or "TEXCOORD_0" not in attrs:
                raise GlbError("textured primitives need NORMAL and TEXCOORD_0")
            if "POSITION" not in attrs:
                raise GlbError("primitive has no POSITION attribute")
            positions = _read_accessor(doc, bin_chunk, attrs["POSITION"], "VEC3", FLOAT_COMPONENTS)
            normals = _read_accessor(doc, bin_chunk, attrs["NORMAL"], "VEC3", FLOAT_COMPONENTS)
            uvs = _read_accessor(doc, bin_chunk, attrs["TEXCOORD_0"], "VEC2", FLOAT_COMPONENTS)
            for key, values in (("POSITION", positions), ("NORMAL", normals), ("TEXCOORD_0", uvs)):
                if len(values) != len(positions):
                    raise GlbError(f"{key} count differs from POSITION")
                if not np.all(np.isfinite(values)):
                    raise GlbError(f"{key} values must be finite")
            if np.any(uvs < -UV_TOLERANCE) or np.any(uvs > 1 + UV_TOLERANCE):
                raise GlbError("TEXCOORD_0 must lie in [0, 1]")
    for node in nodes:
        if not np.all(np.isfinite(node.positions)):
            raise GlbError(f"node {node.name!r} has non-finite positions")
    sizes: list[list[int]] = []
    for image in images:
        if "uri" in image:
            raise GlbError("external images are not allowed")
        mime = image.get("mimeType")
        if not isinstance(mime, str) or mime not in IMAGE_FORMATS:
            raise GlbError(f"unsupported image type {mime!r}")
        raw = _view_bytes(doc, bin_chunk, image.get("bufferView"))
        try:
            with Image.open(io.BytesIO(raw)) as img:
                fmt, (width, height) = img.format, img.size
        except (OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
            raise GlbError("image bytes cannot be decoded") from exc
        if fmt != IMAGE_FORMATS[mime]:
            raise GlbError(f"image data is {fmt}, not {mime}")
        if not (1 <= width <= MAX_TEXTURE_SIDE and 1 <= height <= MAX_TEXTURE_SIDE):
            raise GlbError(f"image is {width}x{height}; each side must be 1..{MAX_TEXTURE_SIDE}")
        sizes.append([width, height])
    if not any(len(node.triangles) for node in nodes):
        raise GlbError("textured GLB has no triangles")
    # Bounds fold in node by node: stacking every node's vertices would copy them all again.
    low = np.full(3, np.inf)
    high = np.full(3, -np.inf)
    for node in nodes:
        if len(node.positions):
            low = np.minimum(low, node.positions.min(axis=0))
            high = np.maximum(high, node.positions.max(axis=0))
    return {
        "triangles": sum(len(node.triangles) for node in nodes),
        "vertices": sum(len(node.positions) for node in nodes),
        "images": sizes,
        "materials": len(materials),
        "bounds": {"min": [float(v) for v in low], "max": [float(v) for v in high]},
        # An empty material list is not "unlit": there is nothing to light or unlight.
        "unlit": bool(materials)
        and all(UNLIT_EXTENSION in _object(m, "extensions") for m in materials),
    }
