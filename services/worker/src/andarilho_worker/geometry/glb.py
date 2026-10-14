"""Minimal deterministic glTF 2.0 binary (GLB) writer and reader.

Only what collision geometry needs: one mesh per node, float32 VEC3 positions,
uint32 triangle indices, no materials. Nodes never carry a transform, so every
node transform is identity and vertex positions are canonical world coordinates.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

GLB_MAGIC = 0x46546C67  # "glTF"
CHUNK_JSON = 0x4E4F534A
CHUNK_BIN = 0x004E4942
FLOAT = 5126
UINT32 = 5125
UINT16 = 5123
UINT8 = 5121
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963
IDENTITY = (1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)
MAX_GLB_BYTES = 64 * 1024 * 1024


@dataclass
class MeshPart:
    name: str
    kind: str
    positions: np.ndarray  # (N, 3) float
    indices: np.ndarray  # (M, 3) int
    extras: dict[str, Any] = field(default_factory=dict)


def _pad(data: bytes, fill: bytes) -> bytes:
    return data + fill * ((4 - len(data) % 4) % 4)


def write_glb(parts: list[MeshPart], *, generator: str = "andarilho-worker") -> bytes:
    bin_blob = bytearray()
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    meshes: list[dict[str, Any]] = []
    nodes: list[dict[str, Any]] = []
    for part in parts:
        pos = np.ascontiguousarray(part.positions, dtype="<f4")
        idx = np.ascontiguousarray(part.indices, dtype="<u4")
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0:
            raise ValueError(f"{part.name}: positions must be a non-empty (N, 3) array")
        if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0 or int(idx.max()) >= len(pos):
            raise ValueError(f"{part.name}: invalid triangle indices")
        if not np.all(np.isfinite(pos)):
            raise ValueError(f"{part.name}: non-finite vertex")
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
    json_bytes = _pad(json.dumps(doc, separators=(",", ":"), sort_keys=True).encode(), b" ")
    bin_bytes = _pad(bytes(bin_blob), b"\x00")
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
    out = bytearray(struct.pack("<III", GLB_MAGIC, 2, total))
    out += struct.pack("<II", len(json_bytes), CHUNK_JSON) + json_bytes
    out += struct.pack("<II", len(bin_bytes), CHUNK_BIN) + bin_bytes
    return bytes(out)


class GlbError(ValueError):
    pass


@dataclass
class GlbNode:
    name: str
    kind: str
    positions: np.ndarray
    triangles: np.ndarray  # (M, 3) int indices into positions


def _node_is_identity(node: dict[str, Any]) -> bool:
    if "matrix" in node:
        m = node["matrix"]
        if len(m) != 16 or any(
            not math.isclose(a, b, abs_tol=1e-9) for a, b in zip(m, IDENTITY, strict=True)
        ):
            return False
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    return (
        all(math.isclose(v, 0, abs_tol=1e-9) for v in t)
        and all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(r, [0, 0, 0, 1], strict=True))
        and all(math.isclose(v, 1, abs_tol=1e-9) for v in s)
    )


def read_glb(data: bytes) -> tuple[dict[str, Any], list[GlbNode]]:
    """Parse a GLB, enforce identity node transforms, and return world-space meshes."""
    if len(data) < 20 or len(data) > MAX_GLB_BYTES:
        raise GlbError("GLB size out of range")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != GLB_MAGIC or version != 2 or length != len(data):
        raise GlbError("not a glTF 2.0 binary")
    json_len, json_type = struct.unpack_from("<II", data, 12)
    if json_type != CHUNK_JSON or 20 + json_len > len(data):
        raise GlbError("missing JSON chunk")
    doc = json.loads(data[20 : 20 + json_len])
    offset = 20 + json_len
    bin_chunk = b""
    if offset + 8 <= len(data):
        bin_len, bin_type = struct.unpack_from("<II", data, offset)
        if bin_type != CHUNK_BIN or offset + 8 + bin_len > len(data):
            raise GlbError("invalid BIN chunk")
        bin_chunk = data[offset + 8 : offset + 8 + bin_len]
    accessors = doc.get("accessors", [])
    views = doc.get("bufferViews", [])

    def read_accessor(index: int) -> np.ndarray:
        acc = accessors[index]
        view = views[acc["bufferView"]]
        if view.get("buffer", 0) != 0:
            raise GlbError("external buffers are not allowed")
        comps = {"SCALAR": 1, "VEC3": 3}[acc["type"]]
        dtype = {FLOAT: "<f4", UINT32: "<u4", UINT16: "<u2", UINT8: "u1"}[acc["componentType"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        stride = view.get("byteStride")
        itemsize = np.dtype(dtype).itemsize * comps
        if stride not in (None, itemsize):
            raise GlbError("interleaved buffers are not supported")
        end = start + itemsize * acc["count"]
        if end > len(bin_chunk) or end > view.get("byteOffset", 0) + view["byteLength"]:
            raise GlbError("accessor out of range")
        arr = np.frombuffer(bin_chunk, dtype=dtype, count=acc["count"] * comps, offset=start)
        return arr.reshape(-1, comps) if comps > 1 else arr

    nodes_out: list[GlbNode] = []
    all_nodes = doc.get("nodes", [])
    for node in all_nodes:
        name = str(node.get("name", ""))
        if not _node_is_identity(node):
            raise GlbError(f"node {name!r} has a non-identity transform")
        if "mesh" not in node:
            continue
        mesh = doc["meshes"][node["mesh"]]
        for prim in mesh.get("primitives", []):
            if prim.get("mode", 4) != 4:
                raise GlbError("only triangle primitives are supported")
            pos = read_accessor(prim["attributes"]["POSITION"]).astype(np.float64)
            if "indices" in prim:
                idx = read_accessor(prim["indices"]).astype(np.int64)
            else:
                idx = np.arange(len(pos), dtype=np.int64)
            if idx.size % 3 or (idx.size and int(idx.max()) >= len(pos)):
                raise GlbError("invalid indices")
            kind = str(node.get("extras", {}).get("kind", "unknown"))
            nodes_out.append(GlbNode(name, kind, pos, idx.reshape(-1, 3)))
    return doc, nodes_out
