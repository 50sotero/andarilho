"""3D Gaussian splat PLY (binary little-endian) reading, cleaning and writing."""

from __future__ import annotations

import mmap
from collections.abc import Buffer, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REQUIRED = (
    "x", "y", "z",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
)  # fmt: skip
SH_C0 = 0.28209479177387814
MAX_HEADER_BYTES = 64 * 1024
MAX_SPLATS = 20_000_000
_TYPES = {
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "short": "<i2", "int16": "<i2", "ushort": "<u2", "uint16": "<u2",
    "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4",
}  # fmt: skip


class PlyError(ValueError):
    pass


@dataclass
class SplatCloud:
    data: np.ndarray  # structured array, one record per splat

    @property
    def count(self) -> int:
        return int(len(self.data))

    def positions(self) -> np.ndarray:
        return np.column_stack([self.data["x"], self.data["y"], self.data["z"]]).astype(np.float64)


def parse_header(blob: Buffer) -> tuple[int, np.dtype, int]:
    head = bytes(blob[:MAX_HEADER_BYTES])  # works for bytes and memory maps alike
    end = head.find(b"end_header\n")
    if not head.startswith(b"ply\n") or end < 0:
        raise PlyError("not a PLY file")
    header = head[:end].decode("ascii", errors="strict").splitlines()
    data_offset = end + len(b"end_header\n")
    fmt = None
    elements: list[tuple[str, int, list[tuple[str, str]]]] = []
    for line in header[1:]:
        parts = line.split()
        if not parts or parts[0] in ("comment", "obj_info"):
            continue
        if parts[0] == "format":
            fmt = parts[1]
        elif parts[0] == "element":
            elements.append((parts[1], int(parts[2]), []))
        elif parts[0] == "property":
            if not elements:
                raise PlyError("property before element")
            if parts[1] == "list":
                raise PlyError("list properties are not supported")
            if parts[1] not in _TYPES:
                raise PlyError(f"unsupported property type {parts[1]}")
            elements[-1][2].append((parts[2], _TYPES[parts[1]]))
    if fmt != "binary_little_endian":
        raise PlyError("only binary_little_endian PLY is supported")
    if not elements or elements[0][0] != "vertex":
        raise PlyError("first element must be vertex")
    name, count, props = elements[0]
    if count <= 0 or count > MAX_SPLATS:
        raise PlyError("splat count out of range")
    dtype = np.dtype(props)
    missing = [p for p in REQUIRED if p not in dtype.names]
    if missing:
        raise PlyError(f"missing splat properties: {', '.join(missing)}")
    return count, dtype, data_offset


def read_splat_ply(blob: Buffer) -> SplatCloud:
    count, dtype, offset = parse_header(blob)
    need = offset + count * dtype.itemsize
    if len(blob) < need:
        raise PlyError("PLY truncated")
    return SplatCloud(np.frombuffer(blob, dtype=dtype, count=count, offset=offset).copy())


def read_splat_ply_file(path: Path) -> SplatCloud:
    """Parse a PLY on disk through a memory map, so large exports are never read inline."""
    if path.stat().st_size == 0:
        raise PlyError("not a PLY file")  # an empty file cannot be memory-mapped
    with path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
        return read_splat_ply(mapped)


def _ply_header(cloud: SplatCloud, comments: Sequence[str]) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {cloud.count}")
    inverse = {"<f4": "float", "<f8": "double", "|u1": "uchar", "|i1": "char",
               "<i2": "short", "<u2": "ushort", "<i4": "int", "<u4": "uint"}  # fmt: skip
    for name in cloud.data.dtype.names:
        kind = cloud.data.dtype[name].newbyteorder("<").str
        lines.append(f"property {inverse[kind]} {name}")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def _little_endian(cloud: SplatCloud) -> np.ndarray:
    return cloud.data.astype(cloud.data.dtype.newbyteorder("<"), copy=False)


def write_splat_ply(cloud: SplatCloud, *, comments: Sequence[str] = ()) -> bytes:
    return _ply_header(cloud, comments) + _little_endian(cloud).tobytes()


def write_splat_ply_file(cloud: SplatCloud, path: Path, *, comments: Sequence[str] = ()) -> int:
    """Write a PLY straight to disk (no in-memory copy of a large cloud); return its size."""
    with path.open("wb") as fh:
        fh.write(_ply_header(cloud, comments))
        _little_endian(cloud).tofile(fh)
    return path.stat().st_size


def drop_non_finite(cloud: SplatCloud) -> tuple[SplatCloud, int]:
    mask = np.ones(cloud.count, dtype=bool)
    for name in cloud.data.dtype.names:
        column = cloud.data[name]
        if np.issubdtype(column.dtype, np.floating):
            mask &= np.isfinite(column)
    return SplatCloud(cloud.data[mask]), int((~mask).sum())


def recenter(cloud: SplatCloud) -> tuple[SplatCloud, list[float]]:
    """Translate splats so their median position is at the origin (baked into data)."""
    center = np.median(cloud.positions(), axis=0)
    data = cloud.data.copy()
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = data[name] - center[axis]
    return SplatCloud(data), [float(v) for v in center]


def splat_dtype() -> np.dtype:
    return np.dtype([(name, "<f4") for name in REQUIRED])
