"""3D Gaussian splat PLY (binary little-endian) reading, cleaning and writing."""

from __future__ import annotations

from collections.abc import Sequence
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
CHUNK_SPLATS = 1 << 20  # records per chunk when streaming a PLY on disk
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


def parse_header(blob: bytes) -> tuple[int, np.dtype, int]:
    try:
        return _parse_header(blob[:MAX_HEADER_BYTES])
    except PlyError:
        raise
    except (ValueError, IndexError) as exc:  # non-ASCII text, a bad count, a short line
        raise PlyError("malformed PLY header") from exc


def _parse_header(head: bytes) -> tuple[int, np.dtype, int]:
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


def read_splat_ply(blob: bytes) -> SplatCloud:
    """Parse a small in-memory PLY. Exports on disk go through the streaming functions below."""
    count, dtype, offset = parse_header(blob)
    need = offset + count * dtype.itemsize
    if len(blob) < need:
        raise PlyError("PLY truncated")
    return SplatCloud(np.frombuffer(blob, dtype=dtype, count=count, offset=offset).copy())


def _ply_header(dtype: np.dtype, count: int, comments: Sequence[str]) -> bytes:
    lines = ["ply", "format binary_little_endian 1.0"]
    lines += [f"comment {c}" for c in comments]
    lines.append(f"element vertex {count}")
    inverse = {"<f4": "float", "<f8": "double", "|u1": "uchar", "|i1": "char",
               "<i2": "short", "<u2": "ushort", "<i4": "int", "<u4": "uint"}  # fmt: skip
    for name in dtype.names:
        kind = dtype[name].newbyteorder("<").str
        lines.append(f"property {inverse[kind]} {name}")
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def write_splat_ply(cloud: SplatCloud, *, comments: Sequence[str] = ()) -> bytes:
    little = cloud.data.astype(cloud.data.dtype.newbyteorder("<"), copy=False)
    return _ply_header(cloud.data.dtype, cloud.count, comments) + little.tobytes()


def _finite(records: np.ndarray) -> np.ndarray:
    """Mask of records whose floating-point values are all finite."""
    mask = np.ones(len(records), dtype=bool)
    for name in records.dtype.names:
        column = records[name]
        if np.issubdtype(column.dtype, np.floating):
            mask &= np.isfinite(column)
    return mask


def drop_non_finite(cloud: SplatCloud) -> tuple[SplatCloud, int]:
    mask = _finite(cloud.data)
    return SplatCloud(cloud.data[mask]), int((~mask).sum())


def recenter(cloud: SplatCloud) -> tuple[SplatCloud, list[float]]:
    """Translate splats so their median position is at the origin (baked into data)."""
    center = np.median(cloud.positions(), axis=0)
    data = cloud.data.copy()
    for axis, name in enumerate(("x", "y", "z")):
        data[name] = data[name] - center[axis]
    return SplatCloud(data), [float(v) for v in center]


@dataclass(frozen=True)
class PlyLayout:
    """Where a validated PLY's splat records sit in its file."""

    count: int
    dtype: np.dtype
    offset: int


def inspect_splat_ply_file(path: Path) -> PlyLayout:
    """Validate a PLY's header and length on disk without reading its records."""
    with path.open("rb") as fh:
        count, dtype, offset = parse_header(fh.read(MAX_HEADER_BYTES))
    if path.stat().st_size < offset + count * dtype.itemsize:
        raise PlyError("PLY truncated")
    return PlyLayout(count, dtype, offset)


@dataclass(frozen=True)
class CleanedPly:
    count: int  # splats written
    dropped: int  # splats removed because a value was NaN or infinite
    center: list[float]  # median position subtracted from every splat
    positions: np.ndarray  # (count, 3) float64 positions as written


def clean_splat_ply_file(
    src: Path, dst: Path, *, comments: Sequence[str] = (), chunk: int = CHUNK_SPLATS
) -> CleanedPly:
    """``drop_non_finite`` + ``recenter`` for a PLY on disk, streamed to ``dst`` in chunks.

    Records are read through a read-only memory map one chunk at a time: a first
    pass keeps only the finite splats' positions (24 bytes each) for the median,
    a second writes the filtered, recentered chunks. Memory stays near one chunk
    plus those positions however large the export, and the output is identical
    to the in-memory functions.
    """
    layout = inspect_splat_ply_file(src)
    records = np.memmap(src, dtype=layout.dtype, mode="r", offset=layout.offset,
                        shape=(layout.count,))  # fmt: skip
    axes = ("x", "y", "z")
    positions = np.empty((layout.count, 3), dtype=np.float64)
    kept = 0
    for start in range(0, layout.count, chunk):
        part = records[start : start + chunk]
        mask = _finite(part)
        n = int(mask.sum())
        for axis, name in enumerate(axes):
            positions[kept : kept + n, axis] = part[name][mask]
        kept += n
    positions = positions[:kept]
    center = np.median(positions, axis=0) if kept else np.zeros(3)
    written = 0
    with dst.open("wb") as fh:
        fh.write(_ply_header(layout.dtype, kept, comments))
        for start in range(0, layout.count, chunk):
            part = records[start : start + chunk]
            out = part[_finite(part)]  # a copy; the map stays read-only
            for axis, name in enumerate(axes):
                out[name] = out[name] - center[axis]
                positions[written : written + len(out), axis] = out[name]
            out.tofile(fh)  # parse_header only yields little-endian types
            written += len(out)
    del records, part  # release the map so the caller can delete the source
    return CleanedPly(kept, layout.count - kept, [float(v) for v in center], positions)


def splat_dtype() -> np.dtype:
    return np.dtype([(name, "<f4") for name in REQUIRED])
