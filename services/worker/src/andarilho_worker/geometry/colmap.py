"""COLMAP binary models: cameras.bin, images.bin and points3D.bin (little-endian).

Layouts (COLMAP 3.12):

* ``cameras.bin``: uint64 count, then per camera ``"<iiQQ"`` (camera id, model id,
  width, height) followed by that model's float64 parameters. Parameter counts:
  0 SIMPLE_PINHOLE 3, 1 PINHOLE 4, 2 SIMPLE_RADIAL 4, 3 RADIAL 5, 4 OPENCV 8,
  5 OPENCV_FISHEYE 8, 6 FULL_OPENCV 12, 7 FOV 5, 8 SIMPLE_RADIAL_FISHEYE 4,
  9 RADIAL_FISHEYE 5, 10 THIN_PRISM_FISHEYE 12.
* ``images.bin``: uint64 count, then per image ``"<idddddddi"`` (image id, quaternion
  w x y z, translation x y z, camera id), a NUL-terminated UTF-8 name, a uint64 count
  of 2D points and that many ``"<ddq"`` entries (x, y, point3D id).
* ``points3D.bin``: uint64 count, then per point ``"<QdddBBBd"`` (point id, xyz, rgb,
  reprojection error) followed by a uint64 track length and that many ``"<ii"``
  entries (image id, point2D index).

Poses are world-to-camera: ``x_cam = R x_world + t`` with the quaternion giving R.
The camera axes are COLMAP's: x right, y down, z forward. COLMAP 3.12 also writes
``rigs.bin`` and ``frames.bin``; they are not read here.

Every reader checks each record against the data length and raises ``ValueError``
on a truncated file, on a non-UTF-8 name or on an unknown camera model.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# COLMAP camera model id -> (name, number of float64 parameters).
CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}
POINTS_LIMIT = 2_000_000

_MODEL_IDS = {name: model_id for model_id, (name, _) in CAMERA_MODELS.items()}
_COUNT = struct.Struct("<Q")
_CAMERA_HEAD = struct.Struct("<iiQQ")
_IMAGE_HEAD = struct.Struct("<idddddddi")
_POINTS2D_ENTRY = struct.Struct("<ddq")
_POINT_HEAD = struct.Struct("<QdddBBBdQ")  # the track length closes the fixed part
_TRACK_ENTRY = struct.Struct("<ii")


@dataclass(frozen=True)
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: tuple[float, ...]

    def intrinsics(self) -> np.ndarray:
        """The 3x3 pinhole matrix K, in pixels. Only PINHOLE and SIMPLE_PINHOLE (undistorted data).

        COLMAP measures image coordinates from the top-left corner of the image, so the
        centre of pixel (u, v) sits at (u + 0.5, v + 0.5) and the principal point (cx, cy)
        is generally not an integer.
        """
        if self.model == "PINHOLE":
            fx, fy, cx, cy = self.params
        elif self.model == "SIMPLE_PINHOLE":
            fx, cx, cy = self.params
            fy = fx
        else:
            raise ValueError(f"intrinsics need a PINHOLE camera, not {self.model}")
        return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])


@dataclass(frozen=True)
class ColmapImage:
    id: int
    camera_id: int
    name: str
    qvec: tuple[float, float, float, float]  # (w, x, y, z), world-to-camera
    tvec: tuple[float, float, float]

    def rotation(self) -> np.ndarray:
        """World-to-camera rotation matrix from the normalised quaternion."""
        return quaternion_to_rotation(self.qvec)

    def center(self) -> np.ndarray:
        """Camera centre in world coordinates: -R^T t."""
        return -self.rotation().T @ np.asarray(self.tvec, dtype=np.float64)

    def forward(self) -> np.ndarray:
        """Viewing direction in world coordinates (camera +z)."""
        return self.rotation().T @ np.array([0.0, 0.0, 1.0])

    def right(self) -> np.ndarray:
        """Image-right direction in world coordinates (camera +x)."""
        return self.rotation().T @ np.array([1.0, 0.0, 0.0])

    def up(self) -> np.ndarray:
        """Image-up direction in world coordinates (camera -y, since COLMAP y points down)."""
        return self.rotation().T @ np.array([0.0, -1.0, 0.0])


@dataclass(frozen=True)
class SparsePoints:
    xyz: np.ndarray  # (N, 3) float64
    rgb: np.ndarray  # (N, 3) uint8
    error: np.ndarray  # (N,) float64, mean reprojection error in pixels
    track_length: np.ndarray  # (N,) int64


def quaternion_to_rotation(qvec: Sequence[float]) -> np.ndarray:
    """Rotation matrix of a (w, x, y, z) quaternion, normalised first."""
    w, x, y, z = (float(v) for v in qvec)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if not norm > 0 or not math.isfinite(norm):
        raise ValueError("camera quaternion must be finite and non-zero")
    return _rotation_matrix(w / norm, x / norm, y / norm, z / norm)


def _rotation_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    """The rotation matrix of ``(w, x, y, z)`` as given, without normalising it."""
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])  # fmt: skip


def _unpack(layout: struct.Struct, data: bytes, offset: int, what: str) -> tuple[Any, ...]:
    if offset < 0 or offset + layout.size > len(data):
        raise ValueError(f"{what} is truncated")
    return layout.unpack_from(data, offset)


def _count(data: bytes, what: str) -> int:
    (count,) = _unpack(_COUNT, data, 0, what)
    return int(count)


def _camera_model(model_id: int) -> tuple[str, int]:
    try:
        return CAMERA_MODELS[model_id]
    except KeyError:
        raise ValueError(f"unknown COLMAP camera model {model_id}") from None


def read_cameras_bin(path: Path) -> dict[int, ColmapCamera]:
    """Every camera of a COLMAP cameras.bin, keyed by camera id."""
    data = path.read_bytes()
    cameras: dict[int, ColmapCamera] = {}
    offset = _COUNT.size
    for _ in range(_count(data, "cameras.bin")):
        camera_id, model_id, width, height = _unpack(_CAMERA_HEAD, data, offset, "cameras.bin")
        offset += _CAMERA_HEAD.size
        model, count = _camera_model(model_id)
        params_layout = struct.Struct(f"<{count}d")
        params = _unpack(params_layout, data, offset, "cameras.bin")
        offset += params_layout.size
        cameras[camera_id] = ColmapCamera(
            camera_id, model, int(width), int(height), tuple(float(p) for p in params)
        )
    return cameras


def _decode_name(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("image name is not valid UTF-8") from exc


def read_images_bin(path: Path) -> list[ColmapImage]:
    """Every registered image of a COLMAP images.bin, sorted by name (the walk order).

    The 2D observations are skipped: the frame needs poses only.
    """
    data = path.read_bytes()
    images: list[ColmapImage] = []
    offset = _COUNT.size
    for _ in range(_count(data, "images.bin")):
        image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = _unpack(
            _IMAGE_HEAD, data, offset, "images.bin"
        )
        offset += _IMAGE_HEAD.size
        end = data.find(b"\x00", offset)
        if end < 0:
            raise ValueError("images.bin is truncated")
        name = _decode_name(data[offset:end])
        offset = end + 1
        (points,) = _unpack(_COUNT, data, offset, "images.bin")
        offset += _COUNT.size
        offset += int(points) * _POINTS2D_ENTRY.size
        if offset > len(data):
            raise ValueError("images.bin is truncated")
        images.append(
            ColmapImage(
                int(image_id),
                int(camera_id),
                name,
                (float(qw), float(qx), float(qy), float(qz)),
                (float(tx), float(ty), float(tz)),
            )
        )
    return sorted(images, key=lambda image: image.name)


def _point_at(data: bytes, offset: int) -> tuple[tuple[Any, ...], int]:
    """The fixed fields of the points3D record at ``offset``, and the offset of the next record.

    The track entries are skipped; the record's track length is the last field.
    """
    head = _unpack(_POINT_HEAD, data, offset, "points3D.bin")
    end = offset + _POINT_HEAD.size + int(head[-1]) * _TRACK_ENTRY.size
    if end > len(data):
        raise ValueError("points3D.bin is truncated")
    return head, end


def _scan_xyz(data: bytes, count: int) -> np.ndarray:
    """XYZ of the first ``count`` points of a points3D.bin payload (the rest is not kept)."""
    xyz: list[tuple[float, float, float]] = []
    offset = _COUNT.size
    for _ in range(count):
        head, offset = _point_at(data, offset)
        xyz.append((head[1], head[2], head[3]))
    return np.array(xyz, dtype=np.float64).reshape(-1, 3)


def _scan_points3d(data: bytes, count: int) -> SparsePoints:
    """The first ``count`` points of a points3D.bin payload, with colour, error and track."""
    xyz: list[tuple[float, float, float]] = []
    rgb: list[tuple[int, int, int]] = []
    error: list[float] = []
    track: list[int] = []
    offset = _COUNT.size
    for _ in range(count):
        head, offset = _point_at(data, offset)
        _pid, x, y, z, r, g, b, err, length = head
        xyz.append((x, y, z))
        rgb.append((r, g, b))
        error.append(err)
        track.append(int(length))
    return SparsePoints(
        xyz=np.array(xyz, dtype=np.float64).reshape(-1, 3),
        rgb=np.array(rgb, dtype=np.uint8).reshape(-1, 3),
        error=np.array(error, dtype=np.float64),
        track_length=np.array(track, dtype=np.int64),
    )


def read_points3d_full(path: Path, limit: int = POINTS_LIMIT) -> SparsePoints:
    """Every point of a points3D.bin; a file with more than ``limit`` points is refused."""
    data = path.read_bytes()
    count = _count(data, "points3D.bin")
    if count > limit:
        raise ValueError(f"points3D.bin has {count} points, above the limit of {limit}")
    return _scan_points3d(data, count)


def read_points3d(path: Path, limit: int = POINTS_LIMIT) -> np.ndarray:
    """XYZ of COLMAP points3D.bin (bounded: at most ``limit`` points are read)."""
    data = path.read_bytes()
    return _scan_xyz(data, min(_count(data, "points3D.bin"), limit))


def read_camera_poses(path: Path) -> list[dict[str, object]]:
    """Registered cameras of a COLMAP images.bin: name, world centre, viewing direction.

    The centre is ``-R^T t`` and the camera looks along ``R^T (0, 0, 1)`` (OpenCV
    convention). Sorted by name, which is the walk order for extracted frames.

    The quaternion is used as stored, not normalised, as the original reader did (COLMAP
    always writes unit quaternions, so this matches ``ColmapImage.center()``).
    """
    poses: list[dict[str, object]] = []
    for image in read_images_bin(path):
        rotation = _rotation_matrix(*image.qvec)
        center = -rotation.T @ np.asarray(image.tvec, dtype=np.float64)
        forward = rotation.T @ np.array([0.0, 0.0, 1.0])
        poses.append({"name": image.name, "center": center.tolist(), "forward": forward.tolist()})
    return poses


def _model_id(name: str) -> int:
    try:
        return _MODEL_IDS[name]
    except KeyError:
        raise ValueError(f"unknown COLMAP camera model {name}") from None


def write_cameras_bin(path: Path, cameras: Sequence[ColmapCamera]) -> None:
    """Write cameras.bin (tests and fixtures)."""
    blob = [_COUNT.pack(len(cameras))]
    for camera in cameras:
        model_id = _model_id(camera.model)
        count = CAMERA_MODELS[model_id][1]
        if len(camera.params) != count:
            raise ValueError(f"{camera.model} takes {count} parameters")
        blob.append(_CAMERA_HEAD.pack(camera.id, model_id, camera.width, camera.height))
        blob.append(struct.pack(f"<{count}d", *camera.params))
    path.write_bytes(b"".join(blob))


def write_images_bin(
    path: Path,
    images: Sequence[ColmapImage],
    points2d: Sequence[Sequence[tuple[float, float, int]]] | None = None,
) -> None:
    """Write images.bin (tests and fixtures); ``points2d`` defaults to no observations."""
    if points2d is None:
        points2d = [() for _ in images]
    if len(points2d) != len(images):
        raise ValueError("one list of 2D points per image")
    blob = [_COUNT.pack(len(images))]
    for image, observations in zip(images, points2d, strict=True):
        if "\x00" in image.name:
            raise ValueError("image names cannot contain NUL")
        blob.append(_IMAGE_HEAD.pack(image.id, *image.qvec, *image.tvec, image.camera_id))
        blob.append(image.name.encode("utf-8") + b"\x00")
        blob.append(_COUNT.pack(len(observations)))
        blob.extend(_POINTS2D_ENTRY.pack(x, y, pid) for x, y, pid in observations)
    path.write_bytes(b"".join(blob))


def write_points3d_bin(path: Path, points: SparsePoints) -> None:
    """Write points3D.bin with ids 1..N and zero track entries (tests and fixtures)."""
    blob = [_COUNT.pack(len(points.xyz))]
    for index in range(len(points.xyz)):
        x, y, z = (float(v) for v in points.xyz[index])
        r, g, b = (int(v) for v in points.rgb[index])
        length = int(points.track_length[index])
        blob.append(
            _POINT_HEAD.pack(index + 1, x, y, z, r, g, b, float(points.error[index]), length)
        )
        blob.append(bytes(length * _TRACK_ENTRY.size))
    path.write_bytes(b"".join(blob))
