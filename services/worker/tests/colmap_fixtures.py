"""Synthetic COLMAP scenes for the frame and normalizing tests.

The scenes are canonical (+Y up, metres) and are recorded the way COLMAP would store
them: a tilted, yawed, shrunken and shifted world, with each camera's pose written as
COLMAP writes it (``camera_in_colmap``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from andarilho_worker.geometry.colmap import ColmapImage, SparsePoints

UP = np.array([0.0, 1.0, 0.0])
COLMAP_SCALE = 0.37  # one canonical metre is 0.37 COLMAP units
FLOOR, CEILING = 0, 1  # the kinds from 2 up are the four walls


def similarity_matrix_problem(m: Sequence[float]) -> str | None:
    """Accept only rotation + translation + positive uniform scale (column-major)."""
    if len(m) != 16:
        return "matrix must have 16 entries"
    if abs(m[3]) > 1e-9 or abs(m[7]) > 1e-9 or abs(m[11]) > 1e-9 or abs(m[15] - 1) > 1e-9:
        return "matrix must be affine"
    cols = [np.array(m[0:3]), np.array(m[4:7]), np.array(m[8:11])]
    norms = [float(np.linalg.norm(c)) for c in cols]
    scale = norms[0]
    if scale <= 1e-12 or any(abs(n - scale) > 1e-6 * max(1.0, scale) for n in norms):
        return "matrix must use a positive uniform scale"
    a, b, c = (col / scale for col in cols)
    if max(abs(a @ b), abs(a @ c), abs(b @ c)) > 1e-6:
        return "matrix rotation must be orthogonal"
    if float(np.linalg.det(np.column_stack([a, b, c]))) <= 0:
        return "matrix must not contain a reflection"
    return None


def apply_affine(matrix: list[float], points: np.ndarray) -> np.ndarray:
    m = np.array(matrix).reshape(4, 4).T
    return points @ m[:3, :3].T + m[:3, 3]


def axis_rotation(axis: np.ndarray, degrees: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    a = math.radians(degrees)
    return np.eye(3) + math.sin(a) * k + (1 - math.cos(a)) * k @ k


def colmap_transform() -> tuple[np.ndarray, float, np.ndarray]:
    """The (rotation, scale, shift) that takes canonical coordinates to COLMAP's world."""
    rotation = axis_rotation(np.array([0.0, 1.0, 0.0]), 33.0)
    rotation = rotation @ axis_rotation(np.array([1.0, 0.0, 0.0]), 12.0)
    return rotation, COLMAP_SCALE, np.array([12.0, -4.0, 7.0])


def quaternion_of(r: np.ndarray) -> tuple[float, float, float, float]:
    """(w, x, y, z) of a rotation matrix, from the largest of its diagonal terms."""
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return 0.25 * s, (r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return (r[2, 1] - r[1, 2]) / s, 0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s
    if r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return (r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return (r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s


def camera_in_colmap(
    image_id: int, centre: np.ndarray, forward: np.ndarray, transform
) -> ColmapImage:
    """A canonical camera at ``centre`` looking along ``forward``, as COLMAP records it."""
    a, scale, b = transform
    f = forward / np.linalg.norm(forward)
    right = np.cross(f, UP)
    right /= np.linalg.norm(right)
    image_up = np.cross(right, f)
    # COLMAP's camera axes are x right, y down, z forward: the columns are right, down, forward.
    world_from_camera = np.column_stack([right, -image_up, f])
    rotation = world_from_camera.T @ a.T  # world-to-camera, in COLMAP's world
    translation = -rotation @ (scale * (a @ centre) + b)
    return ColmapImage(
        image_id,
        1,
        f"{image_id:04d}.jpg",
        quaternion_of(rotation),
        tuple(float(v) for v in translation),
    )


def l_walk(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Canonical centres and forwards: 25 frames along +X, then 15 along +Z.

    The phone is held 1.5 m up and pitched 15 degrees down; each heading wanders a little.
    """
    pitch = math.radians(15.0)
    centres, forwards = [], []
    for i in range(40):
        if i < 25:
            centre = np.array([0.6 + 3.8 * i / 24, 1.5, 0.6])
            yaw = rng.normal(0.0, math.radians(8.0))
        else:
            centre = np.array([4.4, 1.5, 0.6 + 2.3 * (i - 24) / 15])
            yaw = math.pi / 2 + rng.normal(0.0, math.radians(8.0))
        heading = np.array([math.cos(yaw), 0.0, math.sin(yaw)])
        centres.append(centre)
        forwards.append(heading * math.cos(pitch) - UP * math.sin(pitch))
    return np.array(centres), np.array(forwards)


def make_room(
    rng: np.random.Generator, length: float = 5.0, width: float = 3.5, height: float = 2.6
) -> tuple[np.ndarray, np.ndarray]:
    """A length x width x height m room with 1 cm noise, and the kind of each point.

    Points are (X, Y, Z) with Y up: the floor is at Y = 0 and the ceiling at Y = height.
    """
    floor = np.column_stack(
        [rng.uniform(0, length, 700), np.zeros(700), rng.uniform(0, width, 700)]
    )
    ceiling = np.column_stack(
        [rng.uniform(0, length, 400), np.full(400, height), rng.uniform(0, width, 400)]
    )
    walls = [
        np.column_stack([rng.uniform(0, length, 225), rng.uniform(0, height, 225), np.full(225, z)])
        for z in (0.0, width)
    ] + [
        np.column_stack([np.full(225, x), rng.uniform(0, height, 225), rng.uniform(0, width, 225)])
        for x in (0.0, length)
    ]
    points = np.vstack([floor, ceiling, *walls])
    kind = np.repeat([FLOOR, CEILING, 2, 3, 4, 5], [700, 400, 225, 225, 225, 225])
    return points + rng.normal(0, 0.01, points.shape), kind


def to_sparse(points: np.ndarray, transform, rng: np.random.Generator) -> SparsePoints:
    """Canonical points as COLMAP stores them, with random errors (0.2 to 1.5 px) and tracks."""
    a, scale, b = transform
    n = len(points)
    return SparsePoints(
        xyz=scale * points @ a.T + b,
        rgb=rng.integers(0, 256, (n, 3)).astype(np.uint8),
        error=rng.uniform(0.2, 1.5, n),
        track_length=rng.integers(3, 12, n).astype(np.int64),
    )


def with_trust(points: SparsePoints, error: np.ndarray, track: np.ndarray) -> SparsePoints:
    """The same points with new reprojection errors (px) and track lengths."""
    return SparsePoints(
        points.xyz, points.rgb, np.asarray(error, float), np.asarray(track, np.int64)
    )


def join_points(*parts: SparsePoints) -> SparsePoints:
    return SparsePoints(
        np.vstack([p.xyz for p in parts]),
        np.vstack([p.rgb for p in parts]),
        np.concatenate([p.error for p in parts]),
        np.concatenate([p.track_length for p in parts]),
    )


def cameras_for(centres, forwards, transform) -> list[ColmapImage]:
    return [
        camera_in_colmap(i + 1, c, f, transform)
        for i, (c, f) in enumerate(zip(centres, forwards, strict=True))
    ]
