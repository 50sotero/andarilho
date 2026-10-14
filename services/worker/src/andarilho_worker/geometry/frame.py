"""Up direction, floor, estimated scale and spawn from the camera path (README section 9).

COLMAP leaves scale arbitrary, and Nerfstudio's ``up`` orientation assumes the
cameras were held level on average. A phone filming a walkthrough points down,
which tilts that axis (about 10 degrees on the first real capture). The camera
of a walking person stays above one floor, so this module:

* takes the up direction from the normal of the dominant plane under the camera
  path (RANSAC over the splats near it) when that plane lies within
  ``MAX_TILT_DEGREES`` of the trainer's up and every camera is above it;
* sets the scale so the median camera height above that floor is
  ``CAMERA_HEIGHT_METERS``. This is an estimate: ``scaleStatus`` stays
  ``estimated`` until the user measures a known distance;
* puts the spawn where the walk started, on the floor, facing the way the first
  camera faced, and proposes a floor around the camera path.

Without a credible plane it keeps the trainer's up, assumes the floor one typical
splat distance below the lowest camera, and says so in ``warnings``.

Frames: the camera poses are COLMAP world-from-camera centres and viewing
directions. ``colmap_to_export`` applies Nerfstudio's COLMAP world swap, its
dataparser transform and scale, and the normalizing stage's recentering, which is
the frame of the splats the viewer decodes. The returned matrix maps that frame to
the canonical world (+Y up, nominal meters) as a proper rotation, a positive
uniform scale and a translation.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

CAMERA_HEIGHT_METERS = 1.5  # a phone held while walking
MAX_TILT_DEGREES = 30.0
PATH_MARGIN_METERS = 0.75  # proposed floor around the camera path
MIN_ROOM_HEIGHT, MAX_ROOM_HEIGHT = 2.2, 4.0
RANSAC_ITERATIONS = 2000
RANSAC_SAMPLE = 200_000
NEAR_PATH = 2.5  # splats within this many typical distances of a camera are candidates
INLIER_DISTANCE = 0.02  # plane inliers, in typical splat-to-camera distances
MIN_CLEARANCE = 0.2  # every camera at least this far above the floor, same unit
MIN_INLIERS = 200
REFINE_ITERATIONS = 5
REFINE_BAND = 3.0  # times INLIER_DISTANCE

# Nerfstudio Z up -> Andarilho Y up: (x, y, z) -> (x, z, -y)
Z_UP_TO_Y_UP = np.array([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]])


@dataclass(frozen=True)
class CameraPath:
    """Cameras in walk order: centres and unit viewing directions."""

    centers: np.ndarray  # (n, 3)
    forwards: np.ndarray  # (n, 3)

    @classmethod
    def from_record(cls, cameras: Sequence[Mapping[str, Any]]) -> CameraPath:
        ordered = sorted(cameras, key=lambda c: str(c["name"]))
        return cls(
            np.array([c["center"] for c in ordered], dtype=np.float64).reshape(-1, 3),
            np.array([c["forward"] for c in ordered], dtype=np.float64).reshape(-1, 3),
        )


def colmap_to_export(
    points: np.ndarray,
    transform: Sequence[Sequence[float]],
    scale: float,
    center: Sequence[float],
    *,
    directions: bool = False,
) -> np.ndarray:
    """COLMAP world -> the recentered export frame (directions skip translation and scale)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    p = np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])  # Nerfstudio's COLMAP world swap
    t = np.asarray(transform, dtype=np.float64)
    if directions:
        out = p @ t[:, :3].T
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    return (p @ t[:, :3].T + t[:, 3]) * float(scale) - np.asarray(center, dtype=np.float64)


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    """The proper rotation taking the unit ``normal`` to +Z (shortest arc)."""
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(normal, z)
    sin, cos = float(np.linalg.norm(axis)), float(normal @ z)
    if sin < 1e-9:
        return np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + k + k @ k * ((1 - cos) / (sin * sin))


def fit_floor_plane(
    points: np.ndarray, cameras: np.ndarray, *, typical: float, seed: int
) -> tuple[np.ndarray, float, int] | None:
    """``(normal, offset, inliers)`` of the plane ``normal . p + offset = 0`` under the path.

    The normal points up (towards the cameras); ``None`` when no plane within
    ``MAX_TILT_DEGREES`` of +Z keeps every camera above it with enough support.
    """
    if len(points) < 3:
        return None
    rng = np.random.default_rng(seed)
    if len(points) > RANSAC_SAMPLE:
        points = points[rng.choice(len(points), RANSAC_SAMPLE, replace=False)]
    up = np.array([0.0, 0.0, 1.0])
    min_cos = math.cos(math.radians(MAX_TILT_DEGREES))
    inlier, clearance = INLIER_DISTANCE * typical, MIN_CLEARANCE * typical
    best: tuple[int, np.ndarray, float] | None = None
    for _ in range(RANSAC_ITERATIONS):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(b - a, c - a)
        length = float(np.linalg.norm(normal))
        if length < 1e-12:
            continue
        normal = normal / length
        if normal @ up < 0:
            normal = -normal
        if normal @ up < min_cos:
            continue
        offset = -float(normal @ a)
        if (cameras @ normal + offset).min() < clearance:
            continue
        count = int((np.abs(points @ normal + offset) < inlier).sum())
        if best is None or count > best[0]:
            best = (count, normal, offset)
    if best is None or best[0] < MIN_INLIERS:
        return None
    _, normal, offset = best
    # Refine by least squares over a wider band, a few times: the best sample plane
    # sits on one patch of a noisy floor, the refit settles on the floor as a whole.
    for _ in range(REFINE_ITERATIONS):
        members = points[np.abs(points @ normal + offset) < REFINE_BAND * inlier]
        if len(members) < MIN_INLIERS:
            break
        centroid = members.mean(axis=0)
        refined = np.linalg.svd(members - centroid, full_matrices=False)[2][2]
        refined = refined if refined @ up > 0 else -refined
        refined_offset = -float(refined @ centroid)
        if refined @ up < min_cos or (cameras @ refined + refined_offset).min() < clearance:
            break
        normal, offset = refined, refined_offset
    inliers = int((np.abs(points @ normal + offset) < inlier).sum())
    return normal, offset, inliers


def nearest_distance(points: np.ndarray, targets: np.ndarray, chunk: int = 16_384) -> np.ndarray:
    """Distance from each point to its nearest target, in bounded-memory chunks."""
    out = np.empty(len(points))
    for start in range(0, len(points), chunk):
        part = points[start : start + chunk]
        out[start : start + len(part)] = np.min(
            np.linalg.norm(part[:, None, :] - targets[None, :, :], axis=2), axis=1
        )
    return out


@dataclass(frozen=True)
class FrameEstimate:
    matrix: list[float]  # worldFromDecodedRender, column-major
    meters_per_unit: float
    method: str  # "floor_plane" or "trainer_up"
    tilt_degrees: float
    floor_inliers: int
    spawn_feet: list[float]
    spawn_yaw: float
    min_xz: list[float]
    max_xz: list[float]
    height: float
    camera_heights: list[float]  # min, median, max (meters)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "tiltDegrees": round(self.tilt_degrees, 2),
            "floorInliers": self.floor_inliers,
            "metersPerUnit": round(self.meters_per_unit, 6),
            "cameraHeightMeters": CAMERA_HEIGHT_METERS,
            "cameraHeights": [round(h, 3) for h in self.camera_heights],
            "warnings": list(self.warnings),
        }


def _column_major(rotation_scale: np.ndarray, translation: np.ndarray) -> list[float]:
    m = np.eye(4)
    m[:3, :3] = rotation_scale
    m[:3, 3] = translation
    return [float(v) for v in m.T.reshape(-1)]


def estimate_frame(positions: np.ndarray, path: CameraPath, *, seed: int = 0) -> FrameEstimate:
    """Estimate the canonical frame from export-frame splat positions and the camera path."""
    cams = path.centers
    if len(cams) == 0:
        raise ValueError("no cameras")
    rng = np.random.default_rng(seed)
    sample = positions
    if len(sample) > RANSAC_SAMPLE:
        sample = sample[rng.choice(len(sample), RANSAC_SAMPLE, replace=False)]
    nearest = nearest_distance(sample, cams[:: max(1, len(cams) // 64)])
    typical = float(np.median(nearest)) or 1.0
    near = sample[nearest < NEAR_PATH * typical]
    warnings: list[str] = []
    fit = fit_floor_plane(near, cams, typical=typical, seed=seed)
    if fit is not None:
        normal, offset, inliers = fit
        rotation = rotation_to_z(normal)
        floor_z, method = -offset, "floor_plane"
    else:
        rotation, inliers, method = np.eye(3), 0, "trainer_up"
        floor_z = float(cams[:, 2].min()) - typical
        warnings.append(
            "The floor under the walk could not be found; check the floor height and scale."
        )
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(rotation[2] @ [0, 0, 1.0])))))
    level = cams @ rotation.T
    heights = level[:, 2] - floor_z
    scale = CAMERA_HEIGHT_METERS / float(np.median(heights))
    linear = scale * (Z_UP_TO_Y_UP @ rotation)
    translation = np.array([0.0, -scale * floor_z, 0.0])
    world = cams @ linear.T + translation
    heading = path.forwards @ (Z_UP_TO_Y_UP @ rotation).T
    yaw = math.atan2(-float(heading[0, 0]), -float(heading[0, 2]))
    lo = world[:, [0, 2]].min(axis=0) - PATH_MARGIN_METERS
    hi = world[:, [0, 2]].max(axis=0) + PATH_MARGIN_METERS
    near_world = near @ linear.T + translation
    ceiling = float(np.percentile(near_world[:, 1], 98)) if len(near_world) else MAX_ROOM_HEIGHT
    spread = float(np.ptp(world[:, 1]))
    if spread > CAMERA_HEIGHT_METERS:
        warnings.append(
            f"The camera height varies by {spread:.1f} m along the walk; "
            "the reconstruction may be bent. Check the floor in review."
        )
    return FrameEstimate(
        matrix=_column_major(linear, translation),
        meters_per_unit=scale,
        method=method,
        tilt_degrees=tilt,
        floor_inliers=inliers,
        spawn_feet=[float(world[0, 0]), 0.0, float(world[0, 2])],
        spawn_yaw=yaw,
        min_xz=[float(v) for v in lo],
        max_xz=[float(v) for v in hi],
        height=max(MIN_ROOM_HEIGHT, min(MAX_ROOM_HEIGHT, ceiling)),
        camera_heights=[
            float(world[:, 1].min()),
            float(np.median(world[:, 1])),
            float(world[:, 1].max()),
        ],  # fmt: skip
        warnings=warnings,
    )
