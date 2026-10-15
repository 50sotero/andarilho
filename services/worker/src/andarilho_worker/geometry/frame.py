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

``frame_from_sparse`` serves the textured-layout pipeline, which has no splats: it
starts from COLMAP's own frame (arbitrary up, arbitrary scale) and takes the up
direction from the camera poses (the phone's right axis stays level while it is
panned or pitched; on a straight walk the cross product of right and walk gives
it). It then levels the sparse points, turns the walls onto the X and Z axes (the
dominant wall direction, from nearest-neighbour pairs or the camera headings) and
hands the result to ``estimate_frame``. Its matrix maps raw COLMAP world coordinates
to the canonical world (worldFromColmap).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from andarilho_worker.geometry.colmap import ColmapImage, SparsePoints

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

# Sparse points (frame_from_sparse). A 1,443-point capture is far below MIN_INLIERS,
# so the sparse path asks for a fraction of the candidates instead.
SPARSE_MAX_ERROR_PX = 2.0  # reprojection error of a point worth trusting
SPARSE_FALLBACK_ERROR_PX = 4.0
SPARSE_MIN_TRACK = 3
SPARSE_MIN_POINTS = 50
SPARSE_MIN_FALLBACK_POINTS = 20
SPARSE_YAW_SPREAD = 0.02  # middle eigenvalue of the right-vector second moment
SPARSE_STATIONARY = 0.01  # camera motion below this share of the cloud's extent
SPARSE_INLIER_FRACTION = 0.05
# Smoothed peak share of a wall histogram (see _histogram_peak). Measured (40 seeds
# where stated): 1/90 (0.011) for a flat histogram; 0.0256 for the L-room fixture (1 cm
# noise, 2,000 points, two wall directions); 0.016 for a straight single-wall hallway.
# The spec's 0.04 rejects the L-room: its walk falls back to camera headings and the
# walls come out 2.4 to 2.5 deg off, against the 1.5 deg bound. So this deviates from the
# spec: the points path uses 0.02 and needs SPARSE_WALL_MIN_POINTS. Structureless clouds
# in the room box reach 0.02 by chance at 60 points (max 0.025, 5 of 40 seeds), but not
# at 120 (max 0.019) or 200 points (max 0.017), so the gate sits at 200.
SPARSE_WALL_PEAK = 0.02
SPARSE_WALL_MIN_POINTS = 200
SPARSE_WALL_SAMPLE = 3000
SPARSE_NEIGHBOURS = 6
SPARSE_NEIGHBOUR_RADIUS = 3.0  # times the median nearest-neighbour distance
SPARSE_MIN_PAIRS = 100
# The walk's steps: the resultant length of their quadrupled directions is near 1 for a
# walk along one axis (0.77 for steps spread +/- 20 degrees) and near 0 for random steps.
SPARSE_MIN_WALK_STEPS = 3
SPARSE_WALK_RESULTANT = 0.5
HEADING_BINS = 90  # one-degree bins over a 90-degree wall period
HEADING_SMOOTH = 3  # the histogram peak is smoothed over +/- this many bins

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
    points: np.ndarray,
    cameras: np.ndarray,
    *,
    typical: float,
    seed: int,
    min_inliers: int = MIN_INLIERS,
) -> tuple[np.ndarray, float, int] | None:
    """``(normal, offset, inliers)`` of the plane ``normal . p + offset = 0`` under the path.

    The normal points up (towards the cameras); ``None`` when no plane within
    ``MAX_TILT_DEGREES`` of +Z keeps every camera above it with at least
    ``min_inliers`` support.
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
    if best is None or best[0] < min_inliers:
        return None
    _, normal, offset = best
    # Refine by least squares over a wider band, a few times: the best sample plane
    # sits on one patch of a noisy floor, the refit settles on the floor as a whole.
    for _ in range(REFINE_ITERATIONS):
        members = points[np.abs(points @ normal + offset) < REFINE_BAND * inlier]
        if len(members) < min_inliers:
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
    # Column-major 4x4 similarity mapping the input frame to the canonical world (+Y up,
    # metres, floor at y = 0): worldFromDecodedRender for splats, worldFromColmap for
    # sparse points.
    matrix: list[float]
    meters_per_unit: float
    method: str  # "floor_plane" or "trainer_up"
    # Angle from the input's up to the floor normal. Splats: the trainer's up. Sparse
    # points: the camera-derived up, since the points are levelled before the floor fit.
    tilt_degrees: float
    floor_inliers: int
    spawn_feet: list[float]
    spawn_yaw: float
    min_xz: list[float]
    max_xz: list[float]
    height: float
    camera_heights: list[float]  # min, median, max (meters)
    warnings: list[str] = field(default_factory=list)
    wall_yaw_degrees: float = 0.0  # sparse points only: the wall direction levelled out
    wall_yaw_source: str = "none"  # "points", "walk", "cameras" or "none"

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "method": self.method,
            "tiltDegrees": round(self.tilt_degrees, 2),
            "floorInliers": self.floor_inliers,
            "metersPerUnit": round(self.meters_per_unit, 6),
            "cameraHeightMeters": CAMERA_HEIGHT_METERS,
            "cameraHeights": [round(h, 3) for h in self.camera_heights],
            "warnings": list(self.warnings),
        }
        if self.wall_yaw_source != "none":
            out["wallYawDegrees"] = round(self.wall_yaw_degrees, 2)
            out["wallYawSource"] = self.wall_yaw_source
        return out


def _column_major(rotation_scale: np.ndarray, translation: np.ndarray) -> list[float]:
    m = np.eye(4)
    m[:3, :3] = rotation_scale
    m[:3, 3] = translation
    return [float(v) for v in m.T.reshape(-1)]


def estimate_frame(
    positions: np.ndarray,
    path: CameraPath,
    *,
    seed: int = 0,
    min_inliers_fraction: float | None = None,
) -> FrameEstimate:
    """Estimate the canonical frame from export-frame splat positions and the camera path.

    ``min_inliers_fraction`` asks the floor to be supported by that share of the points
    near the path (at least 20); ``None`` keeps ``MIN_INLIERS``.
    """
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
    min_inliers = (
        MIN_INLIERS
        if min_inliers_fraction is None
        else max(20, int(min_inliers_fraction * len(near)))
    )
    warnings: list[str] = []
    fit = fit_floor_plane(near, cams, typical=typical, seed=seed, min_inliers=min_inliers)
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


def frame_from_sparse(
    points: SparsePoints, images: Sequence[ColmapImage], *, seed: int = 0
) -> FrameEstimate:
    """Canonical frame from COLMAP's sparse points and camera poses (worldFromColmap).

    Raw COLMAP world coordinates map to +Y up, metres, a floor at y = 0 and the
    proposed floor's centre (XZ) at the origin. The matrix is a proper rotation, a
    positive uniform scale and a translation. The cameras are taken in walk order
    (sorted by name, as ``CameraPath.from_record`` does), so the first one sets the
    spawn. Raises ``ValueError`` without cameras or points.
    """
    if len(images) == 0:
        raise ValueError("no cameras")
    xyz = _reliable_points(points)
    if len(xyz) == 0:
        raise ValueError("no sparse points")
    walk = sorted(images, key=lambda image: image.name)
    centres = np.array([image.center() for image in walk])
    rights = np.array([image.right() for image in walk])
    ups = np.array([image.up() for image in walk])
    forwards = np.array([image.forward() for image in walk])
    extent = float(np.ptp(xyz, axis=0).max())
    up = _up_from_cameras(centres, rights, ups, forwards, extent)
    level = rotation_to_z(up)
    levelled = xyz @ level.T
    yaw, source = _wall_yaw(levelled, centres @ level.T, forwards @ level.T, seed)
    turn = _rotation_about_z(-yaw)  # walls onto the X and Y axes of the levelled frame
    est = estimate_frame(
        levelled @ turn.T,
        CameraPath((centres @ level.T) @ turn.T, (forwards @ level.T) @ turn.T),
        seed=seed,
        min_inliers_fraction=SPARSE_INLIER_FRACTION,
    )
    colmap_to_pre = np.eye(4)
    colmap_to_pre[:3, :3] = turn @ level
    world_from_colmap = np.array(est.matrix, dtype=np.float64).reshape(4, 4).T @ colmap_to_pre
    # Recentre XZ on the middle of the proposed floor, so that middle is the origin.
    lo, hi = np.array(est.min_xz), np.array(est.max_xz)
    middle = np.array([(lo[0] + hi[0]) / 2, 0.0, (lo[1] + hi[1]) / 2])
    world_from_colmap[:3, 3] -= middle
    return FrameEstimate(
        matrix=_column_major(world_from_colmap[:3, :3], world_from_colmap[:3, 3]),
        meters_per_unit=est.meters_per_unit,
        method=est.method,
        tilt_degrees=est.tilt_degrees,
        floor_inliers=est.floor_inliers,
        spawn_feet=(np.array(est.spawn_feet) - middle).tolist(),
        spawn_yaw=est.spawn_yaw,
        min_xz=(lo - middle[[0, 2]]).tolist(),
        max_xz=(hi - middle[[0, 2]]).tolist(),
        height=est.height,
        camera_heights=list(est.camera_heights),
        warnings=list(est.warnings),
        wall_yaw_degrees=math.degrees(yaw),
        wall_yaw_source=source,
    )


def _reliable_points(points: SparsePoints) -> np.ndarray:
    """The sparse points worth trusting: a real track and a small reprojection error."""
    keep = (points.track_length >= SPARSE_MIN_TRACK) & (points.error < SPARSE_MAX_ERROR_PX)
    if keep.sum() < SPARSE_MIN_POINTS:
        keep = points.error < SPARSE_FALLBACK_ERROR_PX
    if keep.sum() < SPARSE_MIN_FALLBACK_POINTS:
        keep = np.ones(len(points.xyz), dtype=bool)
    return points.xyz[keep]


def _up_from_cameras(
    centres: np.ndarray,
    rights: np.ndarray,
    ups: np.ndarray,
    forwards: np.ndarray,
    extent: float,
) -> np.ndarray:
    """Unit up direction in COLMAP world coordinates, from the camera poses.

    A phone held level keeps its right vector horizontal, so the one direction every
    right vector avoids is up: the smallest eigenvector of their second moment. When
    the right vectors barely vary (a straight walk), up is right x walk instead, with
    the walk taken from the camera centres; a stationary camera uses its mean image-up.
    """
    mean_up = ups.mean(axis=0)
    values, vectors = np.linalg.eigh(rights.T @ rights / len(rights))  # ascending
    if values[1] >= SPARSE_YAW_SPREAD:
        up = vectors[:, 0]
    else:
        offsets = centres - centres.mean(axis=0)
        spread, axes = np.linalg.eigh(offsets.T @ offsets / len(centres))
        walk = axes[:, -1]
        if walk @ forwards.mean(axis=0) < 0:
            walk = -walk
        cross = np.cross(rights.mean(axis=0), walk)
        moving = math.sqrt(max(float(spread[-1]), 0.0)) > SPARSE_STATIONARY * extent
        up = cross if moving and np.linalg.norm(cross) > 1e-6 else mean_up
    if up @ mean_up < 0:
        up = -up
    return up / np.linalg.norm(up)


def _rotation_about_z(radians: float) -> np.ndarray:
    c, s = math.cos(radians), math.sin(radians)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _wall_yaw(
    levelled: np.ndarray, centres: np.ndarray, forwards: np.ndarray, seed: int
) -> tuple[float, str]:
    """Dominant wall direction in the levelled frame, mod 90 degrees, and where it came from.

    Wall points give it from nearest-neighbour pairs (their XY direction runs along the
    wall), when there are at least SPARSE_WALL_MIN_POINTS points. Otherwise the walk
    itself does: people walk along corridors and room walls, so the camera path's steps,
    weighted by length, point along them. A phone looking around while walking spreads
    its headings, so they come last. A flat histogram from every source means no
    direction, which is zero.
    """
    histogram, pairs = _wall_pair_histogram(levelled, seed)
    trusted = len(levelled) >= SPARSE_WALL_MIN_POINTS and pairs >= SPARSE_MIN_PAIRS
    if trusted:
        yaw, strength = _histogram_peak(histogram)
        if strength >= SPARSE_WALL_PEAK:
            return yaw, "points"
    steps = np.diff(centres[:, :2], axis=0)
    lengths = np.linalg.norm(steps, axis=1)
    moved = lengths > 1e-9
    if moved.sum() >= SPARSE_MIN_WALK_STEPS:
        # A handheld walk wobbles by several degrees, so take the length-weighted circular
        # mean of the step directions with a 90-degree period (angles quadrupled) rather
        # than a narrow histogram peak; its resultant length says how consistent it is.
        # Random steps reach about 2 / sqrt(n) by chance, so few steps need a higher bar.
        angles = np.arctan2(steps[moved, 1], steps[moved, 0])
        weights = lengths[moved]
        resultant = np.sum(weights * np.exp(4j * angles)) / weights.sum()
        effective = weights.sum() ** 2 / float(np.sum(weights**2))
        if abs(resultant) >= max(SPARSE_WALK_RESULTANT, 2.0 / math.sqrt(effective)):
            return float(np.angle(resultant)) / 4.0, "walk"
    # Limitation: the heading gate is weak. Uniform random headings clear SPARSE_WALL_PEAK
    # in all 40 seeds at 1, 3, 10 and 40 cameras (median strengths 0.143, 0.048, 0.043 and
    # 0.025), so "cameras" is a weak hint about the walls, not a confirmed direction.
    headings = forwards[:, :2]
    headings = headings[np.linalg.norm(headings, axis=1) > 1e-9]
    yaw, strength = _histogram_peak(_mod90_histogram(np.arctan2(headings[:, 1], headings[:, 0])))
    if strength >= SPARSE_WALL_PEAK:
        return yaw, "cameras"
    return 0.0, "none"


def _wall_pair_histogram(points: np.ndarray, seed: int) -> tuple[np.ndarray, int]:
    """Histogram of XY directions (mod 90 degrees) of nearby point pairs, and the pair count."""
    n = len(points)
    if n < 3:
        return np.zeros(HEADING_BINS), 0
    points = points - points.mean(axis=0)  # directions do not depend on the origin
    rng = np.random.default_rng(seed)
    queries = rng.choice(n, size=min(n, SPARSE_WALL_SAMPLE), replace=False)
    k = min(SPARSE_NEIGHBOURS, n - 1)
    neighbours, distances = _nearest_neighbours(points, queries, k)
    radius = SPARSE_NEIGHBOUR_RADIUS * float(np.median(distances.min(axis=1)))
    vectors = points[neighbours] - points[queries][:, None, :]
    keep = (distances <= radius) & (distances > 1e-12)
    keep &= np.linalg.norm(vectors[..., :2], axis=2) > 1e-12  # a vertical pair has no heading
    xy = vectors[keep][:, :2]
    return _mod90_histogram(np.arctan2(xy[:, 1], xy[:, 0])), int(keep.sum())


def _nearest_neighbours(
    points: np.ndarray, queries: np.ndarray, k: int
) -> tuple[np.ndarray, np.ndarray]:
    """The ``k`` nearest other points of each query row: indices into ``points``, and distances.

    Exact, as a brute-force search is, at a cost near the cloud size rather than the cloud
    size times the number of queries. The points are binned on a uniform grid of about
    ``n ** (1/3)`` cells per side. A query searches the cells within ``r`` steps of its own
    cell and stops once its k-th neighbour lies within ``r`` cell widths: a point in a cell
    further out is more than ``r`` cell widths away, so it cannot be a nearer neighbour.
    """
    lo = points.min(axis=0)
    span = float(np.max(points.max(axis=0) - lo))
    cell = max(span, 1e-9) / max(1.0, len(points) ** (1 / 3))
    coords = np.floor((points - lo) / cell).astype(np.int64)
    dims = coords.max(axis=0) + 1
    keys = np.ravel_multi_index(coords.T, tuple(int(d) for d in dims))
    order = np.argsort(keys, kind="stable")
    cell_keys, starts, counts = np.unique(keys[order], return_index=True, return_counts=True)
    neighbours = np.empty((len(queries), k), dtype=np.int64)
    distances = np.empty((len(queries), k))
    for row, query in enumerate(queries):
        ring = 1
        while True:
            near = _cell_members(coords[query], ring, dims, (cell_keys, starts, counts, order))
            near = near[near != query]
            dist = np.linalg.norm(points[near] - points[query], axis=1)
            covered = ring >= int(dims.max())  # every cell searched: nothing is left out
            if len(dist) >= k:
                best = np.argpartition(dist, k - 1)[:k]
                if covered or float(dist[best].max()) <= ring * cell:
                    neighbours[row] = near[best]
                    distances[row] = dist[best]
                    break
            ring += 1
    return neighbours, distances


def _cell_members(
    centre: np.ndarray,
    ring: int,
    dims: np.ndarray,
    index: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Indices of the points in the grid cells within ``ring`` steps of ``centre``."""
    cell_keys, starts, counts, order = index
    steps = np.arange(-ring, ring + 1)
    offsets = np.stack(np.meshgrid(steps, steps, steps, indexing="ij"), axis=-1).reshape(-1, 3)
    cells = centre + offsets
    cells = cells[np.all((cells >= 0) & (cells < dims), axis=1)]
    keys = np.ravel_multi_index(cells.T, tuple(int(d) for d in dims))
    pos = np.minimum(np.searchsorted(cell_keys, keys), len(cell_keys) - 1)
    pos = pos[cell_keys[pos] == keys]
    if len(pos) == 0:
        return np.empty(0, dtype=np.int64)
    return np.concatenate([order[starts[p] : starts[p] + counts[p]] for p in pos])


def _mod90_histogram(angles: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Counts (or weights) of angles (radians) folded into one 90-degree period, by degree."""
    bins = np.floor(np.mod(np.degrees(angles), 90.0)).astype(np.int64) % HEADING_BINS
    return np.bincount(bins, weights=weights, minlength=HEADING_BINS).astype(np.float64)


def _histogram_peak(histogram: np.ndarray) -> tuple[float, float]:
    """``(angle, strength)``: the period's direction (radians) and the peak's share.

    The histogram is smoothed over +/- HEADING_SMOOTH bins; the strength is the
    smoothed peak as a share of the smoothed total (about 1 / HEADING_BINS when flat).
    The angle is the circular mean of the raw counts in the peak window.
    """
    offsets = np.arange(-HEADING_SMOOTH, HEADING_SMOOTH + 1)
    smooth = sum(np.roll(histogram, int(offset)) for offset in offsets)
    total = float(smooth.sum())
    if total <= 0:
        return 0.0, 0.0
    peak = int(np.argmax(smooth))
    window = (peak + offsets) % HEADING_BINS
    # A 90-degree period maps onto a full turn when the angle is quadrupled.
    phase = np.radians(window + 0.5) * 4.0
    weights = histogram[window]
    mean = math.atan2(
        float((weights * np.sin(phase)).sum()), float((weights * np.cos(phase)).sum())
    )
    return (mean / 4.0) % (math.pi / 2), float(smooth[peak] / total)
