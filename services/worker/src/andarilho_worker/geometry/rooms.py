"""Rooms, walls, doors and furniture for a proposal, from the walk and the scene analysis.

The camera path is the evidence of where the rooms are. Each camera belongs to the room of
its nearest keyframe, and the analysis labels keyframes with rooms. A room's outline is the
bounding box of its cameras, widened by the proposal margin and snapped to the wall-height
points nearest it, as the single-room proposal is. Rooms that overlap are split midway
between their camera clouds. Doors go where the walk crosses from one room into another and
where the analysis saw a door in a wall; furniture goes where the analysis saw it on the floor.

Touching rooms share a wall, which validation does not allow between floor polygons, so the
floor of several rooms is the outline of their union.
"""

from __future__ import annotations

import copy
import math
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from andarilho_worker.geometry.frame import PATH_MARGIN_METERS, nearest_distance
from andarilho_worker.geometry.grounding import GroundedDoor, GroundedObject
from andarilho_worker.geometry.layout import floor_supports, rectangle_layout
from andarilho_worker.geometry.polygon import signed_area

# How far a side may move, and what a wall must look like to move it. The points are
# histogrammed in bins across the window, so a wall shows as a peak above the clutter.
SNAP_RANGE = 0.75  # metres either side of a side, where its wall is looked for
SNAP_BIN = 0.05  # metres per histogram bin
SNAP_MIN_POINTS = 8  # the peak bin holds at least this many points
SNAP_PEAK_RATIO = 2.0  # ... and at least this many times the median non-empty bin
SNAP_MIN_SPAN = 1.5  # the rectangle never shrinks below this on either axis, metres
SNAP_CLEARANCE = 0.3  # every camera keeps at least this distance from a moved side
# Wall-height evidence: above the floor clutter and below the ceiling.
WALL_MIN_HEIGHT = 0.3  # metres above the floor
WALL_MAX_SHARE = 0.85  # of the room height
EXTEND_REACH = 4.0  # metres a side may be pushed out to a wall it did not snap to

# (side, coordinate of its normal in (x, y, z): 0 for X or 2 for Z, then 0 for the minimum
# or 1 for the maximum of that axis).
SIDES: tuple[tuple[str, int, int], ...] = (
    ("west", 0, 0),
    ("east", 0, 1),
    ("south", 2, 0),
    ("north", 2, 1),
)

# Rooms and walls.
ROOM_MARGIN = PATH_MARGIN_METERS  # the proposal around a room's cameras
MIN_ROOM_KEYFRAMES = 2  # a room needs this many keyframes ...
MIN_ROOM_CAMERAS = 4  # ... and this many cameras, or it joins the room before it
MIN_SPLIT_SIDE = 0.1  # an overlap is never split to leave a room thinner than this
MIN_WALL = 0.05  # shorter pieces of wall are not built
SHARED_EPS = 1e-6  # two sides lie on one line when their coordinates agree to this
WALL_THICKNESS = 0.1
WALL_CONFIDENCE = 0.3  # the rectangle proposal's wall confidence
WALK_CROSSING_TOLERANCE = 1.0  # metres a walk step between two rooms may pass a shared wall by
DOOR_WIDTH = 0.9  # an open doorway the walk passes through
DOOR_HEIGHT = 2.05
OPEN_DOOR_CONFIDENCE = 0.5
MIN_DOOR_WIDTH = 0.6
EDGE_MARGIN = 0.05  # a door keeps this far from the ends of its wall
WALL_SIDE_PROBE = 0.25  # metres either side of a wall, to tell interior from exterior

# Grounded doors.
DOOR_WALL_REACH = 0.5  # a door may sit this far from a wall's centreline
DOOR_WALL_ANGLE = math.radians(25.0)
DOOR_WIDTH_RANGE = (0.6, 1.6)
DOOR_TOP_RANGE = (1.8, 2.3)
DUPLICATE_DOOR_RANGE = 0.4  # two doors this close on one wall are one door
MAX_DOORS_PER_WALL = 8
GROUNDED_DOOR_CONFIDENCE = 0.5

# Grounded furniture.
MAX_FURNITURE = 64
FURNITURE_WALL_REACH = 0.6  # an object this close to a wall faces away from it
WALL_FACE_GAP = 0.02  # the back of an object stands this far off the wall's face
ROOM_CLEARANCE = 0.1  # an object fits its room with this much to spare
ROOM_HEIGHT_MARGIN = 0.02
OVERLAP_LIMIT = 0.4  # of its own area, an object may overlap one placed before it
OUTSIDE_REACH = 0.5  # an object this far outside the floor is moved in, else dropped
MIN_FURNITURE_SIDE = 0.05
DEFAULT_ROOM_HEIGHT = 2.5
SPAWN_CLEARANCE = 0.35  # metres between the spawn and a solid object
SPAWN_GRID = 0.1
FACE_PROBE = 0.2  # metres off a wall, to read which side of it is the room
GRID_STEP = 0.05
FIT_RADIUS = 0.6  # how far an object may slide to fit inside an irregular floor


@dataclass(frozen=True, eq=False)
class RoomCluster:
    """The walk's cameras that belong to one room of the analysis."""

    id: str  # the room id the analysis gave, for example "r1"
    indices: np.ndarray  # positions of its cameras in the walk, in walk order
    cameras: np.ndarray  # (n, 2) camera (x, z) positions, in walk order
    keyframes: tuple[int, ...]  # the keyframes the analysis saw this room in
    entry: tuple[float, float] | None  # midpoint of the walk's first step into it, if any


# --------------------------------------------------------------------------
# Snapping and extending sides to wall-height density
# --------------------------------------------------------------------------


def _threshold(counts: np.ndarray) -> float:
    """The fewest points a bin may hold to read as a wall, given the typical bin."""
    return max(float(SNAP_MIN_POINTS), SNAP_PEAK_RATIO * float(np.median(counts[counts > 0])))


def _density_peak(values: np.ndarray, side: float) -> float | None:
    """The median of the densest bin of ``values`` near ``side``, when it reads as a wall.

    The bins span SNAP_RANGE either side of ``side``. The peak must pass ``_threshold``, so a
    flat scatter of clutter never moves a side.
    """
    bins = round(2 * SNAP_RANGE / SNAP_BIN)
    index = np.floor((values - (side - SNAP_RANGE)) / SNAP_BIN)
    index = np.clip(index, 0, bins - 1).astype(np.int64)
    counts = np.bincount(index, minlength=bins)
    if not counts.any():
        return None
    peak = int(np.argmax(counts))
    if counts[peak] < _threshold(counts):
        return None
    return float(np.median(values[index == peak]))


def _outward_peak(distances: np.ndarray, reach: float) -> float | None:
    """The median distance of the nearest bin that reads as a wall, within ``reach``.

    ``distances`` are metres outward from a side, each in [0, reach). A bin reads as a wall
    when it passes ``_threshold`` and holds at least half the strongest bin of the window: a
    wall the walk never reached is then found from its near face, while the chance
    fluctuations of walls running alongside the way out are not taken for one.
    """
    bins = max(1, math.ceil(reach / SNAP_BIN))
    index = np.clip(np.floor(distances / SNAP_BIN), 0, bins - 1).astype(np.int64)
    counts = np.bincount(index, minlength=bins)
    if not counts.any():
        return None
    passing = np.flatnonzero((counts >= _threshold(counts)) & (2 * counts >= counts.max()))
    if len(passing) == 0:
        return None
    return float(np.median(distances[index == int(passing[0])]))


def snap_rectangle(
    min_xz: Any,
    max_xz: Any,
    *,
    points: np.ndarray,
    cameras: np.ndarray,
    height: float,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Move each side of a rectangle onto the wall-height density peak of the points near it.

    ``points`` (x, y, z) are the evidence's sparse points and ``cameras`` its camera
    centres, both in the canonical world. A side moves only when its peak passes
    ``_density_peak``, the rectangle stays at least SNAP_MIN_SPAN wide and every camera
    keeps SNAP_CLEARANCE from the new side. Sides are tried in ``SIDES`` order, each
    against the rectangle as already snapped. Returns the new corners and the shift in
    metres of each side that moved.
    """
    lo = np.asarray(min_xz, dtype=np.float64).copy()
    hi = np.asarray(max_xz, dtype=np.float64).copy()
    moved: dict[str, float] = {}
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(cameras) == 0:
        return lo.tolist(), hi.tolist(), moved
    band = (points[:, 1] >= WALL_MIN_HEIGHT) & (points[:, 1] <= WALL_MAX_SHARE * height)
    walls = points[band]
    for name, axis, which in SIDES:
        # The rectangle is indexed (x, z) and the points (x, y, z): ``k`` is this side's
        # rectangle index, ``other`` the other horizontal one, ``along`` the points' one.
        k, other, along = axis // 2, 1 - axis // 2, 2 - axis
        side = float((lo if which == 0 else hi)[k])
        in_extent = (walls[:, along] >= lo[other]) & (walls[:, along] <= hi[other])
        near = walls[in_extent & (np.abs(walls[:, axis] - side) <= SNAP_RANGE)]
        new = _density_peak(near[:, axis], side)
        if new is None:
            continue
        candidate_lo, candidate_hi = lo.copy(), hi.copy()
        if which == 0:
            candidate_lo[k] = new
            gap = cameras[:, axis] - new
        else:
            candidate_hi[k] = new
            gap = new - cameras[:, axis]
        if candidate_hi[k] - candidate_lo[k] < SNAP_MIN_SPAN:
            continue
        if gap.min() < SNAP_CLEARANCE:
            continue
        lo, hi = candidate_lo, candidate_hi
        moved[name] = new - side
    return lo.tolist(), hi.tolist(), moved


def extend_unsnapped_sides(
    min_xz: Any,
    max_xz: Any,
    *,
    points: np.ndarray,
    cameras: np.ndarray,
    height: float,
    snapped: Collection[str],
    reach: float = EXTEND_REACH,
    reach_limits: Mapping[str, float] | None = None,
) -> tuple[list[float], list[float], dict[str, float]]:
    """Push every side that snap_rectangle left alone out to the wall beyond it.

    Looks outward from each side not in ``snapped`` for the nearest wall-height peak within
    ``reach`` metres (narrowed per side by ``reach_limits``), with the same tests as the snap.
    A side moves only when the new position keeps every camera SNAP_CLEARANCE inside. Returns
    the new corners and the shift of each side that moved.
    """
    lo = np.asarray(min_xz, dtype=np.float64).copy()
    hi = np.asarray(max_xz, dtype=np.float64).copy()
    moved: dict[str, float] = {}
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0 or len(cameras) == 0:
        return lo.tolist(), hi.tolist(), moved
    band = (points[:, 1] >= WALL_MIN_HEIGHT) & (points[:, 1] <= WALL_MAX_SHARE * height)
    walls = points[band]
    limits = reach_limits or {}
    for name, axis, which in SIDES:
        if name in snapped:
            continue
        k, other, along = axis // 2, 1 - axis // 2, 2 - axis
        sign = 1.0 if which else -1.0
        side = float((hi if which else lo)[k])
        cap = min(reach, limits.get(name, math.inf))
        if cap <= 0:
            continue
        in_extent = (walls[:, along] >= lo[other]) & (walls[:, along] <= hi[other])
        outward = (walls[:, axis] - side) * sign
        near = walls[in_extent & (outward >= 0) & (outward < cap)]
        found = _outward_peak((near[:, axis] - side) * sign, cap) if len(near) else None
        if found is None or found <= 0:
            continue
        new = side + sign * found
        if (sign * (new - cameras[:, axis])).min() < SNAP_CLEARANCE:
            continue
        if which:
            hi[k] = new
        else:
            lo[k] = new
        moved[name] = new - side
    return lo.tolist(), hi.tolist(), moved


# --------------------------------------------------------------------------
# Segmenting the walk into rooms
# --------------------------------------------------------------------------


def _keyframe_centre(keyframe: Mapping[str, Any]) -> np.ndarray:
    rotation = np.asarray(keyframe["rotation"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(keyframe["translation"], dtype=np.float64).reshape(3)
    return -rotation.T @ translation


def segment_rooms(
    keyframes: Sequence[Mapping[str, Any]],
    frames_reply: Sequence[Mapping[str, Any]],
    cameras_xz: Any,
) -> list[RoomCluster]:
    """Assign each camera of the walk to the room of its nearest labelled keyframe.

    ``keyframes`` are the analysis's posed keyframes and ``frames_reply`` its per-frame
    replies (``room`` labels them; a null room is no evidence). ``cameras_xz`` are the walk's
    camera centres in walk order. Rooms come back in walk order, by the first camera in each.
    A room with fewer than MIN_ROOM_KEYFRAMES keyframes or MIN_ROOM_CAMERAS cameras joins the
    room before it (the first room joins the one after it).
    """
    cameras = np.asarray(cameras_xz, dtype=np.float64).reshape(-1, 2)
    room_of_frame = {int(entry["frame"]): entry.get("room") for entry in frames_reply}
    labelled: list[tuple[int, str, np.ndarray]] = []
    for keyframe in keyframes:
        room = room_of_frame.get(int(keyframe["frame"]))
        if room is None:
            continue
        centre = _keyframe_centre(keyframe)
        labelled.append((int(keyframe["frame"]), str(room), centre[[0, 2]]))
    if not labelled or len(cameras) == 0:
        return []
    centres = np.array([xz for _, _, xz in labelled])
    nearest = np.argmin(np.linalg.norm(cameras[:, None, :] - centres[None, :, :], axis=2), axis=1)
    owner = [labelled[int(i)][1] for i in nearest]
    frames_of: dict[str, list[int]] = {}
    for frame, room, _ in labelled:
        frames_of.setdefault(room, []).append(frame)
    order = list(dict.fromkeys(owner))  # rooms in walk order
    while len(order) > 1:
        cameras_in = Counter(owner)
        small = next(
            (
                i
                for i, room in enumerate(order)
                if len(frames_of.get(room, [])) < MIN_ROOM_KEYFRAMES
                or cameras_in[room] < MIN_ROOM_CAMERAS
            ),
            None,
        )
        if small is None:
            break
        room = order[small]
        target = order[small - 1] if small > 0 else order[small + 1]
        owner = [target if o == room else o for o in owner]
        frames_of[target] = frames_of.get(target, []) + frames_of.pop(room, [])
        order.pop(small)
    clusters: list[RoomCluster] = []
    for room in order:
        indices = np.array([i for i, o in enumerate(owner) if o == room], dtype=np.int64)
        first = int(indices[0])
        entry = None
        if first > 0:
            before, after = cameras[first - 1], cameras[first]
            entry = (float((before[0] + after[0]) / 2), float((before[1] + after[1]) / 2))
        clusters.append(
            RoomCluster(
                id=room,
                indices=indices,
                cameras=cameras[indices],
                keyframes=tuple(sorted(frames_of.get(room, []))),
                entry=entry,
            )
        )
    return clusters


# --------------------------------------------------------------------------
# Laying out several rooms
# --------------------------------------------------------------------------


def _room_box(cameras: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xz = cameras[:, [0, 2]]
    return xz.min(axis=0) - ROOM_MARGIN, xz.max(axis=0) + ROOM_MARGIN


def _point_owners(points: np.ndarray, room_cameras: list[np.ndarray]) -> np.ndarray:
    """The room whose cameras are nearest to each point (an index into ``room_cameras``)."""
    if len(room_cameras) == 1:
        return np.zeros(len(points), dtype=np.int64)
    xz = points[:, [0, 2]]
    distances = np.stack([nearest_distance(xz, cams[:, [0, 2]]) for cams in room_cameras])
    return np.argmin(distances, axis=0)


def _reach_limits(
    lo: np.ndarray, hi: np.ndarray, others: list[tuple[np.ndarray, np.ndarray]]
) -> dict[str, float]:
    """How far each side may move before it would grow into another room's outline."""
    limits: dict[str, float] = {}
    for name, axis, which in SIDES:
        k, other = axis // 2, 1 - axis // 2
        gaps: list[float] = []
        for other_lo, other_hi in others:
            if not (other_lo[other] < hi[other] and other_hi[other] > lo[other]):
                continue  # the two never face each other across this side
            if which and other_lo[k] >= hi[k]:
                gaps.append(float(other_lo[k] - hi[k]))
            if not which and other_hi[k] <= lo[k]:
                gaps.append(float(lo[k] - other_hi[k]))
        if gaps:
            limits[name] = max(0.0, min(gaps))
    return limits


def _split_overlaps(boxes: list[list[np.ndarray]], room_cameras: list[np.ndarray]) -> None:
    """Split every overlap of two rooms midway between their camera clouds, in place.

    The split runs along the axis where the two rooms overlap least, which gives up the least
    floor of either room.
    """
    # A split can open an overlap with a third room, so the pass repeats until nothing moves;
    # the bound only guards against a loop that never settles.
    for _ in range(10):
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                overlap = np.minimum(boxes[i][1], boxes[j][1]) - np.maximum(
                    boxes[i][0], boxes[j][0]
                )
                if overlap.min() <= SHARED_EPS:
                    continue
                axis = int(np.argmin(overlap))
                column = (0, 2)[axis]  # the camera column of this rectangle axis
                cloud_i, cloud_j = room_cameras[i][:, column], room_cameras[j][:, column]
                if cloud_i.mean() <= cloud_j.mean():
                    low, high, low_cloud, high_cloud = i, j, cloud_i, cloud_j
                else:
                    low, high, low_cloud, high_cloud = j, i, cloud_j, cloud_i
                split = (float(low_cloud.max()) + float(high_cloud.min())) / 2
                split = min(
                    max(split, boxes[low][0][axis] + MIN_SPLIT_SIDE),
                    boxes[high][1][axis] - MIN_SPLIT_SIDE,
                )
                boxes[low][1][axis] = split
                boxes[high][0][axis] = split
                changed = True
        if not changed:
            return


def rooms_layout(
    rooms: Sequence[RoomCluster],
    *,
    points: np.ndarray,
    cameras: np.ndarray,
    height: float,
    floor_y: float,
    spawn: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A layout with one rectangle per room, walls between them and doors where the walk passes.

    ``points`` are the evidence's sparse points and ``cameras`` the walk's camera centres
    (x, y, z), the same array the rooms were segmented from. One room gives exactly the
    ``rectangle_layout`` of its outline. Several rooms share a floor, the outline of their
    union, with one wall per edge piece and one interior wall per shared edge.
    """
    if not rooms:
        raise ValueError("at least one room is needed")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cameras = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    room_cameras = [cameras[room.indices] for room in rooms]
    owners = _point_owners(points, room_cameras)
    boxes: list[list[np.ndarray]] = []
    snapped: list[set[str]] = []
    own_points: list[np.ndarray] = []
    for i, cams in enumerate(room_cameras):
        lo, hi = _room_box(cams)
        own = points[owners == i]
        lo, hi, moved = snap_rectangle(lo, hi, points=own, cameras=cams, height=height)
        boxes.append([np.array(lo), np.array(hi)])
        snapped.append(set(moved))
        own_points.append(own)
    for i, cams in enumerate(room_cameras):
        others = [(boxes[j][0], boxes[j][1]) for j in range(len(boxes)) if j != i]
        lo, hi, _ = extend_unsnapped_sides(
            boxes[i][0],
            boxes[i][1],
            points=own_points[i],
            cameras=cams,
            height=height,
            snapped=snapped[i],
            reach_limits=_reach_limits(boxes[i][0], boxes[i][1], others),
        )
        boxes[i] = [np.array(lo), np.array(hi)]
    _split_overlaps(boxes, room_cameras)
    rects = [(float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])) for lo, hi in boxes]
    if len(rects) == 1:
        x0, z0, x1, z1 = rects[0]
        return rectangle_layout((x0, z0), (x1, z1), elevation=floor_y, height=height, spawn=spawn)
    return _assemble(rects, rooms, cameras, height=height, floor_y=floor_y, spawn=spawn)


def _union_outline(
    rects: Sequence[tuple[float, float, float, float]],
) -> tuple[list[list[list[float]]], list[list[list[float]]]]:
    """The outer outlines and the holes of the union of rectangles (x0, z0, x1, z1).

    Built on the grid of every rectangle edge: each filled cell contributes its exposed
    sides, traced into closed loops. Loops run counter-clockwise for outlines, clockwise for
    holes, and straight runs are merged into single edges.
    """
    xs = sorted({v for x0, _, x1, _ in rects for v in (x0, x1)})
    zs = sorted({v for _, z0, _, z1 in rects for v in (z0, z1)})
    cells = np.zeros((len(xs) - 1, len(zs) - 1), dtype=bool)
    for x0, z0, x1, z1 in rects:
        cells[xs.index(x0) : xs.index(x1), zs.index(z0) : zs.index(z1)] = True

    def filled(i: int, j: int) -> bool:
        return 0 <= i < cells.shape[0] and 0 <= j < cells.shape[1] and bool(cells[i, j])

    edges: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(a: tuple[int, int], b: tuple[int, int]) -> None:
        edges.setdefault(a, []).append(b)

    for i, j in zip(*np.nonzero(cells), strict=True):
        i, j = int(i), int(j)
        if not filled(i, j - 1):
            add((i, j), (i + 1, j))
        if not filled(i + 1, j):
            add((i + 1, j), (i + 1, j + 1))
        if not filled(i, j + 1):
            add((i + 1, j + 1), (i, j + 1))
        if not filled(i - 1, j):
            add((i, j + 1), (i, j))
    loops: list[list[tuple[float, float]]] = []
    while edges:
        start = next(iter(edges))
        loop = [start]
        current = start
        while True:
            options = edges[current]
            following = options.pop()
            if not options:
                del edges[current]
            if following == start:
                break
            loop.append(following)
            current = following
        loops.append([(xs[a], zs[b]) for a, b in loop])
    outers: list[list[list[float]]] = []
    holes: list[list[list[float]]] = []
    for loop in loops:
        corners = _straight_runs_merged(loop)
        ring = [[float(x), float(z)] for x, z in corners]
        if signed_area(ring) > 0:
            outers.append(ring)
        else:
            holes.append(ring)
    return outers, holes


def _straight_runs_merged(loop: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    kept = []
    count = len(loop)
    for k in range(count):
        before, here, after = loop[k - 1], loop[k], loop[(k + 1) % count]
        turn = (here[0] - before[0]) * (after[1] - here[1]) - (here[1] - before[1]) * (
            after[0] - here[0]
        )
        if abs(turn) > SHARED_EPS:
            kept.append(here)
    return kept


def _wall_segments(
    rects: Sequence[tuple[float, float, float, float]],
) -> list[tuple[int, float, float, float, tuple[int, int] | None]]:
    """Wall pieces as (fixed axis, coordinate, start, end, rooms): interior when rooms is set.

    An edge is cut where another room shares it: the shared part is one interior wall (made
    once, from the side that faces +) and the rest of the edge is exterior wall.
    """
    segments: list[tuple[int, float, float, float, tuple[int, int] | None]] = []
    for i, rect in enumerate(rects):
        lo = (rect[0], rect[1])
        hi = (rect[2], rect[3])
        for axis in (0, 1):
            other = 1 - axis
            for sign in (1, -1):
                coord = hi[axis] if sign > 0 else lo[axis]
                start, end = lo[other], hi[other]
                shared: list[tuple[float, float, int]] = []
                for j, rect_j in enumerate(rects):
                    if j == i:
                        continue
                    lo_j = (rect_j[0], rect_j[1])
                    hi_j = (rect_j[2], rect_j[3])
                    partner = lo_j[axis] if sign > 0 else hi_j[axis]
                    if abs(partner - coord) > SHARED_EPS:
                        continue
                    a0, a1 = max(start, lo_j[other]), min(end, hi_j[other])
                    if a1 - a0 > MIN_WALL:
                        shared.append((a0, a1, j))
                if sign > 0:
                    segments += [(axis, coord, a0, a1, (i, j)) for a0, a1, j in shared]
                for p0, p1 in _uncovered(start, end, [(a0, a1) for a0, a1, _ in shared]):
                    if p1 - p0 > MIN_WALL:
                        segments.append((axis, coord, p0, p1, None))
    return segments


def _uncovered(
    start: float, end: float, covered: list[tuple[float, float]]
) -> list[tuple[float, float]]:
    pieces: list[tuple[float, float]] = []
    cursor = start
    for a0, a1 in sorted(covered):
        if a0 > cursor:
            pieces.append((cursor, a0))
        cursor = max(cursor, a1)
    if end > cursor:
        pieces.append((cursor, end))
    return pieces


def _walk_crossing(
    segment: tuple[int, float, float, float, tuple[int, int] | None],
    walk_xz: np.ndarray,
    camera_room: np.ndarray,
) -> float | None:
    """Where the walk passes through an interior wall, along the wall; None when it never does.

    The steps of the walk that go from one of the wall's rooms to the other are the crossings.
    Rooms are split midway between keyframes, so a step may switch rooms a little past the
    wall: the answer is the step that passes nearest the wall, when it is within
    WALK_CROSSING_TOLERANCE of it, at the place along the wall where it is nearest.
    """
    axis, coord, start, end, rooms = segment
    if rooms is None:
        return None
    other = 1 - axis
    best: tuple[float, float] | None = None  # (distance from the wall, place along it)
    for k in range(len(walk_xz) - 1):
        if {int(camera_room[k]), int(camera_room[k + 1])} != set(rooms):
            continue
        p, q = walk_xz[k], walk_xz[k + 1]
        dp, dq = p[axis] - coord, q[axis] - coord
        if dp * dq <= 0 and dp != dq:
            t = dp / (dp - dq)  # the step crosses the wall's line
            across, along = 0.0, p[other] + t * (q[other] - p[other])
        else:
            near = p if abs(dp) <= abs(dq) else q
            across, along = abs(near[axis] - coord), float(near[other])
        beyond = max(start - along, along - end, 0.0)
        distance = math.hypot(across, beyond)
        if best is None or distance < best[0]:
            best = (distance, _clamp(along, start, end))
    if best is None or best[0] > WALK_CROSSING_TOLERANCE:
        return None
    return best[1]


def _assemble(
    rects: list[tuple[float, float, float, float]],
    rooms: Sequence[RoomCluster],
    cameras: np.ndarray,
    *,
    height: float,
    floor_y: float,
    spawn: Mapping[str, Any] | None,
) -> dict[str, Any]:
    outers, holes = _union_outline(rects)
    floor = {"id": "floor-0", "elevation": float(floor_y), "polygons": outers, "holes": holes}
    walls: list[dict[str, Any]] = []
    openings: list[dict[str, Any]] = []
    camera_room = np.full(len(cameras), -1, dtype=np.int64)
    for index, room in enumerate(rooms):
        camera_room[room.indices] = index
    walk_xz = cameras[:, [0, 2]]
    for axis, coord, start, end, rooms_pair in _wall_segments(rects):
        wall_id = f"wall-{len(walls) + 1}"
        if axis == 0:
            p0, p1 = [coord, start], [coord, end]
        else:
            p0, p1 = [start, coord], [end, coord]
        walls.append(
            {
                "id": wall_id,
                "floorId": "floor-0",
                "start": p0,
                "end": p1,
                "thickness": WALL_THICKNESS,
                "height": height,
                "confidence": WALL_CONFIDENCE,
                "provenance": "proposal",
            }
        )
        if rooms_pair is None:
            continue
        along = _walk_crossing((axis, coord, start, end, rooms_pair), walk_xz, camera_room)
        if along is None:
            continue
        length = end - start
        width = min(DOOR_WIDTH, length - 2 * EDGE_MARGIN)
        if width < MIN_DOOR_WIDTH:
            continue
        offset = _clamp(along - start - width / 2, EDGE_MARGIN, length - width - EDGE_MARGIN)
        openings.append(
            {
                "id": f"door-{len(openings) + 1}",
                "wallId": wall_id,
                "offset": offset,
                "width": width,
                "bottom": 0.0,
                "top": min(DOOR_HEIGHT, height - EDGE_MARGIN),
                "state": "open",
                "confidence": OPEN_DOOR_CONFIDENCE,
                "provenance": "proposal",
            }
        )
    return {
        "schemaVersion": "1.0",
        "floors": [floor],
        "walls": walls,
        "openings": openings,
        "obstacles": [],
        "furniture": [],
        "landmarks": [],
        "spawn": _room_spawn(floor, rects, spawn, floor_y),
    }


def _room_spawn(
    floor: Mapping[str, Any],
    rects: Sequence[tuple[float, float, float, float]],
    spawn: Mapping[str, Any] | None,
    floor_y: float,
) -> dict[str, Any]:
    if spawn is not None:
        x, z = float(spawn["feet"][0]), float(spawn["feet"][2])
        if floor_supports(floor, (x, z)):
            return {
                "feet": [x, floor_y, z],
                "yawRadians": float(spawn["yawRadians"]),
                "floorId": "floor-0",
            }
    x0, z0, x1, z1 = rects[0]
    return {
        "feet": [(x0 + x1) / 2, floor_y, (z0 + z1) / 2],
        "yawRadians": 0.0,
        "floorId": "floor-0",
    }


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


# --------------------------------------------------------------------------
# Grounded doors
# --------------------------------------------------------------------------


def _segment_distance(point: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
    span = end - start
    length_sq = float(np.dot(span, span))
    t = 0.0 if length_sq == 0 else _clamp(float(np.dot(point - start, span)) / length_sq, 0, 1)
    return float(np.linalg.norm(point - (start + t * span)))


def _wall_for_door(
    walls: Sequence[Mapping[str, Any]], door: GroundedDoor
) -> Mapping[str, Any] | None:
    """The wall that a door sits in: its centreline is near the door, and runs along it."""
    midpoint = np.asarray(door.position, dtype=np.float64)
    direction = np.asarray(door.direction, dtype=np.float64)
    best: tuple[float, Mapping[str, Any]] | None = None
    for wall in walls:
        start, end = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
        length = float(np.linalg.norm(end - start))
        if length < MIN_WALL:
            continue
        along = (end - start) / length
        if abs(float(np.dot(along, direction))) < math.cos(DOOR_WALL_ANGLE):
            continue
        distance = _segment_distance(midpoint, start, end)
        if distance > DOOR_WALL_REACH:
            continue
        if best is None or distance < best[0]:
            best = (distance, wall)
    return None if best is None else best[1]


def _between_rooms(floor: Mapping[str, Any], wall: Mapping[str, Any]) -> bool:
    """Whether both faces of a wall are inside the floor, so the wall parts two rooms."""
    start, end = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
    direction = (end - start) / np.linalg.norm(end - start)
    normal = np.array([-direction[1], direction[0]])
    middle = (start + end) / 2
    return floor_supports(floor, middle + normal * WALL_SIDE_PROBE) and floor_supports(
        floor, middle - normal * WALL_SIDE_PROBE
    )


def _next_id(prefix: str, taken: set[str]) -> str:
    number = 1
    while f"{prefix}-{number}" in taken:
        number += 1
    return f"{prefix}-{number}"


def _ids_in_use(layout: Mapping[str, Any]) -> set[str]:
    items = [
        *layout.get("openings", []),
        *layout.get("obstacles", []),
        *layout.get("furniture", []),
    ]
    return {str(item["id"]) for item in items}


def _take_door(
    openings: list[dict[str, Any]],
    grounded: set[str],
    candidate: dict[str, Any],
    wall: Mapping[str, Any],
) -> bool:
    """Add a grounded door, or keep the better of two that are the same door. True if added."""
    same_wall = [o for o in openings if o["wallId"] == wall["id"]]
    low, high = candidate["offset"], candidate["offset"] + candidate["width"]
    overlapping = [o for o in same_wall if o["offset"] < high and low < o["offset"] + o["width"]]
    if overlapping:
        if len(overlapping) > 1 or overlapping[0]["id"] not in grounded:
            return False  # a door already there, or the spot is someone else's
        old = overlapping[0]
        old_centre = old["offset"] + old["width"] / 2
        new_centre = candidate["offset"] + candidate["width"] / 2
        if abs(new_centre - old_centre) > DUPLICATE_DOOR_RANGE:
            return False
        middle = _wall_length(wall) / 2
        wider = candidate["width"] > old["width"] + 1e-9
        same_width = abs(candidate["width"] - old["width"]) <= 1e-9
        if not (wider or (same_width and abs(new_centre - middle) < abs(old_centre - middle))):
            return False
        openings.remove(old)
        candidate["id"] = old["id"]
        openings.append(candidate)
        return True
    if sum(1 for o in same_wall if o["id"] in grounded) >= MAX_DOORS_PER_WALL:
        return False
    openings.append(candidate)
    grounded.add(candidate["id"])
    return True


def _wall_length(wall: Mapping[str, Any]) -> float:
    return float(np.linalg.norm(np.asarray(wall["end"], float) - np.asarray(wall["start"], float)))


def attach_doors(
    layout: Mapping[str, Any], doors: Sequence[GroundedDoor], room_ids: Sequence[str]
) -> dict[str, Any]:
    """The layout with each grounded door cut into the wall it sits in.

    A door is open only on an interior wall, between two rooms of the layout, and only when
    the agent did not say it leads to a room the layout lacks; every other door is a closed,
    textured panel. Doors that overlap an opening already there are not added again, and two
    grounded doors on one wall within DUPLICATE_DOOR_RANGE are kept as the wider one.
    """
    result = copy.deepcopy(dict(layout))
    walls = list(result.get("walls", []))
    floor = result["floors"][0]
    openings: list[dict[str, Any]] = result.setdefault("openings", [])
    known_rooms = set(room_ids)
    grounded: set[str] = set()
    for door in doors:
        wall = _wall_for_door(walls, door)
        if wall is None:
            continue
        length = _wall_length(wall)
        width = _clamp(door.width, *DOOR_WIDTH_RANGE)
        low, high = EDGE_MARGIN, length - width - EDGE_MARGIN
        if high < low:
            continue  # too short for this door
        start, end = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
        along = float(np.dot(np.asarray(door.position, float) - start, (end - start) / length))
        top = min(_clamp(door.top, *DOOR_TOP_RANGE), float(wall["height"]) - EDGE_MARGIN)
        interior = _between_rooms(floor, wall)
        leads_ok = door.leads_to is None or door.leads_to in known_rooms
        candidate = {
            "id": _next_id("door", _ids_in_use(result)),
            "wallId": wall["id"],
            "offset": _clamp(along - width / 2, low, high),
            "width": width,
            "bottom": 0.0,
            "top": top,
            "state": "open" if interior and leads_ok else "closed",
            "confidence": GROUNDED_DOOR_CONFIDENCE,
            "provenance": "proposal",
        }
        _take_door(openings, grounded, candidate, wall)
    return result


# --------------------------------------------------------------------------
# Grounded furniture
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _WallHit:
    distance: float  # from the object's centre to the wall's centreline
    foot: np.ndarray  # the nearest point on the centreline
    direction: np.ndarray  # the wall's unit direction
    thickness: float


@dataclass(frozen=True, eq=False)
class _Footprint:
    centre: np.ndarray  # (x, z)
    axes: np.ndarray  # (2, 2): the object's width and depth directions, unit
    half: np.ndarray  # (half width, half depth)
    solid: bool

    def corners(self) -> np.ndarray:
        ex, ez = self.axes
        hw, hd = self.half
        return np.array(
            [
                self.centre - hw * ex - hd * ez,
                self.centre + hw * ex - hd * ez,
                self.centre + hw * ex + hd * ez,
                self.centre - hw * ex + hd * ez,
            ]
        )


def _footprint(
    position: np.ndarray, yaw: float, width: float, depth: float, solid: bool
) -> _Footprint:
    # The object frame of the layout contract: local x across the width, local z front to back.
    ex = np.array([math.cos(yaw), -math.sin(yaw)])
    ez = np.array([math.sin(yaw), math.cos(yaw)])
    return _Footprint(position, np.vstack([ex, ez]), np.array([width / 2, depth / 2]), solid)


def _samples(footprint: _Footprint) -> np.ndarray:
    corners = footprint.corners()
    edges = (corners + np.roll(corners, -1, axis=0)) / 2
    return np.vstack([corners, edges, footprint.centre])


def _inside_floor(floor: Mapping[str, Any], footprint: _Footprint) -> bool:
    return all(floor_supports(floor, point) for point in _samples(footprint))


def _floor_bounds(floor: Mapping[str, Any]) -> tuple[float, float, float, float]:
    points = np.array([p for poly in floor["polygons"] for p in poly], dtype=np.float64)
    lo, hi = points.min(axis=0), points.max(axis=0)
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _grid(reach: float, step: float) -> np.ndarray:
    steps = np.arange(-reach, reach + step / 2, step)
    offsets = np.array([(a, b) for a in steps for b in steps], dtype=np.float64)
    return offsets[np.argsort(np.linalg.norm(offsets, axis=1), kind="stable")]


def _nearest_inside(floor: Mapping[str, Any], position: np.ndarray) -> np.ndarray | None:
    for offset in _grid(OUTSIDE_REACH, GRID_STEP):
        candidate = position + offset
        if floor_supports(floor, candidate):
            return candidate
    return None


def _nearest_wall(walls: Sequence[Mapping[str, Any]], position: np.ndarray) -> _WallHit | None:
    best: _WallHit | None = None
    for wall in walls:
        start, end = np.asarray(wall["start"], float), np.asarray(wall["end"], float)
        length = float(np.linalg.norm(end - start))
        if length < MIN_WALL:
            continue
        direction = (end - start) / length
        t = _clamp(float(np.dot(position - start, direction)), 0.0, length)
        foot = start + t * direction
        distance = float(np.linalg.norm(position - foot))
        if best is None or distance < best.distance:
            best = _WallHit(distance, foot, direction, float(wall["thickness"]))
    return best


def _room_normal(floor: Mapping[str, Any], hit: _WallHit, position: np.ndarray) -> np.ndarray:
    """The unit normal of a wall that points into the room the object stands in."""
    normal = np.array([-hit.direction[1], hit.direction[0]])
    room_on_plus = floor_supports(floor, hit.foot + normal * FACE_PROBE)
    room_on_minus = floor_supports(floor, hit.foot - normal * FACE_PROBE)
    if room_on_plus != room_on_minus:
        return normal if room_on_plus else -normal
    # Both faces are the room (a wall between two rooms): the object's own side wins.
    return normal if float(np.dot(position - hit.foot, normal)) >= 0 else -normal


def _snap_quarter_turn(angle: float) -> float:
    return round(angle / (math.pi / 2)) * (math.pi / 2)


def _fit_size(
    floor: Mapping[str, Any], yaw: float, width: float, depth: float
) -> tuple[float, float] | None:
    """Shrink a footprint so it fits the floor's bounds with ROOM_CLEARANCE to spare."""
    x0, z0, x1, z1 = _floor_bounds(floor)
    cos, sin = abs(math.cos(yaw)), abs(math.sin(yaw))
    factor = 1.0
    for room_side, half in (
        (x1 - x0, width / 2 * cos + depth / 2 * sin),
        (z1 - z0, width / 2 * sin + depth / 2 * cos),
    ):
        if 2 * half > room_side - ROOM_CLEARANCE:
            factor = min(factor, (room_side - ROOM_CLEARANCE) / (2 * half))
    width, depth = width * factor, depth * factor
    if min(width, depth) < MIN_FURNITURE_SIDE:
        return None
    return width, depth


def _fit_position(
    floor: Mapping[str, Any], position: np.ndarray, yaw: float, width: float, depth: float
) -> np.ndarray | None:
    """The nearest position at which the footprint lies inside the floor, or None."""
    if _inside_floor(floor, _footprint(position, yaw, width, depth, True)):
        return position
    x0, z0, x1, z1 = _floor_bounds(floor)
    half_x = width / 2 * abs(math.cos(yaw)) + depth / 2 * abs(math.sin(yaw))
    half_z = width / 2 * abs(math.sin(yaw)) + depth / 2 * abs(math.cos(yaw))
    clamped = np.array(
        [
            _clamp(position[0], x0 + half_x + EDGE_MARGIN / 2, x1 - half_x - EDGE_MARGIN / 2),
            _clamp(position[1], z0 + half_z + EDGE_MARGIN / 2, z1 - half_z - EDGE_MARGIN / 2),
        ]
    )
    for offset in _grid(FIT_RADIUS, GRID_STEP):
        candidate = clamped + offset
        if _inside_floor(floor, _footprint(candidate, yaw, width, depth, True)):
            return candidate
    return None


def _side(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> float:
    """Positive when ``p`` lies to the left of the edge a -> b."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def _cross(
    p: tuple[float, float], q: tuple[float, float], sp: float, sq: float
) -> tuple[float, float]:
    """Where the edge p -> q crosses a clipping line, given the signed sides of its ends."""
    t = sp / (sp - sq)
    return (p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]))


def _overlap_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """The area common to two convex quadrilaterals (Sutherland-Hodgman clipping)."""
    if signed_area(clip) < 0:
        clip = clip[::-1]
    polygon = [(float(x), float(z)) for x, z in subject]
    for k in range(len(clip)):
        a = (float(clip[k][0]), float(clip[k][1]))
        b = (float(clip[(k + 1) % len(clip)][0]), float(clip[(k + 1) % len(clip)][1]))
        current, polygon = polygon, []
        for m in range(len(current)):
            p, q = current[m], current[(m + 1) % len(current)]
            sp, sq = _side(a, b, p), _side(a, b, q)
            if sq >= 0:
                if sp < 0:
                    polygon.append(_cross(p, q, sp, sq))
                polygon.append(q)
            elif sp >= 0:
                polygon.append(_cross(p, q, sp, sq))
        if not polygon:
            return 0.0
    return abs(signed_area(polygon))


def _place_one(
    obj: GroundedObject,
    floor: Mapping[str, Any],
    walls: Sequence[Mapping[str, Any]],
    room_height: float,
    viewed_from: np.ndarray | None,
) -> tuple[np.ndarray, float, float, float, float] | None:
    """Where one object stands: (position, yaw, width, depth, height), or None to drop it."""
    width, height, depth = obj.size
    position = np.asarray(obj.position, dtype=np.float64)
    if not floor_supports(floor, position):
        moved = _nearest_inside(floor, position)
        if moved is None:
            return None
        position = moved
    hit = _nearest_wall(walls, position)
    against_wall = hit is not None and hit.distance <= depth / 2 + FURNITURE_WALL_REACH
    normal = None
    if against_wall:
        normal = _room_normal(floor, hit, position)
        yaw = math.atan2(normal[0], normal[1])  # the front faces into the room
    elif viewed_from is not None:
        towards = viewed_from - position
        yaw = _snap_quarter_turn(math.atan2(towards[0], towards[1]))
    else:
        yaw = 0.0
    fitted = _fit_size(floor, yaw, width, depth)
    if fitted is None:
        return None
    width, depth = fitted
    if against_wall:
        # The back of the object stands just off the wall's face, on the room's side.
        position = hit.foot + normal * (hit.thickness / 2 + WALL_FACE_GAP + depth / 2)
    position = _fit_position(floor, position, yaw, width, depth)
    if position is None:
        return None
    return position, yaw, width, depth, min(height, room_height - ROOM_HEIGHT_MARGIN)


def place_furniture(
    layout: Mapping[str, Any], objects: Sequence[GroundedObject], cameras: Any
) -> tuple[dict[str, Any], int]:
    """The layout with grounded furniture placed against the walls, and how many were dropped.

    Objects are placed most confident first. Each faces away from the wall it stands against,
    or else the mean camera position, turned to a quarter turn. An object that would overlap a
    placed one by more than OVERLAP_LIMIT of its own area is dropped, as is one past
    MAX_FURNITURE. Afterwards the spawn moves to the nearest free point of a grid when a
    solid object stands within SPAWN_CLEARANCE of it.
    """
    result = copy.deepcopy(dict(layout))
    floor = result["floors"][0]
    walls = list(result.get("walls", []))
    room_height = max((float(w["height"]) for w in walls), default=DEFAULT_ROOM_HEIGHT)
    cams = np.asarray(cameras, dtype=np.float64).reshape(-1, 3)
    viewed_from = cams[:, [0, 2]].mean(axis=0) if len(cams) else None
    placed: list[dict[str, Any]] = []
    footprints: list[_Footprint] = []
    dropped = 0
    for obj in sorted(objects, key=lambda o: -o.confidence):
        spot = (
            None
            if len(placed) >= MAX_FURNITURE
            else _place_one(obj, floor, walls, room_height, viewed_from)
        )
        if spot is None:
            dropped += 1
            continue
        position, yaw, width, depth, height = spot
        footprint = _footprint(position, yaw, width, depth, bool(obj.solid))
        own_area = width * depth
        if any(
            _overlap_area(footprint.corners(), other.corners()) > OVERLAP_LIMIT * own_area
            for other in footprints
        ):
            dropped += 1
            continue
        footprints.append(footprint)
        placed.append(
            {
                "id": f"furniture-{len(placed) + 1}",
                "floorId": floor["id"],
                "category": obj.category,
                "label": obj.label,
                "position": [float(position[0]), float(position[1])],
                "yawRadians": float(yaw),
                "size": [float(width), float(height), float(depth)],
                "solid": bool(obj.solid),
                "parts": copy.deepcopy(obj.parts),
                "confidence": float(obj.confidence),
                "provenance": "proposal",
            }
        )
    result["furniture"] = placed
    _clear_spawn(result, floor, [f for f in footprints if f.solid])
    return result, dropped


def _distance_to_footprints(points: np.ndarray, footprints: Sequence[_Footprint]) -> np.ndarray:
    """Each point's distance to the nearest footprint (zero inside one)."""
    best = np.full(len(points), math.inf)
    for footprint in footprints:
        rel = points - footprint.centre
        local = np.abs(rel @ footprint.axes.T) - footprint.half
        best = np.minimum(best, np.linalg.norm(np.maximum(local, 0.0), axis=1))
    return best


def _clear_spawn(
    result: dict[str, Any], floor: Mapping[str, Any], solids: list[_Footprint]
) -> None:
    """Move the spawn to the nearest free grid point when a solid object is too close to it."""
    spawn = result["spawn"]
    feet = np.array([spawn["feet"][0], spawn["feet"][2]], dtype=np.float64)
    if not solids or _distance_to_footprints(feet[None, :], solids)[0] >= SPAWN_CLEARANCE:
        return
    x0, z0, x1, z1 = _floor_bounds(floor)
    xs = np.arange(math.ceil(x0 / SPAWN_GRID), math.floor(x1 / SPAWN_GRID) + 1) * SPAWN_GRID
    zs = np.arange(math.ceil(z0 / SPAWN_GRID), math.floor(z1 / SPAWN_GRID) + 1) * SPAWN_GRID
    candidates = np.array([(x, z) for x in xs for z in zs], dtype=np.float64).reshape(-1, 2)
    if len(candidates) == 0:
        return
    clear = _distance_to_footprints(candidates, solids) >= SPAWN_CLEARANCE
    candidates = candidates[clear]
    order = np.argsort(np.linalg.norm(candidates - feet, axis=1), kind="stable")
    for index in order:
        point = candidates[index]
        if floor_supports(floor, point):
            spawn["feet"] = [float(point[0]), float(spawn["feet"][1]), float(point[1])]
            return
