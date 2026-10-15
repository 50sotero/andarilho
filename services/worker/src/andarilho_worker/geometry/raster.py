"""Software depth rasterizer for a layout's triangle soup (COLMAP camera convention).

Camera coordinates follow COLMAP and OpenCV: x right, y down, z forward, and a pixel
is ``K @ (x / z, y / z, 1)``. The centre of the top-left pixel is at ``(0.5, 0.5)``,
so the array index of a continuous pixel position ``(u, v)`` is ``(floor(v), floor(u))``.
Depth is the camera z of the nearest surface along each pixel's ray.
"""

from __future__ import annotations

import numpy as np

BARY_TOLERANCE = -1e-9  # pixel centres on a shared edge belong to both triangles
DEGENERATE_AREA = 1e-12


def _clip_to_near(tri: np.ndarray, near: float) -> list[np.ndarray]:
    """Clip one camera-space triangle to ``z >= near`` (Sutherland-Hodgman on one plane)."""
    poly: list[np.ndarray] = []
    for i in range(3):
        p, q = tri[i], tri[(i + 1) % 3]
        p_in, q_in = p[2] >= near, q[2] >= near
        if p_in:
            poly.append(p)
        if p_in != q_in:
            s = (near - p[2]) / (q[2] - p[2])
            poly.append(p + s * (q - p))
    if len(poly) < 3:
        return []
    return [np.stack([poly[0], poly[k], poly[k + 1]]) for k in range(1, len(poly) - 1)]


def _rasterize(depth: np.ndarray, xy: np.ndarray, inv_z: np.ndarray) -> None:
    """Keep the minimum depth of one screen-space triangle, interpolating 1/z linearly."""
    height, width = depth.shape
    x0, y0 = xy[0]
    x1, y1 = xy[1]
    x2, y2 = xy[2]
    lo_c = max(int(np.floor(xy[:, 0].min())), 0)
    hi_c = min(int(np.floor(xy[:, 0].max())), width - 1)
    lo_r = max(int(np.floor(xy[:, 1].min())), 0)
    hi_r = min(int(np.floor(xy[:, 1].max())), height - 1)
    if lo_c > hi_c or lo_r > hi_r:
        return
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(det) < DEGENERATE_AREA:
        return
    cols = np.arange(lo_c, hi_c + 1) + 0.5
    rows = np.arange(lo_r, hi_r + 1) + 0.5
    px, py = np.meshgrid(cols, rows)
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / det
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / det
    w2 = 1.0 - w0 - w1
    inside = (w0 >= BARY_TOLERANCE) & (w1 >= BARY_TOLERANCE) & (w2 >= BARY_TOLERANCE)
    if not inside.any():
        return
    inv = w0 * inv_z[0] + w1 * inv_z[1] + w2 * inv_z[2]
    z = np.where(inside, 1.0 / np.where(inside, inv, 1.0), np.inf).astype(np.float32)
    block = depth[lo_r : hi_r + 1, lo_c : hi_c + 1]
    np.minimum(block, z, out=block)


def depth_map(
    tris: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    width: int,
    height: int,
    *,
    near: float = 0.05,
) -> np.ndarray:
    """Camera depth (z) of the nearest triangle per pixel; ``inf`` where nothing is drawn.

    ``tris`` are (T, 3, 3) world-space triangles, ``R`` and ``t`` map world to camera,
    and ``K`` is the 3x3 intrinsics for a ``width`` x ``height`` image. Triangles are
    clipped to ``z >= near`` and drawn without back-face culling, so both sides occlude.
    ``near`` must be positive: a vertex on the camera plane would divide by zero.
    """
    if not near > 0:
        raise ValueError("near clipping distance must be positive")
    tris = np.asarray(tris, dtype=np.float64).reshape(-1, 3, 3)
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    if len(tris) == 0:
        return depth
    cam = tris @ R.T + t  # (T, 3, 3): camera coordinates of every vertex
    z_min = cam[:, :, 2].min(axis=1)
    z_max = cam[:, :, 2].max(axis=1)
    in_front = z_min >= near
    straddles = (z_min < near) & (z_max > near)
    clipped = [piece for k in np.flatnonzero(straddles) for piece in _clip_to_near(cam[k], near)]
    tri_cam = np.concatenate([cam[in_front], np.array(clipped, dtype=np.float64).reshape(-1, 3, 3)])
    if len(tri_cam) == 0:
        return depth
    screen = (tri_cam[:, :, :2] / tri_cam[:, :, 2:3]) @ K[:2, :2].T + K[:2, 2]  # (T, 3, 2)
    inv_z = 1.0 / tri_cam[:, :, 2]
    for k in range(len(tri_cam)):
        _rasterize(depth, screen[k], inv_z[k])
    return depth
