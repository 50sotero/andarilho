"""Depth rasterizer (geometry.raster) against analytic ray intersections."""

from __future__ import annotations

import numpy as np
import pytest

from andarilho_worker.geometry.raster import depth_map

W, H = 100, 80
FX = FY = 100.0
CX, CY = 50.0, 40.0
K = np.array([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])
IDENTITY = np.eye(3)
ZERO = np.zeros(3)


def _pixel_rays() -> np.ndarray:
    """Camera-frame ray directions (z = 1) through every pixel centre, shaped (H, W, 3)."""
    cols, rows = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    return np.stack([(cols - CX) / FX, (rows - CY) / FY, np.ones_like(cols)], axis=-1)


def _moller_trumbore(
    dirs: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray, eps: float = 0.0
) -> np.ndarray:
    """Ray parameter t from the origin along each direction (equal to depth z here); inf on a miss."""
    e1, e2 = b - a, c - a
    p = np.cross(dirs, e2)
    det = p @ e1
    s = -a  # origin minus a, the origin being zero
    q = np.cross(s, e1)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        u = (p @ s) * inv
        v = (dirs @ q) * inv
        t = (e2 @ q) * inv
    hit = (np.abs(det) > 1e-12) & (u >= -eps) & (v >= -eps) & (u + v <= 1 + eps) & (t > 0)
    return np.where(hit, t, np.inf)


def _quad(z: float, half: float) -> np.ndarray:
    p = np.array(
        [
            [-half, -half, z],
            [half, -half, z],
            [half, half, z],
            [-half, half, z],
        ]
    )
    return np.array([[p[0], p[1], p[2]], [p[0], p[2], p[3]]])


def test_facing_quad_at_depth_two():
    tris = _quad(2.0, 0.5)
    depth = depth_map(tris, IDENTITY, ZERO, K, W, H)
    # Pixel centres inside the projected quad: u in [25, 75], v in [15, 65].
    rows, cols = np.mgrid[0:H, 0:W]
    expected = (cols >= 25) & (cols < 75) & (rows >= 15) & (rows < 65)
    assert np.array_equal(np.isfinite(depth), expected)
    np.testing.assert_allclose(depth[expected], 2.0, rtol=1e-6)
    assert depth.dtype == np.float32


def test_nearer_quad_occludes_farther_in_either_order():
    far = _quad(4.0, 1.0)
    near = _quad(2.0, 0.25)
    for tris in (np.concatenate([far, near]), np.concatenate([near, far])):
        depth = depth_map(tris, IDENTITY, ZERO, K, W, H)
        rows, cols = np.mgrid[0:H, 0:W]
        in_near = (cols >= 37) & (cols < 63) & (rows >= 27) & (rows < 53)
        in_far = np.isfinite(depth) & ~in_near
        np.testing.assert_allclose(depth[in_near], 2.0, rtol=1e-6)
        np.testing.assert_allclose(depth[in_far], 4.0, rtol=1e-6)
        assert np.isinf(depth[0, 0]) and np.isinf(depth[79, 99])


def test_triangle_crossing_the_near_plane_is_clipped_not_blown_up():
    # Two vertices are behind the camera plane; the front part must still be drawn.
    tri = np.array([[[-1.0, -1.0, -0.5], [1.0, -1.0, 2.0], [0.0, 1.0, 2.0]]])
    depth = depth_map(tri, IDENTITY, ZERO, K, W, H, near=0.05)
    finite = depth[np.isfinite(depth)]
    assert finite.size > 0
    assert np.all(finite >= 0.05 - 1e-6)
    assert not np.isnan(depth).any()
    # Where the front part of the triangle lies, every pixel is covered and exact.
    dirs = _pixel_rays()
    t = _moller_trumbore(dirs, *tri[0])
    front = np.isfinite(t) & (t > 0.05 + 1e-3)
    assert front.sum() > 50
    assert np.all(np.isfinite(depth[front]))
    np.testing.assert_allclose(depth[front], t[front], rtol=1e-5)


def _patch() -> np.ndarray:
    """Two triangles of the plane z = 3 + 0.4 x - 0.3 y over a 3 x 3 m square patch."""

    def height(x: float, y: float) -> float:
        return 3 + 0.4 * x - 0.3 * y

    corners = np.array(
        [[x, y, height(x, y)] for x, y in ((-1.5, -1.5), (1.5, -1.5), (1.5, 1.5), (-1.5, 1.5))]
    )
    return np.array([[corners[0], corners[1], corners[2]], [corners[0], corners[2], corners[3]]])


def test_tilted_plane_matches_analytic_depth():
    # The plane is n . X = 3 with n = (-0.4, 0.3, 1).
    tris = _patch()
    depth = depth_map(tris, IDENTITY, ZERO, K, W, H)
    dirs = _pixel_rays()
    # Pixels whose ray meets the patch (a tiny tolerance keeps pixels on the shared diagonal).
    patch = np.isfinite(_moller_trumbore(dirs, *tris[0], eps=1e-6)) | np.isfinite(
        _moller_trumbore(dirs, *tris[1], eps=1e-6)
    )
    assert patch.sum() > 500
    finite = np.isfinite(depth)
    assert np.all(finite[patch])  # no holes where the patch is in view
    assert np.all(patch[finite])  # nothing drawn outside it
    normal = np.array([-0.4, 0.3, 1.0])
    analytic = 3.0 / (dirs @ normal)
    rel = np.abs(depth[finite] - analytic[finite]) / analytic[finite]
    assert rel.max() < 1e-3


def _rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def test_tilted_plane_seen_through_a_rotated_camera_matches_analytic_depth():
    # World-to-camera rotation about two axes; the camera centre is off the origin.
    R = _rotation_x(-0.2) @ _rotation_y(0.3)
    eye = np.array([0.3, -0.2, 0.5])
    t = -R @ eye
    tris = _patch()
    depth = depth_map(tris, R, t, K, W, H)
    # A pixel's ray leaves the eye along R.T @ d (row-vector form: d @ R). Its camera depth is
    # the ray parameter, because the camera-frame direction d has unit z.
    dirs_world = _pixel_rays() @ R
    normal = np.array([-0.4, 0.3, 1.0])
    t_hit = (3.0 - normal @ eye) / (dirs_world @ normal)
    hit = eye + t_hit[..., None] * dirs_world
    on_patch = (
        (t_hit > 0.05) & (np.abs(hit[..., 0]) < 1.5 - 1e-6) & (np.abs(hit[..., 1]) < 1.5 - 1e-6)
    )
    assert on_patch.sum() > 500
    finite = np.isfinite(depth)
    assert np.array_equal(finite, on_patch)
    rel = np.abs(depth[on_patch] - t_hit[on_patch]) / t_hit[on_patch]
    assert rel.max() < 1e-3


def test_near_plane_must_be_positive():
    with pytest.raises(ValueError, match="near"):
        depth_map(_quad(2.0, 0.5), IDENTITY, ZERO, K, W, H, near=0.0)


def test_camera_motion_is_applied_through_R_and_t():
    # Translating the world by +1 m in z moves the quad 1 m farther from the camera.
    tris = _quad(2.0, 0.5)
    shifted = depth_map(tris, IDENTITY, np.array([0.0, 0.0, 1.0]), K, W, H)
    np.testing.assert_allclose(shifted[np.isfinite(shifted)], 3.0, rtol=1e-6)


def test_empty_input_gives_empty_depth():
    depth = depth_map(np.zeros((0, 3, 3)), IDENTITY, ZERO, K, W, H)
    assert depth.shape == (H, W)
    assert np.all(np.isinf(depth))
