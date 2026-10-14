from __future__ import annotations

import numpy as np

from andarilho_worker.keyframes import laplacian_variance, pick_sharpest, sharpness_from_raw


def _frames() -> np.ndarray:
    rng = np.random.default_rng(7)
    sharp = rng.integers(0, 256, size=(32, 32)).astype(np.uint8)  # high-frequency texture
    blurred = np.full((32, 32), 128, dtype=np.uint8)  # featureless, like heavy motion blur
    return np.stack([blurred, sharp, blurred])


def test_laplacian_variance_ranks_texture_above_blur():
    scores = laplacian_variance(_frames())
    assert scores[0] == 0 and scores[2] == 0 and scores[1] > 1000


def test_sharpness_from_raw_reads_frames_in_batches(tmp_path):
    frames = np.concatenate([_frames()] * 50)  # 150 frames: more than one batch
    path = tmp_path / "score.gray"
    frames.tofile(path)
    scores = sharpness_from_raw(path, size=32)
    assert np.array_equal(scores, laplacian_variance(frames).astype(np.float64))
    path.unlink()  # the memory map is released
    empty = tmp_path / "empty.gray"
    empty.write_bytes(b"")
    assert len(sharpness_from_raw(empty, size=32)) == 0


def test_pick_sharpest_keeps_one_frame_per_window_in_order():
    scores = np.array([1, 5, 2, 9, 1, 1, 3, 2, 8, 0], dtype=float)
    assert pick_sharpest(scores, 5) == [1, 3, 4, 6, 8]  # windows of two; ties keep the first
    assert pick_sharpest(scores, 2) == [3, 8]
    assert pick_sharpest(scores, 10) == list(range(10))
    assert pick_sharpest(scores, 20) == list(range(10))
    assert pick_sharpest(scores, 0) == []
    picked = pick_sharpest(np.arange(1000.0), 400)
    assert len(picked) == 400 and picked == sorted(set(picked))
