"""Sharp keyframe selection for video frames (README section 8, stage A).

FFmpeg decodes candidate frames at a higher rate than the recipe keeps, and a
second, tiny grayscale decode of the same frames is scored for sharpness (the
variance of a discrete Laplacian). The candidates are split into equal
consecutive windows and the sharpest frame of each window is kept, so the
selection stays evenly spread in time (doorway transitions are not dropped)
while motion-blurred frames from fast turns lose to their sharper neighbours.
Sharpness values are only compared within one video; they are not an absolute
blur verdict, which depends on resolution and content.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SCORE_SIZE = 256  # candidates are scored as SCORE_SIZE x SCORE_SIZE grayscale frames
_BATCH = 64


def laplacian_variance(frames: np.ndarray) -> np.ndarray:
    """Per-frame variance of the 4-neighbour Laplacian of ``(n, h, w)`` grayscale frames."""
    f = frames.astype(np.float32)
    lap = f[:, 1:-1, 2:] + f[:, 1:-1, :-2] + f[:, 2:, 1:-1] + f[:, :-2, 1:-1] - 4 * f[:, 1:-1, 1:-1]
    return lap.reshape(len(f), -1).var(axis=1)


def sharpness_from_raw(path: Path, size: int = SCORE_SIZE) -> np.ndarray:
    """Score a raw ``gray`` video file of ``size`` x ``size`` frames, in bounded batches."""
    frame_bytes = size * size
    count = path.stat().st_size // frame_bytes
    raw = np.memmap(path, dtype=np.uint8, mode="r", shape=(count, size, size)) if count else None
    scores = np.empty(count, dtype=np.float64)
    for start in range(0, count, _BATCH):
        scores[start : start + _BATCH] = laplacian_variance(raw[start : start + _BATCH])  # type: ignore[index]
    del raw  # release the map before the caller removes the file
    return scores


def pick_sharpest(scores: np.ndarray, keep: int) -> list[int]:
    """Indices of the sharpest frame in each of ``keep`` equal consecutive windows."""
    n = len(scores)
    if keep >= n:
        return list(range(n))
    if keep <= 0:
        return []
    edges = np.floor(np.arange(keep + 1) * (n / keep)).astype(int)
    windows = zip(edges[:-1], edges[1:], strict=True)
    return [int(lo + np.argmax(scores[lo:hi])) for lo, hi in windows]
