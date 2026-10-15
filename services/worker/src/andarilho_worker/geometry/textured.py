"""The textured visual model: layout surfaces with UVs into shared atlas pages.

``texture.py`` produces a :class:`TexturedMesh` from a layout (and, for a real
capture, the registered frames); ``glb.write_textured_glb`` serializes it. All
positions are canonical world coordinates: metres, +Y up, identity nodes.

UVs follow glTF: ``(0, 0)`` is the top-left corner of the atlas image, ``u``
grows to the right and ``v`` grows downwards, both normalized to ``[0, 1]``.
Triangles wind counter-clockwise seen from the side the normal points to (the
visible side).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# Surface kinds: the collision kinds, the visual-only ceiling, and furniture parts (drawn
# whether or not the piece is solid).
SURFACE_KINDS = (
    "wall",
    "closed_opening",
    "sill",
    "lintel",
    "floor",
    "ceiling",
    "obstacle",
    "furniture",
)


@dataclass
class TexturedPart:
    name: str
    kind: str  # one of SURFACE_KINDS
    positions: np.ndarray  # (N, 3) float, world metres
    normals: np.ndarray  # (N, 3) float, unit length
    uvs: np.ndarray  # (N, 2) float in [0, 1], glTF convention
    indices: np.ndarray  # (M, 3) int, counter-clockwise from the normal side
    page: int  # index into TexturedMesh.pages
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class TexturedMesh:
    parts: list[TexturedPart]
    pages: list[np.ndarray]  # (H, W, 3) uint8 RGB atlas images
    # Fraction of chart texels that some frame saw, per surface kind and "all";
    # procedural (synthetic) textures report 1.0.
    coverage: dict[str, float] = field(default_factory=dict)
    texel_size: float = 0.01  # metres per texel actually used
