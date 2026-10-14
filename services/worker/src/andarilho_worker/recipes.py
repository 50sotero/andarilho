"""Pinned, server-defined pipeline recipes (README sections 8 and 12).

Users choose only a quality tier. Every tool invocation is an argument array
built from a fixed template whose placeholders are filled exclusively with
worker-generated scratch paths and recipe constants; user-supplied flags, paths,
URLs or configuration files are never accepted.

The digest is ``sha256`` of the canonical JSON (``sort_keys``, compact
separators) of :meth:`Recipe.as_dict`, so any change to stage order,
capabilities, parameters, tool versions or argv templates changes the digest.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

RUN_STAGES = (
    "validating",
    "preparing",
    "camera_reconstruction",
    "training",
    "normalizing",
    "geometry_proposal",
    "packaging",
    "scene_validation",
)
GPU_STAGES = frozenset({"camera_reconstruction", "training"})
LAYOUT_STAGES = ("geometry_proposal", "packaging", "scene_validation")

TOOL_VERSIONS = {
    # Must match infra/Dockerfile.worker-gpu; verify each tool's --help in that image.
    "colmap": "3.12.5",
    "nerfstudio": "1.1.5",
    "gsplat": "1.4.0",  # the exact version nerfstudio 1.1.5 requires
    "torch": "2.7.1+cu128",
    "splat-transform": "1.4.0",
    "ffmpeg": "debian-bookworm",
}

# Argument templates. ``{name}`` placeholders are substituted with worker paths
# or recipe constants by :func:`render_argv`; nothing else is interpolated.
ARGV: dict[str, tuple[str, ...]] = {
    "ffprobe": (
        "ffprobe", "-v", "error", "-hide_banner", "-print_format", "json",
        "-show_format", "-show_streams", "{input}",
    ),
    "ffmpeg_frames": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", "{input}", "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-vf", "fps={fps},scale=w='min({long_edge},iw)':h='min({long_edge},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-frames:v", "{max_frames}", "-q:v", "2", "{output_pattern}",
    ),
    # The same frames as ffmpeg_frames, as tiny grayscale raw images for sharpness scoring.
    "ffmpeg_score_frames": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", "{input}", "-map", "0:v:0", "-an", "-sn", "-dn", "-map_metadata", "-1",
        "-vf", "fps={fps},scale={size}:{size},format=gray",
        "-frames:v", "{max_frames}", "-f", "rawvideo", "{output}",
    ),
    "ffmpeg_photo": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", "{input}", "-map", "0:v:0", "-map_metadata", "-1",
        "-vf", "scale=w='min({long_edge},iw)':h='min({long_edge},ih)'"
        ":force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-frames:v", "1", "-q:v", "2", "{output}",
    ),
    "colmap_feature_extractor": (
        "colmap", "feature_extractor", "--database_path", "{database}",
        "--image_path", "{images}", "--ImageReader.camera_model", "OPENCV",
        "--ImageReader.single_camera_per_folder", "1",
        "--SiftExtraction.use_gpu", "1", "--SiftExtraction.max_image_size", "{long_edge}",
        "--SiftExtraction.max_num_features", "8192",
    ),
    "colmap_sequential_matcher": (
        "colmap", "sequential_matcher", "--database_path", "{database}",
        "--SequentialMatching.overlap", "{overlap}", "--SequentialMatching.quadratic_overlap", "1",
        "--SiftMatching.use_gpu", "1",
    ),
    # Photos are not a capture sequence: match them exhaustively (photo-only
    # captures) or against every image through an explicit pair list (mixed).
    "colmap_exhaustive_matcher": (
        "colmap", "exhaustive_matcher", "--database_path", "{database}",
        "--SiftMatching.use_gpu", "1",
    ),
    "colmap_matches_importer": (
        "colmap", "matches_importer", "--database_path", "{database}",
        "--match_list_path", "{pairs}", "--match_type", "pairs",
        "--SiftMatching.use_gpu", "1",
    ),
    "colmap_mapper": (
        "colmap", "mapper", "--database_path", "{database}", "--image_path", "{images}",
        "--output_path", "{sparse}", "--Mapper.multiple_models", "1",
    ),
    "colmap_image_undistorter": (
        "colmap", "image_undistorter", "--image_path", "{images}", "--input_path", "{model}",
        "--output_path", "{output}", "--output_type", "COLMAP", "--max_image_size", "{long_edge}",
    ),
    "ns_train": (
        "ns-train", "splatfacto", "--data", "{data}", "--output-dir", "{output_dir}",
        "--experiment-name", "andarilho", "--timestamp", "{timestamp}",
        "--max-num-iterations", "{iterations}", "--steps-per-save", "{save_steps}",
        "--machine.seed", "{seed}", "--vis", "tensorboard",
        "--viewer.quit-on-train-completion", "True",
        "colmap", "--colmap-path", "sparse/0", "--images-path", "images",
        "--downscale-factor", "1",
    ),
    "ns_export": (
        "ns-export", "gaussian-splat", "--load-config", "{config}", "--output-dir", "{output_dir}",
    ),
    "splat_transform": ("splat-transform", "{input}", "{output}"),
}  # fmt: skip

_PLACEHOLDER = re.compile(r"\{([a-z_]+)\}")


def render_argv(name: str, **values: object) -> list[str]:
    """Fill a pinned template. Unknown or missing placeholders are errors."""
    template = ARGV[name]
    out: list[str] = []
    for token in template:
        needed = _PLACEHOLDER.findall(token)
        for key in needed:
            if key not in values:
                raise KeyError(f"missing placeholder {key} for {name}")
        out.append(_PLACEHOLDER.sub(lambda m: str(values[m.group(1)]), token))
    unused = set(values) - {k for t in template for k in _PLACEHOLDER.findall(t)}
    if unused:
        raise KeyError(f"unexpected values for {name}: {sorted(unused)}")
    return out


@dataclass(frozen=True)
class StageSpec:
    stage: str
    capability: str
    timeout_seconds: int


@dataclass(frozen=True)
class Recipe:
    name: str
    quality: str
    stages: tuple[StageSpec, ...]
    params: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "quality": self.quality,
            "stages": [
                {"stage": s.stage, "capability": s.capability, "timeoutSeconds": s.timeout_seconds}
                for s in self.stages
            ],
            "params": dict(self.params),
            "tools": TOOL_VERSIONS,
            "argv": {k: list(v) for k, v in ARGV.items()},
        }

    @property
    def digest(self) -> str:
        blob = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return "sha256:" + hashlib.sha256(blob).hexdigest()

    def stage_names(self) -> list[str]:
        return [s.stage for s in self.stages]

    def spec(self, stage: str) -> StageSpec:
        for s in self.stages:
            if s.stage == stage:
                return s
        raise KeyError(stage)

    def next_stage(self, stage: str, *, layout_pipeline: bool = False) -> StageSpec | None:
        order = [s for s in self.stages if not layout_pipeline or s.stage in LAYOUT_STAGES]
        names = [s.stage for s in order]
        if stage not in names:
            raise KeyError(stage)
        index = names.index(stage)
        return order[index + 1] if index + 1 < len(order) else None

    def manifest_entry(self) -> dict[str, Any]:
        """The ``recipe`` object the API stores in the first job's input manifest."""
        return {
            "name": self.name,
            "digest": self.digest,
            "stages": [{"stage": s.stage, "capability": s.capability} for s in self.stages],
        }


TIMEOUTS = {
    "validating": 300,
    "preparing": 1800,
    "camera_reconstruction": 3 * 3600,
    "training": 3 * 3600,
    "normalizing": 1800,
    "geometry_proposal": 600,
    "packaging": 600,
    "scene_validation": 600,
}


def _specs(stages: Sequence[str]) -> tuple[StageSpec, ...]:
    return tuple(StageSpec(s, "gpu" if s in GPU_STAGES else "cpu", TIMEOUTS[s]) for s in stages)


COMMON = {
    "maxFrames": 400,
    "longEdge": 1600,
    "maxVideos": 8,
    "maxPhotos": 400,
    "minRegisteredViews": 30,
    "minRegisteredRatio": 0.8,
    # Measured on the first real walkthrough (handheld phone video, fast room
    # transitions): 2 fps registered 28 of 141 frames, 4 fps 122 of 283, and 4 fps
    # chosen as the sharpest of 12 fps candidates 155 of 283. The sequential
    # window spans the same ~5 s at 4 fps as 10 frames did at 2 fps.
    "fps": 4,
    "keyframeOversample": 3,
    "sequentialOverlap": 20,
    "seed": 42,
}

RECIPES: dict[str, Recipe] = {
    "preview": Recipe(
        "preview-v1",
        "preview",
        _specs(RUN_STAGES),
        {**COMMON, "iterations": 7000, "saveSteps": 2000},
    ),
    "standard": Recipe(
        "standard-v1",
        "standard",
        _specs(RUN_STAGES),
        {**COMMON, "iterations": 30000, "saveSteps": 5000},
    ),
    "synthetic": Recipe(
        "synthetic-v1",
        "synthetic",
        _specs([s for s in RUN_STAGES if s not in GPU_STAGES]),
        {"fixture": "synthetic-two-room"},
    ),
}


@cache
def recipe_for(quality: str) -> Recipe:
    try:
        return RECIPES[quality]
    except KeyError:
        raise KeyError(f"unknown quality tier {quality!r}") from None


def recipe_by_name(name: str) -> Recipe | None:
    return next((r for r in RECIPES.values() if r.name == name), None)
