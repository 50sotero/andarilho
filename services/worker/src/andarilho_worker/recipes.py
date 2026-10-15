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
    "normalizing",
    "scene_analysis",
    "geometry_proposal",
    "texturing",
    "packaging",
    "scene_validation",
)
# Stages retired from the pipeline whose historical job rows (and attempt prefixes) may
# still exist after an upgrade: the Gaussian-splat trainer. Matches the API's
# LEGACY_JOB_STAGES. Nothing new runs in them, but their outputs can still be cleaned up.
LEGACY_RUN_STAGES = ("training",)
# Only COLMAP needs a GPU; texturing and everything after it run on the CPU.
GPU_STAGES = frozenset({"camera_reconstruction"})
# A layout save re-runs these stages on the revision's own layout, reusing the run's frames
# and poses (see stages/texture.py).
LAYOUT_STAGES = ("geometry_proposal", "texturing", "packaging", "scene_validation")

TOOL_VERSIONS = {
    # Must match infra/Dockerfile.worker-gpu; verify each tool's --help in that image.
    "colmap": "3.12.5",
    "ffmpeg": "debian-bookworm",
}

# Argument templates. ``{name}`` placeholders are substituted with worker paths
# or recipe constants by :func:`render_argv`; nothing else is interpolated.
ARGV: dict[str, tuple[str, ...]] = {
    "ffprobe": (
        "ffprobe", "-v", "error", "-hide_banner", "-print_format", "json",
        "-show_format", "-show_streams", "{input}",
    ),
    # Upload inspection: decode an image's pixels once. A truncated or corrupt
    # file makes the decoder fail, and -xerror turns that into a non-zero exit.
    "ffmpeg_decode_image": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
        "-err_detect", "explode", "-i", "{input}", "-map", "0:v:0", "-frames:v", "1",
        "-f", "null", "-",
    ),
    # Upload inspection: decode every frame of a video once. A corrupt stream makes
    # the decoder fail (-xerror); a file cut at a packet boundary decodes cleanly
    # but stops early, which the final "frame=" count in {progress} shows.
    "ffmpeg_decode_video": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-xerror",
        "-i", "{input}", "-map", "0:v:0", "-an", "-sn", "-dn", "-f", "null",
        "-progress", "{progress}", "-stats_period", "5", "-nostats", "-",
    ),
    # Upload inspection: walk a floor-plan PDF's page tree (see pdf_inspect.py).
    # {python} is the worker's own interpreter.
    "pdf_inspect": ("{python}", "-I", "-m", "andarilho_worker.pdf_inspect", "{input}"),
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
    # The scene's poster: one prepared (already metadata-free) frame as WebP.
    "ffmpeg_poster": (
        "ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
        "-i", "{input}", "-map_metadata", "-1",
        "-vf", "scale=w='min({width},iw)':h=-2",
        "-frames:v", "1", "-c:v", "libwebp", "-quality", "80", "{output}",
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
    "normalizing": 1800,
    "scene_analysis": 600,
    "geometry_proposal": 600,
    "texturing": 1800,
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
    # The textured model (texturing stage): the target texel size in metres, coarsened
    # automatically when the atlas would need more than maxPages pages of pageSize texels,
    # and the JPEG quality of the atlas pages. Like every parameter, it is in the digest.
    "texture": {"texelSize": 0.01, "pageSize": 2048, "maxPages": 8, "jpegQuality": 85},
    # The scene analysis (scene_analysis stage): how many posed keyframes and registered
    # photos are shown to the AI model, and how they are downscaled before they leave the
    # worker. The floor plan is sent at its own size limit.
    "analysis": {
        "keyframes": 12,
        "maxPhotos": 6,
        "imageLongEdge": 1024,
        "floorPlanLongEdge": 1600,
        "jpegQuality": 80,
    },
}

# Capture intake rules shared with the API, which publishes them to the web app
# (scripts/generate-contracts.py copies them into the API's recipes.json). Not part
# of any recipe digest.
INTAKE = {
    # Photos alone rarely connect rooms; fewer than this never register reliably.
    "minPhotosWithoutVideo": 30,
}

RECIPES: dict[str, Recipe] = {
    "preview": Recipe(
        "preview-v3",
        "preview",
        _specs(RUN_STAGES),
        {**COMMON},
    ),
    "standard": Recipe(
        "standard-v3",
        "standard",
        _specs(RUN_STAGES),
        {**COMMON},
    ),
    "synthetic": Recipe(
        "synthetic-v3",
        "synthetic",
        _specs([s for s in RUN_STAGES if s not in GPU_STAGES]),
        {"fixture": "synthetic-two-room", "texture": COMMON["texture"]},
    ),
}

# Earlier versions that runs queued before a deployment may still be pinned to. Each lists
# only stages this worker still runs, so an in-flight run finishes on the recipe it was
# submitted with instead of failing on the upgrade. Their digests are computed exactly as
# they were published (tests pin them); a version whose stages are gone (the
# Gaussian-splat v1) is not kept, and its runs fail with RECIPE_MISMATCH.
_V2_STAGES = tuple(s for s in RUN_STAGES if s != "scene_analysis")
_V2_COMMON = {k: v for k, v in COMMON.items() if k != "analysis"}
RETAINED_RECIPES: tuple[Recipe, ...] = (
    Recipe("preview-v2", "preview", _specs(_V2_STAGES), {**_V2_COMMON}),
    Recipe("standard-v2", "standard", _specs(_V2_STAGES), {**_V2_COMMON}),
    Recipe(
        "synthetic-v2",
        "synthetic",
        _specs([s for s in _V2_STAGES if s not in GPU_STAGES]),
        {"fixture": "synthetic-two-room", "texture": COMMON["texture"]},
    ),
)


@cache
def recipe_for(quality: str) -> Recipe:
    try:
        return RECIPES[quality]
    except KeyError:
        raise KeyError(f"unknown quality tier {quality!r}") from None


def recipe_for_run(quality: str, digest: str) -> Recipe | None:
    """The recipe a run was submitted with: the current one or a retained earlier version."""
    for recipe in (RECIPES.get(quality), *RETAINED_RECIPES):
        if recipe is not None and recipe.quality == quality and recipe.digest == digest:
            return recipe
    return None


def recipe_by_name(name: str) -> Recipe | None:
    return next((r for r in RECIPES.values() if r.name == name), None)
