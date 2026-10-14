"""Server-defined reconstruction recipes (README sections 8 and 12).

Clients choose only a quality tier; they can never send flags, URLs or paths.
Each recipe lists its stages with the worker capability that claims them, a
resource budget and fixed stage parameters. ``recipe_digest`` is the SHA-256 of
the recipe's canonical JSON, recorded on the run for provenance.

The ``synthetic`` recipe skips the two GPU stages (camera_reconstruction and
training) so a CPU-only worker can complete a fixture scene end to end. It is
available only when ``ALLOW_SYNTHETIC_RECIPE`` is enabled (never in production).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from andarilho_api.domain.canonical import sha256_hex
from andarilho_api.models import RUN_STAGES

GPU_STAGES = frozenset({"camera_reconstruction", "training"})
RECIPE_VERSION = 1


@dataclass(frozen=True)
class Recipe:
    name: str
    stages: tuple[str, ...]
    reserved_gpu_seconds: int
    reserved_cpu_seconds: int
    params: dict[str, Any]

    def stage_list(self) -> list[dict[str, str]]:
        return [
            {"stage": s, "capability": "gpu" if s in GPU_STAGES else "cpu"} for s in self.stages
        ]

    def budget(self, run_max_seconds: int) -> dict[str, int]:
        return {
            "maxSeconds": run_max_seconds,
            "reservedGpuSeconds": min(self.reserved_gpu_seconds, run_max_seconds),
            "reservedCpuSeconds": min(self.reserved_cpu_seconds, run_max_seconds),
        }

    def document(self, run_max_seconds: int) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": RECIPE_VERSION,
            "stages": self.stage_list(),
            "budget": self.budget(run_max_seconds),
            "params": self.params,
        }

    def digest(self, run_max_seconds: int) -> str:
        return sha256_hex(self.document(run_max_seconds))


_FULL = RUN_STAGES
_SYNTHETIC = tuple(s for s in RUN_STAGES if s not in GPU_STAGES)

RECIPES: dict[str, Recipe] = {
    "preview": Recipe(
        name="preview",
        stages=_FULL,
        reserved_gpu_seconds=1800,
        reserved_cpu_seconds=900,
        params={"maxFrames": 300, "frameLongEdge": 1600, "trainingIterations": 7000},
    ),
    "standard": Recipe(
        name="standard",
        stages=_FULL,
        reserved_gpu_seconds=3600,
        reserved_cpu_seconds=1200,
        params={"maxFrames": 600, "frameLongEdge": 2048, "trainingIterations": 30000},
    ),
    "synthetic": Recipe(
        name="synthetic",
        stages=_SYNTHETIC,
        reserved_gpu_seconds=0,
        reserved_cpu_seconds=300,
        params={"fixture": "synthetic-two-room"},
    ),
}


def get_recipe(quality: str) -> Recipe:
    return RECIPES[quality]
