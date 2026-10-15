"""Server-defined reconstruction recipes (README sections 8 and 12).

Clients choose only a quality tier; they can never send flags, URLs or paths.
The pipeline definition (stage list, capabilities, parameters and digest) is
pinned once, in the worker's ``recipes.py``; ``scripts/generate-contracts.py``
copies it into ``recipes.json`` beside this module, so the digest recorded on a
run is exactly the recipe the worker executes. The worker rejects a run whose
digest it does not recognise. Budget reservations are API cost policy.

The ``synthetic`` recipe skips the two GPU stages (camera_reconstruction and
training) so a CPU-only worker can complete a fixture scene end to end. It is
available only when ``ALLOW_SYNTHETIC_RECIPE`` is enabled (never in production).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_PINNED: dict[str, Any] = json.loads(
    (Path(__file__).with_name("recipes.json")).read_text(encoding="utf-8")
)

# Capture intake rules the worker enforces (published to the web app).
INTAKE: dict[str, Any] = dict(_PINNED["intake"])

# Reserved seconds per quality tier (cost policy, not part of the pipeline digest).
_RESERVATIONS = {
    "preview": (1800, 900),
    "standard": (3600, 1200),
    "synthetic": (0, 300),
}


@dataclass(frozen=True)
class Recipe:
    quality: str
    name: str
    digest: str
    stages: tuple[str, ...]
    stage_list: tuple[dict[str, str], ...]
    params: dict[str, Any]
    reserved_gpu_seconds: int
    reserved_cpu_seconds: int

    def budget(self, run_max_seconds: int) -> dict[str, int]:
        return {
            "maxSeconds": run_max_seconds,
            "reservedGpuSeconds": min(self.reserved_gpu_seconds, run_max_seconds),
            "reservedCpuSeconds": min(self.reserved_cpu_seconds, run_max_seconds),
        }


def _load(quality: str) -> Recipe:
    entry = _PINNED[quality]
    gpu, cpu = _RESERVATIONS[quality]
    return Recipe(
        quality=quality,
        name=entry["name"],
        digest=entry["digest"],
        stages=tuple(s["stage"] for s in entry["stages"]),
        stage_list=tuple(dict(s) for s in entry["stages"]),
        params=dict(entry["params"]),
        reserved_gpu_seconds=gpu,
        reserved_cpu_seconds=cpu,
    )


RECIPES: dict[str, Recipe] = {quality: _load(quality) for quality in _RESERVATIONS}


def get_recipe(quality: str) -> Recipe:
    return RECIPES[quality]
