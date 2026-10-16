"""Recipe cost policy and stage contract: GPU time is COLMAP's, texturing runs on CPU."""

from __future__ import annotations

import pytest

from andarilho_api.domain.recipes import RECIPES
from andarilho_api.models import RUN_STAGES

# (GPU, CPU) reserved seconds per tier. GPU covers camera reconstruction only and fell to
# about a third of the splat-era budget; CPU grew by the 1800 s texturing allowance.
RESERVED = {
    "preview": (600, 2700),
    "standard": (1200, 3000),
    "synthetic": (0, 2100),
}
TIERS = sorted(RESERVED)


@pytest.mark.parametrize("quality", TIERS)
def test_reservations_split_gpu_for_colmap_from_cpu_for_texturing(quality):
    recipe = RECIPES[quality]
    assert (recipe.reserved_gpu_seconds, recipe.reserved_cpu_seconds) == RESERVED[quality]


def test_the_budget_caps_reservations_at_the_run_limit():
    assert RECIPES["standard"].budget(1000) == {
        "maxSeconds": 1000,
        "reservedGpuSeconds": 1000,
        "reservedCpuSeconds": 1000,
    }


@pytest.mark.parametrize("quality", TIERS)
def test_recipe_stages_are_run_stages_in_pipeline_order(quality):
    names = [s["stage"] for s in RECIPES[quality].stage_list]
    assert set(names) <= set(RUN_STAGES)
    assert names == [s for s in RUN_STAGES if s in names]


@pytest.mark.parametrize("quality", TIERS)
def test_recipes_texture_the_layout_and_use_gpu_only_for_colmap(quality):
    stages = {s["stage"]: s["capability"] for s in RECIPES[quality].stage_list}
    assert "training" not in stages
    assert stages["texturing"] == "cpu"
    gpu = {name for name, capability in stages.items() if capability == "gpu"}
    assert gpu == (set() if quality == "synthetic" else {"camera_reconstruction"})
