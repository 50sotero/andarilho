"""``scene_analysis`` (CPU): rooms, doors and furniture recognised by an AI model.

Placeholder: the stage reports that it was skipped, so geometry_proposal falls back to
the geometric proposal.
"""

from __future__ import annotations

from andarilho_worker.stages.base import StageContext, StageResult, finish


def run(ctx: StageContext) -> StageResult:
    return finish(ctx, {"status": "skipped"}, outputs={"analysisStatus": "skipped"})
