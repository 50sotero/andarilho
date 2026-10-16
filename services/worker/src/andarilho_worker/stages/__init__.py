"""Stage registry: canonical stage name -> stage function."""

from __future__ import annotations

from andarilho_worker.stages import (
    analyze_scene,
    cleanup,
    cleanup_artifacts,
    cleanup_upload,
    inspect_upload,
    normalize,
    package_scene,
    prepare,
    propose_geometry,
    reconstruct,
    texture,
    validate,
    validate_scene,
)
from andarilho_worker.stages.base import Stage

STAGES: dict[str, Stage] = {
    "validating": validate.run,
    "preparing": prepare.run,
    "camera_reconstruction": reconstruct.run,
    "normalizing": normalize.run,
    "scene_analysis": analyze_scene.run,
    "geometry_proposal": propose_geometry.run,
    "texturing": texture.run,
    "packaging": package_scene.run,
    "scene_validation": validate_scene.run,
    "upload_inspection": inspect_upload.run,
    "project_cleanup": cleanup.run,
    "upload_cleanup": cleanup_upload.run,
    "artifact_cleanup": cleanup_artifacts.run,
}
