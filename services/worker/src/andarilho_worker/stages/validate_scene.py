"""``scene_validation``: validate the packaged revision and commit it as the draft.

Checks: JSON Schema, finite numbers, similarity matrix, asset prefixes, spawn
floor/bounds, asset bytes/sha256, render header, collision GLB (identity nodes,
finite vertices within bounds), spawn capsule clearance (radius 0.25 m, height
1.7 m) and floor support against the approved layout.

On success the fenced completion transaction marks the revision ``valid`` with
its manifest and ``manifest_sha256`` (sha256 of canonical JSON), inserts the
artifact rows, advances ``scenes.current_draft_revision_id`` and
``projects.active_draft_revision_id`` (never moving them back to an older
revision) and, for a reconstruction run, marks the run succeeded. On failure the
revision becomes ``invalid`` with findings.
"""

from __future__ import annotations

from typing import Any

from psycopg.types.json import Jsonb

from andarilho_worker.artifacts import insert_artifact
from andarilho_worker.errors import StageError
from andarilho_worker.stages.base import StageContext, StageResult, finish
from andarilho_worker.validation import loads_strict, manifest_sha256, validate_scene


def _load(conn: Any, ctx: StageContext, revision_id: str) -> dict[str, Any] | None:
    return conn.execute(
        "SELECT id, scene_id, revision, layout, validation_state FROM scene_revisions "
        "WHERE id = %s AND project_id = %s",
        (revision_id, ctx.project_id),
    ).fetchone()


def _invalidate(revision_id: str, findings: list[dict[str, str]]):
    def hook(conn: Any, job: Any) -> None:
        conn.execute(
            "UPDATE scene_revisions SET validation_state = 'invalid', validation_findings = %s "
            "WHERE id = %s AND validation_state = 'pending'",
            (Jsonb(findings), revision_id),
        )

    return hook


def run(ctx: StageContext) -> StageResult:
    revision_id = str(ctx.output("sceneRevisionId"))
    revision = ctx.query(lambda conn: _load(conn, ctx, revision_id))
    if revision is None:
        raise StageError("MISSING_INPUT")
    try:
        manifest = loads_strict(ctx.store.get_bytes(str(ctx.output("manifestKey"))))
    except (ValueError, FileNotFoundError) as exc:
        findings = [{"code": "MANIFEST", "message": "manifest missing or not valid JSON"}]
        raise StageError(
            "SCENE_VALIDATION_FAILED", on_fail=_invalidate(revision_id, findings)
        ) from exc
    ctx.progress("Validating the scene.")
    findings = [
        f.as_dict()
        for f in validate_scene(
            manifest, ctx.store.get_bytes, project_id=ctx.project_id, layout=revision["layout"]
        )
    ]
    if not findings and manifest.get("sceneRevisionId") != revision_id:
        findings.append({"code": "REVISION", "message": "manifest names another revision"})
    if findings:
        raise StageError(
            "SCENE_VALIDATION_FAILED",
            details={"findings": findings[:20]},
            on_fail=_invalidate(revision_id, findings),
        )
    digest = manifest_sha256(manifest)
    files = dict(ctx.output("packagedFiles"))
    revision_number = int(revision["revision"])
    scene_id = str(revision["scene_id"])

    def commit(conn: Any, job: Any) -> None:
        ids = {}
        for kind in ("render", "collision", "manifest"):
            entry = files[kind]
            ids[kind] = insert_artifact(
                conn,
                project_id=ctx.project_id,
                run_id=ctx.run_id,
                job_id=ctx.job_id,
                kind=f"scene_{kind}",
                key=entry["key"],
                version=entry.get("version"),
                sha256=entry["sha256"],
                size=entry["bytes"],
                provenance={
                    "sceneRevisionId": revision_id,
                    "recipeDigest": ctx.recipe_digest,
                    "format": entry["format"],
                },
                validation_state="valid",
            )
        updated = conn.execute(
            """
            UPDATE scene_revisions
            SET validation_state = 'valid', manifest = %s, manifest_sha256 = %s,
                manifest_artifact_id = %s, validation_findings = '[]'::jsonb
            WHERE id = %s AND project_id = %s AND validation_state = 'pending'
            RETURNING id
            """,
            (Jsonb(manifest), digest, ids["manifest"], revision_id, ctx.project_id),
        ).fetchone()
        if updated is None:
            raise StageError("SCENE_VALIDATION_FAILED", "The revision is no longer pending.")
        conn.execute(
            """
            UPDATE scenes AS s SET current_draft_revision_id = %(rev)s, updated_at = now()
            WHERE s.id = %(scene)s AND s.project_id = %(project)s
              AND (s.current_draft_revision_id IS NULL
                   OR (SELECT r.revision FROM scene_revisions r
                       WHERE r.id = s.current_draft_revision_id) <= %(num)s)
            """,
            {"rev": revision_id, "scene": scene_id, "project": ctx.project_id,
             "num": revision_number},
        )  # fmt: skip
        conn.execute(
            """
            UPDATE projects AS p SET active_draft_revision_id = %(rev)s, updated_at = now()
            WHERE p.id = %(project)s AND p.lifecycle = 'active'
              AND (p.active_draft_revision_id IS NULL
                   OR (SELECT r.revision FROM scene_revisions r
                       WHERE r.id = p.active_draft_revision_id AND r.scene_id = %(scene)s)
                      <= %(num)s
                   -- The pointer is on another scene: take it over only when that
                   -- scene is older, so a late older run never hides a newer scene.
                   OR EXISTS (SELECT 1 FROM scene_revisions r
                              JOIN scenes cur ON cur.id = r.scene_id
                              JOIN scenes mine ON mine.id = %(scene)s
                              WHERE r.id = p.active_draft_revision_id
                                AND r.scene_id <> %(scene)s
                                AND (cur.created_at, cur.id) < (mine.created_at, mine.id)))
            """,
            {"rev": revision_id, "scene": scene_id, "project": ctx.project_id,
             "num": revision_number},
        )  # fmt: skip

    return finish(
        ctx,
        {"valid": True, "manifestSha256": digest},
        outputs={"sceneRevisionId": revision_id, "manifestSha256": digest},
        commit=commit,
        finish_run=True,
    )
