"""Which revision a project is published on, derived from its live share links.

``projects.published_revision_id`` is updated when links are created or revoked,
but links also expire on their own. Reads therefore derive publication from the
links that are still live: the stored pointer while it has one, otherwise the
newest live link's revision, otherwise nothing.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable

from sqlalchemy import or_, select
from sqlalchemy.orm import Session

from andarilho_api.deps import utcnow
from andarilho_api.models import Project, ShareLink


def live_published(db: Session, projects: Iterable[Project]) -> dict[uuid.UUID, uuid.UUID | None]:
    projects = list(projects)
    result: dict[uuid.UUID, uuid.UUID | None] = {p.id: None for p in projects}
    candidates = [p.id for p in projects if p.published_revision_id is not None]
    if not candidates:
        return result
    rows = db.execute(
        select(ShareLink.project_id, ShareLink.scene_revision_id)
        .where(
            ShareLink.project_id.in_(candidates),
            ShareLink.revoked_at.is_(None),
            or_(ShareLink.expires_at.is_(None), ShareLink.expires_at > utcnow()),
        )
        .order_by(ShareLink.created_at.desc(), ShareLink.id)
    ).all()
    live: dict[uuid.UUID, list[uuid.UUID]] = {}
    for project_id, revision_id in rows:
        live.setdefault(project_id, []).append(revision_id)
    for p in projects:
        revisions = live.get(p.id, [])
        if p.published_revision_id in revisions:
            result[p.id] = p.published_revision_id
        elif revisions:
            result[p.id] = revisions[0]
    return result


def published_revision_id(db: Session, project: Project) -> uuid.UUID | None:
    return live_published(db, [project])[project.id]
