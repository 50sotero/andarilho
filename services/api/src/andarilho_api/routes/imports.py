"""Existing tours (README section 15): resolve a provider link or listing, attach an embed."""

from __future__ import annotations

import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from andarilho_api import serializers as ser
from andarilho_api.auth import current_subject
from andarilho_api.db import get_db
from andarilho_api.domain import audit
from andarilho_api.domain.imports import (
    SourceUnavailable,
    UnsupportedSource,
    canonical_matterport_url,
    check_listing_url,
    resolve_source,
)
from andarilho_api.domain.membership import WRITERS, load_project, load_scoped
from andarilho_api.errors import ApiError, invalid_state
from andarilho_api.models import TourImport
from andarilho_api.schemas.api import ImportCreate, ImportResolve

router = APIRouter(tags=["imports"])

Subject = Annotated[str, Depends(current_subject)]
Db = Annotated[Session, Depends(get_db)]


def _unsupported() -> ApiError:
    return ApiError(
        422,
        "IMPORT_UNSUPPORTED",
        "Only Matterport tour links and HomeQ listing pages can be imported.",
        suggested_action="Paste a link like https://my.matterport.com/show/?m=... "
        "or a https://www.homeq.se listing.",
    )


@router.post("/projects/{project_id}/imports/resolve")
def resolve_import(
    project_id: uuid.UUID, body: ImportResolve, subject: Subject, db: Db, request: Request
) -> dict[str, Any]:
    """Which tours a link offers and how each can be shown; creates nothing."""
    load_project(db, project_id, subject, roles=WRITERS, require_active=True)
    db.commit()  # release the connection while a listing page is fetched
    try:
        resolution = resolve_source(body.url, fetch=request.app.state.listing_fetcher)
    except UnsupportedSource:
        raise _unsupported() from None
    except SourceUnavailable:
        raise ApiError(
            502,
            "IMPORT_SOURCE_UNAVAILABLE",
            "The listing page could not be read.",
            retryable=True,
            suggested_action="Check the link, or paste the Matterport tour link itself.",
        ) from None
    return resolution.as_dict()


@router.post("/projects/{project_id}/imports", status_code=201)
def create_import(
    project_id: uuid.UUID, body: ImportCreate, subject: Subject, db: Db
) -> dict[str, Any]:
    """Attach a provider tour as an embed. The iframe is built from the model id only."""
    project, _ = load_project(
        db, project_id, subject, roles=WRITERS, lock=True, require_active=True
    )
    try:
        listing = check_listing_url(body.listing_url) if body.listing_url else None
    except UnsupportedSource:
        raise _unsupported() from None
    row = TourImport(
        project_id=project.id,
        created_by=subject,
        kind=body.kind,
        provider="matterport",
        model_id=body.model_id,
        canonical_url=canonical_matterport_url(body.model_id),
        listing_url=listing,
        provenance={"acquisition": "public_embed", "permission": "user_confirmed"},
    )
    db.add(row)
    try:
        db.flush()
    except IntegrityError:
        db.rollback()
        raise invalid_state("This tour is already part of the project.") from None
    audit.record(
        db,
        project_id=project.id,
        actor=subject,
        action="import.created",
        resource_type="import",
        resource_id=row.id,
        data={"kind": row.kind, "provider": row.provider, "modelId": row.model_id},
    )
    db.commit()
    return ser.tour_import(row)


@router.get("/projects/{project_id}/imports")
def list_imports(project_id: uuid.UUID, subject: Subject, db: Db) -> dict[str, Any]:
    load_project(db, project_id, subject)
    rows = db.execute(
        select(TourImport)
        .where(TourImport.project_id == project_id)
        .order_by(TourImport.created_at, TourImport.id)
    ).scalars()
    return {"items": [ser.tour_import(r) for r in rows]}


@router.delete("/imports/{import_id}", status_code=204)
def delete_import(import_id: uuid.UUID, subject: Subject, db: Db) -> Response:
    scoped = load_scoped(db, TourImport, import_id, subject, "import")
    load_project(db, scoped.project_id, subject, roles=WRITERS, lock=True, require_active=True)
    audit.record(
        db,
        project_id=scoped.project_id,
        actor=subject,
        action="import.removed",
        resource_type="import",
        resource_id=scoped.id,
        data={"provider": scoped.provider, "modelId": scoped.model_id},
    )
    db.delete(scoped)
    db.commit()
    return Response(status_code=204)
