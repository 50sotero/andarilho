"""Request bodies for the /v1 API (camelCase, unknown fields rejected)."""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic.alias_generators import to_camel

from andarilho_api.schemas.scene import Layout


class Body(BaseModel):
    model_config = ConfigDict(
        alias_generator=to_camel, populate_by_name=True, extra="forbid", allow_inf_nan=False
    )


Title = Annotated[str, Field(min_length=1, max_length=200)]


def _no_control(value: str) -> bool:
    return not any(ord(c) < 32 or ord(c) == 127 for c in value)


class ProjectCreate(Body):
    title: Title

    @field_validator("title")
    @classmethod
    def _title(cls, value: str) -> str:
        value = value.strip()
        if not value or not _no_control(value):
            raise ValueError("title must be non-blank text without control characters")
        return value


class ProjectUpdate(ProjectCreate):
    pass


class UploadCreate(Body):
    filename: Annotated[str, Field(min_length=1, max_length=255)]
    bytes: Annotated[int, Field(gt=0, le=1 << 50)]
    media_kind: Literal["video", "photo", "floor_plan"]
    content_type: Annotated[str, Field(min_length=3, max_length=100)]

    @field_validator("filename")
    @classmethod
    def _filename(cls, value: str) -> str:
        if not _no_control(value):
            raise ValueError("filename must not contain control characters")
        # Display metadata only: drop any directory components.
        name = value.replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not name:
            raise ValueError("filename must not be blank")
        return name

    @field_validator("content_type")
    @classmethod
    def _content_type(cls, value: str) -> str:
        return value.strip().lower()


PartNumber = Annotated[int, Field(ge=1, le=10000)]


class PartsRequest(Body):
    part_numbers: Annotated[list[PartNumber], Field(min_length=1, max_length=20)]

    @field_validator("part_numbers")
    @classmethod
    def _unique(cls, value: list[int]) -> list[int]:
        if len(set(value)) != len(value):
            raise ValueError("part numbers must be unique")
        return value


class EmptyBody(Body):
    pass


class CaptureSetCreate(Body):
    upload_ids: Annotated[list[uuid.UUID], Field(min_length=1, max_length=400)]
    floor_plan_upload_id: uuid.UUID | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("upload_ids")
    @classmethod
    def _unique(cls, value: list[uuid.UUID]) -> list[uuid.UUID]:
        if len(set(value)) != len(value):
            raise ValueError("upload ids must be unique")
        return value

    @field_validator("metadata")
    @classmethod
    def _bounded(cls, value: dict[str, Any]) -> dict[str, Any]:
        if len(json.dumps(value, allow_nan=False)) > 8192:
            raise ValueError("metadata must be at most 8 KiB of JSON")
        return value


class RunCreate(Body):
    capture_set_id: uuid.UUID
    quality: Literal["preview", "standard", "synthetic"]
    include_supplementary_photos: bool = True


class LayoutSave(Body):
    base_revision_id: uuid.UUID
    layout: Layout


class RevisionRef(Body):
    scene_revision_id: uuid.UUID


class ShareCreate(Body):
    scene_revision_id: uuid.UUID
    expires_at: datetime | None = None


class PublicDeliveryRequest(Body):
    token: Annotated[str, Field(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
