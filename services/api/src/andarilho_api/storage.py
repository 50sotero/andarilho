"""S3 access for the control plane.

Object layout (README section 7). The API chooses every key; user filenames
are display metadata only:

* ``tenants/{project_id}/staging/{upload_id}`` - multipart upload target. The
  browser writes here through short-lived presigned part URLs.
* ``tenants/{project_id}/originals/{upload_id}`` - immutable copy made by the
  API after completion with a version-conditioned server-side copy
  (``CopySourceIfMatch`` on the completed object's ETag). No client upload
  grant ever targets this key. Capture sets and jobs reference only these keys
  (plus ``storage_version`` when bucket versioning is enabled).
* ``tenants/{project_id}/runs/{run_id}/attempts/{stage}/{job_id}/{fence}/...`` -
  worker attempt outputs; an attempt that does not succeed is deleted by its worker
  or swept by the job's next attempt.
* ``tenants/{project_id}/scenes/{scene_revision_id}/...`` - viewer derivatives
  (render/collision/poster/manifest) written by the worker.

Two clients are kept: ``internal`` (``S3_ENDPOINT_URL``) performs server
operations, and ``public`` (``S3_PUBLIC_ENDPOINT_URL``) only signs URLs handed
to browsers, so a browser never receives an internal hostname.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from andarilho_api.settings import Settings


def staging_key(project_id: uuid.UUID, upload_id: uuid.UUID) -> str:
    return f"tenants/{project_id}/staging/{upload_id}"


def immutable_key(project_id: uuid.UUID, upload_id: uuid.UUID) -> str:
    return f"tenants/{project_id}/originals/{upload_id}"


def tenant_prefix(project_id: uuid.UUID) -> str:
    return f"tenants/{project_id}/"


def scene_prefix(project_id: uuid.UUID, scene_revision_id: uuid.UUID) -> str:
    return f"tenants/{project_id}/scenes/{scene_revision_id}/"


class StorageError(Exception):
    """A storage operation failed. Never shown to clients verbatim."""


class PreconditionFailed(StorageError):
    pass


@dataclass(frozen=True)
class Part:
    part_number: int
    etag: str
    size: int


@dataclass(frozen=True)
class ObjectInfo:
    size: int
    etag: str
    version_id: str | None


def _client(settings: Settings, endpoint: str | None) -> Any:
    config = Config(
        signature_version="s3v4",
        retries={"max_attempts": 3, "mode": "standard"},
        s3={"addressing_style": "path" if endpoint else "auto"},
        connect_timeout=5,
        read_timeout=30,
    )
    return boto3.client(
        "s3",
        endpoint_url=endpoint or None,
        region_name=settings.s3_region,
        aws_access_key_id=settings.s3_access_key_id or None,
        aws_secret_access_key=settings.s3_secret_access_key or None,
        config=config,
    )


class Storage:
    def __init__(self, settings: Settings) -> None:
        self.bucket = settings.s3_bucket
        self.internal = _client(settings, settings.s3_endpoint_url)
        public_endpoint = settings.s3_public_endpoint_url or settings.s3_endpoint_url
        self.public = _client(settings, public_endpoint)

    # -- multipart -------------------------------------------------------
    def create_multipart_upload(self, key: str, content_type: str) -> str:
        response = self.internal.create_multipart_upload(
            Bucket=self.bucket, Key=key, ContentType=content_type
        )
        return response["UploadId"]

    def presign_upload_part(
        self, key: str, multipart_upload_id: str, part_number: int, expires_in: int
    ) -> str:
        return self.public.generate_presigned_url(
            "upload_part",
            Params={
                "Bucket": self.bucket,
                "Key": key,
                "UploadId": multipart_upload_id,
                "PartNumber": part_number,
            },
            ExpiresIn=expires_in,
        )

    def list_parts(self, key: str, multipart_upload_id: str) -> list[Part]:
        parts: list[Part] = []
        marker = 0
        while True:
            response = self.internal.list_parts(
                Bucket=self.bucket,
                Key=key,
                UploadId=multipart_upload_id,
                PartNumberMarker=marker,
                MaxParts=1000,
            )
            for item in response.get("Parts", []):
                parts.append(Part(int(item["PartNumber"]), item["ETag"], int(item["Size"])))
            if not response.get("IsTruncated"):
                return parts
            marker = int(response["NextPartNumberMarker"])

    def complete_multipart_upload(
        self, key: str, multipart_upload_id: str, parts: list[Part]
    ) -> None:
        self.internal.complete_multipart_upload(
            Bucket=self.bucket,
            Key=key,
            UploadId=multipart_upload_id,
            MultipartUpload={
                "Parts": [{"PartNumber": p.part_number, "ETag": p.etag} for p in parts]
            },
        )

    def abort_multipart_upload(self, key: str, multipart_upload_id: str) -> None:
        try:
            self.internal.abort_multipart_upload(
                Bucket=self.bucket, Key=key, UploadId=multipart_upload_id
            )
        except ClientError as exc:
            if _code(exc) not in ("NoSuchUpload", "404"):
                raise StorageError("abort failed") from exc
        except BotoCoreError as exc:
            raise StorageError("abort failed") from exc

    # -- objects ---------------------------------------------------------
    def head_object(self, key: str) -> ObjectInfo | None:
        try:
            response = self.internal.head_object(Bucket=self.bucket, Key=key)
        except ClientError as exc:
            if _code(exc) in ("404", "NoSuchKey", "NotFound"):
                return None
            raise StorageError("head failed") from exc
        except BotoCoreError as exc:
            raise StorageError("head failed") from exc
        return ObjectInfo(
            size=int(response["ContentLength"]),
            etag=response["ETag"],
            version_id=_version(response.get("VersionId")),
        )

    def copy_if_match(self, source_key: str, dest_key: str, etag: str) -> str | None:
        """Server-side copy conditioned on the source ETag; returns the new VersionId."""
        try:
            response = self.internal.copy_object(
                Bucket=self.bucket,
                Key=dest_key,
                CopySource={"Bucket": self.bucket, "Key": source_key},
                CopySourceIfMatch=etag,
                MetadataDirective="COPY",
            )
        except ClientError as exc:
            if _code(exc) in ("PreconditionFailed", "412"):
                raise PreconditionFailed("source changed during copy") from exc
            raise StorageError("copy failed") from exc
        except BotoCoreError as exc:
            raise StorageError("copy failed") from exc
        return _version(response.get("VersionId"))

    def put_object(self, key: str, body: bytes, content_type: str) -> str | None:
        response = self.internal.put_object(
            Bucket=self.bucket, Key=key, Body=body, ContentType=content_type
        )
        return _version(response.get("VersionId"))

    def delete_all_versions(self, key: str) -> None:
        """Remove every stored version and delete marker of exactly ``key``.

        The bucket is versioned (``bootstrap-storage.py``): a plain delete would only
        add a delete marker and keep the bytes. On an unversioned bucket the single
        version is listed as ``null`` and removed the same way.
        """
        try:
            targets = []
            paginator = self.internal.get_paginator("list_object_versions")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=key):
                for item in [*page.get("Versions", []), *page.get("DeleteMarkers", [])]:
                    if item["Key"] == key:
                        targets.append({"Key": key, "VersionId": item["VersionId"]})
            for start in range(0, len(targets), 1000):
                response = self.internal.delete_objects(
                    Bucket=self.bucket,
                    Delete={"Objects": targets[start : start + 1000], "Quiet": True},
                )
                if response.get("Errors"):
                    raise StorageError("delete failed")
        except (ClientError, BotoCoreError) as exc:
            raise StorageError("delete failed") from exc

    def presign_get(self, key: str, expires_in: int, version_id: str | None = None) -> str:
        params: dict[str, Any] = {"Bucket": self.bucket, "Key": key}
        if version_id:
            params["VersionId"] = version_id
        return self.public.generate_presigned_url("get_object", Params=params, ExpiresIn=expires_in)

    def check_bucket(self) -> bool:
        try:
            self.internal.head_bucket(Bucket=self.bucket)
        except Exception:  # noqa: BLE001 - readiness only reports up/down
            return False
        return True


def _code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _version(value: Any) -> str | None:
    if not value or value == "null":
        return None
    return str(value)
