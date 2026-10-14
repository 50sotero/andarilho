"""Object storage abstraction and artifact bookkeeping.

Keys always live under ``tenants/{project_id}/``. Stage outputs go to immutable
attempt paths ``tenants/{project_id}/runs/{run_id}/attempts/{stage}/{fence}/...``
so a retried or late attempt can never overwrite another attempt's output;
viewer derivatives go to ``tenants/{project_id}/scenes/{scene_revision_id}/``
and are written with :func:`put_immutable` (never overwritten with different bytes).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from andarilho_worker.errors import StageError, TransientError

_KEY = re.compile(r"^tenants/[A-Za-z0-9_-]+/[A-Za-z0-9._/-]+$")
MAX_INLINE_BYTES = 256 * 1024 * 1024


def check_key(key: str) -> str:
    if not _KEY.match(key) or ".." in key.split("/") or "//" in key:
        raise ValueError(f"invalid object key {key!r}")
    return key


def attempt_prefix(project_id: str, run_id: str, stage: str, fence: int) -> str:
    return f"tenants/{project_id}/runs/{run_id}/attempts/{stage}/{fence}/"


def scene_prefix(project_id: str, scene_revision_id: str) -> str:
    return f"tenants/{project_id}/scenes/{scene_revision_id}/"


def project_prefix(project_id: str) -> str:
    return f"tenants/{project_id}/"


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size: int
    version: str | None


class ObjectStore(Protocol):
    def put_bytes(self, key: str, data: bytes, content_type: str = ...) -> ObjectInfo: ...
    def create_new(self, key: str, data: bytes, content_type: str = ...) -> ObjectInfo | None: ...
    def put_file(self, key: str, path: Path, content_type: str = ...) -> ObjectInfo: ...
    def get_bytes(self, key: str, version: str | None = None, max_bytes: int = ...) -> bytes: ...
    def download(self, key: str, dest: Path, version: str | None = None) -> str: ...
    def head(self, key: str, version: str | None = None) -> ObjectInfo | None: ...
    def copy(self, src: str, dst: str) -> ObjectInfo: ...
    def list_keys(self, prefix: str) -> list[str]: ...
    def delete_prefix(self, prefix: str) -> int: ...
    def abort_multipart_uploads(self, prefix: str) -> int: ...


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class LocalObjectStore:
    """Filesystem-backed store for tests and single-host development (no versioning)."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.multipart_uploads: dict[str, str] = {}  # upload id -> key (simulated)

    def _path(self, key: str) -> Path:
        return self.root.joinpath(*check_key(key).split("/"))

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
        return ObjectInfo(key, len(data), None)

    def create_new(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        """Atomically create ``key``; return ``None`` if it already exists."""
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_bytes(data)
        try:
            os.link(tmp, path)  # fails if the destination exists; never replaces it
        except FileExistsError:
            return None
        finally:
            tmp.unlink(missing_ok=True)
        return ObjectInfo(key, len(data), None)

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        dest = self._path(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(f".{dest.name}.{uuid.uuid4().hex}.tmp")
        shutil.copyfile(path, tmp)
        os.replace(tmp, dest)
        return ObjectInfo(key, dest.stat().st_size, None)

    def get_bytes(self, key: str, version: str | None = None, max_bytes: int = MAX_INLINE_BYTES):
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        if path.stat().st_size > max_bytes:
            raise StageError("BUDGET_EXHAUSTED", "An object is too large to process.")
        return path.read_bytes()

    def download(self, key: str, dest: Path, version: str | None = None) -> str:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, dest)
        return _sha256_file(dest)

    def head(self, key: str, version: str | None = None) -> ObjectInfo | None:
        path = self._path(key)
        return ObjectInfo(key, path.stat().st_size, None) if path.is_file() else None

    def copy(self, src: str, dst: str) -> ObjectInfo:
        return self.put_file(dst, self._path(src))

    def list_keys(self, prefix: str) -> list[str]:
        base = self.root
        out = []
        for path in base.rglob("*"):
            if path.is_file() and not path.name.endswith(".tmp"):
                key = path.relative_to(base).as_posix()
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list_keys(prefix)
        for key in keys:
            self._path(key).unlink(missing_ok=True)
        return len(keys)

    def abort_multipart_uploads(self, prefix: str) -> int:
        doomed = [uid for uid, key in self.multipart_uploads.items() if key.startswith(prefix)]
        for uid in doomed:
            del self.multipart_uploads[uid]
        return len(doomed)


class S3ObjectStore:
    """S3/MinIO store. Versioned buckets return and accept storage versions."""

    def __init__(self, bucket: str, client: Any) -> None:
        self.bucket = bucket
        self.client = client

    @classmethod
    def from_settings(cls, settings: Any) -> S3ObjectStore:
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=settings.s3_endpoint_url,
            region_name=settings.s3_region,
            aws_access_key_id=settings.s3_access_key_id,
            aws_secret_access_key=settings.s3_secret_access_key,
            config=Config(
                signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}
            ),
        )
        return cls(settings.s3_bucket, client)

    def _wrap(self, exc: Exception) -> Exception:
        code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
        if code in ("NoSuchKey", "404", "NotFound", "NoSuchVersion"):
            return FileNotFoundError("object not found")
        return TransientError("TRANSIENT_FAILURE", "Storage is temporarily unavailable.")

    def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        try:
            resp = self.client.put_object(
                Bucket=self.bucket,
                Key=check_key(key),
                Body=data,
                ContentType=content_type,
                ChecksumAlgorithm="SHA256",
            )
        except Exception as exc:  # botocore raises many types
            raise self._wrap(exc) from exc
        return ObjectInfo(key, len(data), resp.get("VersionId"))

    def create_new(self, key: str, data: bytes, content_type: str = "application/octet-stream"):
        """Conditional create (``If-None-Match: *``); return ``None`` if ``key`` exists."""
        try:
            resp = self.client.put_object(
                Bucket=self.bucket,
                Key=check_key(key),
                Body=data,
                ContentType=content_type,
                ChecksumAlgorithm="SHA256",
                IfNoneMatch="*",
            )
        except Exception as exc:
            code = getattr(exc, "response", {}).get("Error", {}).get("Code", "")
            if code in ("PreconditionFailed", "412", "ConditionalRequestConflict"):
                return None
            raise self._wrap(exc) from exc
        return ObjectInfo(key, len(data), resp.get("VersionId"))

    def put_file(self, key: str, path: Path, content_type: str = "application/octet-stream"):
        try:
            self.client.upload_file(
                str(path),
                self.bucket,
                check_key(key),
                ExtraArgs={"ContentType": content_type, "ChecksumAlgorithm": "SHA256"},
            )
            info = self.head(key)
        except Exception as exc:
            raise self._wrap(exc) from exc
        if info is None:
            raise TransientError("TRANSIENT_FAILURE", "Storage is temporarily unavailable.")
        return info

    def _get(self, key: str, version: str | None):
        kwargs = {"Bucket": self.bucket, "Key": check_key(key)}
        if version:
            kwargs["VersionId"] = version
        try:
            return self.client.get_object(**kwargs)
        except Exception as exc:
            raise self._wrap(exc) from exc

    def get_bytes(self, key: str, version: str | None = None, max_bytes: int = MAX_INLINE_BYTES):
        resp = self._get(key, version)
        if resp["ContentLength"] > max_bytes:
            raise StageError("BUDGET_EXHAUSTED", "An object is too large to process.")
        return resp["Body"].read()

    def download(self, key: str, dest: Path, version: str | None = None) -> str:
        """Stream an exact key/version to disk and return the sha256 of the bytes written."""
        resp = self._get(key, version)
        dest.parent.mkdir(parents=True, exist_ok=True)
        h = hashlib.sha256()
        with dest.open("wb") as fh:
            for chunk in resp["Body"].iter_chunks(1024 * 1024):
                h.update(chunk)
                fh.write(chunk)
        return h.hexdigest()

    def head(self, key: str, version: str | None = None) -> ObjectInfo | None:
        kwargs = {"Bucket": self.bucket, "Key": check_key(key)}
        if version:
            kwargs["VersionId"] = version
        try:
            resp = self.client.head_object(**kwargs)
        except Exception as exc:
            wrapped = self._wrap(exc)
            if isinstance(wrapped, FileNotFoundError):
                return None
            raise wrapped from exc
        return ObjectInfo(key, resp["ContentLength"], resp.get("VersionId"))

    def copy(self, src: str, dst: str) -> ObjectInfo:
        try:
            resp = self.client.copy_object(
                Bucket=self.bucket,
                Key=check_key(dst),
                CopySource={"Bucket": self.bucket, "Key": check_key(src)},
            )
        except Exception as exc:
            raise self._wrap(exc) from exc
        info = self.head(dst)
        if info is None:
            raise TransientError("TRANSIENT_FAILURE", "Storage is temporarily unavailable.")
        return ObjectInfo(dst, info.size, resp.get("VersionId") or info.version)

    def _iter_versions(self, prefix: str) -> Iterator[dict[str, str]]:
        paginator = self.client.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for item in page.get("Versions", []) + page.get("DeleteMarkers", []):
                yield {"Key": item["Key"], "VersionId": item["VersionId"]}

    def list_keys(self, prefix: str) -> list[str]:
        try:
            check_key(prefix.rstrip("/") + "/x")
            keys = {v["Key"] for v in self._iter_versions(prefix)}
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                keys.update(item["Key"] for item in page.get("Contents", []))
        except Exception as exc:
            raise self._wrap(exc) from exc
        return sorted(keys)

    def delete_prefix(self, prefix: str) -> int:
        """Delete every object version and delete marker under ``prefix``."""
        check_key(prefix.rstrip("/") + "/x")
        deleted = 0
        try:
            batch: list[dict[str, str]] = []
            for item in self._iter_versions(prefix):
                batch.append(item)
                if len(batch) == 1000:
                    self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
                    deleted += len(batch)
                    batch = []
            if batch:
                self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})
                deleted += len(batch)
            # Unversioned buckets: also remove current objects directly.
            paginator = self.client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                objs = [{"Key": i["Key"]} for i in page.get("Contents", [])]
                if objs:
                    self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": objs})
                    deleted += len(objs)
        except Exception as exc:
            raise self._wrap(exc) from exc
        return deleted

    def abort_multipart_uploads(self, prefix: str) -> int:
        aborted = 0
        try:
            paginator = self.client.get_paginator("list_multipart_uploads")
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                for upload in page.get("Uploads", []):
                    self.client.abort_multipart_upload(
                        Bucket=self.bucket, Key=upload["Key"], UploadId=upload["UploadId"]
                    )
                    aborted += 1
        except Exception as exc:
            raise self._wrap(exc) from exc
        return aborted


def put_immutable(store: ObjectStore, key: str, data: bytes, content_type: str) -> ObjectInfo:
    """Write ``data`` once; an identical existing object is reused, a different one is an error.

    The create is atomic (conditional), so a stale attempt racing a retry can never
    replace bytes that the fenced winner already committed to a manifest.
    """
    created = store.create_new(key, data, content_type)
    if created is not None:
        return created
    existing = store.head(key)
    current = store.get_bytes(key)
    if existing is not None and hashlib.sha256(current).digest() == hashlib.sha256(data).digest():
        return existing
    raise StageError("ARTIFACT_CONFLICT")


def put_json(store: ObjectStore, key: str, value: Mapping[str, Any]) -> tuple[ObjectInfo, str]:
    blob = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    info = store.put_bytes(key, blob, "application/json")
    return info, hashlib.sha256(blob).hexdigest()


def insert_artifact(
    conn: Any,
    *,
    project_id: str,
    run_id: str | None,
    job_id: str | None,
    kind: str,
    key: str,
    version: str | None,
    sha256: str,
    size: int,
    provenance: Mapping[str, Any],
    validation_state: str,
) -> str:
    """Insert (or reuse) an artifacts row; call only inside a fenced transaction."""
    row = conn.execute(
        """
        INSERT INTO artifacts (id, project_id, run_id, job_id, kind, object_key, storage_version,
                               sha256, bytes, provenance, validation_state)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
        ON CONFLICT (object_key) DO UPDATE SET validation_state = EXCLUDED.validation_state
        WHERE artifacts.sha256 = EXCLUDED.sha256 AND artifacts.project_id = EXCLUDED.project_id
        RETURNING id
        """,
        (
            str(uuid.uuid4()),
            project_id,
            run_id,
            job_id,
            kind,
            key,
            version,
            sha256,
            size,
            json.dumps(dict(provenance)),
            validation_state,
        ),
    ).fetchone()
    if row is None:
        raise StageError("ARTIFACT_CONFLICT")
    return str(row["id"])
