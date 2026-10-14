"""Create and configure the local private bucket (README sections 7 and 14).

Idempotent: creates the bucket if missing, enables versioning, applies CORS from
STORAGE_CORS_FILE (MinIO configures CORS server-wide and may answer
NotImplemented, which is tolerated), and adds an abort-incomplete-multipart
lifecycle rule. Intended for local development only; production buckets are
provisioned by infrastructure tooling.

  python /app/scripts/bootstrap-storage.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError, EndpointConnectionError

TOLERATED = {"NotImplemented", "MalformedXML", "InvalidRequest"}


def main() -> int:
    if os.environ.get("APP_ENV") == "production":
        print("bootstrap-storage is for local development only", file=sys.stderr)
        return 2
    bucket = os.environ["S3_BUCKET"]
    s3 = boto3.client(
        "s3",
        endpoint_url=os.environ.get("S3_ENDPOINT_URL") or None,
        region_name=os.environ.get("S3_REGION", "us-east-1"),
        aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )

    for attempt in range(30):
        try:
            s3.list_buckets()
            break
        except EndpointConnectionError:
            time.sleep(min(2.0, 0.2 * (attempt + 1)))
    else:
        print("storage endpoint is not reachable", file=sys.stderr)
        return 1

    try:
        s3.head_bucket(Bucket=bucket)
        print(f"bucket {bucket} exists")
    except ClientError:
        s3.create_bucket(Bucket=bucket)
        print(f"created bucket {bucket}")

    s3.put_bucket_versioning(Bucket=bucket, VersioningConfiguration={"Status": "Enabled"})
    print("versioning enabled")

    cors_file = os.environ.get("STORAGE_CORS_FILE")
    if cors_file and Path(cors_file).exists():
        cors = json.loads(Path(cors_file).read_text(encoding="utf-8"))
        app_url = os.environ.get("NEXT_PUBLIC_APP_URL")
        if app_url:
            for rule in cors["CORSRules"]:
                rule["AllowedOrigins"] = [app_url]
        try:
            s3.put_bucket_cors(Bucket=bucket, CORSConfiguration=cors)
            print("bucket CORS applied")
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") not in TOLERATED:
                raise
            print("bucket CORS is configured server-wide by this storage provider")

    s3.put_bucket_lifecycle_configuration(
        Bucket=bucket,
        LifecycleConfiguration={
            "Rules": [
                {
                    "ID": "abort-incomplete-multipart",
                    "Status": "Enabled",
                    "Filter": {"Prefix": "tenants/"},
                    "AbortIncompleteMultipartUpload": {"DaysAfterInitiation": 2},
                }
            ]
        },
    )
    print("lifecycle rule applied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
