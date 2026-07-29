"""Sync local data/ tree to Cloudflare R2 (S3-compatible)."""

from __future__ import annotations

from pathlib import Path

import boto3
from botocore.client import Config

from flight_agent.config import get_settings


def _client():
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError(
            "R2 is not configured. Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, "
            "R2_SECRET_ACCESS_KEY in .env (see .env.example)."
        )
    return boto3.client(
        "s3",
        endpoint_url=settings.r2_endpoint,
        aws_access_key_id=settings.r2_access_key_id,
        aws_secret_access_key=settings.r2_secret_access_key,
        config=Config(signature_version="s3v4"),
        region_name="auto",
    )


def sync_to_r2(prefix: str = "raw/") -> int:
    """Upload files under data/{prefix} to R2 with the same key layout."""
    settings = get_settings()
    client = _client()
    root = Path(settings.data_dir)
    local_root = root / prefix.rstrip("/")
    if not local_root.exists():
        raise FileNotFoundError(f"Local path not found: {local_root}")

    uploaded = 0
    for path in local_root.rglob("*"):
        if not path.is_file():
            continue
        key = str(path.relative_to(root)).replace("\\", "/")
        client.upload_file(str(path), settings.r2_bucket, key)
        uploaded += 1
        print(f"Uploaded s3://{settings.r2_bucket}/{key}")
    return uploaded


def sync_from_r2(prefix: str = "raw/") -> int:
    """Download R2 objects under prefix into local data/."""
    settings = get_settings()
    client = _client()
    root = Path(settings.data_dir)
    paginator = client.get_paginator("list_objects_v2")
    downloaded = 0
    for page in paginator.paginate(Bucket=settings.r2_bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            dest = root / key
            dest.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(settings.r2_bucket, key, str(dest))
            downloaded += 1
            print(f"Downloaded {key}")
    return downloaded
