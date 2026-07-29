"""Lake I/O: write/read Parquet locally and/or directly to Cloudflare R2."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd

from flight_agent.config import get_settings


def raw_key(*parts: str) -> str:
    """Build an object key under raw/, e.g. raw/bts/year=2024/month=01/flights.parquet."""
    cleaned = [p.strip("/").replace("\\", "/") for p in parts if p]
    return "/".join(["raw", *cleaned])


def write_parquet(
    df: pd.DataFrame,
    relative_path: str,
    *,
    to_r2: bool = False,
    keep_local: bool = True,
) -> str:
    """
    Write a DataFrame as Parquet.

    relative_path: path under raw/, e.g. "bts/year=2024/month=01/flights.parquet"
    Returns a location string (local path and/or s3:// URI).
    """
    relative_path = relative_path.lstrip("/").replace("\\", "/")
    if relative_path.startswith("raw/"):
        relative_path = relative_path[4:]
    key = raw_key(relative_path)
    settings = get_settings()
    locations: list[str] = []

    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    payload = buf.getvalue()

    if keep_local:
        local = Path(settings.data_dir) / key
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(payload)
        locations.append(str(local))

    if to_r2:
        if not settings.r2_configured:
            raise RuntimeError(
                "Direct R2 upload requested but R2 is not configured. "
                "Set R2_ACCOUNT_ID, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY in .env."
            )
        from flight_agent.ingest.r2_sync import _client

        client = _client()
        client.put_object(
            Bucket=settings.r2_bucket,
            Key=key,
            Body=payload,
            ContentType="application/octet-stream",
        )
        uri = f"s3://{settings.r2_bucket}/{key}"
        locations.append(uri)
        print(f"Uploaded {uri} ({len(df):,} rows)")

    if not locations:
        raise ValueError("write_parquet requires keep_local=True and/or to_r2=True")

    return locations[-1] if to_r2 else locations[0]


def read_parquet(relative_path: str) -> pd.DataFrame:
    """Read Parquet from local data/ first, then fall back to R2."""
    relative_path = relative_path.lstrip("/").replace("\\", "/")
    if relative_path.startswith("raw/"):
        relative_path = relative_path[4:]
    key = raw_key(relative_path)
    settings = get_settings()
    local = Path(settings.data_dir) / key
    if local.exists():
        return pd.read_parquet(local)

    if not settings.r2_configured:
        raise FileNotFoundError(
            f"Missing {local} and R2 is not configured for fallback."
        )

    from flight_agent.ingest.r2_sync import _client

    client = _client()
    obj = client.get_object(Bucket=settings.r2_bucket, Key=key)
    return pd.read_parquet(io.BytesIO(obj["Body"].read()))


def parquet_root_for_dbt(*, prefer_r2: bool = False) -> str:
    """
    Root passed to dbt read_parquet globs.
    Local: /abs/path/data/raw
    R2:    s3://bucket/raw
    """
    settings = get_settings()
    if prefer_r2:
        if not settings.r2_configured:
            raise RuntimeError("prefer_r2=True but R2 is not configured.")
        return f"s3://{settings.r2_bucket}/raw"
    return str((Path(settings.data_dir) / "raw").resolve())
