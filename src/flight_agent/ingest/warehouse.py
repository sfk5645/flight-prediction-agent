"""DuckDB warehouse sync: curated Parquet + .duckdb on R2 (minimal local disk)."""

from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import duckdb

from flight_agent.config import get_settings
from flight_agent.ingest.r2_sync import _client

# Gold marts written by dbt (exported to curated/ on R2).
MART_TABLES = (
    "flt_airport_hour_stats",
    "flt_carrier_delay_stats",
    "flt_route_delay_stats",
    "flt_flights_clean",
    "flt_flights_with_weather",
)

DUCKDB_R2_KEY = "warehouse/flight_agent.duckdb"
CURATED_PREFIX = "curated/"


def _log(msg: str) -> None:
    print(msg, flush=True)


def curated_uri(table: str) -> str:
    settings = get_settings()
    return f"s3://{settings.r2_bucket}/{CURATED_PREFIX}{table}.parquet"


def curated_key(table: str) -> str:
    return f"{CURATED_PREFIX}{table}.parquet"


def duckdb_r2_uri() -> str:
    settings = get_settings()
    return f"s3://{settings.r2_bucket}/{DUCKDB_R2_KEY}"


def keep_duckdb_local() -> bool:
    """When False (default), purge local .duckdb after pushing to R2."""
    return bool(get_settings().duckdb_keep_local)


def configure_r2(con: duckdb.DuckDBPyConnection) -> None:
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("R2 is not configured.")
    endpoint = settings.r2_endpoint.replace("https://", "").replace("http://", "")
    con.execute("INSTALL httpfs;")
    con.execute("LOAD httpfs;")
    con.execute(f"SET s3_endpoint='{endpoint}';")
    con.execute(f"SET s3_access_key_id='{settings.r2_access_key_id}';")
    con.execute(f"SET s3_secret_access_key='{settings.r2_secret_access_key}';")
    con.execute("SET s3_url_style='path';")
    con.execute("SET s3_use_ssl=true;")
    con.execute("SET s3_region='auto';")


def curated_on_r2() -> bool:
    settings = get_settings()
    if not settings.r2_configured:
        return False
    client = _client()
    key = curated_key("flt_flights_with_weather")
    try:
        client.head_object(Bucket=settings.r2_bucket, Key=key)
        return True
    except Exception:  # noqa: BLE001
        return False


def warehouse_available() -> bool:
    path = Path(get_settings().duckdb_path)
    return path.exists() or curated_on_r2()


def export_marts_to_r2(*, local_db: Path | None = None) -> list[str]:
    """
    Export dbt mart tables to R2 curated/*.parquet.

    Writes each table to a temp local Parquet file, uploads via boto3, then
    deletes the temp file (avoids flaky DuckDB httpfs COPY for multi-GB tables).
    """
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("export_marts_to_r2 requires R2 credentials.")
    db = Path(local_db or settings.duckdb_path)
    if not db.exists():
        raise FileNotFoundError(f"Local DuckDB missing at {db}; run dbt build first.")

    client = _client()
    con = duckdb.connect(str(db), read_only=True)
    written: list[str] = []
    try:
        for table in MART_TABLES:
            exists = con.execute(
                "select count(*) from information_schema.tables "
                "where table_name = ? and table_schema = 'main'",
                [table],
            ).fetchone()[0]
            if not exists:
                _log(f"Skip export (missing table): {table}")
                continue

            with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                _log(f"Exporting {table} → temp Parquet…")
                # Escape path for DuckDB string literal
                escaped = str(tmp_path).replace("'", "''")
                con.execute(f"COPY (SELECT * FROM {table}) TO '{escaped}' (FORMAT PARQUET)")
                size_mb = tmp_path.stat().st_size / (1024**2)
                key = curated_key(table)
                _log(f"Uploading {table} ({size_mb:.1f} MiB) → s3://{settings.r2_bucket}/{key}")
                client.upload_file(str(tmp_path), settings.r2_bucket, key)
                uri = curated_uri(table)
                written.append(uri)
                _log(f"Exported {table} → {uri}")
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()
        return written
    finally:
        con.close()


def upload_duckdb_to_r2(*, local_db: Path | None = None) -> str:
    """Upload the .duckdb file to R2 (backup / optional pull)."""
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("upload_duckdb_to_r2 requires R2 credentials.")
    db = Path(local_db or settings.duckdb_path)
    if not db.exists():
        raise FileNotFoundError(f"Local DuckDB missing at {db}")

    client = _client()
    size_gb = db.stat().st_size / (1024**3)
    _log(f"Uploading DuckDB ({size_gb:.2f} GiB) → {duckdb_r2_uri()} …")
    client.upload_file(str(db), settings.r2_bucket, DUCKDB_R2_KEY)
    _log(f"Uploaded {duckdb_r2_uri()}")
    return duckdb_r2_uri()


def download_duckdb_from_r2(*, force: bool = False) -> Path:
    """Download warehouse/flight_agent.duckdb from R2 to the local path."""
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("download_duckdb_from_r2 requires R2 credentials.")
    dest = Path(settings.duckdb_path)
    if dest.exists() and not force:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    client = _client()
    _log(f"Downloading {duckdb_r2_uri()} → {dest} …")
    client.download_file(settings.r2_bucket, DUCKDB_R2_KEY, str(dest))
    _log(f"Downloaded {dest}")
    return dest


def delete_local_duckdb() -> bool:
    """Remove local .duckdb (and WAL) to free disk."""
    settings = get_settings()
    db = Path(settings.duckdb_path)
    removed = False
    for path in (db, Path(str(db) + ".wal")):
        if path.exists():
            path.unlink()
            _log(f"Deleted {path}")
            removed = True
    return removed


def push_warehouse_to_r2(*, delete_local: bool | None = None) -> dict[str, object]:
    """
    Export curated marts + upload .duckdb to R2.
    By default deletes the local DuckDB afterward when FLIGHT_DUCKDB_KEEP_LOCAL=false.
    """
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("push_warehouse_to_r2 requires R2 credentials.")

    # Upload the .duckdb first so a crash mid-export still has a warehouse backup.
    duck_uri = upload_duckdb_to_r2()
    curated = export_marts_to_r2()
    should_delete = (not keep_duckdb_local()) if delete_local is None else delete_local
    deleted = False
    if should_delete:
        deleted = delete_local_duckdb()
    return {
        "curated": curated,
        "duckdb": duck_uri,
        "deleted_local": deleted,
    }


def _attach_r2_views(con: duckdb.DuckDBPyConnection) -> None:
    """Create views over curated Parquet (+ bronze weather) on R2."""
    configure_r2(con)
    settings = get_settings()
    for table in MART_TABLES:
        uri = curated_uri(table)
        con.execute(
            f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{uri}')"
        )
    weather_glob = f"s3://{settings.r2_bucket}/raw/weather/**/*.parquet"
    con.execute(
        f"""
        CREATE OR REPLACE VIEW stg_weather AS
        SELECT
          cast(date as date) as weather_date,
          upper(cast(airport as varchar)) as airport,
          cast(temperature_2m_mean as double) as temperature_2m_mean,
          cast(precipitation_sum as double) as precipitation_sum,
          cast(windspeed_10m_max as double) as windspeed_10m_max,
          cast(weathercode as integer) as weathercode,
          cast(lat as double) as lat,
          cast(lon as double) as lon
        FROM read_parquet('{weather_glob}', hive_partitioning=true, union_by_name=true)
        WHERE date IS NOT NULL AND airport IS NOT NULL
        """
    )


@contextmanager
def warehouse_connection(*, read_only: bool = True) -> Iterator[duckdb.DuckDBPyConnection]:
    """
    Open the analytics warehouse.

    Prefer local DuckDB if present; otherwise query curated Parquet on R2
    via an in-memory DuckDB (no large local file required).
    """
    settings = get_settings()
    db = Path(settings.duckdb_path)
    if db.exists():
        con = duckdb.connect(str(db), read_only=read_only)
        try:
            yield con
        finally:
            con.close()
        return

    if not settings.r2_configured or not curated_on_r2():
        raise FileNotFoundError(
            f"DuckDB missing at {db} and curated marts not found on R2. "
            "Run `flight dbt build --from-r2` then `flight warehouse push`."
        )

    con = duckdb.connect(":memory:")
    try:
        _attach_r2_views(con)
        yield con
    finally:
        con.close()
