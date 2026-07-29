"""Retain only the last N months of lake data (local + R2 + DuckDB)."""

from __future__ import annotations

import io
import re
import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb
import pandas as pd

from flight_agent.config import get_settings, load_project_config

_BTS_MONTH_RE = re.compile(r"year=(\d{4})/month=(\d{2})")
_R2_BTS_RE = re.compile(r"raw/bts/year=(\d{4})/month=(\d{2})(?:/|$)")
_R2_WEATHER_RE = re.compile(r"^raw/weather/airport=[^/]+/weather\.parquet$")


@dataclass
class PruneResult:
    cutoff: date
    local_bts_removed: list[str]
    local_weather_rows_removed: int
    r2_bts_objects_removed: int
    r2_weather_rows_removed: int
    duckdb_rows_removed: int

    @property
    def r2_objects_removed(self) -> int:
        """Backward-compatible alias: BTS object deletes on R2."""
        return self.r2_bts_objects_removed


def retention_months(cfg: dict | None = None) -> int:
    cfg = cfg or load_project_config()
    return int(cfg.get("retention_months", 48))


def cutoff_date(as_of: date | None = None, months: int | None = None) -> date:
    """First day of the oldest month still retained (inclusive)."""
    as_of = as_of or date.today()
    months = months if months is not None else retention_months()
    y, m = as_of.year, as_of.month
    m -= months
    while m <= 0:
        m += 12
        y -= 1
    return date(y, m, 1)


def _ym_before(year: int, month: int, cutoff: date) -> bool:
    return (year, month) < (cutoff.year, cutoff.month)


def filter_weather_frame(df: pd.DataFrame, cutoff: date) -> tuple[pd.DataFrame, int]:
    """Keep weather rows on/after cutoff. Returns (filtered_df, rows_dropped)."""
    if df.empty or "date" not in df.columns:
        return df, 0
    dates = pd.to_datetime(df["date"], errors="coerce")
    keep = dates >= pd.Timestamp(cutoff)
    dropped = int((~keep).sum())
    return df.loc[keep].reset_index(drop=True), dropped


def prune_local_bts(cutoff: date, dry_run: bool = False) -> list[str]:
    settings = get_settings()
    bts_root = Path(settings.raw_dir) / "bts"
    removed: list[str] = []
    if not bts_root.exists():
        return removed

    for month_dir in sorted(bts_root.glob("year=*/month=*")):
        rel = month_dir.relative_to(bts_root).as_posix()
        m = _BTS_MONTH_RE.search(month_dir.as_posix())
        if not m:
            continue
        year, month = int(m.group(1)), int(m.group(2))
        if _ym_before(year, month, cutoff):
            removed.append(f"bts/{rel}")
            if not dry_run:
                shutil.rmtree(month_dir)
                year_dir = month_dir.parent
                if year_dir.exists() and not any(year_dir.iterdir()):
                    year_dir.rmdir()
    return removed


def prune_local_weather(cutoff: date, dry_run: bool = False) -> int:
    """Drop weather rows older than cutoff from per-airport parquet files."""
    settings = get_settings()
    weather_root = Path(settings.raw_dir) / "weather"
    rows_removed = 0
    if not weather_root.exists():
        return 0

    for path in weather_root.glob("airport=*/weather.parquet"):
        df = pd.read_parquet(path)
        filtered, dropped = filter_weather_frame(df, cutoff)
        if dropped == 0:
            continue
        rows_removed += dropped
        if not dry_run:
            filtered.to_parquet(path, index=False)
    return rows_removed


def _delete_r2_keys(client, bucket: str, keys: list[str], dry_run: bool) -> int:
    if not keys:
        return 0
    if dry_run:
        for key in keys:
            print(f"[dry-run] would delete s3://{bucket}/{key}")
        return len(keys)

    deleted = 0
    objects = [{"Key": k} for k in keys]
    for i in range(0, len(objects), 1000):
        chunk = objects[i : i + 1000]
        client.delete_objects(Bucket=bucket, Delete={"Objects": chunk, "Quiet": True})
        deleted += len(chunk)
        for item in chunk:
            print(f"Deleted s3://{bucket}/{item['Key']}")
    return deleted


def prune_r2_bts(cutoff: date, dry_run: bool = False) -> int:
    """Delete R2 BTS month partitions older than the retention window."""
    from flight_agent.ingest.r2_sync import _client

    settings = get_settings()
    if not settings.r2_configured:
        print("R2 not configured; skipping cloud BTS prune.")
        return 0

    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    to_delete: list[str] = []
    for page in paginator.paginate(Bucket=settings.r2_bucket, Prefix="raw/bts/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = _R2_BTS_RE.search(key)
            if not m:
                continue
            year, month = int(m.group(1)), int(m.group(2))
            if _ym_before(year, month, cutoff):
                to_delete.append(key)
    return _delete_r2_keys(client, settings.r2_bucket, to_delete, dry_run)


def prune_r2_weather(cutoff: date, dry_run: bool = False) -> int:
    """
    Trim old rows from R2 weather Parquet objects (same window as BTS).

    Downloads each weather file, drops rows before cutoff, re-uploads
    (or deletes the object if nothing remains).
    """
    from flight_agent.ingest.r2_sync import _client

    settings = get_settings()
    if not settings.r2_configured:
        print("R2 not configured; skipping cloud weather prune.")
        return 0

    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    weather_keys: list[str] = []
    for page in paginator.paginate(Bucket=settings.r2_bucket, Prefix="raw/weather/"):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if _R2_WEATHER_RE.match(key):
                weather_keys.append(key)

    rows_removed = 0
    for key in weather_keys:
        obj = client.get_object(Bucket=settings.r2_bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        filtered, dropped = filter_weather_frame(df, cutoff)
        if dropped == 0:
            continue
        rows_removed += dropped
        uri = f"s3://{settings.r2_bucket}/{key}"
        if dry_run:
            print(f"[dry-run] would drop {dropped:,} rows from {uri}")
            continue
        if filtered.empty:
            client.delete_object(Bucket=settings.r2_bucket, Key=key)
            print(f"Deleted empty weather object {uri}")
        else:
            buf = io.BytesIO()
            filtered.to_parquet(buf, index=False)
            client.put_object(
                Bucket=settings.r2_bucket,
                Key=key,
                Body=buf.getvalue(),
                ContentType="application/octet-stream",
            )
            print(f"Rewrote {uri} (−{dropped:,} rows, {len(filtered):,} kept)")
    return rows_removed


def prune_r2(cutoff: date, dry_run: bool = False) -> tuple[int, int]:
    """Prune R2 BTS partitions and weather rows. Returns (bts_objects, weather_rows)."""
    bts = prune_r2_bts(cutoff, dry_run=dry_run)
    weather = prune_r2_weather(cutoff, dry_run=dry_run)
    return bts, weather


def prune_duckdb(cutoff: date, dry_run: bool = False) -> int:
    """Remove rows older than cutoff from materialised DuckDB marts."""
    settings = get_settings()
    db_path = Path(settings.duckdb_path)
    if not db_path.exists():
        return 0

    tables_with_date = [
        ("flt_flights_clean", "fl_date"),
        ("flt_flights_with_weather", "fl_date"),
        ("stg_weather", "weather_date"),  # may be a view — delete only if table
    ]
    removed = 0
    con = duckdb.connect(str(db_path))
    try:
        existing_tables = {
            r[0]
            for r in con.execute(
                """
                select table_name from information_schema.tables
                where table_schema = 'main' and table_type = 'BASE TABLE'
                """
            ).fetchall()
        }
        existing = {
            r[0]
            for r in con.execute(
                "select table_name from information_schema.tables where table_schema = 'main'"
            ).fetchall()
        }
        for table, col in tables_with_date:
            if table not in existing_tables:
                continue
            count = con.execute(
                f"select count(*) from {table} where {col} < ?",
                [cutoff],
            ).fetchone()[0]
            removed += int(count)
            if count and not dry_run:
                con.execute(f"delete from {table} where {col} < ?", [cutoff])
                print(f"DuckDB: deleted {count:,} rows from {table} before {cutoff}")
        if not dry_run and removed > 0 and "flt_flights_clean" in existing:
            if "flt_route_delay_stats" in existing_tables:
                con.execute("drop table if exists flt_route_delay_stats")
                con.execute(
                    """
                    create table flt_route_delay_stats as
                    select
                      origin, dest, op_unique_carrier,
                      count(*) as n_flights,
                      avg(arr_delay) as avg_arr_delay,
                      avg(arr_delay_15) as pct_delay_15,
                      avg(distance) as avg_distance,
                      avg(taxi_out) as avg_taxi_out,
                      avg(taxi_in) as avg_taxi_in,
                      avg(nas_delay) as avg_nas_delay,
                      avg(carrier_delay) as avg_carrier_delay,
                      avg(late_aircraft_delay) as avg_late_aircraft_delay,
                      avg(crs_elapsed_time) as avg_crs_elapsed_time
                    from flt_flights_clean
                    group by 1, 2, 3
                    """
                )
            if "flt_carrier_delay_stats" in existing_tables:
                con.execute("drop table if exists flt_carrier_delay_stats")
                con.execute(
                    """
                    create table flt_carrier_delay_stats as
                    select
                      op_unique_carrier,
                      count(*) as n_flights,
                      avg(arr_delay) as avg_arr_delay,
                      avg(arr_delay_15) as pct_delay_15,
                      avg(dep_delay) as avg_dep_delay,
                      avg(taxi_out) as avg_taxi_out,
                      avg(nas_delay) as avg_nas_delay,
                      avg(late_aircraft_delay) as avg_late_aircraft_delay
                    from flt_flights_clean
                    group by 1
                    """
                )
            if "flt_airport_hour_stats" in existing:
                print(
                    "DuckDB: refreshed route/carrier aggregates; run `flight dbt build` "
                    "to fully refresh airport-hour stats from pruned Parquet."
                )
    finally:
        con.close()
    return removed


def prune_all(
    *,
    months: int | None = None,
    as_of: date | None = None,
    dry_run: bool = False,
    include_r2: bool = True,
) -> PruneResult:
    months = months if months is not None else retention_months()
    cutoff = cutoff_date(as_of=as_of, months=months)
    print(f"Retention window: {months} months → keep data on/after {cutoff.isoformat()}")

    local_bts = prune_local_bts(cutoff, dry_run=dry_run)
    for rel in local_bts:
        print(f"{'[dry-run] would remove' if dry_run else 'Removed'} local {rel}")

    weather_rows = prune_local_weather(cutoff, dry_run=dry_run)
    if weather_rows:
        print(
            f"{'[dry-run] would drop' if dry_run else 'Dropped'} "
            f"{weather_rows:,} local weather rows before {cutoff}"
        )

    r2_bts = 0
    r2_weather = 0
    if include_r2:
        r2_bts, r2_weather = prune_r2(cutoff, dry_run=dry_run)
        if r2_weather:
            print(
                f"{'[dry-run] would drop' if dry_run else 'Dropped'} "
                f"{r2_weather:,} R2 weather rows before {cutoff}"
            )

    duck_rows = prune_duckdb(cutoff, dry_run=dry_run)
    return PruneResult(
        cutoff=cutoff,
        local_bts_removed=local_bts,
        local_weather_rows_removed=weather_rows,
        r2_bts_objects_removed=r2_bts,
        r2_weather_rows_removed=r2_weather,
        duckdb_rows_removed=duck_rows,
    )
