"""Build ML training frames from DuckDB / R2 curated marts."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from flight_agent.config import load_project_config

# Leakage-safe: no same-flight taxi/NAS/dep_delay/cause minutes.
BASE_COLUMNS = [
    "op_unique_carrier",
    "origin",
    "dest",
    "fl_month",
    "fl_dow",
    "crs_dep_hour",
    "distance",
    "crs_elapsed_time",
    "is_weekend",
    "is_peak_hour",
    "origin_hour_sched_flights",
    "dest_hour_sched_flights",
    "origin_day_sched_flights",
    "origin_temp_c",
    "origin_precip_mm",
    "origin_wind_kmh",
    "origin_weathercode",
    "dest_temp_c",
    "dest_precip_mm",
    "dest_wind_kmh",
    "dest_weathercode",
    "route_hist_pct_delay_15",
    "origin_hist_avg_taxi_out",
    "origin_hist_avg_nas_delay",
    "origin_hist_avg_carrier_delay",
    "origin_hist_avg_weather_delay",
    "origin_hist_avg_late_aircraft_delay",
    "origin_hist_hour_ops",
    "origin_hist_hour_pct_delay_15",
    "dest_hist_avg_taxi_in",
    "dest_hist_hour_ops",
    "dest_hist_hour_pct_delay_15",
    "carrier_hist_pct_delay_15",
    "carrier_hist_avg_taxi_out",
    "carrier_hist_avg_late_aircraft_delay",
]

DERIVED_COLUMNS = [
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "precip_total_mm",
    "wind_max_kmh",
    "origin_congestion_risk",
    "dest_congestion_risk",
    "route_carrier_risk",
    "bad_weather",
    "peak_x_origin_delay",
]

FEATURE_COLUMNS = BASE_COLUMNS + DERIVED_COLUMNS
CATEGORICAL = ["op_unique_carrier", "origin", "dest"]
TARGET = "arr_delay_15"
REGRESSION_TARGET = "arr_delay"


DEFAULTS = {
    "distance": 800.0,
    "crs_elapsed_time": 150.0,
    "is_weekend": 0,
    "is_peak_hour": 0,
    "origin_hour_sched_flights": 20.0,
    "dest_hour_sched_flights": 20.0,
    "origin_day_sched_flights": 200.0,
    "origin_temp_c": 15.0,
    "origin_precip_mm": 0.0,
    "origin_wind_kmh": 10.0,
    "origin_weathercode": 0,
    "dest_temp_c": 15.0,
    "dest_precip_mm": 0.0,
    "dest_wind_kmh": 10.0,
    "dest_weathercode": 0,
    "route_hist_pct_delay_15": 0.2,
    "origin_hist_avg_taxi_out": 16.0,
    "origin_hist_avg_nas_delay": 5.0,
    "origin_hist_avg_carrier_delay": 5.0,
    "origin_hist_avg_weather_delay": 2.0,
    "origin_hist_avg_late_aircraft_delay": 5.0,
    "origin_hist_hour_ops": 50.0,
    "origin_hist_hour_pct_delay_15": 0.2,
    "dest_hist_avg_taxi_in": 8.0,
    "dest_hist_hour_ops": 50.0,
    "dest_hist_hour_pct_delay_15": 0.2,
    "carrier_hist_pct_delay_15": 0.2,
    "carrier_hist_avg_taxi_out": 16.0,
    "carrier_hist_avg_late_aircraft_delay": 5.0,
}


def default_train_rows(cfg: dict | None = None) -> int:
    """Rows to pull into pandas (full lake is ~10M+ and OOMs on laptops)."""
    cfg = cfg or load_project_config()
    return int((cfg.get("model") or {}).get("train_rows", 800_000))


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclical time + interaction features (leakage-safe)."""
    out = df.copy()
    hour = out["crs_dep_hour"].astype(float).clip(0, 23).fillna(12.0)
    month = out["fl_month"].astype(float).clip(1, 12).fillna(6.0)
    out["hour_sin"] = np.sin(2 * math.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * math.pi * hour / 24.0)
    out["month_sin"] = np.sin(2 * math.pi * month / 12.0)
    out["month_cos"] = np.cos(2 * math.pi * month / 12.0)
    out["precip_total_mm"] = (
        out["origin_precip_mm"].fillna(0.0) + out["dest_precip_mm"].fillna(0.0)
    )
    out["wind_max_kmh"] = np.maximum(
        out["origin_wind_kmh"].fillna(0.0), out["dest_wind_kmh"].fillna(0.0)
    )
    out["origin_congestion_risk"] = (
        out["origin_hist_hour_pct_delay_15"].fillna(0.2)
        * np.log1p(out["origin_hist_hour_ops"].fillna(50.0))
    )
    out["dest_congestion_risk"] = (
        out["dest_hist_hour_pct_delay_15"].fillna(0.2)
        * np.log1p(out["dest_hist_hour_ops"].fillna(50.0))
    )
    out["route_carrier_risk"] = (
        0.6 * out["route_hist_pct_delay_15"].fillna(0.2)
        + 0.4 * out["carrier_hist_pct_delay_15"].fillna(0.2)
    )
    out["bad_weather"] = (
        (out["precip_total_mm"] > 2.0) | (out["wind_max_kmh"] > 40.0)
    ).astype(int)
    out["peak_x_origin_delay"] = (
        out["is_peak_hour"].fillna(0).astype(float)
        * out["origin_hist_hour_pct_delay_15"].fillna(0.2)
    )
    return out


def build_training_frame(sample_limit: int | None = None) -> pd.DataFrame:
    """
    Build a training frame from the warehouse (local DuckDB or R2 curated).

    Samples in DuckDB before materializing to pandas unless sample_limit is 0
    (full lake — may OOM).
    """
    from flight_agent.ingest.warehouse import warehouse_connection

    cfg = load_project_config()
    hubs = [str(h).upper() for h in cfg.get("hubs", [])]
    hub_pair_only = bool((cfg.get("model") or {}).get("hub_pair_only", False))
    if sample_limit is None:
        sample_limit = default_train_rows(cfg)
    use_sample = sample_limit is not None and int(sample_limit) > 0
    n = int(sample_limit) if use_sample else 0

    hub_sql = ", ".join(f"'{h}'" for h in hubs) if hubs else "''"
    if hub_pair_only and hubs:
        hub_filter = f"and f.origin in ({hub_sql}) and f.dest in ({hub_sql})"
        scope = "hub↔hub only"
    else:
        # Keep lake as ingested (typically origin OR dest is a configured hub).
        hub_filter = ""
        scope = "hub_pair_only=false (all lake flights)"

    inner = f"""
    with route_hist as (
      select origin, dest, op_unique_carrier, pct_delay_15 as route_hist_pct_delay_15
      from flt_route_delay_stats
    )
    select
      f.fl_date,
      f.op_unique_carrier,
      f.origin,
      f.dest,
      f.fl_month,
      f.fl_dow,
      f.crs_dep_hour,
      coalesce(f.distance, {DEFAULTS['distance']}) as distance,
      coalesce(f.crs_elapsed_time, {DEFAULTS['crs_elapsed_time']}) as crs_elapsed_time,
      coalesce(f.is_weekend, 0) as is_weekend,
      coalesce(f.is_peak_hour, 0) as is_peak_hour,
      coalesce(f.origin_hour_sched_flights, {DEFAULTS['origin_hour_sched_flights']}) as origin_hour_sched_flights,
      coalesce(f.dest_hour_sched_flights, {DEFAULTS['dest_hour_sched_flights']}) as dest_hour_sched_flights,
      coalesce(f.origin_day_sched_flights, {DEFAULTS['origin_day_sched_flights']}) as origin_day_sched_flights,
      coalesce(f.origin_temp_c, {DEFAULTS['origin_temp_c']}) as origin_temp_c,
      coalesce(f.origin_precip_mm, {DEFAULTS['origin_precip_mm']}) as origin_precip_mm,
      coalesce(f.origin_wind_kmh, {DEFAULTS['origin_wind_kmh']}) as origin_wind_kmh,
      coalesce(f.origin_weathercode, {DEFAULTS['origin_weathercode']}) as origin_weathercode,
      coalesce(f.dest_temp_c, {DEFAULTS['dest_temp_c']}) as dest_temp_c,
      coalesce(f.dest_precip_mm, {DEFAULTS['dest_precip_mm']}) as dest_precip_mm,
      coalesce(f.dest_wind_kmh, {DEFAULTS['dest_wind_kmh']}) as dest_wind_kmh,
      coalesce(f.dest_weathercode, {DEFAULTS['dest_weathercode']}) as dest_weathercode,
      coalesce(r.route_hist_pct_delay_15, {DEFAULTS['route_hist_pct_delay_15']}) as route_hist_pct_delay_15,
      coalesce(f.origin_hist_avg_taxi_out, {DEFAULTS['origin_hist_avg_taxi_out']}) as origin_hist_avg_taxi_out,
      coalesce(f.origin_hist_avg_nas_delay, {DEFAULTS['origin_hist_avg_nas_delay']}) as origin_hist_avg_nas_delay,
      coalesce(f.origin_hist_avg_carrier_delay, {DEFAULTS['origin_hist_avg_carrier_delay']}) as origin_hist_avg_carrier_delay,
      coalesce(f.origin_hist_avg_weather_delay, {DEFAULTS['origin_hist_avg_weather_delay']}) as origin_hist_avg_weather_delay,
      coalesce(f.origin_hist_avg_late_aircraft_delay, {DEFAULTS['origin_hist_avg_late_aircraft_delay']}) as origin_hist_avg_late_aircraft_delay,
      coalesce(f.origin_hist_hour_ops, {DEFAULTS['origin_hist_hour_ops']}) as origin_hist_hour_ops,
      coalesce(f.origin_hist_hour_pct_delay_15, {DEFAULTS['origin_hist_hour_pct_delay_15']}) as origin_hist_hour_pct_delay_15,
      coalesce(f.dest_hist_avg_taxi_in, {DEFAULTS['dest_hist_avg_taxi_in']}) as dest_hist_avg_taxi_in,
      coalesce(f.dest_hist_hour_ops, {DEFAULTS['dest_hist_hour_ops']}) as dest_hist_hour_ops,
      coalesce(f.dest_hist_hour_pct_delay_15, {DEFAULTS['dest_hist_hour_pct_delay_15']}) as dest_hist_hour_pct_delay_15,
      coalesce(f.carrier_hist_pct_delay_15, {DEFAULTS['carrier_hist_pct_delay_15']}) as carrier_hist_pct_delay_15,
      coalesce(f.carrier_hist_avg_taxi_out, {DEFAULTS['carrier_hist_avg_taxi_out']}) as carrier_hist_avg_taxi_out,
      coalesce(f.carrier_hist_avg_late_aircraft_delay, {DEFAULTS['carrier_hist_avg_late_aircraft_delay']}) as carrier_hist_avg_late_aircraft_delay,
      f.arr_delay_15,
      f.arr_delay
    from flt_flights_with_weather f
    left join route_hist r
      on f.origin = r.origin
     and f.dest = r.dest
     and f.op_unique_carrier = r.op_unique_carrier
    where f.arr_delay_15 is not null
      and f.arr_delay is not null
      and f.crs_dep_hour is not null
      {hub_filter}
    """

    if use_sample:
        query = f"select * from ({inner}) using sample {n}"
        print(
            f"Building training frame: reservoir sample {n:,} rows ({scope})…",
            flush=True,
        )
    else:
        query = inner
        print(f"Building training frame: ALL rows ({scope})…", flush=True)

    with warehouse_connection(read_only=True) as con:
        df = con.execute(query).df()
    df = add_derived_features(df)
    print(
        f"Training frame ready: {len(df):,} rows, "
        f"delay_rate={float(df[TARGET].mean()):.3f}, "
        f"hub_pair_only={hub_pair_only}",
        flush=True,
    )
    return df
