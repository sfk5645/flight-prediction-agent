"""Shared prediction / stats services used by API and agent tools."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from flight_agent.config import get_settings
from flight_agent.features.build import DEFAULTS, FEATURE_COLUMNS, add_derived_features
from flight_agent.ingest.warehouse import warehouse_available, warehouse_connection


@lru_cache
def load_model():
    from flight_agent.train.r2_model import ensure_model_artifacts

    settings = get_settings()
    ensure_model_artifacts()
    path = Path(settings.model_dir) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `flight train` or `flight model pull`."
        )
    return joblib.load(path)


@lru_cache
def load_meta() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.model_dir) / "meta.json"
    if path.exists():
        return json.loads(path.read_text())
    return {"feature_columns": FEATURE_COLUMNS, "threshold": 0.5}


def decision_threshold() -> float:
    meta = load_meta()
    try:
        return float(meta.get("threshold", 0.5))
    except (TypeError, ValueError):
        return 0.5


def _enrich_from_lake(
    *,
    op_unique_carrier: str,
    origin: str,
    dest: str,
    crs_dep_hour: int,
    fl_dow: int,
) -> dict[str, Any]:
    """Fill congestion / reliability defaults (one light warehouse session)."""
    features = dict(DEFAULTS)
    features["is_weekend"] = 1 if fl_dow in (0, 6) else 0
    features["is_peak_hour"] = 1 if crs_dep_hour in range(6, 10) or crs_dep_hour in range(16, 21) else 0

    if not warehouse_available():
        return features

    origin_u, dest_u, carrier_u = origin.upper(), dest.upper(), op_unique_carrier.upper()
    with warehouse_connection(read_only=True, light=True) as con:
        route = _query_route_stats(con, origin_u, dest_u, carrier_u)
        origin_c = _query_airport_congestion(con, origin_u, crs_dep_hour)
        dest_c = _query_airport_congestion(con, dest_u, crs_dep_hour)
        carrier = _query_carrier_stats(con, carrier_u)

    if route.get("pct_delay_15") is not None:
        features["route_hist_pct_delay_15"] = float(route["pct_delay_15"])
    if route.get("avg_distance") is not None:
        features["distance"] = float(route["avg_distance"])
    if route.get("avg_crs_elapsed_time") is not None:
        features["crs_elapsed_time"] = float(route["avg_crs_elapsed_time"])

    if origin_c.get("avg_taxi_out") is not None:
        features["origin_hist_avg_taxi_out"] = float(origin_c["avg_taxi_out"])
    if origin_c.get("avg_nas_delay") is not None:
        features["origin_hist_avg_nas_delay"] = float(origin_c["avg_nas_delay"])
    if origin_c.get("avg_carrier_delay") is not None:
        features["origin_hist_avg_carrier_delay"] = float(origin_c["avg_carrier_delay"])
    if origin_c.get("avg_weather_delay") is not None:
        features["origin_hist_avg_weather_delay"] = float(origin_c["avg_weather_delay"])
    if origin_c.get("avg_late_aircraft_delay") is not None:
        features["origin_hist_avg_late_aircraft_delay"] = float(
            origin_c["avg_late_aircraft_delay"]
        )
    if origin_c.get("n_operations") is not None:
        features["origin_hist_hour_ops"] = float(origin_c["n_operations"])
        features["origin_hour_sched_flights"] = float(
            origin_c.get("n_departures") or origin_c["n_operations"]
        )
    if origin_c.get("pct_delay_15") is not None:
        features["origin_hist_hour_pct_delay_15"] = float(origin_c["pct_delay_15"])

    if dest_c.get("avg_taxi_in") is not None:
        features["dest_hist_avg_taxi_in"] = float(dest_c["avg_taxi_in"])
    if dest_c.get("n_operations") is not None:
        features["dest_hist_hour_ops"] = float(dest_c["n_operations"])
        features["dest_hour_sched_flights"] = float(
            dest_c.get("n_arrivals") or dest_c["n_operations"]
        )
    if dest_c.get("pct_delay_15") is not None:
        features["dest_hist_hour_pct_delay_15"] = float(dest_c["pct_delay_15"])

    if carrier.get("pct_delay_15") is not None:
        features["carrier_hist_pct_delay_15"] = float(carrier["pct_delay_15"])
    if carrier.get("avg_taxi_out") is not None:
        features["carrier_hist_avg_taxi_out"] = float(carrier["avg_taxi_out"])
    if carrier.get("avg_late_aircraft_delay") is not None:
        features["carrier_hist_avg_late_aircraft_delay"] = float(
            carrier["avg_late_aircraft_delay"]
        )

    return features


def predict_delay(
    *,
    op_unique_carrier: str,
    origin: str,
    dest: str,
    fl_month: int,
    fl_dow: int,
    crs_dep_hour: int,
    distance: float | None = None,
    crs_elapsed_time: float | None = None,
    origin_temp_c: float | None = None,
    origin_precip_mm: float | None = None,
    origin_wind_kmh: float | None = None,
    origin_weathercode: int | None = None,
    dest_temp_c: float | None = None,
    dest_precip_mm: float | None = None,
    dest_wind_kmh: float | None = None,
    dest_weathercode: int | None = None,
    route_hist_pct_delay_15: float | None = None,
    fetch_weather: bool = False,
) -> dict[str, Any]:
    enriched = _enrich_from_lake(
        op_unique_carrier=op_unique_carrier,
        origin=origin,
        dest=dest,
        crs_dep_hour=crs_dep_hour,
        fl_dow=fl_dow,
    )

    overrides = {
        "distance": distance,
        "crs_elapsed_time": crs_elapsed_time,
        "origin_temp_c": origin_temp_c,
        "origin_precip_mm": origin_precip_mm,
        "origin_wind_kmh": origin_wind_kmh,
        "origin_weathercode": origin_weathercode,
        "dest_temp_c": dest_temp_c,
        "dest_precip_mm": dest_precip_mm,
        "dest_wind_kmh": dest_wind_kmh,
        "dest_weathercode": dest_weathercode,
        "route_hist_pct_delay_15": route_hist_pct_delay_15,
    }
    for k, v in overrides.items():
        if v is not None:
            enriched[k] = v

    # Optional live weather from R2 (slow). Agent should call the weather tool instead.
    if fetch_weather:
        if origin_temp_c is None:
            wx = get_weather(origin)
            if wx.get("temperature_2m_mean") is not None:
                enriched["origin_temp_c"] = float(wx["temperature_2m_mean"])
            if wx.get("precipitation_sum") is not None:
                enriched["origin_precip_mm"] = float(wx["precipitation_sum"])
            if wx.get("windspeed_10m_max") is not None:
                enriched["origin_wind_kmh"] = float(wx["windspeed_10m_max"])
            if wx.get("weathercode") is not None:
                enriched["origin_weathercode"] = int(wx["weathercode"])
        if dest_temp_c is None:
            wxd = get_weather(dest)
            if wxd.get("temperature_2m_mean") is not None:
                enriched["dest_temp_c"] = float(wxd["temperature_2m_mean"])
            if wxd.get("precipitation_sum") is not None:
                enriched["dest_precip_mm"] = float(wxd["precipitation_sum"])
            if wxd.get("windspeed_10m_max") is not None:
                enriched["dest_wind_kmh"] = float(wxd["windspeed_10m_max"])
            if wxd.get("weathercode") is not None:
                enriched["dest_weathercode"] = int(wxd["weathercode"])

    row = pd.DataFrame(
        [
            {
                "op_unique_carrier": op_unique_carrier.upper(),
                "origin": origin.upper(),
                "dest": dest.upper(),
                "fl_month": fl_month,
                "fl_dow": fl_dow,
                "crs_dep_hour": crs_dep_hour,
                **{
                    c: enriched.get(c, DEFAULTS.get(c, 0))
                    for c in FEATURE_COLUMNS
                    if c
                    not in {
                        "op_unique_carrier",
                        "origin",
                        "dest",
                        "fl_month",
                        "fl_dow",
                        "crs_dep_hour",
                    }
                    and c
                    not in {
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
                    }
                },
            }
        ]
    )
    row = add_derived_features(row)
    cols = load_meta().get("feature_columns") or FEATURE_COLUMNS
    for c in cols:
        if c not in row.columns:
            row[c] = DEFAULTS.get(c, 0)
    row = row[list(cols)]

    model = load_model()
    try:
        proba = float(model.predict_proba(row)[0, 1])
    except ValueError:
        load_model.cache_clear()
        load_meta.cache_clear()
        model = load_model()
        proba = float(model.predict_proba(row)[0, 1])

    thr = decision_threshold()
    label = int(proba >= thr)
    drivers = {
        "origin_hist_avg_taxi_out": enriched["origin_hist_avg_taxi_out"],
        "origin_hist_avg_nas_delay": enriched["origin_hist_avg_nas_delay"],
        "origin_hour_sched_flights": enriched["origin_hour_sched_flights"],
        "origin_hist_hour_pct_delay_15": enriched["origin_hist_hour_pct_delay_15"],
        "route_hist_pct_delay_15": enriched["route_hist_pct_delay_15"],
        "carrier_hist_pct_delay_15": enriched["carrier_hist_pct_delay_15"],
        "is_peak_hour": enriched["is_peak_hour"],
    }
    return {
        "delay_probability": round(proba, 4),
        "predicted_delay_15": label,
        "threshold": round(thr, 4),
        "inputs": row.iloc[0].to_dict(),
        "congestion_drivers": drivers,
        "interpretation": (
            "Likely delayed ≥15 min" if label else "Likely on-time (<15 min late)"
        ),
    }


def _row_to_dict(row: pd.DataFrame) -> dict[str, Any]:
    if row.empty:
        return {}
    rec = row.iloc[0].to_dict()
    for k, v in list(rec.items()):
        if hasattr(v, "item"):
            rec[k] = v.item()
        elif hasattr(v, "isoformat"):
            rec[k] = v.isoformat()
    return rec


def _query_route_stats(
    con: Any,
    origin: str,
    dest: str,
    carrier: str | None,
) -> dict[str, Any]:
    if carrier:
        row = con.execute(
            """
            select *
            from flt_route_delay_stats
            where origin = ? and dest = ? and op_unique_carrier = ?
            """,
            [origin, dest, carrier],
        ).fetchdf()
    else:
        row = con.execute(
            """
            select origin, dest,
                   sum(n_flights) as n_flights,
                   sum(avg_arr_delay * n_flights) / nullif(sum(n_flights), 0) as avg_arr_delay,
                   sum(pct_delay_15 * n_flights) / nullif(sum(n_flights), 0) as pct_delay_15,
                   sum(avg_distance * n_flights) / nullif(sum(n_flights), 0) as avg_distance,
                   sum(avg_taxi_out * n_flights) / nullif(sum(n_flights), 0) as avg_taxi_out,
                   sum(avg_taxi_in * n_flights) / nullif(sum(n_flights), 0) as avg_taxi_in,
                   sum(avg_nas_delay * n_flights) / nullif(sum(n_flights), 0) as avg_nas_delay,
                   sum(avg_crs_elapsed_time * n_flights) / nullif(sum(n_flights), 0) as avg_crs_elapsed_time
            from flt_route_delay_stats
            where origin = ? and dest = ?
            group by 1, 2
            """,
            [origin, dest],
        ).fetchdf()
    if row.empty:
        return {
            "origin": origin,
            "dest": dest,
            "op_unique_carrier": carrier,
            "n_flights": 0,
            "avg_arr_delay": None,
            "pct_delay_15": None,
            "note": "No historical flights for this route in the lake.",
        }
    return _row_to_dict(row)


def _query_airport_congestion(con: Any, airport: str, hour: int) -> dict[str, Any]:
    row = con.execute(
        """
        select *
        from flt_airport_hour_stats
        where airport = ? and hour = ?
        """,
        [airport, hour],
    ).fetchdf()
    if row.empty:
        return {
            "airport": airport,
            "hour": hour,
            "note": "No congestion profile for this airport/hour.",
        }
    return _row_to_dict(row)


def _query_carrier_stats(con: Any, carrier: str) -> dict[str, Any]:
    row = con.execute(
        """
        select * from flt_carrier_delay_stats
        where op_unique_carrier = ?
        """,
        [carrier],
    ).fetchdf()
    if row.empty:
        return {"op_unique_carrier": carrier, "note": "No carrier stats."}
    return _row_to_dict(row)


def get_route_stats(
    origin: str,
    dest: str,
    carrier: str | None = None,
) -> dict[str, Any]:
    origin_u, dest_u = origin.upper(), dest.upper()
    carrier_u = carrier.upper() if carrier else None
    return _get_route_stats_cached(origin_u, dest_u, carrier_u)


@lru_cache(maxsize=512)
def _get_route_stats_cached(
    origin_u: str,
    dest_u: str,
    carrier_u: str | None,
) -> dict[str, Any]:
    if not warehouse_available():
        return {
            "origin": origin_u,
            "dest": dest_u,
            "op_unique_carrier": carrier_u,
            "n_flights": 0,
            "avg_arr_delay": None,
            "pct_delay_15": None,
            "note": "Warehouse not built; run `flight dbt build --from-r2` then `flight warehouse push`.",
        }
    with warehouse_connection(read_only=True, light=True) as con:
        return _query_route_stats(con, origin_u, dest_u, carrier_u)


def get_airport_congestion(airport: str, hour: int) -> dict[str, Any]:
    """Historical congestion profile for an airport at a clock hour."""
    return _get_airport_congestion_cached(airport.upper(), int(hour))


@lru_cache(maxsize=512)
def _get_airport_congestion_cached(airport: str, hour: int) -> dict[str, Any]:
    if not warehouse_available():
        return {"airport": airport, "hour": hour, "note": "Warehouse not built."}
    with warehouse_connection(read_only=True, light=True) as con:
        return _query_airport_congestion(con, airport, hour)


def get_carrier_stats(carrier: str) -> dict[str, Any]:
    return _get_carrier_stats_cached(carrier.upper())


@lru_cache(maxsize=128)
def _get_carrier_stats_cached(carrier: str) -> dict[str, Any]:
    if not warehouse_available():
        return {"op_unique_carrier": carrier, "note": "Warehouse not built."}
    with warehouse_connection(read_only=True, light=True) as con:
        return _query_carrier_stats(con, carrier)


def get_weather(airport: str, date: str | None = None) -> dict[str, Any]:
    """Weather for one airport — reads a single R2 object (not the full hive)."""
    from flight_agent.ingest.warehouse import configure_r2

    airport = airport.upper()
    settings = get_settings()
    db = Path(settings.duckdb_path)

    # Local DuckDB may have stg_weather from dbt.
    if db.exists():
        with warehouse_connection(read_only=True, light=True) as con:
            try:
                if date:
                    df = con.execute(
                        """
                        select * from stg_weather
                        where airport = ? and weather_date = cast(? as date)
                        """,
                        [airport, date],
                    ).fetchdf()
                else:
                    df = con.execute(
                        """
                        select * from stg_weather
                        where airport = ?
                        order by weather_date desc
                        limit 1
                        """,
                        [airport],
                    ).fetchdf()
            except Exception:  # noqa: BLE001
                df = pd.DataFrame()
        if not df.empty:
            return _row_to_dict(df)

    if not settings.r2_configured:
        return {"airport": airport, "note": "R2 not configured; no weather."}

    import duckdb

    uri = f"s3://{settings.r2_bucket}/raw/weather/airport={airport}/weather.parquet"
    con = duckdb.connect(":memory:")
    try:
        configure_r2(con)
        if date:
            df = con.execute(
                f"""
                select
                  cast(date as date) as weather_date,
                  upper(cast(airport as varchar)) as airport,
                  cast(temperature_2m_mean as double) as temperature_2m_mean,
                  cast(precipitation_sum as double) as precipitation_sum,
                  cast(windspeed_10m_max as double) as windspeed_10m_max,
                  cast(weathercode as integer) as weathercode
                from read_parquet('{uri}')
                where cast(date as date) = cast(? as date)
                limit 1
                """,
                [date],
            ).fetchdf()
        else:
            df = con.execute(
                f"""
                select
                  cast(date as date) as weather_date,
                  upper(cast(airport as varchar)) as airport,
                  cast(temperature_2m_mean as double) as temperature_2m_mean,
                  cast(precipitation_sum as double) as precipitation_sum,
                  cast(windspeed_10m_max as double) as windspeed_10m_max,
                  cast(weathercode as integer) as weathercode
                from read_parquet('{uri}')
                order by date desc
                limit 1
                """
            ).fetchdf()
    except Exception as exc:  # noqa: BLE001
        return {"airport": airport, "date": date, "note": f"No weather: {exc}"}
    finally:
        con.close()

    if df.empty:
        return {"airport": airport, "date": date, "note": "No weather rows found."}
    return _row_to_dict(df)


def load_metrics() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.model_dir) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
