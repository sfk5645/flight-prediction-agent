"""Shared prediction / stats services used by API and agent tools."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from flight_agent.codes import (
    normalize_airport,
    normalize_carrier,
    parse_weather_when,
)
from flight_agent.config import get_settings
from flight_agent.features.build import DEFAULTS, FEATURE_COLUMNS, add_derived_features
from flight_agent.ingest.warehouse import warehouse_available, warehouse_connection


@lru_cache
def load_model():
    settings = get_settings()
    path = Path(settings.model_dir) / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Run `flight train` first."
        )
    return joblib.load(path)


@lru_cache
def load_regressor():
    """Optional arrival-delay minutes regressor (may be missing on older artifacts)."""
    settings = get_settings()
    path = Path(settings.model_dir) / "model_regressor.joblib"
    if not path.exists():
        return None
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
    """Fill congestion / reliability defaults from warehouse lookups."""
    features = dict(DEFAULTS)
    features["is_weekend"] = 1 if fl_dow in (0, 6) else 0
    features["is_peak_hour"] = 1 if crs_dep_hour in range(6, 10) or crs_dep_hour in range(16, 21) else 0

    route = get_route_stats(origin, dest, op_unique_carrier)
    if route.get("pct_delay_15") is not None:
        features["route_hist_pct_delay_15"] = float(route["pct_delay_15"])
    if route.get("avg_distance") is not None:
        features["distance"] = float(route["avg_distance"])
    if route.get("avg_crs_elapsed_time") is not None:
        features["crs_elapsed_time"] = float(route["avg_crs_elapsed_time"])

    origin_c = get_airport_congestion(origin, crs_dep_hour)
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
        features["origin_hour_sched_flights"] = float(origin_c.get("n_departures") or origin_c["n_operations"])
    if origin_c.get("pct_delay_15") is not None:
        features["origin_hist_hour_pct_delay_15"] = float(origin_c["pct_delay_15"])

    # Dest: use same clock hour as a proxy when arrival hour unknown
    dest_c = get_airport_congestion(dest, crs_dep_hour)
    if dest_c.get("avg_taxi_in") is not None:
        features["dest_hist_avg_taxi_in"] = float(dest_c["avg_taxi_in"])
    if dest_c.get("n_operations") is not None:
        features["dest_hist_hour_ops"] = float(dest_c["n_operations"])
        features["dest_hour_sched_flights"] = float(dest_c.get("n_arrivals") or dest_c["n_operations"])
    if dest_c.get("pct_delay_15") is not None:
        features["dest_hist_hour_pct_delay_15"] = float(dest_c["pct_delay_15"])

    carrier = get_carrier_stats(op_unique_carrier)
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
) -> dict[str, Any]:
    op_unique_carrier = normalize_carrier(op_unique_carrier) or op_unique_carrier.upper()
    origin = normalize_airport(origin) or origin.upper()
    dest = normalize_airport(dest) or dest.upper()
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

    # Pull hourly weather from lake when not provided (matched to CRS dep hour)
    weather_as_of: dict[str, Any] = {}
    if origin_temp_c is None:
        wx = get_weather(origin, hour=crs_dep_hour)
        if wx.get("temperature_2m_mean") is not None:
            enriched["origin_temp_c"] = float(wx["temperature_2m_mean"])
        if wx.get("precipitation_sum") is not None:
            enriched["origin_precip_mm"] = float(wx["precipitation_sum"])
        if wx.get("windspeed_10m_max") is not None:
            enriched["origin_wind_kmh"] = float(wx["windspeed_10m_max"])
        if wx.get("weathercode") is not None:
            enriched["origin_weathercode"] = int(wx["weathercode"])
        if wx.get("weather_date") is not None:
            weather_as_of["origin_weather_date"] = wx["weather_date"]
        if wx.get("weather_hour") is not None:
            weather_as_of["origin_weather_hour"] = wx["weather_hour"]
    if dest_temp_c is None:
        # Dest arrival hour unknown at predict time — use same clock hour as dep
        # unless caller overrides; keeps hour-specific conditions in the model.
        wxd = get_weather(dest, hour=crs_dep_hour)
        if wxd.get("temperature_2m_mean") is not None:
            enriched["dest_temp_c"] = float(wxd["temperature_2m_mean"])
        if wxd.get("precipitation_sum") is not None:
            enriched["dest_precip_mm"] = float(wxd["precipitation_sum"])
        if wxd.get("windspeed_10m_max") is not None:
            enriched["dest_wind_kmh"] = float(wxd["windspeed_10m_max"])
        if wxd.get("weathercode") is not None:
            enriched["dest_weathercode"] = int(wxd["weathercode"])
        if wxd.get("weather_date") is not None:
            weather_as_of["dest_weather_date"] = wxd["weather_date"]
        if wxd.get("weather_hour") is not None:
            weather_as_of["dest_weather_hour"] = wxd["weather_hour"]

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
        load_regressor.cache_clear()
        model = load_model()
        proba = float(model.predict_proba(row)[0, 1])

    thr = decision_threshold()
    label = int(proba >= thr)

    predicted_minutes: float | None = None
    regressor = load_regressor()
    if regressor is not None:
        try:
            minutes = float(regressor.predict(row)[0])
        except ValueError:
            load_regressor.cache_clear()
            regressor = load_regressor()
            minutes = float(regressor.predict(row)[0]) if regressor is not None else None
        if minutes is not None:
            meta = load_meta()
            lo = float(meta.get("regression_clip_min", -30) or -30)
            hi = float(meta.get("regression_clip_max", 240) or 240)
            predicted_minutes = float(np.clip(minutes, lo, hi))

    drivers = {
        "origin_hist_avg_taxi_out": enriched["origin_hist_avg_taxi_out"],
        "origin_hist_avg_nas_delay": enriched["origin_hist_avg_nas_delay"],
        "origin_hour_sched_flights": enriched["origin_hour_sched_flights"],
        "origin_hist_hour_pct_delay_15": enriched["origin_hist_hour_pct_delay_15"],
        "route_hist_pct_delay_15": enriched["route_hist_pct_delay_15"],
        "carrier_hist_pct_delay_15": enriched["carrier_hist_pct_delay_15"],
        "is_peak_hour": enriched["is_peak_hour"],
        "origin_precip_mm": enriched.get("origin_precip_mm"),
        "origin_wind_kmh": enriched.get("origin_wind_kmh"),
    }
    if predicted_minutes is not None:
        risk = (
            f"≥15 min late risk {proba:.0%}"
            if label
            else f"on-time risk favored ({proba:.0%} ≥15 min)"
        )
        if predicted_minutes >= 15:
            interpretation = (
                f"{risk}; expected arrival delay ≈ {predicted_minutes:.0f} minutes"
            )
        elif predicted_minutes >= 5:
            interpretation = (
                f"{risk}; expected arrival ≈ {predicted_minutes:.0f} minutes late"
            )
        elif predicted_minutes >= -5:
            interpretation = (
                f"{risk}; expected arrival near schedule ({predicted_minutes:+.0f} min)"
            )
        else:
            interpretation = (
                f"{risk}; expected arrival ≈ {predicted_minutes:.0f} minutes vs schedule"
            )
    else:
        interpretation = (
            "Likely delayed ≥15 min" if label else "Likely on-time (<15 min late)"
        )

    out: dict[str, Any] = {
        "delay_probability": round(proba, 4),
        "predicted_delay_15": label,
        "threshold": round(thr, 4),
        "inputs": row.iloc[0].to_dict(),
        "congestion_drivers": drivers,
        "interpretation": interpretation,
    }
    if predicted_minutes is not None:
        out["predicted_arr_delay_minutes"] = round(predicted_minutes, 1)
    if weather_as_of:
        out["weather_as_of"] = weather_as_of
    return out


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


def get_route_stats(
    origin: str,
    dest: str,
    carrier: str | None = None,
) -> dict[str, Any]:
    origin_c = normalize_airport(origin) or origin.upper()
    dest_c = normalize_airport(dest) or dest.upper()
    carrier_c = normalize_carrier(carrier) if carrier else None
    if not warehouse_available():
        return {
            "origin": origin_c,
            "dest": dest_c,
            "op_unique_carrier": carrier_c,
            "n_flights": 0,
            "avg_arr_delay": None,
            "pct_delay_15": None,
            "note": "Warehouse not built; run `flight dbt build --from-r2` then `flight warehouse push`.",
        }
    with warehouse_connection(read_only=True) as con:
        if carrier_c:
            row = con.execute(
                """
                select *
                from flt_route_delay_stats
                where origin = ? and dest = ? and op_unique_carrier = ?
                """,
                [origin_c, dest_c, carrier_c],
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
                [origin_c, dest_c],
            ).fetchdf()

    if row.empty:
        return {
            "origin": origin_c,
            "dest": dest_c,
            "op_unique_carrier": carrier_c,
            "n_flights": 0,
            "avg_arr_delay": None,
            "pct_delay_15": None,
            "note": "No historical flights for this route in the lake.",
        }
    return _row_to_dict(row)


def get_airport_congestion(airport: str, hour: int) -> dict[str, Any]:
    """Historical congestion profile for an airport at a clock hour."""
    airport_c = normalize_airport(airport) or airport.upper()
    hour_i = int(hour) % 24
    if not warehouse_available():
        return {"airport": airport_c, "hour": hour_i, "note": "Warehouse not built."}
    with warehouse_connection(read_only=True) as con:
        row = con.execute(
            """
            select *
            from flt_airport_hour_stats
            where airport = ? and hour = ?
            """,
            [airport_c, hour_i],
        ).fetchdf()
    if row.empty:
        return {
            "airport": airport_c,
            "hour": hour_i,
            "note": "No congestion profile for this airport/hour.",
        }
    return _row_to_dict(row)


def get_carrier_stats(carrier: str) -> dict[str, Any]:
    carrier_c = normalize_carrier(carrier) or carrier.upper()
    if not warehouse_available():
        return {"op_unique_carrier": carrier_c, "note": "Warehouse not built."}
    with warehouse_connection(read_only=True) as con:
        row = con.execute(
            """
            select * from flt_carrier_delay_stats
            where op_unique_carrier = ?
            """,
            [carrier_c],
        ).fetchdf()
    if row.empty:
        return {"op_unique_carrier": carrier_c, "note": "No carrier stats."}
    return _row_to_dict(row)


def _annotate_weather_units(result: dict[str, Any]) -> dict[str, Any]:
    """
    Make Open-Meteo units explicit for the LLM / API consumers.

    Lake stores °C and km/h; US users often assume °F without labels.
    """
    out = dict(result)
    out["units"] = {
        "temperature": "celsius",
        "precipitation": "mm",
        "wind": "km/h",
    }
    temp_c = out.get("temperature_2m_mean")
    if temp_c is not None:
        try:
            c = float(temp_c)
            out["temperature_c"] = round(c, 1)
            out["temperature_f"] = round(c * 9.0 / 5.0 + 32.0, 1)
        except (TypeError, ValueError):
            pass
    wind = out.get("windspeed_10m_max")
    if wind is not None:
        try:
            out["wind_kmh"] = round(float(wind), 1)
        except (TypeError, ValueError):
            pass
    precip = out.get("precipitation_sum")
    if precip is not None:
        try:
            out["precipitation_mm"] = round(float(precip), 2)
        except (TypeError, ValueError):
            pass
    out["summary"] = (
        f"{out.get('temperature_c', temp_c)}°C "
        f"({out.get('temperature_f', '?')}°F), "
        f"precip {out.get('precipitation_mm', precip)} mm, "
        f"wind {out.get('wind_kmh', wind)} km/h"
        if temp_c is not None
        else None
    )
    return out


def get_weather(
    airport: str,
    date: str | None = None,
    hour: int | None = None,
) -> dict[str, Any]:
    """
    Hourly weather for an airport.

    ``date`` may be free text (today / YYYY-MM-DD / today at 10pm).
    ``hour`` is 0–23; when omitted, uses hour parsed from ``date`` or the
    latest available hour for that airport/day.
    """
    airport_c = normalize_airport(airport) or airport.upper()
    iso_date, parsed_hour, date_note = parse_weather_when(date)
    hour_i = hour if hour is not None else parsed_hour
    if hour_i is not None:
        hour_i = int(hour_i) % 24

    if not warehouse_available():
        return {"airport": airport_c, "note": "Warehouse not built."}
    try:
        with warehouse_connection(read_only=True) as con:
            df = None
            if iso_date is not None and hour_i is not None:
                df = con.execute(
                    """
                    select * from stg_weather
                    where airport = ?
                      and weather_date = cast(? as date)
                      and weather_hour = ?
                    """,
                    [airport_c, iso_date, hour_i],
                ).fetchdf()
            elif iso_date is not None:
                df = con.execute(
                    """
                    select * from stg_weather
                    where airport = ? and weather_date = cast(? as date)
                    order by weather_hour desc
                    limit 1
                    """,
                    [airport_c, iso_date],
                ).fetchdf()
            elif hour_i is not None:
                df = con.execute(
                    """
                    select * from stg_weather
                    where airport = ? and weather_hour = ?
                    order by weather_date desc
                    limit 1
                    """,
                    [airport_c, hour_i],
                ).fetchdf()

            if df is None or df.empty:
                latest = con.execute(
                    """
                    select * from stg_weather
                    where airport = ?
                    order by weather_date desc, weather_hour desc
                    limit 1
                    """,
                    [airport_c],
                ).fetchdf()
                if not latest.empty and (iso_date is not None or hour_i is not None):
                    date_note = "; ".join(
                        p
                        for p in [
                            date_note,
                            "Exact hour/date miss; showing latest lake weather hour.",
                        ]
                        if p
                    )
                df = latest
    except Exception as exc:  # noqa: BLE001
        return {
            "airport": airport_c,
            "date": iso_date or date,
            "hour": hour_i,
            "note": f"Weather lookup failed: {exc}",
        }
    if df is None or df.empty:
        out = {
            "airport": airport_c,
            "date": iso_date or date,
            "hour": hour_i,
            "note": "No hourly weather rows found for that airport "
            "(lake weather is hourly and may lag the calendar).",
        }
        if date_note:
            out["date_note"] = date_note
        return out
    result = _row_to_dict(df)
    if date_note:
        result["date_note"] = date_note
    return _annotate_weather_units(result)


def load_metrics() -> dict[str, Any]:
    settings = get_settings()
    path = Path(settings.model_dir) / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text())
