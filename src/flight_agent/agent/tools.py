"""Tool functions for the ops agent (in-process, no HTTP required)."""

from __future__ import annotations

import json
from typing import Optional

from flight_agent.codes import normalize_airport, normalize_carrier
from flight_agent.serve import services


def tool_predict_delay(
    op_unique_carrier: str,
    origin: str,
    dest: str,
    fl_month: int,
    fl_dow: int,
    crs_dep_hour: int,
    distance: Optional[float] = None,
    origin_precip_mm: Optional[float] = None,
    origin_wind_kmh: Optional[float] = None,
) -> str:
    """Predict probability that a flight arrives 15+ minutes late (uses congestion + weather)."""
    try:
        result = services.predict_delay(
            op_unique_carrier=normalize_carrier(op_unique_carrier) or op_unique_carrier,
            origin=normalize_airport(origin) or origin,
            dest=normalize_airport(dest) or dest,
            fl_month=fl_month,
            fl_dow=fl_dow,
            crs_dep_hour=crs_dep_hour,
            distance=distance,
            origin_precip_mm=origin_precip_mm,
            origin_wind_kmh=origin_wind_kmh,
        )
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    return json.dumps(result)


def tool_route_stats(origin: str, dest: str, carrier: Optional[str] = None) -> str:
    """Historical delay + taxi/NAS stats for a route (optional carrier filter)."""
    try:
        return json.dumps(
            services.get_route_stats(
                normalize_airport(origin) or origin,
                normalize_airport(dest) or dest,
                normalize_carrier(carrier) if carrier else None,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


def tool_weather(
    airport: str,
    date: Optional[str] = None,
    hour: Optional[int] = None,
) -> str:
    """Hourly weather for an airport; date YYYY-MM-DD/today; optional hour 0-23."""
    try:
        return json.dumps(
            services.get_weather(
                normalize_airport(airport) or airport,
                date,
                hour=hour,
            )
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps(
            {"error": str(exc), "airport": airport, "date": date, "hour": hour}
        )


def tool_airport_congestion(airport: str, hour: int) -> str:
    """Historical congestion for airport at hour: taxi, NAS delay, ops volume, delay rate."""
    try:
        return json.dumps(
            services.get_airport_congestion(
                normalize_airport(airport) or airport, hour
            )
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})


def tool_model_metrics() -> str:
    """Return offline evaluation metrics for the delay model."""
    try:
        return json.dumps(
            services.load_metrics() or {"note": "No metrics yet; run flight train."}
        )
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
