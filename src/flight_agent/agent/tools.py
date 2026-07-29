"""Tool functions for the ops agent (in-process, no HTTP required)."""

from __future__ import annotations

import json
from typing import Optional

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
            op_unique_carrier=op_unique_carrier,
            origin=origin,
            dest=dest,
            fl_month=fl_month,
            fl_dow=fl_dow,
            crs_dep_hour=crs_dep_hour,
            distance=distance,
            origin_precip_mm=origin_precip_mm,
            origin_wind_kmh=origin_wind_kmh,
        )
    except FileNotFoundError as exc:
        return json.dumps({"error": str(exc)})
    return json.dumps(result)


def tool_route_stats(origin: str, dest: str, carrier: Optional[str] = None) -> str:
    """Historical delay + taxi/NAS stats for a route (optional carrier filter)."""
    return json.dumps(services.get_route_stats(origin, dest, carrier))


def tool_weather(airport: str, date: Optional[str] = None) -> str:
    """Weather features for an airport IATA code, optional date YYYY-MM-DD."""
    return json.dumps(services.get_weather(airport, date))


def tool_airport_congestion(airport: str, hour: int) -> str:
    """Historical congestion for airport at hour: taxi, NAS delay, ops volume, delay rate."""
    return json.dumps(services.get_airport_congestion(airport, hour))


def tool_model_metrics() -> str:
    """Return offline evaluation metrics for the delay model."""
    return json.dumps(services.load_metrics() or {"note": "No metrics yet; run flight train."})
