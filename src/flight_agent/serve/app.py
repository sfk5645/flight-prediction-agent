"""FastAPI app: health, predict, route-stats, weather, congestion."""

from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from flight_agent import __version__
from flight_agent.serve import services

app = FastAPI(
    title="Flight Delay Prediction API",
    description="Predict ≥15-minute arrival delays for US hub flights (weather + congestion).",
    version=__version__,
)


class PredictRequest(BaseModel):
    op_unique_carrier: str = Field(..., examples=["DL"])
    origin: str = Field(..., examples=["ATL"])
    dest: str = Field(..., examples=["LAX"])
    fl_month: int = Field(..., ge=1, le=12, examples=[7])
    fl_dow: int = Field(..., ge=0, le=6, description="0=Sunday in DuckDB extract(dow)", examples=[1])
    crs_dep_hour: int = Field(..., ge=0, le=23, examples=[8])
    distance: Optional[float] = None
    crs_elapsed_time: Optional[float] = None
    origin_temp_c: Optional[float] = None
    origin_precip_mm: Optional[float] = None
    origin_wind_kmh: Optional[float] = None
    origin_weathercode: Optional[int] = None
    dest_temp_c: Optional[float] = None
    dest_precip_mm: Optional[float] = None
    dest_wind_kmh: Optional[float] = None
    dest_weathercode: Optional[int] = None
    route_hist_pct_delay_15: Optional[float] = None


@app.get("/health")
def health() -> dict:
    model_ok = False
    try:
        services.load_model()
        model_ok = True
    except FileNotFoundError:
        model_ok = False
    return {"status": "ok", "version": __version__, "model_loaded": model_ok}


@app.post("/predict")
def predict(body: PredictRequest) -> dict:
    try:
        return services.predict_delay(**body.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/route-stats")
def route_stats(
    origin: str = Query(...),
    dest: str = Query(...),
    carrier: Optional[str] = Query(None),
) -> dict:
    return services.get_route_stats(origin, dest, carrier)


@app.get("/weather")
def weather(
    airport: str = Query(...),
    date: Optional[str] = Query(None, description="YYYY-MM-DD"),
) -> dict:
    return services.get_weather(airport, date)


@app.get("/congestion")
def congestion(
    airport: str = Query(..., description="IATA code"),
    hour: int = Query(..., ge=0, le=23, description="Local clock hour 0-23"),
) -> dict:
    """Historical taxi / NAS / volume congestion profile for airport×hour."""
    return services.get_airport_congestion(airport, hour)


@app.get("/carrier-stats")
def carrier_stats(carrier: str = Query(...)) -> dict:
    return services.get_carrier_stats(carrier)


@app.get("/metrics")
def metrics() -> dict:
    return services.load_metrics()
