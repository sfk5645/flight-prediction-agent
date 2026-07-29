"""Download OurAirports metadata and filter to configured hubs."""

from __future__ import annotations

from io import BytesIO

import httpx
import pandas as pd

from flight_agent.config import ensure_dirs, load_project_config
from flight_agent.ingest.storage import read_parquet, write_parquet

OURAIRPORTS_URL = "https://davidmegginson.github.io/ourairports-data/airports.csv"


def ingest_airports(*, to_r2: bool = False, keep_local: bool = True) -> str:
    ensure_dirs()
    cfg = load_project_config()
    hubs = set(cfg["hubs"])

    try:
        with httpx.Client(timeout=60.0, follow_redirects=True) as client:
            resp = client.get(OURAIRPORTS_URL)
            resp.raise_for_status()
            df = pd.read_csv(BytesIO(resp.content))
    except Exception as exc:  # noqa: BLE001
        print(f"OurAirports download failed ({exc}); using embedded hub metadata.")
        df = pd.DataFrame(_EMBEDDED_HUBS)

    if "iata_code" in df.columns:
        df = df[df["iata_code"].isin(hubs)].copy()
        df = df.rename(
            columns={
                "iata_code": "airport",
                "latitude_deg": "lat",
                "longitude_deg": "lon",
                "municipality": "city",
                "iso_country": "country",
                "name": "airport_name",
            }
        )
        keep = [
            c
            for c in ["airport", "airport_name", "lat", "lon", "city", "country", "type"]
            if c in df.columns
        ]
        df = df[keep].drop_duplicates(subset=["airport"])
    else:
        df = df[df["airport"].isin(hubs)].copy()

    return write_parquet(
        df,
        "airports/airports.parquet",
        to_r2=to_r2,
        keep_local=keep_local,
    )


def load_airports_frame() -> pd.DataFrame:
    """Load airports from local lake or R2."""
    try:
        return read_parquet("airports/airports.parquet")
    except FileNotFoundError:
        ingest_airports(keep_local=True)
        return read_parquet("airports/airports.parquet")


_EMBEDDED_HUBS = [
    {"airport": "LAX", "airport_name": "Los Angeles International Airport", "lat": 33.9425, "lon": -118.4081, "city": "Los Angeles", "country": "US", "type": "large_airport"},
    {"airport": "JFK", "airport_name": "John F Kennedy International Airport", "lat": 40.6413, "lon": -73.7781, "city": "New York", "country": "US", "type": "large_airport"},
    {"airport": "ORD", "airport_name": "Chicago O'Hare International Airport", "lat": 41.9742, "lon": -87.9073, "city": "Chicago", "country": "US", "type": "large_airport"},
    {"airport": "DEN", "airport_name": "Denver International Airport", "lat": 39.8561, "lon": -104.6737, "city": "Denver", "country": "US", "type": "large_airport"},
    {"airport": "ATL", "airport_name": "Hartsfield-Jackson Atlanta International Airport", "lat": 33.6407, "lon": -84.4277, "city": "Atlanta", "country": "US", "type": "large_airport"},
    {"airport": "IAD", "airport_name": "Washington Dulles International Airport", "lat": 38.9531, "lon": -77.4565, "city": "Washington", "country": "US", "type": "large_airport"},
    {"airport": "DFW", "airport_name": "Dallas Fort Worth International Airport", "lat": 32.8998, "lon": -97.0403, "city": "Dallas-Fort Worth", "country": "US", "type": "large_airport"},
]
