"""Ingest BTS On-Time Performance data (download or synthetic sample)."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd

from flight_agent.config import ensure_dirs, load_project_config
from flight_agent.ingest.schedule import IngestWindow, YearMonth
from flight_agent.ingest.storage import write_parquet

# Public pre-zipped monthly On-Time files from BTS TranStats
BTS_ZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

KEEP_COLUMNS = {
    "FlightDate": "fl_date",
    "FL_DATE": "fl_date",
    "Reporting_Airline": "op_unique_carrier",
    "OP_UNIQUE_CARRIER": "op_unique_carrier",
    "Flight_Number_Reporting_Airline": "op_carrier_fl_num",
    "OP_CARRIER_FL_NUM": "op_carrier_fl_num",
    "Origin": "origin",
    "ORIGIN": "origin",
    "Dest": "dest",
    "DEST": "dest",
    "CRSDepTime": "crs_dep_time",
    "CRS_DEP_TIME": "crs_dep_time",
    "CRSArrTime": "crs_arr_time",
    "CRS_ARR_TIME": "crs_arr_time",
    "DepDelay": "dep_delay",
    "DEP_DELAY": "dep_delay",
    "ArrDelay": "arr_delay",
    "ARR_DELAY": "arr_delay",
    "Cancelled": "cancelled",
    "CANCELLED": "cancelled",
    "Diverted": "diverted",
    "DIVERTED": "diverted",
    "Distance": "distance",
    "DISTANCE": "distance",
    # Congestion / ops timing
    "TaxiOut": "taxi_out",
    "TAXI_OUT": "taxi_out",
    "TaxiIn": "taxi_in",
    "TAXI_IN": "taxi_in",
    "CRSElapsedTime": "crs_elapsed_time",
    "CRS_ELAPSED_TIME": "crs_elapsed_time",
    "ActualElapsedTime": "actual_elapsed_time",
    "ACTUAL_ELAPSED_TIME": "actual_elapsed_time",
    "AirTime": "air_time",
    "AIR_TIME": "air_time",
    # Delay cause minutes (used for historical airport profiles, not same-flight labels)
    "CarrierDelay": "carrier_delay",
    "CARRIER_DELAY": "carrier_delay",
    "WeatherDelay": "weather_delay",
    "WEATHER_DELAY": "weather_delay",
    "NASDelay": "nas_delay",
    "NAS_DELAY": "nas_delay",
    "SecurityDelay": "security_delay",
    "SECURITY_DELAY": "security_delay",
    "LateAircraftDelay": "late_aircraft_delay",
    "LATE_AIRCRAFT_DELAY": "late_aircraft_delay",
}

CANONICAL = [
    "fl_date",
    "op_unique_carrier",
    "op_carrier_fl_num",
    "origin",
    "dest",
    "crs_dep_time",
    "crs_arr_time",
    "dep_delay",
    "arr_delay",
    "cancelled",
    "diverted",
    "distance",
    "taxi_out",
    "taxi_in",
    "crs_elapsed_time",
    "actual_elapsed_time",
    "air_time",
    "carrier_delay",
    "weather_delay",
    "nas_delay",
    "security_delay",
    "late_aircraft_delay",
]


@dataclass
class BtsIngestResult:
    window: IngestWindow
    written: list[str]
    months_written: list[YearMonth]

    def __bool__(self) -> bool:
        return bool(self.written)


def ingest_bts(
    use_sample: bool = False,
    *,
    to_r2: bool = False,
    keep_local: bool = True,
    incremental: bool = True,
    rolling: bool | None = None,
) -> BtsIngestResult:
    """
    Ingest BTS On-Time partitions for the resolved window.

    By default uses rolling mode (latest published month + retention window)
    and skips months already present locally and/or on R2.
    """
    from flight_agent.ingest.schedule import months_to_ingest, resolve_ingest_window

    ensure_dirs()
    cfg = load_project_config()
    hubs = set(cfg["hubs"])
    window = resolve_ingest_window(cfg, rolling=rolling, use_sample=use_sample)
    # When writing to R2, skip based on R2 presence; otherwise use local lake.
    targets = months_to_ingest(
        window,
        incremental=incremental,
        check_local=not to_r2,
        check_r2=to_r2,
    )

    print(
        f"BTS window: {window}"
        + (f"; ingesting {len(targets)} missing month(s)" if incremental else "")
    )
    if incremental and not targets:
        print("BTS: lake already up to date — nothing to download.")
        return BtsIngestResult(window=window, written=[], months_written=[])

    written: list[str] = []
    months_written: list[YearMonth] = []
    for ym in targets:
        year, month = ym.year, ym.month
        rel = f"bts/year={year}/month={month:02d}/flights.parquet"

        if use_sample:
            df = _synthetic_month(year, month, hubs, cfg["sample"]["rows_per_month"])
        else:
            df = _download_month(year, month, hubs)
            if df is None:
                print(f"BTS download failed for {year}-{month:02d}; using synthetic sample.")
                df = _synthetic_month(year, month, hubs, cfg["sample"]["rows_per_month"])

        loc = write_parquet(df, rel, to_r2=to_r2, keep_local=keep_local)
        written.append(loc)
        months_written.append(ym)
        print(f"Wrote {len(df):,} rows → {loc}")
    return BtsIngestResult(window=window, written=written, months_written=months_written)


def _download_month(year: int, month: int, hubs: set[str]) -> pd.DataFrame | None:
    url = BTS_ZIP_URL.format(year=year, month=month)
    try:
        with httpx.Client(timeout=180.0, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code != 200:
                print(f"HTTP {resp.status_code} for {url}")
                return None
            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
                with zf.open(csv_name) as fh:
                    df = pd.read_csv(fh, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        print(f"Download error {year}-{month:02d}: {exc}")
        return None

    rename = {c: KEEP_COLUMNS[c] for c in df.columns if c in KEEP_COLUMNS}
    df = df.rename(columns=rename)
    missing = [c for c in CANONICAL if c not in df.columns]
    for col in missing:
        df[col] = np.nan
    df = df[CANONICAL].copy()
    df["origin"] = df["origin"].astype(str).str.upper()
    df["dest"] = df["dest"].astype(str).str.upper()
    df = df[df["origin"].isin(hubs) | df["dest"].isin(hubs)].copy()
    df["fl_date"] = pd.to_datetime(df["fl_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def _synthetic_month(
    year: int, month: int, hubs: set[str], rows: int
) -> pd.DataFrame:
    rng = np.random.default_rng(year * 100 + month)
    hubs_list = sorted(hubs)
    carriers = ["DL", "AA", "UA", "B6", "WN", "AS"]
    import calendar

    last = calendar.monthrange(year, month)[1]
    days = rng.integers(1, last + 1, size=rows)
    origins = rng.choice(hubs_list, size=rows)
    dests = rng.choice(hubs_list, size=rows)
    for i in range(rows):
        if origins[i] == dests[i]:
            dests[i] = hubs_list[(hubs_list.index(origins[i]) + 1) % len(hubs_list)]

    # Delay distribution: mostly on-time, fat tail late; busier hubs slightly worse
    busy = np.isin(origins, ["ATL", "ORD", "DFW", "LAX"])
    base = rng.normal(loc=5, scale=18, size=rows)
    base[busy] += rng.uniform(0, 12, size=busy.sum())
    late_boost = rng.random(rows) < 0.22
    base[late_boost] += rng.uniform(20, 120, size=late_boost.sum())
    if month in (12, 1, 2):
        base += rng.uniform(0, 15, size=rows)

    crs_dep = rng.integers(500, 2300, size=rows)
    crs_dep = (crs_dep // 100) * 100 + (crs_dep % 100) % 60
    distance = rng.integers(200, 2500, size=rows).astype(float)
    crs_elapsed = np.clip(distance / 7.5 + rng.normal(30, 10, size=rows), 45, 420)

    taxi_out = np.clip(rng.normal(16, 8, size=rows) + busy * 6, 5, 90)
    taxi_in = np.clip(rng.normal(8, 4, size=rows), 2, 45)
    nas = np.clip(np.where(late_boost, rng.uniform(0, 40, size=rows), rng.uniform(0, 5, size=rows)), 0, None)
    carrier = np.clip(rng.uniform(0, 15, size=rows) * late_boost, 0, None)
    wx_delay = np.clip(rng.uniform(0, 20, size=rows) * (month in (12, 1, 2)), 0, None)
    late_ac = np.clip(rng.uniform(0, 25, size=rows) * late_boost, 0, None)

    return pd.DataFrame(
        {
            "fl_date": [f"{year:04d}-{month:02d}-{d:02d}" for d in days],
            "op_unique_carrier": rng.choice(carriers, size=rows),
            "op_carrier_fl_num": rng.integers(1, 3000, size=rows),
            "origin": origins,
            "dest": dests,
            "crs_dep_time": crs_dep,
            "crs_arr_time": (crs_dep + (crs_elapsed.astype(int) // 60) * 100 + (crs_elapsed.astype(int) % 60))
            % 2400,
            "dep_delay": np.round(base - 3, 1),
            "arr_delay": np.round(base, 1),
            "cancelled": 0,
            "diverted": 0,
            "distance": distance,
            "taxi_out": np.round(taxi_out, 1),
            "taxi_in": np.round(taxi_in, 1),
            "crs_elapsed_time": np.round(crs_elapsed, 1),
            "actual_elapsed_time": np.round(crs_elapsed + base * 0.3, 1),
            "air_time": np.round(np.clip(crs_elapsed - taxi_out - taxi_in, 20, None), 1),
            "carrier_delay": np.round(carrier, 1),
            "weather_delay": np.round(wx_delay, 1),
            "nas_delay": np.round(nas, 1),
            "security_delay": 0.0,
            "late_aircraft_delay": np.round(late_ac, 1),
        }
    )
