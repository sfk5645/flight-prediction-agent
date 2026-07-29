"""Ingest daily weather from Open-Meteo for configured hubs."""

from __future__ import annotations

import httpx
import pandas as pd

from flight_agent.config import ensure_dirs, load_project_config
from flight_agent.ingest.airports import load_airports_frame
from flight_agent.ingest.schedule import YearMonth, resolve_ingest_window
from flight_agent.ingest.storage import read_parquet, write_parquet

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"


def _month_bounds(year: int, month: int) -> tuple[str, str]:
    import calendar

    last = calendar.monthrange(year, month)[1]
    return f"{year:04d}-{month:02d}-01", f"{year:04d}-{month:02d}-{last:02d}"


def _months_in_frame(df: pd.DataFrame) -> set[YearMonth]:
    if df.empty or "date" not in df.columns:
        return set()
    dates = pd.to_datetime(df["date"], errors="coerce")
    out: set[YearMonth] = set()
    for ts in dates.dropna():
        out.add(YearMonth(int(ts.year), int(ts.month)))
    return out


def _load_existing_weather(rel: str) -> pd.DataFrame | None:
    try:
        return read_parquet(rel)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read existing weather {rel}: {exc}")
        return None


def ingest_weather(
    use_sample: bool = False,
    *,
    to_r2: bool = False,
    keep_local: bool = True,
    incremental: bool = True,
    rolling: bool | None = None,
) -> list[str]:
    """
    Ingest daily weather for the resolved window.

    Incremental mode merges only missing months into existing airport files.
    """
    ensure_dirs()
    cfg = load_project_config()
    hubs = cfg["hubs"]
    window = resolve_ingest_window(cfg, rolling=rolling, use_sample=use_sample)
    print(f"Weather window: {window}")

    airports = load_airports_frame().set_index("airport")

    written: list[str] = []
    client: httpx.Client | None = None
    if not use_sample:
        client = httpx.Client(timeout=90.0, follow_redirects=True)

    try:
        for airport in hubs:
            if airport not in airports.index:
                print(f"Skipping weather for {airport}: missing coordinates")
                continue
            lat = float(airports.loc[airport, "lat"])
            lon = float(airports.loc[airport, "lon"])
            rel = f"weather/airport={airport}/weather.parquet"

            existing = _load_existing_weather(rel) if incremental else None
            have = _months_in_frame(existing) if existing is not None else set()
            targets = [m for m in window.months if m not in have] if incremental else list(window.months)

            if incremental and not targets and existing is not None:
                # Still trim to window + rewrite if needed for retention alignment
                existing = existing.copy()
                existing["_ym"] = pd.to_datetime(existing["date"], errors="coerce")
                lo = pd.Timestamp(year=window.start.year, month=window.start.month, day=1)
                hi = pd.Timestamp(year=window.end.year, month=window.end.month, day=1) + pd.offsets.MonthEnd(0)
                trimmed = existing[(existing["_ym"] >= lo) & (existing["_ym"] <= hi)].drop(columns=["_ym"])
                if len(trimmed) == len(existing):
                    print(f"Weather {airport}: already up to date")
                    continue
                loc = write_parquet(trimmed, rel, to_r2=to_r2, keep_local=keep_local)
                written.append(loc)
                continue

            frames: list[pd.DataFrame] = []
            if existing is not None and incremental:
                frames.append(existing)

            for ym in targets:
                year, month = ym.year, ym.month
                if use_sample:
                    frames.append(_synthetic_weather(airport, year, month, lat, lon))
                    continue
                start, end = _month_bounds(year, month)
                params = {
                    "latitude": lat,
                    "longitude": lon,
                    "start_date": start,
                    "end_date": end,
                    "daily": ",".join(cfg["weather"]["variables"]),
                    "timezone": cfg["weather"].get("timezone", "UTC"),
                }
                try:
                    assert client is not None
                    resp = client.get(OPEN_METEO_ARCHIVE, params=params)
                    resp.raise_for_status()
                    daily = resp.json().get("daily", {})
                    if not daily or "time" not in daily:
                        continue
                    frame = pd.DataFrame(daily)
                    frame = frame.rename(columns={"time": "date"})
                    frame["airport"] = airport
                    frame["lat"] = lat
                    frame["lon"] = lon
                    frames.append(frame)
                except Exception as exc:  # noqa: BLE001
                    print(f"Weather fetch failed for {airport} {year}-{month:02d}: {exc}")
                    frames.append(_synthetic_weather(airport, year, month, lat, lon))

            if not frames:
                continue
            merged = pd.concat(frames, ignore_index=True)
            merged["date"] = pd.to_datetime(merged["date"], errors="coerce")
            merged = merged.dropna(subset=["date"]).sort_values("date")
            merged = merged.drop_duplicates(subset=["date"], keep="last")
            lo = pd.Timestamp(year=window.start.year, month=window.start.month, day=1)
            hi = pd.Timestamp(year=window.end.year, month=window.end.month, day=1) + pd.offsets.MonthEnd(0)
            merged = merged[(merged["date"] >= lo) & (merged["date"] <= hi)]
            merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")
            loc = write_parquet(merged, rel, to_r2=to_r2, keep_local=keep_local)
            written.append(loc)
            print(f"Weather {airport}: wrote {len(merged):,} days (+{len(targets)} month(s)) → {loc}")
    finally:
        if client is not None:
            client.close()
    return written


def _synthetic_weather(
    airport: str, year: int, month: int, lat: float, lon: float
) -> pd.DataFrame:
    import calendar

    last = calendar.monthrange(year, month)[1]
    dates = [f"{year:04d}-{month:02d}-{d:02d}" for d in range(1, last + 1)]
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_mean": [15.0 + (month - 6) * 1.5] * n,
            "precipitation_sum": [0.5] * n,
            "windspeed_10m_max": [20.0] * n,
            "weathercode": [1] * n,
            "airport": airport,
            "lat": lat,
            "lon": lon,
        }
    )
