"""Ingest daily weather from Open-Meteo for configured hubs."""

from __future__ import annotations

import time
from datetime import date, timedelta

import httpx
import pandas as pd

from flight_agent.config import ensure_dirs, get_settings, load_project_config
from flight_agent.ingest.airports import load_airports_frame
from flight_agent.ingest.schedule import resolve_weather_window
from flight_agent.ingest.storage import read_parquet, write_parquet

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
_USER_AGENT = "flight-prediction-agent/1.0 (github.com/sfk5645/flight-prediction-agent)"
# Free forecast API allows up to 92 past days — used for daily freshness.
_FORECAST_PAST_DAYS_MAX = 92


def _load_existing_weather(rel: str, *, prefer_r2: bool) -> pd.DataFrame | None:
    """Load prior weather; when writing to R2 prefer the cloud object over stale local."""
    settings = get_settings()
    relative = rel.lstrip("/").replace("\\", "/")
    if relative.startswith("raw/"):
        relative = relative[4:]
    key = f"raw/{relative}"

    if prefer_r2 and settings.r2_configured:
        try:
            import io

            from flight_agent.ingest.r2_sync import _client

            obj = _client().get_object(Bucket=settings.r2_bucket, Key=key)
            return pd.read_parquet(io.BytesIO(obj["Body"].read()))
        except Exception as exc:  # noqa: BLE001
            print(f"R2 weather miss for {key}: {exc}")

    try:
        return read_parquet(rel)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        print(f"Could not read existing weather {rel}: {exc}")
        return None


def _http_client() -> httpx.Client:
    return httpx.Client(
        timeout=90.0,
        follow_redirects=True,
        headers={"User-Agent": _USER_AGENT},
    )


def _frame_from_daily(
    daily: dict, *, airport: str, lat: float, lon: float
) -> pd.DataFrame:
    if not daily or "time" not in daily:
        return pd.DataFrame()
    frame = pd.DataFrame(daily).rename(columns={"time": "date"})
    frame["airport"] = airport
    frame["lat"] = lat
    frame["lon"] = lon
    return frame


def _get_json(client: httpx.Client, url: str, params: dict) -> dict:
    resp = client.get(url, params=params)
    if resp.status_code in (429, 403):
        time.sleep(2.0)
        resp = client.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


def _fetch_archive_range(
    client: httpx.Client,
    *,
    lat: float,
    lon: float,
    start: date,
    end: date,
    variables: list[str],
    timezone: str,
    airport: str,
) -> pd.DataFrame:
    if end < start:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    cursor = start
    while cursor <= end:
        chunk_end = min(end, date(cursor.year, 12, 31))
        payload = _get_json(
            client,
            OPEN_METEO_ARCHIVE,
            {
                "latitude": lat,
                "longitude": lon,
                "start_date": cursor.isoformat(),
                "end_date": chunk_end.isoformat(),
                "daily": ",".join(variables),
                "timezone": timezone,
            },
        )
        frame = _frame_from_daily(
            payload.get("daily", {}), airport=airport, lat=lat, lon=lon
        )
        if not frame.empty:
            frames.append(frame)
        time.sleep(0.3)
        cursor = chunk_end + timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fetch_forecast_recent(
    client: httpx.Client,
    *,
    lat: float,
    lon: float,
    start: date,
    end: date,
    variables: list[str],
    timezone: str,
    airport: str,
    as_of: date,
) -> pd.DataFrame:
    """Pull recent daily weather via forecast past_days (better for 'through today')."""
    if end < start:
        return pd.DataFrame()
    past_days = min(_FORECAST_PAST_DAYS_MAX, (as_of - start).days + 1)
    past_days = max(past_days, 1)
    payload = _get_json(
        client,
        OPEN_METEO_FORECAST,
        {
            "latitude": lat,
            "longitude": lon,
            "daily": ",".join(variables),
            "timezone": timezone,
            "past_days": past_days,
            "forecast_days": 1,
        },
    )
    frame = _frame_from_daily(
        payload.get("daily", {}), airport=airport, lat=lat, lon=lon
    )
    if frame.empty:
        return frame
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end)
    frame = frame[(frame["date"] >= lo) & (frame["date"] <= hi)]
    return frame


def _fetch_weather_range(
    client: httpx.Client,
    *,
    lat: float,
    lon: float,
    start: date,
    end: date,
    variables: list[str],
    timezone: str,
    airport: str,
    as_of: date,
) -> pd.DataFrame:
    """
    Archive for older history; forecast past_days for the recent window
    (covers calendar today even when archive lags).
    """
    if end < start:
        return pd.DataFrame()

    recent_start = max(start, as_of - timedelta(days=_FORECAST_PAST_DAYS_MAX - 1))
    frames: list[pd.DataFrame] = []

    if start < recent_start:
        hist_end = min(end, recent_start - timedelta(days=1))
        if hist_end >= start:
            frames.append(
                _fetch_archive_range(
                    client,
                    lat=lat,
                    lon=lon,
                    start=start,
                    end=hist_end,
                    variables=variables,
                    timezone=timezone,
                    airport=airport,
                )
            )

    if end >= recent_start:
        frames.append(
            _fetch_forecast_recent(
                client,
                lat=lat,
                lon=lon,
                start=max(start, recent_start),
                end=end,
                variables=variables,
                timezone=timezone,
                airport=airport,
                as_of=as_of,
            )
        )

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ingest_weather(
    use_sample: bool = False,
    *,
    to_r2: bool = False,
    keep_local: bool = True,
    incremental: bool = True,
    rolling: bool | None = None,
    through_today: bool | None = None,
    as_of: date | None = None,
) -> list[str]:
    """
    Ingest daily weather for hubs.

    Live rolling mode extends through calendar ``as_of`` (default today) so
    serving can use fresh weather while BTS flight labels lag and update weekly.

    Incremental mode loads existing lake weather (preferring R2 when uploading)
    and only fetches the missing date gap + a short lookback for revisions.
    Live fetch failures skip the gap (no synthetic poison) unless ``use_sample``.
    """
    ensure_dirs()
    cfg = load_project_config()
    hubs = cfg["hubs"]
    as_of = as_of or date.today()
    window = resolve_weather_window(
        cfg,
        as_of=as_of,
        rolling=rolling,
        use_sample=use_sample,
        through_today=through_today,
    )
    print(f"Weather window: {window} (as_of={as_of.isoformat()})")

    airports = load_airports_frame().set_index("airport")
    prefer_r2 = bool(to_r2 and get_settings().r2_configured)
    lookback_days = int((cfg.get("weather") or {}).get("daily_lookback_days", 7))
    variables = list(cfg["weather"]["variables"])
    timezone = cfg["weather"].get("timezone", "UTC")

    window_start = date(window.start.year, window.start.month, 1)
    window_end = as_of

    written: list[str] = []
    client: httpx.Client | None = None
    if not use_sample:
        client = _http_client()

    try:
        for airport in hubs:
            if airport not in airports.index:
                print(f"Skipping weather for {airport}: missing coordinates")
                continue
            lat = float(airports.loc[airport, "lat"])
            lon = float(airports.loc[airport, "lon"])
            rel = f"weather/airport={airport}/weather.parquet"

            existing = (
                _load_existing_weather(rel, prefer_r2=prefer_r2)
                if incremental
                else None
            )

            frames: list[pd.DataFrame] = []
            fetch_start = window_start
            if existing is not None and incremental and not existing.empty:
                frames.append(existing)
                existing_dates = pd.to_datetime(existing["date"], errors="coerce")
                last = existing_dates.max()
                if pd.notna(last):
                    last_d = last.date()
                    fetch_start = max(
                        window_start, last_d - timedelta(days=lookback_days - 1)
                    )

            if fetch_start > window_end and existing is not None:
                merged = _finalize_weather_frame(existing, window_start, window_end)
                if len(merged) == len(existing):
                    print(f"Weather {airport}: already up to date through {window_end}")
                    continue
                loc = write_parquet(merged, rel, to_r2=to_r2, keep_local=keep_local)
                written.append(loc)
                continue

            if use_sample:
                frames.append(
                    _synthetic_weather_range(airport, fetch_start, window_end, lat, lon)
                )
            else:
                assert client is not None
                try:
                    chunk = _fetch_weather_range(
                        client,
                        lat=lat,
                        lon=lon,
                        start=fetch_start,
                        end=window_end,
                        variables=variables,
                        timezone=timezone,
                        airport=airport,
                        as_of=as_of,
                    )
                    if chunk.empty:
                        print(
                            f"Weather {airport}: Open-Meteo returned no rows "
                            f"for {fetch_start}→{window_end}"
                        )
                    else:
                        frames.append(chunk)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"Weather fetch failed for {airport} "
                        f"{fetch_start}→{window_end}: {exc}"
                    )
                    if not frames:
                        print(f"Weather {airport}: leaving lake unchanged (no fallback)")
                        continue

            if not frames:
                continue
            merged = _finalize_weather_frame(
                pd.concat(frames, ignore_index=True), window_start, window_end
            )
            loc = write_parquet(merged, rel, to_r2=to_r2, keep_local=keep_local)
            written.append(loc)
            print(
                f"Weather {airport}: wrote {len(merged):,} days "
                f"(fetched {fetch_start}→{window_end}) → {loc}"
            )
    finally:
        if client is not None:
            client.close()
    return written


def _finalize_weather_frame(
    df: pd.DataFrame, window_start: date, window_end: date
) -> pd.DataFrame:
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out = out.dropna(subset=["date"]).sort_values("date")
    out = out.drop_duplicates(subset=["date"], keep="last")
    lo = pd.Timestamp(window_start)
    hi = pd.Timestamp(window_end)
    out = out[(out["date"] >= lo) & (out["date"] <= hi)]
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.reset_index(drop=True)


def _synthetic_weather_range(
    airport: str, start: date, end: date, lat: float, lon: float
) -> pd.DataFrame:
    dates: list[str] = []
    temps: list[float] = []
    cursor = start
    while cursor <= end:
        dates.append(cursor.isoformat())
        temps.append(15.0 + (cursor.month - 6) * 1.5)
        cursor += timedelta(days=1)
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "temperature_2m_mean": temps,
            "precipitation_sum": [0.5] * n,
            "windspeed_10m_max": [20.0] * n,
            "weathercode": [1] * n,
            "airport": airport,
            "lat": lat,
            "lon": lon,
        }
    )
