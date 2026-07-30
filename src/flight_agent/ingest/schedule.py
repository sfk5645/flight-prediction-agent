"""Resolve rolling ingest windows and discover newly published BTS months."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import httpx

from flight_agent.config import get_settings, load_project_config
from flight_agent.ingest.retention import cutoff_date, retention_months

# Keep in sync with flight_agent.ingest.bts.BTS_ZIP_URL
BTS_ZIP_URL = (
    "https://transtats.bts.gov/PREZIP/"
    "On_Time_Reporting_Carrier_On_Time_Performance_1987_present_{year}_{month}.zip"
)

_BTS_LOCAL_RE = re.compile(r"year=(\d{4})/month=(\d{2})")
_R2_BTS_RE = re.compile(r"raw/bts/year=(\d{4})/month=(\d{2})/")


@dataclass(frozen=True)
class YearMonth:
    year: int
    month: int

    def __str__(self) -> str:
        return f"{self.year:04d}-{self.month:02d}"

    def as_tuple(self) -> tuple[int, int]:
        return self.year, self.month


@dataclass
class IngestWindow:
    start: YearMonth
    end: YearMonth
    mode: str  # "rolling" | "fixed"
    months: list[YearMonth]

    def __str__(self) -> str:
        return f"{self.mode} {self.start} → {self.end} ({len(self.months)} months)"


def _iter_months(start: YearMonth, end: YearMonth) -> list[YearMonth]:
    out: list[YearMonth] = []
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        out.append(YearMonth(y, m))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return out


def _shift_months(ym: YearMonth, delta: int) -> YearMonth:
    """Shift YearMonth by delta months (can be negative)."""
    idx = ym.year * 12 + (ym.month - 1) + delta
    year = idx // 12
    month = idx % 12 + 1
    return YearMonth(year, month)


def bts_month_available(year: int, month: int, client: httpx.Client | None = None) -> bool:
    """Return True if the BTS PREZIP for year/month exists (HTTP 200)."""
    url = BTS_ZIP_URL.format(year=year, month=month)
    close = False
    if client is None:
        client = httpx.Client(timeout=30.0, follow_redirects=True)
        close = True
    try:
        # Prefer HEAD; some CDNs mishandle it — fall back to ranged GET
        resp = client.head(url)
        if resp.status_code == 200:
            return True
        if resp.status_code in (403, 405, 501):
            resp = client.get(url, headers={"Range": "bytes=0-0"})
            return resp.status_code in (200, 206)
        return False
    except Exception:  # noqa: BLE001
        return False
    finally:
        if close:
            client.close()


def discover_latest_bts_month(
    as_of: date | None = None,
    max_lookback: int = 8,
) -> YearMonth:
    """
    Walk backward from as_of's month until a published BTS PREZIP is found.
    BTS typically lags ~1–2 months.
    """
    as_of = as_of or date.today()
    cursor = YearMonth(as_of.year, as_of.month)
    with httpx.Client(timeout=30.0, follow_redirects=True) as client:
        for _ in range(max_lookback):
            if bts_month_available(cursor.year, cursor.month, client=client):
                return cursor
            cursor = _shift_months(cursor, -1)
    # Fallback: assume ~2 month lag if discovery fails
    return _shift_months(YearMonth(as_of.year, as_of.month), -2)


def list_existing_bts_months(
    *,
    check_local: bool = True,
    check_r2: bool = False,
) -> set[YearMonth]:
    """Months already present locally and/or on R2."""
    found: set[YearMonth] = set()
    settings = get_settings()

    if check_local:
        bts_root = Path(settings.raw_dir) / "bts"
        if bts_root.exists():
            for path in bts_root.glob("year=*/month=*/flights.parquet"):
                m = _BTS_LOCAL_RE.search(path.as_posix())
                if m:
                    found.add(YearMonth(int(m.group(1)), int(m.group(2))))

    if check_r2 and settings.r2_configured:
        from flight_agent.ingest.r2_sync import _client

        client = _client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.r2_bucket, Prefix="raw/bts/"):
            for obj in page.get("Contents", []):
                m = _R2_BTS_RE.search(obj["Key"])
                if m:
                    found.add(YearMonth(int(m.group(1)), int(m.group(2))))
    return found


def resolve_ingest_window(
    cfg: dict | None = None,
    *,
    as_of: date | None = None,
    rolling: bool | None = None,
    use_sample: bool = False,
) -> IngestWindow:
    """
    Resolve which months to target.

    mode=rolling (default): end = latest published BTS month;
    start = max(retention cutoff, end - retention_months + 1).

    mode=fixed: use date_range start/end from config.
    """
    cfg = cfg or load_project_config()
    dr = cfg.get("date_range") or {}
    mode = str(dr.get("mode", "rolling")).lower()
    if rolling is False:
        mode = "fixed"
    elif rolling is True:
        mode = "rolling"

    if mode == "fixed":
        start = YearMonth(int(dr["start_year"]), int(dr["start_month"]))
        end = YearMonth(int(dr["end_year"]), int(dr["end_month"]))
    else:
        as_of = as_of or date.today()
        if use_sample:
            # Avoid network in demo/CI; assume ~2 month publishing lag
            end = _shift_months(YearMonth(as_of.year, as_of.month), -2)
        else:
            end = discover_latest_bts_month(as_of=as_of)
        months_keep = retention_months(cfg)
        # Inclusive window of `months_keep` months ending at `end`
        start = _shift_months(end, -(months_keep - 1))
        # Also respect absolute retention cutoff from "today"
        cut = cutoff_date(as_of=as_of, months=months_keep)
        cut_ym = YearMonth(cut.year, cut.month)
        if start.as_tuple() < cut_ym.as_tuple():
            start = cut_ym

    if start.as_tuple() > end.as_tuple():
        start = end

    months = _iter_months(start, end)
    return IngestWindow(start=start, end=end, mode=mode, months=months)


def months_to_ingest(
    window: IngestWindow,
    *,
    incremental: bool = True,
    check_local: bool = True,
    check_r2: bool = False,
) -> list[YearMonth]:
    """Filter window to months not already in the lake (when incremental)."""
    if not incremental:
        return list(window.months)
    existing = list_existing_bts_months(check_local=check_local, check_r2=check_r2)
    return [m for m in window.months if m not in existing]


def resolve_weather_window(
    cfg: dict | None = None,
    *,
    as_of: date | None = None,
    rolling: bool | None = None,
    use_sample: bool = False,
    through_today: bool | None = None,
) -> IngestWindow:
    """
    Weather coverage window.

    By default (weather.through_today: true) the end is the current calendar
    month through ``as_of`` (today), while the start follows the same retention
    cutoff as BTS. That lets daily weather stay fresh even though BTS labels
    lag ~1–2 months and only refresh weekly.

    Fixed / sample / through_today=false keep weather aligned to the BTS window
    (deterministic demos and CI).
    """
    cfg = cfg or load_project_config()
    as_of = as_of or date.today()
    bts_window = resolve_ingest_window(
        cfg, as_of=as_of, rolling=rolling, use_sample=use_sample
    )
    wx_cfg = cfg.get("weather") or {}
    if through_today is None:
        through_today = bool(wx_cfg.get("through_today", True))

    if (
        not through_today
        or use_sample
        or bts_window.mode == "fixed"
        or rolling is False
    ):
        return bts_window

    end = YearMonth(as_of.year, as_of.month)
    start = bts_window.start
    if start.as_tuple() > end.as_tuple():
        start = end
    months = _iter_months(start, end)
    return IngestWindow(
        start=start,
        end=end,
        mode="rolling_weather",
        months=months,
    )


def weather_months_to_fetch(
    window: IngestWindow,
    have: set[YearMonth],
    *,
    incremental: bool = True,
    as_of: date | None = None,
) -> list[YearMonth]:
    """
    Months to download for weather.

    Incremental mode skips months already present, but always re-fetches the
    current calendar month (and the previous month early in the month) so new
    days land on a daily refresh.
    """
    as_of = as_of or date.today()
    if not incremental:
        return list(window.months)

    targets = [m for m in window.months if m not in have]
    force: list[YearMonth] = [YearMonth(as_of.year, as_of.month)]
    if as_of.day <= 3:
        force.append(_shift_months(force[0], -1))
    for m in force:
        if m in window.months and m not in targets:
            targets.append(m)
    targets.sort(key=lambda ym: ym.as_tuple())
    return targets
