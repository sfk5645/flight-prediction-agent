"""Normalize airline / airport names and free-text dates for tools."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

# BTS / IATA operating-carrier codes commonly in this lake.
CARRIER_ALIASES: dict[str, str] = {
    "UA": "UA",
    "UNITED": "UA",
    "UNITED AIRLINES": "UA",
    "UNITED AIR": "UA",
    "AA": "AA",
    "AMERICAN": "AA",
    "AMERICAN AIRLINES": "AA",
    "DL": "DL",
    "DELTA": "DL",
    "DELTA AIR LINES": "DL",
    "DELTA AIRLINES": "DL",
    "B6": "B6",
    "JETBLUE": "B6",
    "JET BLUE": "B6",
    "WN": "WN",
    "SOUTHWEST": "WN",
    "SOUTHWEST AIRLINES": "WN",
    "AS": "AS",
    "ALASKA": "AS",
    "ALASKA AIRLINES": "AS",
    "NK": "NK",
    "SPIRIT": "NK",
    "SPIRIT AIRLINES": "NK",
    "F9": "F9",
    "FRONTIER": "F9",
    "FRONTIER AIRLINES": "F9",
    "G4": "G4",
    "ALLEGIANT": "G4",
    "HA": "HA",
    "HAWAIIAN": "HA",
    "HAWAIIAN AIRLINES": "HA",
    "OO": "OO",
    "SKYWEST": "OO",
    "YX": "YX",
    "REPUBLIC": "YX",
    "9E": "9E",
    "ENDEAVOR": "9E",
    "MQ": "MQ",
    "ENVOY": "MQ",
    "OH": "OH",
    "PSA": "OH",
    "YX REPUBLIC": "YX",
}

# Hub IATA codes + common English names / city nicknames.
AIRPORT_ALIASES: dict[str, str] = {
    "LAX": "LAX",
    "LOS ANGELES": "LAX",
    "LOS ANGELES INTERNATIONAL": "LAX",
    "LA": "LAX",
    "JFK": "JFK",
    "JOHN F KENNEDY": "JFK",
    "KENNEDY": "JFK",
    "NEW YORK JFK": "JFK",
    "NYC JFK": "JFK",
    "ORD": "ORD",
    "OHARE": "ORD",
    "O'HARE": "ORD",
    "O HARE": "ORD",
    "CHICAGO OHARE": "ORD",
    "CHICAGO O'HARE": "ORD",
    "CHICAGO": "ORD",
    "DEN": "DEN",
    "DENVER": "DEN",
    "DENVER INTERNATIONAL": "DEN",
    "ATL": "ATL",
    "ATLANTA": "ATL",
    "HARTSFIELD": "ATL",
    "HARTSFIELD JACKSON": "ATL",
    "IAD": "IAD",
    "DULLES": "IAD",
    "WASHINGTON DULLES": "IAD",
    "DULLES INTERNATIONAL": "IAD",
    "DFW": "DFW",
    "DALLAS": "DFW",
    "DALLAS FORT WORTH": "DFW",
    "DALLAS-FORT WORTH": "DFW",
    "FORT WORTH": "DFW",
}


def _clean_token(value: str) -> str:
    s = value.strip().upper()
    s = s.replace("’", "'")
    s = re.sub(r"[^\w\s'/.-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_carrier(value: str | None) -> str | None:
    """Map airline name or code → 2-letter BTS op_unique_carrier (or None)."""
    if value is None:
        return None
    key = _clean_token(value)
    if not key:
        return None
    if key in CARRIER_ALIASES:
        return CARRIER_ALIASES[key]
    # Strip trailing "AIRLINES" / "AIR LINES" and retry
    for suffix in (" AIRLINES", " AIR LINES", " AIR"):
        if key.endswith(suffix):
            base = key[: -len(suffix)].strip()
            if base in CARRIER_ALIASES:
                return CARRIER_ALIASES[base]
    if len(key) == 2 and key.isalnum():
        return key
    return key  # last resort: uppercase as given (may miss)


def normalize_airport(value: str | None) -> str | None:
    """Map airport / city name → IATA code when known."""
    if value is None:
        return None
    key = _clean_token(value)
    if not key:
        return None
    if key in AIRPORT_ALIASES:
        return AIRPORT_ALIASES[key]
    # Drop trailing "AIRPORT" / "INTERNATIONAL"
    for suffix in (" INTERNATIONAL AIRPORT", " AIRPORT", " INTERNATIONAL"):
        if key.endswith(suffix):
            base = key[: -len(suffix)].strip()
            if base in AIRPORT_ALIASES:
                return AIRPORT_ALIASES[base]
    if len(key) == 3 and key.isalpha():
        return key
    return key


def find_carriers_in_text(text: str) -> list[str]:
    """Find carrier codes mentioned by name or IATA in free text (longest match first)."""
    upper = _clean_token(text)
    found: list[str] = []
    for alias in sorted(CARRIER_ALIASES.keys(), key=len, reverse=True):
        if len(alias) <= 3:
            if not re.search(rf"\b{re.escape(alias)}\b", upper):
                continue
        elif alias not in upper:
            continue
        code = CARRIER_ALIASES[alias]
        if code not in found:
            found.append(code)
    return found


def find_airports_in_text(text: str) -> list[str]:
    """Find hub airports mentioned by name or IATA (longest alias first)."""
    upper = _clean_token(text)
    found: list[str] = []
    for alias in sorted(AIRPORT_ALIASES.keys(), key=len, reverse=True):
        # Short tokens (IATA / "LA") need word boundaries to avoid "DELAY"→LAX.
        if len(alias) <= 3:
            if not re.search(rf"\b{re.escape(alias)}\b", upper):
                continue
        elif alias not in upper:
            continue
        code = AIRPORT_ALIASES[alias]
        if code not in found:
            found.append(code)
    return found


_ISO_DATE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_US_DATE = re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b")


def parse_weather_date(raw: str | None) -> tuple[str | None, str | None]:
    """
    Normalize a free-text date for daily weather lookups.

    Returns (YYYY-MM-DD or None for 'latest', optional note).
    Time-of-day phrases are ignored (weather is daily, not hourly).
    """
    if raw is None:
        return None, None
    text = str(raw).strip()
    if not text:
        return None, None

    lower = text.lower().strip()
    note_parts: list[str] = []

    # Strip clock-time so "today at 10pm" → "today"
    stripped = re.sub(
        r"\b(at\s+)?\d{1,2}(:\d{2})?\s*(a\.?m\.?|p\.?m\.?)\b",
        "",
        lower,
        flags=re.I,
    )
    stripped = re.sub(r"\b\d{1,2}:\d{2}\b", "", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip(" ,;")
    if stripped != lower.strip():
        note_parts.append(
            "Weather in this lake is daily (not hourly); time of day was ignored."
        )

    if stripped in ("", "latest", "most recent", "last"):
        return None, "; ".join(note_parts) or None

    if stripped in ("today", "now") or stripped.startswith("today"):
        return date.today().isoformat(), "; ".join(
            note_parts
            + ["Resolved 'today' to calendar date (may be ahead of lake coverage)."]
        )

    if stripped in ("yesterday",) or stripped.startswith("yesterday"):
        return (date.today() - timedelta(days=1)).isoformat(), "; ".join(
            note_parts + ["Resolved 'yesterday' to calendar date."]
        )

    m = _ISO_DATE.search(text)
    if m:
        try:
            datetime.strptime(m.group(1), "%Y-%m-%d")
            return m.group(1), "; ".join(note_parts) or None
        except ValueError:
            pass

    m = _US_DATE.search(stripped)
    if m:
        month, day, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            d = date(year, month, day)
            return d.isoformat(), "; ".join(note_parts) or None
        except ValueError:
            pass

    # Strict ISO attempt on the whole string
    try:
        d = datetime.strptime(stripped[:10], "%Y-%m-%d").date()
        return d.isoformat(), "; ".join(note_parts) or None
    except ValueError:
        pass

    return None, (
        f"Could not parse date '{raw}' (expected YYYY-MM-DD, today, or yesterday); "
        "using latest available weather day."
        + (f" {'; '.join(note_parts)}" if note_parts else "")
    )
