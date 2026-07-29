"""Local cache of small curated marts for fast serve/UI (free-tier friendly)."""

from __future__ import annotations

import os
import threading
from pathlib import Path

from flight_agent.config import get_settings
from flight_agent.ingest.r2_sync import _client
from flight_agent.ingest.warehouse import (
    SERVE_MART_TABLES,
    curated_key,
    curated_on_r2,
)

_log_lock = threading.Lock()


def _log(msg: str) -> None:
    with _log_lock:
        print(msg, flush=True)


def serve_cache_dir() -> Path:
    """
    Directory for downloaded agg Parquet files.

    Prefer project data/serve_cache; on ephemeral hosts (Streamlit Cloud) use /tmp.
    Override with FLIGHT_SERVE_CACHE_DIR.
    """
    override = os.environ.get("FLIGHT_SERVE_CACHE_DIR", "").strip()
    if override:
        path = Path(override)
    else:
        # Streamlit Cloud / ephemeral: prefer /tmp so we don't fill the repo mount.
        if Path("/mount/src").exists() or os.environ.get("STREAMLIT_SHARING_MODE"):
            path = Path("/tmp/flight_serve_cache")
        else:
            path = Path(get_settings().data_dir) / "serve_cache"
    path.mkdir(parents=True, exist_ok=True)
    return path


def serve_cache_paths() -> dict[str, Path]:
    root = serve_cache_dir()
    return {t: root / f"{t}.parquet" for t in SERVE_MART_TABLES}


def serve_cache_ready() -> bool:
    paths = serve_cache_paths()
    return all(p.exists() and p.stat().st_size > 0 for p in paths.values())


def sync_serve_cache(*, force: bool = False) -> list[str]:
    """
    Download small curated agg marts from R2 into the local serve cache.

    These files are typically tens of MB total — fine for free Streamlit / laptop.
    """
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("R2 is not configured; cannot sync serve cache.")
    if not curated_on_r2("flt_route_delay_stats"):
        raise FileNotFoundError(
            "curated/flt_route_delay_stats.parquet missing on R2. "
            "Run `flight dbt build --from-r2` / warehouse push first."
        )

    if serve_cache_ready() and not force:
        _log(f"Serve cache already ready at {serve_cache_dir()}")
        return [str(p) for p in serve_cache_paths().values()]

    client = _client()
    downloaded: list[str] = []
    for table, dest in serve_cache_paths().items():
        if dest.exists() and dest.stat().st_size > 0 and not force:
            downloaded.append(str(dest))
            continue
        key = curated_key(table)
        _log(f"Downloading s3://{settings.r2_bucket}/{key} → {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(settings.r2_bucket, key, str(dest))
        downloaded.append(str(dest))
    _log(f"Serve cache ready ({len(downloaded)} files) at {serve_cache_dir()}")
    return downloaded


def ensure_serve_cache() -> Path:
    """Sync from R2 if needed; return cache directory."""
    if not serve_cache_ready():
        sync_serve_cache(force=False)
    return serve_cache_dir()
