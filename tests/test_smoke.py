from pathlib import Path

from flight_agent.config import get_settings, load_project_config
from flight_agent.ingest.airports import ingest_airports
from flight_agent.ingest.bts import ingest_bts
from flight_agent.ingest.weather import ingest_weather
from flight_agent.serve.app import PredictRequest


def test_project_config_hubs():
    cfg = load_project_config()
    assert set(cfg["hubs"]) == {"LAX", "JFK", "ORD", "DEN", "ATL", "IAD", "DFW"}
    assert cfg["retention_months"] == 48
    assert cfg["model"].get("hub_pair_only") is False


def test_retention_cutoff_and_local_prune(tmp_path, monkeypatch):
    from datetime import date

    from flight_agent.ingest.retention import cutoff_date, prune_local_bts

    assert cutoff_date(as_of=date(2026, 7, 28), months=48) == date(2022, 7, 1)

    data_dir = tmp_path / "data"
    monkeypatch.setenv("FLIGHT_DATA_DIR", str(data_dir))
    get_settings.cache_clear()
    settings = get_settings()

    old = settings.raw_dir / "bts" / "year=2020" / "month=01"
    keep = settings.raw_dir / "bts" / "year=2024" / "month=06"
    old.mkdir(parents=True)
    keep.mkdir(parents=True)
    (old / "flights.parquet").write_text("x")
    (keep / "flights.parquet").write_text("y")

    removed = prune_local_bts(date(2022, 7, 1), dry_run=False)
    assert any("2020" in r for r in removed)
    assert not old.exists()
    assert keep.exists()


def test_sample_ingest(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    monkeypatch.setenv("FLIGHT_DATA_DIR", str(data_dir))
    get_settings.cache_clear()

    cfg = {
        "hubs": ["LAX", "JFK", "ORD", "DEN", "ATL", "IAD", "DFW"],
        "date_range": {
            "mode": "fixed",
            "start_year": 2024,
            "start_month": 1,
            "end_year": 2024,
            "end_month": 1,
        },
        "retention_months": 48,
        "sample": {"rows_per_month": 50},
        "weather": {
            "timezone": "UTC",
            "variables": [
                "temperature_2m_mean",
                "precipitation_sum",
                "windspeed_10m_max",
                "weathercode",
            ],
        },
    }
    monkeypatch.setattr("flight_agent.ingest.bts.load_project_config", lambda: cfg)
    monkeypatch.setattr("flight_agent.ingest.weather.load_project_config", lambda: cfg)
    monkeypatch.setattr("flight_agent.ingest.airports.load_project_config", lambda: cfg)
    monkeypatch.setattr("flight_agent.ingest.schedule.load_project_config", lambda: cfg)

    path = ingest_airports()
    assert path
    assert ingest_weather(use_sample=True, incremental=False)
    result = ingest_bts(use_sample=True, incremental=False)
    assert result.written
    assert (data_dir / "raw" / "bts").exists()

    # Incremental re-run should skip the existing month
    skipped = ingest_bts(use_sample=True, incremental=True)
    assert skipped.written == []
    assert skipped.months_written == []


def test_rolling_window_uses_retention(monkeypatch):
    from datetime import date

    from flight_agent.ingest.schedule import YearMonth, resolve_ingest_window

    cfg = {
        "date_range": {"mode": "rolling"},
        "retention_months": 48,
    }
    monkeypatch.setattr(
        "flight_agent.ingest.schedule.discover_latest_bts_month",
        lambda as_of=None: YearMonth(2026, 5),
    )
    window = resolve_ingest_window(cfg, as_of=date(2026, 7, 28))
    assert window.mode == "rolling"
    assert window.end == YearMonth(2026, 5)
    assert window.start == YearMonth(2022, 7)
    assert len(window.months) == 47  # Jul 2022 → May 2026 inclusive under cutoff


def test_should_auto_retrain_on_window_move(tmp_path, monkeypatch):
    from flight_agent.ingest.schedule import IngestWindow, YearMonth
    from flight_agent.train.retrain import mark_trained, should_auto_retrain

    monkeypatch.setenv("FLIGHT_MODEL_DIR", str(tmp_path / "models"))
    get_settings.cache_clear()

    window = IngestWindow(
        start=YearMonth(2022, 7),
        end=YearMonth(2026, 5),
        mode="rolling",
        months=[],
    )
    needed, reason = should_auto_retrain(window, new_bts_months=0, pruned_bts=False)
    assert needed and reason == "no_model_artifact"

    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "model.joblib").write_text("x")
    mark_trained(window, reason="test")

    needed, reason = should_auto_retrain(window, new_bts_months=0, pruned_bts=False)
    assert not needed

    needed, reason = should_auto_retrain(window, new_bts_months=1, pruned_bts=False)
    assert needed and "new_bts" in reason

    moved = IngestWindow(
        start=YearMonth(2022, 8),
        end=YearMonth(2026, 6),
        mode="rolling",
        months=[],
    )
    needed, reason = should_auto_retrain(moved, new_bts_months=0, pruned_bts=False)
    assert needed



def test_write_parquet_local_only(tmp_path, monkeypatch):
    import pandas as pd

    from flight_agent.ingest.storage import write_parquet

    monkeypatch.setenv("FLIGHT_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    loc = write_parquet(
        pd.DataFrame({"a": [1, 2]}),
        "airports/airports.parquet",
        to_r2=False,
        keep_local=True,
    )
    assert Path(loc).exists()


def test_filter_weather_frame():
    from datetime import date

    import pandas as pd

    from flight_agent.ingest.retention import filter_weather_frame

    df = pd.DataFrame(
        {
            "date": ["2020-01-01", "2024-06-01", "2025-01-01"],
            "temperature_2m_mean": [1.0, 2.0, 3.0],
        }
    )
    filtered, dropped = filter_weather_frame(df, date(2022, 7, 1))
    assert dropped == 1
    assert list(filtered["date"]) == ["2024-06-01", "2025-01-01"]


def test_predict_request_model():
    body = PredictRequest(
        op_unique_carrier="DL",
        origin="ATL",
        dest="LAX",
        fl_month=6,
        fl_dow=1,
        crs_dep_hour=8,
    )
    assert body.origin == "ATL"
