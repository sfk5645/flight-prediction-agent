"""Optional Evidently data-drift report."""

from __future__ import annotations

from pathlib import Path

from flight_agent.config import get_settings
from flight_agent.features.build import FEATURE_COLUMNS, build_training_frame


def run_drift_report() -> Path:
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as exc:
        raise ImportError(
            "Install optional drift extras: `uv sync --extra drift`"
        ) from exc

    settings = get_settings()
    df = build_training_frame()
    if len(df) < 200:
        raise RuntimeError("Need at least 200 rows for a meaningful drift report.")

    mid = len(df) // 2
    reference = df.iloc[:mid][FEATURE_COLUMNS]
    current = df.iloc[mid:][FEATURE_COLUMNS]

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    out = Path(settings.model_dir) / "drift_report.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out))
    return out
