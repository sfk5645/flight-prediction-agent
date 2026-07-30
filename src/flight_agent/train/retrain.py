"""Auto-retrain when the rolling lake window advances or new months land."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from flight_agent.config import get_settings, load_project_config
from flight_agent.ingest.schedule import IngestWindow, YearMonth


@dataclass
class TrainState:
    window_start: YearMonth
    window_end: YearMonth
    trained_at: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_start": str(self.window_start),
            "window_end": str(self.window_end),
            "trained_at": self.trained_at,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainState:
        def _parse(key: str) -> YearMonth:
            raw = str(data[key])
            y, m = raw.split("-", 1)
            return YearMonth(int(y), int(m))

        return cls(
            window_start=_parse("window_start"),
            window_end=_parse("window_end"),
            trained_at=str(data.get("trained_at", "")),
            reason=str(data.get("reason", "")),
        )


def train_state_path() -> Path:
    return Path(get_settings().model_dir) / "train_state.json"


def model_artifact_exists() -> bool:
    return (Path(get_settings().model_dir) / "model.joblib").exists()


def load_train_state() -> TrainState | None:
    path = train_state_path()
    if not path.exists():
        return None
    try:
        return TrainState.from_dict(json.loads(path.read_text()))
    except Exception:  # noqa: BLE001
        return None


def save_train_state(window: IngestWindow, *, reason: str) -> Path:
    settings = get_settings()
    Path(settings.model_dir).mkdir(parents=True, exist_ok=True)
    state = TrainState(
        window_start=window.start,
        window_end=window.end,
        trained_at=datetime.now(timezone.utc).isoformat(),
        reason=reason,
    )
    path = train_state_path()
    path.write_text(json.dumps(state.to_dict(), indent=2))
    return path


def auto_retrain_enabled(cfg: dict | None = None) -> bool:
    cfg = cfg or load_project_config()
    model = cfg.get("model") or {}
    return bool(model.get("auto_retrain", True))


def should_auto_retrain(
    window: IngestWindow,
    *,
    new_bts_months: int = 0,
    pruned_bts: bool = False,
    weather_updated: bool = False,
) -> tuple[bool, str]:
    """
    Decide whether the sliding window moved enough to warrant a retrain.

    Triggers:
    - no model or no prior train_state
    - new BTS months landed
    - weather lake refreshed (daily Open-Meteo or weekly co-ingest)
    - retention prune removed BTS months (left edge slid)
    - resolved window start/end differs from last trained window
    """
    if not model_artifact_exists():
        return True, "no_model_artifact"

    state = load_train_state()
    if state is None:
        return True, "no_train_state"

    if new_bts_months > 0:
        return True, f"new_bts_months={new_bts_months}"

    if weather_updated:
        return True, "weather_updated"

    if pruned_bts:
        return True, "retention_prune_moved_window"

    if window.end.as_tuple() != state.window_end.as_tuple():
        return True, f"window_end {state.window_end}→{window.end}"

    if window.start.as_tuple() != state.window_start.as_tuple():
        return True, f"window_start {state.window_start}→{window.start}"

    return False, "window_unchanged"


def mark_trained(window: IngestWindow, *, reason: str) -> Path:
    return save_train_state(window, reason=reason)
