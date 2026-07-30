"""Push / pull trained model artifacts to Cloudflare R2 for Streamlit Cloud."""

from __future__ import annotations

from pathlib import Path

from flight_agent.config import get_settings
from flight_agent.ingest.r2_sync import _client

MODEL_R2_PREFIX = "models/"
MODEL_FILES = (
    "model.joblib",
    "model_regressor.joblib",
    "meta.json",
    "metrics.json",
    "train_state.json",
)


def model_r2_key(filename: str) -> str:
    return f"{MODEL_R2_PREFIX}{filename}"


def push_model_to_r2(model_dir: Path | None = None) -> list[str]:
    """Upload local model artifacts to R2 under models/."""
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("R2 is not configured; cannot push model.")
    root = Path(model_dir or settings.model_dir)
    client = _client()
    uploaded: list[str] = []
    for name in MODEL_FILES:
        path = root / name
        if not path.exists():
            continue
        key = model_r2_key(name)
        client.upload_file(str(path), settings.r2_bucket, key)
        uploaded.append(key)
        print(f"Uploaded s3://{settings.r2_bucket}/{key}", flush=True)
    if "models/model.joblib" not in uploaded and not (root / "model.joblib").exists():
        raise FileNotFoundError(f"No model.joblib in {root}; run `flight train` first.")
    return uploaded


def pull_model_from_r2(
    model_dir: Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Download model artifacts from R2 into the local model dir."""
    settings = get_settings()
    if not settings.r2_configured:
        raise RuntimeError("R2 is not configured; cannot pull model.")
    root = Path(model_dir or settings.model_dir)
    root.mkdir(parents=True, exist_ok=True)
    local_model = root / "model.joblib"
    if local_model.exists() and not force:
        return root

    client = _client()
    for name in MODEL_FILES:
        key = model_r2_key(name)
        dest = root / name
        try:
            client.head_object(Bucket=settings.r2_bucket, Key=key)
        except Exception:  # noqa: BLE001
            if name == "model.joblib":
                raise FileNotFoundError(
                    f"Model not on R2 at s3://{settings.r2_bucket}/{key}. "
                    "Train and run `flight model push`, or wait for the weekly Actions job."
                ) from None
            continue
        client.download_file(settings.r2_bucket, key, str(dest))
        print(f"Downloaded {key} → {dest}", flush=True)
    return root


def ensure_model_artifacts(*, force_pull: bool = False) -> Path:
    """
    Ensure model.joblib exists locally.

    Prefer local file; otherwise download from R2 when credentials are present.
    """
    settings = get_settings()
    root = Path(settings.model_dir)
    local_model = root / "model.joblib"
    if local_model.exists() and not force_pull:
        return root
    if settings.r2_configured:
        return pull_model_from_r2(root, force=force_pull or not local_model.exists())
    if not local_model.exists():
        raise FileNotFoundError(
            f"Model not found at {local_model}. Run `flight train`, or set R2_* "
            "and push/pull with `flight model push` / `flight model pull`."
        )
    return root
