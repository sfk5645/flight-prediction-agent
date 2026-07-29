"""Optional Hugging Face Hub publish for model + sample dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from flight_agent.config import get_settings


def publish_to_hf(model_path: Path, sample_path: Path, metrics: dict[str, Any]) -> None:
    settings = get_settings()
    if not settings.hf_token:
        raise RuntimeError("HF_TOKEN is not set; cannot publish.")
    if not settings.hf_repo_id:
        raise RuntimeError("HF_REPO_ID is not set; cannot publish.")

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=settings.hf_token)
    create_repo(settings.hf_repo_id, repo_type="model", exist_ok=True, token=settings.hf_token)
    api.upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo="model.joblib",
        repo_id=settings.hf_repo_id,
        repo_type="model",
    )
    metrics_file = model_path.parent / "metrics.json"
    if metrics_file.exists():
        api.upload_file(
            path_or_fileobj=str(metrics_file),
            path_in_repo="metrics.json",
            repo_id=settings.hf_repo_id,
            repo_type="model",
        )
    card = model_path.parent / "README_MODEL.md"
    card.write_text(
        _model_card(metrics),
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=str(card),
        path_in_repo="README.md",
        repo_id=settings.hf_repo_id,
        repo_type="model",
    )

    if settings.hf_dataset_repo_id and sample_path.exists():
        create_repo(
            settings.hf_dataset_repo_id,
            repo_type="dataset",
            exist_ok=True,
            token=settings.hf_token,
        )
        api.upload_file(
            path_or_fileobj=str(sample_path),
            path_in_repo="sample_features.parquet",
            repo_id=settings.hf_dataset_repo_id,
            repo_type="dataset",
        )

    print(f"Published model to https://huggingface.co/{settings.hf_repo_id}")


def _model_card(metrics: dict[str, Any]) -> str:
    return f"""---
tags:
  - tabular-classification
  - aviation
  - flight-delay
library_name: sklearn
---

# Flight delay classifier (`arr_delay_15`)

XGBoost pipeline predicting whether a US hub flight arrives ≥15 minutes late.

## Metrics

```json
{json.dumps(metrics, indent=2)}
```

## Features

Carrier, origin/dest, schedule time features, Open-Meteo weather, historical route delay rate.

## License

Model trained on public BTS / Open-Meteo / OurAirports data for research and demo use.
"""
