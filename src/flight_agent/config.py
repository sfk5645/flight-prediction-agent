"""Application settings and project config loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "project.yaml"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    data_dir: Path = Field(default=ROOT / "data", validation_alias="FLIGHT_DATA_DIR")
    duckdb_path: Path = Field(
        default=ROOT / "data" / "duckdb" / "flight_agent.duckdb",
        validation_alias="FLIGHT_DUCKDB_PATH",
    )
    # When False (default), after warehouse push / dbt --from-r2 the local .duckdb
    # is deleted so gold lives on R2 as curated Parquet (+ optional .duckdb backup).
    duckdb_keep_local: bool = Field(
        default=False, validation_alias="FLIGHT_DUCKDB_KEEP_LOCAL"
    )
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db", validation_alias="FLIGHT_MLFLOW_TRACKING_URI"
    )
    model_dir: Path = Field(
        default=ROOT / "models" / "local", validation_alias="FLIGHT_MODEL_DIR"
    )

    r2_account_id: str = Field(default="", validation_alias="R2_ACCOUNT_ID")
    r2_access_key_id: str = Field(default="", validation_alias="R2_ACCESS_KEY_ID")
    r2_secret_access_key: str = Field(default="", validation_alias="R2_SECRET_ACCESS_KEY")
    r2_bucket: str = Field(default="flight-prediction-lake", validation_alias="R2_BUCKET")
    r2_endpoint_url: str = Field(default="", validation_alias="R2_ENDPOINT_URL")

    hf_token: str = Field(default="", validation_alias="HF_TOKEN")
    hf_repo_id: str = Field(default="", validation_alias="HF_REPO_ID")
    hf_dataset_repo_id: str = Field(default="", validation_alias="HF_DATASET_REPO_ID")

    # Groq LLM for the LangGraph ops agent (free tier: llama-3.1-8b-instant)
    groq_api_key: str = Field(default="", validation_alias="GROQ_API_KEY")
    groq_model: str = Field(
        default="llama-3.1-8b-instant", validation_alias="GROQ_MODEL"
    )

    api_host: str = Field(default="127.0.0.1", validation_alias="FLIGHT_API_HOST")
    api_port: int = Field(default=8000, validation_alias="FLIGHT_API_PORT")

    @property
    def raw_dir(self) -> Path:
        return Path(self.data_dir) / "raw"

    @property
    def curated_dir(self) -> Path:
        return Path(self.data_dir) / "curated"

    @property
    def r2_endpoint(self) -> str:
        if self.r2_endpoint_url:
            return self.r2_endpoint_url
        if self.r2_account_id:
            return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"
        return ""

    @property
    def r2_configured(self) -> bool:
        return bool(self.r2_access_key_id and self.r2_secret_access_key and self.r2_endpoint)


@lru_cache
def get_settings() -> Settings:
    from flight_agent.runtime_secrets import apply_runtime_secrets

    apply_runtime_secrets()
    return Settings()


@lru_cache
def load_project_config(path: Path | None = None) -> dict[str, Any]:
    cfg_path = path or CONFIG_PATH
    with cfg_path.open() as f:
        return yaml.safe_load(f)


def ensure_dirs(settings: Settings | None = None) -> None:
    s = settings or get_settings()
    for p in (
        s.raw_dir / "bts",
        s.raw_dir / "weather",
        s.raw_dir / "airports",
        s.curated_dir,
        Path(s.duckdb_path).parent,
        Path(s.model_dir),
    ):
        p.mkdir(parents=True, exist_ok=True)
