"""Map Streamlit Cloud secrets (and similar) into process env for Settings."""

from __future__ import annotations

import os

# Flat secrets matching .env / GitHub Actions / Streamlit Secrets.
_ENV_KEYS = (
    "GROQ_API_KEY",
    "GROQ_MODEL",
    "R2_ACCOUNT_ID",
    "R2_ACCESS_KEY_ID",
    "R2_SECRET_ACCESS_KEY",
    "R2_BUCKET",
    "R2_ENDPOINT_URL",
    "FLIGHT_MODEL_DIR",
    "FLIGHT_DATA_DIR",
    "FLIGHT_DUCKDB_PATH",
    "HF_TOKEN",
    "HF_REPO_ID",
)


def apply_runtime_secrets() -> None:
    """
    Copy Streamlit secrets into os.environ when env vars are unset.

    Safe to call outside Streamlit (no-op). Call before get_settings().
    """
    try:
        import streamlit as st
    except ImportError:
        return

    try:
        secrets = st.secrets
    except Exception:  # noqa: BLE001
        return

    for key in _ENV_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = secrets.get(key)
        except Exception:  # noqa: BLE001
            value = None
        if value is None or value == "":
            continue
        os.environ[key] = str(value)
