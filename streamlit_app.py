"""
Streamlit Community Cloud entrypoint.

Deploy: Main file path = streamlit_app.py
Secrets: see .streamlit/secrets.toml.example (GROQ_*, R2_*).
GitHub Actions trains and pushes model.joblib to R2; the UI pulls it on demand.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Shared UI (set_page_config is first Streamlit call inside this module).
from flight_agent.ui import app as _app  # noqa: F401
