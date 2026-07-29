"""
Hugging Face Spaces (Streamlit) entrypoint.

Prefer deploying via Streamlit Community Cloud with repo-root streamlit_app.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flight_agent.ui import app as _app  # noqa: F401
