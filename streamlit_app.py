"""Hugging Face Spaces / Streamlit default entrypoint.

HF Spaces looks for ``streamlit_app.py`` at the repo root by default.
The real UI lives in ``flight_agent.ui.app``; this file re-exports it.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from flight_agent.ui.app import *  # noqa: E402,F401,F403
