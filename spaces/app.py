"""
Hugging Face Spaces (Streamlit) entrypoint.

Create a Space with SDK=streamlit and either:
1. Point to this repo, set app file to spaces/app.py, or
2. Copy this folder into a Space repo.

Free CPU Space is enough for the UI; run `flight demo` locally (or in a
Space startup script) so models/local/model.joblib exists, or download from HF Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Re-export the main Streamlit app
from flight_agent.ui.app import *  # noqa: E402,F401,F403
