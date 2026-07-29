# Flight Delay Ops Agent — Hugging Face Space

Streamlit UI for the flight delay prediction agent.

## Setup

1. Duplicate this Space or connect the GitHub repo.
2. In Space settings, set secrets if needed (`OLLAMA` is typically local-only;
   on Spaces the UI uses the **fallback tool summary** unless you wire a free
   cloud LLM).
3. Pre-build artifacts: run `flight demo` in CI and commit `models/local/` for
   demos, or download `model.joblib` from your HF model repo at startup.

## Local

```bash
uv sync
uv run flight demo
uv run flight ui
```
