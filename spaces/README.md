# Flight Delay Ops Agent — optional Hugging Face Space

Primary free UI host is **Streamlit Community Cloud** (`streamlit_app.py` +
`requirements.txt` at the repo root). This folder is an alternate entrypoint.

## Streamlit Community Cloud (recommended)

1. Deploy the GitHub repo at [share.streamlit.io](https://share.streamlit.io).
2. Main file: `streamlit_app.py`
3. Secrets: `GROQ_API_KEY`, `R2_*` (see `.streamlit/secrets.toml.example`).
4. Ensure GitHub Actions (or `flight model push`) has uploaded `models/model.joblib` to R2.

## Hugging Face Space

1. Create a Space with SDK=streamlit; app file `spaces/app.py` or point at `streamlit_app.py`.
2. Add the same secrets as above.
3. Without `GROQ_API_KEY`, the UI uses the deterministic tool summary fallback.
