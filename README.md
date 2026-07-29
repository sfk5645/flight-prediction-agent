# Flight Delay Prediction Agent

End-to-end **data engineering + MLOps + AI agent** project for US aviation delays.

Public flight and weather data land as Parquet (local and optional Cloudflare R2),
are transformed with **dbt-duckdb**, train an **XGBoost** ≥15-minute arrival-delay
model tracked in **MLflow**, and power a **LangGraph + Groq** ops agent with a
**Streamlit** chat UI — all on free / free-tier tooling.

## Architecture

```text
BTS + Open-Meteo + OurAirports
        │
        ▼
  Parquet lake (data/ or R2)
        │
        ▼
   dbt-duckdb marts
        │
        ├──────────────► FastAPI (/predict, /route-stats, /weather)
        ▼                              │
   XGBoost + MLflow                    ▼
        │                    LangGraph agent (Groq)
        ▼                              │
  models/local                    Streamlit UI
```

## Stack (free-first)

| Layer | Tool |
|---|---|
| Lake | Parquet + Cloudflare R2 (optional) |
| SQL engine | DuckDB |
| Transforms | dbt-duckdb |
| Orchestration | GitHub Actions |
| ML | XGBoost + scikit-learn + MLflow |
| API | FastAPI |
| Agent | LangGraph + Groq (`llama-3.1-8b-instant`) |
| UI | Streamlit Community Cloud |

**Hubs in scope:** LAX, JFK, ORD, DEN, ATL, IAD, DFW (see `configs/project.yaml`).

## Quick start (local demo, offline-friendly)

Requirements: Python 3.11+, [uv](https://github.com/astral-sh/uv). Optional: a free
[Groq](https://console.groq.com) API key for full agent chat.

```bash
# Install
uv sync

# Sample ingest → dbt → train (no cloud accounts required)
uv run flight demo

# API
uv run flight serve
# open http://127.0.0.1:8000/docs

# Chat UI (uses Groq if GROQ_API_KEY is set; otherwise tool fallback)
uv run flight ui
```

One-shot agent question:

```bash
uv run flight agent "Will DL ATL to LAX on a Monday at 8am be delayed?"
```

## Data engineering

1. **Ingest** (rolling + incremental by default)
   - Discovers the latest published BTS PREZIP month, targets the last
     `retention_months` (48), and **skips months already in the lake / R2**
   - **Auto-retrain:** when new months land or the window slides, ingest runs
     `dbt build` + `train` and writes `models/local/train_state.json`
     (`model.auto_retrain: true`; disable with `--no-retrain`)
   - `flight ingest all --to-r2 --no-keep-local` — pull only missing months → R2
   - `flight ingest all --full` — re-download the whole window
   - `flight ingest all --fixed` — use static `date_range` start/end in config
   - `flight ingest all --sample` — synthetic bronze for demos/CI
   - `flight ingest prune` — drop data older than 48 months (also after `ingest all`)
2. **Transform** — `flight dbt build --from-r2` reads bronze from R2, builds marts,
   then **pushes gold to R2** (`curated/*.parquet` + `warehouse/flight_agent.duckdb`)
   and deletes the local DuckDB by default (`FLIGHT_DUCKDB_KEEP_LOCAL=false`).
   Train/serve query curated Parquet on R2 in-memory — no large local DB required.
   Manual: `flight warehouse push` / `flight warehouse pull`.
3. **Cloud lake** — set R2 vars in `.env`, then either:
   - Direct upload: `flight ingest all --to-r2 --no-keep-local`
   - Or legacy bulk sync of local files: `flight ingest sync-r2 --prefix raw/`
   Retention prune deletes R2 BTS partitions **and** trims R2 weather rows
   older than 48 months (same window as local).
4. **Schedule** — [`.github/workflows/weekly-lake-update.yml`](.github/workflows/weekly-lake-update.yml)
   runs Mondays: incremental R2 ingest, auto-retrain when the window moves, then
   **`flight model push`** so Streamlit Cloud can pull `models/model.joblib`.
   Daily sample smoke: [`daily-pipeline.yml`](.github/workflows/daily-pipeline.yml).

## Deploy (free): GitHub Actions + Streamlit Cloud

**Split of responsibilities**

| Where | What |
|---|---|
| GitHub Actions | Ingest → dbt → train → push model + curated lake to R2 |
| Streamlit Community Cloud | Chat UI + Groq agent; pulls model from R2; queries curated Parquet on R2 |

### 1. GitHub secrets

In the repo → Settings → Secrets and variables → Actions, add:

- `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET`
- optional `R2_ENDPOINT_URL`

Run **Weekly lake update** once (Actions → Run workflow). Or push your current local model:

```bash
uv run flight model push
```

### 2. Streamlit Community Cloud

1. Push this repo to GitHub (if you have not already).
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app.
3. Main file: `streamlit_app.py`
4. Python requirements: `requirements.txt` (already at repo root).
5. Secrets (App settings → Secrets) — copy from [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example):

```toml
GROQ_API_KEY = "gsk_..."
GROQ_MODEL = "llama-3.1-8b-instant"
R2_ACCOUNT_ID = "..."
R2_ACCESS_KEY_ID = "..."
R2_SECRET_ACCESS_KEY = "..."
R2_BUCKET = "flight-prediction-agent"
```

On startup the app downloads `models/model.joblib` (and meta/metrics) from R2 when missing locally.

### 3. Local model sync helpers

```bash
uv run flight model push   # after train
uv run flight model pull   # download into models/local
uv run flight train --publish-r2   # train and upload (default when R2 is configured)
uv run flight serve-cache sync     # download small agg marts for fast UI tools
```

Chat tools use `data/serve_cache/` (or `/tmp/flight_serve_cache` on Streamlit Cloud)
instead of querying multi‑GB R2 Parquet on every question. Streamlit is pinned to
`<1.60` to avoid blank-page issues during long runs.

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + model loaded flag |
| `POST /predict` | Delay probability (+ congestion drivers) |
| `GET /route-stats` | Historical route delay / taxi / NAS |
| `GET /congestion` | Airport×hour congestion profile |
| `GET /carrier-stats` | Carrier reliability summary |
| `GET /weather` | Curated weather features |
| `GET /metrics` | Offline eval metrics |

## Agent

Tools (in-process): `predict_delay`, `route_stats`, `weather`, `airport_congestion`, `model_metrics`.

```bash
# Set GROQ_API_KEY in .env (https://console.groq.com/keys)
uv run flight ui
```

Without Groq, the agent returns a deterministic tool summary so demos still work.

## Configuration

- [`configs/project.yaml`](configs/project.yaml) — hubs, `date_range.mode` (`rolling`|`fixed`), retention, model hyperparams
- [`.env.example`](.env.example) — R2, HF, Groq, paths

With `date_range.mode: rolling` (default), the lake tracks the latest BTS publish
and keeps `retention_months` of history. Set `mode: fixed` for a pinned window.

## MLOps

```bash
uv run flight train
uv run flight train --publish-r2   # push to R2 for Streamlit (default when R2_* set)
uv run flight train --publish-hf   # optional HF Hub (needs HF_TOKEN + HF_REPO_ID)
```

Prefer letting ingest auto-retrain: when the rolling window advances, `flight ingest all`
runs dbt + train and records the window in `models/local/train_state.json`.
Use `--no-retrain` to ingest without training; set `model.auto_retrain: false` in
`configs/project.yaml` to turn it off by default.
Artifacts land in `models/local/` and (when R2 is configured) under `s3://…/models/`.
MLflow uses `sqlite:///mlflow.db` (override with `FLIGHT_MLFLOW_TRACKING_URI`).

On macOS, if XGBoost fails to load OpenMP, install `brew install libomp` — otherwise
training automatically falls back to sklearn `HistGradientBoostingClassifier`.

Optional drift report:

```bash
uv sync --extra drift
uv run flight drift
```


- **Target:** `arr_delay_15` — arrival delay ≥ 15 minutes (BTS definition)
- **Features:** schedule + peak/weekend, same-day bank volume, weather, historical taxi/NAS/carrier congestion profiles, route reliability
- **Model:** `OneHotEncoder` + `XGBClassifier` sklearn pipeline
- **Data:** Public BTS On-Time, Open-Meteo, OurAirports (synthetic sample available for CI)

## Project layout

```text
configs/               Hub + date + model config
dbt/                   dbt-duckdb project
streamlit_app.py       Streamlit Community Cloud entrypoint
requirements.txt       Slim UI deps for Streamlit Cloud
.streamlit/            Config + secrets example
src/flight_agent/
  ingest/              BTS, weather, airports, R2 sync
  features/            Training frame from DuckDB marts
  train/               MLflow training + R2/HF publish + drift
  serve/               FastAPI + shared services
  agent/               LangGraph tools + graph
  ui/                  Streamlit chat
  cli.py               `flight` CLI
.github/workflows/     Weekly lake+model, daily smoke, CI
spaces/                Optional HF Spaces helper
```

## License

Demo / research project. Respect BTS, Open-Meteo, and OurAirports terms when
redistributing data.
